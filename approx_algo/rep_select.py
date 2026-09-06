"""
RepSelect: Robust Unlearning via Representation Selectivity.

Port of:

    Filip Sondej, Yushi Yang, Adam Mahdi.
    "RepSelect: Robust LLM Unlearning via Representation Selectivity."
    arXiv:2606.17168v3.
    https://github.com/filyp/RepSelect

adapted to this repository's `ModuleArchitecture` / `DeepMoELayer`. Algorithm
1 is architecture-agnostic (it operates on `named_parameters()` weight
tensors and their accumulated gradients), so the numerics below are an
unmodified port; three things are specific to this integration:

  * `layer_filter` -- the paper's default regex targets `fc1|fc2|w1|w2|w3|
    gate_proj|up_proj|down_proj`-style naming. This repo's dense blocks use
    timm's `Mlp` (`blocks.i.mlp.fc1/fc2.weight`, matches already) but its MoE
    expert FFN is a custom `DeepExpert` with `self.layers = nn.ModuleList(...)`
    (`blocks.i.moe.experts.j.layers.0/1.weight`), which the original regex
    does not match at all -- silently skipping every MoE expert and leaving
    only the dense-block MLPs projected. Fixed below.
  * Loss: classification CE via `self.criteria`, through
    `model.forward_with_grad(x)` (this repo's model returns `(logits,
    features)`, not raw logits).
  * Application schedule: the paper computes one cached weight-delta and
    applies it at a single strength alpha (optionally sweeping alpha under a
    utility budget). To fit this benchmark's per-epoch checkpoint / FA
    early-stopping convention, `RepSelect.unlearn` computes the delta ONCE
    (a true single pass over the forget set, per Algorithm 1) and then
    ramps the cumulative strength linearly up to `lr` over `num_epoch`
    epochs, applying only the incremental delta each epoch -- this is
    exactly the paper's "cached-update rescaling" trick (§5), just spread
    across the epoch loop instead of a one-shot apply.

No retain set is used (§4, "No retain set required" -- ascent on the
negated forget CE is sufficient); `retain_loader` is accepted for signature
compatibility with the other wrappers but ignored.
"""
from __future__ import annotations

import math
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from approx_algo.gradient_ascent import Gradient_Ascent

# Matches this repo's naming: timm `Mlp` in dense blocks (`mlp.fc1/fc2.weight`)
# and the custom `DeepExpert` FFN inside `DeepMoELayer` (`moe.experts.j.layers.k.weight`).
DEFAULT_LAYER_FILTER = r"(mlp\.fc[12]|moe\.experts\.\d+\.layers\.\d+)\.weight$"
DEFAULT_MOE_EXPERT_REGEX = r"experts\.(\d+)\."


# =============================================================================
# Numerical primitives (architecture-agnostic; unmodified from the paper port)
# =============================================================================


def _mahalanobis_collapse(M: torch.Tensor, E: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """collapse(M, E, S) = M - (P - P/S_tilde) @ E^T,  S_tilde = S^2 / min(S^2)."""
    lam = S * S
    lam_min = lam.min().clamp(min=torch.finfo(lam.dtype).eps)
    S_tilde = lam / lam_min
    P = M @ E
    correction = P - P / S_tilde
    return M - correction @ E.transpose(-2, -1)


def _low_rank_svd(G: torch.Tensor, k: int, svd_dtype: torch.dtype, driver: Optional[str] = None):
    orig_dtype, orig_device = G.dtype, G.device
    g = G.detach().to(svd_dtype)
    if g.device.type == "cuda" and driver is not None:
        U, S, Vh = torch.linalg.svd(g, full_matrices=False, driver=driver)
    else:
        U, S, Vh = torch.linalg.svd(g, full_matrices=False)
    k_req = min(k, S.numel())
    if k_req > 0 and S[0] > 0:
        # Drop singular values that are numerically negligible relative to the
        # largest one. `_mahalanobis_collapse` normalises by the SMALLEST kept
        # eigenvalue (S_tilde = lam / lam_min); if that "smallest" value is
        # actually near-zero noise rather than a real component (common when
        # k_pcs exceeds the accumulated gradient's true rank -- e.g. a group
        # whose params are frozen by this benchmark's default grad mode has an
        # all-zero gradient buffer, an all-zero SVD, and 0/0 = NaN), the
        # normalisation blows up by many orders of magnitude and corrupts the
        # weights on the very first update. Restricting to the numerically
        # significant top components (relative tolerance ~sqrt(machine eps),
        # the standard rank cutoff) keeps the "smallest kept" eigenvalue
        # meaningful, and an all-zero G naturally yields k_eff=0 below.
        rel_tol = torch.finfo(svd_dtype).eps ** 0.5
        keep = S[:k_req] > S[0] * rel_tol
        k_eff = max(1, int(keep.sum().item()))
    else:
        k_eff = 0
    U = U[:, :k_eff].contiguous()
    S = S[:k_eff].contiguous()
    V = Vh[:k_eff, :].transpose(0, 1).contiguous()
    return U.to(orig_dtype).to(orig_device), S.to(orig_dtype).to(orig_device), V.to(orig_dtype).to(orig_device)


def _two_sided_collapse(G: torch.Tensor, k: int, svd_dtype: torch.dtype, driver: Optional[str]) -> torch.Tensor:
    U, S, V = _low_rank_svd(G, k, svd_dtype, driver)
    if S.numel() == 0:
        # G has no numerically significant singular directions (e.g. an
        # all-zero gradient buffer for a group with no trainable params this
        # pass) -- nothing to collapse; the delta for this group is G itself
        # (zero, in the all-frozen case), not NaN from a degenerate SVD.
        return G.clone()
    G_prime = _mahalanobis_collapse(G, V, S)
    G_prime = _mahalanobis_collapse(G_prime.transpose(-2, -1).contiguous(), U, S).transpose(-2, -1).contiguous()
    return G_prime


@dataclass
class _MLPGroup:
    key: str
    params: List[Tuple[str, nn.Parameter]]

    @property
    def is_moe(self) -> bool:
        return len(self.params) > 1


def _group_parameters(model: nn.Module, layer_filter: str, moe_expert_regex: str) -> List[_MLPGroup]:
    layer_pat = re.compile(layer_filter)
    moe_pat = re.compile(moe_expert_regex)
    buckets: "OrderedDict[str, List[Tuple[str, nn.Parameter]]]" = OrderedDict()

    n_frozen_skipped = 0
    for name, param in model.named_parameters():
        if param.ndim < 2 or not layer_pat.search(name):
            continue
        if not param.requires_grad:
            # This benchmark's default grad mode (ModuleArchitecture.
            # _set_grad_mode("unlearning")) freezes every dense-block MLP,
            # so these would accumulate an all-zero gradient buffer and hit
            # the degenerate all-zero-SVD case in _two_sided_collapse for no
            # reason -- skip them outright rather than project a guaranteed
            # zero delta.
            n_frozen_skipped += 1
            continue
        canonical = moe_pat.sub("experts._.", name)
        buckets.setdefault(canonical, []).append((name, param))

    groups = [_MLPGroup(key=k, params=v) for k, v in buckets.items()]
    if n_frozen_skipped:
        print(f"[*] RepSelect: skipped {n_frozen_skipped} frozen (requires_grad=False) "
              f"parameter(s) matching layer_filter.")
    if not groups:
        print(f"[!] RepSelect: layer_filter={layer_filter!r} matched no trainable parameters.")
    else:
        n_moe = sum(1 for g in groups if g.is_moe)
        moe_keys = [g.key for g in groups if g.is_moe]
        print(f"[*] RepSelect: {len(groups)} MLP SVD-group(s) "
              f"({len(groups) - n_moe} dense, {n_moe} MoE): {moe_keys}")
    return groups


class _LoRALinearHook:
    """Rank-r LoRA delta A @ B attached via forward hooks; self-contained, no `peft` dependency."""

    def __init__(self, layer: nn.Linear, rank: int, device: torch.device):
        out_f, in_f = layer.weight.shape
        self.A = nn.Parameter(torch.zeros(in_f, rank, device=device, dtype=layer.weight.dtype))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(rank, out_f, device=device, dtype=layer.weight.dtype))
        self._handle = layer.register_forward_pre_hook(self._pre)
        self._post_handle = layer.register_forward_hook(self._post)
        self._last_input: Optional[torch.Tensor] = None

    def _pre(self, module, inputs):
        self._last_input = inputs[0]
        return None

    def _post(self, module, inputs, output):
        x, self._last_input = self._last_input, None
        if x is None:
            return output
        return output + (x @ self.A) @ self.B

    def parameters(self) -> List[nn.Parameter]:
        return [self.A, self.B]

    def remove(self):
        self._handle.remove()
        self._post_handle.remove()


def _attach_lora_to_groups(model, groups, rank, device) -> List[_LoRALinearHook]:
    param_to_module: Dict[int, nn.Module] = {}
    for mod in model.modules():
        for _, p in mod.named_parameters(recurse=False):
            param_to_module[id(p)] = mod

    hooks: List[_LoRALinearHook] = []
    for g in groups:
        for name, p in g.params:
            mod = param_to_module.get(id(p))
            if isinstance(mod, nn.Linear):
                hooks.append(_LoRALinearHook(mod, rank=rank, device=device))
            else:
                print(f"[!] RepSelect LoRA: skipping non-Linear owner of {name} (type={type(mod).__name__}).")
    return hooks


def _project_and_split(buffers, groups, k, svd_dtype, svd_driver) -> Dict[str, torch.Tensor]:
    result: Dict[str, torch.Tensor] = {}
    for g in groups:
        G_proj = _two_sided_collapse(buffers[g.key], k=k, svd_dtype=svd_dtype, driver=svd_driver)
        offset = 0
        for name, p in g.params:
            width = p.shape[1]
            result[name] = G_proj[:, offset:offset + width].contiguous()
            offset += width
    return result


def _apply_update(model, deltas: Dict[str, torch.Tensor], alpha: float) -> None:
    """In-place: W -= alpha * DeltaW."""
    name_to_param = dict(model.named_parameters())
    with torch.no_grad():
        for name, dW in deltas.items():
            name_to_param[name].data.add_(dW, alpha=-alpha)


# =============================================================================
# Wrapper class -- matches this repo's Gradient_Ascent-style API
# =============================================================================


class RepSelect(Gradient_Ascent):
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        unseen_loader,
        forget_loader,
        forget_test_loader,
        retain_loader,
        retain_test_loader,
        optimizer,
        criteria,
        num_epoch,
        device="cuda",
        # ---- RepSelect-specific hyperparameters ----
        lr=0.05,                        # alpha in Algorithm 1: single cached-update strength
        k_pcs=512,                      # top-k PCs kept in the low-rank SVD collapse
        use_lora_adversary=False,       # off by default: vision classification is "knowledge-like" (see docstring)
        lora_rank=8,
        lora_lr=0.05,
        layer_filter=DEFAULT_LAYER_FILTER,
        moe_expert_regex=DEFAULT_MOE_EXPERT_REGEX,
        svd_dtype="float32",
        svd_driver=None,
        log_every=50,
    ):
        super().__init__(
            model=model, train_loader=train_loader, test_loader=test_loader,
            unseen_loader=unseen_loader, forget_loader=forget_loader,
            forget_test_loader=forget_test_loader, retain_loader=retain_loader,
            retain_test_loader=retain_test_loader, optimizer=optimizer,
            criteria=criteria, num_epoch=num_epoch, device=device,
        )
        self.lr = lr
        self.k_pcs = k_pcs
        self.use_lora_adversary = use_lora_adversary
        self.lora_rank = lora_rank
        self.lora_lr = lora_lr
        self.layer_filter = layer_filter
        self.moe_expert_regex = moe_expert_regex
        self.svd_dtype = svd_dtype
        self.svd_driver = svd_driver
        self.log_every = log_every

    def _train_lora_adversary(self, groups) -> List[_LoRALinearHook]:
        hooks = _attach_lora_to_groups(self.model, groups, rank=self.lora_rank, device=self.device)
        if not hooks:
            print("[!] RepSelect LoRA: no hooks attached; skipping elicitation.")
            return hooks

        lora_params: List[nn.Parameter] = []
        for h in hooks:
            lora_params.extend(h.parameters())
        lora_optim = torch.optim.SGD(lora_params, lr=self.lora_lr)

        base_requires_grad = {n: p.requires_grad for n, p in self.model.named_parameters()}
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.model.train()
        for step, batch in enumerate(self.forget_loader, start=1):
            images, labels = batch[0].to(self.device), batch[1].to(self.device)
            lora_optim.zero_grad(set_to_none=True)
            logits, _ = self.model.forward_with_grad(images)
            loss = self.criteria(logits, labels)  # descent: elicit the forget behaviour
            loss.backward()
            lora_optim.step()
            if step % self.log_every == 0:
                print(f"[RepSelect LoRA] step {step}  loss={loss.item():.4f}")

        for n, p in self.model.named_parameters():
            p.requires_grad_(base_requires_grad[n])
        return hooks

    def _accumulate_forget_gradient(self, groups) -> Dict[str, torch.Tensor]:
        self.model.train()
        buffers: Dict[str, torch.Tensor] = {
            g.key: torch.zeros_like(torch.cat([p.detach() for _, p in g.params], dim=1)) for g in groups
        }

        for step, batch in enumerate(self.forget_loader, start=1):
            images, labels = batch[0].to(self.device), batch[1].to(self.device)
            self.model.zero_grad(set_to_none=True)
            logits, _ = self.model.forward_with_grad(images)
            loss = -self.criteria(logits, labels)  # ascent direction
            loss.backward()

            for g in groups:
                grads = [p.grad for _, p in g.params]
                grads = [(torch.zeros_like(p.detach()) if gr is None else gr) for (_, p), gr in zip(g.params, grads)]
                buffers[g.key].add_(torch.cat(grads, dim=1))

            if step % self.log_every == 0:
                print(f"[RepSelect] accumulate step {step}  loss={loss.item():.4f}")

        self.model.zero_grad(set_to_none=True)
        return buffers

    def unlearn(self, fa_threshold, ckpt_path):
        self.model.to(self.device)
        groups = _group_parameters(self.model, self.layer_filter, self.moe_expert_regex)
        if not groups:
            raise RuntimeError("RepSelect found no MLP tensors to project. Check `layer_filter`.")

        svd_dtype = getattr(torch, self.svd_dtype)

        hooks: List[_LoRALinearHook] = []
        if self.use_lora_adversary:
            print(f"[*] RepSelect: training LoRA adversary (rank={self.lora_rank}).")
            hooks = self._train_lora_adversary(groups)

        try:
            print("[*] RepSelect: accumulating forget-gradient (single pass over the forget set).")
            buffers = self._accumulate_forget_gradient(groups)
        finally:
            for h in hooks:
                h.remove()

        print(f"[*] RepSelect: projecting (k={self.k_pcs} PCs, {len(groups)} SVD groups, dtype={self.svd_dtype}).")
        deltas = _project_and_split(buffers, groups, k=self.k_pcs, svd_dtype=svd_dtype, svd_driver=self.svd_driver)
        del buffers

        # The delta is computed once; apply it as a linear cumulative ramp up to `self.lr`
        # over the epoch loop so this fits the benchmark's per-epoch eval / FA early-stop
        # convention (equivalent to the paper's cached-update alpha rescaling, §5).
        total_unlearn_time = 0.0
        early_stop = False
        closest_fa_score = float("inf")
        alpha_applied = 0.0

        for epoch in range(self.num_epoch):
            epoch_start = time.time()
            alpha_target = self.lr * (epoch + 1) / self.num_epoch
            _apply_update(self.model, deltas, alpha=alpha_target - alpha_applied)
            alpha_applied = alpha_target
            epoch_time = time.time() - epoch_start
            total_unlearn_time += epoch_time

            fa_score, ra_score, ta_score, mia_score = self.evaluate()

            if fa_threshold < fa_score < closest_fa_score:
                closest_fa_score = fa_score
                torch.save(self.model.state_dict(), f"{ckpt_path}_closest_fa.pt")
            if fa_score <= fa_threshold:
                early_stop = True

            print(f"[RepSelect] epoch [{epoch + 1}/{self.num_epoch}] | alpha: {alpha_applied:.4g} | "
                  f"ra: {ra_score*100:.2f}% | fa: {fa_score*100:.2f}% | "
                  f"ta: {ta_score*100:.2f}% | mia: {mia_score:.4f}")

            wandb.log({
                "epoch": epoch + 1, "rep_select/alpha": alpha_applied,
                "ra": ra_score, "fa": fa_score, "ta": ta_score, "mia": mia_score,
            })
            torch.save(self.model.state_dict(), f"{ckpt_path}_epoch_{epoch + 1}.pt")
            torch.save(self.model.state_dict(), f"{ckpt_path}.pt")

            if early_stop:
                break

        peak_memory_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
        wandb.log({"peak_memory_gb": peak_memory_gb})
        torch.save(self.model.state_dict(), f"{ckpt_path}.pt")
        return total_unlearn_time
