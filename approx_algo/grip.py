"""
GRIP: Geometric Routing Invariance Preservation for MoE Unlearning.

Port of:

    Andy Zhu, Rongzhe Wei, Yupu Gu, Pan Li.
    "GRIP: Algorithm-Agnostic Machine Unlearning for Mixture-of-Experts
     via Geometric Router Constraints."
    COLM 2026. arXiv:2601.16905v3.
    https://github.com/AndyZhu311/GRIP

adapted to this repository's `ModuleArchitecture` (a MoE-ViT: `DeepMoELayer`
with `self.router = nn.Linear(model_dim, num_experts, bias=False)` and
`self.experts = nn.ModuleList([DeepExpert(...), ...])`, routed per patch
token). The default `router_filter` regex matches `...moe.router.weight`
directly, so no adjustment was needed there; `top_k_experts` should be set to
match the run's `gate_k`.

GRIP is a *wrapper*, not a standalone algorithm: it composes with a base
forget/retain objective (here, Gradient Difference, matching the paper's
§A.2 Eq. 6) and only constrains how the router is allowed to move, forcing
the unlearning pressure into expert parameters instead of letting the base
objective take the "routing shortcut" (redirecting forget-queries to a
different expert rather than erasing the knowledge).

One integration wrinkle specific to this benchmark: `ModuleArchitecture.
_set_grad_mode("unlearning")` (called unconditionally in unlearn.py before
any algorithm wrapper is constructed) freezes the router along with
everything outside the MoE experts. Since GRIP's entire premise is
constraining router *updates*, a permanently frozen router makes Method A
vacuous (the projected delta is always zero) and makes Method B's
correction the only thing that ever moves the router. `unlock_router=True`
(the default here) re-enables `requires_grad` on router weights before
unlearning starts, and the optimizer is rebuilt over the resulting trainable
set -- matching the paper's assumption that the base algorithm is free to
push on the router before GRIP constrains it.

The two variants
-----------------
* Method A (`method="A"`, training-time enforcement): after every
  optimizer.step(), project the router's weight update into the null space
  of the retain-input subspace (equality) and then through a Randomized
  Kaczmarz pass enforcing the pre-unlearning selection margins (inequality).
* Method B (`method="B"`, post-training correction -- recommended default):
  run the base algorithm unconstrained, then apply one closed-form
  ridge-regularised realignment per router at the end, guarded by a cosine
  acceptance check.

See MU/grip_baseline/grip_files/INTEGRATION_NOTES.md for the original
adaptation notes (router-input capture semantics, null-space saturation
caveats for small-E vision MoE, etc.) -- retained here as design context;
the class below is the actual integration into this repo's wrapper/dispatch
conventions.
"""
from __future__ import annotations

import inspect
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from approx_algo.gradient_ascent import Gradient_Ascent


# =============================================================================
# Router identification and input capture (architecture-agnostic primitives)
# =============================================================================


def _find_routers(model: nn.Module, filter_pattern: str) -> List[Tuple[str, nn.Linear, nn.Module]]:
    """Enumerate MoE router linear layers matching the filter.

    Returns list of (parameter_name, router_linear, parent_module) tuples.
    """
    pat = re.compile(filter_pattern)
    name_to_module = {name: mod for name, mod in model.named_modules()}
    param_name_to_module_name = {}
    for module_name, module in model.named_modules():
        for pname, p in module.named_parameters(recurse=False):
            param_name_to_module_name[f"{module_name}.{pname}" if module_name else pname] = module_name

    routers: List[Tuple[str, nn.Linear, nn.Module]] = []
    for pname, param in model.named_parameters():
        if not pat.search(pname):
            continue
        if not isinstance(param, torch.Tensor) or param.ndim != 2:
            continue
        module_name = param_name_to_module_name.get(pname)
        if module_name is None:
            continue
        router_mod = name_to_module.get(module_name)
        if not isinstance(router_mod, nn.Linear):
            continue
        parent_name = ".".join(module_name.split(".")[:-1])
        parent_mod = name_to_module.get(parent_name, model)
        routers.append((pname, router_mod, parent_mod))

    if not routers:
        print(f"[!] GRIP: router_filter={filter_pattern!r} matched no Linear layers.")
    else:
        print(f"[*] GRIP: identified {len(routers)} router layer(s): {[r[0] for r in routers]}")
    return routers


def _unpack_batch(batch, device):
    imgs, labels = batch[0], batch[1]
    return imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)


class _RouterInputCapture:
    """Forward-hook capture of router INPUTS across a set of routers.

    `run()` iterates a loader, collects each router's pre-forward input
    (a per-patch-token vector -- this repo's `DeepMoELayer.forward` already
    flattens `(B, S, D) -> (B*S, D)` before calling the router), and returns
    `{param_name: X in R^(d, N)}`.
    """

    def __init__(self, routers: List[Tuple[str, nn.Linear, nn.Module]]):
        self._routers = routers
        self._buffers: Dict[str, List[torch.Tensor]] = {n: [] for n, _, _ in routers}
        self._handles = []

    def _make_hook(self, name: str):
        buf = self._buffers[name]

        def hook(module, inputs):
            x = inputs[0]
            if x.ndim >= 2:
                d = x.shape[-1]
                buf.append(x.reshape(-1, d).detach())
            return None

        return hook

    def __enter__(self):
        self._handles = [
            router.register_forward_pre_hook(self._make_hook(name))
            for name, router, _ in self._routers
        ]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self._handles:
            h.remove()
        self._handles = []

    @torch.no_grad()
    def run(self, model, loader, device, max_images, tokens_per_image=None) -> Dict[str, torch.Tensor]:
        model.eval()
        n_seen = 0
        for n in self._buffers:
            self._buffers[n].clear()

        for batch in loader:
            if n_seen >= max_images:
                break
            imgs, _ = _unpack_batch(batch, device)
            take = min(imgs.shape[0], max_images - n_seen)
            model(imgs[:take])
            n_seen += take

        out: Dict[str, torch.Tensor] = {}
        for name, chunks in self._buffers.items():
            if not chunks:
                continue
            all_x = torch.cat(chunks, dim=0)
            if tokens_per_image is not None and n_seen > 0 and all_x.shape[0] % n_seen == 0:
                T = all_x.shape[0] // n_seen
                if tokens_per_image < T:
                    x_per_img = all_x.reshape(n_seen, T, -1)
                    all_x = x_per_img[:, :tokens_per_image, :].reshape(-1, x_per_img.shape[-1])
            out[name] = all_x.transpose(0, 1).contiguous()  # (d, N)

        for n in self._buffers:
            self._buffers[n].clear()
        return out


@torch.no_grad()
def _compute_pre_selections(router: nn.Linear, X: torch.Tensor, top_k: int) -> torch.Tensor:
    """Router (E, d) x retain inputs X (d, N) -> top-k expert indices per input, (N, top_k)."""
    scores = router.weight.detach() @ X
    if router.bias is not None:
        scores = scores + router.bias.detach().unsqueeze(1)
    top = scores.topk(top_k, dim=0).indices
    return top.transpose(0, 1).contiguous().sort(dim=1).values


@dataclass
class _ExpertConstraint:
    U_ret: torch.Tensor       # (d, r_j) orthonormal basis of the retain subspace that selected expert j
    X_ineq: torch.Tensor      # (d, |I_j^c|) retain inputs that did NOT select expert j
    tau_ineq: torch.Tensor    # (|I_j^c|,) selection margins
    n_selected: int


def _precompute_constraints_for_router(
    router: nn.Linear, X: torch.Tensor, S_pre: torch.Tensor, top_k: int, epsilon_null: float,
) -> List[_ExpertConstraint]:
    E, d = router.weight.shape
    N = X.shape[1]
    device = X.device
    constraints: List[_ExpertConstraint] = []

    selected_mask = torch.zeros(E, N, dtype=torch.bool, device=device)
    for k in range(top_k):
        selected_mask.scatter_(0, S_pre[:, k].unsqueeze(0), True)

    scores = router.weight.detach() @ X
    if router.bias is not None:
        scores = scores + router.bias.detach().unsqueeze(1)

    for j in range(E):
        sel = selected_mask[j]
        idx_sel = sel.nonzero(as_tuple=True)[0]
        idx_notsel = (~sel).nonzero(as_tuple=True)[0]

        if idx_sel.numel() > 0:
            X_eq = X[:, idx_sel]
            U, S, _ = torch.linalg.svd(X_eq, full_matrices=False)
            keep = (S * S) > epsilon_null
            r_j = int(keep.sum().item())
            U_ret = U[:, :r_j].contiguous() if r_j > 0 else torch.zeros(d, 0, device=device, dtype=X.dtype)
        else:
            U_ret = torch.zeros(d, 0, device=device, dtype=X.dtype)

        if idx_notsel.numel() > 0:
            scores_notsel = scores[:, idx_notsel]
            s_j_notsel = scores_notsel[j]
            selected_scores = scores_notsel.gather(0, S_pre[idx_notsel].t())
            min_selected = selected_scores.min(dim=0).values
            tau = min_selected - s_j_notsel
            X_ineq = X[:, idx_notsel].contiguous()
        else:
            X_ineq = torch.zeros(d, 0, device=device, dtype=X.dtype)
            tau = torch.zeros(0, device=device, dtype=X.dtype)

        constraints.append(_ExpertConstraint(U_ret=U_ret, X_ineq=X_ineq, tau_ineq=tau, n_selected=idx_sel.numel()))
    return constraints


def _null_space_project(u: torch.Tensor, U_ret: torch.Tensor) -> torch.Tensor:
    if U_ret.shape[1] == 0:
        return u
    return u - U_ret @ (U_ret.transpose(0, 1) @ u)


def _kaczmarz_project(
    u: torch.Tensor, X_ineq: torch.Tensor, tau_ineq: torch.Tensor,
    epsilon: float, max_iters: int, generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    if X_ineq.shape[1] == 0 or max_iters <= 0:
        return u

    row_norms_sq = (X_ineq * X_ineq).sum(dim=0)
    total = row_norms_sq.sum().clamp(min=1e-30)
    probs = row_norms_sq / total

    n = X_ineq.shape[1]
    u_out = u.clone()

    def all_satisfied(vec):
        residuals = (X_ineq.transpose(0, 1) @ vec) - (tau_ineq - epsilon)
        return bool((residuals <= 0).all().item())

    if all_satisfied(u_out):
        return u_out

    for _ in range(max_iters):
        i = int(torch.multinomial(probs, 1, generator=generator).item())
        x_i = X_ineq[:, i]
        residual = (u_out @ x_i).item() - (tau_ineq[i].item() - epsilon)
        if residual > 0:
            u_out = u_out - (residual / row_norms_sq[i].item()) * x_i

    return u_out


def _project_router_update(delta_W, per_expert, epsilon_margin, kaczmarz_max_iters, generator):
    E, d = delta_W.shape
    projected = torch.empty_like(delta_W)
    for j in range(E):
        u = _null_space_project(delta_W[j], per_expert[j].U_ret)
        u = _kaczmarz_project(
            u, X_ineq=per_expert[j].X_ineq, tau_ineq=per_expert[j].tau_ineq,
            epsilon=epsilon_margin, max_iters=kaczmarz_max_iters, generator=generator,
        )
        projected[j] = u
    return projected


@torch.no_grad()
def _apply_ptc_correction(
    router: nn.Linear, Theta_original: torch.Tensor, X_original: torch.Tensor,
    X_unlearned: torch.Tensor, ridge: float, acceptance_cosine: float, layer_name: str,
) -> bool:
    """Eq. 5: Delta = Theta X (X~^T (X~ X~^T + lambda I)^-1) - Theta~; guarded by a cosine acceptance check."""
    Theta_tilde = router.weight.data
    d = X_unlearned.shape[0]

    XtXt = X_unlearned @ X_unlearned.transpose(0, 1)
    reg = XtXt + ridge * torch.eye(d, device=XtXt.device, dtype=XtXt.dtype)
    try:
        M = torch.linalg.solve(reg, X_unlearned)
    except RuntimeError as e:
        print(f"[!] [GRIP-B] {layer_name}: linear solve failed ({e}); skipping.")
        return False
    X_tilde_pinv = M.transpose(0, 1)

    target = Theta_original @ X_original @ X_tilde_pinv
    delta = target - Theta_tilde

    pre_scores = (Theta_original @ X_original).reshape(-1)
    post_scores = ((Theta_tilde + delta) @ X_unlearned).reshape(-1)
    cos = F.cosine_similarity(pre_scores.unsqueeze(0), post_scores.unsqueeze(0)).item()
    if cos < acceptance_cosine:
        print(f"[!] [GRIP-B] {layer_name}: acceptance check FAILED "
              f"(cos={cos:.3f} < {acceptance_cosine:.3f}). Skipping correction "
              f"(consider Method A for this layer).")
        return False

    router.weight.data.add_(delta)
    print(f"[*] [GRIP-B] {layer_name}: applied correction (acceptance cos={cos:.3f}).")
    return True


# =============================================================================
# Wrapper class -- matches this repo's Gradient_Ascent-style API
# =============================================================================


class GRIP(Gradient_Ascent):
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
        # ---- GRIP-specific hyperparameters ----
        method="B",                    # "A" (training-time enforcement) or "B" (post-training correction)
        retain_weight=1.0,             # alpha in the Gradient-Difference base objective
        grad_clip=1.0,
        router_filter=r"(mlp|moe)\.(gate|router)\.weight$",
        top_k_experts=2,               # must match the run's gate_k
        retain_cache_size=128,         # retain IMAGES cached for router-input capture
        tokens_per_image=1,            # see module docstring: small-E vision MoE saturates null space fast
        epsilon_null=1e-2,             # Method A: null-space eigenvalue cutoff
        epsilon_margin=1e-2,           # Method A: Kaczmarz safety margin
        kaczmarz_max_iters=100,        # Method A: iteration budget
        ptc_ridge=1e-6,                # Method B: ridge regularizer
        ptc_acceptance_cosine=0.95,    # Method B: per-layer acceptance check
        unlock_router=True,            # override ModuleArchitecture's default router freeze (see module docstring)
    ):
        super().__init__(
            model=model, train_loader=train_loader, test_loader=test_loader,
            unseen_loader=unseen_loader, forget_loader=forget_loader,
            forget_test_loader=forget_test_loader, retain_loader=retain_loader,
            retain_test_loader=retain_test_loader, optimizer=optimizer,
            criteria=criteria, num_epoch=num_epoch, device=device,
        )
        self.method = method
        self.retain_weight = retain_weight
        self.grad_clip = grad_clip
        self.router_filter = router_filter
        self.top_k_experts = top_k_experts
        self.retain_cache_size = retain_cache_size
        self.tokens_per_image = tokens_per_image
        self.epsilon_null = epsilon_null
        self.epsilon_margin = epsilon_margin
        self.kaczmarz_max_iters = kaczmarz_max_iters
        self.ptc_ridge = ptc_ridge
        self.ptc_acceptance_cosine = ptc_acceptance_cosine
        self.unlock_router = unlock_router

    def _reinit_optimizer(self):
        """Rebuild self.optimizer over the current trainable set (same pattern as Module/SEUF)."""
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        opt_class = type(self.optimizer)
        valid_kwargs = inspect.signature(opt_class.__init__).parameters.keys()
        filtered = {k: v for k, v in self.optimizer.defaults.items() if k in valid_kwargs}
        self.optimizer = opt_class(trainable, **filtered)

    def _capture_retain_reps(self, routers) -> Dict[str, torch.Tensor]:
        with _RouterInputCapture(routers) as cap:
            return cap.run(
                model=self.model, loader=self.retain_loader, device=self.device,
                max_images=self.retain_cache_size, tokens_per_image=self.tokens_per_image,
            )

    def _cycle(self, loader):
        while True:
            for b in loader:
                yield b

    def _gd_step_loss(self, f_batch, r_batch):
        """Base objective: Gradient Difference, paper §A.2 Eq. 6: L = -CE(forget) + alpha*CE(retain)."""
        f_imgs, f_labels = _unpack_batch(f_batch, self.device)
        r_imgs, r_labels = _unpack_batch(r_batch, self.device)
        logits_f, _ = self.model.forward_with_grad(f_imgs)
        logits_r, _ = self.model.forward_with_grad(r_imgs)
        forget_loss = self.criteria(logits_f, f_labels)
        retain_loss = self.criteria(logits_r, r_labels)
        return -forget_loss + self.retain_weight * retain_loss

    def unlearn(self, fa_threshold, ckpt_path):
        self.model.train()
        routers = _find_routers(self.model, self.router_filter)
        if not routers:
            print("[!] GRIP: no MoE routers matched `router_filter`; running the base "
                  "Gradient-Difference objective with no router constraint.")

        if self.unlock_router and routers:
            n_unlocked = 0
            for _, router, _ in routers:
                for p in router.parameters():
                    if not p.requires_grad:
                        p.requires_grad = True
                        n_unlocked += 1
            if n_unlocked:
                print(f"[*] GRIP: unlocked {n_unlocked} router tensor(s) frozen by the "
                      f"benchmark's default grad mode; rebuilding optimizer.")
                self._reinit_optimizer()

        if self.method.upper() == "B":
            return self._unlearn_method_b(routers, fa_threshold, ckpt_path)
        elif self.method.upper() == "A":
            return self._unlearn_method_a(routers, fa_threshold, ckpt_path)
        else:
            raise ValueError(f"GRIP.method must be 'A' or 'B'; got {self.method!r}")

    # -------------------------------------------------------------------
    # Method B: Post-Training Correction (recommended)
    # -------------------------------------------------------------------
    def _unlearn_method_b(self, routers, fa_threshold, ckpt_path):
        Theta_original = {name: router.weight.data.detach().clone() for name, router, _ in routers}
        X_original = self._capture_retain_reps(routers) if routers else {}

        total_unlearn_time, total_steps = 0.0, 0
        early_stop, closest_fa_score = False, float("inf")
        r_iter = self._cycle(self.retain_loader)

        for epoch in range(self.num_epoch):
            epoch_start = time.time()
            self.model.train()
            total_loss, n_steps = 0.0, 0

            for f_batch in self.forget_loader:
                r_batch = next(r_iter)
                self.optimizer.zero_grad()
                loss = self._gd_step_loss(f_batch, r_batch)
                loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                total_loss += loss.item()
                n_steps += 1
                total_steps += 1

            avg_loss = total_loss / max(n_steps, 1)
            epoch_time = time.time() - epoch_start
            total_unlearn_time += epoch_time

            fa_score, ra_score, ta_score, mia_score = self.evaluate()
            self.model.train()

            if fa_threshold < fa_score < closest_fa_score:
                closest_fa_score = fa_score
                torch.save(self.model.state_dict(), f"{ckpt_path}_closest_fa.pt")
            if fa_score <= fa_threshold:
                early_stop = True

            print(f"[GRIP-B] epoch [{epoch + 1}/{self.num_epoch}] | loss: {avg_loss:.4f} | "
                  f"ra: {ra_score*100:.2f}% | fa: {fa_score*100:.2f}% | "
                  f"ta: {ta_score*100:.2f}% | mia: {mia_score:.4f}")

            wandb.log({
                "epoch": epoch + 1, "unlearn_loss": avg_loss,
                "ra": ra_score, "fa": fa_score, "ta": ta_score, "mia": mia_score,
            })
            torch.save(self.model.state_dict(), f"{ckpt_path}_epoch_{epoch + 1}.pt")
            torch.save(self.model.state_dict(), f"{ckpt_path}.pt")

            if early_stop:
                break

        applied = 0
        if routers:
            print("[GRIP-B] base training done; capturing post-unlearning retain reps and "
                  "applying the closed-form router correction.")
            X_unlearned = self._capture_retain_reps(routers)
            for name, router, _ in routers:
                if name not in X_original or name not in X_unlearned:
                    continue
                if _apply_ptc_correction(
                    router=router, Theta_original=Theta_original[name],
                    X_original=X_original[name], X_unlearned=X_unlearned[name],
                    ridge=self.ptc_ridge, acceptance_cosine=self.ptc_acceptance_cosine,
                    layer_name=name,
                ):
                    applied += 1
            print(f"[GRIP-B] applied correction to {applied}/{len(routers)} router(s).")

            fa_score, ra_score, ta_score, mia_score = self.evaluate()
            print(f"[GRIP-B] post-correction | ra: {ra_score*100:.2f}% | fa: {fa_score*100:.2f}% | "
                  f"ta: {ta_score*100:.2f}% | mia: {mia_score:.4f}")
            wandb.log({
                "grip/routers_corrected": applied, "grip/routers_total": len(routers),
                "fa_post_correction": fa_score, "ra_post_correction": ra_score,
                "ta_post_correction": ta_score, "mia_post_correction": mia_score,
            })

        peak_memory_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
        wandb.log({"peak_memory_gb": peak_memory_gb})
        torch.save(self.model.state_dict(), f"{ckpt_path}.pt")
        return total_unlearn_time

    # -------------------------------------------------------------------
    # Method A: Training-Time Enforcement
    # -------------------------------------------------------------------
    def _unlearn_method_a(self, routers, fa_threshold, ckpt_path):
        constraints: Dict[str, List[_ExpertConstraint]] = {}
        if routers:
            print(f"[GRIP-A] capturing retain-set router inputs ({self.retain_cache_size} images) "
                  f"and precomputing per-expert constraint operators.")
            reps = self._capture_retain_reps(routers)
            for name, router, _ in routers:
                if name not in reps:
                    continue
                X = reps[name]
                S_pre = _compute_pre_selections(router, X, self.top_k_experts)
                constraints[name] = _precompute_constraints_for_router(
                    router=router, X=X, S_pre=S_pre,
                    top_k=self.top_k_experts, epsilon_null=self.epsilon_null,
                )
                r_sizes = [c.U_ret.shape[1] for c in constraints[name]]
                print(f"[GRIP-A] {name}: E={router.weight.shape[0]} experts, "
                      f"U_ret ranks min={min(r_sizes)} mean={sum(r_sizes)/max(len(r_sizes),1):.1f} max={max(r_sizes)}")

        # torch.multinomial (used by the Kaczmarz sampler below) requires the
        # generator's device to match the tensor's device -- a CPU generator
        # against CUDA tensors raises a RuntimeError. The original bundle
        # hardcoded device="cpu" here, which crashes on any real GPU run.
        generator = torch.Generator(device=self.device)
        generator.manual_seed(0)

        total_unlearn_time, total_steps = 0.0, 0
        early_stop, closest_fa_score = False, float("inf")
        r_iter = self._cycle(self.retain_loader)

        for epoch in range(self.num_epoch):
            epoch_start = time.time()
            self.model.train()
            total_loss, n_steps = 0.0, 0

            for f_batch in self.forget_loader:
                r_batch = next(r_iter)
                self.optimizer.zero_grad()
                loss = self._gd_step_loss(f_batch, r_batch)
                loss.backward()
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                pre_weights = {
                    name: router.weight.data.detach().clone()
                    for name, router, _ in routers if name in constraints
                }
                self.optimizer.step()

                with torch.no_grad():
                    for name, router, _ in routers:
                        if name not in constraints:
                            continue
                        W_old = pre_weights[name]
                        W_new = router.weight.data
                        delta = W_new - W_old
                        projected_delta = _project_router_update(
                            delta, constraints[name], self.epsilon_margin, self.kaczmarz_max_iters, generator,
                        )
                        router.weight.data.copy_(W_old + projected_delta)

                total_loss += loss.item()
                n_steps += 1
                total_steps += 1

            avg_loss = total_loss / max(n_steps, 1)
            epoch_time = time.time() - epoch_start
            total_unlearn_time += epoch_time

            fa_score, ra_score, ta_score, mia_score = self.evaluate()
            self.model.train()

            if fa_threshold < fa_score < closest_fa_score:
                closest_fa_score = fa_score
                torch.save(self.model.state_dict(), f"{ckpt_path}_closest_fa.pt")
            if fa_score <= fa_threshold:
                early_stop = True

            print(f"[GRIP-A] epoch [{epoch + 1}/{self.num_epoch}] | loss: {avg_loss:.4f} | "
                  f"ra: {ra_score*100:.2f}% | fa: {fa_score*100:.2f}% | "
                  f"ta: {ta_score*100:.2f}% | mia: {mia_score:.4f}")

            wandb.log({
                "epoch": epoch + 1, "unlearn_loss": avg_loss,
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
