"""
SEUF: Selected-Expert Unlearning Framework.

Port of Zhuang et al., ACL 2025 ("SEUF: Is Unlearning One Expert Enough for
Mixture-of-Experts LLMs?") adapted from LLM unlearning (WMDP/RWKU with GA on
Qwen1.5-MoE and DeepSeek-V2-Lite) to the vision MoE setting used by this
repository (PACS/OfficeHome with ModuleArchitecture).

The framework has three components (paper Alg. 1):

  1. Expert attribution: one-shot, offline. Compute per-(layer, expert)
     affinity as the mean gating score over a calibration subset of the
     forget set, then select the top-M experts (paper: M=1 optimal).

  2. Selective parameter freezing: only the target expert weights and, if
     enabled, their layer's router are trainable. Everything else is frozen.
     Reproduces the paper's 0.06% tunable-parameter ratio for MoE LLMs.

  3. Router anchor loss (Eq. 3): L2 loss between the router's post-softmax
     output and a target one-hot (or 1/M) vector on the selected expert set.
     Prevents the expert-selection shift that causes non-target experts to
     be inadvertently unlearned.

The base forget/retain losses (GA or GDiff) are configurable via
`seuf_forget_loss` -- SEUF is framework-agnostic in the paper, so any of the
existing MoDULE forget losses could plug in here in principle.

Unlike `Module` in this repo, SEUF runs expert attribution ONCE at the start
of unlearning rather than recomputing per epoch. This is deliberate: the
paper (Fig. 3) shows selection stability is the mechanism, not per-epoch
re-selection.
"""
import copy
import inspect
import time

import torch
import torch.nn.functional as F
import wandb

from approx_algo.gradient_ascent import Gradient_Ascent


class SEUF(Gradient_Ascent):
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
        # ---- SEUF-specific hyperparameters ----
        seuf_M=1,                       # top-M experts to unlearn per layer (paper: M=1 optimal)
        seuf_scope="same_layer",        # "same_layer" or "cross_layer" (paper: same_layer for M>1)
        seuf_alpha=1.0,                 # anchor loss weight (paper Eq. 4, alpha; paper: alpha=1 optimal)
        seuf_lambda=1.0,                # retain loss weight (paper Eq. 4, lambda)
        seuf_calibration_batches=None,  # cap the attribution pass; None = use full forget_loader
        seuf_forget_loss="ga",          # "ga" (gradient ascent) or "gdiff" (gradient difference)
        seuf_include_router=True,       # unfreeze router along with target expert
        seuf_include_head=False,        # unfreeze classification head too (off by paper default)
        seuf_calibration_loader=None,   # optional explicit calibration loader; falls back to forget_loader
    ):
        super().__init__(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            unseen_loader=unseen_loader,
            forget_loader=forget_loader,
            forget_test_loader=forget_test_loader,
            retain_loader=retain_loader,
            retain_test_loader=retain_test_loader,
            optimizer=optimizer,
            criteria=criteria,
            num_epoch=num_epoch,
            device=device,
        )

        # SEUF-specific
        self.seuf_M = seuf_M
        self.seuf_scope = seuf_scope
        self.seuf_alpha = seuf_alpha
        self.seuf_lambda = seuf_lambda
        self.seuf_calibration_batches = seuf_calibration_batches
        self.seuf_forget_loss = seuf_forget_loss
        self.seuf_include_router = seuf_include_router
        self.seuf_include_head = seuf_include_head
        self.seuf_calibration_loader = seuf_calibration_loader

    # ----------------------------------------------------------------------
    # Phase 1: expert attribution (offline, one-shot)
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def _seuf_attribution(self, calibration_loader):
        """
        Paper Eq. 2. For each (layer, expert), average the post-softmax
        gating score across all tokens of a calibration subset of the
        forget set. Rank per layer, take top-M.

        Returns:
            selected: dict[layer_idx -> list[expert_idx]]
            affinity: list of [num_experts] cpu tensors, one per MoE layer
        """
        self.model.eval()
        moe_layers = self._get_moe_layers()
        L = len(moe_layers)

        affinity_sums = [None] * L
        token_counts = [0] * L
        n_batches = 0

        for batch in calibration_loader:
            if (self.seuf_calibration_batches is not None
                    and n_batches >= self.seuf_calibration_batches):
                break
            images = batch[0].to(self.device)
            self.model(images)  # populates .last_pi_all on each MoE layer

            for l, m in enumerate(moe_layers):
                pi = m.last_pi_all  # [n_tokens, num_experts]
                batch_sum = pi.sum(dim=0)
                affinity_sums[l] = batch_sum if affinity_sums[l] is None \
                    else affinity_sums[l] + batch_sum
                token_counts[l] += pi.size(0)
            n_batches += 1

        # affinity[l][e] = mean gating score of expert e at layer l over the
        # calibration tokens; sums to 1 across experts (each row of pi did).
        affinity = [(a / max(t, 1)).cpu() for a, t in zip(affinity_sums, token_counts)]

        selected = self._select_top_M(affinity)
        return selected, affinity, n_batches

    def _select_top_M(self, affinity):
        """
        Two selection strategies from the paper (Sec. 4, "Selection of top-M"):

        - same_layer:   For each layer, take the top-M experts of that layer.
                        Every layer gets M target experts.
        - cross_layer:  Take the top-M (layer, expert) pairs across ALL
                        layers. Some layers may end up with zero target
                        experts (and thus stay fully frozen).

        Paper finding: same_layer > cross_layer when M > 1, because
        cross-layer gradient updates disrupt feature hierarchies (Insight 4).
        """
        L = len(affinity)
        if self.seuf_scope == "same_layer":
            return {l: affinity[l].topk(self.seuf_M).indices.tolist()
                    for l in range(L)}

        if self.seuf_scope == "cross_layer":
            # Flatten (layer, expert) pairs; take global top-M
            flat_vals = []
            flat_meta = []  # list of (layer, expert)
            for l, a in enumerate(affinity):
                for e in range(a.numel()):
                    flat_vals.append(float(a[e]))
                    flat_meta.append((l, e))
            top_idx = sorted(range(len(flat_vals)),
                             key=lambda i: flat_vals[i], reverse=True)[:self.seuf_M]
            selected = {}
            for i in top_idx:
                l, e = flat_meta[i]
                selected.setdefault(l, []).append(e)
            # Layers not in `selected` get an empty list -> stay frozen
            for l in range(L):
                selected.setdefault(l, [])
            return selected

        raise ValueError(
            f"Unknown seuf_scope: {self.seuf_scope!r}. "
            f"Expected 'same_layer' or 'cross_layer'."
        )

    # ----------------------------------------------------------------------
    # Phase 2 setup: selective parameter freezing
    # ----------------------------------------------------------------------
    def _apply_seuf_freezing(self, selected_per_layer, moe_layers):
        """
        Freeze everything, then selectively unfreeze:
          - target expert weights (always)
          - the target expert's layer router (if seuf_include_router)
          - classifier head (if seuf_include_head; off by default per paper)

        Overrides whatever ModuleArchitecture._set_grad_mode("unlearning")
        set up before this wrapper was constructed (that default freezes the
        router and the dense-block MLPs entirely) -- SEUF needs its own,
        finer-grained freezing plan.
        """
        for param in self.model.parameters():
            param.requires_grad = False

        for l, m in enumerate(moe_layers):
            selected = selected_per_layer.get(l, [])
            if not selected:
                continue
            for e in selected:
                for param in m.experts[e].parameters():
                    param.requires_grad = True
            if self.seuf_include_router:
                for param in m.router.parameters():
                    param.requires_grad = True

        if self.seuf_include_head and hasattr(self.model, "classifier_head"):
            for param in self.model.classifier_head.parameters():
                param.requires_grad = True

    # ----------------------------------------------------------------------
    # SEUF-specific losses
    # ----------------------------------------------------------------------
    def _anchor_loss(self, moe_layers, selected_per_layer):
        """
        Paper Eq. 3:
            L_anchor^(l) = || g^(l) - a^(l) ||_2^2

        where g^(l) is the router's post-softmax output on the current batch
        and a^(l) is the target vector with 1.0 on the selected expert (for
        M=1) or 1/M on each selected expert (for M>1, giving a valid target
        distribution).

        Averaged over tokens within a layer, then averaged across layers
        that have selected experts. Layers with an empty selected set
        contribute zero (they stay frozen anyway).
        """
        loss = torch.tensor(0.0, device=self.device)
        n_layers = 0

        for l, m in enumerate(moe_layers):
            selected = selected_per_layer.get(l, [])
            if not selected:
                continue

            pi = m.last_pi_all  # [n_tokens, num_experts], post-softmax
            a = torch.zeros(m.num_experts, device=self.device, dtype=pi.dtype)
            for e in selected:
                a[e] = 1.0 / len(selected)  # sums to 1

            # Per-token squared L2, then mean over tokens.
            per_token_l2sq = (pi - a.unsqueeze(0)).pow(2).sum(dim=-1)
            loss = loss + per_token_l2sq.mean()
            n_layers += 1

        if n_layers > 0:
            loss = loss / n_layers
        return loss

    def _forget_loss(self, logits_f, labels_f):
        """
        Paper is agnostic to the base unlearning algorithm. We support:
          - ga:     gradient ascent  (-CE on forget set)
          - gdiff:  gradient difference (same forget term; retain acts as pull)
        For gdiff the retain loss elsewhere provides the differencing signal.
        """
        if self.seuf_forget_loss in ("ga", "gdiff"):
            return -self.criteria(logits_f, labels_f)
        raise ValueError(f"Unknown seuf_forget_loss: {self.seuf_forget_loss!r}")

    def _retain_loss(self, logits_r, labels_r):
        return self.criteria(logits_r, labels_r)

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def _get_moe_layers(self):
        return [m for _, m in self.model.named_modules()
                if m.__class__.__name__ == "DeepMoELayer"]

    def _report_tunable(self):
        total = sum(p.numel() for p in self.model.parameters())
        tunable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        ratio = tunable / max(total, 1)
        print(f"[*] SEUF tunable params: {tunable:,} / {total:,} "
              f"({ratio * 100:.4f}%)")
        return ratio, tunable, total

    def _reinit_optimizer(self):
        """
        Rebuild the optimizer over only trainable parameters. Same pattern as
        Module.unlearn, needed because self.optimizer was built over the full
        parameter set at construction time and would otherwise hold frozen
        params in its state.
        """
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        opt_class = type(self.optimizer)
        valid_kwargs = inspect.signature(opt_class.__init__).parameters.keys()
        filtered = {k: v for k, v in self.optimizer.defaults.items()
                    if k in valid_kwargs}
        self.optimizer = opt_class(trainable, **filtered)

    def _log_attribution(self, selected_per_layer, affinity, n_batches):
        """One-shot wandb log of what SEUF attribution picked."""
        payload = {"seuf/attribution/calibration_batches": n_batches}
        for l, experts in selected_per_layer.items():
            for e in experts:
                payload[f"seuf/attribution/layer{l}/expert{e}/affinity"] = float(affinity[l][e])
                payload[f"seuf/attribution/layer{l}/expert{e}/selected"] = 1
        wandb.log(payload)

    # ----------------------------------------------------------------------
    # Main entry point
    # ----------------------------------------------------------------------
    def unlearn(self, fa_threshold, ckpt_path):
        # ---- Phase 1: attribution (one-shot, before any weight updates) ----
        calib_loader = self.seuf_calibration_loader or self.forget_loader
        print(f"[*] SEUF Phase 1: running expert attribution on "
              f"{'a custom loader' if self.seuf_calibration_loader else 'the forget loader'} "
              f"(M={self.seuf_M}, scope={self.seuf_scope})")
        selected_per_layer, affinity, n_batches = self._seuf_attribution(calib_loader)

        print(f"[*] SEUF attribution complete (used {n_batches} batch(es)):")
        for l, es in sorted(selected_per_layer.items()):
            if not es:
                print(f"    layer {l}: (none selected, layer stays frozen)")
                continue
            scores = ", ".join(f"e{e}={affinity[l][e]:.4f}" for e in es)
            print(f"    layer {l}: {scores}")
        self._log_attribution(selected_per_layer, affinity, n_batches)

        # ---- Phase 2 setup: freeze non-target params, rebuild optimizer ----
        moe_layers = self._get_moe_layers()
        self._apply_seuf_freezing(selected_per_layer, moe_layers)
        ratio, tunable, total = self._report_tunable()
        wandb.log({
            "seuf/tunable_param_ratio": ratio,
            "seuf/tunable_params": tunable,
            "seuf/total_params": total,
        })
        self._reinit_optimizer()

        # ---- Training loop ----
        total_unlearn_time = 0.0
        total_unlearn_steps = 0
        early_stop = False
        closest_fa_score = float("inf")

        print(f"[*] SEUF Phase 2: unlearning with forget_loss={self.seuf_forget_loss}, "
              f"alpha(anchor)={self.seuf_alpha}, lambda(retain)={self.seuf_lambda}")

        for epoch in range(self.num_epoch):
            epoch_start = time.time()
            self.model.train()

            loss_totals = {"total": 0.0, "forget": 0.0, "retain": 0.0, "anchor": 0.0}
            n_steps = 0
            retain_iter = iter(self.retain_loader) if self.retain_loader is not None else None

            for forget_batch in self.forget_loader:
                images_f = forget_batch[0].to(self.device)
                labels_f = forget_batch[1].to(self.device)

                self.optimizer.zero_grad()

                # Forward on forget batch -- populates .last_pi_all needed by anchor loss
                logits_f, _ = self.model.forward_with_grad(images_f)
                loss_forget = self._forget_loss(logits_f, labels_f)
                loss_anchor = self._anchor_loss(moe_layers, selected_per_layer)

                # Retain pass (optional). For GDiff, this is the paired retain
                # signal; for GA with lambda>0 it's a stability regularizer.
                loss_retain = torch.tensor(0.0, device=self.device)
                if retain_iter is not None:
                    try:
                        retain_batch = next(retain_iter)
                    except StopIteration:
                        retain_iter = iter(self.retain_loader)
                        retain_batch = next(retain_iter)
                    images_r = retain_batch[0].to(self.device)
                    labels_r = retain_batch[1].to(self.device)
                    logits_r, _ = self.model.forward_with_grad(images_r)
                    loss_retain = self._retain_loss(logits_r, labels_r)

                # Paper Eq. 4: L = L_forget + lambda * L_retain + alpha * L_anchor
                total_loss = loss_forget + self.seuf_lambda * loss_retain + self.seuf_alpha * loss_anchor
                total_loss.backward()
                self.optimizer.step()

                loss_totals["total"] += total_loss.item()
                loss_totals["forget"] += loss_forget.item()
                loss_totals["retain"] += loss_retain.item()
                loss_totals["anchor"] += loss_anchor.item()
                n_steps += 1
                total_unlearn_steps += 1

            avg = {k: v / max(n_steps, 1) for k, v in loss_totals.items()}
            epoch_time = time.time() - epoch_start
            total_unlearn_time += epoch_time

            print(f"[*] End of epoch {epoch + 1}. Running full evaluation...")
            fa_score, ra_score, ta_score, mia_score = self.evaluate()
            self.model.train()

            if fa_threshold < fa_score < closest_fa_score:
                closest_fa_score = fa_score
                print(f"[!] New closest FA at epoch end: {closest_fa_score * 100:.2f}%. "
                      f"Saving fallback checkpoint...")
                torch.save(self.model.state_dict(), f"{ckpt_path}_closest_fa.pt")

            if fa_score <= fa_threshold:
                print(f"[*] Target condition met (FA = {fa_score * 100:.2f}% <= "
                      f"{fa_threshold * 100:.2f}%).")
                early_stop = True

            print(f"--> Epoch [{epoch + 1}/{self.num_epoch}] | Time: {epoch_time:.2f}s | "
                  f"Loss: {avg['total']:.4f}")
            print(f"--> Losses: forget={avg['forget']:.4f} retain={avg['retain']:.4f} "
                  f"anchor={avg['anchor']:.4f}")
            print(f"--> Metrics: RA: {ra_score * 100:.2f}% | FA: {fa_score * 100:.2f}% | "
                  f"TA: {ta_score * 100:.2f}% | MIA: {mia_score:.4f}")
            print("-" * 40)

            wandb.log({
                "epoch": epoch + 1,
                "unlearn_loss": avg["total"],
                "seuf/forget_loss": avg["forget"],
                "seuf/retain_loss": avg["retain"],
                "seuf/anchor_loss": avg["anchor"],
                "fa": fa_score,
                "ra": ra_score,
                "ta": ta_score,
                "mia": mia_score,
                "unlearn_steps_accum": total_unlearn_steps,
            })

            torch.save(self.model.state_dict(), f"{ckpt_path}_epoch_{epoch + 1}.pt")
            torch.save(self.model.state_dict(), f"{ckpt_path}.pt")

            if early_stop:
                break

        print(f"[*] SEUF unlearning finished. Total Steps: {total_unlearn_steps} | "
              f"Total Time: {total_unlearn_time:.2f}s")

        peak_memory_gb = (torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)
                         if torch.cuda.is_available() else 0.0)
        wandb.log({
            "peak_memory_gb": peak_memory_gb,
        })

        torch.save(self.model.state_dict(), f"{ckpt_path}.pt")
        return total_unlearn_time
