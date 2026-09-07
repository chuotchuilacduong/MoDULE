"""
QMUL baseline for the MoDULE unlearning benchmark.

Ported from Green-Edge-AI-Group/MU-Q-MUL ("Robust Machine Unlearning for
Quantized Neural Networks via Adaptive Gradient Reweighting with Similar
Labels", ICCV 2025), specifically its `RL_min` method.

Two ideas, both ported:

  1. "Similar labels" relabeling: instead of assigning each forget-set sample
     a uniformly random wrong label (as in plain Random Labeling / SalUn),
     assign it the label whose predicted probability is *closest* to the
     ground-truth class's probability (excluding the ground truth itself) --
     i.e. the most "confusable" alternative under the current model. The
     intuition is that pushing toward a label the model already considers
     plausible is a gentler, more stable unlearning signal than pushing
     toward an arbitrary random one.

  2. Adaptive gradient reweighting: before each epoch, measure the average
     per-batch gradient norm on the forget set and on the retain set
     separately (using the plain, unweighted criterion), then weight the
     combined loss inversely to whichever side has the larger norm -- so
     forget and retain updates stay balanced instead of one side dominating.

What's NOT ported: the paper's actual subject (quantized weights/activations)
-- MoDULE doesn't quantize its architectures, and neither of the two ideas
above requires it; they're general-purpose stabilization techniques the
paper happens to validate under quantization noise.

Adaptation vs. the original repo: the original repo relabels the forget set
by mutating the underlying Dataset's `.targets`/`.labels` array once per
epoch (dataset-attribute-name-specific, and awkward with MoDULE's
Subset+ApplyTransform wrapping). Instead, relabeling happens on-the-fly per
batch inside the training loop -- matching how MoDULE's own approx_algo/salun.py
already does its random relabeling. Functionally equivalent, no dataset
introspection needed.

Fix vs. the ported source: `_avg_grad_norm` and `_compute_saliency_map` are
meant to be read-only measurement passes (their only purpose is to read
`.grad` values; neither ever calls `optimizer.step()`), and both correctly
call `self.model.eval()` up front for that reason. But the original code
then measured gradients through `self.model.forward_with_grad(images)`,
which unconditionally calls `self.train()` internally -- silently undoing
the `.eval()` call from the very first batch onward. For a BatchNorm
architecture (e.g. the original bundle's own example config, model_name:
resnet50 -- this repo's config/experiments/pacs_qmul.yaml instead targets
module_small_patch16_224 for consistency with the other baselines)
that means every "measurement" pass was quietly updating BatchNorm running
statistics from forget/retain images, and for a MoE architecture it means
DeepMoELayer's training-only router noise leaked into the gradient-norm /
saliency measurement. Fixed below by calling the model directly (`self.model
(images)`, honoring whatever mode `.eval()`/`.train()` set beforehand)
instead of `forward_with_grad`.

Performance note: `_adaptive_weights` runs once per epoch and previously did
a full forward+backward scan over BOTH forget_loader and retain_loader just
to produce two scalars, on top of the real training loop below (whose
iteration count is only len(forget_loader)). retain_loader is typically much
larger (e.g. forget_ratio=0.1 -> ~9x), so that measurement alone could cost
more than the rest of the epoch combined. By default the retain-side
measurement is now capped to len(forget_loader) batches -- see
`_adaptive_weights` for the reasoning; pass `full_norm_scan=True` to restore
the original exhaustive-scan behavior.

Optionally supports layering a SalUn-style weight-saliency mask on top
(mask_ratio != None), exactly like the original repo's RL_min does when
called with a `mask` argument from generate_mask.py -- reuses the same
AdamW-safe restore mechanism as approx_algo/salun.py.
"""
import time
import torch
import torch.nn.functional as F
import wandb

from approx_algo.gradient_ascent import Gradient_Ascent


class QMUL(Gradient_Ascent):
    def __init__(self, *args, use_similar_labels=True, mask_ratio=None,
                 full_norm_scan=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_similar_labels = use_similar_labels
        self.mask_ratio = mask_ratio
        self.full_norm_scan = full_norm_scan
        self.masks = {}
        self.theta0 = {}

    # ------------------------------------------------------------------ #
    # "Similar labels": most-confusable wrong class per forget sample.
    # ------------------------------------------------------------------ #
    def _confusable_labels(self, images, labels):
        with torch.no_grad():
            logits, _ = self.model.inference(images)
            probs = F.softmax(logits, dim=-1)
            gt_prob = probs.gather(1, labels.unsqueeze(1))
            distance = torch.abs(probs - gt_prob)
            # exclude the ground-truth class itself from the search
            distance.scatter_(1, labels.unsqueeze(1), float('inf'))
            confusable = distance.argmin(dim=1)
        return confusable

    # ------------------------------------------------------------------ #
    # Adaptive gradient reweighting between forget/retain.
    # ------------------------------------------------------------------ #
    def _avg_grad_norm(self, loader, max_batches=None):
        # Read-only measurement pass: stays in eval() throughout (calling
        # the model directly rather than via forward_with_grad, which would
        # force train() and leak into BatchNorm running stats / MoE router
        # noise -- see module docstring). Still computes real gradients:
        # eval() only changes forward-pass behavior, not autograd.
        self.model.eval()
        norms = []
        # Check the cap BEFORE pulling the next batch (rather than
        # enumerate()+break) so an exhausted cap never triggers one extra
        # __getitem__/collate from the underlying DataLoader.
        loader_iter = iter(loader)
        count = 0
        while max_batches is None or count < max_batches:
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            images, labels = batch[0].to(self.device), batch[1].to(self.device)
            logits, _ = self.model(images)
            loss = self.criteria(logits, labels)
            self.optimizer.zero_grad()
            loss.backward()
            norm = sum(p.grad.norm(2).item() for p in self.model.parameters() if p.grad is not None)
            norms.append(norm)
            count += 1
        self.optimizer.zero_grad()
        return sum(norms) / max(len(norms), 1)

    def _adaptive_weights(self):
        # This measurement runs once per epoch and its only output is two
        # scalars (forget_weight, retain_weight) -- but a naive full scan
        # costs a complete extra forward+backward pass over BOTH loaders,
        # on top of the real training loop below (whose iteration count is
        # only len(forget_loader), since it cycles retain_loader alongside
        # each forget batch via next(retain_iter)). retain_loader is
        # typically much larger than forget_loader (e.g. forget_ratio=0.1
        # -> ~9x), so the retain-side measurement alone can cost more than
        # the entire rest of the epoch. Capping it to len(forget_loader)
        # batches brings this measurement's cost in line with the training
        # loop it feeds, while still giving a fresh (retain_loader reshuffles
        # each epoch when shuffle=True) sample every epoch. forget_loader
        # itself is left uncapped -- it's already the smaller side, and it's
        # the set whose signal matters more here. Set full_norm_scan=True to
        # restore the original exhaustive-scan behavior if you want the most
        # stable possible estimate regardless of cost.
        forget_norm = self._avg_grad_norm(self.forget_loader)
        retain_cap = None if self.full_norm_scan else len(self.forget_loader)
        retain_norm = self._avg_grad_norm(self.retain_loader, max_batches=retain_cap)
        total = forget_norm + retain_norm
        if total == 0:
            return 0.5, 0.5
        if forget_norm > retain_norm:
            retain_weight = forget_norm / total
            forget_weight = retain_norm / total
        else:
            forget_weight = retain_norm / total
            retain_weight = forget_norm / total
        return forget_weight, retain_weight

    # ------------------------------------------------------------------ #
    # Optional SalUn-style saliency mask (see approx_algo/salun.py for the
    # AdamW-momentum-drift rationale behind the restore step).
    # ------------------------------------------------------------------ #
    def _compute_saliency_map(self):
        # Same read-only-pass rationale as _avg_grad_norm above.
        self.model.eval()
        self.optimizer.zero_grad()
        for batch in self.forget_loader:
            images, labels = batch[0].to(self.device), batch[1].to(self.device)
            logits, _ = self.model(images)
            self.criteria(logits, labels).backward()

        param_names = [name for name, p in self.model.named_parameters() if p.grad is not None]
        flat_grads = torch.cat([
            self.model.get_parameter(name).grad.detach().abs().view(-1) for name in param_names
        ])
        if flat_grads.numel() == 0:
            return
        threshold_index = int(flat_grads.numel() * self.mask_ratio)
        ranks = torch.argsort(torch.argsort(-flat_grads))
        hard_mask = (ranks < threshold_index).float()

        start = 0
        for name in param_names:
            numel = self.model.get_parameter(name).numel()
            self.masks[name] = hard_mask[start:start + numel].view_as(self.model.get_parameter(name))
            start += numel

        with torch.no_grad():
            self.theta0 = {name: self.model.get_parameter(name).detach().clone() for name in self.masks}
        self.optimizer.zero_grad()

    def _restore_masked_params(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name not in self.masks:
                    continue
                mask = self.masks[name].to(device=param.device, dtype=param.dtype)
                if torch.count_nonzero(1 - mask) == 0:
                    continue
                param.data.mul_(mask).add_(self.theta0[name].to(param.device) * (1 - mask))
                state = self.optimizer.state.get(param, None)
                if state is None:
                    continue
                for key in ("momentum_buffer", "exp_avg", "exp_avg_sq"):
                    if key in state and state[key] is not None:
                        state[key].mul_(mask)

    # ------------------------------------------------------------------ #
    def unlearn(self, fa_threshold, ckpt_path):
        if self.mask_ratio is not None:
            self._compute_saliency_map()

        total_unlearn_time = 0.0
        for epoch in range(self.num_epoch):
            epoch_start = time.time()

            forget_weight, retain_weight = self._adaptive_weights()
            print(f"[*] QMUL epoch {epoch+1}: forget_weight={forget_weight:.4f} "
                  f"retain_weight={retain_weight:.4f}")

            self.model.train()
            total_loss = 0.0
            retain_iter = iter(self.retain_loader)

            for forget_batch in self.forget_loader:
                try:
                    retain_batch = next(retain_iter)
                except StopIteration:
                    retain_iter = iter(self.retain_loader)
                    retain_batch = next(retain_iter)

                images_f = forget_batch[0].to(self.device)
                labels_f = forget_batch[1].to(self.device)
                images_r = retain_batch[0].to(self.device)
                labels_r = retain_batch[1].to(self.device)

                if self.use_similar_labels:
                    new_labels_f = self._confusable_labels(images_f, labels_f)
                else:
                    logits_probe, _ = self.model.inference(images_f)
                    num_classes = logits_probe.shape[1]
                    shifts = torch.randint(1, num_classes, labels_f.shape, device=self.device)
                    new_labels_f = (labels_f + shifts) % num_classes

                images = torch.cat([images_f, images_r], dim=0)
                targets = torch.cat([new_labels_f, labels_r], dim=0)
                weights = torch.cat([
                    torch.full_like(labels_f, forget_weight, dtype=torch.float),
                    torch.full_like(labels_r, retain_weight, dtype=torch.float),
                ])

                self.optimizer.zero_grad()
                logits, _ = self.model.forward_with_grad(images)
                per_sample_loss = F.cross_entropy(logits, targets, reduction='none')
                loss = (weights * per_sample_loss).sum() / weights.numel()
                loss.backward()

                if self.mask_ratio is not None:
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and name in self.masks:
                            param.grad.data *= self.masks[name]

                self.optimizer.step()

                if self.mask_ratio is not None:
                    self._restore_masked_params()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.forget_loader)
            epoch_time = time.time() - epoch_start
            total_unlearn_time += epoch_time

            print(f"[*] evaluating epoch {epoch+1}...")
            fa_score, ra_score, ta_score, mia_score = self.evaluate()
            print(f"--> Epoch [{epoch+1}/{self.num_epoch}] | Time: {epoch_time:.2f}s | "
                  f"Reweighted Loss: {avg_loss:.4f}")
            print(f"--> Metrics: RA: {ra_score*100:.2f}% | FA: {fa_score*100:.2f}% | "
                  f"TA: {ta_score*100:.2f}% | MIA: {mia_score:.4f}")
            print("-" * 40)

            wandb.log({
                "epoch": epoch + 1, "unlearn_loss": avg_loss,
                "forget_weight": forget_weight, "retain_weight": retain_weight,
                "ra": ra_score, "fa": fa_score, "ta": ta_score, "mia": mia_score,
            })
            torch.save(self.model.state_dict(), f"{ckpt_path}_epoch_{epoch+1}.pt")

            if fa_score <= fa_threshold:
                print(f"[*] early stopping triggered at epoch {epoch+1} (FA <= {fa_threshold})")
                break

        peak_memory_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
        wandb.log({"total_unlearn_time_sec": total_unlearn_time, "peak_memory_gb": peak_memory_gb})
        torch.save(self.model.state_dict(), f"{ckpt_path}.pt")
        return total_unlearn_time
