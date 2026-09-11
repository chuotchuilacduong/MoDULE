"""
SSD baseline for the MoDULE unlearning benchmark.

Ported from if-loops/selective-synaptic-dampening ("Fast Machine Unlearning
Without Retraining Through Selective Synaptic Dampening", AAAI/ICLR-TP),
specifically the "clean implementation" in src/ssd.py::ParameterPerturber.

Method (no gradient-descent training loop at all):
  1. Compute a Fisher-like importance for every parameter -- the average
     squared gradient of the standard classification loss -- once over the
     forget set, once over the FULL training set (not just retain: this is
     the "how generally important is this weight overall" baseline).
  2. For any parameter where its forget-set importance exceeds
     `selection_weighting` times its overall importance (i.e. it's
     disproportionately tied to the samples being forgotten), multiply it by
     a dampening factor derived from the ratio of the two importances,
     capped at `lower_bound` so weights never get amplified past their
     original value -- only shrunk toward zero.

This is a purely deterministic, single-shot weight edit (like POUR-P): no
optimizer.step() is ever called. Unlike POUR/ERM-KTP/SCADA, it has no
class-specific structure -- it works identically for unlearn_setting:
random, class, or domain, since it only needs forget_loader and train_loader.

Adaptation vs. the original repo:
  - Batch unpacking: the original repo's datasets.py yields 3-tuples
    (image, _, label) (an unused index column). MoDULE's loaders yield plain
    (image, label) 2-tuples, so calc_importance() unpacks batch[0]/batch[1]
    directly.
  - Model call: the original calls the model as a plain callable expecting
    raw logits. MoDULE architectures return (logits, features); adapted to
    unpack accordingly.
  - min_layer/max_layer/forget_threshold/magnitude_diff from the original
    parameters dict are dead code in modify_weight() even in the upstream
    "clean implementation" (never referenced there) -- dropped here rather
    than carried along as unused config surface.
  - train_loader fallback: unlearn.py never constructs a full training-set
    loader for the unlearn phase (train_loader=None is passed uniformly to
    every algorithm wrapper -- that loader only exists in learn.py's base
    training run), but SSD's "overall importance" baseline explicitly needs
    one. When self.train_loader is None, it's synthesized here from
    forget_loader + retain_loader -- together they ARE the full training set
    the base checkpoint was trained on, which is exactly the paper's
    "how generally important is this weight overall" premise.
"""
import time
import torch
import torch.nn as nn
import wandb

from approx_algo.gradient_ascent import Gradient_Ascent


class SSD(Gradient_Ascent):
    def __init__(self, *args, dampening_constant=1.0, selection_weighting=10.0,
                 lower_bound=1.0, exponent=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.dampening_constant = dampening_constant
        self.selection_weighting = selection_weighting
        self.lower_bound = lower_bound
        self.exponent = exponent

    def _zerolike_params(self):
        return {name: torch.zeros_like(p, device=p.device) for name, p in self.model.named_parameters()}

    def _calc_importance(self, batches, num_batches):
        """Average (over batches) squared gradient of the CE loss, per parameter.

        `batches` is any iterable of (images, labels, ...) batches; `num_batches`
        is passed separately since the synthesized forget+retain fallback (see
        module docstring) isn't a single sized loader.
        """
        self.model.eval()
        importances = self._zerolike_params()
        for batch in batches:
            images, labels = batch[0].to(self.device), batch[1].to(self.device)
            self.optimizer.zero_grad()
            logits, _ = self.model(images)
            loss = self.criteria(logits, labels)
            loss.backward()
            for (name, p) in self.model.named_parameters():
                if p.grad is not None:
                    importances[name] += p.grad.data.clone().pow(2)
        for name in importances:
            importances[name] /= float(num_batches)
        self.optimizer.zero_grad()
        return importances

    def _full_train_batches(self):
        """The "full training set" the paper's overall-importance baseline
        needs. unlearn.py passes train_loader=None uniformly (see module
        docstring), so fall back to forget_loader + retain_loader, which
        together cover everything the loaded checkpoint was trained on."""
        if self.train_loader is not None:
            return self.train_loader, len(self.train_loader)

        def combined():
            yield from self.forget_loader
            yield from self.retain_loader
        return combined(), len(self.forget_loader) + len(self.retain_loader)

    def _modify_weight(self, original_importance, forget_importance):
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                oimp = original_importance[name]
                fimp = forget_importance[name]

                # Synapse Selection with parameter alpha (selection_weighting)
                oimp_norm = oimp.mul(self.selection_weighting)
                locations = torch.where(fimp > oimp_norm)

                # Synapse Dampening with parameter lambda (dampening_constant)
                weight = ((oimp.mul(self.dampening_constant)).div(fimp + 1e-12)).pow(self.exponent)
                update = weight[locations]
                # bounded by lower_bound so parameters only ever shrink, never grow
                min_locs = torch.where(update > self.lower_bound)
                update[min_locs] = self.lower_bound
                p[locations] = p[locations].mul(update)

    def unlearn(self, fa_threshold, ckpt_path):
        start = time.time()

        forget_importance = self._calc_importance(self.forget_loader, len(self.forget_loader))
        full_train_batches, n_full_train = self._full_train_batches()
        original_importance = self._calc_importance(full_train_batches, n_full_train)
        self._modify_weight(original_importance, forget_importance)

        total_unlearn_time = time.time() - start
        print(f"[*] SSD: dampened salient parameters in {total_unlearn_time:.4f}s "
              f"(2 importance passes, no gradient-descent steps).")

        fa_score, ra_score, ta_score, mia_score = self.evaluate()
        print(f"--> Metrics: RA: {ra_score*100:.2f}% | FA: {fa_score*100:.2f}% | "
              f"TA: {ta_score*100:.2f}% | MIA: {mia_score:.4f}")

        peak_memory_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
        wandb.log({
            "epoch": 1, "ra": ra_score, "fa": fa_score, "ta": ta_score, "mia": mia_score,
            "total_unlearn_time_sec": total_unlearn_time, "peak_memory_gb": peak_memory_gb,
        })

        torch.save(self.model.state_dict(), f"{ckpt_path}_epoch_1.pt")
        torch.save(self.model.state_dict(), f"{ckpt_path}.pt")
        return total_unlearn_time
