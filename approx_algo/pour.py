"""
POUR baseline for the MoDULE unlearning benchmark.

Ported from ale256/representation_unlearning ("POUR: A Provably Optimal
Method for Unlearning Representations via Neural Collapse", CVPR 2026).
Two variants, both included here:

  POUR-P (`ETF_replace` in the original repo) -- a closed-form, single-shot
  operation with NO training. Under Neural Collapse, a well-trained
  classifier's weight vectors form a simplex Equiangular Tight Frame (ETF).
  To forget a class, build the orthogonal projector onto the complement of
  the subspace spanned by the forget class(es)' weight vector(s), apply it to
  every *other* class's weight vector, and zero out the forget class's own
  row/bias. No forward/backward pass over any data at all.

  POUR-D (`etf_finetune` in the original repo) -- propagates that same
  projection into the backbone via distillation: a frozen teacher (a copy of
  the pre-unlearning model) produces features on the *forget set*, those
  features get projected by the same P, and the student's feature extractor
  is fine-tuned to match the projected targets. Notably (and this is a
  faithful detail, not a MoDULE convention) the original method trains on
  forget_loader only -- retain data is explicitly not used, since the
  intent is to move the forget-class representations toward the "class
  doesn't exist" projection, not to also rehearse retain accuracy.

Only meaningful for unlearn_setting: class (forget_classes must be set) --
the projector is built from specific class weight vectors, which has no
analogue for a random subset or a whole domain. unlearn.py enforces this
before dispatch (same pattern as its ERM-KTP / SCADA checks).

Adaptation notes vs. the original repo:
  - Classifier lookup: the original repo searches named_modules() for a
    Linear layer ending in .fc/.classifier/.head (or falls back to the last
    Linear it finds). MoDULE's BaseArchitecture already exposes a
    `classifier_head` attribute uniformly, so we use that directly with the
    string-search as a fallback for any custom architecture that doesn't
    follow the convention. (Not applicable to architecture/spm.py -- SPM's
    classifier_head is a non-parametric memory-relation head, not a Linear
    with per-class weight vectors, so POUR has nothing to project there.)
  - Feature capture: the original repo uses a forward hook on the classifier
    to grab its input. MoDULE architectures already return (logits, features)
    from forward(), so no hook is needed.
  - Multi-class forget: the original functions take a single `forget_class:
    int`. MoDULE's `forget_classes` is a list, so the projector here is
    generalized to project out the whole subspace spanned by all forget
    class vectors: P = I - V^T (V V^T)^+ V (pseudo-inverse handles the case
    where forget vectors aren't exactly orthogonal). For a single forget
    class this reduces exactly to the original's P = I - v v^T.
"""
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from approx_algo.gradient_ascent import Gradient_Ascent


def _find_classifier_head(model):
    if hasattr(model, 'classifier_head') and isinstance(model.classifier_head, nn.Linear):
        return model.classifier_head
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.endswith(('.fc', '.classifier', '.head')):
            return module
    for name, module in reversed(list(model.named_modules())):
        if isinstance(module, nn.Linear):
            return module
    raise ValueError("POUR: could not locate a classifier (nn.Linear) layer in the model.")


def _build_projection(classifier, forget_classes, device):
    """P = I - V^T (V V^T)^+ V, projecting out span(forget class weight vectors)."""
    with torch.no_grad():
        W = classifier.weight.data.clone()             # [C, D]
        V = F.normalize(W[forget_classes], p=2, dim=1)  # [k, D]
        D = W.shape[1]
        gram = V @ V.T                                   # [k, k]
        gram_pinv = torch.linalg.pinv(gram)
        P = torch.eye(D, device=device) - V.T @ gram_pinv @ V
    return P


def _etf_loss(student_features, targets, loss_function):
    if loss_function == "cosine":
        f_s_n = F.normalize(student_features, p=2, dim=1)
        t_n = F.normalize(targets, p=2, dim=1)
        return 1.0 - (f_s_n * t_n).sum(dim=1).mean()
    elif loss_function == "mse_normalized":
        f_s_n = F.normalize(student_features, p=2, dim=1)
        t_n = F.normalize(targets, p=2, dim=1)
        return F.mse_loss(f_s_n, t_n)
    elif loss_function == "l2_regression":
        return F.mse_loss(student_features, targets)
    elif loss_function == "huber":
        return F.smooth_l1_loss(student_features, targets)
    elif loss_function == "angular":
        f_s_n = F.normalize(student_features, p=2, dim=1)
        t_n = F.normalize(targets, p=2, dim=1)
        cos_sim = torch.clamp((f_s_n * t_n).sum(dim=1), -1.0 + 1e-7, 1.0 - 1e-7)
        return torch.acos(cos_sim).mean()
    raise ValueError(f"Unknown loss_function: {loss_function}")


class POUR_P(Gradient_Ascent):
    """Closed-form, single-shot projection. No training loop -- unlearn() is O(1)."""

    def __init__(self, *args, forget_classes, **kwargs):
        super().__init__(*args, **kwargs)
        if not forget_classes:
            raise ValueError("POUR_P requires forget_classes (unlearn_setting: class).")
        self.forget_classes = forget_classes if isinstance(forget_classes, list) else [forget_classes]
        self.classifier = _find_classifier_head(self.model)

    def unlearn(self, fa_threshold, ckpt_path):
        start = time.time()

        P = _build_projection(self.classifier, self.forget_classes, self.device)
        with torch.no_grad():
            W = self.classifier.weight.data.clone()
            new_W = W @ P.T
            new_W[self.forget_classes] = 0.0
            self.classifier.weight.data.copy_(new_W)
            if self.classifier.bias is not None:
                self.classifier.bias.data[self.forget_classes] = 0.0

        total_unlearn_time = time.time() - start
        print(f"[*] POUR-P: projected out {len(self.forget_classes)} forget class direction(s) "
              f"in {total_unlearn_time:.4f}s -- no gradient steps taken.")

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


class POUR_D(Gradient_Ascent):
    """
    Distills the POUR-P projection into the backbone. Trains on forget_loader
    ONLY (this is faithful to the original repo, not an oversight -- see
    module docstring). retain_loader is still used for evaluate()/RA metric.
    """

    def __init__(self, *args, forget_classes, loss_function='l2_regression',
                 freeze_classifier=True, freeze_bn_stats=True, **kwargs):
        super().__init__(*args, **kwargs)
        if not forget_classes:
            raise ValueError("POUR_D requires forget_classes (unlearn_setting: class).")
        self.forget_classes = forget_classes if isinstance(forget_classes, list) else [forget_classes]
        self.loss_function = loss_function
        self.freeze_classifier = freeze_classifier
        self.freeze_bn_stats = freeze_bn_stats
        self.classifier = _find_classifier_head(self.model)

    def _set_bn_eval(self):
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()

    def unlearn(self, fa_threshold, ckpt_path):
        P = _build_projection(self.classifier, self.forget_classes, self.device)

        teacher = copy.deepcopy(self.model).to(self.device)
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()

        if self.freeze_classifier:
            for p in self.classifier.parameters():
                p.requires_grad_(False)

        params = [p for p in self.model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("POUR-D: no trainable parameters left to fine-tune "
                              "(check freeze_classifier).")
        # Uses its own SGD + cosine schedule, matching the original repo,
        # rather than the shared AdamW optimizer other MoDULE baselines use --
        # this method's convergence behavior was tuned against SGD in the paper.
        optimizer = torch.optim.SGD(params, lr=self.optimizer.param_groups[0]['lr'],
                                     momentum=0.9, weight_decay=5e-4, nesterov=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epoch)

        total_unlearn_time = 0.0
        for epoch in range(self.num_epoch):
            epoch_start = time.time()
            self.model.train()
            if self.freeze_bn_stats:
                self._set_bn_eval()

            total_loss = 0.0
            for batch in self.forget_loader:
                images = batch[0].to(self.device)

                with torch.no_grad():
                    _, f_t = teacher(images)
                    targets = f_t @ P.T

                logits_s, f_s = self.model(images)
                loss = _etf_loss(f_s, targets, self.loss_function)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            scheduler.step()
            avg_loss = total_loss / len(self.forget_loader)
            epoch_time = time.time() - epoch_start
            total_unlearn_time += epoch_time

            print(f"[*] evaluating epoch {epoch+1}...")
            fa_score, ra_score, ta_score, mia_score = self.evaluate()
            print(f"--> Epoch [{epoch+1}/{self.num_epoch}] | Time: {epoch_time:.2f}s | "
                  f"{self.loss_function} loss: {avg_loss:.4f}")
            print(f"--> Metrics: RA: {ra_score*100:.2f}% | FA: {fa_score*100:.2f}% | "
                  f"TA: {ta_score*100:.2f}% | MIA: {mia_score:.4f}")
            print("-" * 40)

            wandb.log({
                "epoch": epoch + 1, "pour_d_loss": avg_loss,
                "ra": ra_score, "fa": fa_score, "ta": ta_score, "mia": mia_score,
            })
            torch.save(self.model.state_dict(), f"{ckpt_path}_epoch_{epoch+1}.pt")

            if fa_score <= fa_threshold:
                print(f"[*] early stopping triggered at epoch {epoch+1} (FA <= {fa_threshold})")
                break

        if self.freeze_classifier:
            for p in self.classifier.parameters():
                p.requires_grad_(True)

        peak_memory_gb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3) if torch.cuda.is_available() else 0.0
        wandb.log({"total_unlearn_time_sec": total_unlearn_time, "peak_memory_gb": peak_memory_gb})
        torch.save(self.model.state_dict(), f"{ckpt_path}.pt")
        return total_unlearn_time
