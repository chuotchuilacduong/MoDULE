"""
SCADA baseline for the MoDULE unlearning benchmark.

Ported/adapted from D-Arnav/SCADA ("Source Models Leak What They Shouldn't:
Unlearning Zero-Shot Transfer in Domain Adaptation Through Adversarial
Optimization", CVPR 2026).

What's ported: the paper's actual unlearning mechanism -- for each forget
class, optimize a small set of *synthetic* images (`AdversarialSample`) so the
current classifier is highly confident they belong to that class, then train
the classifier to instead predict a "not-forget-class" pseudo-label on those
synthetic samples (mislabeling loss `mu_loss`), interleaved with ordinary
supervised steps on real data. This adversarially erodes the forget-class
decision region without ever touching real forget-set images.

What's intentionally dropped: the original repo runs this jointly with SFDA2
(a source-free domain-adaptation self-training loss with feature/prototype
memory banks, via the `tllib` library) because their setting is "unlearn a
class's zero-shot transfer from a source domain classifier onto a new target
domain." MoDULE's benchmark is single-dataset retain/forget (no separate
source/target domains to adapt between), so the domain-adaptation loss term is
replaced with a standard supervised loss on `retain_loader` -- the same role
`retain_loader` plays in your other baselines (finetune, scrub, sg_unlearn).
This keeps the paper's actual contribution (the adversarial minimax
mislabeling step) faithful while dropping ~800KB of unrelated `tllib` DA code.

Only meaningful for unlearn_setting: class (forget_classes must be set) -- it
has no notion of forgetting a random subset or a whole domain, since the
adversarial sample is optimized per target *class*. unlearn.py enforces this
before dispatch (same pattern as its ERM-KTP check); __init__ here only
double-checks forget_classes is non-empty in case this class is ever
constructed directly.
"""
import time
import torch
import torch.nn.functional as F
from torch.optim import Adam
import wandb

from approx_algo.gradient_ascent import Gradient_Ascent


class _ForeverLoader:
    """Cycles a DataLoader indefinitely (stand-in for tllib's ForeverDataIterator)."""
    def __init__(self, loader):
        self.loader = loader
        self._iter = iter(loader)

    def __next__(self):
        try:
            return next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            return next(self._iter)


class _Sample(torch.nn.Module):
    """A learnable batch of synthetic images (ported as-is from utils/utils.py)."""
    def __init__(self, num_samples, channels, size, device):
        super().__init__()
        self.param = torch.nn.Parameter(torch.randn(num_samples, channels, size, size, device=device))

    def forward(self):
        return self.param


class AdversarialSample:
    """
    Maintains a synthetic sample optimized to look like `adv_classes` to the
    current classifier. Ported from utils/utils.py::AdversarialSample, with
    the model call swapped from `classifier(x)` (SCADA's raw nn.Module) to
    `classifier(x)[0]` matching MoDULE architectures' (logits, features)
    forward contract.
    """
    def __init__(self, adv_classes, num_classes, num_samples, channels, size, device):
        self.adv_classes = adv_classes
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.device = device
        self.sample = _Sample(num_samples, channels, size, device)

    def _adv_labels(self):
        labels = torch.zeros(self.num_samples, self.num_classes, device=self.device)
        labels[:, self.adv_classes] = 1.0 / len(self.adv_classes)
        return labels

    def learn_init(self, classifier, steps_per_epoch=10, epochs=5, lr=0.1):
        optimizer = Adam(self.sample.parameters(), lr=lr)
        adv_labels = self._adv_labels()
        classifier.eval()
        loss_val = 0.0
        for _ in range(epochs):
            loss_val = 0.0
            for _ in range(steps_per_epoch):
                sample = self.sample().to(self.device)
                logits, _ = classifier(sample)
                loss = F.cross_entropy(logits, adv_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_val += loss.item()
            loss_val /= steps_per_epoch
        classifier.train()
        return loss_val

    def update(self, classifier, lr=1e-3):
        optimizer = Adam(self.sample.parameters(), lr=lr)
        adv_labels = self._adv_labels()
        classifier.eval()
        sample = self.sample().to(self.device)
        logits, _ = classifier(sample)
        loss = F.cross_entropy(logits, adv_labels)
        loss_val = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        classifier.train()
        return loss_val


class SCADA_MiniMax(Gradient_Ascent):
    def __init__(self, *args, forget_classes, m_alpha=10.0, m_samples=4,
                 m_update=1, m_label='rescaled', iter_per_epoch=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not forget_classes:
            raise ValueError("SCADA_MiniMax requires forget_classes (unlearn_setting: class).")
        self.forget_classes = forget_classes
        self.m_alpha = m_alpha
        self.m_samples = m_samples
        self.m_update = max(1, m_update)
        self.m_label = m_label
        self.iter_per_epoch = iter_per_epoch
        self.num_classes = getattr(self.model, 'num_classes')

    def _pseudo_label(self, logits, adv_classes):
        if self.m_label == 'rescaled':
            labels = logits.detach().clone()
            labels[:, adv_classes] = -float('inf')
            return F.softmax(labels, dim=1)
        elif self.m_label == 'uniform':
            labels = torch.ones_like(logits)
            labels[:, adv_classes] = 0
            return labels / labels.sum(dim=1, keepdim=True)
        elif self.m_label == 'random':
            labels = torch.zeros_like(logits)
            retain_classes = list(set(range(logits.size(1))) - set(adv_classes))
            for row in range(logits.size(0)):
                labels[row, retain_classes[torch.randint(0, len(retain_classes), (1,)).item()]] = 1.0
            return labels
        raise ValueError(f"Unknown m_label: {self.m_label}")

    def unlearn(self, fa_threshold, ckpt_path):
        self.model.train()
        sample_batch = next(iter(self.retain_loader))
        channels, img_size = sample_batch[0].shape[1], sample_batch[0].shape[-1]

        num_forget_classes = len(self.forget_classes)
        iter_per_epoch = self.iter_per_epoch or len(self.retain_loader)
        iter_per_class = max(1, iter_per_epoch // num_forget_classes)

        adversarial_samples = [
            AdversarialSample([c], self.num_classes, self.m_samples, channels, img_size, self.device)
            for c in self.forget_classes
        ]
        retain_iter = _ForeverLoader(self.retain_loader)

        total_unlearn_time = 0.0
        for epoch in range(self.num_epoch):
            epoch_start = time.time()
            total_loss, total_mu_loss = 0.0, 0.0

            for i in range(iter_per_epoch):
                class_idx = min(i // iter_per_class, num_forget_classes - 1)
                adv = adversarial_samples[class_idx]

                if i % iter_per_class == 0:
                    adv.learn_init(self.model)
                elif i % self.m_update == 0:
                    adv.update(self.model)

                self.model.train()
                images, labels = sample_batch_to_device(next(retain_iter), self.device)

                sample = adv.sample().detach().to(self.device)
                adv_logits, _ = self.model.forward_with_grad(sample)
                pseudo_labels = self._pseudo_label(adv_logits, adv.adv_classes)
                mu_loss = F.cross_entropy(adv_logits, pseudo_labels)

                retain_logits, _ = self.model.forward_with_grad(images)
                retain_loss = self.criteria(retain_logits, labels)

                loss = retain_loss + mu_loss * self.m_alpha

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_mu_loss += mu_loss.item()

            avg_loss = total_loss / iter_per_epoch
            avg_mu_loss = total_mu_loss / iter_per_epoch
            epoch_time = time.time() - epoch_start
            total_unlearn_time += epoch_time

            print(f"[*] evaluating epoch {epoch+1}...")
            fa_score, ra_score, ta_score, mia_score = self.evaluate()

            print(f"--> Epoch [{epoch+1}/{self.num_epoch}] | Time: {epoch_time:.2f}s | "
                  f"Loss: {avg_loss:.4f} | MU Loss: {avg_mu_loss:.4f}")
            print(f"--> Metrics: RA: {ra_score*100:.2f}% | FA: {fa_score*100:.2f}% | "
                  f"TA: {ta_score*100:.2f}% | MIA: {mia_score:.4f}")
            print("-" * 40)

            wandb.log({
                "epoch": epoch + 1,
                "unlearn_loss": avg_loss,
                "mu_loss": avg_mu_loss,
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


def sample_batch_to_device(batch, device):
    return batch[0].to(device), batch[1].to(device)
