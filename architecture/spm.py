"""
SPM (Semi-Parametric Model) architecture adapter for the MoDULE benchmark.

Ported from amberyzheng/spm_unlearning (models/spm_classifier.py) and reshaped
to match the duck-typed interface the rest of this repo expects from a model:

    logits, features = model.forward_with_grad(images)   # train / unlearn step
    logits, features = model.inference(images)            # eval
    model.load_state_dict(...) / model.state_dict()       # checkpointing

Unlike the other architectures in this repo (resnet / deit / module), SPM is
*non-parametric at test time*: predictions are made by relating a query image
embedding to a bank of "support" sample embeddings (the memory), aggregated
through a small mixture-of-pairwise-relation-experts head. "Unlearning" a
sample means removing it from that support bank -- see
approx_algo/spm_unlearn.py, which is the actual point of this baseline.

IMPORTANT: forward() returns *log-probabilities*, not raw logits (matches the
original SPM paper, trained with NLLLoss on log(moe_output)). learn.py /
unlearn.py select nn.NLLLoss() as `criteria` for the 'spm' model_name branch.
`self.returns_log_probs = True` below lets metric/mia.py apply the matching
per-sample loss (nll_loss, not cross_entropy -- see that file for why it
matters: cross_entropy on an already-log-softmax'd input silently computes
the wrong per-sample loss).

Differences from the original spm_unlearning repo (intentional, for benchmark fit):
  - No PyTorch Lightning / FAISS dependency. MoDULE's datasets (PACS, OfficeHome,
    CIFAR-100, Tiny-ImageNet) are small enough that we keep the support bank as a
    plain in-memory tensor and do exact (not approximate) attention over it. If you
    later benchmark on something ImageNet-scale, swap the exact attention in
    PairwiseRelationExpert for a FAISS IndexFlatL2 top-k lookup as in the
    original repo's test_step.
  - Uses `timm` for the encoder (like architecture/resnet.py) instead of the
    original repo's custom CIFAR ResNet, so pretrained weights + the rest of the
    codebase's conventions (model_name string, embed_dim = featurizer.num_features)
    stay consistent with your other architectures.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class PairwiseRelationExpert(nn.Module):
    """Ported as-is (renamed for clarity) from spm_unlearning/models/spm_classifier.py."""

    def __init__(self, embed_dim_q, embed_dim_s, num_classes, hidden_dim=None):
        super().__init__()
        self.num_classes = num_classes
        hidden_dim = hidden_dim or embed_dim_q
        self.linear_q = nn.Linear(embed_dim_q, hidden_dim, bias=False)
        self.linear_s = nn.Linear(embed_dim_s, hidden_dim, bias=False)
        self.linear_out = nn.Linear(hidden_dim, 1)
        self.hidden_dim = hidden_dim

    def forward(self, q, support_emb, support_labels):
        # q: (B, D)  support_emb: (N, D)  support_labels: (N,)
        unique_labels, inv_indices = torch.unique(support_labels, sorted=True, return_inverse=True)
        one_hot = F.one_hot(inv_indices, num_classes=unique_labels.size(0)).float()  # (N, C_small)

        q_trans = self.linear_q(q)                     # (B, hidden_dim)
        s_trans = self.linear_s(support_emb)            # (N, hidden_dim)
        h = torch.matmul(q_trans, s_trans.transpose(0, 1)) / math.sqrt(self.hidden_dim)  # (B, N)
        weights = F.softmax(h, dim=1)                    # (B, N)
        out_small = torch.matmul(weights, one_hot)        # (B, C_small)

        out = q.new_zeros(q.size(0), self.num_classes)
        for i, lab in enumerate(unique_labels):
            if lab < self.num_classes:  # guards against sentinel/ignore labels
                out[:, lab] = out_small[:, i]
        return out


class GatingNetwork(nn.Module):
    def __init__(self, embed_dim, num_experts):
        super().__init__()
        self.fc = nn.Linear(2 * embed_dim, num_experts)

    def forward(self, q, support_emb):
        s_mean = support_emb.mean(dim=0, keepdim=True).expand(q.size(0), -1)
        return F.softmax(self.fc(torch.cat([q, s_mean], dim=-1)), dim=-1)  # (B, num_experts)


class SPM_MoEHead(nn.Module):
    def __init__(self, embed_dim, num_classes, num_experts=4):
        super().__init__()
        self.experts = nn.ModuleList([
            PairwiseRelationExpert(embed_dim, embed_dim, num_classes) for _ in range(num_experts)
        ])
        self.gating = GatingNetwork(embed_dim, num_experts)
        self.num_classes = num_classes

    def forward(self, q, support_emb, support_labels):
        expert_out = torch.stack(
            [expert(q, support_emb, support_labels) for expert in self.experts], dim=1
        )                                                  # (B, num_experts, C)
        gate = self.gating(q, support_emb).unsqueeze(-1)   # (B, num_experts, 1)
        probs = torch.sum(gate * expert_out, dim=1)         # (B, C), sums ~1 over seen classes
        return probs


class SPMArchitecture(nn.Module):
    """
    Drop-in replacement for ResNetArchitecture / ModuleArchitecture etc., but
    NOT a BaseArchitecture subclass: BaseArchitecture's forward() composes a
    single-argument featurizer -> classifier_head pipeline, while SPM's head
    needs three arguments (query embedding, support embeddings, support
    labels) it can't source from `x` alone. Implements the same duck-typed
    forward_with_grad/inference/state_dict contract instead.
    """

    SUPPORTED_MODELS = ['spm_resnet18', 'spm_resnet34', 'spm_resnet50']

    def __init__(self, model_name='spm_resnet18', num_classes=7, pretrained=True,
                 num_experts=4, support_size=256, device="cuda"):
        super().__init__()
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"model '{model_name}' is not supported. choose from: {self.SUPPORTED_MODELS}")

        backbone = model_name.replace('spm_', '')
        self.featurizer = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        embed_dim = self.featurizer.num_features
        self.classifier_head = SPM_MoEHead(embed_dim, num_classes, num_experts=num_experts)

        self.model_name = model_name
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.support_size = support_size
        self.device = device
        # metric/mia.py needs this to pick nll_loss instead of cross_entropy
        # for the per-sample membership signal (see module docstring).
        self.returns_log_probs = True
        self.to(self.device)

        # Support ("memory") bank -- NOT persisted in state_dict by default.
        # It is data-derived (train/retain embeddings), not learned, and its
        # whole purpose in this benchmark is to be rebuilt post-unlearning,
        # so persisting a stale copy inside a checkpoint would be misleading.
        self.register_buffer("support_embeddings", torch.zeros(1, embed_dim), persistent=False)
        self.register_buffer("support_labels", torch.zeros(1, dtype=torch.long), persistent=False)
        self._support_ready = False

    # ------------------------------------------------------------------ #
    # Support-bank management -- this IS the "unlearning" mechanism.
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def build_support(self, loader, max_support=None):
        """Recompute the memory bank from a given loader (e.g. retain_loader
        after excluding the forget set). This is the entire unlearning step
        for SPM: samples simply never make it back into the bank."""
        self.eval()
        max_support = max_support or self.support_size
        embs, labels = [], []
        n_seen = 0
        for batch in loader:
            images, lbls = batch[0].to(self.device), batch[1].to(self.device)
            embs.append(self.featurizer(images))
            labels.append(lbls)
            n_seen += images.size(0)
            if n_seen >= max_support:
                break
        support_embeddings = torch.cat(embs, dim=0)[:max_support]
        support_labels = torch.cat(labels, dim=0)[:max_support]
        self.support_embeddings = support_embeddings
        self.support_labels = support_labels
        self._support_ready = True
        return support_embeddings.size(0)

    # ------------------------------------------------------------------ #
    # Forward interface, matching architecture/based_model.py's contract.
    # ------------------------------------------------------------------ #
    def forward(self, x, support_batch=None):
        q_emb = self.featurizer(x)
        if support_batch is not None:
            support_imgs, support_labels = support_batch
            support_emb = self.featurizer(support_imgs)
        else:
            if not self._support_ready:
                raise RuntimeError(
                    "SPM support bank is empty. Call model.build_support(loader) "
                    "before inference/training (done once after pretrain load, "
                    "and again inside the unlearning step)."
                )
            support_emb, support_labels = self.support_embeddings, self.support_labels

        probs = self.classifier_head(q_emb, support_emb, support_labels)
        log_probs = torch.log(probs.clamp(min=1e-8))
        return log_probs, q_emb

    def inference(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x.to(self.device))

    def forward_with_grad(self, x):
        self.train()
        return self.forward(x.to(self.device))
