import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score


def _per_sample_loss(model, loader, device):
    """
    per-sample cross-entropy loss for every item in `loader`.

    the loss is the membership signal: samples the model was trained on are
    memorized and score lower than held-out samples drawn from the same
    distribution. a scalar loss is used instead of the raw logit vector because
    logits also encode class identity, which would let the attack separate the
    two sets on class rather than on membership.
    """
    losses = []
    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device)
            labels = batch[1].to(device)

            # inference() handles eval() + no_grad() internally
            logits, _ = model.inference(images)
            losses.append(
                F.cross_entropy(logits, labels, reduction="none").float().cpu().numpy()
            )

    return np.concatenate(losses, axis=0)


def mia(model, forget_loader, unseen_loader, device="cuda", seed=42):
    """
    membership inference attack score against the forget set.

    trains a logistic regression to tell forget-set samples (members, label 1)
    from held-out samples (non-members, label 0) using per-sample loss, and
    reports its cross-validated accuracy.

    args:
        model: architecture inherited from BaseArchitecture.
        forget_loader: the forget set, under a deterministic transform.
        unseen_loader: never-trained-on samples matched to the forget set in
            class/domain composition and using the same deterministic
            transform. any other difference between the two sets (augmentation,
            class mix) is picked up by the attack and inflates the score.
        device: "cuda" or "cpu".
        seed: controls the balancing subsample and the cross-validation splits.

    returns:
        float: attack accuracy in [0, 1]. 0.5 means the attack cannot do better
        than chance, i.e. the forget set is indistinguishable from held-out
        data. the metric is two-sided: a model that has not forgotten scores
        above 0.5 (forget loss below unseen), and so does an over-unlearned
        model (forget loss far above unseen).
    """
    forget_losses = _per_sample_loss(model, forget_loader, device)
    unseen_losses = _per_sample_loss(model, unseen_loader, device)

    # balance the two sides so the attack cannot exploit a majority-class baseline
    min_len = min(len(forget_losses), len(unseen_losses))
    if min_len == 0:
        raise ValueError("Length of forget set or unseen set is 0")

    # random subsample rather than a head slice: both loaders run with
    # shuffle=False, so the first min_len items are a biased (dataset-ordered)
    # slice rather than a representative one.
    rng = np.random.default_rng(seed)
    if len(forget_losses) > min_len:
        forget_losses = forget_losses[rng.choice(len(forget_losses), min_len, replace=False)]
    if len(unseen_losses) > min_len:
        unseen_losses = unseen_losses[rng.choice(len(unseen_losses), min_len, replace=False)]

    samples_mia = np.concatenate((unseen_losses, forget_losses), axis=0).reshape(-1, 1)

    # target variables: 0 for non-members (unseen), 1 for members (forget)
    labels_mia = np.array([0] * min_len + [1] * min_len)

    # gradient ascent can drive forget losses into the tens, so standardize to
    # keep the solver well conditioned. a global affine rescale of a 1-D feature
    # does not change which threshold the attack can reach.
    std = samples_mia.std()
    if std > 0:
        samples_mia = (samples_mia - samples_mia.mean()) / std

    attack_model = LogisticRegression(max_iter=1000)
    # the matched non-member pool is small (a few hundred samples on PACS), so
    # average over many splits to keep the score from swinging on split noise.
    cv = StratifiedShuffleSplit(n_splits=50, test_size=0.2, random_state=seed)

    mia_scores = cross_val_score(
        attack_model, samples_mia, labels_mia, cv=cv, scoring="accuracy"
    )

    return mia_scores.mean()
