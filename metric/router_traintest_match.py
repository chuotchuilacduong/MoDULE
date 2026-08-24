"""
Per-epoch router train/test consistency check.

For each domain, aggregates the mean routing-mass vector (over all tokens) on
that domain's TRAIN images and on its held-out TEST images, then compares the
top-k_u expert set between the two sides, per MoE layer.

Both loaders must use TEST-style transforms and this module puts the model in
eval() mode -- otherwise router noise_std and training augmentation would
contaminate what is supposed to be a same-input-distribution comparison.
"""
import torch


@torch.no_grad()
def _per_domain_mass(model, loader, moe_layers, device):
    """
    Mean routing-mass vector across all tokens in `loader`, per MoE layer.
    Returns a list of [num_experts] tensors, one per layer, matching the
    ordering of `moe_layers`.
    """
    model.eval()
    mass_sum = None
    tok_count = 0

    for batch in loader:
        images = batch[0].to(device)
        model(images)  # populates each layer's .last_pi_all

        batch_mass = [m.last_pi_all.sum(dim=0) for m in moe_layers]
        # every MoE layer sees the same number of tokens per forward pass on
        # a given input, so any layer's count is representative.
        tok_count += moe_layers[0].last_pi_all.size(0)

        if mass_sum is None:
            mass_sum = batch_mass
        else:
            mass_sum = [a + b for a, b in zip(mass_sum, batch_mass)]

    if mass_sum is None:
        return None
    return [m / max(tok_count, 1) for m in mass_sum]


@torch.no_grad()
def domain_train_test_expert_match(
    model,
    per_domain_train_loaders,
    per_domain_test_loaders,
    device,
    k_u=1,
    domain_names=None,
):
    """
    Per domain, per MoE layer: does the top-k_u expert set picked from that
    domain's TRAIN images match the top-k_u picked from its TEST images?

    Args:
        model: ModuleArchitecture instance (must expose DeepMoELayer modules
            with .last_pi_all populated after a forward pass).
        per_domain_train_loaders: {domain_id: DataLoader} over train images
            of that domain, using TEST transforms.
        per_domain_test_loaders: {domain_id: DataLoader} over test images of
            that domain, using TEST transforms.
        device: "cuda" or "cpu".
        k_u: size of the top-k expert set to compare. 1 = strict top-1 match.
            Set to gate_k for a set-match consistent with Eq. 7's selection.
        domain_names: optional list; if provided, metric keys use the human
            name instead of the numeric id.

    Returns:
        dict of wandb-ready scalar metrics. Headline entries:
            router_match/overall/exact_match_rate  -- fraction of (domain,
                layer) pairs whose top-k_u sets are identical.
            router_match/overall/mean_topk_overlap -- average |intersection|/k_u
                across (domain, layer) pairs (partial credit).
        Plus per-layer and per-(domain, layer) breakdowns.
    """
    moe_layers = [m for m in model.modules() if type(m).__name__ == 'DeepMoELayer']
    L = len(moe_layers)
    if L == 0:
        return {}

    domains = sorted(set(per_domain_train_loaders.keys()) & set(per_domain_test_loaders.keys()))
    if not domains:
        return {}

    def dname(d):
        if domain_names is not None and 0 <= d < len(domain_names):
            return str(domain_names[d])
        return f"domain{d}"

    metrics = {}
    per_layer_hits = [0] * L
    per_layer_overlap = [0.0] * L
    total_overlap = 0.0
    total_pairs = 0
    active_domains = 0

    for d in domains:
        tr_mass = _per_domain_mass(model, per_domain_train_loaders[d], moe_layers, device)
        te_mass = _per_domain_mass(model, per_domain_test_loaders[d],  moe_layers, device)
        if tr_mass is None or te_mass is None:
            continue
        active_domains += 1

        for l in range(L):
            _, tr_top = tr_mass[l].topk(k_u)
            _, te_top = te_mass[l].topk(k_u)
            tr_set = set(tr_top.tolist())
            te_set = set(te_top.tolist())
            overlap = len(tr_set & te_set) / k_u  # 0..1
            l1 = (tr_mass[l] - te_mass[l]).abs().sum().item()

            key = f"router_match/{dname(d)}/layer{l}"
            metrics[f"{key}/topk_overlap"] = overlap
            metrics[f"{key}/l1"] = l1

            if overlap == 1.0:
                per_layer_hits[l] += 1
            per_layer_overlap[l] += overlap
            total_overlap += overlap
            total_pairs += 1

    if active_domains == 0:
        return {}

    for l in range(L):
        metrics[f"router_match/overall/layer{l}/exact_match_rate"] = per_layer_hits[l] / active_domains
        metrics[f"router_match/overall/layer{l}/mean_topk_overlap"] = per_layer_overlap[l] / active_domains

    metrics["router_match/overall/exact_match_rate"] = sum(per_layer_hits) / max(L * active_domains, 1)
    metrics["router_match/overall/mean_topk_overlap"] = total_overlap / max(total_pairs, 1)
    metrics["router_match/overall/num_domains"] = active_domains
    metrics["router_match/overall/k_u"] = k_u
    return metrics