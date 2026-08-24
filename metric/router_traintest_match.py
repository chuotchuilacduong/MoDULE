"""
Per-epoch router train/test consistency check for MoDULE.

For each (domain, MoE layer) pair, aggregates two distributions over experts
on the domain's TRAIN images and on its held-out TEST images:

  routing_mass   pi_bar_{d,l,m}   = mean softmax probability over tokens
                                     that expert m received from the router.
                                     (soft, pre-top-k)

  selection_freq q_bar_{d,l,m}    = fraction of tokens for which expert m
                                     was in the top-gate_k selected set.
                                     (hard, what the forward pass ROUTED to)

Then compares train vs test in both spaces:

  - top-k_u overlap    -- how many of the top-k_u experts by that measure
                          agree between train and test (0..1).
  - TV distance        -- 1/2 * sum_m |p_train[m] - p_test[m]|, in [0, 1].
                          "Fraction of probability mass that must be shifted
                          to make the two distributions agree."

Both loaders must use TEST-style transforms and this module puts the model in
eval() mode so noise_std and training augmentation don't contaminate what is
supposed to be a same-input-distribution comparison.

Naming convention for wandb keys:

  router_match/<domain>/layer<l>/mass_topk_overlap    scalar per (d, l)
  router_match/<domain>/layer<l>/mass_tv              scalar per (d, l)
  router_match/<domain>/layer<l>/sel_topk_overlap     scalar per (d, l)
  router_match/<domain>/layer<l>/sel_tv               scalar per (d, l)

  router_match/<domain>/layer<l>/expert<e>/mass_train scalar per (d, l, e)
  router_match/<domain>/layer<l>/expert<e>/mass_test  scalar per (d, l, e)
  router_match/<domain>/layer<l>/expert<e>/mass_diff  scalar per (d, l, e)
  router_match/<domain>/layer<l>/expert<e>/sel_train  scalar per (d, l, e)
  router_match/<domain>/layer<l>/expert<e>/sel_test   scalar per (d, l, e)
  router_match/<domain>/layer<l>/expert<e>/sel_diff   scalar per (d, l, e)

  router_match/overall/layer<l>/mass_{exact_match_rate,mean_topk_overlap,mean_tv}
  router_match/overall/layer<l>/sel_{exact_match_rate,mean_topk_overlap,mean_tv}
  router_match/overall/mass_{exact_match_rate,mean_topk_overlap,mean_tv}
  router_match/overall/sel_{exact_match_rate,mean_topk_overlap,mean_tv}
  router_match/overall/{num_domains,k_u,gate_k}
"""
import torch


@torch.no_grad()
def _per_domain_distributions(model, loader, moe_layers, device):
    """
    Runs `loader` through `model` in eval mode and, per MoE layer, returns
    two length-`num_experts` tensors aggregated over all tokens:

      mass -- mean of the router softmax vector (pi_all).
      sel  -- fraction of tokens where each expert was in the top-gate_k set.

    Both live on CPU (small, one row per expert). Returns None if the loader
    yielded no batches.
    """
    model.eval()

    num_layers = len(moe_layers)
    mass_sum = [None] * num_layers   # accumulators on GPU, moved to CPU at the end
    sel_count = [None] * num_layers
    tok_count = 0

    for batch in loader:
        images = batch[0].to(device)
        model(images)  # populates every layer's .last_pi_all

        # every layer sees the same token count on a shared forward pass, so we
        # just read it from the first layer.
        n_tokens = moe_layers[0].last_pi_all.size(0)
        tok_count += n_tokens

        for l, m in enumerate(moe_layers):
            pi = m.last_pi_all                        # [n_tokens, num_experts]
            # mean routing mass (soft, pre-top-k): straight sum, divide later.
            batch_mass = pi.sum(dim=0)                # [num_experts]

            # selection frequency (hard, what forward actually routed to):
            # topk on pi_all reproduces the layer's own routing because
            # noise_std is off in eval() and no masking is active outside
            # unlearn mode. Scatter each selected position into a per-expert
            # counter.
            gate_k = m.gate_k
            _, topk_idx = pi.topk(gate_k, dim=-1)     # [n_tokens, gate_k]
            batch_sel = torch.zeros(
                m.num_experts, device=pi.device, dtype=pi.dtype,
            )
            batch_sel.scatter_add_(
                0,
                topk_idx.reshape(-1),                 # [n_tokens * gate_k]
                torch.ones(topk_idx.numel(), device=pi.device, dtype=pi.dtype),
            )

            if mass_sum[l] is None:
                mass_sum[l] = batch_mass
                sel_count[l] = batch_sel
            else:
                mass_sum[l] += batch_mass
                sel_count[l] += batch_sel

    if tok_count == 0:
        return None

    mass = [(m / tok_count).cpu() for m in mass_sum]
    # sel_count[l] sums to n_tokens * gate_k across experts, so dividing by
    # n_tokens gives "fraction of tokens where expert e was selected" in
    # [0, 1] per expert (summing to gate_k across experts). We keep it in
    # this human-readable form and derive a distribution (sum=1) locally
    # when needed for TV.
    sel = [(s / tok_count).cpu() for s in sel_count]
    return mass, sel


def _tv_distance(p, q):
    """1/2 * L1 between two vectors, treating them as distributions
    (assumes they already sum to the same total, e.g. both to 1 or both to
    gate_k). Returns a plain float in [0, 1]."""
    return 0.5 * (p - q).abs().sum().item()


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
    Per domain, per MoE layer, compares expert usage between the domain's
    train images and its held-out test images -- in both the ROUTING MASS
    (soft, pre-top-k) and the SELECTION FREQUENCY (hard, top-gate_k) spaces.

    Args:
        model: ModuleArchitecture instance whose DeepMoELayer submodules
            populate .last_pi_all on forward and expose .gate_k / .num_experts.
        per_domain_train_loaders: {domain_id: DataLoader} over train images
            of that domain, TEST transforms.
        per_domain_test_loaders: {domain_id: DataLoader} over test images of
            that domain, TEST transforms.
        device: "cuda" or "cpu".
        k_u: size of top-k expert set to compare between train and test.
            1 = strict argmax match. Set to gate_k for a set-match consistent
            with Eq. 7's selection.
        domain_names: optional list; if provided, metric keys use names
            instead of numeric ids.

    Returns:
        Flat dict of wandb-ready scalar metrics. See module docstring for the
        full key layout. Empty dict if no domain contributed data (e.g. the
        dataset had no .domains attribute upstream).
    """
    moe_layers = [m for m in model.modules() if type(m).__name__ == 'DeepMoELayer']
    L = len(moe_layers)
    if L == 0:
        return {}

    domains = sorted(set(per_domain_train_loaders.keys()) & set(per_domain_test_loaders.keys()))
    if not domains:
        return {}

    # gate_k is per-layer in principle; report the first layer's value in
    # bookkeeping (the config forces a single scalar in practice).
    gate_k = int(moe_layers[0].gate_k)

    def dname(d):
        if domain_names is not None and 0 <= d < len(domain_names):
            return str(domain_names[d])
        return f"domain{d}"

    metrics = {}

    # per-layer running sums (across domains)
    mass_hits_per_layer = [0] * L
    mass_overlap_per_layer = [0.0] * L
    mass_tv_per_layer = [0.0] * L
    sel_hits_per_layer = [0] * L
    sel_overlap_per_layer = [0.0] * L
    sel_tv_per_layer = [0.0] * L
    active_domains = 0

    for d in domains:
        tr = _per_domain_distributions(model, per_domain_train_loaders[d], moe_layers, device)
        te = _per_domain_distributions(model, per_domain_test_loaders[d],  moe_layers, device)
        if tr is None or te is None:
            continue
        tr_mass, tr_sel = tr
        te_mass, te_sel = te
        active_domains += 1

        for l in range(L):
            num_experts = tr_mass[l].numel()

            # ------- routing mass (soft, pre-top-k) -------
            _, tr_mass_top = tr_mass[l].topk(k_u)
            _, te_mass_top = te_mass[l].topk(k_u)
            mass_overlap = len(set(tr_mass_top.tolist()) & set(te_mass_top.tolist())) / k_u
            mass_tv = _tv_distance(tr_mass[l], te_mass[l])

            # ------- selection frequency (hard, top-gate_k) -------
            _, tr_sel_top = tr_sel[l].topk(k_u)
            _, te_sel_top = te_sel[l].topk(k_u)
            sel_overlap = len(set(tr_sel_top.tolist()) & set(te_sel_top.tolist())) / k_u
            # normalise to a distribution (sums to 1) so TV lands in [0, 1]
            tr_sel_dist = tr_sel[l] / max(gate_k, 1)
            te_sel_dist = te_sel[l] / max(gate_k, 1)
            sel_tv = _tv_distance(tr_sel_dist, te_sel_dist)

            base = f"router_match/{dname(d)}/layer{l}"
            metrics[f"{base}/mass_topk_overlap"] = mass_overlap
            metrics[f"{base}/mass_tv"] = mass_tv
            metrics[f"{base}/sel_topk_overlap"] = sel_overlap
            metrics[f"{base}/sel_tv"] = sel_tv

            # per-expert raw values -- so wandb charts can show train, test,
            # and |diff| lines per expert for any (domain, layer) drill-down.
            for e in range(num_experts):
                mt, mp = float(tr_mass[l][e]), float(te_mass[l][e])
                st, sp = float(tr_sel[l][e]),  float(te_sel[l][e])
                metrics[f"{base}/expert{e}/mass_train"] = mt
                metrics[f"{base}/expert{e}/mass_test"] = mp
                metrics[f"{base}/expert{e}/mass_diff"] = abs(mt - mp)
                metrics[f"{base}/expert{e}/sel_train"] = st
                metrics[f"{base}/expert{e}/sel_test"] = sp
                metrics[f"{base}/expert{e}/sel_diff"] = abs(st - sp)

            # per-layer accumulators (aggregated across domains below)
            if mass_overlap == 1.0:
                mass_hits_per_layer[l] += 1
            if sel_overlap == 1.0:
                sel_hits_per_layer[l] += 1
            mass_overlap_per_layer[l] += mass_overlap
            sel_overlap_per_layer[l] += sel_overlap
            mass_tv_per_layer[l] += mass_tv
            sel_tv_per_layer[l] += sel_tv

    if active_domains == 0:
        return {}

    # -------- per-layer aggregates (across domains) --------
    for l in range(L):
        metrics[f"router_match/overall/layer{l}/mass_exact_match_rate"] = mass_hits_per_layer[l] / active_domains
        metrics[f"router_match/overall/layer{l}/mass_mean_topk_overlap"] = mass_overlap_per_layer[l] / active_domains
        metrics[f"router_match/overall/layer{l}/mass_mean_tv"] = mass_tv_per_layer[l] / active_domains
        metrics[f"router_match/overall/layer{l}/sel_exact_match_rate"] = sel_hits_per_layer[l] / active_domains
        metrics[f"router_match/overall/layer{l}/sel_mean_topk_overlap"] = sel_overlap_per_layer[l] / active_domains
        metrics[f"router_match/overall/layer{l}/sel_mean_tv"] = sel_tv_per_layer[l] / active_domains

    # -------- overall aggregates (across domain x layer) --------
    total_pairs = L * active_domains
    metrics["router_match/overall/mass_exact_match_rate"] = sum(mass_hits_per_layer) / max(total_pairs, 1)
    metrics["router_match/overall/mass_mean_topk_overlap"] = sum(mass_overlap_per_layer) / max(total_pairs, 1)
    metrics["router_match/overall/mass_mean_tv"] = sum(mass_tv_per_layer) / max(total_pairs, 1)
    metrics["router_match/overall/sel_exact_match_rate"] = sum(sel_hits_per_layer) / max(total_pairs, 1)
    metrics["router_match/overall/sel_mean_topk_overlap"] = sum(sel_overlap_per_layer) / max(total_pairs, 1)
    metrics["router_match/overall/sel_mean_tv"] = sum(sel_tv_per_layer) / max(total_pairs, 1)

    metrics["router_match/overall/num_domains"] = active_domains
    metrics["router_match/overall/k_u"] = k_u
    metrics["router_match/overall/gate_k"] = gate_k
    return metrics