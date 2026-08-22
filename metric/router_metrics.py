import torch


@torch.no_grad()
def mass_and_entropy(model, loader, device):
    """
    One forward pass over `loader`. Per MoE layer, returns the mean routing-mass
    vector (same computation as Module._get_routing_mass, approx_algo/module.py:115-136,
    standalone so it doesn't need a Module instance) and the mean per-token entropy
    of last_pi_all (-sum_m pi_m log pi_m), mirroring Eq. 3's sparsity target.
    """
    model.eval()
    moe_layers = [m for m in model.modules() if type(m).__name__ == 'DeepMoELayer']
    L = len(moe_layers)

    masses = None
    total_tokens = 0
    entropy_sum = [0.0] * L

    for batch in loader:
        images = batch[0].to(device)
        model(images)

        batch_masses = []
        num_tokens = 0
        for l, m in enumerate(moe_layers):
            pi = m.last_pi_all  # [tokens, num_experts]
            batch_masses.append(pi.sum(dim=0))
            num_tokens = pi.size(0)
            ent = -(pi * (pi + 1e-8).log()).sum(dim=-1)  # [tokens]
            entropy_sum[l] += ent.sum().item()

        masses = batch_masses if masses is None else [a + b for a, b in zip(masses, batch_masses)]
        total_tokens += num_tokens

    mass_vecs = [m / max(total_tokens, 1) for m in masses]
    entropies = [s / max(total_tokens, 1) for s in entropy_sum]
    return mass_vecs, entropies


def _router_mask_hook(allowed_experts):
    def hook(module, inputs, output):
        mask = torch.ones_like(output, dtype=torch.bool)
        mask[:, allowed_experts] = False
        return output.masked_fill(mask, float('-inf'))
    return hook


@torch.no_grad()
def forced_routing_accuracy(model, loader, moe_layers, selected_experts_per_layer, device, invert_subset=False):
    """
    Forces each MoE layer to route only within `selected_experts_per_layer[l]`
    (or its complement if invert_subset=True), then measures plain classification
    accuracy with zero weight updates.

    Implemented via a forward hook on DeepMoELayer.router (masking excluded expert
    logits to -inf before the softmax), NOT via DeepMoELayer.allowed_experts.
    allowed_experts also sets active_k = len(allowed_experts) (architecture/module.py:68),
    so the necessity condition (6-expert complement) would average 6 experts under a
    near-flat softmax instead of the normal top-gate_k -- a confound that would bias
    necessity_gap independently of whether the selected set is actually meaningful.
    Masking router logits keeps active_k == gate_k in every condition, so only "which
    experts are reachable" varies, isolating the thing this test is meant to measure.
    Runs in eval() (no_grad), where noise_std is already inactive.
    """
    model.eval()
    num_experts = moe_layers[0].num_experts
    hooks = []
    try:
        for l, m in enumerate(moe_layers):
            selected = selected_experts_per_layer[l]
            if invert_subset:
                allowed = [e for e in range(num_experts) if e not in selected]
            else:
                allowed = list(selected)
            allowed_t = torch.as_tensor(allowed, device=device, dtype=torch.long)
            hooks.append(m.router.register_forward_hook(_router_mask_hook(allowed_t)))

        correct, total = 0, 0
        for batch in loader:
            images = batch[0].to(device)
            labels = batch[1].to(device)
            logits, _ = model.inference(images)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        return correct / max(total, 1)
    finally:
        for h in hooks:
            h.remove()


@torch.no_grad()
def expert_hit_rate(model, loader, moe_layers, selected_experts_per_layer, device):
    """
    Unmasked, natural forward pass (router behaves as trained). Per image, per layer:
    does ANY token's actual top-gate_k expert set intersect selected_experts_per_layer[l]?
    Same aggregation as Module._create_filtered_retain_loader (approx_algo/module.py:341-387)
    and the ARR metric in module_diagnostics.py::compute_localization_diagnostics --
    applied here per-rule instead of only for the `diff` selection.

    Returns one hit-rate per MoE layer.
    """
    model.eval()
    L = len(moe_layers)
    hit_counts = [0] * L
    total = 0

    for batch in loader:
        images = batch[0].to(device)
        B = images.size(0)
        model(images)
        total += B

        for l, m in enumerate(moe_layers):
            selected = selected_experts_per_layer[l]
            if not selected:
                continue
            _, topk_indices = m.last_pi_all.topk(m.gate_k, dim=-1)  # [B*S, gate_k]
            S = topk_indices.size(0) // B
            topk_indices = topk_indices.view(B, S, m.gate_k)
            sel_t = torch.as_tensor(selected, device=device)
            hit = torch.isin(topk_indices, sel_t).any(dim=-1).any(dim=-1)  # [B]
            hit_counts[l] += hit.sum().item()

    return [c / max(total, 1) for c in hit_counts]
