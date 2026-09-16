"""
L_sep: route-separation regulariser = -I(G; E), mutual information between a
group label G (domain or class) and the expert E a token is routed to.

Motivation (docs/review_online_spec_and_route_separation.md): on the PACS base
model, expert 7 sits in the top-2 of 3/4 domains and Photo/Art share their whole
top-2 set at layer 2. None of L_sp / L_bal / L_div penalises that -- L_sp only
concentrates each *sample's* route, L_bal only flattens the marginal load, L_div
only decorrelates expert *outputs*. Two groups can both be sharply routed to the
same expert and incur zero penalty from all three. Yet shared dominant experts
are exactly what makes retain exposure E_r large and Lemma 1 vacuous.

    I(G;E) = H(E) - H(E|G)
           = sum_{g,e} P(g,e) log P(g,e) / (P(g) P(e))

  * H(E|G) low  -> each group concentrates its routing mass on few experts
                   (surrogate for forget coverage C_f, Eq. 7).
  * H(E)   high -> marginal load is flat over experts, i.e. the SAME pressure
                   as L_bal. Collapse onto one expert gives I = 0, so unlike a
                   naive concentration penalty it cannot be gamed.

Prior art: Mod-Squad (Chen et al., CVPR 2023) uses the same task-expert MI for
multi-task MoE; AEA applies it per domain. Here it is used as a deletion-
localisability prior.

P(g,e) is estimated from the batch as the mean gated mass (Eq. 29/30, the
quantity C_f and E_r are defined on) or from the raw router softmax pi:

    W[g, :] = mean_{tokens of group g} mass[t, :]           (rows sum to 1)
    P(g)    = token share of group g in the batch
    P(g,e)  = P(g) * W[g,e]

The returned loss is  -I / min(log G, log M)  in [-1, 0], so lambda_sep is
comparable across G=4 domains and G=7 (or 65) classes.

Sanity values (numpy mirror, M=8, k=2, 4 domains, see the review doc):
    fully disjoint routing      I = log 4 = 1.386   (loss = -1.00)
    current PACS base (shared)  I = 0.641           (loss = -0.46)
    uniform / collapsed         I = 0               (loss =  0.00)
"""
import math

import torch


def group_mass_matrix(mass, group_ids, num_groups):
    """mass: [N_tokens, M] (>=0); group_ids: [N_tokens] long in [0, num_groups).

    Returns (W [G, M] row-normalised, p_g [G] token share, count [G])."""
    G = int(num_groups)
    M = mass.size(-1)
    onehot = torch.zeros(mass.size(0), G, device=mass.device, dtype=mass.dtype)
    onehot.scatter_(1, group_ids.view(-1, 1), 1.0)
    count = onehot.sum(dim=0)                                  # [G]
    summed = onehot.T @ mass                                   # [G, M]
    W = summed / count.clamp(min=1.0).unsqueeze(1)
    W = W / W.sum(dim=1, keepdim=True).clamp(min=1e-8)
    p_g = count / count.sum().clamp(min=1.0)
    return W, p_g, count


def mutual_information(W, p_g, eps=1e-8):
    """I(G;E) in nats from row-stochastic W [G, M] and group prior p_g [G]."""
    joint = p_g.unsqueeze(1) * W                               # [G, M]
    p_e = joint.sum(dim=0, keepdim=True)                       # [1, M]
    ratio = joint / (p_g.unsqueeze(1) * p_e).clamp(min=eps)
    return (joint * (ratio.clamp(min=eps)).log()).sum()


def loss_route_separation(moe_modules, group_ids, num_groups, *, use_gated=True,
                          ema_states=None, ema_alpha=0.0, return_parts=False):
    """-I(G;E) averaged over MoE layers, normalised to [-1, 0].

    moe_modules : list of (name, DeepMoELayer) that have just done a forward pass
    group_ids   : [B] long (one id per IMAGE); expanded to tokens by repeat.
    num_groups  : G
    use_gated   : True  -> last_gate_mass (normalised active gates, Eq. 29)
                  False -> last_pi_all    (raw router softmax; Mod-Squad's choice)
    ema_states  : dict name -> W_ema; with ema_alpha > 0, W is blended across
                  batches (needed when a batch holds ~1 sample per group, e.g.
                  CIFAR-100). The EMA term is detached; gradient flows through the
                  current batch only.
    """
    G = int(num_groups)
    if G < 2 or len(moe_modules) == 0:
        return None
    total = None
    parts = {}
    for name, m in moe_modules:
        mass = m.last_gate_mass if use_gated else m.last_pi_all
        if mass is None:
            continue
        M = mass.size(-1)
        S = mass.size(0) // group_ids.size(0)
        gid = group_ids.repeat_interleave(S).to(mass.device)
        W, p_g, count = group_mass_matrix(mass, gid, G)
        if ema_states is not None and ema_alpha > 0.0:
            prev = ema_states.get(name)
            if prev is not None:
                W = ema_alpha * prev.detach() + (1.0 - ema_alpha) * W
            ema_states[name] = W.detach()
            # groups absent from this batch fall back to their EMA row (no gradient)
        mi = mutual_information(W, p_g)
        norm = min(math.log(G), math.log(M))
        l = -mi / norm
        parts[name] = l.detach()
        total = l if total is None else total + l
    if total is None:
        return None
    total = total / len(parts)
    return (total, parts) if return_parts else total
