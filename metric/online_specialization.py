"""
Per-epoch, per-MoE-layer expert-specialization metrics logged to wandb DURING
training (the "online" counterpart to the end-of-training diagnostics in
module_diagnostics.py / _run_final_router_diagnostics).

It reuses the router probabilities each DeepMoELayer already caches on the
training forward pass (`last_pi_all`, `last_pi`) -- so it adds NO extra
inference. You feed it the model + this batch's labels/domains once per batch,
then log() once per epoch.

Metrics per MoE layer (name = module path, e.g. blocks.10.moe):
  gate_entropy          mean entropy of the top-gate_k gate weights (last_pi).
                        This is what the sparse-routing loss trains on.
  gate_entropy_norm     gate_entropy / log(gate_k)   -> 0 = fully committed, 1 = 50/50.
  alltoken_entropy      mean entropy of the full softmax (last_pi_all).
  alltoken_entropy_norm alltoken_entropy / log(num_experts).
  dead_experts          # experts whose global routing mass < dead_expert_threshold.
  balance_deviation     sum_e (load_e - 1/E)^2   (0 = perfectly uniform load).
  max_load              largest per-expert routing-mass share (collapse warning).
  mean_domain_purity    per expert: fraction of its assigned images from its top
                        domain; averaged over experts. Chance = 1/num_domains.
  mean_class_purity     same, per class. Chance = 1/num_classes.

  mean_domain_purity_used / mean_class_purity_used
                        same purity but averaged ONLY over experts that received
                        >= 1 image this epoch. mean_*_purity counts a dead expert
                        as 0 and so can legitimately fall below chance; the _used
                        variant separates "how pure are the used experts" from
                        "how many experts are used".
  cover                 # experts that received >= 1 image (hard top-k).
  domain_mi_norm /      I(G;E) / min(log G, log M) computed from the gated-mass
  class_mi_norm         group x expert matrix (Eq. 29/30). 1 = every group owns
                        its own experts, 0 = routing independent of the group.
                        This is exactly what L_sep (metric/route_separation.py)
                        maximises, so it is the online read of route separation.
  domain_overlap /      mean over group pairs of sum_e min(W_g,e, W_g',e):
  class_overlap         soft-mass overlap in [0,1] (1 = identical routing).
  domain_topk_jaccard / mean pairwise Jaccard of the groups' top-gate_k expert
  class_topk_jaccard    sets (the Fig. 6/7 "shared dominant expert" statistic).

Overall roll-ups (mean over layers): online_spec/overall/{mean_gate_entropy_norm,
mean_domain_purity, mean_class_purity, total_dead_experts, mean_domain_mi_norm,
mean_class_mi_norm, ...} plus the chance lines domain_purity_chance / class_purity_chance.

CAVEAT: during training noise_std > 0, so these reflect the *noisy* routing the
model is optimizing under, not a clean eval pass. That's fine for watching a
trend; for the definitive clean read use the eval-mode router_traintest_match
block and the end-of-training diagnostics.
"""
import numpy as np
import torch
import wandb


class OnlineSpecTracker:
    def __init__(self, num_experts, gate_k, num_domains=0, num_classes=0,
                 dead_expert_threshold=0.01):
        self.E = int(num_experts)
        self.gate_k = int(gate_k)
        self.D = int(num_domains)
        self.C = int(num_classes)
        self.dead_thr = float(dead_expert_threshold)
        self._load = {}       # name -> [E]   summed routing mass
        self._tok = {}        # name -> float summed token count
        self._gk_ent = {}     # name -> float summed top-k gate entropy
        self._at_ent = {}     # name -> float summed all-token entropy
        self._dom = {}        # name -> [E, D] per-image domain tally
        self._cls = {}        # name -> [E, C] per-image class tally
        self._gdom = {}       # name -> [D, E] summed gated mass per domain (token level)
        self._gcls = {}       # name -> [C, E] summed gated mass per class
        self._gdom_n = {}     # name -> [D] token count per domain
        self._gcls_n = {}     # name -> [C] token count per class

    @staticmethod
    def _entropy(p, eps=1e-8):
        return -(p * (p + eps).log()).sum(dim=-1)

    def _ensure(self, name):
        if name not in self._load:
            self._load[name] = torch.zeros(self.E)
            self._tok[name] = 0.0
            self._gk_ent[name] = 0.0
            self._at_ent[name] = 0.0
            self._dom[name] = torch.zeros(self.E, max(self.D, 1))
            self._cls[name] = torch.zeros(self.E, max(self.C, 1))
            self._gdom[name] = torch.zeros(max(self.D, 1), self.E)
            self._gcls[name] = torch.zeros(max(self.C, 1), self.E)
            self._gdom_n[name] = torch.zeros(max(self.D, 1))
            self._gcls_n[name] = torch.zeros(max(self.C, 1))

    @torch.no_grad()
    def update(self, model, labels, domains=None):
        """Call once per training batch, AFTER the forward pass (so last_pi_all /
        last_pi are populated) and BEFORE the next forward pass.

        labels:  [B] long tensor of class ids for this batch.
        domains: [B] long tensor of domain ids, or None (datasets without domains).
        """
        labels = labels.detach().to("cpu").long()
        B = labels.size(0)
        if domains is not None:
            domains = domains.detach().to("cpu").long()

        moe = [(n, m) for n, m in model.named_modules()
               if m.__class__.__name__ == "DeepMoELayer"]

        for name, m in moe:
            if m.last_pi_all is None:
                continue
            self._ensure(name)
            pi_all = m.last_pi_all.detach().float().cpu()   # [B*S, E]
            gk = m.last_pi.detach().float().cpu()            # [B*S, gate_k]
            S = max(pi_all.size(0) // B, 1)

            self._load[name] += pi_all.sum(dim=0)
            self._tok[name] += float(pi_all.size(0))
            self._at_ent[name] += self._entropy(pi_all).sum().item()
            self._gk_ent[name] += self._entropy(gk).sum().item()

            # per-image assignment: mean pi over tokens, then top-gate_k
            # (same convention as module_diagnostics.extract_router_diagnostics).
            # group x expert gated mass (Eq. 30) for the MI / overlap statistics.
            gm = m.last_gate_mass
            gm = gm.detach().float().cpu() if gm is not None else pi_all
            tok_ones = torch.ones(gm.size(0))
            if self.C > 0:
                lab_tok = labels.repeat_interleave(S)
                self._gcls[name].index_add_(0, lab_tok, gm)
                self._gcls_n[name].index_add_(0, lab_tok, tok_ones)
            if self.D > 0 and domains is not None:
                dom_tok = domains.repeat_interleave(S)
                self._gdom[name].index_add_(0, dom_tok, gm)
                self._gdom_n[name].index_add_(0, dom_tok, tok_ones)

            pi_img = pi_all.view(B, S, self.E).mean(dim=1)   # [B, E]
            _, topk = pi_img.topk(self.gate_k, dim=-1)        # [B, gate_k]
            for k in range(self.gate_k):
                e = topk[:, k]
                if self.C > 0:
                    self._cls[name].view(-1).index_add_(0, e * self.C + labels, torch.ones(B))
                if self.D > 0 and domains is not None:
                    self._dom[name].view(-1).index_add_(0, e * self.D + domains, torch.ones(B))

    def _group_stats(self, summed, count):
        """summed: [G, E] gated mass, count: [G] tokens. Returns dict or None."""
        present = count > 0
        if int(present.sum().item()) < 2:
            return None
        W = summed[present] / count[present].unsqueeze(1)
        W = W / W.sum(dim=1, keepdim=True).clamp(min=1e-8)      # [G', E]
        p_g = count[present] / count[present].sum()
        joint = p_g.unsqueeze(1) * W
        p_e = joint.sum(dim=0, keepdim=True)
        mi = (joint * (joint / (p_g.unsqueeze(1) * p_e).clamp(min=1e-8)).clamp(min=1e-8).log()).sum()
        norm = min(np.log(W.size(0)), np.log(self.E))
        G = W.size(0)
        overlaps, jacc = [], []
        top = W.topk(min(self.gate_k, self.E), dim=1).indices
        for i in range(G):
            for j in range(i + 1, G):
                overlaps.append(torch.minimum(W[i], W[j]).sum().item())
                a, b = set(top[i].tolist()), set(top[j].tolist())
                jacc.append(len(a & b) / len(a | b))
        return {
            "mi_norm": float(mi.item() / norm) if norm > 0 else 0.0,
            "overlap": float(np.mean(overlaps)),
            "topk_jaccard": float(np.mean(jacc)),
        }

    def log(self, epoch, prefix="online_spec", to_wandb=True):
        """Call once at the end of an epoch. Returns the payload dict."""
        payload = {"epoch": epoch}
        log_E = float(np.log(self.E)) if self.E > 1 else 1.0
        log_gk = float(np.log(self.gate_k)) if self.gate_k > 1 else 0.0

        gnorm_list, domp_list, clsp_list, dead_total = [], [], [], 0
        roll = {}   # overall roll-ups for the new group statistics

        def _acc(key, val):
            roll.setdefault(key, []).append(val)
        for name in self._load:
            tok = max(self._tok[name], 1.0)
            load = self._load[name] / tok                    # [E], sums ~1
            dead = int((load < self.dead_thr).sum().item())
            dead_total += dead
            bal_dev = float(((load - 1.0 / self.E) ** 2).sum().item())
            gk_ent = self._gk_ent[name] / tok
            at_ent = self._at_ent[name] / tok
            gk_norm = (gk_ent / log_gk) if log_gk > 0 else 0.0
            gnorm_list.append(gk_norm)

            payload[f"{prefix}/{name}/gate_entropy"] = gk_ent
            payload[f"{prefix}/{name}/gate_entropy_norm"] = gk_norm
            payload[f"{prefix}/{name}/alltoken_entropy"] = at_ent
            payload[f"{prefix}/{name}/alltoken_entropy_norm"] = at_ent / log_E
            payload[f"{prefix}/{name}/dead_experts"] = dead
            payload[f"{prefix}/{name}/balance_deviation"] = bal_dev
            payload[f"{prefix}/{name}/max_load"] = float(load.max().item())

            if self.D > 0 and self._dom[name].sum() > 0:
                used = self._dom[name].sum(dim=1) > 0
                share = self._dom[name] / self._dom[name].sum(dim=1, keepdim=True).clamp(min=1)
                pur = share.max(dim=1).values
                dp = float(pur.mean().item())
                payload[f"{prefix}/{name}/mean_domain_purity"] = dp
                payload[f"{prefix}/{name}/mean_domain_purity_used"] = float(pur[used].mean().item())
                payload[f"{prefix}/{name}/cover"] = int(used.sum().item())
                domp_list.append(dp)
                gs = self._group_stats(self._gdom[name], self._gdom_n[name])
                if gs is not None:
                    for k, v in gs.items():
                        payload[f"{prefix}/{name}/domain_{k}"] = v
                        _acc(f"mean_domain_{k}", v)
            if self.C > 0 and self._cls[name].sum() > 0:
                used = self._cls[name].sum(dim=1) > 0
                share = self._cls[name] / self._cls[name].sum(dim=1, keepdim=True).clamp(min=1)
                pur = share.max(dim=1).values
                cp = float(pur.mean().item())
                payload[f"{prefix}/{name}/mean_class_purity"] = cp
                payload[f"{prefix}/{name}/mean_class_purity_used"] = float(pur[used].mean().item())
                payload.setdefault(f"{prefix}/{name}/cover", int(used.sum().item()))
                clsp_list.append(cp)
                gs = self._group_stats(self._gcls[name], self._gcls_n[name])
                if gs is not None:
                    for k, v in gs.items():
                        payload[f"{prefix}/{name}/class_{k}"] = v
                        _acc(f"mean_class_{k}", v)

        if gnorm_list:
            payload[f"{prefix}/overall/mean_gate_entropy_norm"] = float(np.mean(gnorm_list))
        if domp_list:
            payload[f"{prefix}/overall/mean_domain_purity"] = float(np.mean(domp_list))
        if clsp_list:
            payload[f"{prefix}/overall/mean_class_purity"] = float(np.mean(clsp_list))
        payload[f"{prefix}/overall/total_dead_experts"] = dead_total
        for k, vals in roll.items():
            payload[f"{prefix}/overall/{k}"] = float(np.mean(vals))
        # chance baselines, so the wandb panels have a reference line to read against.
        if self.D > 0:
            payload[f"{prefix}/overall/domain_purity_chance"] = 1.0 / self.D
        if self.C > 0:
            payload[f"{prefix}/overall/class_purity_chance"] = 1.0 / self.C

        if to_wandb:
            wandb.log(payload)
        return payload