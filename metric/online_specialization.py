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

Two overall scalars are also logged (mean over layers) for a single at-a-glance
line chart: online_spec/overall/{mean_gate_entropy_norm, mean_domain_purity,
mean_class_purity, total_dead_experts}.

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
            pi_img = pi_all.view(B, S, self.E).mean(dim=1)   # [B, E]
            _, topk = pi_img.topk(self.gate_k, dim=-1)        # [B, gate_k]
            for k in range(self.gate_k):
                e = topk[:, k]
                if self.C > 0:
                    self._cls[name].view(-1).index_add_(0, e * self.C + labels, torch.ones(B))
                if self.D > 0 and domains is not None:
                    self._dom[name].view(-1).index_add_(0, e * self.D + domains, torch.ones(B))

    def log(self, epoch, prefix="online_spec", to_wandb=True):
        """Call once at the end of an epoch. Returns the payload dict."""
        payload = {"epoch": epoch}
        log_E = float(np.log(self.E)) if self.E > 1 else 1.0
        log_gk = float(np.log(self.gate_k)) if self.gate_k > 1 else 0.0

        gnorm_list, domp_list, clsp_list, dead_total = [], [], [], 0
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
                share = self._dom[name] / self._dom[name].sum(dim=1, keepdim=True).clamp(min=1)
                dp = float(share.max(dim=1).values.mean().item())
                payload[f"{prefix}/{name}/mean_domain_purity"] = dp
                domp_list.append(dp)
            if self.C > 0 and self._cls[name].sum() > 0:
                share = self._cls[name] / self._cls[name].sum(dim=1, keepdim=True).clamp(min=1)
                cp = float(share.max(dim=1).values.mean().item())
                payload[f"{prefix}/{name}/mean_class_purity"] = cp
                clsp_list.append(cp)

        if gnorm_list:
            payload[f"{prefix}/overall/mean_gate_entropy_norm"] = float(np.mean(gnorm_list))
        if domp_list:
            payload[f"{prefix}/overall/mean_domain_purity"] = float(np.mean(domp_list))
        if clsp_list:
            payload[f"{prefix}/overall/mean_class_purity"] = float(np.mean(clsp_list))
        payload[f"{prefix}/overall/total_dead_experts"] = dead_total
        # chance baselines, so the wandb panels have a reference line to read against.
        if self.D > 0:
            payload[f"{prefix}/overall/domain_purity_chance"] = 1.0 / self.D
        if self.C > 0:
            payload[f"{prefix}/overall/class_purity_chance"] = 1.0 / self.C

        if to_wandb:
            print("online_spec")
            wandb.log(payload)
        return payload