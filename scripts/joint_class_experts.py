#!/usr/bin/env python3
"""Record ModULE's expert selection for each joint request (1/2/3 classes).

ModULE selects, per MoE layer, the top-k_u experts by rho = mass_forget - alpha *
mass_retain, where the routing mass is measured over the *union* of the requested
forget classes (the forget loader of that request). k_u is fixed at 4 for every
request, so the expert-update budget does not grow with the number of classes.
This reproduces the selection the run starts from (epoch 1, original checkpoint)
and writes it next to the MIA results.

  python scripts/joint_class_experts.py
"""
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from unlearn import ApplyTransform  # noqa: E402
from architecture.module import ModuleArchitecture  # noqa: E402
from dataset.transform.forget_train_transform import get_forget_train_transform  # noqa: E402
from dataset.transform.retain_train_transform import get_retain_train_transform  # noqa: E402
from scripts.joint_class_mia import REQUESTS, build_splits, BASE_CKPT  # noqa: E402


@torch.no_grad()
def routing_mass(model, loader, device):
    """identical to Module._get_routing_mass: mean router probability per expert."""
    model.eval()
    masses, total = None, 0
    for batch in loader:
        model(batch[0].to(device))
        cur = [m.last_pi_all.sum(dim=0) for _, m in model.named_modules()
               if m.__class__.__name__ == "DeepMoELayer"]
        n = [m.last_pi_all.size(0) for _, m in model.named_modules()
             if m.__class__.__name__ == "DeepMoELayer"][0]
        masses = cur if masses is None else [a + b for a, b in zip(masses, cur)]
        total += n
    return [m / max(total, 1) for m in masses]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open(REPO / "config/joint_class/module_joint_c3.yaml"))
    k_u, alpha, bs = cfg["k_u"], cfg["alpha"], cfg["batch_size"]
    ds, train_ids, _, _ = build_splits()
    model = ModuleArchitecture(model_name=cfg["model_name"], num_classes=7, pretrained=False,
                               moe_layers=cfg["moe_layers"], num_experts=cfg["num_experts"],
                               expert_depth=cfg["expert_depth"], expert_hidden_ratio=cfg["expert_hidden_ratio"],
                               gate_k=cfg["gate_k"], device=device)
    model.load_state_dict(torch.load(REPO / BASE_CKPT, map_location=device))
    n_total = sum(p.numel() for p in model.parameters())

    out = {"k_u": k_u, "alpha": alpha, "selection": "diff (mass_forget - alpha*mass_retain) over the UNION "
           "of the requested forget classes; k_u fixed across requests", "requests": {}}
    for tag, cls in REQUESTS.items():
        f_ids = [i for i in train_ids if int(ds.labels[i]) in cls]
        r_ids = [i for i in train_ids if int(ds.labels[i]) not in cls]
        fl = DataLoader(ApplyTransform(Subset(ds, f_ids), get_forget_train_transform()), batch_size=bs, num_workers=4)
        rl = DataLoader(ApplyTransform(Subset(ds, r_ids), get_retain_train_transform()), batch_size=bs, num_workers=4)
        fm, rm = routing_mass(model, fl, device), routing_mass(model, rl, device)
        sel = [sorted((fm[l] - alpha * rm[l]).topk(k_u).indices.tolist()) for l in range(len(fm))]
        # parameters ModULE updates: selected experts + classifier head (update_scope in the config)
        moe = [m for _, m in model.named_modules() if m.__class__.__name__ == "DeepMoELayer"]
        n_upd = sum(sum(p.numel() for p in moe[l].experts[e].parameters()) for l in range(len(moe)) for e in sel[l])
        n_upd += sum(p.numel() for p in model.classifier_head.parameters()) if hasattr(model, "classifier_head") else 0
        out["requests"][tag] = {
            "forget_classes": [f"{c}:{ds.class_names[c]}" for c in cls],
            "n_forget_train": len(f_ids), "n_retain_train": len(r_ids),
            "selected_experts_per_moe_layer": sel,
            "n_experts_updated_total": sum(len(s) for s in sel),
            "updated_params_experts_and_head": int(n_upd),
            "updated_params_fraction": round(n_upd / n_total, 4),
            "update_scope": cfg["update_scope"],
        }
        print(f"{tag:10s} forget={len(f_ids):5d} retain={len(r_ids):5d} | experts/layer {sel} "
              f"| {sum(len(s) for s in sel)} experts, {n_upd:,} params ({100*n_upd/n_total:.2f}%)")
    p = REPO / "results" / "joint_class_mia" / "modules_expert_selection.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"[*] wrote {p}")


if __name__ == "__main__":
    main()
