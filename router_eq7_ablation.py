"""
Simple ablation: is the router (Phase 1) and Eq. 7's selection (Phase 2) reasonable?
See implement.md for the full write-up. Self-contained -- does its own domain split
rather than touching learn.py/unlearn.py.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms

from dataset.pytorch_dataset.pacs import PACSDataset
from dataset.transform.test_transform import get_test_transform
from architecture.module import ModuleArchitecture
from metric.router_metrics import mass_and_entropy, forced_routing_accuracy, expert_hit_rate


class ApplyTransform(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        self.resize = transforms.Resize((224, 224))

    def __getitem__(self, idx):
        data = self.subset[idx]
        image = self.resize(data[0])
        if self.transform:
            image = self.transform(image)
        return (image,) + data[1:]

    def __len__(self):
        return len(self.subset)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_forget_retain_loaders(cfg, batch_size):
    full_dataset = PACSDataset(root_dir=cfg['data_dir'], transform=None)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    test_size = int(0.1 * total_size)
    unseen_size = total_size - train_size - test_size

    generator = torch.Generator().manual_seed(cfg['seed'])
    train_subset, _, _ = random_split(full_dataset, [train_size, test_size, unseen_size], generator=generator)

    forget_idx = [i for i in train_subset.indices if full_dataset.domains[i] == 0]
    retain_idx = [i for i in train_subset.indices if full_dataset.domains[i] != 0]

    forget_loader = DataLoader(
        ApplyTransform(Subset(full_dataset, forget_idx), get_test_transform()),
        batch_size=batch_size, shuffle=False, num_workers=4)
    retain_loader = DataLoader(
        ApplyTransform(Subset(full_dataset, retain_idx), get_test_transform()),
        batch_size=batch_size, shuffle=False, num_workers=4)
    return forget_loader, retain_loader


def build_model(cfg, checkpoint_path, device):
    model = ModuleArchitecture(
        model_name=cfg['model_name'],
        num_classes=7,  # PACS
        pretrained=False,
        moe_layers=cfg.get('moe_layers'),
        num_experts=cfg['num_experts'],
        expert_depth=cfg['expert_depth'],
        expert_hidden_ratio=cfg['expert_hidden_ratio'],
        gate_k=cfg['gate_k'],
        device=device,
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    return model


def run_phase1(model, forget_loader, retain_loader, device, num_experts, use_wandb):
    forget_mass, forget_entropy = mass_and_entropy(model, forget_loader, device)
    retain_mass, retain_entropy = mass_and_entropy(model, retain_loader, device)

    log_E = float(np.log(num_experts))
    print("\n" + "=" * 70)
    print("PHASE 1 -- router sanity check (does Lsp/Lbal look reasonable pre-Eq.7?)")
    print("=" * 70)

    wandb_payload = {}
    for l in range(len(forget_mass)):
        print(f"\n-- MoE layer {l} --")
        print(f"forget (art_painting) mass: {[round(x, 4) for x in forget_mass[l].tolist()]}")
        print(f"retain (other 3)      mass: {[round(x, 4) for x in retain_mass[l].tolist()]}")
        print(f"entropy: forget={forget_entropy[l]:.4f}  retain={retain_entropy[l]:.4f}"
              f"  (max log({num_experts})={log_E:.4f}, no sparsity)")

        bal_f = ((forget_mass[l] - 1.0 / num_experts) ** 2).sum().item()
        bal_r = ((retain_mass[l] - 1.0 / num_experts) ** 2).sum().item()
        print(f"balance deviation: forget={bal_f:.6f}  retain={bal_r:.6f}  (near 0 = balanced)")

        if use_wandb:
            wandb_payload.update({
                f"phase1/layer{l}/forget_entropy": forget_entropy[l],
                f"phase1/layer{l}/retain_entropy": retain_entropy[l],
                f"phase1/layer{l}/entropy_max": log_E,
                f"phase1/layer{l}/forget_balance_dev": bal_f,
                f"phase1/layer{l}/retain_balance_dev": bal_r,
            })
            for e in range(num_experts):
                wandb_payload[f"phase1/layer{l}/forget_mass/expert{e}"] = forget_mass[l][e].item()
                wandb_payload[f"phase1/layer{l}/retain_mass/expert{e}"] = retain_mass[l][e].item()
            wandb_payload[f"phase1/layer{l}/mass_table"] = wandb.Table(
                columns=["expert", "forget_mass", "retain_mass"],
                data=[[e, forget_mass[l][e].item(), retain_mass[l][e].item()] for e in range(num_experts)])

    if use_wandb:
        wandb.log(wandb_payload)

    return forget_mass, retain_mass


def run_phase2(model, forget_loader, retain_loader, device, forget_mass, retain_mass,
               num_experts, k_u, alphas, use_wandb):
    print("\n" + "=" * 70)
    print("PHASE 2 -- Eq. 7 selection check")
    print("=" * 70)

    moe_layers = [m for m in model.modules() if type(m).__name__ == 'DeepMoELayer']
    L = len(moe_layers)

    rules = {}
    prev_sets = None
    alpha_invariant = True
    for alpha in alphas:
        sel = []
        for l in range(L):
            rho = forget_mass[l] - alpha * retain_mass[l]
            _, idx = rho.topk(k_u)
            sel.append(sorted(idx.tolist()))
        rules[f"diff (α={alpha})"] = sel
        if prev_sets is not None and sel != prev_sets:
            alpha_invariant = False
        prev_sets = sel

    if alpha_invariant and len(alphas) > 1:
        print("\n[!] Selected expert sets are IDENTICAL across every alpha tested. The alpha "
              "penalty has no effect on selection at this k_u -- this alone would explain a "
              "flat FA/RA response to alpha in the full unlearn() sweep. Proceeding anyway to "
              "measure the (single, alpha-invariant) diff set against the baselines.")

    sel_bottom = []
    for l in range(L):
        rho = forget_mass[l] - 1.0 * retain_mass[l]
        _, idx = torch.topk(-rho, k_u)
        sel_bottom.append(sorted(idx.tolist()))
    rules["bottom"] = sel_bottom

    rand_gen = torch.Generator().manual_seed(0)
    sel_random = []
    for l in range(L):
        perm = torch.randperm(num_experts, generator=rand_gen)[:k_u]
        sel_random.append(sorted(perm.tolist()))
    rules["random"] = sel_random

    print("\nSelected expert sets per rule (per layer):")
    for name, sel in rules.items():
        print(f"  {name:<16} {sel}")

    print(f"\n{'rule':<16}{'specificity':>13}{'necessity_gap':>16}{'forget_hit_rate':>18}{'retain_leak_rate':>18}")
    table_rows = []
    wandb_payload = {}

    for name, sel in rules.items():
        FFA_suff = forced_routing_accuracy(model, forget_loader, moe_layers, sel, device, invert_subset=False)
        FRA_suff = forced_routing_accuracy(model, retain_loader, moe_layers, sel, device, invert_subset=False)
        FFA_nec = forced_routing_accuracy(model, forget_loader, moe_layers, sel, device, invert_subset=True)
        FRA_nec = forced_routing_accuracy(model, retain_loader, moe_layers, sel, device, invert_subset=True)

        specificity = FFA_suff - FRA_suff
        necessity_gap = FRA_nec - FFA_nec

        forget_hits = expert_hit_rate(model, forget_loader, moe_layers, sel, device)
        retain_hits = expert_hit_rate(model, retain_loader, moe_layers, sel, device)
        forget_hit_rate_avg = sum(forget_hits) / len(forget_hits)
        retain_leak_rate_avg = sum(retain_hits) / len(retain_hits)

        print(f"{name:<16}{specificity:13.4f}{necessity_gap:16.4f}{forget_hit_rate_avg:18.4f}{retain_leak_rate_avg:18.4f}")

        table_rows.append([name, specificity, necessity_gap, forget_hit_rate_avg, retain_leak_rate_avg,
                            FFA_suff, FRA_suff, FFA_nec, FRA_nec])
        wandb_payload.update({
            f"phase2/{name}/specificity": specificity,
            f"phase2/{name}/necessity_gap": necessity_gap,
            f"phase2/{name}/forget_hit_rate": forget_hit_rate_avg,
            f"phase2/{name}/retain_leak_rate": retain_leak_rate_avg,
            f"phase2/{name}/FFA_suff": FFA_suff,
            f"phase2/{name}/FRA_suff": FRA_suff,
            f"phase2/{name}/FFA_nec": FFA_nec,
            f"phase2/{name}/FRA_nec": FRA_nec,
        })

    if use_wandb:
        wandb.log(wandb_payload)
        wandb.log({"phase2/table": wandb.Table(
            columns=["rule", "specificity", "necessity_gap", "forget_hit_rate", "retain_leak_rate",
                     "FFA_suff", "FRA_suff", "FFA_nec", "FRA_nec"],
            data=table_rows)})


def main():
    parser = argparse.ArgumentParser(description="Router Eq. 7 selection ablation.")
    parser.add_argument('--config', type=str, default='config/learn/pacs_module.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--k_u', type=int, default=2)
    parser.add_argument('--alphas', type=str, default='0,0.5,1.0,2.0')
    parser.add_argument('--no-wandb', action='store_true')
    cli = parser.parse_args()
    alphas = [float(a) for a in cli.alphas.split(',')]

    with open(cli.config, 'r') as f:
        full_cfg = yaml.safe_load(f)
    cfg = full_cfg['learn'] if 'learn' in full_cfg else full_cfg

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg['seed'])

    print(f"[*] device: {device}")
    print(f"[*] checkpoint: {cli.checkpoint}")
    print(f"[*] k_u={cli.k_u}  alphas={alphas}")

    batch_size = cfg.get('batch_size', 128)
    forget_loader, retain_loader = build_forget_retain_loaders(cfg, batch_size)
    print(f"[*] forget (art_painting): {len(forget_loader.dataset)} images | "
          f"retain (cartoon/photo/sketch): {len(retain_loader.dataset)} images")

    model = build_model(cfg, cli.checkpoint, device)

    use_wandb = not cli.no_wandb
    if use_wandb:
        dataset = cfg.get('dataset', 'unknown')
        wandb.init(
            project="MoE",
            name=f"router_eq7ablation_{dataset}_forget-artpainting_ku{cli.k_u}_{Path(cli.checkpoint).stem}",
            tags=['router_ablation', 'eq7', dataset],
            config={**cfg, "k_u": cli.k_u, "alphas": alphas},
            settings=wandb.Settings(start_method='thread'),
        )

    forget_mass, retain_mass = run_phase1(
        model, forget_loader, retain_loader, device, cfg['num_experts'], use_wandb)
    run_phase2(
        model, forget_loader, retain_loader, device, forget_mass, retain_mass,
        cfg['num_experts'], cli.k_u, alphas, use_wandb)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
