"""
Ablation: does the "diff" selection score (rho_m = forget_mass - alpha * retain_mass,
approx_algo/module.py:494) computed on the WHOLE pooled forget/retain set pick the same
experts as if it were computed separately per class or per domain within that set?

The production code (`Module.unlearn`) only ever computes one pooled forget_mass and
one pooled retain_mass per layer, then does top-k over rho_m = forget_mass - alpha *
retain_mass to choose which experts to update. This script recomputes rho_m at three
granularities on the SAME checkpoint and forget/retain split, and reports whether the
pooled ("normal") top-k selection actually represents what each class/domain subgroup
inside the forget set would have picked on its own.
"""
import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, random_split, Subset

from dataset.pytorch_dataset.pacs import PACSDataset
from dataset.transform.test_transform import get_test_transform
from architecture.module import ModuleArchitecture
from module_diagnostics import ApplyTransform, set_seed


def get_domain(dataset, idx):
    if hasattr(dataset, 'domains'):
        return dataset.domains[idx]
    data_tuple = dataset[idx]
    val = data_tuple[2].item() if isinstance(data_tuple[2], torch.Tensor) else data_tuple[2]
    return int(val)


def build_forget_retain_loaders(args, full_dataset, batch_size):
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    test_size = int(0.1 * total_size)
    unseen_size = total_size - train_size - test_size
    generator = torch.Generator().manual_seed(args.seed)
    train_subset, test_subset, unseen_subset = random_split(
        full_dataset, [train_size, test_size, unseen_size], generator=generator
    )

    unlearn_setting = getattr(args, 'unlearn_setting', 'random')
    if unlearn_setting == 'random':
        forget_size = int(getattr(args, 'forget_ratio', 0.1) * train_size)
        forget_subset, retain_subset = random_split(
            train_subset, [forget_size, train_size - forget_size], generator=generator
        )
    elif unlearn_setting in ('class', 'domain'):
        attr_name = 'forget_classes' if unlearn_setting == 'class' else 'forget_domains'
        target_list = getattr(args, attr_name, [0])
        if not isinstance(target_list, list):
            target_list = [target_list]
        f_idx, r_idx = [], []
        for idx in train_subset.indices:
            val = full_dataset.labels[idx] if unlearn_setting == 'class' else get_domain(full_dataset, idx)
            (f_idx if val in target_list else r_idx).append(idx)
        forget_subset = Subset(full_dataset, f_idx)
        retain_subset = Subset(full_dataset, r_idx)
    else:
        raise ValueError(f"Unsupported unlearn_setting: {unlearn_setting}")

    forget_loader = DataLoader(ApplyTransform(forget_subset, get_test_transform()), batch_size=batch_size, shuffle=False)
    retain_loader = DataLoader(ApplyTransform(retain_subset, get_test_transform()), batch_size=batch_size, shuffle=False)
    return forget_loader, retain_loader, unlearn_setting


@torch.no_grad()
def collect_mass_buckets(model, loader, device, num_experts, num_classes, num_domains, moe_layers):
    """Single pass; returns per-layer aggregate mass plus per-class and per-domain mass buckets."""
    L = len(moe_layers)
    mass_agg = torch.zeros(L, num_experts)
    tok_agg = torch.zeros(L)
    mass_by_class = torch.zeros(L, num_classes, num_experts)
    tok_by_class = torch.zeros(L, num_classes)
    mass_by_domain = torch.zeros(L, num_domains, num_experts)
    tok_by_domain = torch.zeros(L, num_domains)

    model.eval()
    for batch in loader:
        images = batch[0].to(device)
        labels = batch[1].to(device).long()
        domains = batch[2].to(device).long()
        B = images.size(0)

        model(images)

        for l, m in enumerate(moe_layers):
            pi_flat = m.last_pi_all.detach().cpu()  # [B*S, E]
            S = pi_flat.size(0) // B

            mass_agg[l] += pi_flat.sum(dim=0)
            tok_agg[l] += pi_flat.size(0)

            labels_tok = labels.repeat_interleave(S).cpu()
            mass_by_class[l].index_add_(0, labels_tok, pi_flat)
            tok_by_class[l].index_add_(0, labels_tok, torch.ones(labels_tok.size(0)))

            domains_tok = domains.repeat_interleave(S).cpu()
            mass_by_domain[l].index_add_(0, domains_tok, pi_flat)
            tok_by_domain[l].index_add_(0, domains_tok, torch.ones(domains_tok.size(0)))

    mass_agg = mass_agg / tok_agg.clamp(min=1).unsqueeze(-1)
    mass_by_class = mass_by_class / tok_by_class.clamp(min=1).unsqueeze(-1)
    mass_by_domain = mass_by_domain / tok_by_domain.clamp(min=1).unsqueeze(-1)
    return {
        "agg": mass_agg, "by_class": mass_by_class, "tok_by_class": tok_by_class,
        "by_domain": mass_by_domain, "tok_by_domain": tok_by_domain,
    }


def topk_experts(rho, k):
    _, idx = rho.topk(k, dim=-1)
    return set(idx.tolist())


def jaccard(a, b):
    return len(a & b) / max(len(a | b), 1)


def main():
    parser = argparse.ArgumentParser(description="Pi-score (forget-retain diff) selection ablation: aggregate vs per-class vs per-domain.")
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--k_u', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--min-tokens', type=float, default=1000.0,
                         help="Skip a class/domain subgroup whose forget-side token count is below this "
                              "(too few tokens -> its own pi score is noisy).")
    cmd_args = parser.parse_args()

    config_path = cmd_args.config
    if config_path is None:
        derived = Path(cmd_args.checkpoint).resolve().parent.parent / "learn.yaml"
        if not derived.exists():
            raise FileNotFoundError(f"--config not given and no sibling learn.yaml at {derived}.")
        config_path = str(derived)
        print(f"[*] --config not given, using: {config_path}")

    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    args = argparse.Namespace(**yaml_config)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = cmd_args.batch_size or args.batch_size

    if args.dataset != 'pacs':
        raise NotImplementedError("This ablation currently assumes PACS-style (image, label, domain) data.")

    full_dataset = PACSDataset(root_dir=args.data_dir, transform=None)
    num_classes = len(full_dataset.class_names)
    num_domains = len(full_dataset.domain_names)
    class_names = full_dataset.class_names
    domain_names = full_dataset.domain_names

    forget_loader, retain_loader, unlearn_setting = build_forget_retain_loaders(args, full_dataset, batch_size)
    print(f"[*] unlearn_setting: {unlearn_setting} | forget images: {len(forget_loader.dataset)} | retain images: {len(retain_loader.dataset)}")

    model = ModuleArchitecture(
        model_name=args.model_name,
        num_classes=num_classes,
        moe_layers=getattr(args, 'moe_layers', None),
        num_experts=args.num_experts,
        expert_depth=args.expert_depth,
        expert_hidden_ratio=args.expert_hidden_ratio,
        gate_k=args.gate_k,
        device=device,
    )
    checkpoint = torch.load(cmd_args.checkpoint, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    moe_layers = [m for m in model.modules() if type(m).__name__ == 'DeepMoELayer']
    L = len(moe_layers)
    num_experts = args.num_experts
    k_u = cmd_args.k_u
    alpha = cmd_args.alpha

    print("[*] Collecting pi mass on FORGET set (aggregate + per-class + per-domain)...")
    fg = collect_mass_buckets(model, forget_loader, device, num_experts, num_classes, num_domains, moe_layers)
    print("[*] Collecting pi mass on RETAIN set (aggregate + per-class + per-domain)...")
    rt = collect_mass_buckets(model, retain_loader, device, num_experts, num_classes, num_domains, moe_layers)

    for l in range(L):
        print(f"\n{'='*78}\nMoE layer {l}  (k_u={k_u}, alpha={alpha})\n{'='*78}")

        rho_agg = fg["agg"][l] - alpha * rt["agg"][l]
        top_agg = topk_experts(rho_agg, k_u)
        print(f"[normal / aggregate]  rho={[round(x,4) for x in rho_agg.tolist()]}")
        print(f"                       top-{k_u} experts = {sorted(top_agg)}")

        print(f"\n  per-class (forget-set classes; skip if forget-side tokens < {cmd_args.min_tokens:.0f}):")
        print(f"    {'class':<12}{'fg_tok':>10}{'top-k':>14}{'match agg':>12}{'jaccard':>10}")
        for c in range(num_classes):
            if fg["tok_by_class"][l, c] < cmd_args.min_tokens:
                continue
            retain_mass_c = rt["by_class"][l, c] if rt["tok_by_class"][l, c] >= cmd_args.min_tokens else rt["agg"][l]
            rho_c = fg["by_class"][l, c] - alpha * retain_mass_c
            top_c = topk_experts(rho_c, k_u)
            match = "yes" if top_c == top_agg else "NO"
            print(f"    {class_names[c]:<12}{int(fg['tok_by_class'][l,c]):>10}{str(sorted(top_c)):>14}{match:>12}{jaccard(top_c, top_agg):>10.2f}")

        print(f"\n  per-domain (forget-set domains; skip if forget-side tokens < {cmd_args.min_tokens:.0f}):")
        print(f"    {'domain':<14}{'fg_tok':>10}{'top-k':>14}{'match agg':>12}{'jaccard':>10}")
        for d in range(num_domains):
            if fg["tok_by_domain"][l, d] < cmd_args.min_tokens:
                continue
            retain_mass_d = rt["by_domain"][l, d] if rt["tok_by_domain"][l, d] >= cmd_args.min_tokens else rt["agg"][l]
            rho_d = fg["by_domain"][l, d] - alpha * retain_mass_d
            top_d = topk_experts(rho_d, k_u)
            match = "yes" if top_d == top_agg else "NO"
            print(f"    {domain_names[d]:<14}{int(fg['tok_by_domain'][l,d]):>10}{str(sorted(top_d)):>14}{match:>12}{jaccard(top_d, top_agg):>10.2f}")


if __name__ == "__main__":
    main()
