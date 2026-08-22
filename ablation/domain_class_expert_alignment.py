"""
Ablation: does the router's expert preference for a joint (domain, class) slice
-- e.g. "art_painting / dog" -- generalize from TRAIN images to held-out TEST
images of that same slice?

Existing ablations (`router_train_test_consistency.py`, `pi_score_train_test_ablation.py`)
only check domain-only or class-only train/test consistency (marginals). This script
buckets routing mass jointly by (domain, class) so a mismatch localized to a specific
slice -- e.g. only "sketch / dog" flips expert between train and test -- doesn't get
averaged away inside a domain- or class-level aggregate.

Natural routing only (model.eval(), no unlearning expert masking): for each MoE layer
and each (domain, class) pair with enough tokens on both sides, compares the top-k_u
experts by average pi-mass on TRAIN images of that pair vs TEST images of that pair.
"""
import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.pytorch_dataset.pacs import PACSDataset
from dataset.transform.test_transform import get_test_transform
from architecture.module import ModuleArchitecture
from module_diagnostics import ApplyTransform, set_seed


@torch.no_grad()
def collect_pair_mass(model, loader, device, num_experts, num_classes, num_domains, moe_layers):
    """Single pass; returns per-layer routing mass and token counts bucketed by
    (domain, class) pair. Pair index = domain * num_classes + label."""
    L = len(moe_layers)
    num_pairs = num_domains * num_classes
    mass_by_pair = torch.zeros(L, num_pairs, num_experts)
    tok_by_pair = torch.zeros(L, num_pairs)

    model.eval()
    for batch in loader:
        images = batch[0].to(device)
        labels = batch[1].to(device).long()
        domains = batch[2].to(device).long()
        B = images.size(0)

        model(images)

        pair_ids = (domains * num_classes + labels).cpu()

        for l, m in enumerate(moe_layers):
            pi_flat = m.last_pi_all.detach().cpu()  # [B*S, num_experts]
            S = pi_flat.size(0) // B
            pair_tok = pair_ids.repeat_interleave(S)
            mass_by_pair[l].index_add_(0, pair_tok, pi_flat)
            tok_by_pair[l].index_add_(0, pair_tok, torch.ones(pair_tok.size(0)))

    mass_by_pair = mass_by_pair / tok_by_pair.clamp(min=1).unsqueeze(-1)
    return mass_by_pair, tok_by_pair


def topk_experts(mass, k):
    _, idx = mass.topk(k, dim=-1)
    return set(idx.tolist())


def jaccard(a, b):
    return len(a & b) / max(len(a | b), 1)


def main():
    parser = argparse.ArgumentParser(
        description="Domain x class expert-selection consistency: train vs test.")
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--k_u', type=int, default=2,
                         help="Top-k experts to compare by pi mass (matches the unlearning k_u convention).")
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--min-tokens', type=float, default=150.0,
                         help="Skip a (domain, class) pair unless both train- and test-side token counts clear this.")
    parser.add_argument('--wandb-run-id', type=str, default=None,
                         help="Resume and log into this existing wandb run id instead of creating a new one.")
    parser.add_argument('--no-wandb', action='store_true')
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
    class_names = full_dataset.class_names
    domain_names = full_dataset.domain_names
    num_classes = len(class_names)
    num_domains = len(domain_names)
    pair_names = [f"{domain_names[d]}/{class_names[c]}"
                  for d in range(num_domains) for c in range(num_classes)]

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    test_size = int(0.1 * total_size)
    unseen_size = total_size - train_size - test_size
    generator = torch.Generator().manual_seed(args.seed)
    train_subset, test_subset, _ = random_split(
        full_dataset, [train_size, test_size, unseen_size], generator=generator
    )
    train_loader = DataLoader(ApplyTransform(train_subset, get_test_transform()), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(ApplyTransform(test_subset, get_test_transform()), batch_size=batch_size, shuffle=False)
    print(f"[*] train images: {len(train_subset)} | test images: {len(test_subset)}")

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

    use_wandb = not cmd_args.no_wandb
    if use_wandb:
        import wandb
        if cmd_args.wandb_run_id:
            wandb.init(project="MoE", id=cmd_args.wandb_run_id, resume="must")
        else:
            wandb.init(
                project="MoE",
                name=f"domain_class_alignment_{Path(cmd_args.checkpoint).parent.parent.name}_{Path(cmd_args.checkpoint).stem}",
                tags=['pi_score_ablation', 'train_test', 'domain_class', getattr(args, 'dataset', 'unknown')],
                config={**yaml_config, "checkpoint": str(cmd_args.checkpoint), "k_u": k_u},
            )

    print("[*] Collecting pi mass on TRAIN split, bucketed by (domain, class)...")
    mass_tr, tok_tr = collect_pair_mass(model, train_loader, device, num_experts, num_classes, num_domains, moe_layers)
    print("[*] Collecting pi mass on TEST split, bucketed by (domain, class)...")
    mass_te, tok_te = collect_pair_mass(model, test_loader, device, num_experts, num_classes, num_domains, moe_layers)

    wandb_rows = [] if use_wandb else None

    for l in range(L):
        print(f"\n{'='*100}\nMoE layer {l}  (k_u={k_u}, min_tokens={cmd_args.min_tokens:.0f})\n{'='*100}")
        print(f"  {'domain/class':<24}{'tr_tok':>9}{'te_tok':>9}{'top-k(tr)':>13}{'top-k(te)':>13}{'match':>8}{'jaccard':>10}")

        n_eval, n_match, jac_sum = 0, 0, 0.0
        for i in range(len(pair_names)):
            if tok_tr[l, i] < cmd_args.min_tokens or tok_te[l, i] < cmd_args.min_tokens:
                continue
            top_tr = topk_experts(mass_tr[l, i], k_u)
            top_te = topk_experts(mass_te[l, i], k_u)
            match = top_tr == top_te
            jac = jaccard(top_tr, top_te)
            n_eval += 1
            n_match += int(match)
            jac_sum += jac
            print(f"  {pair_names[i]:<24}{int(tok_tr[l, i]):>9}{int(tok_te[l, i]):>9}"
                  f"{str(sorted(top_tr)):>13}{str(sorted(top_te)):>13}{('yes' if match else 'NO'):>8}{jac:>10.2f}")
            if wandb_rows is not None:
                wandb_rows.append([
                    l, domain_names[i // num_classes], class_names[i % num_classes],
                    int(tok_tr[l, i]), int(tok_te[l, i]),
                    str(sorted(top_tr)), str(sorted(top_te)), match, jac,
                ])

        match_rate = n_match / n_eval if n_eval else float('nan')
        mean_jac = jac_sum / n_eval if n_eval else float('nan')
        print(f"\n  [summary] layer {l}: {n_match}/{n_eval} pairs matched ({match_rate * 100:.1f}%), "
              f"mean jaccard = {mean_jac:.3f} (skipped {len(pair_names) - n_eval} pairs below min-tokens)")

        if use_wandb:
            import wandb
            wandb.log({
                f"domain_class_alignment/layer{l}/pair_match_rate": match_rate,
                f"domain_class_alignment/layer{l}/pair_mean_jaccard": mean_jac,
                f"domain_class_alignment/layer{l}/pairs_evaluated": n_eval,
            })

    if use_wandb:
        import wandb
        table = wandb.Table(
            columns=["layer", "domain", "class", "train_tokens", "test_tokens",
                     "topk_train", "topk_test", "match", "jaccard"],
            data=wandb_rows,
        )
        wandb.log({"domain_class_alignment/table": table})
        wandb.finish()


if __name__ == "__main__":
    main()
