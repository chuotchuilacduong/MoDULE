"""
Ablation: pi-score (router probability) expert selection, train vs test, broken down
by class and by domain -- not just the aggregate/pooled score.

Motivation: `Module._get_routing_mass` (approx_algo/module.py:121) and the "diff"/"ratio"
selection rules only ever look at ONE pooled pi-mass vector per layer. This script asks:
if you computed the same top-k expert selection separately per class or per domain, would
it agree with what you'd pick using the whole train split -- and does that selection
still hold on held-out TEST images of the same class/domain? This is a pure learning-phase
diagnostic (single checkpoint, plain 80/10/10 train/test split, no forget/retain involved).
"""
import argparse
import sys
from pathlib import Path

import torch
import wandb
import yaml
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.pytorch_dataset.pacs import PACSDataset
from dataset.transform.test_transform import get_test_transform
from architecture.module import ModuleArchitecture
from module_diagnostics import ApplyTransform, set_seed


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


def topk_experts(mass, k):
    _, idx = mass.topk(k, dim=-1)
    return set(idx.tolist())


def jaccard(a, b):
    return len(a & b) / max(len(a | b), 1)


def report_group(name_list, mass_tr, tok_tr, mass_te, tok_te, top_agg_tr, k_u, min_tokens, label, wandb_rows=None, layer=None):
    print(f"\n  per-{label} (skip if train-side tokens < {min_tokens:.0f}):")
    print(f"    {label:<12}{'tr_tok':>9}{'te_tok':>9}{'top-k(tr)':>13}{'top-k(te)':>13}{'tr==te':>9}{'jac(tr,te)':>12}{'tr==agg':>10}")
    for i in range(len(name_list)):
        if tok_tr[i] < min_tokens:
            continue
        top_tr = topk_experts(mass_tr[i], k_u)
        top_te = topk_experts(mass_te[i], k_u) if tok_te[i] >= min_tokens else None
        te_str = str(sorted(top_te)) if top_te is not None else "(too few)"
        match_tr_te = "yes" if (top_te is not None and top_tr == top_te) else ("NO" if top_te is not None else "-")
        jac_val = jaccard(top_tr, top_te) if top_te is not None else None
        jac = f"{jac_val:.2f}" if jac_val is not None else "-"
        match_agg = "yes" if top_tr == top_agg_tr else "NO"
        print(f"    {name_list[i]:<12}{int(tok_tr[i]):>9}{int(tok_te[i]):>9}{str(sorted(top_tr)):>13}{te_str:>13}{match_tr_te:>9}{jac:>12}{match_agg:>10}")
        if wandb_rows is not None:
            wandb_rows.append([
                layer, label, name_list[i], int(tok_tr[i]), int(tok_te[i]),
                str(sorted(top_tr)), te_str, match_tr_te, jac_val if jac_val is not None else -1.0, match_agg,
            ])


def main():
    parser = argparse.ArgumentParser(description="Pi-score train-vs-test selection ablation, per class and per domain.")
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--k_u', type=int, default=2, help="Top-k experts to select by pi mass (same k_u convention as unlearning).")
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--min-tokens', type=float, default=500.0)
    parser.add_argument('--wandb-run-id', type=str, default=None,
                         help="Resume and log into this existing wandb run id (e.g. the checkpoint's own training run) "
                              "instead of creating a new one.")
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
    num_classes = len(full_dataset.class_names)
    num_domains = len(full_dataset.domain_names)
    class_names = full_dataset.class_names
    domain_names = full_dataset.domain_names

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
        if cmd_args.wandb_run_id:
            wandb.init(project="MoE", id=cmd_args.wandb_run_id, resume="must")
        else:
            wandb.init(
                project="MoE",
                name=f"pi_score_traintest_{Path(cmd_args.checkpoint).parent.parent.name}_{Path(cmd_args.checkpoint).stem}",
                tags=['pi_score_ablation', 'train_test', getattr(args, 'dataset', 'unknown')],
                config={**yaml_config, "checkpoint": str(cmd_args.checkpoint), "k_u": k_u},
            )

    print("[*] Collecting pi mass on TRAIN split (aggregate + per-class + per-domain)...")
    tr = collect_mass_buckets(model, train_loader, device, num_experts, num_classes, num_domains, moe_layers)
    print("[*] Collecting pi mass on TEST split (aggregate + per-class + per-domain)...")
    te = collect_mass_buckets(model, test_loader, device, num_experts, num_classes, num_domains, moe_layers)

    wandb_rows = [] if use_wandb else None

    for l in range(L):
        print(f"\n{'='*90}\nMoE layer {l}  (k_u={k_u})\n{'='*90}")

        top_agg_tr = topk_experts(tr["agg"][l], k_u)
        top_agg_te = topk_experts(te["agg"][l], k_u)
        agg_match = top_agg_tr == top_agg_te
        print(f"[normal / aggregate]  train top-{k_u} = {sorted(top_agg_tr)}  |  test top-{k_u} = {sorted(top_agg_te)}"
              f"  |  match: {'yes' if agg_match else 'NO'}")

        if use_wandb:
            wandb.log({
                f"pi_score/layer{l}/aggregate_train_test_match": int(agg_match),
                f"pi_score/layer{l}/aggregate_jaccard": jaccard(top_agg_tr, top_agg_te),
            })

        report_group(class_names, tr["by_class"][l], tr["tok_by_class"][l],
                     te["by_class"][l], te["tok_by_class"][l], top_agg_tr, k_u, cmd_args.min_tokens, "class",
                     wandb_rows=wandb_rows, layer=l)
        report_group(domain_names, tr["by_domain"][l], tr["tok_by_domain"][l],
                     te["by_domain"][l], te["tok_by_domain"][l], top_agg_tr, k_u, cmd_args.min_tokens, "domain",
                     wandb_rows=wandb_rows, layer=l)

    if use_wandb:
        table = wandb.Table(
            columns=["layer", "group_type", "name", "train_tokens", "test_tokens",
                     "topk_train", "topk_test", "train_eq_test", "jaccard_train_test", "train_eq_aggregate"],
            data=wandb_rows,
        )
        wandb.log({"pi_score/train_test_selection_table": table})
        wandb.finish()


if __name__ == "__main__":
    main()
