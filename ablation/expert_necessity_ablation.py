"""
Ablation: were the experts learned WELL in the learning phase?

The purity/entropy tables in module_diagnostics are *descriptive* -- they show how
routing is distributed, but not whether each expert carries real, non-redundant
function. This ablation tests that CAUSALLY, by constraining which experts the
router may reach at inference (no weight updates) and watching accuracy move:

  sufficiency   keep only the top-j experts per layer (ranked by routing mass)
                and force routing within them. If accuracy already holds at
                small j, a few experts carry the task.
  necessity     route through the COMPLEMENT (drop the top-t experts). A large
                accuracy drop => those experts carry unique function (well
                learned); a small drop => they are redundant.
  ranking test  keep j RANDOM experts vs the top-j. If top-j clearly beats
                random-j the router has *differentiated* its experts; if they
                are equal the experts are interchangeable (poorly specialised).

Routing is constrained by masking the router logits to -inf for disallowed
experts (metric.router_metrics.forced_routing_accuracy), which keeps
active_k == gate_k in every condition -- so only WHICH experts are reachable
changes, never how many fire. Everything runs in eval()/no_grad, so noise_std
is inactive and no gradients are taken. To keep every condition well defined we
only ever leave >= gate_k experts reachable.

PACS only (image, label, domain), consistent with the other ablations.
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.pytorch_dataset.pacs import PACSDataset
from dataset.transform.test_transform import get_test_transform
from architecture.module import ModuleArchitecture

from module_diagnostics import ApplyTransform, set_seed, evaluate_accuracy
from metric.router_metrics import mass_and_entropy, forced_routing_accuracy, expert_hit_rate


def main():
    parser = argparse.ArgumentParser(
        description="Causal expert necessity/sufficiency/redundancy ablation (learning-phase check).")
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'test'],
                        help="Which split to evaluate on (default: held-out test).")
    parser.add_argument('--random-draws', type=int, default=3,
                        help="Random-expert draws to average for the ranking test.")
    parser.add_argument('--tolerance', type=float, default=2.0,
                        help="Accuracy drop (percentage points) still considered 'no real loss' "
                             "when reporting the minimal sufficient expert count.")
    parser.add_argument('--wandb', action='store_true',
                        help="Log scalars/curves to wandb (off by default -- prints to stdout).")
    cmd_args = parser.parse_args()

    config_path = cmd_args.config
    if config_path is None:
        derived = Path(cmd_args.checkpoint).resolve().parent.parent / "learn.yaml"
        if not derived.exists():
            raise FileNotFoundError(f"--config not given and no sibling learn.yaml at {derived}.")
        config_path = str(derived)
        print(f"[*] --config not given, using: {config_path}")

    import yaml
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    args = argparse.Namespace(**yaml_config)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = cmd_args.batch_size or args.batch_size

    if args.dataset != 'pacs':
        raise NotImplementedError("This ablation currently assumes PACS-style (image, label, domain) data.")

    full_dataset = PACSDataset(root_dir=args.data_dir, transform=None)
    num_classes = 7

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    test_size = int(0.1 * total_size)
    unseen_size = total_size - train_size - test_size
    generator = torch.Generator().manual_seed(args.seed)
    train_subset, test_subset, _ = random_split(
        full_dataset, [train_size, test_size, unseen_size], generator=generator)

    subset = test_subset if cmd_args.split == 'test' else train_subset
    loader = DataLoader(ApplyTransform(subset, get_test_transform()), batch_size=batch_size, shuffle=False)
    print(f"[*] split={cmd_args.split} | images={len(subset)}")

    model = ModuleArchitecture(
        model_name=args.model_name, num_classes=num_classes,
        moe_layers=getattr(args, 'moe_layers', None),
        num_experts=args.num_experts, expert_depth=args.expert_depth,
        expert_hidden_ratio=args.expert_hidden_ratio, gate_k=args.gate_k, device=device,
        gate_norm=getattr(args, 'gate_norm', 'softmax'))
    checkpoint = torch.load(cmd_args.checkpoint, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    moe_layers = [m for m in model.modules() if type(m).__name__ == 'DeepMoELayer']
    L = len(moe_layers)
    E = args.num_experts
    gk = args.gate_k
    if L == 0:
        raise ValueError("No DeepMoELayer found -- is this a module_* checkpoint?")
    print(f"[*] MoE layers={L} | experts/layer={E} | gate_k={gk}")

    # ---- baseline (all experts, natural routing) ------------------------------
    # evaluate_accuracy already returns a PERCENTAGE (100*correct/total);
    # forced_routing_accuracy below returns a fraction, hence the 100.0 there only.
    baseline = evaluate_accuracy(model, loader, device)
    print(f"\n[baseline] full-model accuracy (all {E} experts): {baseline:.2f}%")

    # ---- rank experts per layer by routing mass -------------------------------
    mass_vecs, entropies = mass_and_entropy(model, loader, device)
    ranking = [torch.as_tensor(m).argsort(descending=True).tolist() for m in mass_vecs]
    print("\n[per-layer expert routing mass] (desc) and top-gate_k hit rate")
    hit = expert_hit_rate(model, loader, moe_layers,
                          [r[:gk] for r in ranking], device)
    for l in range(L):
        m = torch.as_tensor(mass_vecs[l])
        order = ranking[l]
        pretty = ", ".join(f"e{e}:{m[e].item():.3f}" for e in order)
        print(f"  layer {l}: {pretty}")
        print(f"           entropy={entropies[l]:.3f}/log(E)={np.log(E):.3f} | "
              f"top-{gk} hit rate={hit[l]:.3f}")

    def per_layer_top(j):
        return [r[:j] for r in ranking]

    def per_layer_random(j, seed):
        rng = random.Random(seed)
        return [rng.sample(range(E), j) for _ in range(L)]

    # ---- sufficiency + ranking sweep (keep top-j vs random-j) -----------------
    print(f"\n{'='*70}\nSUFFICIENCY + RANKING  (route within j experts/layer)\n{'='*70}")
    print(f"  {'j':>3}{'acc(top-j)':>13}{'acc(rand-j)':>13}{'top-rand':>11}{'vs full':>10}")
    suff_rows = []
    min_sufficient = E
    for j in range(gk, E + 1):
        acc_top = 100.0 * forced_routing_accuracy(model, loader, moe_layers, per_layer_top(j), device)
        rand_accs = [100.0 * forced_routing_accuracy(model, loader, moe_layers,
                                                      per_layer_random(j, 1000 + s), device)
                     for s in range(cmd_args.random_draws)]
        acc_rand = float(np.mean(rand_accs))
        print(f"  {j:>3}{acc_top:>13.2f}{acc_rand:>13.2f}{acc_top - acc_rand:>11.2f}{acc_top - baseline:>10.2f}")
        suff_rows.append((j, acc_top, acc_rand))
        if acc_top >= baseline - cmd_args.tolerance and j < min_sufficient:
            min_sufficient = j

    # ---- necessity sweep (drop top-t, route through the rest) -----------------
    print(f"\n{'='*70}\nNECESSITY  (drop the top-t experts/layer, route through the rest)\n{'='*70}")
    print(f"  {'t':>3}{'kept':>6}{'acc(complement)':>18}{'drop vs full':>15}")
    nec_rows = []
    for t in range(1, E - gk + 1):
        acc_comp = 100.0 * forced_routing_accuracy(
            model, loader, moe_layers, per_layer_top(t), device, invert_subset=True)
        drop = baseline - acc_comp
        print(f"  {t:>3}{E - t:>6}{acc_comp:>18.2f}{drop:>15.2f}")
        nec_rows.append((t, acc_comp, drop))

    # ---- verdict --------------------------------------------------------------
    top_gk = next(a for j, a, _ in suff_rows if j == gk)
    rand_gk = next(r for j, _, r in suff_rows if j == gk)
    diff_gk = top_gk - rand_gk
    drop_top1 = nec_rows[0][2] if nec_rows else float('nan')
    print(f"\n{'='*70}\nVERDICT\n{'='*70}")
    print(f"  differentiation : top-{gk} vs random-{gk} = +{diff_gk:.2f} pts "
          f"({'clear -> experts differentiated' if diff_gk > 5 else 'small -> experts interchangeable / weakly specialised'})")
    print(f"  redundancy      : ~{min_sufficient}/{E} experts keep accuracy within "
          f"{cmd_args.tolerance:.0f} pts of full "
          f"({'lean, well used' if min_sufficient >= E - gk else 'several redundant experts'})")
    print(f"  necessity(top-1): dropping the single most-used expert/layer costs {drop_top1:.2f} pts "
          f"({'meaningful' if drop_top1 > 2 else 'negligible -> redundant'})")

    if cmd_args.wandb:
        import wandb
        wandb.init(project="MoE",
                   name=f"expert_necessity_{Path(cmd_args.checkpoint).stem}",
                   tags=['ablation', 'expert_necessity', args.dataset],
                   config={**yaml_config, "split": cmd_args.split})
        wandb.log({"necessity/baseline_acc": baseline,
                   "necessity/differentiation_gk": diff_gk,
                   "necessity/min_sufficient_experts": min_sufficient,
                   "necessity/drop_top1": drop_top1})
        wandb.log({"necessity/sufficiency_curve": wandb.Table(
            columns=["j", "acc_top_j", "acc_rand_j"], data=[list(r) for r in suff_rows])})
        wandb.log({"necessity/necessity_curve": wandb.Table(
            columns=["t_dropped", "acc_complement", "drop"], data=[list(r) for r in nec_rows])})
        wandb.finish()


if __name__ == "__main__":
    main()