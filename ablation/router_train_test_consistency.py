"""
Ablation: does the router's per-domain expert selection on the TRAIN split
match its selection on the TEST split (same domains, held-out images)?

For each MoE layer and each domain, computes the per-domain all-token routing
mass vector separately on train_loader and test_loader, then reports the
top-1 (argmax) expert on each split and the L1 distance between the two full
mass vectors. A mismatched top-1 expert or large L1 distance means the router
routes a domain differently depending on whether the images were seen during
training -- i.e. the routing rule doesn't generalize to held-out data for
that domain.
"""
import argparse
import sys
from pathlib import Path

import torch
import yaml

from torch.utils.data import DataLoader, random_split, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.pytorch_dataset.pacs import PACSDataset
from dataset.transform.test_transform import get_test_transform
from architecture.module import ModuleArchitecture

from module_diagnostics import (
    ApplyTransform, set_seed, extract_router_diagnostics,
)


def get_domain(dataset, idx):
    if hasattr(dataset, 'domains'):
        return dataset.domains[idx]
    data_tuple = dataset[idx]
    val = data_tuple[2].item() if isinstance(data_tuple[2], torch.Tensor) else data_tuple[2]
    return int(val)


def per_domain_mass(diag, layer):
    mass = diag["mass_all_sum"][layer] / diag["token_count_per_domain"][layer].clamp(min=1).unsqueeze(-1)
    return mass  # [num_domains, num_experts]


def main():
    parser = argparse.ArgumentParser(description="Train vs. test per-domain routing consistency ablation.")
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=None)
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
    num_classes = 7
    domain_names = full_dataset.domain_names
    num_domains = len(domain_names)

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

    print("[*] Running router diagnostics on TRAIN split...")
    diag_train = extract_router_diagnostics(
        model=model, dataloader=train_loader, device=device,
        num_experts=args.num_experts, gate_k=args.gate_k,
        num_classes=num_classes, num_domains=num_domains,
    )
    print("[*] Running router diagnostics on TEST split...")
    diag_test = extract_router_diagnostics(
        model=model, dataloader=test_loader, device=device,
        num_experts=args.num_experts, gate_k=args.gate_k,
        num_classes=num_classes, num_domains=num_domains,
    )

    L = diag_train["num_layers"]
    for l in range(L):
        mass_tr = per_domain_mass(diag_train, l)
        mass_te = per_domain_mass(diag_test, l)

        print(f"\n{'='*78}\nMoE layer {l} -- train vs test per-domain routing mass\n{'='*78}")
        header = "  domain".ljust(14) + "top1(train)".rjust(13) + "top1(test)".rjust(12) + "match".rjust(8) + "L1(train,test)".rjust(16)
        print(header)
        for d in range(num_domains):
            top1_tr = int(mass_tr[d].argmax())
            top1_te = int(mass_te[d].argmax())
            l1 = (mass_tr[d] - mass_te[d]).abs().sum().item()
            match = "yes" if top1_tr == top1_te else "NO"
            print(
                f"  {domain_names[d]:<12}"
                f"{('e' + str(top1_tr)):>13}"
                f"{('e' + str(top1_te)):>12}"
                f"{match:>8}"
                f"{l1:>16.4f}"
            )

        print(f"\n  full mass vectors (train):")
        for d in range(num_domains):
            print(f"    {domain_names[d]:<12} " + " ".join(f"{v:.4f}" for v in mass_tr[d].tolist()))
        print(f"  full mass vectors (test):")
        for d in range(num_domains):
            print(f"    {domain_names[d]:<12} " + " ".join(f"{v:.4f}" for v in mass_te[d].tolist()))


if __name__ == "__main__":
    main()
