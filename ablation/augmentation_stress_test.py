"""
Ablation: augmentation stress test for expert-selection stability.

For each sample, computes the top-k_u expert set (by that image's own
token-averaged pi-mass) on the clean image, then on N heavily-augmented views
of the same image, and checks whether the augmented views still route to the
same expert set. Reports the % of augmented views that match (view-level) and
the % of samples where a majority of their views matched (sample-level "still
keeps its expert" rate), overall and broken down by domain/class.

This is a per-sample robustness check, not a train/test generalization check
(see router_train_test_consistency.py / pi_score_train_test_ablation.py /
domain_class_expert_alignment.py for that): a sample can be train/test-aligned
but still be fragile to input perturbation.

Note: because heavy augmentation (rotation, affine, perspective) does not
preserve patch-to-patch spatial correspondence, token-position-aligned
comparison between clean and augmented views is meaningless -- a sample's
"expert set" is instead defined as the top-k_u experts by mean pi-mass
averaged over that image's own tokens.
"""
import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset.pytorch_dataset.pacs import PACSDataset
from architecture.module import ModuleArchitecture
from module_diagnostics import set_seed

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_clean_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_stress_transform():
    return transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomAffine(0, translate=(0.2, 0.2), shear=20, scale=(0.6, 1.4)),
        transforms.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.3),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
    ])


class RawResized(Dataset):
    """Wraps a Subset of a transform=None PACSDataset; returns PIL images resized to 224x224."""

    def __init__(self, subset):
        self.subset = subset
        self.resize = transforms.Resize((224, 224))

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label, domain = self.subset[idx]
        return self.resize(image), label, domain


def topk_experts(mass, k):
    _, idx = mass.topk(k, dim=-1)
    return set(idx.tolist())


def print_breakdown(title, name_list, retained, total):
    print(f"\n  per-{title}:")
    print(f"    {title:<16}{'samples':>10}{'retained':>10}{'rate':>10}")
    for i in range(len(name_list)):
        if total[i] == 0:
            continue
        rate = retained[i] / total[i]
        print(f"    {name_list[i]:<16}{int(total[i]):>10}{int(retained[i]):>10}{rate:>10.2%}")


def main():
    parser = argparse.ArgumentParser(
        description="Augmentation stress test: does a sample keep its expert set under heavy augmentation?")
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--k_u', type=int, default=2,
                         help="Top-k experts to compare by pi mass (matches the unlearning k_u convention).")
    parser.add_argument('--num-augments', type=int, default=5, help="N augmented views per sample.")
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

    if args.dataset != 'pacs':
        raise NotImplementedError("This ablation currently assumes PACS-style (image, label, domain) data.")

    full_dataset = PACSDataset(root_dir=args.data_dir, transform=None)
    class_names = full_dataset.class_names
    domain_names = full_dataset.domain_names
    num_classes = len(class_names)
    num_domains = len(domain_names)

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    test_size = int(0.1 * total_size)
    unseen_size = total_size - train_size - test_size
    generator = torch.Generator().manual_seed(args.seed)
    _, test_subset, _ = random_split(
        full_dataset, [train_size, test_size, unseen_size], generator=generator
    )
    test_pool = RawResized(test_subset)
    test_loader = DataLoader(test_pool, batch_size=1, shuffle=False, collate_fn=lambda b: b[0])
    print(f"[*] test images: {len(test_pool)} | augmented views per image: {cmd_args.num_augments}")

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
    model.eval()

    moe_layers = [m for m in model.modules() if type(m).__name__ == 'DeepMoELayer']
    L = len(moe_layers)
    k_u = cmd_args.k_u
    N = cmd_args.num_augments

    clean_tf = get_clean_transform()
    stress_tf = get_stress_transform()

    use_wandb = not cmd_args.no_wandb
    if use_wandb:
        import wandb
        if cmd_args.wandb_run_id:
            wandb.init(project="MoE", id=cmd_args.wandb_run_id, resume="must")
        else:
            wandb.init(
                project="MoE",
                name=f"aug_stress_{Path(cmd_args.checkpoint).parent.parent.name}_{Path(cmd_args.checkpoint).stem}",
                tags=['pi_score_ablation', 'augmentation_stress', getattr(args, 'dataset', 'unknown')],
                config={**yaml_config, "checkpoint": str(cmd_args.checkpoint), "k_u": k_u, "num_augments": N},
            )

    # per-layer accumulators
    view_total = torch.zeros(L)
    view_matched = torch.zeros(L)
    sample_total = torch.zeros(L)
    sample_retained = torch.zeros(L)
    domain_total = torch.zeros(L, num_domains)
    domain_retained = torch.zeros(L, num_domains)
    class_total = torch.zeros(L, num_classes)
    class_retained = torch.zeros(L, num_classes)

    print("[*] Running stress test over the full test split...")
    with torch.no_grad():
        for i, (image, label, domain) in enumerate(test_loader):
            batch = torch.stack([clean_tf(image)] + [stress_tf(image) for _ in range(N)]).to(device)
            model(batch)

            for l, m in enumerate(moe_layers):
                pi_flat = m.last_pi_all.detach().cpu()  # [(N+1)*S, num_experts]
                S = pi_flat.size(0) // (N + 1)
                mass = pi_flat.view(N + 1, S, -1).mean(dim=1)  # [N+1, num_experts]

                clean_top = topk_experts(mass[0], k_u)
                matched = sum(topk_experts(mass[v], k_u) == clean_top for v in range(1, N + 1))

                view_total[l] += N
                view_matched[l] += matched
                sample_total[l] += 1
                retained = matched > N / 2
                sample_retained[l] += int(retained)
                domain_total[l, domain] += 1
                domain_retained[l, domain] += int(retained)
                class_total[l, label] += 1
                class_retained[l, label] += int(retained)

            if (i + 1) % 200 == 0:
                print(f"    ...{i + 1}/{len(test_pool)} samples processed")

    wandb_rows = [] if use_wandb else None

    for l in range(L):
        print(f"\n{'='*90}\nMoE layer {l}  (k_u={k_u}, N={N})\n{'='*90}")
        view_rate = (view_matched[l] / view_total[l]).item()
        sample_rate = (sample_retained[l] / sample_total[l]).item()
        print(f"[overall]  view-level match rate: {view_rate:.2%}  |  "
              f"sample-level retained (majority vote): {sample_rate:.2%} "
              f"({int(sample_retained[l])}/{int(sample_total[l])})")

        print_breakdown("domain", domain_names, domain_retained[l], domain_total[l])
        print_breakdown("class", class_names, class_retained[l], class_total[l])

        if wandb_rows is not None:
            wandb_rows.append([l, "overall", "overall", int(sample_total[l]), sample_rate])
            for d in range(num_domains):
                if domain_total[l, d] > 0:
                    wandb_rows.append([l, "domain", domain_names[d], int(domain_total[l, d]),
                                        (domain_retained[l, d] / domain_total[l, d]).item()])
            for c in range(num_classes):
                if class_total[l, c] > 0:
                    wandb_rows.append([l, "class", class_names[c], int(class_total[l, c]),
                                        (class_retained[l, c] / class_total[l, c]).item()])
            import wandb
            wandb.log({
                f"stress/layer{l}/view_match_rate": view_rate,
                f"stress/layer{l}/sample_retention_rate": sample_rate,
            })

    if use_wandb:
        import wandb
        table = wandb.Table(
            columns=["layer", "group_type", "name", "num_samples", "retention_rate"],
            data=wandb_rows,
        )
        wandb.log({"stress/table": table})
        wandb.finish()


if __name__ == "__main__":
    main()
