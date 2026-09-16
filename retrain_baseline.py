# gold-standard "retrain" baseline for the Perf. Gap ablation metric:
# trains a fresh module model that never sees the forget set, using the
# exact same seeded train/test/forget/retain split as unlearn.py, so its
# FA/RA/TA/MIA are directly comparable to the unlearned-model runs.
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from torchvision import transforms
import wandb
import yaml

from dataset.pytorch_dataset.pacs import PACSDataset
from dataset.pytorch_dataset.officehome import OfficeHomeDataset

from dataset.transform.train_transform import get_train_transform
from dataset.transform.test_transform import get_test_transform
from dataset.transform.forget_test_transform import get_forget_test_transform
from dataset.transform.retain_test_transform import get_retain_test_transform
from dataset.transform.unseen_transform import get_unseen_transform

from architecture.module import ModuleArchitecture
from approx_algo.module import Module


class ApplyTransform(Dataset):
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
        torch.cuda.reset_peak_memory_stats()


def get_domain(dataset, idx):
    if hasattr(dataset, 'domains'):
        return dataset.domains[idx]
    data_tuple = dataset[idx]
    domain_val = data_tuple[2].item() if isinstance(data_tuple[2], torch.Tensor) else data_tuple[2]
    return int(domain_val)


def main():
    parser = argparse.ArgumentParser(description="Retrain-from-scratch gold-standard baseline (excludes forget set).")
    parser.add_argument('--config', type=str, required=True)
    cmd_args = parser.parse_args()

    with open(cmd_args.config, 'r') as f:
        yaml_config = yaml.safe_load(f)
    args = argparse.Namespace(**yaml_config)

    yaml_filename = os.path.splitext(os.path.basename(cmd_args.config))[0]
    output_dir = os.path.join(getattr(args, 'output_root', 'runs'), yaml_filename)
    os.makedirs(output_dir, exist_ok=True)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    unlearn_setting = args.unlearn_setting
    # tên theo quy ước main table nếu config cung cấp metadata; nếu không thì giữ như cũ
    run_name = getattr(args, 'study_name', None) or (
        f"Retraining__PACS__{unlearn_setting}__{yaml_filename}__seed{args.seed}"
        if getattr(args, 'baseline_name', None) == "Retraining"
        else f"retrain_{unlearn_setting}_{yaml_filename}_seed{args.seed}")

    print("\n" + "=" * 40)
    print(f"[*] retrain-from-scratch baseline (gold standard, never sees forget set)")
    print(f"[*] config: {cmd_args.config}")
    print(f"[*] device: {device}")

    wandb.init(project="MoE", name=run_name, config=yaml_config,
               group=getattr(args, 'wandb_group', None),
               tags=getattr(args, 'wandb_tags', None))

    if args.dataset == 'pacs':
        full_dataset = PACSDataset(root_dir=args.data_dir, transform=None)
        num_classes = 7
    elif args.dataset == 'officehome':
        full_dataset = OfficeHomeDataset(root_dir=args.data_dir, transform=None)
        num_classes = 65
    else:
        raise ValueError(f"Unsupported dataset for this baseline: {args.dataset}")

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    test_size = int(0.1 * total_size)
    unseen_size = total_size - train_size - test_size

    # same seed + same split call as unlearn.py -> identical train/test/unseen indices.
    generator = torch.Generator().manual_seed(args.seed)
    train_subset, test_subset, unseen_subset = random_split(
        full_dataset, [train_size, test_size, unseen_size], generator=generator
    )

    if unlearn_setting == 'class':
        forget_classes = args.forget_classes
        if not isinstance(forget_classes, list):
            forget_classes = [forget_classes]
        print(f"[*] target classes to hold out of training: {forget_classes}")

        forget_train_indices, retain_train_indices, retain_test_indices = [], [], []
        # non-member side of the MIA, matched to the forget set in class composition
        # (see unlearn.py for why the pool cannot be the whole unseen split).
        mia_unseen_indices = []
        for idx in train_subset.indices:
            label = full_dataset.labels[idx]
            (forget_train_indices if label in forget_classes else retain_train_indices).append(idx)
        for idx in test_subset.indices:
            label = full_dataset.labels[idx]
            if label not in forget_classes:
                retain_test_indices.append(idx)
            else:
                mia_unseen_indices.append(idx)
        for idx in unseen_subset.indices:
            if full_dataset.labels[idx] in forget_classes:
                mia_unseen_indices.append(idx)

        forget_subset = Subset(full_dataset, forget_train_indices)
        retain_subset = Subset(full_dataset, retain_train_indices)
        test_subset = Subset(full_dataset, retain_test_indices)

    elif unlearn_setting == 'domain':
        forget_domains = args.forget_domains
        if not isinstance(forget_domains, list):
            forget_domains = [forget_domains]
        print(f"[*] target domains to hold out of training: {forget_domains}")

        forget_train_indices, retain_train_indices, retain_test_indices = [], [], []
        # non-member side of the MIA, matched to the forget set in domain composition.
        mia_unseen_indices = []
        for idx in train_subset.indices:
            domain = get_domain(full_dataset, idx)
            (forget_train_indices if domain in forget_domains else retain_train_indices).append(idx)
        for idx in test_subset.indices:
            domain = get_domain(full_dataset, idx)
            if domain not in forget_domains:
                retain_test_indices.append(idx)
            else:
                mia_unseen_indices.append(idx)
        for idx in unseen_subset.indices:
            if get_domain(full_dataset, idx) in forget_domains:
                mia_unseen_indices.append(idx)

        forget_subset = Subset(full_dataset, forget_train_indices)
        retain_subset = Subset(full_dataset, retain_train_indices)
        test_subset = Subset(full_dataset, retain_test_indices)
    else:
        raise ValueError("unlearn_setting must be 'class' or 'domain'")

    print(f"[*] retain(train): {len(retain_subset)} | forget(held out, never trained on): {len(forget_subset)} "
          f"| test: {len(test_subset)} | unseen: {len(unseen_subset)}")

    retain_train_loader = DataLoader(ApplyTransform(retain_subset, get_train_transform()),
                                     batch_size=args.batch_size, shuffle=True, num_workers=4)
    forget_test_loader = DataLoader(ApplyTransform(forget_subset, get_forget_test_transform()),
                                    batch_size=args.batch_size, shuffle=False, num_workers=4)
    retain_test_loader = DataLoader(ApplyTransform(retain_subset, get_retain_test_transform()),
                                    batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(ApplyTransform(test_subset, get_test_transform()),
                             batch_size=args.batch_size, shuffle=False, num_workers=4)
    unseen_loader = DataLoader(ApplyTransform(unseen_subset, get_unseen_transform()),
                               batch_size=args.batch_size, shuffle=False, num_workers=4)

    # MIA non-member pool, same deterministic transform as forget_test_loader.
    mia_unseen_subset = Subset(full_dataset, mia_unseen_indices)
    mia_unseen_loader = DataLoader(ApplyTransform(mia_unseen_subset, get_test_transform()),
                                   batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"[*] mia non-member pool: {len(mia_unseen_subset)} (matched to forget set, deterministic transform)")

    model = ModuleArchitecture(
        model_name=args.model_name,
        num_classes=num_classes,
        pretrained=args.pretrained,
        moe_layers=getattr(args, 'moe_layers', None),
        num_experts=args.num_experts,
        expert_depth=args.expert_depth,
        expert_hidden_ratio=args.expert_hidden_ratio,
        gate_k=args.gate_k,
            gate_norm=getattr(args, 'gate_norm', 'softmax'),
        device=device,
    )
    model._set_grad_mode("learning")

    criteria = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    algo_wrapper = Module(
        model=model,
        train_loader=retain_train_loader,   # gold standard: forget data never enters training.
        test_loader=test_loader,
        unseen_loader=unseen_loader,
        forget_loader=forget_test_loader,   # unused by learn(); kept only for API compatibility.
        forget_test_loader=forget_test_loader,
        retain_loader=retain_train_loader,
        retain_test_loader=retain_test_loader,
        optimizer=optimizer,
        criteria=criteria,
        num_epoch=args.epochs,
        lambda_sparse=getattr(args, 'lambda_sparse', 1.0),
        lambda_balance=getattr(args, 'lambda_balance', 1.0),
        lambda_div=getattr(args, 'lambda_div', 1.0),
        alpha=1.0,
        beta=1.0,
        gamma=1.0,
        eta=1.0,
        k_u=getattr(args, 'k_u', 4),
        domain_names=getattr(full_dataset, 'domain_names', None),
        class_names=getattr(full_dataset, 'class_names', None),
        device=device,
    )

    algo_wrapper.set_mia_unseen_loader(mia_unseen_loader)
    algo_wrapper.learn_eval_every = getattr(args, 'learn_eval_every', 0)

    ckpt_prefix = os.path.join(output_dir, yaml_filename)
    print(f"\n[*] starting retrain-from-scratch phase ({args.epochs} epochs, batch_size={args.batch_size})")
    algo_wrapper.learn(ckpt_path=ckpt_prefix, ema_alpha=getattr(args, 'ema_alpha', 0.9))

    wandb.finish()


if __name__ == "__main__":
    main()
