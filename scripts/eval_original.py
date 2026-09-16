#!/usr/bin/env python3
"""Baseline 'Original model' cho Table 1: đánh giá checkpoint nền TRƯỚC khi unlearn.

unlearn.py không làm được việc này vì evaluate() nằm trong vòng lặp epoch, nên
`epochs: 0` sẽ không log chỉ số nào. Script dựng lại đúng split mà unlearn.py dựng
(cùng seed, cùng transform, cùng non-member pool cho MIA) rồi tính FA/RA/TA/MIA
trên mô hình chưa hề bị unlearn.
"""
import argparse, csv, sys
from pathlib import Path
import torch, yaml, wandb
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset.pytorch_dataset.pacs import PACSDataset
from dataset.pytorch_dataset.officehome import OfficeHomeDataset
from dataset.transform.forget_test_transform import get_forget_test_transform
from dataset.transform.retain_test_transform import get_retain_test_transform
from dataset.transform.test_transform import get_test_transform
from architecture.module import ModuleArchitecture
from metric.fa import forget_acc
from metric.ra import retain_acc
from metric.ta import test_acc
from metric.mia import mia


class ApplyTransform(Dataset):
    def __init__(self, subset, transform=None):
        self.subset, self.transform = subset, transform
        self.resize = transforms.Resize((224, 224))
    def __getitem__(self, i):
        d = self.subset[i]
        img = self.resize(d[0])
        return ((self.transform(img) if self.transform else img),) + d[1:]
    def __len__(self):
        return len(self.subset)


def get_domain(ds, i):
    return ds.domains[i] if hasattr(ds, "domains") else int(ds[i][2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="config unlearn để lấy split/kiến trúc")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--checkpoint", help="ghi đè pretrained_model_path (vd checkpoint retrain)")
    ap.add_argument("--baseline", default="Original", help="tên baseline cho W&B/CSV")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    args = argparse.Namespace(**cfg)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    scen = args.unlearn_setting
    bs = getattr(args, "batch_size", 128)

    ds_name = str(getattr(args, 'dataset', 'pacs')).lower()
    full = (OfficeHomeDataset if ds_name == 'officehome' else PACSDataset)(root_dir=args.data_dir, transform=None)
    ds_label = 'OfficeHome' if ds_name == 'officehome' else 'PACS'
    n = len(full); tr = int(0.8 * n); te = int(0.1 * n)
    g = torch.Generator().manual_seed(args.seed)
    train_s, test_s, unseen_s = random_split(full, [tr, te, n - tr - te], generator=g)

    key = (lambda i: full.labels[i]) if scen == "class" else (lambda i: get_domain(full, i))
    targets = getattr(args, "forget_classes", None) or getattr(args, "forget_domains", None)
    f_idx = [i for i in train_s.indices if key(i) in targets]
    r_idx = [i for i in train_s.indices if key(i) not in targets]
    rt_idx = [i for i in test_s.indices if key(i) not in targets]
    mia_idx = ([i for i in test_s.indices if key(i) in targets]
               + [i for i in unseen_s.indices if key(i) in targets])

    L = lambda idx, t: DataLoader(ApplyTransform(Subset(full, idx), t), batch_size=bs,
                                  shuffle=False, num_workers=4)
    forget_l = L(f_idx, get_forget_test_transform())
    retain_l = L(r_idx, get_retain_test_transform())
    test_l = L(rt_idx, get_test_transform())
    mia_l = L(mia_idx, get_test_transform())
    print(f"[*] {scen}: forget={len(f_idx)} retain={len(r_idx)} test={len(rt_idx)} mia_pool={len(mia_idx)}")

    model = ModuleArchitecture(model_name=args.model_name, num_classes=len(full.class_names), pretrained=False,
                               moe_layers=getattr(args, "moe_layers", None),
                               num_experts=args.num_experts, expert_depth=args.expert_depth,
                               expert_hidden_ratio=args.expert_hidden_ratio,
                               gate_k=args.gate_k, device=dev,
                               mlp_ratio=getattr(args, 'mlp_ratio', 4.0), gate_norm=getattr(args, 'gate_norm', 'softmax'))
    ckpt = a.checkpoint or args.pretrained_model_path
    print(f"[*] checkpoint: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model.eval()

    fa = forget_acc(model, forget_l, dev)
    ra = retain_acc(model, retain_l, dev)
    ta = test_acc(model, test_l, dev)
    m = mia(model, forget_l, mia_l, dev)
    print(f"--> Metrics: RA: {ra*100:.2f}% | FA: {fa*100:.2f}% | TA: {ta*100:.2f}% | MIA: {m:.4f}")

    if not a.no_wandb:
        stem = f"{a.baseline.lower()}_{ds_name}_{scen}"
        suffix = getattr(args, "study_name", "").split("__seed")[-1] if getattr(args, "study_name", "") else str(args.seed)
        name = f"{a.baseline}__{ds_label}__{scen}__{stem}__seed{suffix}"
        wandb.init(project="MoE", name=name, group=getattr(args, "wandb_group", f"MainTable_{ds_label}_seed42"),
                   tags=["main_table", ds_label, scen, a.baseline, f"seed{args.seed}"],
                   config={**cfg, "baseline_name": a.baseline, "scenario": scen,
                           "config_name": stem, "config_path": a.config,
                           "dataset_name": ds_label, "pretrained_model_path": ckpt})
        wandb.log({"fa": fa, "ra": ra, "ta": ta, "mia": m, "epoch": 0})
        wandb.finish()
        print(f"[*] W&B: {name}")

    out = Path("results/main_table_pacs_seed42/original_results.csv")
    new = not out.exists()
    with out.open("a", newline="") as fh:
        w = csv.writer(fh)
        if new: w.writerow(["baseline", "scenario", "fa", "ra", "ta", "mia", "checkpoint"])
        w.writerow([a.baseline, scen, round(fa*100, 2), round(ra*100, 2), round(ta*100, 2),
                    round(m, 4), ckpt])
    print(f"[*] -> {out}")


if __name__ == "__main__":
    main()
