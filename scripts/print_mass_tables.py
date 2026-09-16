"""In bảng domain x expert và class x expert (routing mass) của một checkpoint, trên TRAIN và TEST,
để nhìn trực tiếp expert có chuyên biệt hoá theo domain / lớp hay không.

    python scripts/print_mass_tables.py --checkpoint runs/_base_models/<x>/checkpoints/<y>.pt \
        [--config <learn.yaml>] [--mass gated|pi] [--split train,test]

Mỗi ô = tỉ lệ mass của nhóm (hàng) rơi vào expert (cột); hàng tổng = 1. Đều = 1/M.
Kèm: top-k expert của mỗi nhóm, tỉ lệ max/uniform, số nhóm dùng chung expert (overlap),
I(G;E)/min(log G, log M) (1 = mỗi nhóm có expert riêng, 0 = routing độc lập với nhóm).
"""
import argparse
import math
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader, Subset, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from architecture.module import ModuleArchitecture, DeepMoELayer
from dataset.pytorch_dataset.pacs import PACSDataset
from dataset.pytorch_dataset.officehome import OfficeHomeDataset
from dataset.transform.test_transform import get_test_transform
from module_diagnostics import ApplyTransform, set_seed
from metric.route_separation import mutual_information


def load_config(ckpt, cfg_path):
    if cfg_path is None:
        cand = os.path.join(os.path.dirname(os.path.dirname(ckpt)), "learn.yaml")
        if not os.path.exists(cand):
            raise FileNotFoundError(f"không thấy learn.yaml cạnh checkpoint, truyền --config: {cand}")
        cfg_path = cand
    d = yaml.safe_load(open(cfg_path))
    return d.get("learn", d)


def build_loaders(cfg, batch_size):
    set_seed(cfg["seed"])
    ds_name = str(cfg["dataset"]).lower()
    if ds_name == "pacs":
        full = PACSDataset(root_dir=cfg["data_dir"], transform=None)
    elif ds_name == "officehome":
        full = OfficeHomeDataset(root_dir=cfg["data_dir"], transform=None)
    else:
        raise ValueError("chỉ hỗ trợ dataset có domain (pacs / officehome)")
    n = len(full)
    n_tr, n_te = int(0.8 * n), int(0.1 * n)
    g = torch.Generator().manual_seed(cfg["seed"])
    tr, te, _ = random_split(full, [n_tr, n_te, n - n_tr - n_te], generator=g)
    tf = get_test_transform()
    mk = lambda sub: DataLoader(ApplyTransform(sub, tf), batch_size=batch_size, shuffle=False, num_workers=4)
    return full, {"train": mk(tr), "test": mk(te)}


@torch.no_grad()
def mass_on(model, loader, device, num_classes, num_domains, use_gated):
    model.eval()
    moe = [(n, m) for n, m in model.named_modules() if isinstance(m, DeepMoELayer)]
    cm = {n: torch.zeros(num_classes, m.num_experts) for n, m in moe}
    dm = {n: torch.zeros(num_domains, m.num_experts) for n, m in moe}
    for batch in loader:
        x = batch[0].to(device); y = batch[1].long(); d = batch[2].long()
        model.inference(x)
        for n, m in moe:
            mass = (m.last_gate_mass if use_gated else m.last_pi_all).detach().float().cpu()
            S = mass.size(0) // y.size(0)
            cm[n].index_add_(0, y.repeat_interleave(S), mass)
            dm[n].index_add_(0, d.repeat_interleave(S), mass)
    norm = lambda t: t / t.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return {n: norm(cm[n]) for n in cm}, {n: norm(dm[n]) for n in dm}


def print_table(W, names, k, title):
    G, M = W.shape
    print(f"\n  {title}  (hàng tổng = 1, đều = {1/M:.3f})")
    print("  " + f"{'':14s}" + "".join(f"{'e'+str(e):>7s}" for e in range(M)) + f"   top-{k}   max/unif")
    tops = []
    for g in range(G):
        row = W[g]
        top = row.topk(min(k, M)).indices.tolist(); tops.append(set(top))
        cells = "".join(f"{v:7.3f}" for v in row.tolist())
        print(f"  {names[g][:14]:14s}{cells}   {str(top):8s} {row.max().item()*M:6.2f}x")
    # expert usage: which groups have expert e in their top-k
    print("  " + f"{'#groups top-k':14s}" + "".join(f"{sum(e in t for t in tops):7d}" for e in range(M)))
    jacc = []
    for i in range(G):
        for j in range(i + 1, G):
            jacc.append(len(tops[i] & tops[j]) / len(tops[i] | tops[j]))
    pg = torch.full((G,), 1.0 / G)
    mi = mutual_information(W, pg).item() / min(math.log(G), math.log(M))
    col = W.mean(dim=0)
    print(f"  mean pairwise top-{k} Jaccard = {sum(jacc)/len(jacc):.3f} | I(G;E) norm = {mi:.3f}"
          f" | tải theo expert: max/min = {col.max().item()/max(col.min().item(),1e-8):.1f}x")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--mass", choices=["gated", "pi"], default="gated",
                    help="gated = mass sau top-k (Eq. 29, cái L_bal/L_sep tối ưu); pi = softmax đầy đủ")
    ap.add_argument("--split", default="train,test")
    ap.add_argument("--batch-size", type=int, default=128)
    a = ap.parse_args()
    cfg = load_config(a.checkpoint, a.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full, loaders = build_loaders(cfg, a.batch_size)
    C, D = len(full.class_names), len(full.domain_names)
    model = ModuleArchitecture(
        model_name=cfg["model_name"], num_classes=C, pretrained=False, device=device,
        moe_layers=cfg.get("moe_layers"), num_experts=cfg["num_experts"], expert_depth=cfg["expert_depth"],
        expert_hidden_ratio=cfg["expert_hidden_ratio"], gate_k=cfg["gate_k"],
        mlp_ratio=cfg.get("mlp_ratio", 4.0), gate_norm=cfg.get("gate_norm", "softmax"))
    sd = torch.load(a.checkpoint, map_location=device)
    sd = sd.get("model_state_dict", sd.get("state_dict", sd))
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()}, strict=False)
    model.to(device)
    k = int(cfg["gate_k"])
    print(f"checkpoint: {a.checkpoint}\nmass: {a.mass} | gate_k={k} | M={cfg['num_experts']} | lambda_sep={cfg.get('lambda_sep', 0.0)}")
    for split in a.split.split(","):
        cm, dm = mass_on(model, loaders[split], device, C, D, a.mass == "gated")
        for layer in cm:
            short = layer.replace("featurizer.model.", "")
            print(f"\n=== {split.upper()} | {short} ===")
            print_table(dm[layer], full.domain_names, k, "DOMAIN x expert")
            print_table(cm[layer], full.class_names, k, "CLASS x expert")


if __name__ == "__main__":
    main()
