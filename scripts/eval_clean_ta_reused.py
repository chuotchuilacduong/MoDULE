#!/usr/bin/env python3
"""Measure Clean TA for the base checkpoints REUSED by the number-of-experts
ablation (M=8 and M=12), which have no learning log in this ablation.

Clean TA = test accuracy of the base model before any unlearning, on the same
seeded test split unlearn.py builds (unlearn_setting: random -> the full 10%
test split). Each checkpoint is evaluated at the gate_k it was trained with.

Writes results/clean_ta_reused.json, which scripts/collect_abla_num_experts.py
reads. Values are measured here, never entered by hand. Runs on CPU by default
so it does not contend with training on the GPU.
"""
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset.pytorch_dataset.pacs import PACSDataset  # noqa: E402
from dataset.transform.test_transform import get_test_transform  # noqa: E402
from architecture.module import ModuleArchitecture  # noqa: E402

DEVICE = "cuda" if (len(sys.argv) > 1 and sys.argv[1] == "--cuda" and torch.cuda.is_available()) else "cpu"

# (M, gate_k, checkpoint) for the reused base models
REUSED = [
    (8, 2, REPO_ROOT / "runs/_base_models/3ae530097a/checkpoints/learn_best.pt"),
    (12, 4, REPO_ROOT / "runs/_base_models/pacs_M12_k4_seed42/checkpoints/pacs_module_base_M12_k4_best.pt"),
]


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


def main():
    full = PACSDataset(root_dir=str(REPO_ROOT / "dataset/data_folder/pacs"), transform=None)
    n = len(full)
    tr, te = int(0.8 * n), int(0.1 * n)
    g = torch.Generator().manual_seed(42)
    _, test_subset, _ = random_split(full, [tr, te, n - tr - te], generator=g)
    loader = DataLoader(ApplyTransform(test_subset, get_test_transform()),
                        batch_size=32, shuffle=False, num_workers=4)
    print(f"[*] device={DEVICE} | test split: {len(test_subset)} images")

    out = {}
    for M, gate_k, ckpt in REUSED:
        if not ckpt.exists():
            print(f"[!] missing checkpoint for M={M}: {ckpt}")
            continue
        model = ModuleArchitecture(
            model_name="module_small_patch16_224", num_classes=7, pretrained=False,
            moe_layers="FFFFFFFFFFSS", num_experts=M, expert_depth=2,
            expert_hidden_ratio=2, gate_k=gate_k, device=DEVICE,
        )
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in loader:
                logits, _ = model.inference(batch[0].to(DEVICE))
                correct += (logits.argmax(1).cpu() == batch[1]).sum().item()
                total += batch[1].size(0)
        acc = correct / total
        out[str(M)] = {"clean_ta": acc, "gate_k": gate_k, "checkpoint": str(ckpt.relative_to(REPO_ROOT)),
                       "correct": correct, "total": total}
        print(f"M={M} (gate_k={gate_k}): Clean TA = {acc*100:.2f}%  ({correct}/{total})", flush=True)

    dest = REPO_ROOT / "results" / "clean_ta_reused.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"[*] written to {dest}")


if __name__ == "__main__":
    main()
