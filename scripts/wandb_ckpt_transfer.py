#!/usr/bin/env python3
"""Move checkpoints between machines through a W&B artifact.

The machines here cannot ssh/scp to each other, so W&B is the transport: this
machine pushes the checkpoints it already trained, the server pulls them and skips
the corresponding training. Files keep their repo-relative path inside the
artifact, so the pull writes them exactly where the scripts expect them.

  # on this machine (checkpoints already trained):
  python scripts/wandb_ckpt_transfer.py push
  # on the server:
  python scripts/wandb_ckpt_transfer.py pull

Default payload = everything the joint class-unlearning MIA experiment needs and
that is expensive to recompute: the three Retrain references (50 epochs each) and
the 1-class ModULE/SEUF runs. `--with-base` also ships the original M=12/k=4 base
checkpoint (the server normally already has it).
"""
import argparse
import os
import sys
from pathlib import Path

import wandb

REPO = Path(__file__).resolve().parents[1]
PROJECT = "chuotchuilacduong-hanoi-university-of-science-and-technology/MoE"
ARTIFACT = "joint_class_pacs_m12k4_seed42"

PAYLOAD = [
    "runs/sequential/pacs_retrain_class/retrain_class_0/retrain_class_0.pt",
    "runs/sequential/pacs_retrain_class/retrain_class_0-1/retrain_class_0-1.pt",
    "runs/sequential/pacs_retrain_class/retrain_class_0-1-2/retrain_class_0-1-2.pt",
    "runs/joint_class/module_joint_c1/checkpoints/unlearned_module_module_joint_c1.pt",
    "runs/joint_class/seuf_joint_c1/checkpoints/unlearned_seuf_seuf_joint_c1.pt",
]
# the 1-class joint runs are identical to the sequential stage-1 runs (verified);
# on this machine only the sequential copies exist, so map them onto the joint paths
ALIASES = {
    "runs/joint_class/module_joint_c1/checkpoints/unlearned_module_module_joint_c1.pt":
        "runs/sequential/pacs_module_seq_class/stage_1/checkpoints/unlearned_module_stage_1.pt",
    "runs/joint_class/seuf_joint_c1/checkpoints/unlearned_seuf_seuf_joint_c1.pt":
        "runs/sequential/pacs_seuf_seq_class/stage_1/checkpoints/unlearned_seuf_stage_1.pt",
}
BASE = "runs/_base_models/pacs_M12_k4_seed42/checkpoints/pacs_module_base_M12_k4_best.pt"


def push(args):
    files = list(PAYLOAD) + ([BASE] if args.with_base else [])
    plan, missing = [], []
    for rel in files:
        src = REPO / rel
        if not src.exists() and rel in ALIASES:
            src = REPO / ALIASES[rel]
        (plan if src.exists() else missing).append((rel, src))
    if missing:
        print("[!] missing locally, not uploaded:")
        for rel, src in missing:
            print("   ", rel)
    if not plan:
        sys.exit("[!] nothing to upload")
    total = sum(s.stat().st_size for _, s in plan) / 1e6
    print(f"[*] uploading {len(plan)} file(s), {total:.0f} MB, as artifact '{ARTIFACT}'")
    run = wandb.init(project=PROJECT.split("/")[-1], entity=PROJECT.split("/")[0],
                     job_type="ckpt-transfer", name=f"push_{ARTIFACT}",
                     tags=["joint_class", "ckpt-transfer"])
    art = wandb.Artifact(ARTIFACT, type="model",
                         description="PACS M=12/k=4 seed 42: Retrain references for forget sets "
                                     "{0},{0,1},{0,1,2} (50 ep, ImageNet init) and the 1-class joint "
                                     "ModULE/SEUF unlearned models (20 ep, no early stop).")
    for rel, src in plan:
        art.add_file(str(src), name=rel)   # name = repo-relative destination path
        print(f"    + {rel}  ({src.stat().st_size/1e6:.0f} MB, from {src.relative_to(REPO)})")
    run.log_artifact(art)
    art.wait()
    print(f"[*] done. On the server: python scripts/wandb_ckpt_transfer.py pull")
    run.finish()


def pull(args):
    api = wandb.Api(timeout=300)
    art = api.artifact(f"{PROJECT}/{ARTIFACT}:{args.version}", type="model")
    tmp = REPO / ".wandb_ckpt_dl"
    print(f"[*] downloading {ARTIFACT}:{args.version} ...")
    root = Path(art.download(root=str(tmp)))
    n = 0
    for f in sorted(root.rglob("*.pt")):
        rel = f.relative_to(root)
        dst = REPO / rel
        if dst.exists() and not args.force:
            print(f"    = keep existing {rel}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.replace(f, dst)
        print(f"    -> {rel}  ({dst.stat().st_size/1e6:.0f} MB)")
        n += 1
    print(f"[*] placed {n} checkpoint(s); leftover download dir: {tmp} (safe to delete)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push"); p.add_argument("--with-base", action="store_true")
    g = sub.add_parser("pull"); g.add_argument("--version", default="latest"); g.add_argument("--force", action="store_true")
    a = ap.parse_args()
    (push if a.cmd == "push" else pull)(a)
