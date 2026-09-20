"""Tải checkpoint đã được learn.py đẩy lên W&B (wandb.save) về máy khác.
    python scripts/fetch_wandb_ckpt.py --run g0inran5 --file checkpoints/learn_best.pt \
        --out runs/_base_models/aaedd1da4d/checkpoints/learn_best.pt
"""
import argparse, os, shutil, wandb
ap=argparse.ArgumentParser(); ap.add_argument("--run",required=True); ap.add_argument("--file",required=True); ap.add_argument("--out",required=True)
ap.add_argument("--project",default="chuotchuilacduong-hanoi-university-of-science-and-technology/MoE")
a=ap.parse_args()
r=wandb.Api().run(f"{a.project}/{a.run}")
tmp=os.path.join(os.path.dirname(a.out) or ".", "_wandb_dl"); os.makedirs(tmp,exist_ok=True)
p=r.file(a.file).download(root=tmp, replace=True); os.makedirs(os.path.dirname(a.out) or ".",exist_ok=True)
shutil.move(p.name, a.out); shutil.rmtree(tmp, ignore_errors=True)
print("saved", a.out, os.path.getsize(a.out)//1_000_000, "MB")
