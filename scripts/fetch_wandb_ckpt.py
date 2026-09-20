"""Tải checkpoint đã được learn.py đẩy lên W&B (wandb.save) về máy khác.
    python scripts/fetch_wandb_ckpt.py --run g0inran5 --file checkpoints/learn_best.pt \
        --out runs/_base_models/aaedd1da4d/checkpoints/learn_best.pt
Thử File.download() trước; nếu service W&B báo bận thì tải thẳng qua HTTP bằng API key trong ~/.netrc.
"""
import argparse, os, shutil, netrc, requests, wandb
ap=argparse.ArgumentParser(); ap.add_argument("--run",required=True); ap.add_argument("--file",required=True); ap.add_argument("--out",required=True)
ap.add_argument("--project",default="chuotchuilacduong-hanoi-university-of-science-and-technology/MoE")
a=ap.parse_args()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
r=wandb.Api(timeout=120).run(f"{a.project}/{a.run}")
f=r.file(a.file)
try:
    tmp=os.path.join(os.path.dirname(a.out) or ".", "_wandb_dl"); os.makedirs(tmp,exist_ok=True)
    p=f.download(root=tmp, replace=True); shutil.move(p.name, a.out); shutil.rmtree(tmp, ignore_errors=True)
except Exception as e:
    print(f"[!] File.download() lỗi ({type(e).__name__}); tải qua HTTP...")
    key=netrc.netrc().authenticators("api.wandb.ai")[2]
    with requests.get(f.url, auth=("api", key), stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with open(a.out,"wb") as fh:
            for chunk in resp.iter_content(1<<20): fh.write(chunk)
print("saved", a.out, os.path.getsize(a.out)//1_000_000, "MB")
