#!/usr/bin/env python3
"""Collect results for the forget-set-size ablation.

Everything is parsed from the run logs and configs written by
scripts/run_abla_forget_size.sh -- no metric value is entered by hand.

Reports the final unlearning epoch (epoch 20); no best-epoch or threshold
selection is applied. Routing Overlap is the RFO (retain-forget routing overlap)
from the Eq.7 localization diagnostics. |Mf| is the number of experts selected
for update per MoE layer, i.e. k_u, taken from the run's own log line.
"""
import csv
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_ROOT / "results" / "logs"
CONFIG_DIR = REPO_ROOT / "config" / "experiments"
OUT_CSV = REPO_ROOT / "results" / "abla_forget_size.csv"

PCTS = [1, 5, 10, 20, 40]

EPOCH_RE = re.compile(r"Epoch \[(\d+)/(\d+)\]")
METRICS_RE = re.compile(
    r"RA:\s*([\d.]+)%\s*\|\s*FA:\s*([\d.]+)%\s*\|\s*TA:\s*([\d.]+)%\s*\|\s*MIA:\s*([\d.]+)"
)
TIME_RE = re.compile(r"Total [Tt]ime:\s*([\d.]+)s")
RFO_RE = re.compile(r"Retain-forget routing overlap \(RFO\):\s*([\d.]+)")
SPLIT_RE = re.compile(r"split sizes -> retain:\s*(\d+)\s*\|\s*forget:\s*(\d+)")
MF_RE = re.compile(r"Using alpha=[\d.]+, k_u=(\d+)")
WANDB_ID_RE = re.compile(r"run-\d{8}_\d{6}-(\w+)")


def read(path):
    return path.read_text() if path.exists() else ""


def wandb_id(name):
    m = WANDB_ID_RE.search(read(LOG_DIR / f"{name}.wandb_dir").strip())
    return m.group(1) if m else ""


def fmt_runtime(sec):
    if sec in ("", None):
        return "-"
    sec = int(float(sec))
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


def main():
    rows = []
    for p in PCTS:
        name = f"abla_unlearn_forget_size_{p}pct_M12_ku4_k4_seed42"
        text = read(LOG_DIR / f"{name}.log")
        cfg_path = CONFIG_DIR / f"{name}.yaml"
        cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}

        epochs = EPOCH_RE.findall(text)
        metrics = METRICS_RE.findall(text)
        times = TIME_RE.findall(text)
        rfos = RFO_RE.findall(text)
        split = SPLIT_RE.search(text)
        mf = MF_RE.search(text)

        ok = bool(epochs and metrics)
        ra, fa, ta, mia = metrics[-1] if ok else ("", "", "", "")
        n_forget = int(split.group(2)) if split else ""
        n_retain = int(split.group(1)) if split else ""
        n_train = (n_forget + n_retain) if split else ""
        rows.append({
            "forget_size_pct_requested": p,
            "forget_samples_actual": n_forget,
            "train_samples": n_train,
            "forget_pct_actual": round(100.0 * n_forget / n_train, 2) if split else "",
            "mf_selected_experts_per_layer": int(mf.group(1)) if mf else cfg.get("ku", ""),
            "run_name": name,
            "fa": float(fa) / 100.0 if ok else "",
            "ra": float(ra) / 100.0 if ok else "",
            "ta": float(ta) / 100.0 if ok else "",
            "mia": float(mia) if ok else "",
            "routing_overlap_rfo": float(rfos[-1]) if rfos else "",
            "final_epoch": int(epochs[-1][0]) if ok else "",
            "total_epochs": int(epochs[-1][1]) if ok else "",
            "runtime_sec": float(times[-1]) if times else "",
            "wandb_run_id": wandb_id(name),
            # the repo builds forget sets on the fly from the seeded split; there is
            # no saved index file. sets are nested across percentages.
            "forget_index_file": "(none - generated in-process by unlearn.py random_split, nested)",
            "checkpoint": cfg.get("checkpoint", ""),
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\nABLATION — FORGET-SET SIZE")
    print("(PACS, M=12, learn_k=4, ku=4, seed=42, nested random forget sets, final epoch)")
    print()
    print(f"{'Forget Size':<12}|{'|Mf|':>5} |{'FA':>9} |{'RA':>9} |{'TA':>9} |"
          f"{'Routing Overlap':>16} |{'Runtime':>9} | W&B Run")
    print("-" * 100)
    for r in rows:
        label = f"{r['forget_size_pct_requested']}%"
        if r["fa"] == "":
            print(f"{label:<12}|  MISSING (run not completed)")
            continue
        rfo = f"{r['routing_overlap_rfo']:16.4f}" if r["routing_overlap_rfo"] != "" else f"{'-':>16}"
        print(f"{label:<12}|{r['mf_selected_experts_per_layer']:>5} |{r['fa']*100:8.2f}% |"
              f"{r['ra']*100:8.2f}% |{r['ta']*100:8.2f}% |{rfo} |"
              f"{fmt_runtime(r['runtime_sec']):>9} | {r['wandb_run_id'] or '-'}")
    print("\nactual forget sample counts:")
    for r in rows:
        if r["forget_samples_actual"] != "":
            print(f"  {r['forget_size_pct_requested']:>3}% -> {r['forget_samples_actual']:>5} / "
                  f"{r['train_samples']} train samples ({r['forget_pct_actual']}%)")
    print(f"\n[*] results written to {OUT_CSV}")


if __name__ == "__main__":
    main()
