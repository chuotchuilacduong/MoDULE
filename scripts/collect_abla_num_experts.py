#!/usr/bin/env python3
"""Collect results for the number-of-experts ablation.

Everything is parsed from the run logs and configs written by
scripts/run_abla_num_experts.sh -- no metric value is entered by hand.

Clean TA comes from the corresponding LEARNING run's [Final Metrics] line (the
base model before unlearning). FA/RA/TA/MIA come from the final unlearning epoch
(epoch 20); no best-epoch or threshold selection is applied. Routing Overlap is
the RFO (retain-forget routing overlap) from the Eq.7 localization diagnostics,
enabled via run_eq7_diagnostics in each unlearning config.
"""
import csv
import json
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_ROOT / "results" / "logs"
CONFIG_DIR = REPO_ROOT / "config" / "experiments"
OUT_CSV = REPO_ROOT / "results" / "abla_num_experts.csv"

MS = [4, 8, 12, 16, 24]

# M=8 and M=12 reuse pre-existing base checkpoints and so have no learning log in
# this ablation; their Clean TA is measured by scripts/eval_clean_ta_reused.py.
_REUSED_JSON = REPO_ROOT / "results" / "clean_ta_reused.json"
REUSED_CLEAN_TA = json.loads(_REUSED_JSON.read_text()) if _REUSED_JSON.exists() else {}

EPOCH_RE = re.compile(r"Epoch \[(\d+)/(\d+)\]")
METRICS_RE = re.compile(
    r"RA:\s*([\d.]+)%\s*\|\s*FA:\s*([\d.]+)%\s*\|\s*TA:\s*([\d.]+)%\s*\|\s*MIA:\s*([\d.]+)"
)
TIME_RE = re.compile(r"Total [Tt]ime:\s*([\d.]+)s")
RFO_RE = re.compile(r"Retain-forget routing overlap \(RFO\):\s*([\d.]+)")
CLEAN_TA_RE = re.compile(r"\[Final Metrics\].*?ta:\s*([\d.]+)%", re.S)
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
    for M in MS:
        un = f"abla_unlearn_num_experts_M{M}_ku4_k4_seed42"
        ln = f"abla_learn_num_experts_M{M}_k4_seed42"
        utext = read(LOG_DIR / f"{un}.log")
        ltext = read(LOG_DIR / f"{ln}.log")
        cfg_path = CONFIG_DIR / f"{un}.yaml"
        cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}

        epochs = EPOCH_RE.findall(utext)
        metrics = METRICS_RE.findall(utext)
        times = TIME_RE.findall(utext)
        rfos = RFO_RE.findall(utext)
        clean = CLEAN_TA_RE.search(ltext)
        clean_ta = float(clean.group(1)) / 100.0 if clean else REUSED_CLEAN_TA.get(str(M), {}).get("clean_ta", "")

        ok = bool(epochs and metrics)
        ra, fa, ta, mia = metrics[-1] if ok else ("", "", "", "")
        rows.append({
            "M": M,
            "run_name": un,
            "learn_run_name": ln,
            "clean_ta": clean_ta,
            "fa": float(fa) / 100.0 if ok else "",
            "ra": float(ra) / 100.0 if ok else "",
            "ta": float(ta) / 100.0 if ok else "",
            "mia": float(mia) if ok else "",
            # RFO after unlearning (the UNLEARN-phase diagnostic is logged last)
            "routing_overlap_rfo": float(rfos[-1]) if rfos else "",
            "final_epoch": int(epochs[-1][0]) if ok else "",
            "total_epochs": int(epochs[-1][1]) if ok else "",
            "runtime_sec": float(times[-1]) if times else "",
            "wandb_run_id": wandb_id(un),
            "learn_wandb_run_id": wandb_id(ln),
            "ku": cfg.get("ku", ""),
            "learn_k": cfg.get("learn_k", ""),
            "seed": cfg.get("seed", ""),
            "unlearn_setting": cfg.get("unlearn_setting", ""),
            "forget_ratio": cfg.get("forget_ratio", ""),
            "checkpoint": cfg.get("checkpoint", ""),
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\nABLATION — NUMBER OF EXPERTS")
    print("(PACS, learn_k=4, ku=4, seed=42, random forget 10% of train, final epoch)")
    print()
    print(f"{'M':<5}|{'Clean TA':>10} |{'FA':>9} |{'RA':>9} |{'MIA':>8} |"
          f"{'Routing Overlap':>16} |{'Runtime':>9} | W&B Run")
    print("-" * 100)
    for r in rows:
        if r["fa"] == "":
            print(f"{r['M']:<5}|  MISSING (run not completed)")
            continue
        cta = f"{r['clean_ta']*100:9.2f}%" if r["clean_ta"] != "" else f"{'-':>10}"
        rfo = f"{r['routing_overlap_rfo']:16.4f}" if r["routing_overlap_rfo"] != "" else f"{'-':>16}"
        print(f"{r['M']:<5}|{cta} |{r['fa']*100:8.2f}% |{r['ra']*100:8.2f}% |{r['mia']:8.4f} |"
              f"{rfo} |{fmt_runtime(r['runtime_sec']):>9} | {r['wandb_run_id'] or '-'}")
    print(f"\n[*] results written to {OUT_CSV}")


if __name__ == "__main__":
    main()
