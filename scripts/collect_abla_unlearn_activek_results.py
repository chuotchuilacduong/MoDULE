#!/usr/bin/env python3
"""Collect results for the class/domain active-routing-k unlearning ablation from
the actual run logs and configs produced by
scripts/run_abla_unlearn_activek_class_domain.sh.

Does not accept or fabricate metric values -- everything is parsed from the
per-run log file (results/logs/<run_name>.log) written by the launcher, and from
the run's own yaml config. Reports the FINAL epoch (epoch 20); no best-epoch or
threshold-based selection is applied anywhere.
"""
import csv
import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_ROOT / "results" / "logs"
CONFIG_DIR = REPO_ROOT / "config" / "experiments"
OUT_CSV = REPO_ROOT / "results" / "abla_unlearn_activek_class_domain.csv"

ACTIVE_KS = [2, 4, 6, 8, 10, 12]


def run_name(scenario, k):
    if k == 12:
        return f"abla_unlearn_{scenario}_activek_12_dense_ku12_M12_learnk4_seed42"
    return f"abla_unlearn_{scenario}_activek_{k}_ku12_M12_learnk4_seed42"


RUNS = [(s, k, run_name(s, k)) for s in ("class", "domain") for k in ACTIVE_KS]

EPOCH_RE = re.compile(r"Epoch \[(\d+)/(\d+)\]")
METRICS_RE = re.compile(
    r"RA:\s*([\d.]+)%\s*\|\s*FA:\s*([\d.]+)%\s*\|\s*TA:\s*([\d.]+)%\s*\|\s*MIA:\s*([\d.]+)"
)
TIME_RE = re.compile(r"Total [Tt]ime:\s*([\d.]+)s")
WANDB_ID_RE = re.compile(r"run-\d{8}_\d{6}-(\w+)")


def parse_log(log_path):
    if not log_path.exists():
        return None
    text = log_path.read_text()
    epochs = EPOCH_RE.findall(text)
    metrics = METRICS_RE.findall(text)
    times = TIME_RE.findall(text)
    if not epochs or not metrics:
        return None
    final_epoch, total_epochs = epochs[-1]
    ra, fa, ta, mia = metrics[-1]
    return {
        "final_epoch": int(final_epoch),
        "total_epochs": int(total_epochs),
        "fa": float(fa) / 100.0,
        "ra": float(ra) / 100.0,
        "ta": float(ta) / 100.0,
        "mia": float(mia),
        "runtime_sec": float(times[-1]) if times else "",
    }


def wandb_id(name):
    p = LOG_DIR / f"{name}.wandb_dir"
    if not p.exists():
        return ""
    m = WANDB_ID_RE.search(p.read_text().strip())
    return m.group(1) if m else ""


def fmt_runtime(sec):
    if sec == "" or sec is None:
        return "-"
    sec = int(float(sec))
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"


def main():
    rows = []
    for scenario, k, name in RUNS:
        cfg_path = CONFIG_DIR / f"{name}.yaml"
        cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
        parsed = parse_log(LOG_DIR / f"{name}.log")
        target = cfg.get("target", "")
        ckpt_dir = cfg.get("output_dir", "")
        rows.append({
            "scenario": scenario,
            "unlearn_k": k,
            "dense_routing": k == 12,
            "run_name": name,
            "M": cfg.get("M", ""),
            "learn_k": cfg.get("learn_k", ""),
            "ku": cfg.get("ku", ""),
            "seed": cfg.get("seed", ""),
            "target": target,
            "fa": parsed["fa"] if parsed else "",
            "ra": parsed["ra"] if parsed else "",
            "ta": parsed["ta"] if parsed else "",
            "mia": parsed["mia"] if parsed else "",
            "final_epoch": parsed["final_epoch"] if parsed else "",
            "total_epochs": parsed["total_epochs"] if parsed else "",
            "runtime_sec": parsed["runtime_sec"] if parsed else "",
            "wandb_run_id": wandb_id(name),
            "base_checkpoint": cfg.get("checkpoint", ""),
            "final_checkpoint": f"{ckpt_dir}/unlearned_module_{name}.pt" if ckpt_dir else "",
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    for scenario, title in (("class", "CLASS"), ("domain", "DOMAIN")):
        sub = [r for r in rows if r["scenario"] == scenario]
        print(f"\nABLATION — UNLEARN {title} — ACTIVE TOP-K")
        if sub and sub[0]["target"]:
            print(f"(forgotten target: {sub[0]['target']}, ku=12, M=12, learn_k=4, seed=42, final epoch)")
        print()
        print(f"{'Active k':<12}|{'FA':>9} |{'RA':>9} |{'TA':>9} |{'MIA':>8} |{'Runtime':>9} | W&B Run")
        print("-" * 84)
        for r in sub:
            label = "12 (Dense)" if r["unlearn_k"] == 12 else str(r["unlearn_k"])
            if r["fa"] == "":
                print(f"{label:<12}|{'  MISSING (run not completed)':<50}")
                continue
            print(f"{label:<12}|{r['fa']*100:8.2f}% |{r['ra']*100:8.2f}% |{r['ta']*100:8.2f}% |"
                  f"{r['mia']:8.4f} |{fmt_runtime(r['runtime_sec']):>9} | {r['wandb_run_id'] or '-'}")

    print(f"\n[*] results written to {OUT_CSV}")


if __name__ == "__main__":
    main()
