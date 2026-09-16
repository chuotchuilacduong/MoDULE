#!/usr/bin/env python3
"""Collect results for the class/domain k_u unlearning ablation from the actual
run logs and configs produced by scripts/run_abla_unlearn_class_domain_ku.sh.

Does not accept or fabricate metric values -- everything is parsed from the
per-run log file (results/logs/<run_name>.log) written by the launcher, and
from the run's own yaml config.
"""
import csv
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_ROOT / "results" / "logs"
CONFIG_DIR = REPO_ROOT / "config" / "experiments"
OUT_CSV = REPO_ROOT / "results" / "abla_unlearn_class_domain_ku.csv"

KUS = [2, 4, 6, 8]

RUNS = (
    [("class", ku, f"abla_unlearn_class_ku_{ku}_M12_k4_seed42") for ku in KUS]
    + [("domain", ku, f"abla_unlearn_domain_ku_{ku}_M12_k4_seed42") for ku in KUS]
)

EPOCH_RE = re.compile(r"Epoch \[(\d+)/(\d+)\]")
METRICS_RE = re.compile(
    r"RA:\s*([\d.]+)%\s*\|\s*FA:\s*([\d.]+)%\s*\|\s*TA:\s*([\d.]+)%\s*\|\s*MIA:\s*([\d.]+)"
)
TIME_RE = re.compile(r"Total time:\s*([\d.]+)s")
WANDB_ID_RE = re.compile(r"run-\d{8}_\d{6}-(\w+)")


def load_yaml_simple(path):
    """Tiny flat-yaml reader (avoids a hard dependency on PyYAML for this script)."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def parse_log(log_path):
    if not log_path.exists():
        return None
    text = log_path.read_text()
    epoch_matches = EPOCH_RE.findall(text)
    metrics_matches = METRICS_RE.findall(text)
    time_matches = TIME_RE.findall(text)
    if not epoch_matches or not metrics_matches:
        return None
    final_epoch, total_epochs = epoch_matches[-1]
    ra, fa, ta, mia = metrics_matches[-1]
    runtime = time_matches[-1] if time_matches else ""
    return {
        "final_epoch": int(final_epoch),
        "total_epochs": int(total_epochs),
        "fa": float(fa) / 100.0,
        "ra": float(ra) / 100.0,
        "ta": float(ta) / 100.0,
        "mia": float(mia),
        "runtime_sec": float(runtime) if runtime else "",
    }


def parse_wandb_id(run_name):
    wandb_dir_file = LOG_DIR / f"{run_name}.wandb_dir"
    if not wandb_dir_file.exists():
        return ""
    dir_line = wandb_dir_file.read_text().strip()
    m = WANDB_ID_RE.search(dir_line)
    return m.group(1) if m else ""


def main():
    rows = []
    missing = []
    for scenario, ku, run_name in RUNS:
        config_path = CONFIG_DIR / f"{run_name}.yaml"
        cfg = load_yaml_simple(config_path)
        output_dir = Path(cfg["output_dir"])
        output_checkpoint = output_dir / f"unlearned_{cfg['unlearn_algo']}_{run_name}.pt"
        target = cfg.get("target", "")

        log_path = LOG_DIR / f"{run_name}.log"
        parsed = parse_log(log_path)
        wandb_id = parse_wandb_id(run_name)

        row = {
            "scenario": scenario,
            "ku": ku,
            "run_name": run_name,
            "fa": parsed["fa"] if parsed else "",
            "ra": parsed["ra"] if parsed else "",
            "ta": parsed["ta"] if parsed else "",
            "mia": parsed["mia"] if parsed else "",
            "runtime_sec": parsed["runtime_sec"] if parsed else "",
            "final_epoch": parsed["final_epoch"] if parsed else "",
            "wandb_run_id": wandb_id,
            "output_checkpoint": str(output_checkpoint),
            "output_checkpoint_exists": output_checkpoint.exists(),
            "forgotten_target": target,
            "base_checkpoint": cfg.get("checkpoint", ""),
            "config_path": str(config_path.relative_to(REPO_ROOT)),
        }
        rows.append(row)
        if not parsed:
            missing.append(run_name)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    def fmt(v, pct=False):
        if v == "" or v is None:
            return "MISSING"
        if pct:
            return f"{v * 100:.2f}%"
        return f"{v}"

    def print_table(title, scenario):
        print(f"\n{title}")
        print()
        print("ku | FA | RA | TA | MIA | Runtime | W&B Run")
        for r in rows:
            if r["scenario"] != scenario:
                continue
            print(
                f"{r['ku']} | {fmt(r['fa'], pct=True)} | {fmt(r['ra'], pct=True)} | "
                f"{fmt(r['ta'], pct=True)} | {fmt(r['mia'])} | "
                f"{fmt(r['runtime_sec'])}s | {r['wandb_run_id'] or 'MISSING'}"
            )

    print_table("ABLATION — UNLEARN CLASS", "class")
    print_table("ABLATION — UNLEARN DOMAIN", "domain")

    print(f"\n[*] Results CSV: {OUT_CSV.relative_to(REPO_ROOT)}")
    if missing:
        print(f"[!] Missing/unparsed results for: {missing}")


if __name__ == "__main__":
    main()
