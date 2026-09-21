# sequential class / domain removal: the class-wise (domain-wise) counterpart of
# the "Sequential Unlearning" experiment in CoUn (Khalil et al., NeurIPS 2025,
# Sec. 4.2 / Fig. 6).
#
# CoUn removes 10% of the training data at random every 10 epochs, five times
# (forget ratio 10% -> 50%), each stage starting from the model of the previous
# one, and reports the Average Gap to a Retrain model at every stage. Here a
# stage removes a *group of classes* (or *domains*) instead:
#
#   stage t : unlearn.py starts from the checkpoint of stage t-1 (stage 1 starts
#             from the base model), is fed the classes/domains of request t as
#             the forget set, and everything forgotten so far is excluded from
#             retain/test and included in the FA/MIA evaluation (cumulative,
#             like the growing forget ratio in the paper). optionally a Retrain
#             model that never saw anything forgotten up to stage t is trained
#             so the gap can be reported per stage. that Retrain only depends on
#             the cumulative forget set, so it lives under
#             sequential.retrain_output_root and is shared by every algorithm
#             run with the same stages.
#
# usage:
#   python sequential_unlearn.py --config config/sequential/pacs_module_seq_class.yaml [--retrain]
#
# the yaml is an ordinary unlearn.py config (algo, lr, epochs, base checkpoint,
# ...) plus a `sequential:` block, see config/sequential/*.yaml. one unlearn.py
# run (= one wandb run) per stage, logs/configs/checkpoints under
# sequential.output_root, and a summary.csv with the per-stage metrics and gaps.
import argparse
import copy
import csv
import os
import re
import subprocess
import sys

import yaml

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# same lines the collect_* scripts parse; every approx_algo prints the first one
# per epoch, retrain_baseline.py (Module.learn) prints the second once.
UNLEARN_METRICS_RE = re.compile(
    r"Metrics:\s*RA:\s*([\d.]+)%\s*\|\s*FA:\s*([\d.]+)%\s*\|\s*TA:\s*([\d.]+)%\s*\|\s*MIA:\s*([\d.]+)"
)
RETRAIN_METRICS_RE = re.compile(
    r"\[Final Metrics\]\s*ra:\s*([\d.]+)%\s*\|\s*fa:\s*([\d.]+)%\s*\|\s*ta:\s*([\d.]+)%\s*\|\s*mia:\s*([\d.]+)"
)
TIME_RE = re.compile(r"Total [Tt]ime:\s*([\d.]+)s")

METRIC_KEYS = ["ra", "fa", "ta", "mia"]


def parse_last_metrics(log_path, pattern):
    """RA/FA/TA/MIA of the last evaluation in a log, all in percentage points."""
    if not os.path.exists(log_path):
        return None
    last = None
    for m in pattern.finditer(open(log_path, errors="replace").read()):
        last = m
    if last is None:
        return None
    ra, fa, ta, mia = (float(x) for x in last.groups())
    # the scripts print accuracies in % but the mia score as a 0-1 fraction; put
    # the four on the same scale so the average gap is in points like the paper.
    return {"ra": ra, "fa": fa, "ta": ta, "mia": mia * 100.0}


def parse_time(log_path):
    if not os.path.exists(log_path):
        return None
    m = TIME_RE.findall(open(log_path, errors="replace").read())
    return float(m[-1]) if m else None


def run_logged(cmd, log_path, dry_run):
    """run a command, mirroring its stdout/stderr to the terminal and a log."""
    print(f"[*] $ {' '.join(cmd)}")
    print(f"[*] log: {log_path}")
    if dry_run:
        return 0
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as log, subprocess.Popen(
        cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    ) as proc:
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        proc.wait()
        return proc.returncode


def expand_stages(seq_cfg):
    """
    `stages` may be written explicitly (a list of class/domain id lists) or
    generated from `classes_per_stage` + `num_stages` (consecutive class ids,
    optionally shuffled with `class_order_seed`).
    """
    if "stages" in seq_cfg:
        stages = [list(s) if isinstance(s, (list, tuple)) else [s] for s in seq_cfg["stages"]]
    else:
        per = int(seq_cfg["classes_per_stage"])
        n = int(seq_cfg["num_stages"])
        order = list(range(per * n))
        if seq_cfg.get("class_order_seed") is not None:
            import random
            random.Random(int(seq_cfg["class_order_seed"])).shuffle(order)
        stages = [order[i * per:(i + 1) * per] for i in range(n)]
    seen = set()
    for t, s in enumerate(stages, 1):
        dup = seen & set(s)
        if dup:
            raise ValueError(f"stage {t} repeats ids already removed earlier: {sorted(dup)}")
        seen |= set(s)
    return stages


def main():
    parser = argparse.ArgumentParser(description="Sequential class/domain removal driver (runs unlearn.py once per stage).")
    parser.add_argument("--config", required=True, help="unlearn.py config with a `sequential:` block.")
    parser.add_argument("--retrain", action="store_true",
                        help="also train the per-stage Retrain reference (needs sequential.retrain_config).")
    parser.add_argument("--retrain-only", action="store_true",
                        help="only train the per-stage Retrain references; skip the unlearning chain.")
    parser.add_argument("--force", action="store_true",
                        help="re-run stages whose final checkpoint and metrics already exist.")
    parser.add_argument("--dry-run", action="store_true", help="write the stage configs, run nothing.")
    cmd_args = parser.parse_args()

    with open(cmd_args.config) as f:
        base_cfg = yaml.safe_load(f)
    setting = str(base_cfg.get("unlearn_setting", "random"))
    if setting not in ("class", "domain"):
        raise ValueError("sequential_unlearn.py is for unlearn_setting: class | domain")
    # unlearn.py / retrain_baseline.py keys for the current request and for what
    # earlier stages already removed
    forget_key = f"forget_{setting}s"          # forget_classes | forget_domains
    prev_key = f"prev_forget_{setting}s"       # prev_forget_classes | prev_forget_domains
    seq_cfg = base_cfg.pop("sequential", None)
    if not seq_cfg:
        raise ValueError("config needs a `sequential:` block (stages / output_root ...)")

    stages = expand_stages(seq_cfg)
    forget_mode = str(seq_cfg.get("forget_mode", "new"))
    if forget_mode not in ("new", "cumulative"):
        raise ValueError("sequential.forget_mode must be 'new' or 'cumulative'")

    stem = os.path.splitext(os.path.basename(cmd_args.config))[0]
    out_root = seq_cfg.get("output_root", os.path.join("runs", "sequential", stem))
    cfg_dir = os.path.join(out_root, "configs")
    log_dir = os.path.join(out_root, "logs")
    os.makedirs(cfg_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    # shared across algorithms (see header); default keeps it next to out_root
    retrain_root = seq_cfg.get("retrain_output_root",
                               os.path.join(os.path.dirname(out_root.rstrip("/")), f"retrain_{setting}"))

    unlearn_algo = str(base_cfg.get("unlearn_algo", "finetune")).lower()
    base_study = base_cfg.get("study_name", stem)
    base_ckpt = base_cfg["pretrained_model_path"]
    if not cmd_args.dry_run and not os.path.exists(base_ckpt):
        raise FileNotFoundError(f"base checkpoint not found: {base_ckpt}")

    retrain_tmpl = None
    if cmd_args.retrain or cmd_args.retrain_only:
        rpath = seq_cfg.get("retrain_config")
        if not rpath:
            raise ValueError("--retrain needs sequential.retrain_config (a retrain_baseline.py yaml to use as template)")
        with open(rpath) as f:
            retrain_tmpl = yaml.safe_load(f)
        if retrain_tmpl.get("unlearn_setting") != setting:
            raise ValueError(f"sequential.retrain_config must be a {setting}-setting retrain config")
        os.makedirs(os.path.join(retrain_root, "configs"), exist_ok=True)
        os.makedirs(os.path.join(retrain_root, "logs"), exist_ok=True)

    print("=" * 40)
    print(f"[*] sequential {setting} removal: {len(stages)} stage(s), forget_mode={forget_mode}, algo={unlearn_algo}")
    for t, s in enumerate(stages, 1):
        print(f"    stage {t}: remove {setting}s {s}")
    print(f"[*] output root: {out_root}")
    if retrain_tmpl is not None:
        print(f"[*] retrain root (shared): {retrain_root}")

    rows = []
    prev_ckpt = base_ckpt
    prev_classes = []
    for t, new_classes in enumerate(stages, 1):
        cum_classes = prev_classes + new_classes
        tag = f"stage_{t}"
        print("\n" + "=" * 40)
        print(f"[*] stage {t}/{len(stages)} | new: {new_classes} | cumulative: {cum_classes} "
              f"({len(cum_classes)} {setting}s)")

        row = {"stage": t, "new_ids": " ".join(map(str, new_classes)),
               "num_forgot": len(cum_classes)}

        # ---------- unlearning stage ----------
        if not cmd_args.retrain_only:
            stage_cfg = copy.deepcopy(base_cfg)
            if forget_mode == "new":
                stage_cfg[forget_key] = list(new_classes)
                stage_cfg[prev_key] = list(prev_classes)
            else:
                stage_cfg[forget_key] = list(cum_classes)
                stage_cfg[prev_key] = []
            stage_cfg["pretrained_model_path"] = prev_ckpt
            stage_cfg["output_dir"] = os.path.join(out_root, tag, "checkpoints")
            stage_cfg["study_name"] = f"{base_study}__{tag}"
            stage_cfg["wandb_tags"] = list(base_cfg.get("wandb_tags") or []) + [f"sequential_{setting}", tag]
            stage_cfg.setdefault("wandb_group", f"Sequential_{stem}")
            stage_cfg["sequential_stage"] = t
            stage_cfg[f"sequential_cumulative_{setting}s"] = list(cum_classes)
            stage_cfg["config_path"] = os.path.join(cfg_dir, f"{tag}.yaml")

            cfg_path = os.path.join(cfg_dir, f"{tag}.yaml")
            with open(cfg_path, "w") as f:
                yaml.safe_dump(stage_cfg, f, sort_keys=False)

            # unlearn.py names the checkpoint after the algo and the yaml stem
            final_ckpt = os.path.join(stage_cfg["output_dir"], f"unlearned_{unlearn_algo}_{tag}.pt")
            log_path = os.path.join(log_dir, f"{tag}.log")

            done = os.path.exists(final_ckpt) and parse_last_metrics(log_path, UNLEARN_METRICS_RE) is not None
            if done and not cmd_args.force:
                print(f"[*] stage {t} already finished ({final_ckpt}); skipping (use --force to redo)")
            else:
                rc = run_logged([sys.executable, "unlearn.py", "--config", cfg_path], log_path, cmd_args.dry_run)
                if rc != 0:
                    print(f"[!] stage {t} failed (exit {rc}); stopping the chain")
                    break
                if not cmd_args.dry_run and not os.path.exists(final_ckpt):
                    print(f"[!] stage {t} finished but {final_ckpt} is missing; stopping the chain")
                    break

            m = parse_last_metrics(log_path, UNLEARN_METRICS_RE)
            if m:
                row.update({k: m[k] for k in METRIC_KEYS})
            row["unlearn_time_sec"] = parse_time(log_path)
            prev_ckpt = final_ckpt

        # ---------- retrain reference for this stage ----------
        if retrain_tmpl is not None:
            # named after the cumulative forget set, not the stage index, so two
            # configs with the same stages share it and different ones never collide
            r_stem = f"retrain_{setting}_" + "-".join(map(str, sorted(cum_classes)))
            r_cfg = copy.deepcopy(retrain_tmpl)
            r_cfg[forget_key] = list(cum_classes)
            r_cfg["output_root"] = retrain_root
            r_cfg["study_name"] = f"Retraining__{str(base_cfg.get('dataset', '')).upper()}__seq{setting}__{r_stem}__seed{r_cfg.get('seed', '')}"
            r_cfg["wandb_tags"] = list(retrain_tmpl.get("wandb_tags") or []) + [f"sequential_{setting}", tag]
            r_cfg["wandb_group"] = seq_cfg.get("retrain_wandb_group", f"Sequential_retrain_{setting}")
            r_cfg["sequential_stage"] = t
            r_cfg_path = os.path.join(retrain_root, "configs", f"{r_stem}.yaml")
            r_cfg["config_path"] = r_cfg_path
            with open(r_cfg_path, "w") as f:
                yaml.safe_dump(r_cfg, f, sort_keys=False)

            # retrain_baseline.py writes to <output_root>/<yaml stem>/<yaml stem>.pt
            r_ckpt = os.path.join(retrain_root, r_stem, f"{r_stem}.pt")
            r_log = os.path.join(retrain_root, "logs", f"{r_stem}.log")
            done = os.path.exists(r_ckpt) and parse_last_metrics(r_log, RETRAIN_METRICS_RE) is not None
            if done and not cmd_args.force:
                print(f"[*] retrain for stage {t} already finished ({r_ckpt}); skipping")
            else:
                rc = run_logged([sys.executable, "retrain_baseline.py", "--config", r_cfg_path], r_log, cmd_args.dry_run)
                if rc != 0:
                    print(f"[!] retrain for stage {t} failed (exit {rc}); gap for this stage will be empty")

            rm = parse_last_metrics(r_log, RETRAIN_METRICS_RE)
            if rm:
                row.update({f"retrain_{k}": rm[k] for k in METRIC_KEYS})
                if all(k in row for k in METRIC_KEYS):
                    gaps = {f"gap_{k}": abs(row[k] - rm[k]) for k in METRIC_KEYS}
                    row.update(gaps)
                    row["avg_gap"] = sum(gaps.values()) / len(gaps)

        rows.append(row)
        prev_classes = cum_classes

    # ---------- summary ----------
    if not rows:
        return
    fields = ["stage", "new_ids", "num_forgot", *METRIC_KEYS, "unlearn_time_sec",
              *[f"retrain_{k}" for k in METRIC_KEYS], *[f"gap_{k}" for k in METRIC_KEYS], "avg_gap"]
    summary_path = os.path.join(out_root, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else (f"{r[k]:.2f}" if isinstance(r[k], float) else r[k]))
                        for k in fields})

    print("\n" + "=" * 40)
    print(f"[*] sequential {setting} removal summary ({unlearn_algo}, forget_mode={forget_mode})")
    hdr = f"{'stage':>5} {'#fgt':>5} {'RA':>7} {'FA':>7} {'TA':>7} {'MIA':>7} {'AvgGap':>7}"
    print(hdr)
    for r in rows:
        def fmt(k):
            return f"{r[k]:7.2f}" if isinstance(r.get(k), float) else f"{'-':>7}"
        print(f"{r['stage']:>5} {r['num_forgot']:>5} {fmt('ra')} {fmt('fa')} {fmt('ta')} {fmt('mia')} {fmt('avg_gap')}")
    print(f"[*] written to {summary_path}")


if __name__ == "__main__":
    main()
