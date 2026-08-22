#!/usr/bin/env python3
"""Run base MoE learning followed by one or more unlearning ablations."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run an MoE learning-to-unlearning pipeline from one YAML file."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun existing checkpoints.")
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--name", help="Override the study name from the YAML file.")
    parser.add_argument(
        "--stage",
        choices=("learn", "unlearn", "both"),
        default="both",
        help="Run learning only, unlearning only, or the chained pipeline.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Chosen base checkpoint. Required when --stage unlearn.",
    )
    parser.add_argument(
        "--sweep",
        action="append",
        default=[],
        metavar="KEY=VALUES",
        help="Override a sweep, for example: --sweep 'learn.num_experts=[4,8,12]'.",
    )
    return parser.parse_args()


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)


def stable_id(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def set_nested(config, dotted_key, value):
    section, separator, key = dotted_key.partition(".")
    if not separator or section not in {"learn", "unlearn"} or not key:
        raise ValueError(
            f"Sweep key '{dotted_key}' must have form 'learn.name' or 'unlearn.name'."
        )
    config[section][key] = value


def apply_sweep_overrides(spec, overrides):
    if not overrides:
        return
    parameters = spec.setdefault("sweep", {}).setdefault("parameters", {})
    parameters.clear()
    for override in overrides:
        key, separator, raw_values = override.partition("=")
        if not separator:
            raise ValueError(f"Invalid --sweep '{override}'; expected KEY=[value,...].")
        values = yaml.safe_load(raw_values)
        if not isinstance(values, list) or not values:
            raise ValueError(f"Sweep '{key}' must provide a non-empty YAML list.")
        if not key.startswith(("learn.", "unlearn.")):
            raise ValueError(f"Sweep key '{key}' must start with learn. or unlearn.")
        parameters[key] = values


def expand_trials(spec):
    parameters = spec.get("sweep", {}).get("parameters", {})
    if not parameters:
        return [("default", copy.deepcopy(spec))]

    keys = list(parameters)
    value_lists = []
    for key in keys:
        values = parameters[key]
        if not isinstance(values, list) or not values:
            raise ValueError(f"Sweep values for '{key}' must be a non-empty list.")
        value_lists.append(values)

    trials = []
    for values in itertools.product(*value_lists):
        trial = copy.deepcopy(spec)
        assignments = dict(zip(keys, values))
        for key, value in assignments.items():
            set_nested(trial, key, value)
        label = "__".join(
            f"{key.replace('.', '-')}_{str(value).replace('/', '-')}"
            for key, value in assignments.items()
        )
        trials.append((label, trial))
    return trials


def validate(learn, unlearn):
    if not str(learn.get("model_name", "")).startswith("module_"):
        raise ValueError("learn.model_name must be a supported module_* MoE model.")

    architecture_keys = (
        "dataset", "data_dir", "unlearn_setting", "forget_ratio",
        "forget_classes", "forget_domains", "model_name", "moe_layers",
        "num_experts", "gate_k", "expert_depth", "expert_hidden_ratio", "seed",
    )
    for key in architecture_keys:
        if key in learn:
            unlearn.setdefault(key, learn[key])

    num_experts = int(learn["num_experts"])
    if int(learn["gate_k"]) > num_experts:
        raise ValueError("learn.gate_k cannot exceed learn.num_experts.")
    if int(unlearn.get("k_u", 1)) > num_experts:
        raise ValueError("unlearn.k_u cannot exceed learn.num_experts.")


def run_command(command, dry_run):
    print("$", " ".join(str(part) for part in command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def main():
    args = parse_args()
    spec = load_yaml(args.config)
    if not isinstance(spec, dict) or "learn" not in spec or "unlearn" not in spec:
        raise ValueError("Pipeline YAML must contain 'learn' and 'unlearn' mappings.")

    apply_sweep_overrides(spec, args.sweep)

    if args.stage == "learn":
        invalid = [key for key in spec.get("sweep", {}).get("parameters", {}) if not key.startswith("learn.")]
        if invalid:
            raise ValueError(f"Learning stage accepts only learn.* sweeps, not: {invalid}")

    chosen_checkpoint = None
    if args.stage == "unlearn":
        invalid = [key for key in spec.get("sweep", {}).get("parameters", {}) if not key.startswith("unlearn.")]
        if invalid:
            raise ValueError(f"Unlearning stage accepts only unlearn.* sweeps, not: {invalid}")
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required when --stage unlearn.")
        chosen_checkpoint = args.checkpoint.expanduser().resolve()
        if not chosen_checkpoint.exists() and not args.dry_run:
            raise FileNotFoundError(f"Chosen base checkpoint does not exist: {chosen_checkpoint}")

        # Checkpoints generated by this runner keep their exact learning config
        # two levels above the .pt file. Loading it prevents architecture drift.
        generated_learn_config = chosen_checkpoint.parent.parent / "learn.yaml"
        if generated_learn_config.exists():
            spec["learn"] = load_yaml(generated_learn_config)
            spec["learn"].pop("output_dir", None)
            print(f"Loaded matching learning config: {generated_learn_config}")

    study_name = args.name or spec.get("name", args.config.stem)
    output_root = Path(spec.get("output_root", "runs"))
    if not output_root.is_absolute():
        output_root = REPO_ROOT / output_root
    study_root = output_root / study_name

    trials = expand_trials(spec)
    if args.max_trials is not None:
        trials = trials[: args.max_trials]

    print(f"Study: {study_name} | trials: {len(trials)} | output: {study_root}")
    learned_checkpoints = {}
    learning_manifest = []

    for index, (label, trial) in enumerate(trials, start=1):
        learn = copy.deepcopy(trial["learn"])
        unlearn = copy.deepcopy(trial["unlearn"])
        validate(learn, unlearn)

        print(f"\n[{index}/{len(trials)}] {label}")
        learn_signature = stable_id(learn)

        if args.stage == "unlearn":
            base_checkpoint = chosen_checkpoint
            base_config_path = chosen_checkpoint.parent.parent / "learn.yaml"
        else:
            # Cache base models across studies using the full learning config.
            base_root = output_root / "_base_models" / learn_signature
            base_config_path = base_root / "learn.yaml"
            base_checkpoint = base_root / "checkpoints" / "learn.pt"
            learn["output_dir"] = str(base_root / "checkpoints")
            dump_yaml(base_config_path, learn)

            if learn_signature not in learned_checkpoints:
                if args.force or not base_checkpoint.exists():
                    run_command(
                        [sys.executable, "learn.py", "--config", str(base_config_path)],
                        args.dry_run,
                    )
                else:
                    print(f"[skip] base checkpoint exists: {base_checkpoint}")
                learned_checkpoints[learn_signature] = base_checkpoint
            else:
                print(f"[reuse] base checkpoint: {base_checkpoint}")

            learning_manifest.append(
                {
                    "label": label,
                    "learning_signature": learn_signature,
                    "checkpoint": str(base_checkpoint),
                    "config": str(base_config_path),
                }
            )

        if args.stage == "learn":
            continue

        trial_id = f"{index:03d}_{stable_id({'label': label, 'unlearn': unlearn})}"
        trial_root = study_root / "trials" / trial_id
        unlearn_config_path = trial_root / "unlearn.yaml"
        unlearn["pretrained_model_path"] = str(base_checkpoint)
        unlearn["output_dir"] = str(trial_root / "checkpoints")
        dump_yaml(unlearn_config_path, unlearn)
        dump_yaml(
            trial_root / "metadata.yaml",
            {
                "label": label,
                "base_signature": learn_signature,
                "learn_config": str(base_config_path),
                "unlearn_config": str(unlearn_config_path),
            },
        )

        algorithm = str(unlearn.get("unlearn_algo", "module")).lower()
        final_checkpoint = (
            trial_root / "checkpoints" / f"unlearned_{algorithm}_unlearn.pt"
        )
        if args.force or not final_checkpoint.exists():
            run_command(
                [sys.executable, "unlearn.py", "--config", str(unlearn_config_path)],
                args.dry_run,
            )
        else:
            print(f"[skip] unlearning checkpoint exists: {final_checkpoint}")

    if args.stage in {"learn", "both"}:
        dump_yaml(study_root / "learning_manifest.yaml", learning_manifest)


if __name__ == "__main__":
    main()
