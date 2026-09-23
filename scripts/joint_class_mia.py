#!/usr/bin/env python3
"""Membership leakage under JOINT class unlearning on PACS (1, 2, 3 classes).

Controlled evaluation, reported separately from the main-table MIA. What differs
from `metric/mia.py` (the main-table protocol) is stated in the output manifest:

  * members    = the forgotten TRAINING examples of the requested classes.
    non-members= never-trained examples of the SAME classes (test + unseen splits).
  * the two sides are matched cell-by-cell on (class x domain) and balanced, so
    neither semantics nor group size is a shortcut for the attacker.
  * the number of attacker-fitting and attacker-evaluation samples is FIXED across
    the 1/2/3-class requests (sized from the smallest request), so a larger forget
    set cannot hand the attacker more data.
  * attacker fitting, feature standardisation and final scoring are separated by
    sample id (repeated stratified splits); the same sample ids and the same folds
    are used for every method within a request.
  * the attack feature is the per-sample cross-entropy loss, as in the repo's MIA.
    Reported value = balanced accuracy on held-out attacker-evaluation samples;
    chance = 0.500. Raw values, never clipped and never inverted.

Requests are JOINT, not sequential: every unlearned model starts from the original
checkpoint, and every Retrain model is trained from ImageNet init on its own retain
set (never from the task-finetuned checkpoint).

  python scripts/joint_class_mia.py                     # evaluate + write csv/figure
  python scripts/joint_class_mia.py --skip-missing      # tolerate not-yet-finished runs
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset, random_split

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from unlearn import ApplyTransform  # noqa: E402  (same transform wrapper the runs use)
from architecture.module import ModuleArchitecture  # noqa: E402
from dataset.pytorch_dataset.pacs import PACSDataset  # noqa: E402
from dataset.transform.test_transform import get_test_transform  # noqa: E402

OUT = REPO / "results" / "joint_class_mia"
BASE_CKPT = "runs/_base_models/pacs_M12_k4_seed42/checkpoints/pacs_module_base_M12_k4_best.pt"
# fixed, pre-registered class order (the main table's one-class target first)
REQUESTS = {"1 class": [0], "2 classes": [0, 1], "3 classes": [0, 1, 2]}

# model -> request tag -> (checkpoint, provenance)
def model_table():
    m = {"Original": {}, "ModULE": {}, "SEUF": {}, "Retrain": {}}
    for tag, cls in REQUESTS.items():
        key = "-".join(map(str, cls))
        m["Original"][tag] = (BASE_CKPT, "original base model, no unlearning")
        n = {1: "c1", 2: "c2", 3: "c3"}[len(cls)]
        for method, algo in (("ModULE", "module"), ("SEUF", "seuf")):
            own = f"runs/joint_class/{algo}_joint_{n}/checkpoints/unlearned_{algo}_{algo}_joint_{n}.pt"
            # the 1-class request of the sequential experiment is the same run (same base,
            # same forget set, same hyper-parameters, no preceding stage): reuse it when this
            # machine has it, otherwise use the dedicated joint run.
            reuse = f"runs/sequential/pacs_{algo}_seq_class/stage_1/checkpoints/unlearned_{algo}_stage_1.pt"
            if (REPO / own).exists() or cls != [0]:
                m[method][tag] = (own, f"config/joint_class/{algo}_joint_{n}.yaml (started from the original checkpoint)")
            else:
                m[method][tag] = (reuse, f"reused sequential stage 1: config verified semantically identical to "
                                         f"config/joint_class/{algo}_joint_c1.yaml")
        m["Retrain"][tag] = (f"runs/sequential/pacs_retrain_class/retrain_class_{key}/retrain_class_{key}.pt",
                             "retrained from ImageNet init on the retain set only (50 epochs)")
    return m


def build_splits(seed=42, data_dir="./dataset/data_folder/pacs"):
    """exactly the split unlearn.py / retrain_baseline.py use."""
    ds = PACSDataset(root_dir=data_dir, transform=None)
    n = len(ds)
    tr, te = int(0.8 * n), int(0.1 * n)
    g = torch.Generator().manual_seed(seed)
    train, test, unseen = random_split(ds, [tr, te, n - tr - te], generator=g)
    return ds, list(train.indices), list(test.indices), list(unseen.indices)


def cell(ds, idx):
    return int(ds.labels[idx]), int(ds.domains[idx])


def matched_pairs(ds, member_ids, nonmember_ids, budget, rng):
    """pick the same number of members and non-members in every (class, domain) cell,
    then trim to `budget` per side keeping the cell proportions."""
    mem, non = {}, {}
    for i in member_ids:
        mem.setdefault(cell(ds, i), []).append(i)
    for i in nonmember_ids:
        non.setdefault(cell(ds, i), []).append(i)
    cells = sorted(set(mem) & set(non))
    cap = {c: min(len(mem[c]), len(non[c])) for c in cells}
    total = sum(cap.values())
    take = dict(cap)
    if budget is not None and budget < total:  # largest-remainder allocation
        exact = {c: cap[c] * budget / total for c in cells}
        take = {c: int(np.floor(exact[c])) for c in cells}
        rem = budget - sum(take.values())
        for c in sorted(cells, key=lambda c: exact[c] - take[c], reverse=True)[:rem]:
            take[c] += 1
    out_m, out_n = [], []
    for c in cells:
        k = take[c]
        if k <= 0:
            continue
        out_m += list(rng.choice(mem[c], k, replace=False))
        out_n += list(rng.choice(non[c], k, replace=False))
    return sorted(int(i) for i in out_m), sorted(int(i) for i in out_n), cap


@torch.no_grad()
def per_sample_loss(model, ds, ids, device, batch_size=128):
    """per-sample CE loss under the deterministic test transform (attack feature)."""
    loader = DataLoader(ApplyTransform(Subset(ds, ids), get_test_transform()),
                        batch_size=batch_size, shuffle=False, num_workers=4)
    out = []
    for batch in loader:
        logits, _ = model.inference(batch[0].to(device))
        out.append(F.cross_entropy(logits, batch[1].to(device), reduction="none").float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def accuracy(model, ds, ids, device, batch_size=128):
    if not ids:
        return float("nan"), 0
    loader = DataLoader(ApplyTransform(Subset(ds, ids), get_test_transform()),
                        batch_size=batch_size, shuffle=False, num_workers=4)
    ok = tot = 0
    for batch in loader:
        logits, _ = model.inference(batch[0].to(device))
        ok += (logits.argmax(1) == batch[1].to(device)).sum().item()
        tot += batch[1].numel()
    return ok / tot, tot


def attack(losses, is_member, folds):
    """logistic regression on the 1-D loss feature. Standardisation and fitting use
    attacker-train ids only; the score is balanced accuracy on the held-out ids."""
    accs, aucs = [], []
    for tr_idx, ev_idx in folds:
        x_tr, y_tr = losses[tr_idx].reshape(-1, 1), is_member[tr_idx]
        x_ev, y_ev = losses[ev_idx].reshape(-1, 1), is_member[ev_idx]
        mu, sd = x_tr.mean(), x_tr.std()
        sd = sd if sd > 0 else 1.0
        clf = LogisticRegression(max_iter=1000).fit((x_tr - mu) / sd, y_tr)
        p = clf.predict((x_ev - mu) / sd)
        accs.append(balanced_accuracy_score(y_ev, p))
        s = clf.predict_proba((x_ev - mu) / sd)[:, 1]
        aucs.append(roc_auc_score(y_ev, s) if len(set(y_ev)) > 1 else float("nan"))
    return float(np.mean(accs)), float(np.std(accs)), float(np.nanmean(aucs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--attack-seed", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--eval-frac", type=float, default=0.3, help="held-out share of the attack samples")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--skip-missing", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds, train_ids, test_ids, unseen_ids = build_splits(seed=a.seed)
    names, dnames = ds.class_names, ds.domain_names
    rng = np.random.default_rng(a.seed)

    # ---- per-request sample selection (identical for every method) -----------
    pools, checks = {}, {}
    for tag, cls in REQUESTS.items():
        mem = [i for i in train_ids if int(ds.labels[i]) in cls]
        non = [i for i in test_ids + unseen_ids if int(ds.labels[i]) in cls]
        retain = [i for i in train_ids if int(ds.labels[i]) not in cls]
        _, _, cap = matched_pairs(ds, mem, non, None, np.random.default_rng(0))
        pools[tag] = dict(cls=cls, mem=mem, non=non, retain=retain, cap=cap, max_matched=sum(cap.values()))
        checks[tag] = {
            "forget_retain_disjoint": len(set(mem) & set(retain)) == 0,
            "all_requested_classes_in_forget": sorted({int(ds.labels[i]) for i in mem}) == sorted(cls),
            "nonmembers_never_trained": len(set(non) & set(train_ids)) == 0,
            "n_forget_train": len(mem), "n_retain_train": len(retain), "n_nonmember_pool": len(non),
        }
    budget = min(p["max_matched"] for p in pools.values())   # fixed per side, from the smallest request
    print(f"[*] matched pool per side: " + ", ".join(f"{t}={p['max_matched']}" for t, p in pools.items())
          + f"  -> fixed budget {budget}/side ({2*budget} attack samples per request)")

    for tag, p in pools.items():
        m_ids, n_ids, _ = matched_pairs(ds, p["mem"], p["non"], budget, np.random.default_rng(a.seed))
        p["m_ids"], p["n_ids"] = m_ids, n_ids
        ids = np.array(m_ids + n_ids)
        y = np.array([1] * len(m_ids) + [0] * len(n_ids))
        strat = np.array([f"{y[k]}_{int(ds.labels[i])}" for k, i in enumerate(ids)])
        sss = StratifiedShuffleSplit(n_splits=a.repeats, test_size=a.eval_frac, random_state=a.attack_seed)
        p["ids"], p["y"], p["folds"] = ids, y, list(sss.split(ids.reshape(-1, 1), strat))
        comp_m = Counter((names[int(ds.labels[i])], dnames[int(ds.domains[i])]) for i in m_ids)
        comp_n = Counter((names[int(ds.labels[i])], dnames[int(ds.domains[i])]) for i in n_ids)
        checks[tag]["class_domain_composition_matched"] = comp_m == comp_n
        checks[tag]["n_attack_members"] = len(m_ids)
        checks[tag]["n_attack_nonmembers"] = len(n_ids)
        checks[tag]["n_attacker_fit"] = len(p["folds"][0][0])
        checks[tag]["n_attacker_eval"] = len(p["folds"][0][1])
        checks[tag]["composition"] = {f"{c}/{d}": v for (c, d), v in sorted(comp_m.items())}
        checks[tag]["test_ids_retained_classes"] = len([i for i in test_ids if int(ds.labels[i]) not in p["cls"]])
        checks[tag]["test_ids_all_classes"] = len(test_ids)

    # ---- evaluate every model -----------------------------------------------
    table, per_example, missing = [], [], []
    model = ModuleArchitecture(model_name="module_small_patch16_224", num_classes=7, pretrained=False,
                               moe_layers="FFFFFFFFFFSS", num_experts=12, expert_depth=2,
                               expert_hidden_ratio=2, gate_k=4, device=device)
    for method, per_req in model_table().items():
        for tag, (ck, prov) in per_req.items():
            if not (REPO / ck).exists():
                missing.append(f"{method} / {tag}: {ck}")
                if a.skip_missing:
                    continue
                raise FileNotFoundError(f"missing checkpoint for {method} / {tag}: {ck}")
            model.load_state_dict(torch.load(REPO / ck, map_location=device))
            model.eval()
            p = pools[tag]
            losses = per_sample_loss(model, ds, list(p["ids"]), device, a.batch_size)
            acc, std, auc = attack(losses, p["y"], p["folds"])

            fa, n_fa = accuracy(model, ds, p["mem"], device, a.batch_size)          # forget train set
            ra, n_ra = accuracy(model, ds, p["retain"], device, a.batch_size)       # retain train set
            keep_test = [i for i in test_ids if int(ds.labels[i]) not in p["cls"]]
            ta, n_ta = accuracy(model, ds, keep_test, device, a.batch_size)         # test, retained classes
            ta_all, n_ta_all = accuracy(model, ds, test_ids, device, a.batch_size)  # test, all classes
            table.append(dict(request=tag, n_classes=len(p["cls"]),
                              forget_classes=" ".join(names[c] for c in p["cls"]), method=method,
                              mia_bal_acc=round(acc, 4), mia_std=round(std, 4), mia_auc=round(auc, 4),
                              fa=round(100 * fa, 2), ra=round(100 * ra, 2), ta_retained=round(100 * ta, 2),
                              ta_all_classes=round(100 * ta_all, 2),
                              n_fa=n_fa, n_ra=n_ra, n_ta_retained=n_ta, n_ta_all=n_ta_all,
                              n_attack_per_side=len(p["m_ids"]), checkpoint=ck, provenance=prov))
            print(f"  {tag:10s} {method:9s} MIA {acc:.3f}±{std:.3f} (AUC {auc:.3f}) | "
                  f"FA {100*fa:5.2f} RA {100*ra:5.2f} TA_ret {100*ta:5.2f} TA_all {100*ta_all:5.2f}")
            fold0_eval = set(p["folds"][0][1].tolist())
            for k, sid in enumerate(p["ids"]):
                per_example.append(dict(request=tag, method=method, sample_id=int(sid),
                                        class_id=int(ds.labels[sid]), class_name=names[int(ds.labels[sid])],
                                        domain_id=int(ds.domains[sid]), domain_name=dnames[int(ds.domains[sid])],
                                        is_member=int(p["y"][k]), loss=float(losses[k]),
                                        fold0_split="eval" if k in fold0_eval else "fit"))

    import csv
    with open(OUT / "joint_mia_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys())); w.writeheader(); w.writerows(table)
    with open(OUT / "joint_mia_per_example.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_example[0].keys())); w.writeheader(); w.writerows(per_example)
    with open(OUT / "joint_mia_splits.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["request", "repeat", "sample_id", "assignment"])
        for tag, p in pools.items():
            for r, (tr, ev) in enumerate(p["folds"]):
                for k in tr:
                    w.writerow([tag, r, int(p["ids"][k]), "fit"])
                for k in ev:
                    w.writerow([tag, r, int(p["ids"][k]), "eval"])
    manifest = dict(
        seed=a.seed, attack_seed=a.attack_seed, repeats=a.repeats, eval_frac=a.eval_frac,
        class_order=[f"{c}:{names[c]}" for c in [0, 1, 2]],
        requests={t: [f"{c}:{names[c]}" for c in v] for t, v in REQUESTS.items()},
        fixed_attack_budget_per_side=budget, chance_level=0.5,
        attack="logistic regression on per-sample CE loss; fit+standardise on attacker-fit ids only; "
               "balanced accuracy on held-out ids; 20 repeated stratified splits (membership x class); "
               "identical ids and folds across methods within a request",
        differs_from_main_table_mia=("main table: cross_val_score accuracy over the whole matched pool with global "
                                     "standardisation and no fixed budget across settings; here: explicit id-level "
                                     "fit/eval separation, (class x domain) cell matching, and a per-side sample "
                                     "budget fixed to the 1-class request. Reported separately; existing results untouched."),
        checks=checks, missing_checkpoints=missing)
    (OUT / "joint_mia_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[*] wrote {OUT/'joint_mia_metrics.csv'}, per-example scores, splits, manifest")
    if missing:
        print("[!] missing checkpoints:\n   " + "\n   ".join(missing))
    make_figure(table)


def make_figure(table):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[*] matplotlib missing: no figure"); return
    COLOR = {"Original": "#eb6834", "ModULE": "#2a78d6", "SEUF": "#1baf7a", "Retrain": "#7a7973"}
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#c9c8c1", "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "grid.color": "#e6e5df"})
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True)
    xs_all = sorted({r["n_classes"] for r in table})
    for ax, (key, err, title) in zip(axes, [("mia_bal_acc", "mia_std", "MIA balanced accuracy (chance 0.5)"),
                                            ("ta_retained", None, "Test accuracy, retained classes (%)")]):
        for method in ["Original", "ModULE", "SEUF", "Retrain"]:
            rows = sorted([r for r in table if r["method"] == method], key=lambda r: r["n_classes"])
            if not rows:
                continue
            xs = [r["n_classes"] for r in rows]; ys = [r[key] for r in rows]
            kw = dict(marker="o", ms=4, lw=2, color=COLOR[method], label=method)
            if method == "Retrain":
                kw.update(ls="--", marker="s", ms=3)
            if err:
                ax.errorbar(xs, ys, yerr=[r[err] for r in rows], capsize=2, **kw)
            else:
                ax.plot(xs, ys, **kw)
        ax.set_xlabel("# classes unlearned jointly"); ax.set_xticks(xs_all)
        ax.set_title(title, fontsize=9, loc="left"); ax.grid(axis="y", lw=0.6)
        if key == "mia_bal_acc":
            ax.axhline(0.5, color="#b0afa8", lw=1, ls=":")
            ax.annotate("chance", (xs_all[0], 0.5), xytext=(2, 3), textcoords="offset points",
                        fontsize=7, color="#7a7973")
            ax.set_ylim(0.40, 0.75)
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.suptitle("Membership leakage under joint class unlearning (PACS, one seed, classes dog→elephant→giraffe)",
                 fontsize=9, x=0.01, ha="left")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"joint_class_mia.{ext}", dpi=200)
    print(f"[*] wrote {OUT/'joint_class_mia.png'}")


if __name__ == "__main__":
    main()
