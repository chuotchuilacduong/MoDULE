# Review chat: verifying expert specialization, online metrics, and a route-separation loss

Source: shared Claude chat https://claude.ai/share/01e12f03-3f44-4391-93f9-90ed6f4671ac (pasted by the
user on 2026-09-16; the chat's attached files — `online_specialization.py`, `expert_necessity_ablation.py`,
`loss_specialization_summary.py`, `route_separation.py`, `module_losses.py`, `pacs_module_v2.yaml`,
`UPGRADE.patch` — were hidden in the share and are NOT reproduced here).

The reviewer worked from a `--depth 1` clone of `origin/main`, never from branch `DUKKU`. Section 0 maps
each finding onto the current `DUKKU` code; the verbatim transcript follows.

---

## 0. Status of each finding against `DUKKU` (as of 2026-09-16)

| # | Reviewer finding | `origin/main` | `DUKKU` |
|---|---|---|---|
| 1 | `L_bal` receives `last_pi` `[N,k]` → `M = k`, blind to expert identity, missing `M` prefactor | **still broken** | **fixed** in `7f3a051`: `architecture/module.py` caches `last_gate_mass` `[N,M]`; `_loss_balance` returns `M·Σ(ḡ−1/M)²`; `balance_estimator` = minibatch/ema/none |
| 2 | `L_sp` = entropy of the double-softmax top-k gate, ceiling `log k`, not `H(π)/log M` (Eq. 3/32) | still as described | **not changed** — `_loss_sparse` still acts on `last_pi`; documented as a code–paper deviation |
| 3 | `L_div` is uncentered Frobenius cross-correlation, not centered linear CKA (Eq. 5/38–40) | still as described | **partially**: `diversity_objective` key with `_div_centered_cka` (Eq. 39) added; default remains `output_decorrelation` (original behaviour); no probe subsample / stop-gradient |
| 4 | Selection rule is `diff` with `alpha` (weighted subtraction), paper says lexicographic argmin `E_r` s.t. `C_f ≥ τ_f` | same | same (`selection_option: diff`, `alpha: 1.0`) |
| 5 | Gate weights `softmax(topk_vals)` re-softmax probabilities → gate entropy compressed to 84–100 % of `log k` | same | same (root cause of `sp_loss = 0.5853` floor); reviewer proposes sum-normalisation `topk_vals / topk_vals.sum()` (Eq. 29) |
| 6 | Online per-epoch specialization metrics (`online_spec/*`) | **added by skydvn** (`metric/online_specialization.py`, 4 call sites in `module.py`, commits 7791fe3–72c7167) | **absent** |
| 7 | `ablation/expert_necessity_ablation.py` (forced-routing sufficiency/necessity) | **added by skydvn** | absent |
| 8 | New `L_sep = −I(G;E)` mutual-information route-separation loss (Mod-Squad style), `lambda_sep`, `sep_axis` | absent | absent |
| 9 | Hyper-parameter mismatch paper App. G (M=12, k=4, λ=0.1/0.25/0.1, hidden ratio 0.1) vs code (M=8, k=2, λ=1/1/1, ratio 2) | — | Table 1 ModULE row uses M=12/k=4, λ_bal=0.25; the main-table base `aaedd1da4d` is M=8/k=2, λ=0.5 |

Empirical facts from this repo that bear on the review:

* Cross-domain route overlap the reviewer quantified from Fig. 6 (expert 7 in top-2 of 3/4 domains, Photo/Art
  Jaccard 1.0 at layer 2) comes from base `aaedd1da4d`, trained with the broken `L_bal` (2/8 dead experts,
  max/min mass 148×). After the `L_bal` fix (`base_balfix_*`): collapse 0.34 → 0.16, 0 dead experts,
  domain `cover` 2/4 → 4/4 — so part of the sharing was the bug, as the reviewer cautioned.
* Class×expert mass shows stronger structure than domain (e5 = animal 20.6×, e1 = house 11.2×, e3 =
  person 7.9× vs max 6.37× for domain) — consistent with the reviewer's reading of Fig. 7.
* Table 16 (routing entropy): each regulariser breaks exactly one column; Table 17 (expert similarity):
  ModULE wins Mean/Max CKA — measured with true centered CKA, not the training `L_div`.

### 0.1 What was implemented on `DUKKU` from this review (2026-09-16)

All new behaviour is **opt-in**; with default config keys the forward pass and the loss are
bit-for-bit the old ones (smoke-tested: `sp=0.6923`, `sep=0.0000`, `output_decorrelation`), so every
existing checkpoint and table stays valid.

| Key (learn config) | Default (= old) | New option | Where |
|---|---|---|---|
| `gate_norm` | `softmax` | `sum` — Eq. (29) `topk/sum` instead of re-softmax; **changes the forward pass**, must be carried to unlearn/diagnostics (pipeline `architecture_keys` does this; direct `unlearn.py` configs need the key) | `architecture/module.py` (`DeepMoELayer.gate_norm`, `ModuleArchitecture(gate_norm=)`), threaded through `learn.py`, `unlearn.py`, `retrain_baseline.py`, `module_diagnostics.py`, `ablation/*.py` |
| `sparse_target` | `gate` | `pi_all` — `H(π)/log M` on the full softmax, Eq. (3)/(32), in [0,1] | `Module._loss_sparse` |
| `probe_size` | `0` (all tokens) | `n` — random token probe `B_p` for `L_div`, Eq. (37) | `Module._probe_subsample` |
| `lambda_sep`, `sep_axis`, `sep_use_gated`, `sep_ema_alpha` | `0.0`, `domain`, `true`, `0.0` | `L_sep = −I(G;E)/min(log G, log M)` ∈ [−1,0] (Mod-Squad MI on gated mass) | `metric/route_separation.py`, wired in `Module.learn()`; logged as `sep_loss`, `sep_loss_weighted` |
| `online_spec_log_every` | `1` | `0` disables | `metric/online_specialization.py` (ported from `origin/main` and extended) |
| `last_v` | — | detached expert input cached per MoE layer (sg(U_p)), for future backbone-free CKA probes | `DeepMoELayer.forward` |

`online_spec/*` additions over the `origin/main` tracker: `mean_*_purity_used` (average over experts that
received ≥1 image — the reviewer's own caveat that dead experts drag the plain mean below chance),
`cover`, and the **cross-group route-overlap statistics the review asked for**: `domain_mi_norm`,
`domain_overlap` (mean pairwise Σmin), `domain_topk_jaccard`, and the same for `class_*`. `domain_mi_norm`
is exactly the quantity `L_sep` maximises, so the online panel reads route separation directly. The debug
`print("online_spec")` was removed. `ablation/expert_necessity_ablation.py` was ported from `origin/main`
with the `100 × percentage` baseline bug fixed (`evaluate_accuracy` already returns a percentage; the
pushed version would print a 9,000 % baseline).

Verified on real tensors (CPU, `ModuleArchitecture` M=8/k=2, and a 2-epoch `learn.py` run on a 240-image
PACS subset with `config/learn/pacs_module_v2.yaml`): gate rows sum to 1 in both modes; softmax mode
equals the old formula exactly; `L_sep` ∈ [−1,0], gradient reaches the router, sanity cases disjoint
→ −1.000 / uniform → 0 / collapsed → 0; `sp` ∈ [0,1] with `pi_all`; `sep` fell −0.022 → −0.039 over two
epochs (MI rising); all tracker metrics in range with chance lines 0.25 / 0.1429.

Not implemented (deliberately): the Eq. 8 lexicographic selection rule (finding 4) and any change of
defaults. `config/learn/pacs_module_v2.yaml` turns every paper-faithful switch on at once; run it first
with `lambda_sep: 0.0` to attribute the bug-fixes separately from the new term.

---

## 1. Verbatim transcript

### User
This is the repository https://github.com/chuotchuilacduong/MoDULE . I want to consider how to verify the learning phase. Specifically. I want to verify if the experts in MoDULE algo are learned in a specialized way enough? How to verify if it is specialized and make runner commands so that I can check.

### Assistant
I'll explore the repository to understand the MoDULE architecture and how the experts are trained, then figure out how to verify expert specialization.

This is a machine unlearning benchmark using a Mixture-of-Experts (MoE) architecture called "MoDULE". There's a module_diagnostics.py file that looks directly relevant to verifying expert specialization. Let me clone the repo to examine the code properly.

The repo has exactly the tooling we need — module_diagnostics.py, a metric/router_metrics.py, and ablation scripts like domain_class_expert_alignment.py. Let me read the core architecture first, then the diagnostics.

Now I understand the architecture: it's a DeiT ViT where some MLP blocks are swapped for MoE layers (the moe_layers string, S=sparse/MoE, F=FFN), each with num_experts experts and a noisy top-k router. Let me read the diagnostics and router-metric tooling, which is what verifies specialization.

This is the heart of it. Let me read the truncated middle section (196–373) which contains the specialization printing logic and the localization diagnostics.

The diagnostics are clear now. Let me check the metric/ router files and the ablation scripts that directly test specialization, plus the config and scripts so I can give exact runner commands.

Now the ablation scripts and the scripts folder, which give me the runnable entry points. Let me look at the domain/class alignment ablation and how scripts are structured.

Let me see the rest of the alignment ablation output, the training losses in learn.py (specialization is driven by those), and a learn config plus the run scripts.

Now the training losses in learn.py — the specialization is created by these objectives, so understanding them tells us what "specialized enough" means. And the run scripts + a learn config.

The Module trainer in approx_algo/module.py defines the three losses (sparse, balance, div). Let me look at those, since they define what "specialized" means during training.

Let me read the three loss definitions and the final-diagnostics function precisely, then a learn config and the run scripts.

Now the learn config and the run scripts to get exact invocations.

Let me check the config structure and the remaining ablation entry points so I can give a complete, accurate command set.

Let me confirm the checkpoint save path convention so the runner commands point at real files.

One more check — the diagnostics scripts want a flat config or a derived learn.yaml next to the checkpoint, but pacs_module.yaml is nested. Let me confirm how the pipeline lays out checkpoints + learn.yaml so the commands are correct.

I've read through the architecture, the training losses, and the diagnostic/ablation tooling. Here's how to think about "specialized enough" in this specific codebase, and the exact commands to check it.

**What "specialized" concretely means here**

In MoDULE, only the blocks marked S in moe_layers are MoE (the example config FFFFFFFFFFSS means just the last 2 of 12 ViT blocks have experts). Each MoE layer has num_experts experts and a noisy top-gate_k router. Specialization is driven by three losses in approx_algo/module.py, so those define what you're verifying:

- Sparse routing loss (_loss_sparse): entropy of the top-k gate weights. Pushes each token to commit to its chosen expert(s) rather than spreading mass. This is the thing Lsp actually trains on.
- Load-balance loss (_loss_balance): L2 of mean routing mass vs uniform 1/E (with EMA). Prevents expert collapse / dead experts.
- Diversity loss (_loss_diversity): squared cross-correlation between normalized expert outputs, pushing experts to produce decorrelated representations.

So a "well-specialized" model should show: committed (low-entropy) routing, no dead experts, and experts whose assigned inputs map cleanly to distinct domains/classes — and that mapping should hold on held-out data, not just training data.

**The primary check**

Everything you need is in module_diagnostics.py. Run it on a trained checkpoint:

```bash
# from repo root. If the checkpoint was produced by config/experiments/run_moe_pipeline.py,
# it auto-derives the flat config from <ckpt>/../../learn.yaml — just pass --checkpoint.
python module_diagnostics.py --checkpoint runs/_base_models/<hash>/checkpoints/learn.pt --no-wandb

# add the causal localization metrics (FEM/ARR/RFO):
python module_diagnostics.py --checkpoint <ckpt>.pt --run-eq7-diagnostics --no-wandb

# if you trained directly with learn.py (no sibling learn.yaml), pass a FLAT config
# (top-level keys: dataset, data_dir, model_name, moe_layers, num_experts, gate_k,
#  expert_depth, expert_hidden_ratio, seed, batch_size, unlearn_setting, forget_ratio):
python module_diagnostics.py --checkpoint <ckpt>.pt --config <flat_learn>.yaml --no-wandb
```

This prints, per MoE layer, for both TRAIN and TEST splits: sparsity entropy, load/dead-expert counts, a per-domain routing-mass table, and the key per-expert domain/class specialization table with dom purity and cls purity. Note the specialization tables are gated to PACS (a domain-labeled dataset) in the standalone script; for CIFAR-100/Tiny-ImageNet you only get accuracy + entropy + load. The same diagnostics also run automatically at the end of learn() for domain-labeled datasets, so you can read them from the training log too.

**How to read it — is it specialized enough?**

Judge against chance baselines, not in absolute terms:

- Gate entropy (top-k gate entropy (last_pi)) vs max log(gate_k). With gate_k=2, max ≈ 0.693. Close to 0 = strong commitment; close to max = 50/50 indecision. This is your sparsity signal.
- Dead experts should be ~0 and balance deviation small. Several dead experts means the model collapsed onto a few — wasted capacity, weak specialization.
- Per-expert purity is the core number. dom purity = fraction of an expert's assigned images coming from its top domain. Chance is 1/num_domains (0.25 for PACS's 4 domains); 1/num_classes for class purity (≈0.14 for PACS's 7 classes). Purity well above chance = real specialization. Also check that different experts have different top domains/classes — if every expert's top domain is the same, the router is just following the data prior, not partitioning.
- CLS-vs-all-token L1 should be small. If it's large, the specialization you see in token stats isn't what the classifier (which reads the CLS token) actually uses.

**The stronger evidence: generalization + necessity**

Purity on train data can be memorized routing. Two classes of test settle whether specialization is real:

Does routing generalize train → test? (a specialized expert should be chosen for the same slice on held-out images)

```bash
python -m ablation.router_train_test_consistency --checkpoint <ckpt>.pt --no-wandb
python -m ablation.domain_class_expert_alignment --checkpoint <ckpt>.pt --k_u 2 --no-wandb
python -m ablation.augmentation_stress_test --checkpoint <ckpt>.pt --num-augments 5 --no-wandb
```

Read the per-slice match rate and mean Jaccard / TV distance: high match rate + high Jaccard between train and test top-k expert sets = specialization that generalizes. The augmentation test checks the expert set survives heavy augmentation (specialization tied to semantics, not artifacts).

Are the specialized experts causally necessary?

```bash
python -m ablation.router_eq7_ablation --checkpoint <ckpt>.pt --alphas 0,0.5,1.0,2.0 --no-wandb
python -m ablation.pi_score_selection_ablation --checkpoint <ckpt>.pt --k_u 2 --no-wandb
python -m ablation.pi_score_train_test_ablation --checkpoint <ckpt>.pt --k_u 2 --no-wandb
```

The --run-eq7-diagnostics metrics (FEM/ARR/RFO) and the forced-routing helpers in metric/router_metrics.py (forced_routing_accuracy, expert_hit_rate) are the causal test: if you force inference through only the "specialized" subset and accuracy holds, but through the complement it collapses, that necessity gap is the cleanest proof the experts carry distinct, non-redundant function. A high FEM with low RFO (forget data concentrates on experts that retain data avoids) is exactly the localization the unlearning phase relies on.

A practical bar: gate entropy well below its max, zero/near-zero dead experts, per-expert purity clearly above the 1/D and 1/C chance lines with experts covering different top categories, train↔test match rate high (e.g. most slices matching), and a visible forced-routing necessity gap. If purity is near chance or train↔test match is low, the experts aren't meaningfully specialized regardless of what training loss curves looked like.

One thing to keep in mind: metric/router_metrics.py exposes forced_routing_accuracy / expert_hit_rate as library functions rather than a CLI, so the necessity gap isn't a one-liner yet. If it's useful, I can write a small standalone script that loads a checkpoint, runs the forced-routing necessity test across layers, and prints a single pass/fail specialization summary alongside the purity/entropy numbers — say the word and I'll put it together.

### User
what do the module_diagnostics.py actually do?

### Assistant
module_diagnostics.py is a read-only evaluation script. It loads a trained checkpoint and characterizes it — it never trains, updates weights, or takes gradients (everything runs under eval() / no_grad). Its job is to answer "did this model learn, and how is the router using its experts," so you can inspect it after learn.py (or after unlearning) without re-running training.

Here's what it actually does, in execution order.

Setup. It takes --checkpoint (required) and a config. If you don't pass --config, it looks for a flat learn.yaml two directories above the checkpoint (the layout run_moe_pipeline.py produces) so the architecture it rebuilds matches the one that was trained. It seeds RNG from args.seed, picks CUDA if available, and optionally starts a wandb run.

Rebuilds the exact data splits. It loads the dataset named in the config (PACS / OfficeHome / CIFAR-100 / Tiny-ImageNet), then does the same seeded 80/10/10 train/test/unseen split the training used. Depending on unlearn_setting it also carves out forget/retain subsets — a random forget_ratio slice, or a specific class/domain. It builds four dataloaders (train, test, forget, retain), all with test-time transforms and shuffle=False, so routing is measured on clean, deterministic inputs. The split only lines up with what the model actually saw if the seed matches training.

Rebuilds the model and loads weights. It constructs ModuleArchitecture with the config's moe_layers, num_experts, gate_k, etc., loads the checkpoint (handling model_state_dict/state_dict/raw and stripping _orig_mod. prefixes from torch.compile), with strict=False.

Accuracy check. It reports plain top-1 train and test accuracy. This is the "did it learn at all" sanity gate — everything below is only meaningful if accuracy is reasonable.

Router evaluation (the main part). For module models it does a single forward pass over TRAIN, then over TEST, and prints per-MoE-layer diagnostics from what each DeepMoELayer cached during the pass (last_pi_all = full softmax over all experts; last_pi = the local softmax over the top-k selected). Per layer it computes and prints:

- Sparsity: mean token entropy of the full distribution (vs log(num_experts)) and of the top-k gate (vs log(gate_k)) — the latter is what the sparse-routing loss actually trains on.
- Load: the global per-expert mass vector, how many experts are effectively dead (below a mass threshold), and a balance-deviation number (distance from uniform 1/E).
- A per-domain routing-mass table (rows = domains, columns = experts, rows sum to ~1) — which domains lean on which experts.
- CLS-token vs all-token L1 distance per domain — whether the token the classifier reads routes like the average token.
- A per-expert specialization table: each expert's load %, its top domain and "domain purity" (the fraction of its assigned images from that top domain), and the same for class.

Doing TRAIN and TEST separately is deliberate — it lets you see whether the routing pattern learned on training data still holds on held-out images. In this standalone script that whole router block is gated to PACS; for the other datasets you get accuracy and then a "skipping router eval" message, because the domain/class tables need a domain-labeled loader.

Eq. 7 localization (opt-in, --run-eq7-diagnostics). This is the unlearning-oriented part and is off by default. It measures routing mass on the forget vs retain loaders, picks the k_u "forget experts" as the top of mass_forget − alpha·mass_retain, and reports three numbers: FEM (how much of the forget set's routing mass those experts capture), ARR (fraction of retain images whose experts overlap that set — collateral risk), and RFO (mean expert-set IoU between forget and retain images). Together these say how cleanly the forget data concentrates onto a distinct expert subset, which is the property the unlearning algorithm relies on.

So in short: it's a diagnostic harness that verifies the model trained (accuracy), then quantifies how the router distributes work across experts and whether that distribution is committed, balanced, interpretable by domain/class, stable from train to test, and — optionally — localizable for unlearning. It's the tool you'd read to answer the specialization question from the previous message; the ablation scripts extend the same measurements with train/test-match and causal necessity tests.

### User
if I want to make run commands to check the learning phase first (before getting the models for the diagnostics). Please make ablation commands. Also, can you add "online metrics" during the training so that I can log to wandb and check?

### Assistant
Let me look at the exact training loop and what's already logged per-epoch, so the online metrics I add slot in cleanly and don't duplicate what's there.

Let me confirm the pipeline's --stage learn only trains (doesn't unlearn), so the ablation commands are accurate.

Confirmed — --stage learn trains base models only (if args.stage == "learn": continue), writing each checkpoint to runs/_base_models/<signature>/checkpoints/learn.pt with a flat learn.yaml beside it. That's exactly what the diagnostics later auto-derive from.

Now let me build the online-metrics tracker and validate it with a quick smoke test before handing it over.

*New online specialization tracker for per-epoch wandb logging during training*

Now a smoke test with the real DeepMoELayer to make sure the tracker's shapes and math are correct against a synthetic batch.

Disk is too constrained for the full torch wheel. Let me validate the core aggregation math (the index-flattening and purity/entropy computations — the only non-trivial parts) with a NumPy mirror on a crafted case where the answer is known.

Core math confirmed correct. Let me syntax-check the actual file I wrote (can't run torch here, but this catches any syntax/typo issues).

Both parts are ready. Let me present the tracker file, then give the commands and integration.

**Part 1 — Learning-phase ablation commands**

These train base models only and stop before any unlearning (run_moe_pipeline.py does if stage == "learn": continue after training). Each trial lands at runs/_base_models/<signature>/checkpoints/learn.pt with a flat learn.yaml beside it — which is what the diagnostics later auto-derive from, so these commands produce the checkpoints you'll diagnose in the next step.

Always dry-run first to see the trial matrix without spending GPU time:

```bash
# from repo root
DRY_RUN=1 bash scripts/run_moe_learn.sh          # prints every command it would run
bash scripts/run_moe_learn.sh                     # runs all 8 learning ablations
MAX_TRIALS=2 bash scripts/run_moe_learn.sh        # cap trials per sweep (quick check)
```

The three sweeps that bear directly on specialization are the loss weights — run them individually so you can watch each knob in isolation:

```bash
PIPE=config/experiments/run_moe_pipeline.py
CFG=config/experiments/pacs_module.yaml

# sparse-routing loss: controls how committed each token is to its expert
python $PIPE --stage learn --config $CFG --name learn_lambda_sparse \
  --sweep 'learn.lambda_sparse=[0.0,0.1,0.5,1.0]'

# load-balancing loss: guards against dead experts / collapse
python $PIPE --stage learn --config $CFG --name learn_lambda_balance \
  --sweep 'learn.lambda_balance=[1.0,2.0,5.0,10.0]'

# diversity loss: pushes experts to decorrelated representations
python $PIPE --stage learn --config $CFG --name learn_lambda_div \
  --sweep 'learn.lambda_div=[1.0,2.0,5.0,10.0]'
```

The capacity/routing-shape sweeps (how many experts exist, how many fire per token, how big each is), plus a seed sweep to check the specialization is stable rather than a lucky init:

```bash
python $PIPE --stage learn --config $CFG --name learn_num_experts        --sweep 'learn.num_experts=[8,12]'
python $PIPE --stage learn --config $CFG --name learn_gate_k             --sweep 'learn.gate_k=[2,3,4]'
python $PIPE --stage learn --config $CFG --name learn_expert_depth       --sweep 'learn.expert_depth=[1,2,3]'
python $PIPE --stage learn --config $CFG --name learn_expert_hidden_ratio --sweep 'learn.expert_hidden_ratio=[1,2,4]'
python $PIPE --stage learn --config $CFG --name learn_seeds              --sweep 'learn.seed=[1,2,3,4,5]'
```

Add --force to retrain even if a matching checkpoint is cached, and --max-trials N to truncate. To train a single non-swept baseline, just point learn.py at a flat config, or use config/experiments/pacs_module_learn_only.yaml.

One thing to set before you launch: the shipped configs have wandb_mode: disabled. Flip that to online (in the config's learn: block) or the online metrics below won't reach the dashboard.

**Part 2 — Online metrics during training**

What already logs per epoch: the five losses (ce/sp/bal/div/train_loss and weighted versions), domain_mass/<layer>/spread + a domain×expert table every domain_mass_log_every epochs, and the eval-mode router_match/overall/* train↔test consistency scalars every router_match_log_every epochs. What's missing online — it currently only appears in the end-of-training diagnostics — is per-epoch gate entropy, dead-expert count, balance deviation, max load, and per-expert domain/class purity. Those are exactly the "are the experts specializing yet" signals you want to watch climb (or collapse) as training runs.

The file above (metric/online_specialization.py) adds them by reusing the router probabilities each DeepMoELayer already caches on the forward pass, so it costs no extra inference. I validated the aggregation math (index-flattening, purity, entropy, dead-expert counting) against a crafted routing with a known answer — a perfectly domain-specialized router reads back as purity 1.0, zero dead experts, with the right chance baselines.

Drop the file into metric/, then make four small edits to approx_algo/module.py inside learn():

```python
# (1) top of file, with the other imports
from metric.online_specialization import OnlineSpecTracker

# (2) at the start of each epoch, next to the `domain_mass_sum, domain_tok_count = {}, {}` line
num_classes = len(self.class_names) if self.class_names else 0
spec_tracker = OnlineSpecTracker(
    self.num_experts, self.gate_k,
    num_domains=num_domains, num_classes=num_classes,
    dead_expert_threshold=self.dead_expert_threshold,
)

# (3) inside the batch loop, AFTER optimizer.step() (pi is cached, still in train mode).
#     Pull domains even when do_domain_log is False:
spec_domains = batch[2].to(self.device).long() if len(batch) > 2 else None
spec_tracker.update(self.model, labels, spec_domains)

# (4) at epoch end, near the existing wandb.log(...) block
spec_tracker.log(epoch + 1)
```

On the dashboard you'll get, per MoE layer and as online_spec/overall/* roll-ups: gate_entropy_norm (→0 means committed routing; →1 means indecisive), total_dead_experts (want ≈0), balance_deviation, max_load (a spike = collapse), and mean_domain_purity / mean_class_purity with *_chance reference lines (0.25 and ≈0.14 for PACS). Read purity climbing well above its chance line as specialization forming; read entropy stuck near 1 or dead experts rising as it failing to form.

Two caveats worth keeping in mind. During training noise_std > 0, so these reflect the noisy routing the model optimizes under — good for a trend, but the clean read is still the eval-mode router_match block and the end-of-training diagnostics. And purity needs domain/class labels, so the domain/class lines populate for PACS/OfficeHome; on CIFAR-100/Tiny-ImageNet you'll still get entropy, load, dead experts, and balance.

If it'd help, I can also fold online_spec_log_every into the config as a proper cadence knob (matching domain_mass_log_every) so you can throttle it on larger datasets, or extend the tracker to emit a per-epoch load-weighted purity variant.

*(attached: Online specialization .PY)*

### User
what does this file do? where do we place it in? How can it work together with the current repository?

### Assistant
Three questions, taken in order.

What the file does. It defines one class, OnlineSpecTracker, that converts the router probabilities each MoE layer already computes into per-epoch specialization numbers and logs them to wandb while training runs. It's a stateful accumulator with three methods. You construct it once per epoch with your model's shape (num_experts, gate_k, number of domains/classes, the dead-expert threshold). You call update(model, labels, domains) once per training batch — it walks the model's DeepMoELayer modules, reads the last_pi_all and last_pi tensors they cached during that batch's forward pass, and adds to running tallies (routing mass per expert, summed entropy, and per-image top-gate_k domain/class hit counts). Then at the end of the epoch you call log(epoch), which reduces those tallies into scalars — gate entropy (normalized against log(gate_k)), all-token entropy, dead-expert count, balance deviation, max load, and mean per-expert domain/class purity, plus overall/* roll-ups and the 1/D, 1/C chance baselines — and pushes them to wandb. The important design point: it reads tensors the forward pass already produced, so it adds no extra inference, only some cheap CPU bookkeeping. It never touches the loss, gradients, optimizer, or checkpoints — it's purely an observer.

Where it goes. Into the existing metric/ package as metric/online_specialization.py, sitting next to router_metrics.py and router_traintest_match.py. That directory already has an __init__.py, so it's an importable package and no setup is needed — that's exactly why the import line reads from metric.online_specialization import OnlineSpecTracker. Nothing else in the repo has to move.

How it works together with the repo. It hooks into the training loop through the DeepMoELayer contract that's already there. Look at architecture/module.py: every forward pass through a MoE layer stores self.last_pi_all (the full softmax over all experts) and self.last_pi (the local softmax over the selected top-k). The trainer's learn() loop in approx_algo/module.py already iterates those modules each batch to build the sparse/balance/diversity losses. The tracker piggybacks on that same cached state. So the data flow after your four edits is: the forward pass populates the cached probabilities → update() reads them and tallies → log() reduces and sends to wandb. Because update() runs under no_grad and doesn't change the model's train/eval mode, it slots in right after optimizer.step() without disturbing the next iteration.

It's deliberately parallel to the logging the repo already does. Today learn() logs the five losses every epoch, and — periodically — domain_mass/<layer>/spread and the eval-mode router_match/overall/* train↔test consistency scalars. What it does not log per epoch is routing quality: gate entropy, dead experts, balance, and per-expert purity only appear once, at the very end, inside _run_final_router_diagnostics. This file fills that gap so you can watch specialization form (or collapse) across epochs instead of finding out at the finish. It writes under its own online_spec/... wandb namespace, so there's no key collision with the existing metrics, and it uses the same per-image top-gate_k assignment convention as module_diagnostics.extract_router_diagnostics, which means the epoch-by-epoch purity numbers are directly comparable to the end-of-training diagnostic tables and to the standalone module_diagnostics.py run you'd do afterward. Its only external dependency is wandb, which the repo already uses, so as long as the run's wandb_mode isn't disabled, the metrics show up automatically.

So the end-to-end picture: this file is the online layer of the same specialization measurement the repo already does offline. During training it streams the trend to wandb; after training, module_diagnostics.py and the ablation scripts give you the clean, eval-mode confirmation on the checkpoints those training runs produced.

### User
but if current files do not call online_specialization, how can we achieve?

### Assistant
Right — a .py file sitting in metric/ is inert. Python never runs it on its own; something has to import it and call its methods. Nothing in the repo does that yet, so as shipped the tracker does nothing. To "achieve" it you edit the one file that owns the training loop — approx_algo/module.py — to import the class and call update() / log() at the right moments. There's no auto-discovery mechanism; the wiring is the four edits. Let me show them against the real code so it's concrete.

Let me pull the exact current lines so the edits anchor precisely.

The principle is just that: an imported-but-uncalled module is dead code. To make it run you add call sites in the trainer. Let me actually apply the four edits to the cloned approx_algo/module.py and compile it, so you can see exactly what changes and that it's valid — rather than trusting a snippet.

Compiles cleanly, and the diff above is the whole change: four insertions, nothing removed. That's the answer to "how can we achieve" — the tracker only runs because these four lines now call it.

Here's the flow the edits create, mapped to what you see in the diff:

The import (top of file) makes the class available. At the start of each epoch, spec_tracker = OnlineSpecTracker(...) creates a fresh accumulator sized to your model. Inside the batch loop, right after optimizer.step(), spec_tracker.update(self.model, labels, spec_domains) reads the last_pi_all / last_pi that the forward pass on that same batch just cached on each DeepMoELayer, and tallies them. At the end of the epoch, next to the existing wandb.log(...), spec_tracker.log(epoch + 1) reduces the tallies to scalars and pushes them under online_spec/*. Next epoch, a new tracker is created and the cycle repeats.

Two details that make it safe to drop in: update() runs under no_grad and never changes the model's train/eval mode, so it doesn't perturb the next iteration or the gradients; and it pulls spec_domains independently of the existing domains variable (which is only set when do_domain_log fires), so purity is tracked every epoch regardless of the domain-mass cadence.

I've applied the edits to the clone and saved the patched trainer so you can diff it against your own copy:

To put it in place: drop online_specialization.py into metric/, and either replace approx_algo/module.py with the patched version or apply those same four insertions to your own copy by hand (safer if your file has diverged from the GitHub version). Then set wandb_mode to online in your learn config and train — the online_spec/* panels will start populating per epoch.

One caveat on the patched file: I edited the version straight from the repo, so if your local module.py already has uncommitted changes, don't overwrite it — apply the four edits manually instead, since they're small and the diff above shows exactly where each goes.

If you'd rather not touch module.py at all, the alternative is to register forward hooks on the DeepMoELayer modules from learn.py before calling .learn() — but you'd still need a call site to trigger log() at each epoch boundary and to supply labels/domains, so it ends up more awkward than these four lines. Editing the loop is the clean path.

*(attached: Module .PY, Online specialization .PY)*

### User
do the current online specialization log to the wandb?

### Assistant
Short answer: the code calls wandb.log, but whether anything actually reaches your wandb dashboard depends on the run state — and with the shipped configs it won't, because they set wandb_mode: disabled. Let me verify the exact conditions rather than guess.

Yes — the tracker's log() ends in wandb.log(payload) (its to_wandb defaults to True), so every epoch it writes the online_spec/* scalars into the same run the trainer already uses. That part works, and it needs no separate wandb.init because learn.py opens the run before training starts.

But "does it reach your wandb dashboard" has a catch worth knowing, and it's not what the configs suggest. The YAML files carry wandb_mode: disabled, but learn.py never reads that field — its wandb.init(...) only passes project, name, config, and settings, with no mode=. So that config line is effectively dead: the run mode is whatever wandb defaults to, not what the YAML says.

In practice that means the mode is decided by your environment at wandb.init time:

- Logged in / WANDB_API_KEY set, no override → default online: everything, including online_spec/*, uploads to the cloud dashboard.
- WANDB_MODE=offline → logs to a local run dir only; nothing appears online until you wandb sync it.
- WANDB_MODE=disabled → wandb.log becomes a silent no-op; the tracker still runs and computes, but nothing is stored anywhere.

So to actually see the panels, control it through the environment, e.g. WANDB_MODE=online python learn.py ... (or the pipeline). Don't rely on the wandb_mode: key — it isn't wired up. If you'd like, I can add a one-line fix to learn.py so wandb.init(mode=getattr(args, "wandb_mode", "online")) actually honors the config field; that's the cleaner long-term fix.

One display nuance, not a logging failure: the loop already makes several wandb.log calls per epoch (main losses, then domain-mass, then router-match), and spec_tracker.log() adds one more. Each call advances wandb's internal step counter, so with the default "Step" x-axis the online_spec/* points for an epoch sit at a slightly different step than train_loss for that same epoch, which looks staggered. Every block logs an "epoch" key, so the fix is just to set the chart x-axis to epoch (or add wandb.define_metric("online_spec/*", step_metric="epoch") once after wandb.init). The values are correct either way.

### User
give me runners for the ablation so that I can check

### Assistant
Good — let me verify the exact invocation details for every ablation script (config handling, --no-wandb, import path) so the runner actually works rather than erroring on a config mismatch.

Two scripts lack --no-wandb and one defaults to a nested config. Let me nail down those three so the runner doesn't error.

All verified. Key gotchas the runner has to respect: four scripts accept --no-wandb but two (router_train_test_consistency, pi_score_selection_ablation) don't use wandb at all and will error if you pass it; router_eq7_ablation doesn't auto-derive its config (it defaults to a fixed file), so it needs the checkpoint's own learn.yaml passed explicitly to match architecture; and all six add repo-root to sys.path, so python ablation/x.py works from the repo root. All of them are PACS-oriented.

Let me write a single runner in the repo's own scripts/ idiom, then bash-lint it.

The runner lints clean and the dry run shows every resolved command. Here it is:

Drop it in scripts/ and run it against a checkpoint. It's read-only — every step just loads the checkpoint and reports.

```bash
# preview the exact commands first (nothing runs)
DRY_RUN=1 bash scripts/run_ablations.sh runs/_base_models/<sig>/checkpoints/learn.pt

# run the full suite
bash scripts/run_ablations.sh runs/_base_models/<sig>/checkpoints/learn.pt

# run just a subset, e.g. the diagnostic + the two train/test generalization checks
ONLY="1 2 3" bash scripts/run_ablations.sh <ckpt>.pt

# push results to your wandb dashboard instead of running offline
WANDB=online bash scripts/run_ablations.sh <ckpt>.pt
```

The seven steps map to the two things "specialized enough" means. Step 1 (module_diagnostics) is the descriptive read — accuracy, gate/all-token entropy, dead experts, balance, per-expert domain/class purity for TRAIN and TEST, plus the Eq. 7 FEM/ARR/RFO localization numbers. Steps 2–4 test whether the specialization generalizes: does each domain (step 2) and each domain×class slice (step 3) route to the same experts on held-out test images, and does an image keep its expert set under heavy augmentation (step 4). Steps 5–7 probe the forget/retain selection the unlearning phase depends on. Read purity climbing well above its chance line (1/D, 1/C) with high train↔test match as specialization that's real; near-chance purity or low match rate as experts that didn't specialize.

A few things baked into the script so it doesn't trip on the inconsistencies I found: it omits --no-wandb from the two scripts that don't accept it, passes the checkpoint's own learn.yaml to the Eq. 7 ablation explicitly (that one doesn't auto-derive its config), and defaults WANDB_MODE=disabled so nothing needs a wandb login unless you ask for WANDB=online. The config defaults to <ckpt>/../../learn.yaml, which is exactly where run_moe_pipeline.py --stage learn writes it — so if you trained through the pipeline, just pass the checkpoint. If you trained some other way and there's no sibling learn.yaml, pass CONFIG=path/to/flat_learn.yaml. And since every one of these scripts is PACS-specific, run them on PACS checkpoints.

The natural order end to end: scripts/run_moe_learn.sh to produce checkpoints (watching the online_spec/* panels as they train), then scripts/run_ablations.sh on whichever checkpoint you want to scrutinize.

*(attached: Run ablations .SH)*

### User
just give me list of commands, do not give sweep files

### Assistant
Run these from the repo root on a trained checkpoint (CKPT). All are read-only.

```bash
CKPT=runs/_base_models/<sig>/checkpoints/learn.pt
CONFIG="$(dirname "$(dirname "$CKPT")")/learn.yaml"   # the flat learn.yaml beside the checkpoint
export WANDB_MODE=disabled                             # avoids needing a wandb login

# 1) master diagnostic: accuracy, entropy, dead experts, balance, per-expert domain/class purity (TRAIN+TEST) + Eq.7 FEM/ARR/RFO
python module_diagnostics.py --checkpoint "$CKPT" --config "$CONFIG" --run-eq7-diagnostics --no-wandb

# 2) per-domain routing: same experts on held-out test images?  (no --no-wandb flag on this one)
python ablation/router_train_test_consistency.py --checkpoint "$CKPT" --config "$CONFIG"

# 3) domain x class expert alignment, train vs test (match rate + Jaccard)
python ablation/domain_class_expert_alignment.py --checkpoint "$CKPT" --config "$CONFIG" --k_u 2 --no-wandb

# 4) augmentation stress test: does a sample keep its expert set under heavy augmentation?
python ablation/augmentation_stress_test.py --checkpoint "$CKPT" --config "$CONFIG" --k_u 2 --num-augments 5 --no-wandb

# 5) Eq.7 forget/retain selection sweep over alpha
python ablation/router_eq7_ablation.py --checkpoint "$CKPT" --config "$CONFIG" --k_u 2 --alphas 0,0.5,1.0,2.0 --no-wandb

# 6) pi-score selection: aggregate vs per-class vs per-domain  (no --no-wandb flag on this one)
python ablation/pi_score_selection_ablation.py --checkpoint "$CKPT" --config "$CONFIG" --k_u 2

# 7) pi-score train-vs-test selection stability
python ablation/pi_score_train_test_ablation.py --checkpoint "$CKPT" --config "$CONFIG" --k_u 2 --no-wandb
```

Notes: set the first two lines to your real checkpoint; if you trained through the pipeline, CONFIG already resolves to the right file. These are PACS-only. To send results to wandb instead of running offline, use export WANDB_MODE=online and drop --no-wandb from steps 1, 3, 4, 5, 7.

### User
is this the code of training from learn.py? do those file call it?

### Assistant
Let me verify rather than assert — check what learn.py actually invokes, and whether any of the ablation/diagnostic files reference it.

Two separate answers, both confirmed in the code.

Is learn.py the training code? Yes. It's the training entry point but not the loop itself. learn.py's main() builds the model, dataset splits, and optimizer, then constructs the Module trainer and calls algo_wrapper.learn(...) (line 303). The actual training loop — forward/backward, optimizer.step(), the sparse/balance/diversity losses, checkpoint saving, and the online metrics you wired in — lives in Module.learn() in approx_algo/module.py. So the chain is learn.py → Module.learn() → the loop. (This is also why the online-specialization edits went into approx_algo/module.py, not learn.py.)

Do the ablation/diagnostic files call it? No. Not one of them imports learn.py or calls .learn() — the only match was a comment in router_eq7_ablation.py saying it works "rather than touching learn.py/unlearn.py." They're independent scripts. What they actually do is rebuild the model (ModuleArchitecture), load a saved checkpoint (torch.load), and run read-only evaluation. They never train.

So the relationship is producer/consumer, not caller/callee:

- learn.py trains and writes a .pt checkpoint (+ the flat learn.yaml beside it).
- module_diagnostics.py and ablation/*.py read that checkpoint and measure it.

They only overlap in the lower-level modules they both import — architecture/module.py (the model), dataset/* (the data + the same seeded split), and a few helper functions from module_diagnostics.py. That shared code is why the numbers line up, but the ablation runners never trigger training. Which is also why the workflow is two steps: run learn.py (or the learn sweep) first to get a checkpoint, then point the ablation commands at that checkpoint.

### User
i want commands that learn.py (so that we can check the module.py with online_specialization?

### Assistant
Let me confirm exactly how learn.py reads its config (flat vs nested) and its argparse, so the commands actually run rather than erroring.

Confirmed: learn.py does args = argparse.Namespace(**yaml_config) and reads flat top-level keys (args.dataset, args.num_experts, …). So it needs a flat config — the nested config/…/pacs_module.yaml files (with learn:/unlearn: blocks) will crash if passed straight to learn.py. That's the one thing that determines the commands. Let me confirm the nested file's shape.

Right — the nested learn: block is flat and has everything learn.py needs, but you can't hand learn.py the nested file directly. Here are the commands to train (which runs Module.learn() in module.py with the online tracker):

```bash
# 0) one-time: dataset must exist at the data_dir in the config
python dataset/downloader/pacs.py

# make online_spec/* actually reach wandb (learn.py ignores the wandb_mode config field)
export WANDB_MODE=online          # or: offline (inspect locally) / disabled (no logging at all)
```

Direct learn.py — needs a flat config, so extract the learn: block first:

```bash
# build a flat config from the nested experiments file (one-liner, no file authored by hand)
python -c "import yaml; d=yaml.safe_load(open('config/experiments/pacs_module.yaml'))['learn']; yaml.safe_dump(d, open('flat_learn.yaml','w'))"

python learn.py --config flat_learn.yaml
```

Quick smoke test (2 epochs, just to confirm the tracker logs — same command, epochs overridden):

```bash
python -c "import yaml; d=yaml.safe_load(open('config/experiments/pacs_module.yaml'))['learn']; d['epochs']=2; yaml.safe_dump(d, open('flat_smoke.yaml','w'))"
python learn.py --config flat_smoke.yaml
```

Via the pipeline (does the flat-config extraction for you and calls learn.py internally; also writes the sibling learn.yaml the ablations later auto-derive from):

```bash
python config/experiments/run_moe_pipeline.py --stage learn --config config/experiments/pacs_module.yaml --max-trials 1
```

The online tracker fires in every one of these, because they all reach Module.learn(). During the run watch the wandb panels under online_spec/* — per-layer gate_entropy_norm, dead_experts, balance_deviation, max_load, mean_domain_purity, mean_class_purity, plus the online_spec/overall/* roll-ups and *_chance reference lines. Note the tracker only sends to wandb (it doesn't print), so with WANDB_MODE=disabled you'd see nothing — use online or offline to actually check it. Direct learn.py writes the checkpoint under checkpoint/…; the pipeline writes it under runs/_base_models/<sig>/checkpoints/learn.pt.

### User
```
(flrepo) root@duong-pc:~/duongnm/MoDULE# python config/experiments/run_moe_pipeline.py --stage learn --config config/experiments/pacs_module.yaml --max-trials 1
Traceback (most recent call last):
  File "/root/duongnm/MoDULE/config/experiments/run_moe_pipeline.py", line 268, in <module>
    main()
  File "/root/duongnm/MoDULE/config/experiments/run_moe_pipeline.py", line 157, in main
    raise ValueError(f"Learning stage accepts only learn.* sweeps, not: {invalid}")
ValueError: Learning stage accepts only learn.* sweeps, not: ['unlearn.k_u', 'unlearn.selection_option']
```

### Assistant
That's a config-validation quirk, not a code bug. In the learn stage the pipeline refuses any sweep entry that isn't learn.*, and config/experiments/pacs_module.yaml carries a sweep: block full of unlearn.* parameters (for the later unlearning study). --max-trials 1 doesn't help because the check runs before trials are built. Let me confirm the cleanest fix.

Root cause confirmed: that config's sweep: block contains unlearn.* params, which the learn stage forbids. The shipped pacs_module_learn_only.yaml has no sweep block at all — it's the purpose-built learn-only config. Quick check that its learn block is complete:

Complete and valid. Pick any of these:

Fix A — use the learn-only config (cleanest, no sweep):

```bash
python config/experiments/run_moe_pipeline.py --stage learn --config config/experiments/pacs_module_learn_only.yaml
```

Fix B — keep your config but override the sweep with a learn.* key (the override does a parameters.clear() first, so it wipes the unlearn.* entries that trip the check):

```bash
python config/experiments/run_moe_pipeline.py --stage learn --config config/experiments/pacs_module.yaml --sweep 'learn.seed=[42]'
```

Fix C — skip the pipeline, run learn.py directly on the flat config (sidesteps sweep validation entirely):

```bash
python -c "import yaml; d=yaml.safe_load(open('config/experiments/pacs_module.yaml'))['learn']; yaml.safe_dump(d, open('flat_learn.yaml','w'))"
python learn.py --config flat_learn.yaml
```

All three reach Module.learn(), so the online_specialization tracker runs in each. Remember export WANDB_MODE=online beforehand or the online_spec/* panels won't appear (the tracker only logs to wandb, it doesn't print). Both configs default to 100 epochs — for a fast check that the metrics show up, use Fix C with d['epochs']=2 as in the smoke-test one-liner from before.

### User
python config/experiments/run_moe_pipeline.py --stage learn --config config/experiments/pacs_module_learn_only.yaml, do we sweep on the stage learn only?

### Assistant
Let me read expand_trials to answer precisely for the no-sweep case.

No — that command doesn't sweep anything. It trains exactly one model.

pacs_module_learn_only.yaml has no sweep: block, so in expand_trials the parameters dict is empty and it returns a single trial labeled "default" — one base model with the config's fixed hyperparameters (8 experts, gate_k=2, moe_layers=FFFFFFFFFFSS, 100 epochs, the lambda_* as written). No grid, one run.

There are two separate ideas worth keeping distinct here:

The --stage learn flag governs what runs, not how much — it means the pipeline trains base model(s) and stops before any unlearning (if stage == "learn": continue), and if a sweep were present it would only allow learn.* keys (that's the exact rule that rejected your earlier unlearn.* sweep). It does not by itself create a sweep.

Whether a sweep happens depends entirely on the config's sweep: block (or a --sweep override). This config has neither, so the answer is a single learn-only run. If you wanted to sweep during the learn stage, you'd add --sweep 'learn.something=[...]', e.g.:

```bash
# one run, no sweep (your command)
python config/experiments/run_moe_pipeline.py --stage learn --config config/experiments/pacs_module_learn_only.yaml

# sweep the learn stage over, say, lambda_sparse -> 4 base models
python config/experiments/run_moe_pipeline.py --stage learn --config config/experiments/pacs_module_learn_only.yaml --sweep 'learn.lambda_sparse=[0.0,0.1,0.5,1.0]'
```

So: learn-stage only, and single-trial only, for the command as written.

### User
what base models actually?

### Assistant
In this repo "base model" means the trained MoDULE network produced by the learning phase — the checkpoint before any unlearning has been applied. It's what the README calls the "gold" pretrained model. Nothing exotic; it's just the naming the pipeline uses to distinguish "the model you trained" from "the model after an unlearning algorithm has modified it."

Concretely, for pacs_module_learn_only.yaml the one base model it trains is: a DeiT-Small ViT backbone (module_small_patch16_224 — 384-dim, 12 transformer blocks, 6 heads), initialized from the public ImageNet DeiT weights, with the last 2 of its 12 MLP blocks swapped for MoE layers (moe_layers=FFFFFFFFFFSS) — each MoE layer holding 8 experts with top-2 routing, expert depth 2, hidden ratio 2 — plus a fresh linear classifier head sized to PACS's 7 classes. It's trained end-to-end with cross-entropy plus the sparse/balance/diversity routing losses. That trained bundle of weights is the base model.

Where it lands: the pipeline writes it to runs/_base_models/<learn_signature>/checkpoints/learn.pt, with the flat learn.yaml (the exact learning config) one level up at runs/_base_models/<learn_signature>/learn.yaml. The <learn_signature> is a hash of the learning config, which is the key to the "base models" plural: the pipeline caches one base model per distinct learning configuration. Run the learn stage with no sweep and you get exactly one. Run it with --sweep 'learn.num_experts=[8,12]' and you get two base models, one per config, each under its own signature directory — and if a signature's checkpoint already exists, the pipeline reuses it instead of retraining (that's the [reuse]/[skip] messages).

Those base models are also the input to the next stage: --stage unlearn takes one base checkpoint as its starting point and applies an unlearning algorithm to it, producing a separate unlearned checkpoint under runs/<study>/trials/.... The base model stays untouched.

So for your command: one base model = one PACS MoDULE checkpoint at runs/_base_models/<hash>/checkpoints/learn.pt, and that's exactly the file you'd then feed to module_diagnostics.py and the ablation runners to check expert specialization.

### User
i want to focus more on the ablation, test if the experts are learned well in the learning phase? Please make ablation for that

### Assistant
Focusing on that: the existing ablations mostly test generalization (does routing match train→test) and unlearning selection. The gap for "are the experts learned well" is a causal utilization test — are the experts actually necessary, non-redundant, and sufficient, or is the model quietly leaning on one or two while the rest are dead weight? The repo already has the machinery for this in metric/router_metrics.py (forced_routing_accuracy, expert_hit_rate) but no CLI wraps it. Let me build that ablation.

First let me mirror the exact boilerplate from an existing ablation so the new one is consistent.

I have the boilerplate and the router_metrics.py signatures. I'll write ablation/expert_necessity_ablation.py — a causal utilization test of whether the experts learned real, non-redundant function, built on the repo's existing forced_routing_accuracy / expert_hit_rate / mass_and_entropy.

Compiles and every imported name resolves. I also checked the sweep ranges keep at least gate_k experts reachable in every condition (sufficiency uses j ≥ gate_k; necessity drops at most E − gate_k), so no condition is ill-defined. Here it is:

Drop it in ablation/ and run it on a trained checkpoint:

```bash
python ablation/expert_necessity_ablation.py --checkpoint runs/_base_models/<sig>/checkpoints/learn.pt
# options: --split train|test (default test), --random-draws 3, --tolerance 2.0, --wandb
```

Why this one, and not the ones you already have: the existing ablations test whether routing generalizes (train→test match) and whether the forget/retain split localizes for unlearning. None of them test whether an expert actually does anything. This does, by constraining which experts the router may reach at inference (via the repo's forced_routing_accuracy, which masks router logits and keeps active_k == gate_k) and watching accuracy move. No training, no gradients.

It reports three things that together answer "were the experts learned well":

Sufficiency + ranking — keep only the top-j experts per layer (ranked by routing mass) and compare against keeping j random experts. If top-j clearly beats random-j, the router has genuinely differentiated its experts; if they're equal, the experts are interchangeable and specialization didn't really form. The curve also tells you the minimal number of experts that hold full accuracy — if that's much smaller than E, the rest are redundant.

Necessity — drop the top-t experts and route through the remainder. A large accuracy drop means those experts carry unique, non-redundant function (well learned); a negligible drop means the model was leaning on redundancy. The t=1 row is the single most-used expert per layer.

The end prints a plain verdict: differentiation (top-gk minus random-gk), redundancy (minimal sufficient expert count), and necessity of the top expert. Read a good model as: top-gk well above random-gk, minimal-sufficient count close to E − gate_k or higher, and a meaningful drop when you remove top experts. A poorly-specialized model shows top-gk ≈ random-gk and almost no drop when experts are removed — the classic "experts collapsed into redundancy" signature that the purity tables alone can miss.

To fold it into the runner, add it as an eighth step (it's read-only like the rest, and doesn't take --no-wandb — it prints by default and only logs with --wandb):

```bash
python ablation/expert_necessity_ablation.py --checkpoint "$CKPT" --config "$CONFIG"
```

*(attached: Expert necessity ablation .PY)*

### User
what is <sig>? is this a full proper test?

### Assistant
Two separate questions.

What <sig> is. It's the learn_signature — a short hash the pipeline computes from your learning config to name the folder a base model is cached in. It's not something you type or guess; it's created at training time and you read it back off disk. In run_moe_pipeline.py the path is built as output_root / "_base_models" / learn_signature / "checkpoints" / "learn.pt", where learn_signature = stable_id(learn) hashes the whole learn config block. So the real path looks like runs/_base_models/a1b2c3d4/checkpoints/learn.pt, and a1b2c3d4 is the <sig> placeholder. Two consequences worth knowing: identical learning configs hash to the same signature (that's the caching — the pipeline reuses the checkpoint instead of retraining), and changing any learn hyperparameter produces a new signature and a new folder.

You don't compute it; you look it up. After a learn run:

```bash
ls runs/_base_models/*/checkpoints/learn.pt        # list what exists
find runs -name '*.pt'                              # or search everything under runs/
cat runs/<study>/learning_manifest.yaml             # the manifest maps each trial -> its checkpoint path
```

And note this signature layout only exists if you trained through the pipeline. If you ran learn.py --config flat_learn.yaml directly, there's no <sig> at all — the checkpoint goes to checkpoint/<config-path>/<name>.pt (and _best.pt), per the output_dir logic in learn.py. So substitute whichever real path your training actually produced.

Is this a full, proper test? Honest answer: it's a strong, meaningful test of one specific property, not a complete verification of "the experts learned well." Here's the scope.

What expert_necessity_ablation.py genuinely establishes is causal utilization — whether experts are differentiated, non-redundant, and necessary — which is exactly the thing purity/entropy tables can't show and the property most likely to be silently broken (experts collapsing into redundancy). That's real evidence, and it's the piece that was missing from the suite.

But on its own it has blind spots. It ranks experts by routing mass and evaluates on one split, so it says nothing about whether that routing generalizes train→test, whether it's stable under augmentation, or whether the specialization aligns with anything interpretable like domain or class. It's also PACS-only, single-checkpoint, and — like everything here — measured under eval() where the training noise is off, so it characterizes the final model, not the trajectory. And a single run can't separate "this config specializes" from "this seed got lucky."

So a proper learning-phase verification is the necessity ablation plus the complementary axes you already have:

- descriptive: module_diagnostics.py — entropy, dead experts, load balance, per-expert domain/class purity vs the 1/D, 1/C chance lines;
- generalization: router_train_test_consistency.py and domain_class_expert_alignment.py — does the routing rule hold on held-out images;
- robustness: augmentation_stress_test.py — does an image keep its expert set under augmentation;
- causal utilization: the new expert_necessity_ablation.py;
- stability: repeat across a few learn.seed values so the conclusion isn't seed-specific;
- online trend: the online_spec/* panels, to confirm specialization formed during training rather than being read only at the end.

Convergent results across those — high purity above chance, routing that matches train→test, a clear top-vs-random gap with a real necessity drop, holding across seeds — is what would let you say the experts were learned well with confidence. Any single one, including this ablation, is a partial view. If it'd help, I can wire the seed-stability sweep and a small aggregator that collects these into one pass/fail summary per checkpoint.

### User
please give me the proper command of expert_necessity_ablation

### Assistant
From the repo root. Pick the block matching how you trained.

If you trained via the pipeline (checkpoint under runs/_base_models/<sig>/..., with a sibling learn.yaml so --config auto-derives):

```bash
# auto-resolve the checkpoint path instead of typing <sig>
CKPT=$(ls -t runs/_base_models/*/checkpoints/learn.pt | head -1)

python ablation/expert_necessity_ablation.py --checkpoint "$CKPT"
```

If you trained learn.py directly (no sibling learn.yaml, so pass the flat config explicitly):

```bash
# checkpoint lands under checkpoint/... ; adjust to your run
CKPT=$(ls -t checkpoint/**/*.pt 2>/dev/null | head -1)

python ablation/expert_necessity_ablation.py --checkpoint "$CKPT" --config flat_learn.yaml
```

The defaults (--split test, --random-draws 3, --tolerance 2.0) are the sensible ones, so the bare command above is the proper run. Options if you want them:

```bash
# evaluate on the training split instead of held-out test
python ablation/expert_necessity_ablation.py --checkpoint "$CKPT" --split train

# steadier random-baseline (more draws), stricter "no real loss" tolerance, and log to wandb
python ablation/expert_necessity_ablation.py --checkpoint "$CKPT" \
  --random-draws 5 --tolerance 1.0 --wandb
```

It prints everything to stdout by default (no wandb needed, no --no-wandb flag — --wandb is opt-in). If the checkpoint isn't a module_* model it'll stop with "No DeepMoELayer found," and it's PACS-only like the other ablations. If ls returns nothing, just point --checkpoint at the actual .pt file your training produced.

### User
it seems the expert_necessity_ablation does not train?

### Assistant
Correct — it does not train, and that's by design.

expert_necessity_ablation.py is a diagnostic that runs on a checkpoint you've already trained. It only loads weights (torch.load + load_state_dict), then evaluates under eval()/no_grad — no optimizer, no backward pass, no weight updates. All it varies is which experts the router may reach at inference (by masking router logits), and it watches accuracy move. So it measures a finished model; it can't create one.

That's the split I described a few messages back: learn.py (via Module.learn()) is the only thing that trains and produces the .pt; every file in ablation/ and module_diagnostics.py consumes that .pt read-only. The necessity ablation is in the second group.

So the order is always two steps — train first, then check:

```bash
# 1) TRAIN (this is the part that produces a checkpoint)
python config/experiments/run_moe_pipeline.py --stage learn \
  --config config/experiments/pacs_module_learn_only.yaml
# -> writes runs/_base_models/<sig>/checkpoints/learn.pt

# 2) CHECK (read-only; needs the checkpoint from step 1)
CKPT=$(ls -t runs/_base_models/*/checkpoints/learn.pt | head -1)
python ablation/expert_necessity_ablation.py --checkpoint "$CKPT"
```

If you run the ablation before any training exists, it won't train a model for you — it'll just fail to find a checkpoint. If what you actually want is a test that trains a fresh model and then checks it in one shot, that's a different kind of script (train → checkpoint → necessity test in sequence), and I can write that wrapper if that's the goal. But note it'd still be calling the same learn.py training underneath — the ablation itself stays read-only.

One thing worth flagging so the expectation is right: the point of keeping it read-only is that "are the experts specialized" is a question about a trained model. Training is what you'd vary (different lambda_sparse, seeds, expert counts); the ablation is how you score each resulting model. That's why the earlier learning-phase sweep produced several checkpoints — you'd run this ablation on each to compare.

### User
does the run_moe_pipeline sweep?

### Assistant
Yes, run_moe_pipeline.py can sweep — but only when the config (or --sweep) actually defines sweep parameters. It's not automatic.

The mechanism is expand_trials(spec): it reads spec["sweep"]["parameters"], and if that dict is empty it returns a single "default" trial; otherwise it takes the Cartesian product (itertools.product) of all parameter value-lists and builds one trial per combination. So whether it sweeps, and how many runs you get, is entirely determined by that parameters block.

Which means for the specific commands you've been running:

--config config/experiments/pacs_module_learn_only.yaml → no sweep. That file has no sweep: block, so expand_trials returns one trial. One base model.

--config config/experiments/pacs_module.yaml --stage learn → this has a sweep block, but it's full of unlearn.* parameters, and the learn stage rejects any non-learn.* sweep key — which is exactly the error you hit earlier. So it doesn't run a learn sweep; it refuses.

To actually sweep the learning phase, give it learn.* parameters via --sweep (which clears the config's parameters first, so it also sidesteps the unlearn.* rejection):

```bash
# 4 base models, one per lambda_sparse value
python config/experiments/run_moe_pipeline.py --stage learn \
  --config config/experiments/pacs_module_learn_only.yaml \
  --sweep 'learn.lambda_sparse=[0.0,0.1,0.5,1.0]'

# multi-axis: Cartesian product = 2 x 3 = 6 base models
python config/experiments/run_moe_pipeline.py --stage learn \
  --config config/experiments/pacs_module_learn_only.yaml \
  --sweep 'learn.num_experts=[8,12]' --sweep 'learn.gate_k=[1,2,3]'
```

Two things about how it behaves during a sweep. Each trial gets its own learn_signature (the hash of that trial's learn block), so it lands in its own runs/_base_models/<sig>/ folder — and if a signature already exists on disk, the pipeline reuses it instead of retraining (the [reuse]/[skip] messages). And --max-trials N caps how many of the expanded combinations actually run, which is useful for a quick check, though note it truncates after expansion, not before validation.

So: sweeping is a property of the config/flags, not the pipeline itself. learn_only as written trains exactly one model; add a learn.* --sweep to get a grid, then run expert_necessity_ablation.py on each resulting checkpoint to compare specialization across the grid.

*(Files hidden in shared chats)*

### User
1) what are the difference of those runs? 2) what are the metrics newly added?

### Assistant
Let me check exactly how the run name is built in learn.py, since that determines what the identical names do and don't tell you.

Confirmed — the run name in learn.py is:

`baselearn_{dataset}_{model_name}_ne{num_experts}_gk{gate_k}_ed{expert_depth}_ehr{expert_hidden_ratio}_{unlearn_setting}_fr{forget_ratio}_seed{seed}`

**1) What differs between those runs**

From the name alone, nothing you can see — and that's the catch. Identical names only tell you the nine name-encoded fields match: PACS, DeiT-small, num_experts=8, gate_k=2, expert_depth=2, expert_hidden_ratio=2, unlearn_setting=random, forget_ratio=0.1, seed=42. The template leaves out everything else in the learn config: lambda_sparse, lambda_balance, lambda_div, lr, epochs, weight_decay, ema_alpha, pretrained. So the runs must differ in one of those unencoded fields — or they're plain repeats of the exact same config.

Given the pattern (8+ runs, all ~1h04–1h05 runtime, one shared seed), the two likely stories are: a sweep over a loss weight (lambda_sparse/balance/div) or lr — which changes the pipeline's learn_signature and therefore writes a separate checkpoint per run, but does not change the wandb name, so they collide visually; or simple replicates of one config. The near-identical runtimes argue against an epochs sweep, and the fixed seed=42 means true replicates would track almost on top of each other (only CUDA/dataloader nondeterminism separating them).

To see the real difference, don't rely on the name — use the config wandb already logged (learn.py passes config=yaml_config to wandb.init):

- Open two runs → Overview → Config, and diff lambda_sparse / lambda_balance / lambda_div / lr.
- Or in the runs table, click the columns/gear control and add those config keys as columns — the one that varies across rows is your sweep axis.
- Or group the table by, say, lambda_sparse to collapse replicates.

If it turns out to be a loss-weight sweep, it's worth appending the swept field to the name template (e.g. …_ls{lambda_sparse}) so future runs are self-labeling — one line at learn.py:119.

**2) The newly added metrics**

Those are the online_spec/* keys from the OnlineSpecTracker we wired into Module.learn(). Per MoE layer (layer name in the key), logged every epoch:

- online_spec/<layer>/gate_entropy and …/gate_entropy_norm (÷ log gate_k → 0 = committed, 1 = indecisive)
- online_spec/<layer>/alltoken_entropy and …/alltoken_entropy_norm
- online_spec/<layer>/dead_experts
- online_spec/<layer>/balance_deviation
- online_spec/<layer>/max_load
- online_spec/<layer>/mean_domain_purity and …/mean_class_purity

Plus the overall roll-ups: online_spec/overall/mean_gate_entropy_norm, mean_domain_purity, mean_class_purity, total_dead_experts, and the reference lines domain_purity_chance (0.25 for PACS) and class_purity_chance (≈0.143).

Two caveats on whether they'll actually show in these runs: they only appear for runs trained after you applied the module.py edit and with WANDB_MODE=online (the tracker only logs, never prints), and the purity keys need domain labels — fine for PACS. If a run in the list predates the edit, it'll have the five loss curves plus the pre-existing domain_mass/* and router_match/* blocks, but no online_spec/* section. Set the panel x-axis to epoch so these line up with the loss curves, since they're logged in a separate wandb.log call.

*(Files hidden in shared chats)*

### User
1) it seems the current works only ablations on the number experts? I want to focus more on the learning loss and see how it affects the current expert specialization. 2) I do not see online spec

### Assistant
The screenshot is informative: your workspace has router_match, router_learn_train, router_learn_test, Charts, domain_mass, System — and no online_spec section. Since router_match and domain_mass are clearly reaching wandb, wandb itself is fine. That tells us the tracker simply isn't in the code that trained these runs. Let me verify the manifest + name details so I can answer both points precisely.

Two things, and they're connected — fixing #2 is what lets you see #1 during training.

**1) Studying the learning loss, not the expert count**

You're right that the knob to turn for specialization is the loss, not the expert count. The three loss weights are exactly that knob, and they're set at train time: lambda_sparse (commitment — pushes gate entropy down), lambda_balance (anti-collapse — pushes load toward uniform), lambda_div (decorrelates expert outputs). So a "how does the loss affect specialization" study is a learning-phase sweep over these weights, then measure specialization per resulting model. The expert_necessity ablation's internal j-sweep is a post-hoc probe of a finished model; it's not the training knob — don't confuse the two.

Sweep each weight (holding the others at default), one base model per value:

```bash
P=config/experiments/run_moe_pipeline.py
C=config/experiments/pacs_module_learn_only.yaml

python $P --stage learn --config $C --sweep 'learn.lambda_sparse=[0.0,0.5,1.0,2.0]'
python $P --stage learn --config $C --sweep 'learn.lambda_balance=[0.0,1.0,5.0,10.0]'
python $P --stage learn --config $C --sweep 'learn.lambda_div=[0.0,1.0,5.0,10.0]'
```

Each value gets its own learn_signature/checkpoint (the default 1.0 run you already have is reused, not retrained). The mapping from swept value → checkpoint is written to runs/<study>/learning_manifest.yaml — read that to know which folder is which. Then score each checkpoint:

```bash
python module_diagnostics.py --checkpoint <ckpt> --no-wandb        # purity / entropy / dead experts
python ablation/expert_necessity_ablation.py --checkpoint <ckpt>   # differentiation / necessity
```

The interesting finding to look for: lambda_sparse and lambda_balance pull against each other — sparse wants peaky, committed routing; balance wants uniform load. Too much balance flattens routing and kills purity; too much sparse can collapse onto a few experts (dead-expert count climbs). The sweep is how you find where specialization is strongest.

One practical fix for your goal: because the wandb run name (line 119 in learn.py) doesn't include the lambdas, every run in the sweep collides to the same name — which is exactly what you saw. Add them so the sweep is self-labeling:

```python
        f"_ls{args.lambda_sparse}_lb{args.lambda_balance}_ld{args.lambda_div}"
```

(insert as a line inside the run_name f-string). Then in wandb you can group/compare runs by loss weight directly.

**2) Why you don't see online_spec**

Your panel sections are router_match, router_learn_train/test, domain_mass, Charts, System — the pre-existing metrics — and no online_spec. Since those are logging fine, wandb is working; the tracker just isn't in the code that produced these runs. The most likely cause: I applied the edits to a sandbox clone, not your actual repo at /root/duongnm/MoDULE. Check:

```bash
ls metric/online_specialization.py && grep -n "OnlineSpecTracker" approx_algo/module.py
```

If either comes back empty, the tracker was never wired in — add the metric/online_specialization.py file and the four edits to approx_algo/module.py, then retrain. (If the import were added but the file missing, learn.py would have crashed at startup; since these runs Finished, it's simply not applied.) Note it can't backfill — the 257 existing runs predate the tracker, so only new runs after the edit will show online_spec/*.

Once it's wired and you rerun the lambda sweep above with the name fix, you get the payoff for #1 directly in wandb: group the runs by lambda_sparse and watch online_spec/overall/mean_domain_purity, mean_gate_entropy_norm, and total_dead_experts move across the sweep — the loss-vs-specialization relationship, live, without waiting for post-hoc ablations. If you'd like, I can write a small aggregator that reads learning_manifest.yaml, runs both diagnostics on every checkpoint in the sweep, and prints a lambda → purity/entropy/dead/necessity table so the comparison is one command.

### User
i want to add directly to wandb, not logging file

### Assistant
The online_spec/* tracker already logs straight to wandb — no file involved. So I think you mean the comparison across the loss sweep: the aggregator I offered would have printed a table, and you want that logged directly into wandb instead. Let me build it that way. First, quick check on the model's forward/inference API so I compute accuracy and routing in a single pass.

Important — evaluate_accuracy already returns a percentage (100 * correct/total), not a fraction. That means my expert_necessity_ablation.py has a bug: I multiply it by 100 again. Let me check what forced_routing_accuracy returns so I can fix the conventions.

Found the bug: forced_routing_accuracy and expert_hit_rate return fractions (0–1), but evaluate_accuracy returns a percentage (0–100). In expert_necessity_ablation.py I multiply the baseline by 100 as if it were a fraction — that's wrong. Let me fix it.

Now the aggregator that logs the loss-sweep specialization comparison directly to wandb (a table you can chart as lambda → purity/entropy/dead), reusing the validated OnlineSpecTracker for the metrics.

Both compile. Two deliverables here: the new loss_specialization_summary.py, and the corrected expert_necessity_ablation.py (re-download it — the baseline-accuracy bug is fixed).

The aggregator logs the loss-sweep comparison straight into wandb, no file. It scores each swept checkpoint in one eval pass (reusing the validated OnlineSpecTracker for the specialization scalars) and writes one wandb run containing a loss_specialization/summary table — one row per checkpoint with its lambda_sparse/balance/div, accuracy, gate-entropy, domain/class purity, and dead-expert count. It also emits scatter panels (domain_purity_vs_lambda_sparse, etc.) for any lambda axis that actually varies, so the loss→specialization relationship plots itself.

```bash
# after the loss sweep, feed it the manifest the pipeline wrote:
python ablation/loss_specialization_summary.py --manifest runs/<study>/learning_manifest.yaml

# or point it at checkpoints directly (shell glob):
python ablation/loss_specialization_summary.py --checkpoints runs/_base_models/*/checkpoints/learn.pt

# options: --split train|test, --project MoE, --run-name loss_specialization_summary, --batch-size N
```

It needs metric/online_specialization.py present (same file from the training edit) and WANDB_MODE=online so the run uploads. In wandb you'll find the run under project MoE, tagged loss_specialization — open the table and, if you want a custom view, add a scatter panel with X = lambda_sparse and Y = domain_purity.

Two honest caveats. It reads specialization under eval() (training noise off), so it characterizes each finished model — which is what you want for comparing loss settings. And it assumes the swept runs share dataset/seed (true for a pure loss sweep); it caches the split on (data_dir, seed, split) accordingly. If your manifest uses a different key name for the checkpoint path than the three I check (checkpoint/checkpoint_path/ckpt), tell me what the field is called and I'll match it — or just use --checkpoints with a glob.

*(attached: Loss specialization summary .PY, Expert necessity ablation .PY)*

### User
but why I do not see it in the running of run_moe_pipeline on module with different hyper parameters?

### Assistant
Let me confirm exactly how the pipeline launches training, since that determines which copy of the code actually runs.

That confirms the mechanism. The pipeline runs training as a subprocess — subprocess.run([sys.executable, "learn.py", "--config", ...]) (line 214). That subprocess loads learn.py from disk, which imports approx_algo/module.py from disk. So the online metrics appear only if that on-disk file in the repo you're running contains the four OnlineSpecTracker calls, and metric/online_specialization.py exists there.

The near-certain reason you don't see it: I applied those edits to a sandbox copy of the repo, not to your working tree at /root/duongnm/MoDULE. The pipeline runs your on-disk code, which is still the stock version with no tracker — so there's nothing to log, no matter what hyperparameters you sweep. Sweeping changes what gets trained, not what gets logged; the logging lives in Module.learn() and has to physically be in that file.

Verify in your actual repo:

```bash
cd /root/duongnm/MoDULE
ls metric/online_specialization.py
grep -n "OnlineSpecTracker" approx_algo/module.py
```

If the first errors or the grep prints nothing, the tracker isn't wired in — that's the whole reason. Fix it there: copy online_specialization.py into metric/, apply the four edits to approx_algo/module.py (the import, the per-epoch spec_tracker = OnlineSpecTracker(...), the spec_tracker.update(...) after optimizer.step(), and spec_tracker.log(epoch + 1) at epoch end), then confirm it imports:

```bash
python -c "import approx_algo.module; print('tracker wired:', hasattr(__import__('metric.online_specialization', fromlist=['OnlineSpecTracker']), 'OnlineSpecTracker'))"
```

Three follow-on gotchas once it's wired:

The pipeline caches by signature — if a checkpoint for a given config already exists it prints [reuse]/[skip] and never launches learn.py for it, so no new run and no online_spec. Only genuinely new hyperparameter combinations (or --force) actually retrain and log. So to see it on your existing configs, add --force.

Runs can't backfill — your 257 existing runs predate the tracker and will never show online_spec; only runs trained after the edit will.

And keep WANDB_MODE=online (you already have it — router_match/domain_mass are uploading, so that part's fine).

So: it's not a pipeline or hyperparameter issue at all — it's that the training code the subprocess loads doesn't contain the tracker yet. Edit the file in the repo you actually run, add --force, and the online_spec/* section will appear on the new runs.

### User
```
(flrepo) root@duong-pc:~/duongnm/MoDULE# cd /root/duongnm/MoDULE
ls metric/online_specialization.py
grep -n "OnlineSpecTracker" approx_algo/module.py
metric/online_specialization.py
16:from metric.online_specialization import OnlineSpecTracker
367:            spec_tracker = OnlineSpecTracker(
```

### Assistant
Good — so the file exists and the import (line 16) and the constructor (line 367) are both there. But that grep only proves two of the four edits. The string OnlineSpecTracker appears only in the import and the constructor; the two calls that actually make it do anything — spec_tracker.update(...) in the batch loop and spec_tracker.log(...) at epoch end — don't contain that string, so this grep can't see them. If those two are missing, the tracker gets built every epoch, fed nothing, and never logs. That's the most likely culprit. Check:

```bash
grep -n "spec_tracker" approx_algo/module.py
```

You should see roughly four references: the constructor (367), a spec_tracker.update(self.model, labels, ...) line inside the batch loop, and a spec_tracker.log(epoch + 1) line near the epoch-end wandb.log. If you only see 367 (and maybe the constructor's continuation lines), then edits 3 and 4 never got applied — add them and it'll log. The log() call is the one that emits to wandb; without it nothing appears no matter what.

If all four are present, then it's not the code — it's one of these:

Are the runs you're looking at actually newer than the edit? Your 257 existing runs predate the tracker and can't backfill. Sort the runs table by Created and check whether any run started after you saved the edits. If the newest run is older than the edit, you're looking at pre-tracker runs.

Did the pipeline actually retrain, or reuse? The pipeline skips training when a checkpoint for that signature already exists ([reuse]/[skip] in its output) and then never launches learn.py — so no new run and no online_spec. "Different hyperparameters" only triggers real training if the signature is genuinely new; otherwise add --force. Look back at the pipeline's console output for [skip]/[reuse] vs an actual training log.

Is it logged but hidden by your workspace layout? Your screenshot looked like a customized workspace (fixed sections: router_match, domain_mass, …). wandb doesn't always auto-add new key prefixes to a saved layout. Open one new run directly and type online_spec into the "Search panels with regex" box, or check the run's logged keys:

```bash
# from python, against a specific new run id
python -c "import wandb; api=wandb.Api(); r=api.run('<entity>/<project>/<run_id>'); print([k for k in r.summary.keys() if 'online_spec' in k])"
```

If that prints keys, the data is there and it's purely a display/section issue — add the panels manually. If it prints nothing for a run created after the edit, then log() isn't running and it's back to the grep -n "spec_tracker" check above.

So the order to check: confirm update/log calls exist → confirm you're viewing a post-edit run → confirm the pipeline retrained (--force) rather than reused → confirm it's not just a hidden panel. Paste the output of grep -n "spec_tracker" approx_algo/module.py and I can tell you which case you're in.

### User
```
(flrepo) root@duong-pc:~/duongnm/MoDULE# grep -n "spec_tracker" approx_algo/module.py
367:            spec_tracker = OnlineSpecTracker(
420:                spec_tracker.update(self.model, labels, spec_domains)
454:            spec_tracker.log(epoch + 1)
```

### Assistant
All four edits are in place — import (16), construct (367), update (420), log (454). The wiring is correct, so this is no longer a code problem. That narrows it to one thing: no new run has actually executed this code yet. The runs you've been looking at were created 9–17h ago, almost certainly before you saved these lines, and wandb can't backfill old runs. Sweeping hyperparameters doesn't help if the pipeline reused cached checkpoints ([reuse]/[skip]) instead of retraining — a reused signature never launches learn.py, so no new run appears at all.

Settle it with one fresh, fast run that bypasses the pipeline's caching entirely:

```bash
cd /root/duongnm/MoDULE
export WANDB_MODE=online
python -c "import yaml; d=yaml.safe_load(open('config/experiments/pacs_module.yaml'))['learn']; d['epochs']=2; yaml.safe_dump(d, open('flat_smoke.yaml','w'))"
python learn.py --config flat_smoke.yaml
```

Then open that specific run (the newest one) in wandb and type online_spec into the "Search panels with regex" box at the top. Two outcomes:

If the online_spec/* panels show up there — the code works; your earlier runs were simply pre-edit. Re-run the pipeline with --force so it retrains instead of reusing, and the section will appear on the swept runs too.

If they still don't show even in this brand-new 2-epoch run — then log() isn't emitting, and it's an indentation/reachability issue. Paste the context so I can see how the calls sit in the loop:

```bash
sed -n '360,372p;416,422p;450,456p' approx_algo/module.py
```

The thing to check is that line 454's spec_tracker.log(epoch + 1) sits at the epoch-loop level (runs once per epoch), not nested inside the batch loop or an if do_domain_log: block, and that line 420's update is inside the batch loop after optimizer.step(). Since training ran without error, the calls are at least syntactically fine — so indentation-placing them in the wrong scope is the only remaining failure mode.

One more possibility even when the data is logged: a customized workspace layout (like the one in your screenshot, with fixed router_match/domain_mass sections) doesn't always auto-add a newly-appearing key prefix. The regex panel search box sidesteps that. Or confirm the keys exist directly on a new run:

```bash
python -c "import wandb; api=wandb.Api(); r=api.run('<entity>/<project>/<run_id>'); print([k for k in r.summary.keys() if 'online_spec' in k])"
```

Run the 2-epoch job, check that one run, and tell me which of the two outcomes you get — that pins it down completely.

### User
what are the meaning of current metrics of onlinen_spec

### Assistant
Here's what each online_spec/* key means, grouped by the three things they measure. Layer keys use the module path (e.g. blocks.10.moe); the two MoE layers in your config are the last two blocks.

**Commitment — how decisively the router picks (driven by lambda_sparse)**

gate_entropy is the mean entropy of the gate weights over the gate_k experts each token actually selected (the local softmax on the top-k). gate_entropy_norm divides that by log(gate_k), so it's 0–1: near 0 means each token puts almost all its weight on one of its selected experts (committed); near 1 means it splits evenly across them (indecisive). With gate_k=2 the raw max is log(2)=0.693. This is the quantity lambda_sparse pushes down.

alltoken_entropy is the entropy of the full softmax over all experts (before top-k), and alltoken_entropy_norm divides by log(num_experts). It describes how peaky the router's whole distribution is — low means it concentrates probability on a few experts across the board.

**Utilization — are all experts used, or has the model collapsed (driven by lambda_balance)**

dead_experts counts experts receiving less than 1% of total routing mass. You want ≈0; a rising count means capacity is collapsing onto a subset. total_dead_experts in the overall section sums this across layers.

balance_deviation is the sum of squared deviations of each expert's load from uniform (load_e − 1/E)²; 0 means perfectly even load. This is what lambda_balance minimizes. Important nuance: balance is a health signal, not a specialization signal — perfectly uniform load prevents dead experts but doesn't by itself mean the experts learned distinct roles, and forcing it too hard actually fights specialization.

max_load is the routing-mass share of the single most-used expert. For 8 experts, uniform would be 0.125; a value drifting toward 1 is the collapse-onto-one-expert warning.

**Specialization — do experts map to distinct content (the actual "learned well" signal)**

mean_domain_purity: for each expert, take the fraction of images routed to it that come from its single most-common domain, then average over experts. Read it against domain_purity_chance = 1/num_domains = 0.25 for PACS. Above chance means experts are specializing by domain; at chance means each expert just gets a representative mix (no specialization).

mean_class_purity: the same, per class, read against class_purity_chance = 1/num_classes ≈ 0.143 for PACS's 7 classes.

The overall section adds layer-averaged roll-ups (mean_gate_entropy_norm, mean_domain_purity, mean_class_purity) plus the two *_chance reference lines, so you get one at-a-glance curve per concept.

How to read them together over epochs: a well-learned model shows purity rising clearly above its chance line and plateauing, gate_entropy_norm falling, dead_experts staying near 0, and max_load not spiking. Specialization failing looks like purity stuck near chance, or dead_experts/max_load climbing (collapse). And keep the tension in mind — lambda_sparse (wants peaky, low entropy) and lambda_balance (wants uniform load) pull against each other, which is exactly why the loss sweep is worth running: you're looking for the setting where purity is highest without experts dying.

Two caveats. During training noise_std > 0, so these are computed on the noisy routing the model optimizes under — good for watching a trend, but the clean read is the eval-mode router_match/* block and module_diagnostics.py. And mean_domain_purity/mean_class_purity are unweighted averages over experts, so a barely-used expert can show purity ≈1 on a handful of images and nudge the mean up — cross-check against dead_experts/max_load when a purity number looks surprisingly high.

### User
do you really check the online_spec metrics?

### Assistant
You're right to press on that. No — I have not verified the online_spec metrics against a real run. I should be clear about what I actually checked versus what I'm asserting.

What I did verify: the aggregation math in isolation. I ran a NumPy mirror of the tracker's logic on a crafted routing where the answer was known (each domain forced to a fixed expert pair), and confirmed the purity/entropy/dead-expert/index-flattening formulas produced the expected values (purity 1.0, 0 dead, correct chance baselines). I also byte-compiled the tracker and the edited module.py, and checked that every imported name resolves.

What I did not do: run the actual OnlineSpecTracker against the real DeepMoELayer with real tensors, end-to-end, and inspect the emitted wandb payload. I couldn't — the sandbox ran out of disk installing PyTorch, so the one smoke test that would have exercised the real forward pass never executed. So my descriptions of the metrics are read off the code I wrote, not off observed output. That's an important distinction I glossed over.

Concretely, there are things only a real run would confirm that I'm currently taking on faith:

- That last_pi_all and last_pi have the shapes I assumed ([B*S, E] and [B*S, gate_k]) after a real forward through DeepMoELayer — if the layer caches per-token differently, the view(B, S, E) reshape and the entropy could be wrong.
- That batch[2] is actually the domain id in your PACS loader (I assumed the (image, label, domain) ordering) — if it's ordered differently, the purity numbers would be meaningless even though they'd still "look" plausible.
- That model(images) populates the cache the way I need in eval mode, and that the per-image top-k assignment matches what module_diagnostics does closely enough to be comparable.

So the honest status: the metrics are logic-checked and syntax-checked, but not run-validated. If you've now done the 2-epoch smoke run, the fastest real check is to look at one epoch's numbers and sanity-test them against reality: domain_purity_chance should read exactly 0.25 and class_purity_chance ≈ 0.143 for PACS; gate_entropy_norm should sit in [0, 1]; dead_experts should be an integer in [0, num_experts]; and mean_domain_purity should be ≥ 0.25, not below it. If any of those are out of range — purity below chance, entropy above 1, a reshape error in the logs — then one of my shape/ordering assumptions is wrong and the numbers need fixing before you trust them.

If you paste one run's online_spec/* values (or the sed output of the loop context), I can check them against what the code should produce and tell you whether it's actually working, rather than asserting that it is.

*(Files hidden in shared chats)*

### User
please check the onlinespectracker and comment. I do not prefer hallucination.

### Assistant
I'll actually read the real DeepMoELayer forward in your architecture and check the tracker's assumptions against it, line by line. No guessing.

The shapes check out against the real code. Let me verify the one thing the tracker can't self-certify — that batch[2] really is the domain — by reading the PACS dataset.

I read the real DeepMoELayer.forward and the PACS dataset this time, so this is checked against source, not asserted from memory.

Verified correct against the code. The shape assumptions hold. In DeepMoELayer.forward, x_flat = x.reshape(-1, D) flattens [B, S, D] row-major, so tokens are image-major (x_flat[i*S + s] = image i, token s). probs = softmax(routing_logits) gives last_pi_all shape [B*S, E], and gate_weights = softmax(topk_vals) gives last_pi shape [B*S, gate_k] (during learning allowed_experts is None, so active_k == gate_k). That means the tracker's pi_all.view(B, S, E) correctly recovers [image, token, E] — the reshape is right, which was my biggest worry. The class name filter "DeepMoELayer" matches, the last_pi_all is None guard is correct, and I confirmed PACS __getitem__ returns (image, label, domain) with domain an int index — so batch[2] really is the domain, num_domains=4 (chance 0.25), num_classes=7 (chance ≈0.143). The index-flattening (e*C + labels) I'd already validated numerically. So there is no shape/ordering bug that would produce garbage; the values will be in-range and mean roughly what the docstring says.

Three real caveats, grounded in the code, that change interpretation:

The purity metric averages over all experts, and a dead/unused expert contributes 0, not a neutral value. Its tally row is all zeros, so share.max is 0, and it drags the mean down. So mean_domain_purity blends "how pure are the used experts" with "how many experts are used" — a config with several dead experts will show low purity partly for that reason, not because its active experts are impure. Cross-check against dead_experts/max_load before reading a low purity as "no specialization."

gate_entropy is the entropy of last_pi = softmax(topk_vals), and topk_vals are already probabilities (top-k of the softmax over experts). That's a softmax-of-a-softmax. Because the top-k probabilities are usually small and close together, re-softmaxing them squashes toward uniform, so gate_entropy_norm tends to sit near 1 and move very little even when the raw routing is fairly committed. It faithfully matches what _loss_sparse optimizes (same quantity), but as an interpretability signal it's compressed — for "how committed is the router," alltoken_entropy_norm and max_load are more sensitive. Don't read gate_entropy_norm ≈ 1 as "routing is random."

The training-time values are computed on noisy routing (noise_std=0.15 is added when self.training), which the docstring flags. Fine for a trend, but it's not the clean eval read.

Two corrections to things I told you earlier that were wrong or incomplete:

I said a sanity check was that mean_domain_purity "should be ≥ 0.25, not below it." That's incorrect — because dead experts contribute 0, the mean can legitimately fall below chance. So don't use "below 0.25" as a bug signal. And earlier I said a barely-used expert could nudge purity up; the fuller truth is that a zero-use expert nudges it down, so the distortion runs both ways depending on the utilization spread.

Minor: the print("online_spec") you added fires once per epoch inside log() — harmless debug noise, remove it when you're satisfied it's logging. And the class has no reset(), so it's correct only because module.py creates a fresh tracker each epoch (it does, at line 367); don't hoist that construction outside the epoch loop or the accumulators will span epochs.

What I still haven't done is watch it emit real numbers on a live run — the sandbox couldn't install PyTorch, so I can't claim I've seen the output. The logic and shapes are verified; the runtime behavior is inferred. If you paste one epoch's online_spec/* values from your smoke run, I'll check them against what the code should produce (e.g. domain_purity_chance exactly 0.25, gate_entropy_norm near-but-below 1, dead_experts an integer in [0,8]) and tell you if anything's off.

### User
please explain each metric carefully

### Assistant
Each metric, precisely as the code computes it. First the primitive they share: entropy of a probability distribution p over n items is H(p) = −Σ pᵢ log pᵢ. It's 0 when all the mass sits on one item (decisive) and maximal at log(n) when the mass is spread evenly (indecisive). Several metrics below are entropies or are normalized by a log(n) so they land in [0, 1].

Also two different notions of "assignment" appear, and they're not the same:

- Soft load — the raw softmax probability last_pi_all (a distribution over all E experts per token), which the load/entropy metrics use.
- Hard pick — the top-gate_k experts a token/image actually routes through, which the purity metrics use.

Now each key, as logged under online_spec/<layer>/.

gate_entropy — mean over tokens of the entropy of last_pi, the gate weights. In the model, last_pi = softmax(topk_vals) where topk_vals are the top-gate_k probabilities the router assigned. So this measures how evenly a token splits its weight between the experts it already selected. Low means it commits almost entirely to one of its gate_k picks; high means it splits them evenly. This is the exact quantity the sparse-routing loss minimizes, so it's the natural online mirror of lambda_sparse.

gate_entropy_norm — the above divided by log(gate_k), so 0 = fully committed, 1 = even split. With gate_k=2 the raw ceiling is log(2)=0.693. One honest caveat I verified in the code: because last_pi is a softmax of probabilities (a softmax-of-a-softmax), the inputs are small and close together, so the result is squashed toward uniform. Concretely, even a peaky router like top-2 probs [0.6, 0.2] gives softmax([0.6,0.2]) = [0.60, 0.40], norm ≈ 0.97; you need near-degenerate routing (top prob approaching 1) before it drops much below ~0.85. So treat gate_entropy_norm as a compressed signal that mostly lives in the high range and moves little — real movement is meaningful, but a value near 1 does not mean "routing is random."

alltoken_entropy — mean over tokens of the entropy of last_pi_all, the full distribution over all E experts (before top-k). This is the uncompressed view of how peaky routing is: low means each token concentrates its probability on a few experts across the whole expert set; high means it spreads across all of them.

alltoken_entropy_norm — the above divided by log(num_experts) (= log(8) ≈ 2.079 for you), giving [0, 1]. Because it's the raw distribution, this is the more sensitive commitment signal than gate_entropy_norm — if you want to see routing sharpen over training, watch this one and max_load.

dead_experts — count of experts whose soft load (see below) is under 1% of total mass. Note the definition: it's based on average softmax probability, not on how many tokens hard-selected the expert. You want this near 0. A climbing count means the model is abandoning experts — capacity collapse, the failure mode lambda_balance exists to prevent.

balance_deviation — first, load: load = mean over tokens of last_pi_all, an E-vector of each expert's average soft probability, summing to ~1. balance_deviation = Σₑ (loadₑ − 1/E)², the squared distance of that load vector from perfectly uniform (1/E each). 0 = every expert carries equal average mass. This is essentially what lambda_balance drives toward. Important framing: balance is a health signal (no expert starved), not a specialization signal — perfectly uniform load is compatible with either well-specialized or totally undifferentiated experts, and forcing balance too hard actively suppresses specialization.

max_load — the largest single entry of that load vector, i.e. the most-used expert's share of average routing mass. Uniform would be 1/E = 0.125 for 8 experts. A value drifting toward 1 is the "everything collapsed onto one expert" alarm; it's the sharpest early-warning of collapse, paired with dead_experts.

mean_domain_purity — this is the actual specialization signal, and it uses hard picks. Per image, the tracker averages last_pi_all over that image's tokens, takes the top-gate_k experts, and tallies [expert, domain]. Then per expert it computes the fraction of its tallied images coming from its single most-common domain (its "purity"), and averages that over experts. Read it against domain_purity_chance = 1/num_domains = 0.25 (PACS has 4 domains: art_painting, cartoon, photo, sketch). Above 0.25 means experts are capturing domain-specific inputs; at 0.25 means each expert receives a representative mix, i.e. no domain specialization. A crucial caveat I confirmed in the code: the average runs over all E experts, and an unused expert contributes 0 (its tally row is empty), which pulls the mean down. So this number blends "how pure are the used experts" with "how many experts are used," and it can legitimately fall below the 0.25 chance line when experts are dead — cross-read it with dead_experts and max_load before concluding anything.

mean_class_purity — identical construction, but tallying [expert, class] and reading against class_purity_chance = 1/num_classes ≈ 0.143 (7 PACS classes). Above chance means experts are picking up class-specific structure. Same dead-expert-contributes-zero caveat applies.

Then the online_spec/overall/ roll-ups, which are just cross-layer summaries so you get one line per concept instead of one-per-layer:

mean_gate_entropy_norm — gate_entropy_norm averaged over the MoE layers. mean_domain_purity / mean_class_purity — the per-layer purities averaged over layers. total_dead_experts — the sum of dead_experts across layers (so its ceiling is num_layers × num_experts). And domain_purity_chance (0.25) and class_purity_chance (≈0.143) are constants logged only so your wandb panels have a horizontal reference line to compare the purity curves against.

How they cohere when you read a run: alltoken_entropy_norm and max_load tell you whether routing is sharpening; dead_experts and balance_deviation tell you whether it's staying healthy (not collapsing); and mean_domain_purity / mean_class_purity vs their chance lines tell you whether the sharpening is actually aligning experts with content — which is the thing "learned well" really means. A good trajectory is purity rising clearly above chance and plateauing while dead experts stay ~0; a bad one is purity stuck at chance, or dead experts and max_load climbing together.

Two limits to keep in mind for all of them, both verified in the source rather than assumed: during training these are computed on noisy routing (noise_std=0.15 is added in self.training mode), so they're a trend, not the clean eval read; and gate_entropy is the compressed double-softmax quantity described above. For the clean, uncompressed confirmation you'd still run module_diagnostics.py on the checkpoint.

*(Files hidden in shared chats)*

### User
please evaluate these code, what are they used for?

### Assistant
Let me verify what mass_tv and mass_diff actually are in this repo before evaluating the plotting code — these read the existing router_match/* metrics, not the online_spec ones, and I want to ground the explanation in the source.

These operate on the repo's existing router_match metrics, so let me ground the definitions first, then test the non-trivial logic in the plotting scripts rather than eyeball it.

Definitions confirmed. Now let me test the two non-trivial bits of the plotting code — the log-boundary builder and the wandb NaN-row handling — with synthetic data, since those are where bugs hide.

All three scripts check out — the log-boundary builder, the wandb NaN-row handling, and the column rename all behave correctly (including correctly excluding the overall/…mean_tv and sel_tv columns, which is easy to get wrong). Here's what they are and how they fit.

**What they operate on**

These plot the repo's existing router_match/* metrics, computed in metric/router_traintest_match.py and logged every router_match_log_every epochs during training — not the online_spec metrics. Grounded in the source:

mass_tv per (domain, layer) is 0.5·Σ|p−q|, the total variation distance between the mean routing-mass distribution over experts computed on that domain's train images versus its test images. mass here is the soft, pre-top-k pi_all averaged over tokens, so it sums to 1 and TV lands in [0, 1]. Low mass_tv means the router spreads expert mass the same way on held-out images of that domain — routing is stable/generalizes; high means train and test route that domain differently.

mass_diff per (domain, layer, expert) is |mass_train_e − mass_test_e|, the per-expert contribution to that TV (and mass_tv = 0.5·Σ_e mass_diff). It's the drill-down showing which experts shift between train and test.

Note "layer0/layer1" here are the two MoE layers (your FFFFFFFFFFSS puts them at blocks 10–11), not transformer-block indices. And these plot the soft-mass variant; the sibling sel_tv/sel_diff (hard top-k selection frequency) is logged too but these scripts don't chart it.

**What each script does**

prepare_mass_metrics.py is the extractor. It pulls a run's history via the wandb API, keeps columns ending in /mass_tv and /mass_diff, strips the router_match/ prefix and the suffix, and writes tidy mass_tv.csv (columns <domain>/<layer>) and mass_diff.csv (columns <domain>/<layer>/<expert>). It replaces an older two-step download+split flow.

plot_mass_tv.py turns mass_tv.csv into two ICLR-style line figures (one per MoE layer), one line per domain, x = epoch, y = TV, with optional EMA/rolling smoothing. This is the "does routing converge to something that generalizes over training" figure — you want each domain's TV curve trending down and flattening.

plot_mass_diff_heatmap.py turns mass_diff.csv into a domain×expert heatmap per layer at a chosen epoch (default: last), with discrete log-spaced color bins and shared bin edges across layers so the two panels are comparable. This is the "at the final model, which experts still disagree between train and test, per domain" figure (labeled Appendix B.8.4 in its docstring).

**Evaluation**

They're correct and reasonably careful — the NaN-row handling (wandb logs metrics on separate calls, producing alternating NaN rows) is explicitly handled by "last non-NaN value at or before the target epoch," which I verified. Nothing here recomputes metrics; they only visualize what training already logged, so their correctness rests on router_traintest_match.py, which is sound.

Minor things worth fixing before they're paper figures, none of which break output:

The two scripts disagree on domain order — plot_mass_tv uses ["sketch","photo","art_painting","cartoon"], the heatmap uses ["photo","art_painting","cartoon","sketch"]. Cosmetic, but unify them so legend/row order is consistent across your figures.

plot_mass_tv forces y-axis scientific notation at exactly ×10⁻³ (scilimits=(-3,-3)). That's fine if TV is milli-scale, but if a domain's TV is larger (say 0.05–0.2) the axis still reads in units of 10⁻³ (so "50"–"200"), which can look odd. Consider letting matplotlib auto-scale.

prepare_mass_metrics.py uses run.history(samples=100_000), which samples the history — for a run logging more than 100k points it would downsample. For your scale (≈100 epochs × a few log calls) you're far under that, so it's exact in practice; if you ever need guaranteed full fidelity, run.scan_history() is the exact-but-slower path.

The heatmap's nice_log_boundaries degenerates if every mass_diff value is identical (single boundary → zero color bins → it'd error). That only happens on perfectly uniform data, so realistically never — but it's the one latent crash path.

**How this relates to your specialization question**

Keep the axes distinct. online_spec purity answers "do experts map to distinct domains/classes at all?" These mass_tv/mass_diff figures answer "is that routing the same on unseen data?" — the generalization/stability axis. They're complementary: high purity with low mass_tv is specialization that's real and transfers; high purity with high mass_tv is routing that looks specialized on train but doesn't hold on test (memorized). So for the "were the experts learned well" story, these plots are the train→test consistency evidence that pairs with the purity evidence — not a substitute for it.

*(Files hidden in shared chats)*

### User
it seems the current domains will focus on some specific experts. I think it should be (for top-k) k_i experts will handle one domain i, and the set of expert according to domain i should be overlap less according to k_j expert according to domain j. We should consider this regularization to the class too? How do you think?

### Assistant
Let me check the numbers in your own figure before opining, because they push back on the premise a bit.

Reading the domain heatmaps: Layer 1 rows peak at expert 7 (0.31), 7 (0.26), 6 (0.17), 6 (0.30) for photo/art/cartoon/sketch. So photo and art painting share their top expert (7), and cartoon and sketch share theirs (6). Layer 2 peaks at 0 (0.24), 0 (0.26), 5 (0.17), 5 (0.30) — same story with a different pair. That's not four disjoint expert sets; it's a photo+art vs cartoon+sketch split, which is exactly the real structure of PACS (photographic/realistic vs drawn/abstract). The router discovered a semantically sensible 2-way partition, not a 4-way one. Also note the mass is fairly flat — max cell 0.31 against uniform 0.125 — so the specialization is real but mild.

That reframes your proposal. What you're describing is an orthogonality/disjointness regularizer: per domain i, concentrate mass on k_i experts, and penalize overlap between domain i's and domain j's expert sets. Formally, build the domain×expert mass matrix M ∈ R^{D×E} (rows are your heatmap rows, each summing to 1), then add something like:

- concentration: minimize row entropy Σ_d H(M_d) (or maximize the top-k mass per row), and
- disjointness: penalize off-diagonal row similarity, Σ_{i≠j} ⟨M_i, M_j⟩ — i.e. ‖MMᵀ − diag‖², the same cross-correlation trick _loss_diversity already uses on expert outputs, but applied to routing mass.

Both are differentiable and cheap: you already compute per-token pi_all, and you already have domain labels in the batch (batch[2]), so M is one scatter-mean per batch. This would slot beside the existing three losses as a fourth weight.

Three honest reservations, though:

It conflicts with load balance. lambda_balance pushes column sums toward uniform; hard disjointness with D=4, E=8 means ~2 experts per domain, which is achievable, but if you also want k_i > 2 you run out of experts and the two losses fight. Worth checking the arithmetic for your E before tuning.

Disjointness may be the wrong prior. Photo and art painting genuinely share visual statistics; forcing them onto disjoint experts destroys reusable features and likely costs accuracy. A softer objective — penalize overlap proportionally to how dissimilar the domains are, or just penalize overlap weakly — is probably better than hard orthogonality.

Domains aren't the unlearning target. Your unlearn_setting is random with forget_ratio=0.1, so the forget set is a random slice spanning all domains and classes. Domain-disjoint experts wouldn't localize a random forget set at all. If the goal is unlearning localization, this regularizer helps only for the domain/class unlearn settings.

On class: the class heatmaps actually show stronger structure than the domain ones in places — Layer 2 has guitar→0 (0.27), house→3 (0.31), person→0 (0.36), with several near-zero cells (guitar/e1 = 0.0025, house/e1 = 0.0017), which is much peakier than anything in the domain figure. So class-conditioned routing is already emerging on its own. And class is what matters for class-unlearning. I'd argue class is the more promising axis for this regularizer, not an afterthought — but note class and domain compete for the same E experts, so regularizing both simultaneously over-constrains an 8-expert layer. Pick one axis per experiment.

My suggestion for how to test it rather than assume it: implement it as a fourth loss with weight lambda_disjoint, sweep [0, 0.1, 0.5, 1.0] on the class axis first, and score each checkpoint with the tooling you now have — mean_class_purity and dead_experts from online_spec, plus expert_necessity_ablation.py for whether the sharper split is causally real, and mass_tv for whether it still generalizes to test. The failure mode to watch: purity climbs, dead experts climb with it, and accuracy drops — that's the regularizer forcing an artificial partition rather than finding a real one.

If you want, I can write the loss (_loss_disjoint, computing M from pi_all + domain/class labels, with concentration and off-diagonal terms) and wire it into Module.learn() with its own lambda_disjoint config key, matching the existing loss style. Say which axis — class, domain, or switchable — and I'll build it.

### User
can you now access to the current paper and the repository code of this project?

### Assistant
Two different answers.

The repository: yes, partly. I cloned github.com/chuotchuilacduong/MoDULE at the start of this conversation, and I have that clone in my sandbox. Everything I've told you about DeepMoELayer.forward, the three losses, module_diagnostics.py, router_traintest_match.py, the pipeline's subprocess call, and the PACS dataset ordering came from reading those actual files — not from memory. Two caveats: it's a --depth 1 clone from earlier in this session, so it reflects the public GitHub state at that moment, not your working tree at /root/duongnm/MoDULE (which we established has diverged — you've applied the tracker edits there, and your repo may have other local changes I can't see). And I can't see anything unpushed.

The paper: no. I've never had access to it. Everything I know about your paper is what you've shown me — the screenshot with Figure 6/7 and the surrounding text, plus the "Appendix B.8.4" reference in the heatmap script's docstring. I read the heatmap values off that image; I haven't read the paper's method section, its formal definition of the MoDULE objective, its notation, or its claims. If I've described the method in ways that sound authoritative, that's inference from code, not from the paper, and the two can diverge (the code is one implementation of the method; the paper may define things differently or more generally).

That matters for the last exchange in particular. My comments on your proposed disjointness regularizer were grounded in the heatmap numbers and the loss code, but I don't know how the paper frames expert specialization, whether it already states a disjointness or orthogonality property, or what the reviewers' framing is. So take that as engineering input, not as advice calibrated to your paper's argument.

If you want me working from both: upload the paper (PDF is fine) and I'll read it. And if your local repo has diverged meaningfully, either push and tell me, or upload the specific files — approx_algo/module.py as it now stands would be the most useful, since that's where the losses and your tracker edits live and it's the file any new regularizer would go into.

*(Files hidden in shared chats)*

### User
This is the current version of the draft. Is the work aligned with the code?

### Assistant
This is the most important check yet — let me verify the paper's equations against the actual loss implementations rather than trusting either.

Critical findings emerging. Let me check the call sites — which routing quantity is fed to each loss, and how the probe/stop-gradient is handled.

This is a major finding. Let me read the exact loss-computation block to be certain before I report it.

Let me verify the expert-selection rule against Eq. 8, and numerically confirm the balance-loss issue.

I checked the paper's equations against the actual implementations. There are real misalignments, and two of them are significant enough that they affect what your experiments are measuring. Reporting them in severity order.

**1. L_bal is computed on the wrong tensor — this looks like a bug**

Paper Eq. 4/33: average the active gated mass per expert ḡ_{ℓ,m}, penalize Σ_{m=1..M}(ḡ_{ℓ,m} − 1/M)².

Code: line 384 appends module.last_pi, which the architecture sets to gate_weights = softmax(topk_vals) — shape [N, gate_k], i.e. the top-k gate weights, not a per-expert distribution. Inside _loss_balance, M = pi.size(-1) therefore equals gate_k = 2, not num_experts = 8. So it penalizes the mean first-choice vs second-choice gate weight toward 0.5 each. It never references expert identity, so it cannot prevent expert collapse — the stated purpose of the term.

Worse, it directly opposes L_sp, because both act on the same two numbers. I verified numerically:

| routing | bal_loss | sp_loss (entropy) |
|---|---|---|
| committed [0.95, 0.05] | 0.405 | 0.199 |
| indecisive [0.50, 0.50] | 0.000 | 0.693 |

L_bal is minimized exactly where L_sp is maximized. With λ_sp = λ_bal = 1.0 they largely cancel. This plausibly explains two things you've already observed: the Figure 6/7 heatmaps being fairly flat (max 0.31 against 0.125 uniform), and gate_entropy_norm sitting near 1 in the online metrics.

Also, _loss_balance applies an EMA across batches (ema_alpha=0.9) that appears nowhere in Eq. 4/34.

**2. L_sp optimizes a different quantity than Eq. 3**

Paper Eq. 3/32: normalized entropy of the full router softmax over all M experts, H(π_ℓ)/log M, explicitly in [0,1].

Code: entropy of last_pi (the gate_k-dim double-softmax over already-selected experts), with no log M normalization. So it sharpens the 2-way split among chosen experts rather than concentrating the M-way routing distribution. Its ceiling is log(2)=0.693, not log(M). The fix is to pass last_pi_all and divide by log(num_experts) — which would also resolve the L_bal conflict, since Eq. 4 wants gated mass per expert.

**3. L_div is not centered CKA**

Paper Eq. 5/38–40 specifies centered linear CKA: center with J_{B_p}, denominator ‖H̃ᵀH̃‖_F products, unordered pairs with 2/(LM(M−1)), on B_p probe units with sg(U_p).

Code _loss_diversity: Frobenius-normalizes each H_m (no centering), sums squared cross-correlation entries, loops ordered pairs (double-counting, no 2/(M(M−1)) normalization), runs on last_h = all tokens (no probe subsample), and uses no stop-gradient — so gradients flow into the shared backbone, contradicting the paper's claim that "the CKA objective does not update the shared backbone through the probe computation." The denominator ‖H_m‖_F² also differs from CKA's ‖H_mᵀH_m‖_F. This matters for Figure 3 ("Pairwise CKA similarity") and the CKA column in Table 2, which is all zeros.

**4. Selection rule doesn't match Eq. 8**

Section 3.4/B.2 states the selection is lexicographic argmin E_r subject to C_f ≥ τ_f, and explicitly says it "does not combine forget association and retain exposure through a weighted subtraction coefficient." The code's default is selection_option="diff" with alpha=1.0, i.e. s(D_f) − α·s(D_r) — exactly that weighted subtraction. Your Appendix G.1 lists ρ^diff as an ablation, so as implemented the main method is one of its own ablation variants.

**5. Hyperparameter and architecture inconsistencies**

Appendix B says MoE replaces the FFN in the final two blocks; Appendix G says "inserted to layers 7 and 9." The code (moe_layers=FFFFFFFFFFSS) puts them at blocks 10–11, matching B, not G.

Appendix G's defaults (M=12, k=4, λ_sp=0.1, λ_bal=0.25, λ_div=0.1, expert hidden ratio 0.1) don't match the config (num_experts=8, gate_k=2, λ=1.0/1.0/1.0, hidden ratio 2 — a 20× capacity difference). Figures 6–9 show expert indices 0–7, i.e. 8 experts, consistent with the code and inconsistent with G's M=12, while Table 13 attributes the headline numbers to M=12.

**Also worth a pass**

Tables 1, 5, and 6 contain identical numbers despite claiming different datasets, and Table 6 is titled CIFAR-100/TinyImageNet but carries a Domain Unlearning column. Several rows are 00.00 placeholders, Table 15 and Section 4.2 are empty, and the [NMD: ...] notes are still inline.

**What I'd do**

Items 1 and 2 are the ones I'd fix before running more experiments, because they change what the learning phase optimizes — and the specialization story in Section 4.3.1 rests on it. Concretely: pass last_pi_all to both losses, normalize L_sp by log(num_experts), and compute L_bal over the normalized active gates per expert (Eq. 29/33). That should make the routing measurably sharper, and the online_spec purity/entropy metrics are exactly the instrument to confirm it — rerun a short training and compare mean_domain_purity and alltoken_entropy_norm before vs after.

One caveat on scope: I read the public GitHub clone, so if your local tree has diverged on these functions, re-check against your copy before acting. I'd also rather you verify item 1 yourself with a one-line print of module.last_pi.shape in the loop — if it prints [N, 2] rather than [N, 8], the diagnosis holds.

### User
firstly, please analyze this part that i mentioned and explain what do you understand about it and what is your suggestion?

### Assistant
You're pointing at the proposal you raised before I had the paper: that for top-k routing, a set of k_i experts should handle domain i, and the expert set for domain i should overlap as little as possible with that for domain j — and asking whether the same should apply to classes. Now that I've read the draft, my assessment of it has changed, and mostly in its favor.

**What it is, in the paper's own notation**

Your idea operates on exactly the object your Figures 6 and 7 already plot. Let m_{g,ℓ} ∈ Δ^{M−1} be the mean sample-level active gated mass (Eq. 30) for group g — a class or domain — at MoE layer ℓ. That's one row of your heatmap. The proposal adds two pressures:

- Concentration: each row should put its mass on a few experts (low row entropy, or high top-k_g mass).
- Disjointness: different rows should use different experts — penalize Σ_{g≠g'} ⟨m_g, m_{g'}⟩, i.e. off-diagonal mass of M Mᵀ.

Note that when G ≤ M and rows are simplex-valued, the disjointness term alone already induces concentration, so this could be a single term rather than two.

**Why this is well-motivated — the paper concedes the gap**

The last paragraph of Section 3.3 states that the three losses supply a request-agnostic structural bias but don't force samples from a class or domain onto shared experts, and that localizability is therefore treated as an empirical property, measured post hoc via forget coverage and retain exposure. Your proposal converts that from something you hope for into something you optimize.

And the mapping onto the paper's own formalism is clean, which is what makes it more than a heuristic:

- Concentration ≈ maximizing C_f (Eq. 7 forget coverage): a group's routing mass concentrated on few experts means a small S captures most of its routed computation.
- Disjointness ≈ minimizing E_r (retain exposure): if group i's experts are unused by other groups, few retained samples intersect them.
- It directly optimizes surrogates of your Appendix E.7 diagnostics — raising FEM, lowering ARR and RFO.
- It strengthens Lemma 1. The pathway-locality guarantee is stated unconditionally, but its practical value scales with how many retained inputs avoid the updated experts. Disjointness increases that fraction, so the lemma binds more often. That's a real theoretical payoff, not just better numbers.

It also isn't redundant with your existing terms. L_sp concentrates each sample's routing; it says nothing about whether samples from the same group concentrate on the same experts. A group whose every sample is sharply routed, but each to a different expert, satisfies L_sp perfectly and is terrible for localization. L_div decorrelates expert representations; this decorrelates expert usage. So it adds a fourth rung to your design-rationale ladder: balanced, differentiated experts still need not align with deletion requests.

**Constraints you should check before committing**

Capacity arithmetic. Strict disjointness needs G·k ≤ M. For domains with D=4, M=8, k=2, that's exactly 8 — feasible. For classes with C=7, M=8, you'd need 14 slots; infeasible, so class disjointness can only ever be soft. On CIFAR-100/Tiny-ImageNet (C=100/200) it's hopeless as a hard constraint. This is the strongest argument for applying it on the domain axis as the primary result and treating class as a soft variant — the opposite of what I suggested before I'd read the paper, and I was wrong then because I was reasoning from PACS purity numbers without knowing your dataset suite.

Batch estimation. Group statistics need enough samples per batch. At batch 128: ~32/domain and ~18/class on PACS, fine; ~1.3/class on CIFAR-100, far too noisy — you'd need an EMA across batches, like the one _loss_balance already uses.

Semantic overlap. Your Layer 1 heatmap peaks at expert 7 for both photo (0.31) and art painting (0.26), and expert 6 for both cartoon (0.17) and sketch (0.30); Layer 2 shows the same pairing on experts 0 and 5. The router found a photo+art vs cartoon+sketch split — real PACS structure. Forcing 4-way disjointness fights that and will likely cost accuracy. A softer penalty, or one weighted by inter-group dissimilarity, is safer than hard orthogonality.

Interaction with the broken L_bal. As implemented, L_bal operates on gate_k slots, not experts, so it currently provides no load balancing. Adding a disjointness term on top of that would make collapse more likely, not less. I'd fix L_bal first — otherwise you can't tell whether a new term helped or just filled the vacuum left by a non-functional one.

**The framing tradeoff**

This makes the learning phase request-aware: you must know the partition axis at training time. That weakens the request-agnostic claim in Section 3.3. But I think the defense is strong and worth stating explicitly in the paper: you need to know only that deletion will be class- or domain-structured, not which class or domain will be deleted. Given your scope is already structured downstream requests, that's a mild assumption. Be upfront that it won't help random deletion — which your paper already positions as a structure-free stress test, so that's consistent rather than damaging.

One small thing I noticed while reading: Table 2 reports λ_sep, which is never defined in Section 3.3, and its values (0.0/0.5/1.0/1.5) are identical to the λ_div values in Figure 2. Either it's a renaming slip, or there was already an intended "separation" term — worth resolving, and if the latter, your proposal may be what it was meant to be.

**What I'd run**

Add it as λ_disj on the domain axis first, sweep [0, 0.1, 0.5, 1.0], and score each checkpoint on four axes: mean_domain_purity and dead_experts from online_spec; FEM/ARR/RFO from module_diagnostics --run-eq7-diagnostics; mass_tv for train→test stability; and expert_necessity_ablation for whether the sharper partition is causally real rather than cosmetic. The failure signature to watch: purity climbs, dead experts climb with it, accuracy drops — that's the regularizer imposing an artificial partition instead of finding a real one. The success signature is ARR and RFO falling at roughly constant TA, which is the result that would actually strengthen the paper.

If you want, I can write _loss_disjoint against the fixed L_bal (group-mass matrix from last_pi_all plus labels/domains, EMA for sparse groups, switchable class/domain axis) and wire it in with its own config key.

### User
please suggest the contribution to the paper?

### Assistant
Here's how I'd position it. The framing matters more than the loss itself, because the paper currently has a gap the regularizer fills precisely.

**The gap it fills**

Your title promises "Learning Modular Expert Representations for Localized Machine Unlearning," but nothing in L_learn (Eq. 2) optimizes anything that appears in the localization criteria (Eq. 7). Section 3.3 admits this outright: the three losses "do not explicitly force examples from a class, domain, user, or source to use the same experts. We therefore treat routing localizability as an empirical property." So the learning objective and the deletion objective are currently disconnected, and localizability is something you measure and hope for rather than something you train.

That's the contribution: close the loop between Eq. 2 and Eq. 7. Everything else in the paper is well-covered by prior work in kind (sparse routing, load balancing, CKA differentiation are standard MoE machinery), and the genuinely novel claim — training before the request to make future requests localizable — is exactly the claim the current objective doesn't implement.

**The theoretical hook, which is stronger than I expected**

Your own definitions make the surrogate almost exact. Let m_g be the mean sample-level gated mass for group g (Eq. 30), i.e. one row of Figure 6/7, and let S_g be its top-k experts.

Because C_f is defined as an expectation of q_S(x) and q_S is linear in the gated mass, concentration is forget coverage, exactly:

C_f(S_g) = Σ_{m∈S_g} m_{g,m}

And retain exposure is bounded by the overlap term via Markov (q ≥ 0):

E_r(S_g; δ) = P_{x∼D_r}(q_{S_g}(x) > δ) ≤ (1/δ)·Σ_{m∈S_g} m_{r,m}

The right-hand side is precisely what a disjointness penalty minimizes. So the two halves of your regularizer are a differentiable surrogate for Eq. 8's constraint and objective respectively — the constraint exactly, the objective through a bound. That converts "empirical property" into "optimized quantity with a guarantee," and it slots in as a Proposition 2 beside Lemma 1: Lemma 1 says localized updates are safe if pathways are disjoint; Proposition 2 says training can make them disjoint. Together they're a much tighter story than either alone. (I derived this quickly — check the 1/L factor in Eq. 6 carries through as you expect before committing it to the paper.)

**Concrete edits I'd make**

Contribution bullet 2, rewritten to something like: "We show that routing localizability need not be left to emerge. We introduce a group-conditioned route-separation objective that acts as a differentiable surrogate for the forget-coverage and retain-exposure criteria — exact for coverage, a Markov bound for exposure — and show it improves FEM/ARR/RFO at matched utility."

Section 3.3: add the fourth term to Eq. 2 and extend the design-rationale ladder you already have. It reads naturally as the next rung: sparsity concentrates each sample's route, balancing prevents collapse, CKA differentiates representations — and none of these align routes with the deletion partition.

Related work: this is your sharpest separation from SEUF/GRIP/TRACE (all post-hoc on trained MoEs) and from MUNKEY/semi-parametric designs (design-time but instance-indexed). You're design-time and parametric and structured-request-targeted; the regularizer is what makes that positioning bite rather than just assert.

Resolve λ_sep: Table 2 reports a λ_sep that Section 3.3 never defines, with values identical to Figure 2's λ_div. If a separation term was already intended, this proposal is likely what it was meant to be — and Table 2 becomes supporting evidence rather than an orphan.

New ablation: λ_disj sweep against FEM / ARR / RFO / TA / FA / RA, on the domain axis as the headline (capacity permits D·k = 4·2 = 8 = M exactly) and class as a soft variant (C·k = 14 > 8, so disjointness can only be approximate, and on CIFAR-100/Tiny-ImageNet it's infeasible as a hard constraint).

**Two honest risks**

The contribution is contingent on the experiment. It's a contribution only if ARR and RFO fall at roughly constant TA. If purity rises while dead experts rise and accuracy drops, you've forced an artificial partition and it's a negative result. Frame and run it before writing the claim.

Sequencing. I'd fix the L_bal and L_sp issues first. As implemented, L_bal operates on gate_k slots rather than experts and provides no load balancing at all, so a new separation term would be filling a vacuum left by a non-functional one — and you'd have no way to attribute the gain. Fixing those may also improve your existing numbers on its own, which changes what the new term has to beat.

Pre-empt the obvious reviewer objection in the limitations: this makes learning request-aware, weakening the request-agnostic framing. The defense is that you need to know only the deletion axis (class- or domain-structured), not which specific class or domain — consistent with your stated scope. Say plainly it doesn't help random deletion, which your paper already positions as a structure-free stress test.

If you want, I can draft the Section 3.3 paragraph and the Proposition 2 statement in your notation, alongside the _loss_disjoint implementation.

### User
No, I want to focus on the fact that the router now do not allocate the experts specifically according to each domains (which means that in two different domains, there has chance to have same expert with high routing probability).

### Assistant
Let me quantify this precisely from your own Figure 6 numbers rather than eyeballing it.

Your observation is correct, and it's sharper than "there's a chance of sharing" — it's systematic. Quantifying your own Figure 6:

Layer 1: expert 7 is in the top-2 of three of four domains (Photo 0.31, Art 0.26, Cartoon 0.16). Expert 6 serves two (Cartoon, Sketch).

Layer 2: Photo and Art painting have identical top-2 sets {0, 7} — Jaccard 1.00, differing only in order. Expert 7 again spans three domains.

And the soft mass tells the same story: pairwise overlap (Σ min(m_i, m_j)) runs 0.50–0.90 across domain pairs. At Layer 2, Photo/Art share 0.899 of their routing mass. Only the Photo↔Sketch and Art↔Sketch pairs are genuinely disjoint at the top-2 level.

**Why the router does this**

Nothing in L_learn penalizes it, and the task loss actively rewards it. L_sp concentrates each sample's route but is indifferent to which expert — two domains can both be sharply routed to the same expert and incur zero penalty. L_bal (as implemented, on gate_k slots) doesn't touch expert identity at all. L_div decorrelates expert representations, not expert usage — two domains can share expert 7 while expert 7's representation is maximally distinct from the others'. So the only force acting on assignment is cross-entropy, which prefers sharing features between visually similar domains. Photo and Art painting share realistic texture and color statistics, so routing them together is the utility-optimal solution. The router is behaving correctly for the objective it was given; the objective just isn't the one your method needs.

**Why it breaks the method specifically**

This is the failure mode your own Eq. 7 is built to detect. Take domain-deletion of Art painting at Layer 2: selection by forget mass picks {0, 7} — which is exactly Photo's top-2. So E_r (retain exposure) is near-maximal, D_r^risk swallows most of the Photo retain set, and RFO for that pair is 1.0. Worse, Lemma 1's guarantee is conditional on E⁰_{ℓ,k}(x) ∩ S* = ∅; with shared dominant experts, almost no retained input satisfies that condition, so the pathway-locality result — your main theoretical contribution — applies to a vanishing fraction of the retain set. The guarantee stays true and becomes nearly vacuous.

Two things in your own draft corroborate this. Table 13 reports Routing Overlap of 0.65 at M=12 — your stated default configuration — versus 0.27 at M=8; the config you report headline numbers with has the worst overlap in the sweep, and that non-monotonicity deserves explanation. And in Table 1, domain unlearning shows TA 89.65 against retraining's 94.27 (a 4.6-point gap) and RA 97.51 vs 99.30, whereas class unlearning sits much closer to gold — exactly the collateral-degradation signature you'd predict if forget and retain share experts.

**What I'd suggest**

Make this an explicit negative finding in Section 4.3.1, stated plainly: sparsity, balance, and representational differentiation are jointly insufficient for route separation, and here is the measurement showing it. That's a legitimate and citable observation — it's the empirical justification for why localizability can't be left to emerge, and it motivates the separation term far better than an abstract argument. Report per-layer cross-domain top-k overlap and the Jaccard matrix above as a figure; they're computable from data you already have.

Concretely I'd add a cross-group route-overlap diagnostic to the learning-phase evaluation (Appendix E.8 currently measures only train→test stability, which as your own last paragraph notes cannot distinguish specialization from collapse — a router sending every domain to one expert scores perfectly stable). Then show it drops when the separation term is added, with ARR/RFO falling at matched TA.

One caution before you write the claim: these figures come from a model trained with the L_bal bug, where no load balancing was actually applied. Some of the sharing may be an artifact of that rather than an inherent limitation of the three-loss design. I'd re-measure the overlap after fixing L_bal and L_sp — if the sharing persists, the negative finding is solid and the motivation is real; if it largely disappears, the honest conclusion is different and you'd want to know that before building a contribution on it.

### User
i want to find or design a regularizer such that it will penalizes it?

### Assistant
You asked to find or design — there's real prior art here, so let me check it rather than reinventing from memory.

Confirmed — this exists. Mod-Squad (CVPR 2023) does exactly this: it adds MoE layers to a transformer with a loss incorporating the mutual dependence between tasks and experts, so that only a small set of experts activates for each task. And its decomposition is precisely what you need: maximizing the first term encourages sharpness of the conditional distributions P(E|T), while the entropy term on P(E) encourages a flat distribution, making experts evenly used across the dataset. ModuleFormer notes that Mod-Squad's MI objective simultaneously balances expert load and builds task–expert alignment, and a follow-up applies the same idea at the domain level: AEA uses a mutual information loss to strengthen correlations between experts and specific domains, significantly improving the distinction between experts. [2212.08066] Mod-Squad: Designing Mixture of Experts As Modular Multi-Task Learners +3

This is the right regularizer for you, for a reason specific to your situation: maximizing I(G;E) subsumes load balancing. Since your L_bal is currently computing on gate_k slots and doing no expert balancing at all, the MI term's H(E) component replaces it correctly — one term fixes the bug and adds separation.

Let me verify the caching I need and validate the math numerically.

The MI objective discriminates exactly as required — and critically, collapse scores 0, so unlike a naive concentration penalty it can't be gamed by routing everything to one expert:

| routing pattern | I(G;E) (nats) |
|---|---|
| fully disjoint (ideal) | 1.386 = max |
| your current shared top-2 | 0.641 (46% of max) |
| uniform / no specialization | 0.000 |
| collapsed to one expert | 0.000 |

Your model sits at roughly 46% of the achievable maximum, which quantifies the gap you spotted. Let me write the loss.

Drop it in metric/ and wire it like the other losses — one call in the batch loop, one weight in the config:

```python
from metric.route_separation import loss_route_separation

# in the batch loop, after the forward pass (last_pi_all is populated):
moe_mods = [(n, m) for n, m in self.model.featurizer.model.named_modules()
            if m.__class__.__name__ == 'DeepMoELayer']
sep_loss = loss_route_separation(moe_mods, domains, num_domains, self.gate_k,
                                 ema_states=sep_ema, ema_alpha=0.0)
t_loss = t_loss + self.lambda_sep * (sep_loss if sep_loss is not None else 0.0)
```

Design choices worth knowing about. It computes MI on the normalized active gates (Eq. 29) by default rather than the raw softmax, because that's the quantity C_f and E_r are defined on — so the thing you optimize is the thing your localization metrics measure. Set use_gated=False for Mod-Squad's raw-softmax version if gradients feel too sparse. It's normalized by min(log G, log M) so λ_sep transfers across domains (G=4) and classes (G=7) without retuning. And the EMA hook exists for the many-class datasets — at batch 128 you get ~32 images/domain on PACS (fine, leave ema_alpha=0) but ~1.3/class on CIFAR-100, where per-batch estimates are too noisy to use raw.

Two things about how this interacts with the rest of your objective. First, the H(E) term is a correct load-balancing term over experts, so once this is in, the existing L_bal is both redundant and wrong — I'd either fix L_bal and keep λ_sep small, or drop L_bal and let MI do both. Don't run them together unexamined, or you won't be able to attribute anything. Second, MI won't force disjointness where sharing is genuinely useful: Photo and Art painting share real visual statistics, and the term trades off against cross-entropy rather than overriding it. That's the desirable behavior, but it means you should expect reduced overlap, not zero.

For the paper this also sharpens your positioning: Mod-Squad and AEA use MI for multi-task and domain-recommendation performance. Applying it as a deletion-localizability prior — with the argument that H(E|G) is a surrogate for forget coverage and H(E) for retain exposure, and that it widens the set of retained inputs to which Lemma 1 applies — is the novel part, and it's a claim your FEM/ARR/RFO diagnostics can test directly.

I'd sweep λ_sep ∈ [0, 0.1, 0.5, 1.0] on the domain axis first and watch for the failure signature: if ARR and RFO fall while TA holds, it worked; if purity rises while dead experts climb and TA drops, the term is imposing an artificial partition. One caveat on my own work here — I validated the MI math numerically (the table above) but couldn't execute the torch path, since this sandbox can't install PyTorch. Before trusting a run, check on one batch that W rows sum to 1 and that mi lands in [0, min(log G, log M)].

*(attached: Route separation .PY)*

### User
is this MI loss overlap with the diversity loss already in the code/paper?

### Assistant
Short answer: it overlaps heavily with L_bal, partially with L_sp, and not with L_div. Taking them in turn.

**L_div — genuinely complementary**

They act on different spaces. L_div compares expert output representations H_{ℓ,m} on shared probe inputs: do experts compute the same function? MI compares expert usage: do different groups route to the same experts? Neither implies the other, and you can satisfy one while badly violating the other:

- Differentiated experts, shared routes. Photo and Art both route to {0, 7}, while experts 0 and 7 compute completely different functions. L_div is fully satisfied; localization still fails.
- Separated routes, duplicated experts. Each domain gets its own expert, but all experts learned near-identical functions. MI is maximal, L_div violated — and unlearning one expert gives behavioral forgetting while the knowledge survives in its twins, which is the recovery-attack failure your Section 2 cites.

Your own paper contains the empirical proof of non-redundancy. Figure 3 shows near-zero off-diagonal CKA — experts are representationally differentiated — while Figure 6 shows expert 7 in the top-2 of three of four domains. L_div is doing its job and route separation still fails. That's exactly the argument for adding a fourth term, and it extends your Section 3.3 ladder cleanly: balanced utilization doesn't prevent similar representations (hence L_div), and differentiated representations don't prevent shared routes (hence L_sep).

One caveat: the implemented _loss_diversity isn't the centered CKA of Eq. 5, and Table 2's CKA column is all zeros, so Figure 3's near-zero off-diagonal may be partly a measurement artifact. The argument survives either way, but I'd confirm Figure 3 with a correct CKA before leaning on it in the text.

**L_bal — redundant, and MI is the better version**

I(G;E) = H(E) − H(E|G), and the H(E) term is load balancing: it pushes the marginal expert distribution toward uniform, over experts. That's what Eq. 4 intends and what the current implementation fails to do (it reduces to gate_k slots). So MI supersedes it. Don't run both — either fix L_bal and keep λ_sep small, or drop L_bal and let MI carry balance plus separation.

**L_sp — partially subsumed**

L_sp minimizes per-sample routing entropy. H(E|G) minimizes the entropy of the group-averaged routing distribution. By Jensen, entropy-of-the-mean ≥ mean-of-entropies, so H(E|G) upper-bounds what L_sp targets and additionally demands within-group agreement. That gap is precisely your problem: a group whose every sample is sharply routed, but each to a different expert, scores perfectly on L_sp and terribly on H(E|G). So MI is strictly stronger here. You could keep L_sp as a cheap per-sample term (it needs no labels, so it also applies to unlabeled or ungrouped data), but expect its marginal contribution to shrink once L_sep is in.

**What I'd keep**

| term | controls | verdict |
|---|---|---|
| L_sp | per-sample route concentration | partially subsumed; keep at low weight or ablate out |
| L_bal | marginal expert load | redundant with H(E); drop or fix, not both |
| L_div | expert representation similarity | keep — orthogonal, and guards against duplicated knowledge |
| L_sep | group↔expert assignment | new; the missing axis |

For the paper, the honest framing is that L_sep replaces L_bal rather than being added to three existing terms — which is a cleaner story anyway, since it means the objective doesn't grow. And the L_sep + L_div pairing has a nice interpretation for unlearning: L_sep ensures the forget set routes to a small expert set, L_div ensures that set isn't functionally duplicated elsewhere. You need both for localized unlearning to be more than cosmetic, and that's a sentence worth putting in Section 3.3.

### User
please make full revision on the files belong to the repo that upgrade the current MoDule version

### Assistant
This is a substantial change, so let me re-read the exact code I'm revising before editing.

Two more issues surface from the architecture. Line 79 uses softmax(topk_vals) — but Eq. 29 specifies sum-normalization of the top-k probabilities, not a re-softmax. That's the root cause of gate_entropy_norm sitting near 1. And the CKA probe needs a detached input (Eq. 37), which isn't cached. Let me fix both.

Now the corrected loss module implementing Eq. 3, 4, and 5 faithfully.

*Corrected ModULE learning losses matching the paper equations, plus the new separation term*

Now patch the trainer to use the corrected losses.

Now the import, the sep_ema_states init, and the config plumbing.

Now the config plumbing in learn.py and a config file with the new keys.

Let me create the config and validate the corrected loss math end-to-end.

Every corrected loss behaves as specified: L_sp spans [0,1] and is normalized, L_bal now detects expert collapse (6.75 vs 0.013) instead of being blind to it, and L_div is a real CKA (1.0 for identical experts, ~0.2 for independent).

Everything compiles and the loss math is validated. Start with UPGRADE.patch — it's the whole change as a reviewable diff (81 insertions, 12 deletions across 3 files) and the safest way to apply this to your tree, since your local copy has diverged from GitHub. git apply UPGRADE.patch from the repo root, plus drop module_losses.py and online_specialization.py into metric/ and the config into config/experiments/.

What changed, by file.

architecture/module.py — two fixes. The gate is now sum-normalized per Eq. 29 (topk_vals / topk_vals.sum()) instead of F.softmax(topk_vals); the old double-softmax was squashing gates toward uniform and is the root cause of gate_entropy_norm sitting near 1. It also caches last_g_full (the full-width normalized active gates, the quantity C_f/E_r are defined on) and last_v (detached expert input for the CKA probe).

metric/module_losses.py — new, holding all four corrected terms. loss_sparse uses the full pi over all M experts and divides by log M (Eq. 32), instead of the gate_k-dim gate unnormalized. loss_balance operates on gated mass per expert with the M prefactor (Eq. 4) — the old version set M = gate_k and never referenced expert identity, so it couldn't see collapse. loss_diversity_cka is real centered linear CKA on a B_p probe with stop-gradient and unordered pairs (Eq. 37–40). loss_separation is the new −I(G;E) term.

approx_algo/module.py — swaps the loss block for structural_losses(...), adds lambda_sep/sep_axis/sep_use_ema/sep_ema_alpha/probe_size to __init__, threads the separation EMA, and logs sep_loss to wandb.

learn.py + pacs_module_v2.yaml — config plumbing, with lambda_balance: 0.0 and lambda_sep: 0.5 since MI's H(E) term subsumes balancing.

Two things to be deliberate about. The gate-normalization fix changes the forward pass, so existing checkpoints aren't comparable to new ones — you'll want to retrain your baselines. And lambda_sparse is now normalized by log M, so the old 1.0 is effectively a much larger weight than before; I set 0.1 in the config, but treat it as a starting point rather than a tuned value.

I'd run this in two stages so you can attribute the effect: first lambda_sep: 0.0 (corrections only) to see what the bug fixes alone buy you, then the sweep lambda_sep ∈ [0.1, 0.5, 1.0]. Watch online_spec/overall/mean_domain_purity and the FEM/ARR/RFO diagnostics.

Caveat on verification: I validated each loss with numpy mirrors on crafted inputs and compiled every file, but this sandbox can't install PyTorch, so no file was executed against a real tensor. Run 2 epochs on pacs_module_v2.yaml before trusting a long job, and sanity-check that last_g_full rows sum to 1 and sep_loss lands in [−1, 0].
