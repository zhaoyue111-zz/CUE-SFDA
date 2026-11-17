# DWI Liver Cancer SFDA — Step 1–3 audit

## 1. Executive Summary

- The 101-case frozen VoxTell uncertainty file is the correct **training split**, and its `baseline_dice` was computed before adaptation. The diagnostic source freezes VoxTell and uses GT only after uncertainty calculation. Routing CSVs each contain the same 101 train IDs.
- The joint no-response gate gives the largest observed zero-shot High–Low Dice gap, **0.540** (High 0.631, Low 0.091); it also isolates 17 empty-prediction cases into Skip. Among A/B/C without Skip, Original CUE has the largest gap (0.464), followed by Image (0.456), then Joint (0.448). Joint uncertainty alone did not improve High–Low separation in these group statistics.
- The historical 0.344416 Dynamic-hard run did reach all 101 cases before its first refresh at iteration 552. Its sampling was **mixed across resumes**: the seeded 0.75 branch sampler ran through iteration 550; a resumed unseen-case sampler selected the two remaining Low cases at iterations 551–552. The post-refresh log also matches a restarted unseen-case coverage cycle. The run log does not record a Git SHA, so this is reconstructed from the exact logged sample sequence and the historical source snapshot, not from a run-stamped commit ID.
- Existing adapted per-case metrics are for 44 held-out validation/test cases, not the 101 train cases. Therefore, the requested train High/Low pre/post analysis cannot be computed without new inference. This analysis does **not** run that inference.
- The old `outputs/ablation/summary` C baseline row (Dice 0.283336) has no best iteration and represents iteration 0. For the C_joint comparison below, I use the saved best C_joint checkpoint in `outputs/response_routing/C_joint` and keep it explicitly in the historical response-routing protocol family. No `liver_cancer_mainline` values are mixed into these tables.

## 2. Step 1 — Coverage-aware Dynamic-hard

### Run and checkpoint facts

`outputs/ablation/G_dynamic_hard_final/resolved_config.json` specifies joint routing, High=59 / Low=42, `high_quality_sampling_probability=0.75`, `max_refresh_cycles=1`, and `stage2.mode=none`. `refresh_history.json` records one completed refresh at **iteration 552**, with **101/101** covered cases. `early_stopping.json` records best Dice **0.3444156** at iteration **1250**, and early stop at **1750**. The best checkpoint is `checkpoints/best_iteration_001250_dice_0.344416.pt`.

### Historical implementation and what the run trace shows

The saved output does not contain a run-time Git SHA. The parent snapshot `1f0fabf` predates the committed consensus implementation. Commit `ef005e5` contains both the first available dynamic/coverage implementation and the experiment artifacts; its `_sample_case_index` prefers an unseen case from `high_indices + low_indices` before falling back to the 0.75 branch sampler. The later `825af4f` removes that unseen preference, leaving `seen_cases_since_refresh` as coverage bookkeeping only. The source in `ef005e5` records the sampled primary case after `_train_step`, without testing `effective_patches`, loss, or whether the case changed model parameters; the partner case is sampled separately and is not added to coverage.

The run log and RNG replay show that the 0.344416 run was not coverage-prioritized from iteration 1:

1. From iterations 1–550, the primary case sequence exactly matches seed 42 with the ordinary Bernoulli `p_high=0.75` branch sampler, including the logged repeat at iterations 10 and 100. At iteration 550, that sequence has seen 99 distinct primary cases.
2. On resume from the iteration-550 checkpoint, the unseen-case behavior completes the set: the two missing cases are `0017405680.nii.gz` and `0015978449.nii.gz` (both Low), selected at iterations 551 and 552. Refresh cycle 1 then starts at 552.
3. Refresh clears the coverage set. The unseen-case sampler starts a new High+Low cycle; replay predicts the logged primary samples at iterations 560–650 exactly, and reaches full coverage by iteration 653. Since `max_refresh_cycles=1`, the refresh gate prevents another refresh. Once no unseen cases remain, sampling falls back to `p_high=0.75` for the rest of training.

So the exact observed protocol is a **resume-dependent hybrid**, not an unseen-first sampler for all 552 pre-refresh iterations. Before the resume, branch probability 0.75 applied. During forced coverage completion, it was bypassed; the final two missing cases happened to both be Low. Coverage is based on the primary sampled case, not on the partner. The historical source adds it regardless of effective-patch count.

### Sampler + refresh pseudocode reconstructed from the run

```text
seen = restored primary cases since the last refresh
for each training iteration:
    if consensus is enabled and unseen cases remain:
        primary = uniform_choice(unseen cases in High ∪ Low)
    else:
        branch = High with probability 0.75, otherwise Low
        if that branch is empty, use the other branch
        primary = uniform_choice(branch)

    partner = uniform_choice(all 101 train cases)  # not part of coverage
    train_step(primary, partner)
    seen.add(primary)  # regardless of effective patches / nonzero loss

    if all 101 cases are in seen and refresh_cycle < 1:
        rebuild dynamic teacher from current student
        refresh_cycle += 1
        seen.clear()
```

**To reproduce the observed 0.344416 protocol:** keep the fixed joint 59/42 partition and seed, `p_high=0.75`, separate partner draw, primary-case coverage state persisted in checkpoints, unconditional coverage recording after the train step, unseen-first sampling only while cases remain unseen, full-set refresh, clear coverage after refresh, and `max_refresh_cycles=1`. Preserve the resume boundary too: the initial 1–550 sampling was ordinary stratified sampling; the unseen-first behavior is evidenced after resuming at 550. The later `825af4f` sampler is a different protocol because it no longer changes case selection to complete coverage.

The phase-level evidence is in [`step1_dynamic_hard_protocol.csv`](tables/step1_dynamic_hard_protocol.csv). Exact run-time source identity remains an inference because the logs do not stamp a Git hash.

## 3. Step 2 — Routing Quality Analysis

### Data checks

- `/data/hilab/datasets/WCH_MRI145/DWI/split.json`: 101 train, 44 test.
- `VoxTell_from_disk/diagnostic/dwi_uncertainty_train/per_case_uncertainty.csv`: 101 rows and one `baseline_dice` per train case. `diagnose_uncertainty.py` sets the network frozen and uses `torch.inference_mode`; it computes the uncertainty values before reading GT and uses GT only for offline Dice. The baseline Dice is the unadapted `liver cancer` prompt prediction.
- Routing uses CUE score / image uncertainty / joint uncertainty and response state, not `baseline_dice`. D's Skip consists of the 17 No-response cases; `routing_summary.json` confirms all of those raw predictions are empty.
- The frozen diagnostic CSV has no per-case precision, recall, or predicted-voxel counts. Thus Precision/Recall means are unavailable. Empty-prediction counts below are supported by the routing `No-response` metadata, not inferred from Dice=0.

### High / Low / Skip statistics

Dice thresholds are strict (`<0.1`, `<0.3`, `>0.5`, `>0.7`). SD is population SD; quartiles use linear interpolation. `P/R` are unavailable in the frozen train-case file.

| Routing | Group | N | Mean / median Dice | SD | Q1–Q3 | Dice <.1 / <.3 | Dice >.5 / >.7 | Empty predictions |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Original CUE | High | 59 | .530 / .698 | .336 | .178–.795 | 14 / 17 | 38 / 29 | 0 |
| Original CUE | Low | 42 | .067 / .000 | .172 | .000–.000 | 35 / 38 | 2 / 1 | 17 |
| Image uncertainty | High | 59 | .527 / .700 | .349 | .042–.795 | 16 / 18 | 39 / 30 | 10 |
| Image uncertainty | Low | 42 | .071 / .000 | .144 | .000–.047 | 33 / 37 | 1 / 0 | 7 |
| Joint, no gate | High | 59 | .524 / .700 | .350 | .000–.795 | 16 / 17 | 38 / 30 | 10 |
| Joint, no gate | Low | 42 | .076 / .000 | .155 | .000–.068 | 33 / 38 | 2 / 0 | 7 |
| Joint + gate | High | 49 | .631 / .740 | .283 | .583–.838 | 6 / 7 | 38 / 30 | 0 |
| Joint + gate | Low | 35 | .091 / .000 | .165 | .000–.107 | 26 / 31 | 2 / 0 | 0 |
| Joint + gate | Skip | 17 | .000 / .000 | .000 | .000–.000 | 17 / 17 | 0 / 0 | 17 |

| Routing | High–Low mean Dice gap | High–Low median gap | High with Dice <.1 | Low with Dice >.5 / >.7 |
|---|---:|---:|---:|---:|
| Original CUE | .464 | .698 | 14/59 (23.7%) | 2/42 (4.8%) / 1/42 (2.4%) |
| Image uncertainty | .456 | .700 | 16/59 (27.1%) | 1/42 (2.4%) / 0 |
| Joint, no gate | .448 | .700 | 16/59 (27.1%) | 2/42 (4.8%) / 0 |
| Joint + gate | .540 | .740 | 6/49 (12.2%) | 2/35 (5.7%) / 0 |

Joint + gate has the largest High–Low gap when Skip is reported separately. Its advantage is partly the clean isolation of 17 all-empty/no-response predictions. Among A/B/C, Original CUE has the largest mean gap; Image and Joint are slightly lower. High is not perfectly “easy”: 12–27% of High cases across the four methods have Dice <.1. Low is mostly poor, but a small number are genuinely good.

### Routing identity transitions

| Transition | High→High | High→Low | Low→High | Low→Low | Changed IDs |
|---|---:|---:|---:|---:|---:|
| Original → Image | 43 | 16 | 16 | 26 | 32/101 |
| Original → Joint | 43 | 16 | 16 | 26 | 32/101 |
| Image → Joint | 52 | 7 | 7 | 35 | 14/101 |

The changed identities are low zero-shot Dice cases in both directions. Their means (medians) are: Original→Image High→Low `.142 (.030)`, Low→High `.130 (.000)`; Original→Joint High→Low `.159 (.065)`, Low→High `.135 (.000)`; Image→Joint High→Low `.153 (.071)`, Low→High `.127 (.0003)`. Thus Joint reassigns some cases, but the moved cases are generally weak rather than a clear pool of high-performing cases newly rescued into High.

Full threshold counts, group statistics, transition-cell distributions, and joined train-case identities are in [`step2_routing_quality_summary.csv`](tables/step2_routing_quality_summary.csv), [`step2_routing_transition.csv`](tables/step2_routing_transition.csv), and [`step2_routing_case_join.csv`](tables/step2_routing_case_join.csv). Precision/Recall columns are blank where source data do not provide them.

## 4. Step 3 — High vs Low Adaptation Analysis

### Protocol audit and available overall results

All seven historical routing CSVs contain the same 101-case C_joint partition (0 identity mismatches). Stage2 is `none` in all six ablation configs; the C_joint config omits the Stage2 key, whose code default is `none`. `high_quality_sampling_probability` is 0.75 for all.

| Method | Best / stop | Refresh | Dice | mIoU | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| C_joint | 800 / 1000 | 0 | .338383 | .259746 | .316615 | .528148 |
| Image-only | 1050 / 1550 | 0 | .337883 | .258653 | .295661 | .560773 |
| Prompt-only | 1050 / 1550 | 0 | .337766 | .258515 | .295442 | .560771 |
| Cross-modal | 1050 / 1550 | 0 | .337378 | .258214 | .294842 | .560303 |
| Cross-modal + Language | 800 / 1300 | 0 | .339490 | .259855 | .327632 | .520646 |
| Dynamic-hard | 1250 / 1750 | 1 at 552 | .344416 | .263898 | .299871 | .574800 |
| Dynamic-EMA | 850 / 1350 | 1 at 502 (β=.7) | .336774 | .257303 | .301923 | .542920 |

C_joint uses `outputs/response_routing/C_joint` (1000-iteration protocol). The old ablation summary’s `C baseline` row is not a best-checkpoint result: it has no best/stop iteration and is the iteration-0 Dice .283336. It is not used as the C_joint adapted result above. These are all 44 held-out validation/test metrics, not 101-case train High/Low metrics.

### Train High/Low pre/post evidence is missing

The frozen 101-case file has zero-shot Dice only; it has no per-case Precision/Recall. The saved adapted `metrics_per_case.csv` files for C_joint and all six ablations each contain 44 held-out cases. They do not contain the 101 train IDs. Applying the train partition to those 44 cases would be invalid.

Therefore, **current results are insufficient for a true High/Low pre/post analysis**. Questions A–D below cannot be answered by group from current artifacts:

- Dynamic-hard’s 44-case overall result versus C_joint raises Recall by .0467 and lowers Precision by .0167; the contribution from High versus Low is unknown.
- Cross-modal + Language raises overall Precision versus Cross-modal by .0328 and lowers Recall by .0397; the group responsible is unknown.
- Dynamic-hard’s Recall increase is measurable on the 44 cases but cannot be attributed to train High or Low.
- The current data cannot establish whether high-branch cases regress after adaptation or whether high branch has remaining headroom.

A minimal inference-only script is provided at [`evaluate_train_partition_pre_post.py`](evaluate_train_partition_pre_post.py). It uses the fixed C_joint partition, evaluates frozen VoxTell and the saved best checkpoints over the 101 train cases, writes per-case Dice/Precision/Recall and High/Low aggregates, and does not save prediction masks. It was **not run**. A future invocation is:

```bash
python analysis/evaluate_train_partition_pre_post.py \
  --checkpoint Dynamic-hard=outputs/ablation/G_dynamic_hard_final/checkpoints/best_iteration_001250_dice_0.344416.pt \
  --gt-path /data/hilab/datasets/WCH_MRI145/DWI/labels \
  --output-dir analysis/train_partition_eval
```

That command performs frozen and Dynamic-hard inference over all 101 train cases. Add one `--checkpoint NAME=PATH` argument per additional method when those comparisons are needed. The script was **not run**.

## 5. Findings

1. **Best routing separation:** Joint + No-response gate has the largest numeric High–Low Dice gap when Skip is explicit. Among routings that assign every case High/Low, Original CUE has the largest mean gap; joint uncertainty alone is not better.
2. **Which group benefits from low-branch changes:** unavailable from existing outputs. Overall 44-case metrics cannot be assigned to the 101-case training partition.
3. **Dynamic-hard’s 0.344416 source:** group-level source unknown. Overall it raises Recall while lowering Precision against C_joint on the held-out set; the true High/Low contribution needs the unevaluated train-case metrics.
4. **Cross-modal + Language Precision change:** overall Precision is higher, but group attribution is unknown.
5. **Freeze low branch / shift to High:** current evidence does not establish that high branch is the bottleneck or that low branch can be frozen. The needed train-group evaluation is missing; no Step 4 method is proposed here.

## 6. Data / Protocol Caveats

- **Train vs test:** routing and zero-shot diagnostic rows are 101 train cases. Adapted validation rows are 44 held-out cases. They are not interchangeable.
- **Frozen vs adapted:** `baseline_dice` is the frozen, unadapted VoxTell standard-prompt Dice. The `metrics_per_case.csv` files are adapted checkpoint results on held-out cases.
- **GT boundary:** GT Dice is offline analysis only. The routing builder reads original CUE score, image/joint uncertainty, and response state; the uncertainty diagnostic reads GT only after computing uncertainty.
- **Fixed vs changing groups:** Step 2 compares A/B/C group identities and reports D Skip separately. Step 3 would hold the C_joint 59/42 partition fixed; all historical ablation routing CSVs match it exactly.
- **Historical vs mainline:** the Step 3 metrics above come from `outputs/response_routing` and `outputs/ablation`; `outputs/liver_cancer_mainline` is excluded.
- **Evidence vs inference:** config/checkpoint/metrics files directly establish overall scores and refresh values. The historical hybrid sampler is reconstructed from the run logs plus a seed-42 RNG replay; the run log does not store the exact Git SHA.
- **Unavailable metrics:** frozen train-case Precision/Recall and adapted train-case per-case metrics are absent. No new 3D inference or training was run.

### Tables generated

- `tables/step1_dynamic_hard_protocol.csv`
- `tables/step2_routing_quality_summary.csv`
- `tables/step2_routing_transition.csv`
- `tables/step2_routing_case_join.csv`
- `tables/step3_protocol_audit.csv`
- `tables/step3_overall_existing_results.csv`
- `tables/step3_route_identity_audit.csv`
- `tables/step3_train_group_data_availability.csv`
