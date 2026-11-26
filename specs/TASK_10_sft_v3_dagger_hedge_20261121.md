# Task 10: SFT v3 (states from the hedge policy, hedge + correct-drop targets), reward v2, locked final test set

Written by Claude on 2026-11-21 at about 22:40. Codex executes; Claude checks. Read this whole file before starting.

## 0. Background and scope

**Why.** The user approved Claude's recommendation on 2026-11-21 ("同意，先写任务10任务书"). Task 09 (`eval/hedge_v2/hedge_v2_report.md`; Claude's failure analysis in `claude_checks/hedge_reselect_failure_out_20261121.txt` and `hedge_drop_title_out_20261121.txt` on 3080) found:
1. **Adding a second candidate when unsure has value.** H1 (τ\*=0.99) pooled loss was 806/1,494 = 53.95%, against H0's 957 = 64.06%. Append-mid went from 62.7% to 34.5%.
2. **The model cannot drop correctly.** In reselect, hedged extras in the kept area trigger drops:
   - steps with non-gold in kept that dropped an old gold: H0 37/2,250, H1 546/3,409;
   - 563 of 2,157 dropped segments were gold;
   - only 120 of those 563 were replaced by a same-title non-gold.

   So reselect got worse in every tier (≈65% → 69.5%).
3. **The PLAN §4 reward barely prices tokens.** τ\*=0.99 hedged on 33% of steps, for +38% cached cost.

**The full method this task builds toward.** "When unsure, keep one more fragment; once later context resolves it, drop the extra." SFT v3 teaches the format and the correct drops. GRPO round 2 (Task 11, not this task) tunes *when* to hedge, under reward v2.

**Parts:**
- **A.** Locked final test set from MuSiQue dev. CPU only; nothing is run on it.
- **B.** Reward v2: implementation, tests, and an offline τ table. CPU only.
- **C.** DAgger states: run SFT v2 + the Task 09 hedge wrapper (τ=0.99) closed-loop on train tasks. GPU 1.
- **D.** SFT v3 data and training. GPU 1.
- **E.** Held-out closed-loop eval of the SFT v3 checkpoints. GPU 1.
- **F.** GRPO-readiness sampling probe. GPU 1.

**Not in this task:**
- GRPO;
- reader F1;
- any API call;
- any strategy (model, bm25, oracle, …) on the Part A test set;
- changing Task 05–09 artifacts.

**Pre-registered, not to be changed after seeing results:**
- the target rules in §4;
- the sampling and hyperparameters in §5;
- the checkpoint rule in §6;
- the reward v2 formula in §3.

Report numbers with their n. No "worked / didn't work" wording. One run, point estimates, no CIs.

Copy every item marked "Claude's change" or "Claude's interpretation" into NOTES.

## 1. Machine, resources, hard rules

Same as Task 09 §1:
- 5880, formal project `/root/workplace/ctx-compressor`.
- Large files only under `/root/userfolder/ctx-compressor/`. New dirs: `data/test_dev_v1/`, `dagger_v3/`, `data/sft_v3/`, `runs/sft_v3_<date>/`, `eval_rollouts_v3/`. Small results: `eval/test_dev_v1/`, `eval/reward_v2/`, `eval/sft_run_v3/`.
- **GPU 1 only.** `CUDA_VISIBLE_DEVICES=1`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Before each GPU job, GPU 1 must use < 2,048 MiB. Never touch GPU 0.
- Files written under `/` must total ≤ 300 MB. Use the existing `reader` / `train` / `grpo` envs; no pip installs. If something is missing, print `NEED:` and stop.
- Never delete anything, never overwrite outputs, kill processes only by exact PID.
- Don't touch `mirror/.git`, don't commit. Close all sub-agents before your final message.
- **Reuse code.** Reuse the Task 04/06 SFT builders and trainers, the Task 05/06 eval, and the Task 09 hedge wrapper / scorer / pipeline. Parameterize them; don't rewrite them. Old defaults and old outputs must stay valid.

## 2. A: locked final test set (CPU)

**Source.** MuSiQue-Ans dev (2,417 questions). Exclude every dev question id used anywhere so far:
- the Task 01 acceptance set (12 questions);
- the Task 02/03 feasibility set (`data/feasibility/tasks_dev150.jsonl`, 600 questions);
- anything else you find in the artifacts.

Report the union and how you found it.

**Construction (Claude's interpretation).** Use the same code path as the Task 02 dev feasibility tasks:
- dev-only corpus;
- BM25 top-20 per hop;
- placeholder substitution;
- gold insertion when not retrieved;
- 4 questions per task, hops in original order.

**Reproduction check.** Before building, rebuild the Task 02 150 tasks with this code path and show that the retrieval steps equal `data/feasibility/steps.jsonl` (byte-identical, or field-identical with any difference explained). If they don't reproduce, stop and report.

**Tasks.**
- Shuffle the remaining question ids with `random.Random(20261122)`. Group them into consecutive 4s and drop any remainder.
- Expected ≈ 450 tasks.
- Budgets stay 1876 / 2814 / 5628.

**Lock.** Write:
- `data/test_dev_v1/test_tasks.jsonl` and the retrieval steps;
- `eval/test_dev_v1/LOCK.md`, stating: the sha256s, the exclusion list sha, the seed, and "no strategy has been run on this set; first use requires a Claude-written task".

Add the sha256s to SHA256SUMS.

**Structural stats only.**
- task count;
- hop distribution;
- natural hit rate;
- per-task evidence-token P90 (compare with Task 02's 1,875.6);
- fraction of test gold paragraphs that appear verbatim (title + text) in the train corpus;
- fraction of test gold paragraphs that are gold in any SFT v2 training row.

No strategy outputs, no rewards.

## 3. B: reward v2 (CPU; used from GRPO round 2, not trained on in this task)

**Claude's change to PLAN §4.** The old reward stays for reproducibility; v2 is a new function. Two reasons:
1. **Late misses are nearly free.** Per-step recall weights a missed gold by 1/n_t, where n_t is the gold available so far (up to ≈12). A miss at a late step therefore costs about 4× less than one at step 1, yet the evaluation counts any miss as a lost trajectory.
2. **Tokens are barely priced.** Tokens only count when recall = 1, and only relative to B.

**Definition.** Let A_t = previously revealed or current gold that is available at this step (in kept_before ∪ new). This is the same set as the recall denominator. Invalid or over-budget outputs execute as a no-op (keep_after = kept_before), exactly as in the eval.

```
n_lost_t = |A_t \ keep_after_t|
R_t = - n_lost_t - mu * used_after_t / 1000 - 1[invalid_or_over_budget]
```

- `mu` is the price, in lost-gold units, per 1,000 kept tokens.
- **Implementation.** `src/grpo_reward_v2.py::score_completion_v2(raw, state, mu)`. It runs the eval parser first (same as `grpo_reward.py`) and returns the reward and its parts.
- **Tests.** Every branch:
  - invalid JSON;
  - bad id;
  - append drop;
  - over budget;
  - no loss;
  - 1 and 2 lost;
  - dropping an old gold;
  - the μ term.

  Parity: for valid actions, its keep_after equals `score_completion`'s.

**Offline τ table.**
- Inputs: Task 09's 22,158 train states and scores (`hedge_v2/train/state_scores.jsonl` and the greedy outputs). Use the same open-loop hedge execution as Task 09 §2.
- For μ ∈ {0, 0.1, 0.25, 0.5, 1.0} and every τ in the Task 09 grid, compute:
  - mean reward v2;
  - hedge rate;
  - current-gold miss rate;
  - mean extra tokens.
- Report τ\*(μ), choosing the smaller τ on ties. Give the table pooled and per mode.
- **Consistency.** Reproduce Task 09's reward-v1 τ table exactly from the same code.
- Write it to `eval/reward_v2/offline_tau_table.{json,md}`.
- The user will choose μ for Task 11 from this table. Don't pick one.

## 4. C + D-data: DAgger states and SFT v3 targets

**C. Rollouts (GPU 1).**
- Run the Task 09 hedge wrapper with τ = 0.99 on SFT v2 checkpoint-2000 (base + adapter `7253eae0…1f9c`, Task 09 engine parameters).
- Tasks: the same 400 train tasks as Task 08/09; 2 modes × 3 tiers = 2,400 trajectories, closed loop.
- Output: `dagger_v3/rollouts/steps.jsonl`, with per-step scores, p1, hedge decision and the rendered state.
- **Pre-check.** The 400 task ids equal the Task 08 list, and overlap with the held-out tasks is 0.

**D-data. Targets.** Claude's interpretation; pre-registered. For every DAgger state:
- **Base target** = the pure oracle target from the existing `sft_target` function (the same one SFT v2 used): keep available gold within budget (smallest-token first), drop every non-gold in kept (reselect), add the current gold.
- **Hedge.**
  - Condition: the state's recorded p1 < 0.99.
  - Extra = the highest-scored candidate in `new_ids` that is not in the base target's add list and not in `kept_ids`.
  - It must fit the budget after the base target.
  - When all three hold, append extra to the base target's add list. The format is the same compact JSON as v2, `{"add":[<oracle adds>,<extra>],"drop":[...]}`.
- **Check every target with the eval parser.** It must be valid and within budget.
  - Keep_after must contain every gold the oracle keeps.
  - Extra must be non-gold.
  - The hedge must be present if and only if its condition holds.
- **Same input.** The user text and prompt ids must be exactly what the policy saw in C. Compare user_text sha on every row; compare prompt ids on a ≥ 1% sample.

**Stats** (by mode × tier):
- states;
- fraction with 2 adds;
- fraction of reselect states (step ≥ 1) with non-gold in kept;
- fraction with a non-empty drop;
- mean drops;
- token-length distribution.

**Rows longer than `max_length` 8392.** Exclude them before sampling and report the count. If the count is > 1% of the pool, stop and report.

## 5. D-train: SFT v3

**Sample (pre-registered).**
- **(a)** Exactly Task 06's 20,000 SFT v2 rows. Same ids; byte-identical to `data/sft_v2/train.jsonl` rows selected by `sample_ids.json`.
- **(b)** 10,000 DAgger rows: 5,000 per mode, drawn uniformly from that mode's eligible DAgger pool with `random.Random(20261122)`.
- Total 30,000.
- The eval-loss subset is Task 06's fixed 1,000 held-out samples, unchanged.

**Training.**
- Start from the **base** model with a new LoRA, so SFT v3 vs v2 differ only in data.
- Hyperparameters exactly as Task 06's `train_run_config.json`: batch 1 × accum 16, LoRA r16/α32/0.05 all-linear, lr 1e-4 cosine, warmup 3%, bf16, grad checkpointing, max_length 8392 with truncation `raise`, seed 20261118, 2 epochs.
- That is ≈ 3,750 optimizer steps. Save every 500 steps plus the final step.
- Diff the config against Task 06 field by field. Only paths, row count and the step count may differ.

**Monitoring.** Every ≤ 30 min (alive, step, NaN, VRAM, disk). On NaN or OOM, stop and report. Don't change hyperparameters.

## 6. E: held-out closed-loop eval

**Protocol.** Task 06's exactly:
- 249 held-out tasks × 2 modes × 3 tiers = 1,494 trajectories;
- greedy, `max_tokens` 256;
- same engine and LoRA loading;
- same invalid handling and metrics/diagnostics code.

This is plain greedy: **no hedge wrapper**. Output: `eval_rollouts_v3/<ckpt>/`.

**Pre-checks.** The held-out sha matches Task 05. Step-0 prompts are byte-identical to the Task 06 eval.

**Checkpoints.** 1000, 2000, 3000 and the final step.

**Selection rule** (same as Task 05/06): lowest pooled loss rate → highest pooled final recall → earliest step.

**Extra stats** (per cell and pooled):
- add-count distribution (0 / 1 / 2 / ≥3);
- for outputs with ≥ 2 adds: the fraction that contain the current gold, and the fraction of added fragments that are gold;
- **drop precision**: dropped segments that are non-gold / all dropped segments;
- kept-gold survival;
- the Claude failure-analysis split: steps with non-gold in kept → fraction that dropped an old gold;
- occupancy;
- cached cost;
- first-error breakdown;
- new-gold add rate by same / different title (Task 08 `extra.json` definitions).

## 7. F: GRPO-readiness probe

- Pick 300 DAgger states that are **not** among the 10,000 sampled in §5 (seed 20261123, 150 per mode).
- For each, sample 8 outputs at temperature 1.0, top_k disabled, `max_tokens` 256. Use the Task 07/08 probe code and settings.
- Run on two models: (1) the SFT v3 selected checkpoint; (2) SFT v2 checkpoint-2000.
- Report:
  - fraction of samples with ≥ 2 adds;
  - invalid rate;
  - mean distinct actions per group;
  - fraction of groups with zero reward std under reward v1 (λ=0.3) and under reward v2 for every μ in §3.

  Task 08 reference: 82% of groups had zero std, and 0/177,257 samples added 2.

## 8. Pipeline, report, deliverables

**Order.**
1. Do A and B interactively first.
2. Do C's pre-checks and a 2-task smoke run interactively.
3. Launch C → D-data → D-train → E → F as one detached pipeline. Follow the Task 08/09 pattern: `dagger_v3/pipeline_status.json`, stage receipts, `require` checks, no automatic retry.
4. Verify it is running, print `WAITING:` and end the turn **without a DONE line**. Claude will message you for the wrap-up.

**Report.** `eval/sft_run_v3/sft_v3_report.md` (in Chinese, every number with its n) and `sft_v3_results.json`:
1. the A summary (test-set size, exclusions, lock shas, structural stats);
2. the B τ\*(μ) table;
3. C/D data stats;
4. the training curve;
5. each checkpoint's pooled loss and recall, plus the selected checkpoint;
6. the comparison, pooled and at tight and mid for both modes:
   - rows: SFT v3 selected, SFT v2 ckpt-2000 (Task 06), H0 and H1 (Task 09), GRPO ckpt-500 (Task 08), bm25, oracle (cited);
   - columns: final recall, loss rate, invalid rates, single-step recall, occupancy, cached cost, mean adds per step, drop precision;
7. the F probe table.

**Deliverables.**
- Code and tests. All existing tests (568) must pass.
- `TASK10_SHA256SUMS`, merged into SHA256SUMS. The Task 06–09 manifests must still verify.
- Append-only README / NOTES / HANDOFF.
- Mirror: `rsync -a`, without `--delete`, without touching `.git`, no commit.

**Final message.** In Chinese, then:

```
===AGENTDECK_DONE=== status=<ok|fail> summary=任务10 <one sentence>
```

## 9. Time estimate (Claude)

| Part | Estimate |
|---|---|
| A + B (CPU, interactive) | ~1.5–2 h, partly in parallel with C |
| C: 2,400 hedge trajectories (scoring included) | ~4.5 h |
| D-train: 30k × 2 epochs (Task 06 took 7.2 h for 20k) | ~11 h |
| E: 4 checkpoints, plain greedy | ~3–4 h |
| F: probe | ~0.5 h |

Total GPU ≈ 19–20 h after launch.
