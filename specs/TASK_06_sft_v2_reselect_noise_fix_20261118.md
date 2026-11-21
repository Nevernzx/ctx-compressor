# Task 06: fix reselect training states (noise 0–2), retrain SFT v2, held-out closed-loop eval

Date: 2026-11-18. Dispatched by Claude (brain / checker); executed by Codex, operating via `ssh 5880` from 3080.
Spec: `tasks/PLAN_v2.md` §4 and §7. Prerequisites: everything delivered in Tasks 04 and 05. Copy every item marked "Claude's interpretation" or "Claude's change" into NOTES.

## 0. Background and scope

Task 05 finding (verified independently by Claude): in reselect mode the model trained on v1 data drops gold it had kept earlier. Only 44–67% of kept gold survives each step. Held-out mid-budget final recall is 27.0%, below BM25's 39.3%.

Cause: v1 generates reselect states with `make_oracle_noise_policy`, which adds `randint(1,3)` noise fragments every step. As a result, among reselect training targets at step ≥ 1, 39,536/39,546 have a non-empty drop, averaging 2.00 drops. The model learned "delete something every step". In the closed loop its own kept area is usually all gold (5,111/6,197 steps), yet it still deletes in 77% of those steps.

The user chose option 1 on 11-18: "change the noise count to a random 0–2, so that about a third of the steps need no deletion; keep everything else the same; rerun 20k × 2 epochs + held-out eval".

**Do:**
- **A.** Regenerate the reselect states with noise in 0–2, and produce SFT v2 data.
- **B.** Same sampling and same hyperparameters; retrain SFT v2 for 2 epochs.
- **C.** Held-out closed-loop eval of the 5 checkpoints; select a checkpoint; compare with Task 05.

**Don't (hard rules):**
- No evaluation on dev (test) tasks; no GRPO; no reader F1; no API calls; no batch benchmark (reuse Task 05's choice).
- Do not modify any Task 01–05 artifact, including the old data, the old rollouts, and the default behaviour of the old `make_oracle_noise_policy`.
- Do not rerun Task 05's base model and CPU baselines (`no_compress`/`latest`/`bm25`/`oracle`). Cite them directly, because the held-out tasks and retrieval steps are unchanged; verify this with sha256.
- README / NOTES / HANDOFF may only be appended to. Do not touch the `ctxc` / `reader` / `train` environments, other people's environments, or other people's processes.
- **Do not touch `mirror/.git`, and do not git commit / push** (Claude reviews and then pushes).

## 1. Machine and resources

- **GPU 1 only** (`CUDA_VISIBLE_DEVICES=1`).
- **Wait for the card before training**:
  - As of 22:40 on 11-18, GPU 1 is occupied, with about 40 GB used: about 4.9 GB is the user's MRI_SAM full rerun, about 35 GB belongs to a process outside this container, and utilization is 100%.
  - Do Part A (CPU only) first. Then start the "wait for card → train → eval" pipeline with a **detached waiter script** (nohup/setsid).
  - The waiter polls `nvidia-smi` read-only every 10 minutes. Only after **GPU 1 shows used memory < 2 GB on 2 consecutive polls** does it start training.
  - It writes its status to `runs/sft_20k_v2_*/continuation_status.json`. It never retries, and never kills or touches any other process.
  - While waiting for the card, Codex does not need to keep its session alive: once everything is launched and verified, print a line starting with `WAITING:` (path of the status file, PIDs) and end the turn. Do **not** print a DONE line at that point.
  - After training and eval finish, Claude will tell you to continue with the summary.
- **Large files go under `/root/userfolder/ctx-compressor/`**:
  - `data/traj_train_v2/`, `data/sft_v2/`;
  - `runs/sft_20k_v2_*/`;
  - `eval_rollouts_v2/`.
- **`/` disk-space rule (Claude's change; reason below)**:
  - Task 05 established that `/` is an overlay sharing the same host NVMe as `/etc/hosts`, so raw `df` growth includes writes from other users.
  - The criterion therefore becomes: **the total size of files this task creates or modifies under `/` (overlay) ≤ 300 MB**, measured with `find -xdev -newer <start marker file>` and similar.
  - Still record raw `df /` at start and end, side by side, but it is no longer pass/fail.
- Training is checked at least every 30 minutes (process alive, step, NaN, VRAM, disk). Check via the waiter / monitor log; Codex does not have to be online.
  - On NaN or OOM, stop and report. Do not change hyperparameters and rerun on your own initiative.

## 2. A: regenerate reselect states and SFT v2 data

- **Policy (Claude's interpretation)**: add a parameter to `make_oracle_noise_policy`, `noise_range=(1, 3)`.
  - The default keeps the old behaviour (RNG call order unchanged).
  - v2 uses `noise_range=(0, 2)`, i.e. `want = rng.randint(0, 2)`.
  - Everything else stays the same: seed = task index; `sft_target` is always the pure oracle; append still uses the pure oracle.
- **Retrieval is not redone**: read `steps` from the existing `data/traj_train/{train,heldout}_tasks.jsonl` and only rerun `rollout`.
  - reselect uses the v2 policy;
  - the append rollout is copied as-is.
  - Write to `data/traj_train_v2/` and `data/sft_v2/{train,heldout}.jsonl`, with the same format and meta fields as v1.
- **Consistency checks** (stop and report on any failure):
  1. Calling the default parameters on a v1 task reproduces v1's reselect rollout byte for byte (at least 200 tasks, plus a unit test);
  2. v2 row counts equal v1's (train 88,562 / heldout 4,652);
  3. Every append row is byte-identical to v1;
  4. Every reselect step-0 row is byte-identical to v1;
  5. The step-0 state is identical in both versions: nothing kept yet, and the same returned passages.
- **Statistics** (v1 and v2 side by side, by mode × tier, steps ≥ 1 only):
  - fraction of targets whose drop is non-empty (v2 expected ≈ 2/3);
  - mean drop count;
  - mean number of kept fragments;
  - fraction of steps whose kept area is entirely gold.

## 3. B: training (only the data changes)

- **Sampling**: run Task 05's `sample_sft_subset.py` procedure and seed (`random.Random(20261118)`) unchanged on `data/sft_v2/train.jsonl`.
  - **Verify that the selected (task, step, mode) list is exactly Task 05's `sample_ids.json`**, and that the append rows are byte-identical.
  - The eval-loss subset uses the same procedure on `sft_v2/heldout.jsonl`, and its IDs must likewise match Task 05's.
- **Hyperparameters**: exactly the same as Task 05's `train_run_config.json`. Diff the two configs field by field and write the diff into the report; the only allowed differences are the paths.
  - per-device batch 1 × accumulation 16, `group_by_length=False`;
  - LoRA r=16 / alpha 32 / dropout 0.05, all-linear;
  - lr 1e-4 cosine, warmup 3%, bf16, gradient checkpointing;
  - `max_length=8392`, `truncation_strategy=raise`, seed 20261118;
  - 2 epochs, 2,500 steps, save every 500 steps, eval loss on the fixed 1,000 samples.
- Output to `runs/sft_20k_v2_<date>/`.

## 4. C: held-out closed-loop eval

- **Protocol**: same as Task 05.
  - 249 tasks × 2 modes × 3 budgets = 1,494 trajectories;
  - greedy, `max_tokens=256`;
  - vLLM settings and LoRA loading identical to Task 05;
  - the four kinds of invalid output and their handling identical to Task 05.
  - Output to `eval_rollouts_v2/<checkpoint>/`.
- **Pre-checks**: the sha256 of `heldout_tasks.jsonl` matches Task 05; the step-0 prompts are byte-identical to the v1 eval (the user text is unchanged).
- **Metrics**: same as Task 05 (Claude added a few diagnostic metrics). Compute them for both the 5 v2 checkpoints and the 5 Task 05 checkpoints (read-only from the existing `eval_rollouts/`):
  - per step, the add rate for new gold (new-returned gold that was not kept before → is it in keep_after);
  - per step, the survival rate of kept gold (reselect);
  - among reselect steps ≥ 1 with a valid output and a kept area that is entirely gold, the fraction that still dropped something;
  - for the trajectory's first error, a breakdown by cause: invalid output / new gold not added / old gold dropped / both.
- **Checkpoint selection**: same rule as Task 05 (lowest pooled evidence-loss rate over the 6 cells → highest pooled final recall → earliest step).
- **The report states**:
  - each v2 checkpoint's pooled loss rate and pooled final recall;
  - the selected checkpoint;
  - the following, at the mid budget for both modes, for the v2 selected model, Task 05's checkpoint-2000, the base model (cited), bm25 (cited) and oracle (cited):
    - final recall;
    - evidence loss rate;
    - the four invalid rates;
    - single-step recall;
    - cost (with caching);
    - the diagnostic metrics above.
  - Do not draw a conclusion such as "the fix worked"; only list the numbers, each with its n.

## 5. Deliverables

- **Code** (project directory, synced to mirror):
  - the `noise_range` parameter of `make_oracle_noise_policy` plus unit tests (old default reproduces v1, the new range can yield 0 noise);
  - `scripts/make_sft_v2_data.py` (or an equivalent name);
  - v2 variants or parameterization of the waiter / training / eval / report scripts;
  - all existing tests (462) must still pass.
- **Results** (`eval/sft_run_v2/`, synced to mirror):
  - `sft_v2_report.md` (in Chinese, every number with its n);
  - `sft_v2_results.json`;
  - `data_v2_stats.json`;
  - `loss_curve.csv` / `.svg`;
  - `train_run_config.json`;
  - `config_diff_vs_task05.json`.
- **Documentation**:
  - append to README / NOTES / HANDOFF;
  - update SHA256SUMS (large data: record the path and sha256);
  - Task 01–05 invariant files and document prefixes still verify.
  - rsync into `mirror/` with `-a`, **without `--delete`, never touching `mirror/.git`**.

## 6. Completion criteria

1. A's consistency checks all pass; training runs its 2 epochs with no NaN / OOM; all 5 checkpoints complete the closed-loop eval.
2. Only GPU 1 was used, and released afterwards; files this task wrote under `/` total ≤ 300 MB (raw `df` recorded separately).
3. All tests pass.
4. The final message reports in Chinese:
   - row counts of the v1 / v2 data and the drop-non-empty ratio;
   - time spent waiting for the card and training time;
   - each v2 checkpoint's pooled loss rate;
   - the selected checkpoint;
   - the v2 vs v1 vs bm25 vs oracle comparison in both modes at the mid budget.

   End it with this line:

```
===AGENTDECK_DONE=== status=<ok|fail> summary=任务06 <one sentence>
```
