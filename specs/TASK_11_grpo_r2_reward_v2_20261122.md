# Task 11: GRPO round 2 (reward v2, μ = 0.5, no per-group std scaling) from the Task 10 SFT model — 5880 GPU 1

Written by Claude on 2026-11-22, around 19:30. Codex executes and Claude checks. Read the whole file before you start. Every decision rule below is fixed in advance; do not change any of them after seeing results.

## 0. Background, scope, and what is new compared with Task 08

**Why this task exists.**
- On 2026-11-22 the user said "SFT完了请你继续GRPO或者别的强化学习策略". Claude read that as: once Task 10 finishes, run the second GRPO round from PLAN_v2 §4, using reward v2.
- **μ = 0.5.** This is Claude's recommendation from the Task 10 B table: τ\*≈0.9, hedge rate 15.2%, current-gold miss 3.39%. The user did not choose a different value.

**Why Task 08 (GRPO round 1) barely moved.** Claude's analysis is in `claude_checks/grpo_failure_out_20261121.txt`:
- Only 18% of states were informative (some reward variance within the group of 8 samples).
- In only 4.2% of all states did any sample beat greedy.
- The checkpoint-500 output differed from SFT on 2.2% of identical-input steps.

**What changes in this round (four things, nothing else):**
1. **Start model.** The Task 10 selected SFT v3 checkpoint. The exception rule is in §2.
2. **Reward v2**, μ = 0.5:
   - Formula: `R = −n_lost − 0.5·used_after/1000 − 1[invalid]`.
   - An invalid output executes as a no-op.
   - Use `src/grpo_reward_v2.py::score_completion_v2`. This is the exact function already used for the Task 10 B table and its tests. Do not re-implement it.
3. **`--scale_rewards none`.** The advantage is `r − group mean`, with no division by the group std (the "Dr. GRPO" form).
   - Reason: reward v2 has a cost of ~0.03–0.1 per hedged fragment next to a cost of 1 per lost gold fragment.
   - With per-group std scaling, a group whose 8 samples all keep the gold and differ only in whether they hedge gets advantages of about ±1. That is the same size as a group that differs in losing gold, so training would mostly learn "never hedge".
   - Without std scaling, that tiny token difference stays tiny.
   - swift 3.12.6 in `envs/grpo` supports `scale_rewards ∈ {group, batch, none}`. Claude checked `grpo_trainer.py` lines 421–545.
4. **LR 1.5e-5** (3 × Task 08's 5e-6).
   - With `none`, the advantages in groups that differ in gold loss shrink by that group's std. Typical values: 1 of 8 samples losing gold gives std 0.33; 2 of 8 gives 0.43.
   - Multiplying LR by 3 keeps the effective step on those groups about where Task 08 had it.
   - The KL coefficient stays at 0.04.

**Everything else is Task 08's.** That covers the state counts, the sampling filter, the dataset rule, all other hyperparameters, 500 steps, checkpoint saving, the eval protocol, and the selection rule.

**Pre-registered PLAN §4 targets** (report the numbers only; do not write "worked" or "didn't work"):
1. Tight-budget evidence loss rate, per mode.
2. Pooled loss rate, then pooled final recall.

Also report the held-out mean of reward v2 per step, broken into mean n_lost, mean used/1000 and invalid rate. This is the training objective measured on held-out data.

**Not in scope.** The locked test set (`eval/test_dev_v1/`) stays untouched. It is reserved for the final comparison, and the user decides when that happens.

## 1. Machine, resources, hard rules (same as Tasks 08 and 10)

**Paths.**
- Machine: 5880 (`ssh 5880` from 3080).
- Project: `/root/workplace/ctx-compressor`.
- Large files go only under `/root/userfolder/ctx-compressor/`. New external directories are `grpo_r2/` and `eval_rollouts_grpo_r2/`. Small results go in `eval/grpo_r2/`.

**GPU 1 only.**
- Set `CUDA_VISIBLE_DEVICES=1` and `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Never use GPU 0, even if it looks free.
- Before **every** GPU stage, use the Task 10 resume gate (`scripts/sft_v3_wait_gpu.py` logic):
  - read GPU 1 every 5 min;
  - start only after 3 consecutive readings below 2,048 MiB;
  - time out after 36 h;
  - publish `waiting_gpu` in the status file.

  GPU 1 is shared with the user's MRI_SAM line. **Never touch MRI_SAM processes or directories.**

**Disk and environments.**
- Files written under `/` must total ≤ 300 MB.
- No pip installs. If something is missing, print `NEED:` and stop.

**Never:**
- delete anything;
- overwrite existing outputs (refuse instead);
- write into the directories of Tasks 05–10;
- touch `mirror/.git`, commit, or push;
- kill anything except by exact PID (no `pkill -f`).

**Close all sub-agents before every final message.**

## 2. Phases and how they fit around the still-running Task 10

### Phase P — now, while Task 10 is still running (CPU only)

- Task 10's resume pipeline (PID 2541782) is in D_train. E and F will import project scripts **at the time each stage starts**.
- **Phase P must not change any file listed in `eval/sft_run_v3/TASK10_LAUNCH_SHA256SUMS` or `TASK10_RESUME1_SHA256SUMS`, nor anything Task 10's remaining stages import.**
  - New behaviour goes in **new files**, or in edits deferred to Phase A.
  - Run `sha256sum -c` on both lists at the start and at the end of Phase P, and report the result.
- Write the code and tests for §3:
  - GRPO with reward v2: the plugin, the filter scoring, and the dataset builder;
  - `scale_rewards` / LR as parameters of the training script, with Task 08 defaults kept;
  - the state builder for the start model, without a hedge wrapper;
  - the eval and pipeline driver.
- Run the full CPU test suite: all existing tests plus the new ones must pass.
- No GPU use and no merge yet (the start model is not known yet).
- End with `WAITING: 任务11 准备完成（Phase P）` and **no DONE line**.

### Task 10 wrap-up
- When Task 10's pipeline completes, Claude sends the Task 10 wrap-up message.
- Finish that first, as Task 10 §8 says, ending with the `任务10` DONE line.

### Phase A — after Claude says "开始任务11" (interactive)

1. **Start model.**
   - Read `eval/sft_run_v3/selected_checkpoint.json` and the Task 10 E pooled results.
   - If the selected v3 checkpoint's pooled loss count is ≤ 956/1494 (SFT v2 checkpoint-2000, Task 06), start from it.
   - Otherwise start from SFT v2 checkpoint-2000.
   - Record which one was used and why.
2. **Merge.** Use `scripts/merge_sft_lora.py` (CPU, fp32 merge, saved as bf16) to write `grpo_r2/merged_start/`.
3. **Greedy equivalence.** As in Task 08 A.2, using the n = 30 step-0 fixture:
   - run base + adapter, and the merged model alone (GPU 1, gated);
   - report text and token-id mismatches out of 30.
   - Mismatches are recorded, not a stop condition.
4. **Apply deferred edits.** Apply any edits to shared files that Phase P deferred, now that Task 10 has finished. Rerun the whole test suite.
5. **Launch.**
   - Launch the detached pipeline in §3 (pidfile, `grpo_r2/pipeline_status.json`, a stage receipt per stage, `require` checks, no automatic retry, monitor every 30 s).
   - Check that it is running.
   - Print `WAITING:` and end **without a DONE line**. Claude will message you for the wrap-up.

## 3. Pipeline stages

### F0. Merged-only held-out eval (consistency)

- Use the Task 10 E protocol exactly:
  - 249 held-out tasks × 2 modes × 3 tiers = 1,494 trajectories;
  - greedy, `max_tokens` 256;
  - the same engine settings, seed, invalid handling and metrics.
- Model: the merged start model, with no LoRA.
- Output: `eval_rollouts_grpo_r2/merged_start/`.
- Report:
  - per-cell and pooled differences from the start checkpoint's own Task 10 E rollouts (or Task 06's, if v2 was chosen);
  - the number of steps whose raw output differs, among steps with identical input states.

### B. Train states (start-model policy, on-policy)

- Run greedy closed-loop rollouts of the start model (base + adapter, exactly as in eval, **no hedge wrapper**):
  - **400 train tasks** drawn with seed 20261123;
  - **disjoint from** Task 08 B's 400 tasks and Task 10 C's 400 tasks;
  - all 6 cells → 2,400 trajectories, about 22k states.
- All Task 07/08 `verify_rows` checks must show 0 mismatches:
  - gold, re-encode, re-render;
  - train/held-out overlap 0;
  - locked-test-set overlap 0.
- Report per cell:
  - n, invalid rate, kept-has-non-gold rate, lost-gold rate;
  - add-count distribution;
  - prompt length (min / median / p99 / max) and the number of prompts over 8,392.

### C. Sampling filter (all states)

- Sample every B state 8 times:
  - temperature 1.0, top_p 1.0, top_k −1, `max_tokens` 256, seed 20261123.
- Score each sample with `score_completion_v2(·, μ=0.5)`.
- Write one row per state with:
  - identifiers and cell;
  - the 8 rewards and the 8 (n_lost, used_after, invalid) triples;
  - the 8 add-counts;
  - the population std and the greedy reward.
- **Informative** (pre-registered, as in Task 08) means the population std of the 8 rewards is > 1e-9.
- Report per cell and pooled:
  - zero-std rate;
  - **gold-varying rate**: n_lost or invalid differs within the group;
  - token-only-varying rate;
  - the fraction of groups where some sample beats greedy;
  - the sampled add-count distribution;
  - invalid rate by category;
  - throughput.
- Compare these with the Task 10 F probe on the same model, and with Task 08 C.

### D. GRPO dataset

- Use the Task 08 rule exactly:
  - **2,000 prompts drawn from informative states only**;
  - balanced over the 6 cells, in sorted order, with targets 334 / 334 / 333 / 333 / 333 / 333;
  - shortfall goes round-robin to cells with surplus;
  - seed 20261124, shuffled at the end.
- If fewer than 2,000 states are informative, use all of them.
- **If fewer than 500 are informative, stop before E** and report it (`status=stopped_insufficient_signal`).
- Columns are as in Task 08. The reward state must carry everything `score_completion_v2` needs.
- Check:
  - every row's user text is byte-identical to its B state;
  - all rows are from the train split;
  - the plugin's reward on each row's recorded greedy output equals C's greedy reward (0 mismatches).

### E. GRPO training

- Use `train_grpo.sh` (Task 08's version) with:
  - `MERGED = grpo_r2/merged_start`, `DATA = <D>`, `OUT = grpo_r2/train/`;
  - the **reward v2 plugin** (μ = 0.5);
  - **`--scale_rewards none`** and **LR 1.5e-5**.
- Keep Task 08 exactly for everything else:
  - cosine schedule, warmup 0.03, BETA 0.04;
  - NGEN 8, GEN_TEMP 1.0, TOP_P 1.0, top_k −1;
  - `max_completion_length` 256, `max_length` 8392;
  - PER_DEVICE 4, GRAD_ACC 8, EPOCHS 1, `MAX_STEPS = floor(n_prompts/4)` (500 for 2,000 prompts), SAVE_STEPS 125;
  - VLLM_UTIL 0.30, colocate, enforce_eager, SLEEP_LEVEL 0;
  - a new LoRA: r16 / α32 / dropout 0.05, all-linear;
  - `dynamic_sample` off, GRPO_SEED 20261123;
  - the same advantage estimator and loss_type as Task 08. Record what swift resolved them to.
- Log every step:
  - loss, reward, reward_std, frac_reward_zero_std, kl, grad_norm;
  - completion length and clipped ratio, memory, seconds;
  - **the reward components** from the plugin: mean n_lost, mean used/1000, invalid rate, mean add-count, and the ≥2-add rate.
- Stop on a nonfinite loss or grad_norm, or on OOM.
- Recheck every logged reward in `completions.jsonl` with the eval parser plus `score_completion_v2`. Expect 0 mismatches.

### F. Held-out eval of the GRPO checkpoints

- Evaluate checkpoints 125 / 250 / 375 / 500 (or whichever were saved) as a LoRA on the merged start model, using the F0 protocol.
- Output to `eval_rollouts_grpo_r2/<ckpt>/`.
- Pre-checks:
  - the held-out sha matches;
  - step-0 user text and prompt ids are identical to the Task 10 E eval.
- **Selection rule** (Task 05, among the GRPO checkpoints only): lowest pooled loss rate → highest pooled final recall → earliest step.
- For every eval, F0 included, report the Task 10 E extra stats:
  - add-count distribution;
  - for outputs with ≥2 adds: gold content;
  - drop precision;
  - kept-gold survival;
  - the non-gold-in-kept → old-gold-dropped split;
  - occupancy and cached cost;
  - first-error breakdown;
  - new-gold add rate by same / different title;
  - **the held-out reward v2 per step** (mean, and its components).

## 4. Report (at wrap-up, after Claude's message)

Write `eval/grpo_r2/grpo_r2_report.md` (in Chinese, every number with its n) and `grpo_r2_results.json`.

**Sections:**
- A: which start model and why; merge equivalence; how F0 differs from the start model's eval.
- B / C / D statistics.
- E: training curves as csv + svg (reward, reward components, reward_std, frac_zero_std, kl, ≥2-add rate), wall time, peak memory, and the reward recheck.
- Each GRPO checkpoint's pooled loss and recall, and the selected checkpoint.

**Comparison table**, pooled and at tight and mid for both modes.

Rows:
- GRPO r2 selected;
- the start SFT model (Task 10 E);
- SFT v2 checkpoint-2000 (Task 06);
- GRPO r1 checkpoint-500 (Task 08);
- H1 (Task 09);
- bm25 and oracle (cited).

Columns:
- final recall, loss rate, the four invalid rates, single-step recall;
- occupancy, cached cost;
- mean adds per step, ≥2-add rate, drop precision;
- held-out reward v2 per step.

**State the two pre-registered PLAN §4 metrics as numbers**, start SFT vs GRPO r2 selected.

## 5. Deliverables

- **Code and tests**, synced to the mirror. All existing tests plus the new ones must pass.
- **`eval/grpo_r2/TASK11_SHA256SUMS`**, merged into SHA256SUMS. The Task 06–10 manifests must still verify.
- **README / NOTES / HANDOFF**: append only. Earlier sections stay byte-identical.
- **Mirror**: `rsync -a` with no `--delete`, never touching `.git`, no commit.

**Final message**, in Chinese:
- the start model;
- F0 differences;
- state / informative / gold-varying / dataset counts;
- training time;
- each checkpoint's pooled loss;
- the selected checkpoint;
- the tight and mid comparison;
- adds per step and the ≥2-add rate, start vs GRPO;
- held-out reward v2.

End with:

```
===AGENTDECK_DONE=== status=<ok|fail> summary=任务11 <one sentence>
```

## 6. Time estimate (Claude)

| Stage | Estimate |
|---|---|
| Phase P | ~1 h (CPU, during Task 10) |
| Phase A | ~30–40 min |
| F0 | ~0.5 h |
| B | ~1 h |
| C | ~1–1.5 h |
| D | minutes |
| E | ~5.5–6.5 h (Task 08: 40.4 s/step) |
| F | 4 × ~0.5 h |

Total ≈ 11–12 h after Phase A.

## 7. Addendum (Claude, 2026-11-23 01:20): training smoke test in Phase A

Task 10 is complete. E selected `checkpoint-3000`, with 886/1494 pooled loss. Since 886 ≤ 956, **start from SFT v3 checkpoint-3000.**

**Where the smoke test goes.** In Phase A, after the merge and greedy equivalence and before launching the pipeline, run a **2-optimizer-step GRPO smoke test** on GPU 1. Use the same gate before it.

**Setup.**
- Use the real `train_grpo_r2.sh` / `run_grpo_r2_training.py` path:
  - swift 3.12.6;
  - `--scale_rewards none`;
  - LR 1.5e-5;
  - the `ctx_step_reward_v2` plugin;
  - NGEN 8, PER_DEVICE 4, GRAD_ACC 8.
- Model: the merged start model.
- Data: 8 prompts in the round-two dataset format, taken from Task 08's `grpo_v2` dataset rows. Convert or re-pack them so the reward state carries every field `score_completion_v2` needs.
- `MAX_STEPS=2`.
- Write everything under `grpo_r2/smoke_train/`. This is a separate smoke-only entry point or flag. The formal `train/` path and its checks must stay unchanged.

**What to verify.**
- Both optimizer steps finish.
- The reward components are present and aligned (n = 32 per step).
- The logged advantages / rewards are consistent with no std scaling. Check the swift resolved args and `completions.jsonl`.
- The reward recheck shows 0 mismatches.
- No NaN or OOM. Record peak memory and seconds per step.

**If it fails,** fix the code, rerun the full test suite, and retry. Do not launch the pipeline until the smoke test passes.

**Report** the smoke-test results in the final report under A.
