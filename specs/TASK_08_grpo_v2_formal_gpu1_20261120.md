# Task 08: formal GRPO from SFT v2 checkpoint-2000 on 5880 GPU 1, held-out closed-loop eval

Written by Claude on 2026-11-20 at about 21:30. Codex executes; Claude checks. Read this file fully before starting.

## 0. Background and scope

- On 2026-11-20 at about 21:20 the user said "v2训完了吗，训完了请继续强化学习". Claude's reading of that: start formal GRPO now, on the recommended options:
  - start from **v2 checkpoint-2000**, the checkpoint Task 06 selected (`eval/sft_run_v2/selected_checkpoint.json`, adapter sha256 `7253eae0…1f9c`);
  - use **GPU 1**;
  - run the whole chain without stopping to ask. Every decision rule below is fixed in advance.
- Task 07 already built and verified the whole GRPO toolchain on a stand-in (v1 ckpt-2000):
  - the `envs/grpo` environment;
  - `build_grpo_states.py`, `probe_grpo_sampling.py`, `make_grpo_dataset.py`, `merge_sft_lora.py`;
  - the `train_grpo.sh` script and the `grpo_swift_plugin.py` reward plugin.

  Read `eval/grpo_prep/grpo_prep_report.md` first. **Reuse that code.** Parameterize it, don't rewrite it.
- Task 06 failure analysis (Claude, `claude_checks/heldout_failure_v2_out_20261120.txt` on 3080), for context only:
  - v2 adds **exactly one fragment per step** (mean 1.00, never 0, never 2).
  - It picks the wrong one about 11% of the time, which accounts for most of the 64% pooled loss rate.
  - The budget is not binding: mid-budget occupancy is 0.43.
  - The PLAN §4 reward should favour adding a second fragment when the model is unsure. That is what this task tests.
- Pre-registered from PLAN §4, and not to be changed: where GRPO should beat SFT.
  1. Evidence loss rate at the **tight** budget, per mode.
  2. The closed-loop result: pooled evidence loss rate, then pooled final recall.

  Report the numbers only. Do not write "GRPO worked" or "GRPO didn't work". This line uses one run and point estimates, each with its n, and no CIs.

## 1. Machine, resources, hard rules

- **Machine**: 5880, reached from 3080 with `ssh 5880`. Formal project: `/root/workplace/ctx-compressor`. Large files only under `/root/userfolder/ctx-compressor/`.
  - New external directories: `grpo_v2/` (merged model, states, filter, data, train, logs) and `eval_rollouts_grpo_v2/`.
  - New small results: `eval/grpo_v2/`.
- **GPU 1 only.**
  - Every GPU job sets `CUDA_VISIBLE_DEVICES=1` and `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
  - Before each GPU job, require GPU 1 used memory < 2,048 MiB.
  - Never run anything on GPU 0. It belongs to someone else and is busy again (about 19.5 GB at 21:00).
- **Parameterizing the GPU**: Task 07's scripts hard-code GPU 0. These include:
  - `grpo_environment.sh`, which exports `CUDA_VISIBLE_DEVICES=0`;
  - `train_grpo.sh`, which has a GPU 0 gate and writes `gpu0_monitor.csv`;
  - the `--gpu` choices `(0,)` in `build_grpo_states.py` and `probe_grpo_sampling.py`.

  Add a `GRPO_GPU` / `--gpu` parameter that allows 1. Default behaviour, tests and the existing Task 07 outputs must stay valid. The monitor csv name follows the GPU.
- **Paths**: some scripts require output under `grpo_prep/`. Allow `grpo_v2/` as well. Never write into Task 05–07 directories (`grpo_prep/`, `eval_rollouts/`, `eval_rollouts_v2/`, `runs/`, `eval/sft_run*`, `eval/grpo_prep/`). Refuse to overwrite existing outputs.
- **Disk**: files this task writes under `/` must total ≤ 300 MB. Caches and tmp go under `grpo_v2/` or the existing `grpo_prep/cache`. Record raw `df` separately.
- **Environments**: no pip installs into any existing env. If something is missing, print `NEED:` and stop.
- **Don't**:
  - delete anything;
  - touch `mirror/.git`;
  - commit or push.
- **Killing processes**: only by exact PID, never `pkill -f`. Inside `ssh 5880 '…'`, `pgrep -f` matches the ssh command line itself, so use pidfiles and `kill -0`.
- **Sub-agents**: if you use them, close all of them before your final message.

## 2. A: merge and equivalence (interactive, CPU, then GPU 1)

1. Merge with `scripts/merge_sft_lora.py` (CPU, fp32 merge, saved as bf16):
   - base: `models/Qwen3-1.7B`;
   - adapter: `runs/sft_20k_v2_20261118/checkpoints/checkpoint-2000`;
   - output: `grpo_v2/merged_v2_ckpt2000/`.
2. Greedy equivalence: use the Task 07 step-0 fixture (`grpo_prep/step0_fixture`, n=30) with `build_grpo_states.py --stage reference`.
   - Run base + v2 adapter, and merged model alone.
   - Use `--compare` between the two. Both on GPU 1.
   - Report mismatches out of 30 for text and for token ids.
   - Not identical does not mean stop. Record it, and section F's merged-only eval quantifies it.
3. Train tasks: use `data/traj_train_v2/train_tasks.jsonl`.
   - It differs from `traj_train/` only in the `rollouts` field. Claude checked: 4,735 tasks, step-level fields identical.
   - Re-verify that the fields the builder reads are identical. Keep the held-out overlap checks.
4. Run all CPU tests. They must pass before section 3.

## 3. B–F: detached pipeline on GPU 1

Once A passes:
- launch one detached pipeline (setsid/nohup, pidfile) that runs the stages below in order;
- give it a status file `grpo_v2/pipeline_status.json`: stage, PIDs, start/end UTC, `last_monitor` every 30 s with GPU 1 memory/util, alive, nonfinite/oom;
- verify it is running;
- print a line starting with `WAITING:` (status path, PIDs) and **end the turn without a DONE line**.

Claude monitors and will message you for the wrap-up. Each stage `require`s its checks; on any failure the pipeline stops with `status=failed` and the first error. There is no automatic retry.

### F0. Merged-only held-out eval (consistency)
- Same protocol as Task 05/06:
  - 249 held-out tasks × 2 modes × 3 budgets = 1,494 trajectories;
  - greedy, `max_tokens` 256;
  - identical engine parameters and seed;
  - identical four invalid reasons and handling, and the same metrics and diagnostics code as `heldout_model_v2.py` / `heldout_diagnostics_v2.py`.
- Only the model path changes, to the merged model, with no LoRA.
- Output: `eval_rollouts_grpo_v2/merged_sft_v2/`.
- Report:
  - per-cell and pooled differences from Task 06 v2 checkpoint-2000;
  - the number of steps whose raw output differs, among steps whose input state is identical.

### B. Train states (v2 policy)
- `build_grpo_states.py --stage build`:
  - base + v2 checkpoint-2000 adapter (same as eval), train tasks only;
  - `--n-tasks 400`, `--seed 20261120`, giving 2,400 trajectories, about 22k states.
- All Task 07 `verify_rows` checks must report 0 mismatches: gold, re-encode, re-render, train/held-out overlap 0.
- Report: per-cell n, invalid rate, kept-has-non-gold rate, lost-gold rate, prompt length (min / median / p99 / max), and the count over 8,392.

### C. Offline sampling filter (all states)
- For **every** state from B: n=8, temperature 1.0, top_p 1.0, top_k −1 (no top-k), `max_tokens` 256, seed 20261120.
- Score with `src/grpo_reward.py::score_completion`, λ=0.3. This is the exact reward the plugin uses.
- Write one row per state: identifiers, cell, the 8 rewards, the 8 invalid reasons, the 8 add-counts, population std, and greedy reward.
- **Informative state** (pre-registered): population std of the 8 rewards > 1e-9.
- Report, per cell and pooled:
  - zero-std rate;
  - mean within-group std;
  - the fraction of groups where some sample beats greedy;
  - the invalid rate by category;
  - the distribution of sampled add-counts (0 / 1 / 2 / ≥3);
  - throughput in states/s.
- Extend `probe_grpo_sampling.py` with an all-states mode rather than writing a new sampler, and keep the old mode working.

### D. GRPO dataset
- Use `make_grpo_dataset.py` extended with an `--informative` file from C.
- **2,000 prompts from informative states only**, balanced over the 6 mode × tier cells:
  - cells in sorted order;
  - the targets are 334, 334, 333, 333, 333, 333;
  - a cell with fewer informative states contributes all of them;
  - the shortfall goes round-robin, in sorted order, to cells with surplus;
  - seed 20261121; shuffle at the end.
- If there are fewer than 2,000 informative states in total, use all of them and report it.
- Columns as in Task 07: messages, reward_state_json (with prompt_token_ids_tail), trajectory_id, state_step, cell.
- Check: every row's user text is byte-identical to its B state's `user_text`, and all rows are train-split.

### E. GRPO training
- Use `train_grpo.sh` (Task 07 version, parameterized for the GPU) with:
  - `MERGED=grpo_v2/merged_v2_ckpt2000`, `DATA=<D output>`, `OUT=grpo_v2/train/`;
  - LR 5e-6, cosine, warmup 0.03, BETA 0.04;
  - NGEN 8, GEN_TEMP 1.0, TOP_P 1.0, top_k −1, `max_completion_length` 256, `max_length` 8392;
  - PER_DEVICE 4, GRAD_ACC 8, so 4 prompts × 8 completions per optimizer step;
  - EPOCHS 1, `MAX_STEPS` = floor(n_prompts / 4), which is 500 for 2,000;
  - **SAVE_STEPS 125**;
  - VLLM_UTIL 0.30, colocate, enforce_eager, SLEEP_LEVEL 0;
  - LAMBDA 0.3, GRPO_SEED 20261120;
  - new LoRA r16 / α32 / dropout 0.05, all-linear;
  - `dynamic_sample` off.
- Log every step: loss, reward, reward_std, frac_reward_zero_std, kl, grad_norm, completion mean length / clipped ratio, memory, seconds.
- Stop the pipeline on nonfinite loss / grad_norm or on OOM.
- Recheck every logged reward in `completions.jsonl` with the eval parser, as in Task 07 (`grpo_prep_summary.py` logic). Expect 0 mismatches.

### F. Held-out eval of GRPO checkpoints
- Evaluate every saved checkpoint (125 / 250 / 375 / 500, or whatever was saved) as a LoRA on the merged model.
- Use the F0 protocol exactly. Output to `eval_rollouts_grpo_v2/<checkpoint>/`.
- Pre-checks:
  - held-out sha matches Task 05;
  - step-0 user text is byte-identical to the Task 06 v2 eval;
  - prompt ids are identical.
- **Checkpoint selection**: Task 05's rule, applied among the GRPO checkpoints only: lowest pooled loss rate over the 6 cells → highest pooled final recall → earliest step.
- Extra diagnostics for every eval, F0 included:
  - mean fragments added per valid step;
  - the distribution of add-counts (0 / 1 / 2 / ≥3) per mode;
  - the new-gold add rate when the returned set has another fragment with the gold's title, and when it doesn't.

## 4. Report (wrap-up, after Claude's message)

`eval/grpo_v2/grpo_v2_report.md` (in Chinese, every number with its n) and `grpo_v2_results.json`. Contents:

- **A**: merge equivalence; F0 vs Task 06 differences.
- **B / C / D**: state, filter and dataset statistics as listed above.
- **E**: the training curve (csv + svg: reward, reward_std, frac_zero_std, kl), wall time, peak memory, reward recheck.
- **Pooled results**: each GRPO checkpoint's pooled loss rate and final recall; the selected checkpoint.
- **Comparison table** at the **tight** and **mid** budgets, both modes. Rows:
  - GRPO selected;
  - Task 06 v2 checkpoint-2000 (pre-registered SFT baseline, cited);
  - merged-only (F0);
  - bm25 and oracle (cited).

  Columns:
  - final recall;
  - evidence loss rate;
  - the four invalid rates;
  - single-step recall;
  - occupancy;
  - cost with caching;
  - mean fragments added per step;
  - the diagnostics (new-gold add rate, kept-gold survival, first-error causes).
- The two pre-registered PLAN §4 metrics, stated as numbers for SFT baseline vs GRPO selected.

## 5. Deliverables

- **Code** (synced to mirror): GPU / path parameterization; the all-states sampling mode; the informative dataset option; eval and pipeline scripts for GRPO v2; tests for the new options.
  - All existing tests (501) must still pass.
- **Results**: `eval/grpo_v2/`.
- **Docs**:
  - append a Task 08 section to README / NOTES / HANDOFF;
  - prefixes for Tasks 01–07 stay byte-identical;
  - write `eval/grpo_v2/TASK08_SHA256SUMS` (code, results, and external large files as path + sha256) and merge its relative entries into SHA256SUMS;
  - `scripts/checksums_task06.py check` must still pass, and the Task 07 entries must still verify.
- **Mirror**: `rsync -a` into `mirror/`, without `--delete`, never touching `mirror/.git`, no commit.

## 6. Completion criteria and final message

1. A passes. B checks are 0 mismatch. C and D complete. Training finishes with no NaN / OOM. Every saved checkpoint completes the closed-loop eval.
2. Only GPU 1 was used, and it was released at the end. Files under `/` total ≤ 300 MB.
3. All tests pass. Mirror hashes verify.
4. The final message, in Chinese, reports:
   - A's equivalence and F0's difference from Task 06;
   - the state / informative / dataset counts;
   - training time;
   - each GRPO checkpoint's pooled loss rate;
   - the selected checkpoint;
   - the tight and mid comparison with SFT v2, bm25 and oracle;
   - mean fragments added per step, SFT vs GRPO.

   End it with:

```
===AGENTDECK_DONE=== status=<ok|fail> summary=任务08 <one sentence>
```

## 7. Time estimate (Claude's, for the status messages)

| Stage | Estimate |
|---|---|
| A | ~20 min |
| F0 | ~0.5 h |
| B | ~0.9 h |
| C | ~0.5–1.5 h (measure and report) |
| D | minutes |
| E | ~5.6 h (40.4 s/step × 500, Task 07 smoke) |
| F | 4 × ~0.5 h |

Total ≈ 10–11 h after launch.
