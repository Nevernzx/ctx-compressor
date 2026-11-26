# Task 07: GRPO preparation and smoke test on GPU 0 (no formal GRPO run)

Date: 2026-11-20. Dispatched by Claude (brain / checker); executed by Codex via `ssh 5880` from 3080.
Spec: `tasks/PLAN_v2.md` §4 (reward, GRPO states), §5 (`src/reward.py`), §7 (GRPO config). Prerequisites: Tasks 01–05 delivered; Task 06 is **still running in the background** (see §1). Copy every item marked "Claude's interpretation" / "Claude's change" into NOTES.

## 0. Background and scope

On 11-20 GPU 0 of 5880 became free (until now it was occupied by someone else, so we only used GPU 1). The user asked to continue with RL (GRPO) and chose option 1: "use GPU 0 now to prepare GRPO (environment, reward plugin, GRPO script, single-step memory/speed test using v1 checkpoint-2000 as a stand-in); start the formal GRPO only after the v2 SFT results are in and the starting checkpoint is chosen". The user also allowed using GPU 0 for this. Nobody knows how long GPU 0 stays free.

So this task builds and tests everything the formal GRPO run (Task 08) needs, and measures the numbers needed to plan it. **The starting checkpoint for Task 08 is not decided yet** (v1 `checkpoint-2000` vs the v2 checkpoint selected in Task 06); everything must be parameterized by the SFT adapter path. All model outputs in this task come from the **stand-in** v1 `checkpoint-2000` and must be labelled "stand-in, not a formal result".

**Do:** A environment · B GRPO state builder · C reward plugin · D sampling probe (no training) · E GRPO smoke training · F planning numbers.

**Don't (hard rules):**
- No formal GRPO run; no evaluation on held-out or dev tasks (B/D/E use **train tasks only**; the only held-out read allowed is the read-only parser-equivalence test in C on existing Task 05 raw outputs); no reader F1; no API calls.
- **Do not touch GPU 1 or the running Task 06 pipeline**: waiter PID 1141340, training PID 1404749, and later its eval child. Do not edit, move or re-run any file the waiter will execute later (`wait_sft_v2.py`, `run_heldout_model_v2.sh`, `heldout_model_v2.py`, `heldout_diagnostics_v2.py`, `train_sft_v2*.py`, `sft_v2_*.py`, `train_environment.sh`, `src/*` modules they import, …). New code goes into **new files**; if you need a changed version of an existing module, add a new module or a new optional parameter whose default keeps behaviour byte-identical, and prove it with the existing tests. Do not write into `runs/sft_20k_v2_*`, `eval_rollouts_v2/`, `eval/sft_run_v2/`.
- Do not modify any Task 01–06 artifact. README / NOTES / HANDOFF: append only. Do not touch the `ctxc` / `reader` / `train` envs (cloning one as a read-only source is allowed), other people's environments or processes.
- **Do not touch `mirror/.git`; no git commit / push.**
- Task 06 is not finished: Claude will later ask you to finish its summary. Do not do it in this task.

## 1. Machine and resources

- **GPU 0 only**: every GPU process in this task runs with `CUDA_VISIBLE_DEVICES=0`; never 1.
- GPU 0 is someone else's card that happens to be free. **Before each GPU job**, read `nvidia-smi -i 0`; if used memory is ≥ 2 GB, do not start; wait (poll ≤ every 10 min, max 2 h) and report if it does not free up. If a job dies because another process took memory, stop GPU work and report. Never kill or touch anyone else's process. Record GPU 0 memory before/after each job.
- **Large files under `/root/userfolder/ctx-compressor/`**: `envs/grpo/`, `grpo_prep/` (states, merged model, probe, smoke runs). Point `TMPDIR`, `PIP_CACHE_DIR`, HF / ModelScope caches, vLLM / Triton / torch-extension caches to `/root/userfolder/ctx-compressor/...` before running anything.
- `/` has only about 18 GB free. Same rule as Task 06: **files this task creates or modifies under `/` (overlay) total ≤ 300 MB**, measured with `find -xdev -newer <start marker>`; record raw `df /` at start and end (not pass/fail).
- No proxy on 5880. Install packages only from sources earlier tasks already used successfully (local wheels, the pip index / mirror used by `setup_train_env.sh` / `setup_reader_env.sh`, ModelScope). No new external downloads of models.
- Time box: if the environment (A) is not working after about 1.5 h of effort, stop A, record what failed, and continue B/C/D with the existing `reader` env (vLLM) and report E as blocked.

## 2. A: environment

- New env `envs/grpo`. Target versions = what the project already uses: ms-swift 3.12.6, trl 0.24.0, peft 0.18.0, transformers 4.57.6, torch 2.8.0+cu128 (from `train`), **vLLM 0.11.0** (from `reader`, same as the held-out eval).
- Checks (write `env_check.json`): imports; `swift rlhf` accepts `--rlhf_type grpo`; vLLM can load base Qwen3-1.7B + the stand-in LoRA on GPU 0 with the **same engine settings as the Task 05 held-out eval** and reproduce the Task 05 checkpoint-2000 greedy output on ≥ 20 **train-task** step-0 prompts (compare against a fresh `reader`-env run on the same prompts; outputs must be identical); `pip freeze` saved.
- If vLLM and swift cannot coexist: record why, and use swift GRPO without vLLM (HF generation) for E. Record which path was used.

## 3. B: GRPO state builder (`scripts/build_grpo_states.py`)

PLAN §4: "first run the SFT model closed-loop over the training trajectories and use its own kept results at each step as the input states, so training sees the same states as real use."

- Parameters: base model path, SFT adapter path (or merged model path), task file, number of tasks, seed, output dir, GPU.
- Input: `data/traj_train/train_tasks.jsonl` and the train corpus. **Never** `heldout_tasks.jsonl`.
- Closed loop with **the same code path as the held-out eval**: same render, chat template, state construction, parser, four invalid categories in the same priority order (invalid_json → invalid_candidate_id → append_drop → over_budget; any of them = the step is left unchanged), greedy, `max_tokens=256`, same vLLM settings. Reuse the Task 05 eval modules by import; do not copy-paste them.
- Task sampling: `random.Random(20261120)` over train tasks; every sampled task is run in both modes × 3 tiers (tight 1876 / mid 2814 / loose 5628).
- One record per step state, containing: identifiers (task, mode, tier, budget, step, trajectory_id, source adapter path + sha256 of `adapter_model.safetensors`); the exact prompt (`prompt_token_ids` and the user text) the model saw; and the reward fields `step_reward` needs (`mode`, `budget`, `kept_ids`, `new_ids`, `ntok`, `gold`, plus whatever `apply_action` reads). `gold` must be the state's available gold exactly as `build_traj` defines it (gold seen so far ∩ (kept + new)).
- Smoke run: stand-in v1 `checkpoint-2000`, **30 train tasks** → 180 trajectories. Report state count, prompt-length distribution (max vs `max_length=8392`; never truncate silently — count and report any state that is longer), and per mode × tier: invalid rate, fraction of states whose kept area contains non-gold, fraction with lost gold.
- Checks (write `state_checks.json`): (1) step-0 user text of every sampled trajectory is byte-identical to the corresponding SFT train row (`data/sft_v2/train.jsonl` step 0; equal to v1 for step 0); (2) on ≥ 200 states, `prompt_token_ids` re-encoded via the eval's own prompt function are identical; (3) recomputing `gold` from scratch with `build_traj` logic gives the same lids on all states.

## 4. C: reward (`src/grpo_reward.py` + a swift plugin file)

- Use `src/reward.py::step_reward` **unchanged** as the core; λ is a parameter, default 0.3 (PLAN code default; the λ ablation 0 / 0.2 / 0.5 comes later).
- **Claude's interpretation — reward-invalid must equal eval-invalid.** The completion text the reward parses must be exactly what the eval parser parses (same handling of the `<think>\n\n</think>\n\n` response prefix, whitespace, trailing text, end-of-sequence). A completion is −1 **iff** the eval would classify it as one of the four invalid categories. If `step_reward`'s own parsing differs from the eval parser in any edge case, put a wrapper in front that applies the eval parser first; list every difference you found.
- Register it as a swift ORM (`--external_plugins` / `--reward_funcs`); the state is passed through dataset columns.
- Tests: unit tests for every branch (each invalid category, recall < 1, recall = 1 with the budget bonus, empty gold); an equivalence test that runs the plugin on **all existing Task 05 `checkpoint-2000` held-out raw outputs** (read-only, `eval_rollouts/checkpoint-2000/steps.jsonl`) and checks reward = −1 ⇔ the recorded `invalid_reason` is not null, and otherwise recall matches the recorded keep_after. Write `reward_equivalence.json` (n, mismatches = must be 0).

## 5. D: sampling probe (no training)

Purpose: GRPO only learns from prompt groups whose 8 rewards are not all equal. Measure how many such groups exist before planning the formal run.

- From the B smoke state pool, sample 192 states with `random.Random(20261120)`: 32 per mode × tier.
- For each, generate 8 completions with the stand-in model: temperature 1.0, top_p 1.0, `max_tokens=256`, fixed seed; compute rewards with the C plugin (λ = 0.3).
- Report per mode × tier (n each) and pooled: mean reward; invalid rate (by category); **fraction of groups with zero reward std**; mean within-group std; mean of (best-of-8 reward − greedy reward); fraction of groups where some sample beats the greedy action. Write `sampling_probe.json`.

## 6. E: GRPO smoke training (provisional config, not the formal one)

- **Claude's interpretation — KL reference = the SFT model.** Merge the stand-in SFT LoRA into the base weights (`swift export --merge_lora` or equivalent) → `grpo_prep/merged_v1_ckpt2000/`, and train a **new** LoRA (r16 / α32 / dropout 0.05, all-linear, same as SFT) on top of the merged model. Reason: if the SFT adapter were loaded and trained directly, swift's LoRA reference model (adapter disabled) would be the base Qwen3, which outputs 100% invalid JSON. Verify the merged model reproduces the adapter's greedy outputs on ≥ 20 train-task prompts (identical outputs, or report the differences).
- Provisional settings (record everything): 8 generations per prompt; temperature 1.0; `max_completion_length=256`; `max_length=8392`; bf16; gradient checkpointing; beta (KL) 0.04; lr 5e-6 cosine with 3% warmup; seed 20261120; prompts = the B smoke states (train tasks only). Use vLLM colocate if A succeeded (choose a `vllm_gpu_memory_utilization` that fits next to training; record it), otherwise HF generation. Choose batch / accumulation so that one optimizer step covers 4 prompts × 8 completions; record the exact swift arguments.
- Run **20–30 optimizer steps**. Measure: peak GPU memory (torch reserved and whole-card); seconds per optimizer step, split into generation / reward / forward-backward; reward mean and std per step; fraction of zero-std groups per step; KL; any NaN / OOM. Save a checkpoint at the end. Write `smoke_train.json` and `smoke_train_log.csv`. Do not interpret the reward trend; 30 steps prove nothing.
- **Eval plumbing check**: load the smoke LoRA on top of the merged model with the eval code path (base model path must be a parameter) and run the closed loop on 2 **train** tasks × 2 modes × 3 tiers. Only check that it runs and produces valid records; report the numbers but draw no conclusion.

## 7. F: planning numbers (`projection.json`, plus a table in the report)

From the measured speeds, project wall time on one RTX 5880 for:
- state generation over N train tasks, N ∈ {200, 400, 800} (both modes × 3 tiers each);
- one GRPO epoch over 1,000 / 2,000 / 4,000 prompts × 8 completions;
- the closed-loop held-out eval of one GRPO checkpoint (use Task 05's measured time per checkpoint).

Also list, as **open choices for Claude** (do not decide): lr, beta, number of prompts, mode / tier mix, whether to filter or oversample zero-std groups (with the D numbers), λ, and the save / eval interval. Give the data behind each one.

## 8. Deliverables

- **Code** (project directory, synced to mirror): `scripts/setup_grpo_env.sh`, `scripts/build_grpo_states.py`, `src/grpo_reward.py` + the swift plugin, `scripts/train_grpo.sh` (parameterized: merged base path, data, GPU index, output dir, all hyperparameters as flags), the probe script, the report script, unit tests. All existing tests (472) still pass, plus the new ones.
- **Results** (`eval/grpo_prep/`, synced to mirror): `grpo_prep_report.md` (in Chinese; every number with its n; every model-output number marked "替身 v1 checkpoint-2000，非正式结果"), `env_check.json`, `env_freeze.txt`, `state_checks.json`, `reward_equivalence.json`, `sampling_probe.json`, `smoke_train.json`, `smoke_train_log.csv`, `projection.json`, `resource_check.json` (GPU 0 before / after; GPU 1 untouched; `/` task-owned bytes; raw `df`).
- **Documentation**: append to README / NOTES / HANDOFF; update SHA256SUMS (large files: path + sha256); Task 01–06 invariant files and document prefixes still verify. rsync into `mirror/` with `-a`, **without `--delete`, never touching `mirror/.git`**.

## 9. Completion criteria

1. A works (or its failure is documented and E is reported as blocked); B / C checks all pass (0 mismatches); D and E ran on GPU 0 only; GPU 1 and the Task 06 pipeline were not touched.
2. GPU 0 released at the end (no process of ours left on it); `/` task-owned files ≤ 300 MB.
3. All tests pass.
4. The final message reports in Chinese: environment path / versions; state count and invalid rate of the smoke states; reward-equivalence n and mismatches; the zero-std group fraction per mode × tier; smoke training peak memory and seconds per optimizer step; the projection table; the open choices. End with:

```
===AGENTDECK_DONE=== status=<ok|fail> summary=任务07 <one sentence>
```
