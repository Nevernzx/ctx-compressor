# Task 09: how much does "add a second fragment when unsure" help? Training-free hedge test on SFT v2

Written by Claude on 2026-11-21 at about 09:20. Codex executes; Claude checks. Read this whole file before starting.

## 0. Background and scope

**Why this task exists.**
- Task 08's GRPO gave no gain beyond merge noise. Claude's analysis is in `claude_checks/grpo_failure_out_20261121.txt` on 3080, and its pointer is in NOTES. In short:
  - SFT v2 adds exactly one fragment per step;
  - not one of 177,257 samples added two;
  - so GRPO could never try "hedging".
- The user chose option A1 on 2026-11-21 ("A1 B1"). A1 means: without training, measure the ceiling of adding the second-ranked candidate when the model is unsure, and what it costs.

**Model.** SFT v2 checkpoint-2000: base `models/Qwen3-1.7B` plus adapter `runs/sft_20k_v2_20261118/checkpoints/checkpoint-2000` (sha256 `7253eae0…1f9c`). This is exactly the Task 06 pre-registered SFT baseline. No merged model, no GRPO adapter.

**Pre-registered, not to be changed after seeing results.**
- The candidate score, the hedge rule, the τ grid and the τ selection rule all follow §2.
- τ is chosen **on train states only**. The held-out set is used only for the final runs in §3.
- Report numbers with their n. No "worked / didn't work" conclusions. One run, point estimates, no CIs.

## 1. Machine, resources, hard rules

Same as Task 08 §1:
- 5880, formal project `/root/workplace/ctx-compressor`.
- Large files only under `/root/userfolder/ctx-compressor/`. New dirs: `hedge_v2/`, `eval_rollouts_hedge_v2/`. Small results: `eval/hedge_v2/`.
- **GPU 1 only.** `CUDA_VISIBLE_DEVICES=1`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`. Before each GPU job, GPU 1 must use < 2,048 MiB. Never touch GPU 0.
- Files written under `/` must total ≤ 300 MB.
- Use the existing `reader` or `grpo` env. No pip installs. If something is missing, print `NEED:`.
- Never delete anything and never overwrite outputs. Don't modify Task 05–08 artifacts.
- Don't touch `mirror/.git`, and don't commit. Claude commits.
- Kill processes only by exact PID.
- Close every sub-agent before your final message.

## 2. Candidate score and hedge rule (pre-registered)

**Output format.** Greedy outputs look like `{"add":[N],"drop":[...]}` with no spaces. Task 06 data: 13,830 of 13,956 outputs are `{"add":[N],"drop":[]}`. Candidate local ids have 1–3 digits, and Qwen tokenizes each digit as its own token.

**Score.** For each step, for each candidate local id c in `new_ids` that is not in `kept_ids`:
- score(c) = the summed log-probability, under the step's exact eval prompt (same `inference_prompt` ids as the Task 06 eval), of the assistant text `{"add":[` + str(c) + `],"`, computed by teacher forcing.
- Choose the implementation yourself (vLLM prefix requests reusing the prefix cache, or HF with a KV cache and batched candidate continuations). It must pass these checks, on 300 random valid steps from the Task 06 v2 checkpoint-2000 eval:
  1. argmax score equals the greedy add id (whenever greedy added exactly one id) in **≥ 99%** of steps;
  2. for the greedy id, score(c) equals the sum of vLLM generation logprobs of the same tokens within 0.02 nats;
  3. re-encoding the greedy raw text gives the generated token ids.

  Report all three.
- **Confidence.** p(c) = softmax of the scores over that step's candidates. p1 is the largest; top2 is the candidate with the second-largest p.

**Hedge rule (wrapper around greedy).**
- Run the normal greedy generation and parse it exactly as in the eval.
- If the output is invalid, keep the eval's no-op behaviour and don't hedge.
- Otherwise let extra = the highest-p candidate that is not already in the greedy `add` list and not in `kept_ids`. Hedge when both hold:
  - **p1 < τ**, where p1 is the probability of the highest-scored candidate overall;
  - adding extra still fits the budget after the greedy action (greedy drops applied, greedy adds kept).
- When hedging, the executed action is the greedy action with extra appended to `add`.
- Build the executed action as the JSON string `{"add":[a,extra],"drop":[...]}` and pass it through the **same** `apply_model_output` / `score_completion` code, so validity, budget and recall use identical logic. If the budget doesn't fit, don't hedge. Never create an over-budget action.

**τ grid.** {0 (never), 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01 (always when it fits)}.

**τ selection (train only).**
- On all 22,158 Task 08 train states `grpo_v2/states/states.jsonl`, which come from this exact SFT v2 adapter policy on 400 train tasks:
  - compute the open-loop executed action for every τ;
  - compute its PLAN §4 step reward with `src/grpo_reward.py::score_completion`, λ=0.3.
- **τ\*** = the τ with the highest mean step reward; on a tie, the smaller τ.
- Also report, per τ (pooled and per cell): hedge rate; current-gold miss rate (current gold returned new but not in keep_after); mean extra tokens per step; mean reward; and the hedge hit rate (extra is gold).

## 3. Held-out closed-loop runs (Task 06 protocol)

- Same protocol as Task 05/06:
  - 249 held-out tasks × 2 modes × 3 budgets = 1,494 trajectories;
  - greedy, `max_tokens` 256;
  - same engine parameters, seed and LoRA loading as `heldout_model_v2.py`;
  - same invalid handling, same metrics and diagnostics code.
- The only change is the hedge wrapper. Each run's output goes to `eval_rollouts_hedge_v2/<name>/`.
- Runs:
  - **H0** `nohedge`: the wrapper with τ=0. A consistency check: report differences against Task 06 v2 checkpoint-2000, per cell and pooled, and the number of raw outputs that differ on identical states.
  - **H1** `tau_star`: τ = τ\*.
  - **H2** `always`: τ = 1.01, the ceiling of adding the second candidate whenever it fits.
- Pre-checks: held-out sha matches Task 05; step-0 user text and prompt ids are byte-identical to the Task 06 eval.
- Extra per-run stats:
  - hedge rate per step;
  - hedge hit rate;
  - add-count distribution (0 / 1 / 2 / ≥3);
  - mean kept tokens (occupancy);
  - new-gold add rate split by same-title / different-title (Task 08 `extra.json` definitions).
- Run §2 (train scoring) and §3 as one detached pipeline with a status file `hedge_v2/pipeline_status.json`, following the Task 08 pattern (stage receipts, `require` checks, no automatic retry).
  - Do §2's three implementation checks interactively first.
  - Then launch, verify it is running, print `WAITING:` and end the turn **without a DONE line**. Claude will message you for the wrap-up.

## 4. Report (wrap-up)

`eval/hedge_v2/hedge_v2_report.md` (in Chinese, every number with its n) and `hedge_v2_results.json`.

1. **Implementation checks and τ selection.**
   - The three score checks.
   - The τ table on train states.
   - τ\*.
2. **H0 consistency** with Task 06.
3. **Comparison table, pooled and at the tight and mid budgets, both modes.**
   - Rows: H0, H1, H2, SFT v2 checkpoint-2000 (Task 06, cited), GRPO checkpoint-500 (Task 08, cited), bm25 and oracle (cited).
   - Columns: final recall, evidence loss rate, invalid rates, single-step recall, occupancy, **cost with caching**, mean fragments added per step, hedge rate, hedge hit rate.
4. **Cost vs loss** for H0, H1, H2 and oracle: a small table, plus an svg of pooled cached cost (x) against pooled evidence loss rate (y).

## 5. Deliverables

- **Code**: scoring, hedge wrapper, pipeline and report scripts, with tests. All existing tests (533) must still pass.
- **Results**: `eval/hedge_v2/`, including `TASK09_SHA256SUMS` (code, results, and external paths with sha256). Merge its relative entries into SHA256SUMS. `checksums_task06.py check` and the Task 07/08 manifests must still verify.
- **Docs**: append a Task 09 section to README / NOTES / HANDOFF. Earlier prefixes stay byte-identical.
- **Mirror**: `rsync -a` into `mirror/`, without `--delete`, without touching `.git`, no commit.

## 6. Completion and final message

All checks pass, every run completes, only GPU 1 was used and it is released, files under `/` total ≤ 300 MB, and all tests pass.

The final message is in Chinese and reports:
- the score checks;
- τ\* and the train τ table;
- H0 vs Task 06;
- H0 / H1 / H2 pooled loss, recall and cached cost;
- the tight and mid table.

End with:

```
===AGENTDECK_DONE=== status=<ok|fail> summary=任务09 <one sentence>
```

## 7. Time estimate (Claude)

| Stage | Estimate |
|---|---|
| Implementation checks | ~30 min |
| Train scoring, 22,158 states × ~20 candidates | ~0.5–1 h |
| Each held-out run | ~0.6–0.8 h (greedy plus scoring) |

Total ≈ 3–3.5 h after launch.
