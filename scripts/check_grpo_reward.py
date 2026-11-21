"""Read-only parser/ORM equivalence on the frozen Task 05 raw completion file."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.grpo_swift_plugin import CtxStepReward
from src.grpo_reward import STATE_FIELDS, decode_rollout, score_completion
from src.heldout_eval import INVALID_REASONS
from src.sft_protocol import EMPTY_THINK

EXTERNAL = Path("/root/userfolder/ctx-compressor")
SOURCE = EXTERNAL / "eval_rollouts/checkpoint-2000/steps.jsonl"
LABEL = "替身 v1 checkpoint-2000，非正式结果"
PARSER_DIFFERENCES = [
    "The eval parser rejects duplicate JSON keys anywhere; the old reward's json.loads keeps the last.",
    "The eval parser rejects NaN, Infinity and -Infinity anywhere; the old reward can accept them in ignored metadata.",
    "Eval requires str. The old reward also accepts bytes or bytearray containing a valid JSON object.",
    "Eval catches RecursionError for excessive JSON nesting. The old reward can propagate it.",
]


def check_file(model):
    from transformers import AutoTokenizer

    baseline = json.loads((ROOT / "eval/sft_run/model_rollout_audit.json").read_text())
    expected = next(x for x in baseline["strategies"] if x["strategy"] == "checkpoint-2000")
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True, use_fast=True)
    orm = CtxStepReward(tokenizer)
    errors = Counter()
    counts = Counter()
    examples = []
    digest = hashlib.sha256()
    total = 0
    with SOURCE.open("rb") as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            total += 1
            meta, saved = row["generation_metadata"], row["state"]
            state = {k: saved[k] for k in STATE_FIELDS}
            tail = meta["prompt_token_ids"][-24:]
            tokens = meta["output_token_ids"]
            raw = row["raw_output"]
            packet = json.dumps({**state, "prompt_token_ids_tail": tail})
            reward, = orm([EMPTY_THINK + raw], [packet], [tokens])
            decoded = decode_rollout(tokenizer.backend_tokenizer, tail, tokens)
            detail = score_completion(raw, state, orm.lam)
            reason = row["invalid_reason"]
            counts[reason or "valid"] += 1

            pid_map = saved["pid_by_lid"]
            gold_pids = {pid_map[str(lid)] for lid in state["gold"]}
            keep_after = set(row["keep_after"])
            recall = len(keep_after & gold_pids) / len(gold_pids) if gold_pids else 1.0
            expected_reward = -1.0 if reason else recall
            if not reason and recall == 1.0:
                expected_reward += orm.lam * (1 - row["used_after"] / state["budget"])
            checks = {
                "decoded_text": decoded == raw,
                "invalid_sentinel": (reward == -1.0) == (reason is not None),
                "invalid_category": detail["invalid_reason"] == reason,
                "keep_after": [pid_map[str(lid)] for lid in detail["keep_after"]] == row["keep_after"],
                "available_gold_recall": detail["recall"] == recall == row["recall"],
                "used_after": detail["used_after"] == row["used_after"],
                "reward_value": reward == detail["reward"] == expected_reward,
            }
            failed = [k for k, passed in checks.items() if not passed]
            if failed:
                errors["records"] += 1
                errors.update(failed)
                if len(examples) < 12:
                    examples.append({"trajectory_id": row["trajectory_id"], "step": row["step"], "failed": failed})
    artifact = expected["steps_artifact"]
    source_ok = (digest.hexdigest() == artifact["sha256"] and total == expected["n_step_records"]
                 and SOURCE.stat().st_size == artifact["bytes"])
    return {
        "model_output_label": LABEL, "source": str(SOURCE), "sha256": digest.hexdigest(),
        "matches_task05_audited_artifact": source_ok, "n": total, "n_valid": counts["valid"],
        "invalid_categories": {k: counts[k] for k in INVALID_REASONS},
        "lambda": orm.lam, "mismatches": errors["records"], "mismatches_by_check": dict(errors),
        "examples": examples, "status": "ok" if source_ok and not errors else "fail",
        "core_parser_differences": PARSER_DIFFERENCES,
        "protocol": "Task 05 uses generated text only. The ORM reconstructs this from generated token IDs using "
                    "the same fast DecodeStream and prompt suffix. The renderer-added think block is outside raw "
                    "text. No whitespace stripping, EOS-string replacement or trailing-text repair is performed.",
        "recall_reference": "Map saved available-gold lids to corpus pids; intersect with recorded keep_after pids.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=EXTERNAL / "models/Qwen3-1.7B")
    parser.add_argument("--output", type=Path, default=ROOT / "eval/grpo_prep/reward_equivalence.json")
    args = parser.parse_args()
    result = check_file(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "n", "mismatches", "invalid_categories")}))
    sys.exit(0 if result["status"] == "ok" else 1)
