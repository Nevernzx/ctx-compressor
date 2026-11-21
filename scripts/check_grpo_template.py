#!/usr/bin/env python3
"""Task 07 (Claude): swift's GRPO prompt encoding must equal the eval prompt ids, token for token.

For N state rows, encode messages=[user_text] with the swift qwen3 template using the
Task 04 protocol options (enable_thinking False, non-thinking prefix) in the modes GRPO
uses for rollout ('vllm') and compare with the stored eval prompt_token_ids. CPU only.
"""
import argparse, json, random, sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.sft_protocol import TEMPLATE_OPTIONS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    from swift.llm import InferRequest, get_model_tokenizer, get_template
    _, processor = get_model_tokenizer(str(args.model), load_model=False, model_type="qwen3", use_hf=False)
    opts = {k: v for k, v in TEMPLATE_OPTIONS.items() if k not in ("padding_side", "padding_free")}
    rows = [json.loads(l) for l in open(args.states)]
    picked = random.Random(20261120).sample(rows, min(args.n, len(rows)))
    result = {"n": len(picked), "template_options": opts, "modes": {}}
    for mode in ("vllm", "pt"):
        template = get_template(processor=processor, **opts)
        template.set_mode(mode)
        mismatch, examples = 0, []
        for r in picked:
            enc = template.encode(InferRequest(messages=[{"role": "user", "content": r["messages"][0]["content"]}]))
            ids = list(enc["input_ids"])
            if ids != r["prompt_token_ids"]:
                mismatch += 1
                if len(examples) < 2:
                    tail = processor.decode(ids[-12:]) if hasattr(processor, "decode") else ids[-12:]
                    examples.append({"trajectory_id": r["trajectory_id"], "step": r["step"],
                                     "swift_len": len(ids), "eval_len": len(r["prompt_token_ids"]),
                                     "swift_tail": tail})
        result["modes"][mode] = {"mismatches": mismatch, "examples": examples}
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
