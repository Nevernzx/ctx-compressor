#!/usr/bin/env python3
"""Task 07 (Claude): merge an SFT LoRA into Qwen3-1.7B on CPU (fp32 merge, bf16 save).

GRPO then trains a NEW LoRA on the merged model, so swift's LoRA reference model
(adapter disabled) is the SFT policy, not the base model (which outputs 100% invalid JSON).
"""
import argparse, hashlib, json, os, shutil, sys, time
from pathlib import Path

PREP = Path("/root/userfolder/ctx-compressor/grpo_prep")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, default=Path("/root/userfolder/ctx-compressor/models/Qwen3-1.7B"))
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        sys.exit("merge runs on CPU: set CUDA_VISIBLE_DEVICES=''")
    if not args.out.resolve().is_relative_to(PREP) or args.out.exists():
        sys.exit(f"output must be a new directory under {PREP}")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    t0 = time.time()
    base = AutoModelForCausalLM.from_pretrained(str(args.base), torch_dtype=torch.float32, local_files_only=True)
    merged = PeftModel.from_pretrained(base, str(args.adapter)).merge_and_unload()
    merged = merged.to(torch.bfloat16)
    merged.save_pretrained(str(args.out), safe_serialization=True)
    copied = []
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                 "generation_config.json", "special_tokens_map.json", "added_tokens.json"):
        if (args.base / name).exists():
            shutil.copy2(args.base / name, args.out / name)
            copied.append(name)
    record = {"base": str(args.base), "adapter": str(args.adapter),
              "adapter_sha256": sha256(args.adapter / "adapter_model.safetensors"),
              "merge": "peft merge_and_unload in float32 on CPU, saved as bfloat16 safetensors",
              "tokenizer_files_copied_from_base": copied,
              "weights": {p.name: sha256(p) for p in sorted(args.out.glob("*.safetensors"))},
              "seconds": time.time() - t0}
    (args.out / "merge_record.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(record))


if __name__ == "__main__":
    main()
