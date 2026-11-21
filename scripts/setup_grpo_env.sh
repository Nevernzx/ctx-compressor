#!/usr/bin/env bash
# Build an independent venv from Tasks 03/04 wheels and their approved index.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/grpo_environment.sh
export CUDA_VISIBLE_DEVICES=''
mkdir -p eval/grpo_prep "$GRPO_PREP/logs"
exec 8>"$GRPO_PREP/setup.lock"
flock 8
base_python="$PWD/$(cat logs/environment.path)/bin/python"
if [[ ! -x "$GRPO_ENV/bin/python" ]]; then
    "$base_python" -B -m venv --copies "$GRPO_ENV"
fi
py="$GRPO_ENV/bin/python"
"$py" -B -c 'import os, sys; assert sys.prefix == os.environ["GRPO_ENV"] and sys.version_info[:2] == (3, 11)'

# Recover the original cached wheel names from their archive metadata. The pip
# HTTP cache uses hash names. Link to those archives without writing the cache.
"$py" -B - <<'PY'
from pathlib import Path
from zipfile import ZipFile
from email.parser import BytesParser
from pip._vendor.packaging.tags import sys_tags
import os, re

cache = Path(os.environ["GRPO_ROOT"]) / "cache/pip"
prep = Path(os.environ["GRPO_PREP"])
wheels = prep / "wheelhouse"
wheels.mkdir(exist_ok=True)
tag_order = {str(tag): i for i, tag in enumerate(sys_tags())}
for source in [*cache.glob("wheels/**/*.whl"), *cache.glob("http-v2/**/*.body")]:
    with source.open("rb") as f:
        if f.read(4) != b"PK\x03\x04":
            continue
    with ZipFile(source) as archive:
        entries = [p for p in archive.namelist()
                   if p.count("/") == 1 and p.endswith(".dist-info/WHEEL")]
        if len(entries) != 1:
            continue
        info = BytesParser().parsebytes(archive.read(entries[0]))
        prefix = entries[0].split(".dist-info/")[0]
        tags = [tag for tag in info.get_all("Tag", []) if tag in tag_order]
        if not tags:
            continue
        tag = min(tags, key=tag_order.get)
        destination = wheels / f"{prefix}-{tag}.whl"
        if not destination.exists():
            destination.symlink_to(source)

requirements = {}
for lock in ["reader", "train"]:
    for line in Path(f"requirements.{lock}.lock.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==")
        requirements[re.sub("[-_.]+", "-", name.lower())] = f"{name}=={version}"
# The reader's instrumentator 8.1 requires Starlette >=1, while Swift's
# Gradio 5.50 requires <1. vLLM permits instrumentator 7.1.
requirements["prometheus-fastapi-instrumentator"] = "prometheus-fastapi-instrumentator==7.1.0"
(prep / "requirements.grpo.txt").write_text("\n".join(sorted(requirements.values())) + "\n")
PY
if ! compgen -G "$GRPO_PREP/wheelhouse/prometheus_fastapi_instrumentator-7.1.0-*.whl" >/dev/null; then
    "$py" -B -m pip download --no-deps --only-binary=:all: \
        --index-url https://pypi.tuna.tsinghua.edu.cn/simple --timeout 45 --retries 2 \
        --dest "$GRPO_PREP/wheelhouse" prometheus-fastapi-instrumentator==7.1.0
fi
"$py" -B -m pip install --disable-pip-version-check --no-compile --no-index \
    --find-links "$GRPO_PREP/wheelhouse" -r "$GRPO_PREP/requirements.grpo.txt"
"$py" -B -m pip check
"$py" -B -m pip freeze --all > eval/grpo_prep/env_freeze.txt
"$GRPO_ENV/bin/swift" rlhf --rlhf_type grpo --help > "$GRPO_PREP/logs/swift_rlhf_help.log"
"$py" -B - <<'PY'
import importlib, importlib.metadata as md, json, os, sys, datetime
from pathlib import Path
from transformers import HfArgumentParser
from swift.llm.argument.rlhf_args import RLHFArguments

modules = {"ms-swift": "swift", "trl": "trl", "peft": "peft", "transformers": "transformers",
           "torch": "torch", "vllm": "vllm"}
versions = {dist: md.version(dist) for dist in modules}
for module in modules.values():
    importlib.import_module(module)
import torch
from swift.trainers.rlhf_trainer.grpo_trainer import GRPOTrainer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
assert torch.__version__ == "2.8.0+cu128"
assert not torch.cuda.is_initialized()
expected = {"ms-swift": "3.12.6", "trl": "0.24.0", "peft": "0.18.0", "transformers": "4.57.6",
            "torch": "2.8.0", "vllm": "0.11.0"}
assert versions == expected, versions
parser = HfArgumentParser(RLHFArguments)
namespace = parser.parse_args(["--rlhf_type", "grpo"])
assert namespace.rlhf_type == "grpo"
result = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "prefix": sys.prefix,
          "python": sys.version, "packages": versions, "torch_build": torch.__version__,
          "torch_cuda_version": torch.version.cuda, "imported": list(modules.values()) +
          ["swift.trainers.rlhf_trainer.grpo_trainer.GRPOTrainer", "vllm.LLM", "vllm.SamplingParams", "vllm.lora.request.LoRARequest"],
          "swift_rlhf_cli_help_exit": 0, "swift_rlhf_parser_value": namespace.rlhf_type,
          "cuda_initialized_by_check": torch.cuda.is_initialized(), "pip_check_exit": 0,
          "install_source": "Tasks 03/04 wheels plus their Tsinghua index for prometheus-fastapi-instrumentator 7.1.0 (reader 8.1.0 conflicts with train Gradio/Starlette); isolated venv, shared pins prefer train lock",
          "pip_freeze": "eval/grpo_prep/env_freeze.txt"}
Path("eval/grpo_prep/env_check_cpu.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
PY
