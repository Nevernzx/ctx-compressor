"""The compressor must load and validate the F-stage toolchain before GPU admission."""
import shlex
import sys
from pathlib import Path

import pytest

from scripts import run_reranker_v1_pipeline as pipeline


@pytest.mark.parametrize("compiler_kind", ["unset", "missing", "non_executable", "executable"])
def test_compressor_preflight_runs_before_gpu_gate(tmp_path, monkeypatch, compiler_kind):
    compiler = tmp_path / "gcc"
    if compiler_kind in {"non_executable", "executable"}:
        compiler.touch()
    if compiler_kind == "executable":
        compiler.chmod(0o755)

    work = tmp_path / "task17"
    (work / "stages").mkdir(parents=True)
    historical = tmp_path / "task11" / "cache"
    fake = tmp_path / "grpo_r2_environment.sh"
    cc = "" if compiler_kind == "unset" else str(compiler)
    fake.write_text(
        f"export CC={shlex.quote(cc)} CXX={shlex.quote(str(compiler))}\n"
        f"export GRPO_CACHE={shlex.quote(str(historical))}\n"
        'export TRITON_CACHE_DIR="$GRPO_CACHE/triton" CUDA_DEVICE_ORDER=PCI_BUS_ID\n'
        'mkdir -p "$TRITON_CACHE_DIR"\n'
    )
    monkeypatch.setattr(pipeline, "WORK", work)
    monkeypatch.setattr(pipeline.r, "SHORT", tmp_path / "ipc")
    # pytest's temporary path is longer than the production socket directory.
    monkeypatch.setattr(pipeline, "ipc_record", lambda path: {"sample": str(path)})
    monkeypatch.setattr(pipeline, "GRPO_ENVIRONMENT", fake)
    monkeypatch.setattr(pipeline, "signed_project", pipeline.r.reuse.checksum)
    monkeypatch.setattr(pipeline, "READER_PY", Path(sys.executable))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "unexpected")

    class GateReached(Exception):
        pass

    def gate(name, _publish):
        proof = pipeline.read(work / "stages" / name / "receipt_environment.json")
        assert proof["CC"] == str(compiler)
        assert proof["python"] == sys.executable
        assert proof["TRITON_CACHE_DIR"] == str(work / "runtime_cache" / name / "cache/triton")
        assert proof["environment_script"]["path"] == str(fake)
        assert proof["cuda_visible_devices"] == ""
        raise GateReached

    monkeypatch.setattr(pipeline, "wait_for_gpu", gate)
    if compiler_kind == "executable":
        with pytest.raises(GateReached):
            pipeline.stage("latency-eager", "reranker_v1_latency.py", ["eager"], [], gpu=True)
    else:
        with pytest.raises(ValueError, match="CC must point to an executable C compiler"):
            pipeline.stage("latency-eager", "reranker_v1_latency.py", ["eager"], [], gpu=True)
    assert not historical.exists()
