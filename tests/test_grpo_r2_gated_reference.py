from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import gate_grpo_r2_phase_a as phase_a


@pytest.mark.parametrize("failed", [False, True])
def test_reference_uses_selected_adapter_and_releases_engine(tmp_path, monkeypatch, failed):
    monkeypatch.setattr(phase_a, "RUN", tmp_path)
    shutdown = Mock()
    engine = SimpleNamespace(llm_engine=SimpleNamespace(engine_core=SimpleNamespace(shutdown=shutdown)))
    monkeypatch.setattr(phase_a.protocol, "create_engine", lambda args: (engine, None, {}))
    start = {"path":str(tmp_path/"checkpoint-3000"), "adapter_sha256":"fixture"}
    def generate(opts):
        assert opts.gpu == 1 and opts.source_adapter == opts.adapter == Path(start["path"])
        assert opts.base_model == phase_a.BASE and opts.compare is None and opts.prompts == phase_a.FIXTURE
        phase_a.protocol.create_engine(opts)
        if failed:
            raise ValueError("mock generation failed")
        (opts.output_dir/"greedy_reference.json").write_text("{}")
    monkeypatch.setattr(phase_a.protocol, "reference", generate)
    if failed:
        with pytest.raises(ValueError, match="mock generation failed"):
            phase_a.reference("ref_adapter", start)
    else:
        receipt = phase_a.reference("ref_adapter", start)
        assert receipt["reference"]["bytes"] == 2
    shutdown.assert_called_once()
