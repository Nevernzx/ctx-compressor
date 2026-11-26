"""CPU regression coverage for Task 20's archived-result closeout."""
from ast import literal_eval

import pytest

from scripts import agent_v1_recover_main as recovery, agent_v1_report as report


def test_gold_retention_counting_is_disjoint():
    counts = report.gold_distribution({(2, 0): 3, (2, 1): 4, (1, 1): 2, (4, 2): 5, (4, 4): 6})
    assert counts == {
        "n": 20,
        "counts": {"zero": 3, "one_incomplete": 4, "multiple_incomplete": 5, "all": 8},
        "by_total_gold_and_kept": {"1": {"1": 2}, "2": {"0": 3, "1": 4}, "4": {"2": 5, "4": 6}},
    }


@pytest.fixture
def failed_run():
    state = {
        "status": "failed", "gpu_index": 1, "stage": "main", "stage_pid": None, "automatic_retry": False,
        "lock": {"held": False}, "last_monitor": {"pid": 17, "alive": False, "oom": False, "nonfinite": False},
        "stages": [{"status": "failed", "stage": "main", "exit_code": 1, "uses_gpu": True, "pid": 17}],
        "first_error": "Task 20 storage rule a triggered (closing root use minus opening root use: 350000001 bytes)",
    }
    storage = {
        "status": "failed", "violation": {"rule": "a", "bytes": 350_000_001}, "limit_bytes": 300_000_000,
        "min_free_limit_bytes": 5_000_000_000, "sustained_growth_bytes": 350_000_001,
        "root_before_bytes": 940_000_000_000, "root_after_bytes": 940_350_000_001,
    }
    return state, storage


@pytest.mark.parametrize("rule", ["b", "c"])
def test_archive_recovery_is_limited_to_rule_a(failed_run, rule):
    state, storage = failed_run
    recovery.terminal(state, storage)
    storage["violation"]["rule"] = rule
    with pytest.raises(ValueError, match=r"original rule \(a\) failure"):
        recovery.terminal(state, storage)


def test_original_attempt_must_release_gpu_lock(failed_run):
    state, storage = failed_run
    state["lock"]["held"] = True
    with pytest.raises(ValueError, match="terminal, released GPU"):
        recovery.terminal(state, storage)


def test_recovery_reads_logged_engine_flags(tmp_path):
    agent = {
        "model": "local/qwen", "tokenizer": "local/qwen", "dtype": "bfloat16", "seed": 9,
        "max_model_len": 40960, "enable_prefix_caching": True, "enable_chunked_prefill": True,
        "gpu_memory_utilization": 0.5, "max_num_batched_tokens": 8192, "max_num_seqs": 32,
        "enforce_eager": True, "disable_cascade_attn": True,
    }
    r4 = {**agent, "model": "local/merged", "tokenizer": "local/merged", "enable_lora": True, "max_cpu_loras": 5}
    proof = {"assets": {"agent": {"params": agent}, "r4": {"params": r4}}}
    args = recovery.main.agent_runtime_assets(proof["assets"])
    log = tmp_path / "worker.log"
    entry = "INFO non-default args: " + repr(args["agent"]["params"])
    log.write_text(entry + "\nINFO non-default args: " + repr(r4) + "\n")
    evidence = recovery.engine_launches(log, proof)
    assert [(row["engine"], row["line"]) for row in evidence] == [("agent", 1), ("r4", 2)]
    assert evidence[0]["effective_disable_cascade_attn"] is True
    params = literal_eval(entry.split("args: ", 1)[1])
    params["disable_cascade_attn"] = False
    log.write_text("INFO non-default args: " + repr(params) + "\n")
    with pytest.raises(ValueError, match="settings differ"):
        recovery.engine_launches(log, proof)

