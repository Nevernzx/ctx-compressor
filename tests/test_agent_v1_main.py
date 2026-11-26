"""Offline checks for the Task 20 main scheduler's frozen deployment and parking invariant."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import agent_v1_main as main, agent_v1_residency as residency


def test_task20_main_changes_only_permitted_agent_reservation():
    frozen = {
        "agent": {"params": {"gpu_memory_utilization": 0.65, "max_num_seqs": 32, "disable_cascade_attn": True},
                  "generation": {"temperature": 0, "max_tokens": 256, "seed": 20261117, "stop": ["\nObservation"]}},
        "r4": {"params": {"gpu_memory_utilization": 0.7, "enable_lora": True}, "sampling": {"temperature": 0}},
    }
    prior = deepcopy(frozen)
    effective = main.effective_assets(frozen)
    assert frozen == prior
    assert effective["agent"]["params"]["gpu_memory_utilization"] == 0.50
    effective["agent"]["params"]["gpu_memory_utilization"] = 0.65
    assert effective == prior


def test_task20_weight_transfer_checks_same_worker_and_kv():
    before = {"pid": 17, "kv_buffers": {"12345": 8192},
              "param_layout_sha256": "same-layout", "param_bytes": {"cuda:0": 4096}, "sampled_weights_sha256": "weights"}
    parked = {**before, "param_bytes": {"cpu": 4096}}
    residency.check_transfer(before, parked, before)
    with pytest.raises(ValueError, match="KV allocations"):
        residency.check_transfer(before, parked, {**before, "kv_buffers": {"54321": 8192}})
    with pytest.raises(ValueError, match="weight sample"):
        residency.check_transfer(before, {**parked, "sampled_weights_sha256": "other"})


def test_task20_cache_reset_accepts_v1_none_and_rejects_active_requests():
    reset = Mock(return_value=None)
    llm = SimpleNamespace(reset_prefix_cache=reset,
                          llm_engine=SimpleNamespace(get_num_unfinished_requests=lambda: 0))
    residency.clear_cache(llm)
    reset.assert_called_once()
    llm.llm_engine.get_num_unfinished_requests = lambda: 1
    with pytest.raises(ValueError, match="in flight"):
        residency.clear_cache(llm)
    reset.assert_called_once()



def test_task20_uses_named_worker_commands_and_keeps_r4_constructor_unchanged():
    assets = {"agent": {"params": {}}, "r4": {"params": {"max_tokens": 256}}}
    runtime = main.agent_runtime_assets(assets)
    assert runtime["agent"]["params"]["worker_extension_cls"] == residency.EXTENSION
    assert assets["agent"]["params"] == {} and runtime["r4"] == assets["r4"]
    rpc = Mock(side_effect=[[{"result": "state"}], [{"param_bytes": {"cpu": 1024}}]])
    llm = SimpleNamespace(collective_rpc=rpc,
                          llm_engine=SimpleNamespace(get_num_unfinished_requests=lambda: 0))
    assert residency.state(llm, True) == {"result": "state"}
    assert residency.move(llm, "cpu", True) == {"param_bytes": {"cpu": 1024}}
    assert [call.args[0] for call in rpc.call_args_list] == ["task20_cuda_state", "task20_move_weights"]


def test_task20_pilot_has_separate_paths_and_requests_two_tasks(monkeypatch):
    for key in ("PILOT", "ROOT", "SMALL", "STATUS", "PID", "LOGS", "CACHE", "PRE", "CONFIG",
                "GPU_RECEIPT", "GPU_LOG", "EPISODES"):
        monkeypatch.setattr(main, key, getattr(main, key))
    formal_dir, formal_cfg = main.ROOT, main.CONFIG
    assert main.selected_task_count() == 300
    main.enable_pilot()
    assert main.ROOT == main.PILOT_ROOT and main.ROOT != formal_dir and main.CONFIG != formal_cfg
    assert main.selected_task_count() == 2
    assert main.argv("python", "worker")[-2:] == ["worker", "--pilot"]
