"""The held-out reader loader stores each task's ID in the map key."""
import sys
from types import SimpleNamespace

from scripts import reranker_v1_rollouts as rollout


def test_prepare_heldout_restores_map_key_before_closed_loop(tmp_path, monkeypatch):
    task_id = 37
    questions = [f"Question {i}" for i in range(4)]
    example_ids = [f"example-{i}" for i in range(4)]
    task_row = {
        "questions": questions,
        "example_ids": example_ids,
        "steps": [
            {"new": [0, 1], "gold": hop, "question_index": i,
             "hop_index": hop, "example_id": example}
            for i, example in enumerate(example_ids) for hop in range(2)
        ],
        "metadata": [
            {"question": q, "example_id": e, "answer": "answer",
             "answer_aliases": [], "question_hops": 2, "gold": {0, 1}}
            for q, e in zip(questions, example_ids, strict=True)
        ],
        "all_gold": {0, 1},
    }
    tasks = {task_id: task_row}
    corpus = [
        {"pid": pid, "ntok": 2, "title": str(pid), "paragraph_text": "evidence", "text": "evidence"}
        for pid in range(2)
    ]
    patch = monkeypatch.setattr
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    out = tmp_path / "heldout_reader"
    patch(rollout, "SETUP", {"heldout": (out, 1, 120)})
    patch(rollout, "SMALL", tmp_path)
    patch(rollout, "LOOPS", tmp_path / "rollouts")
    rollout.reuse.write_new(tmp_path / "tau_selection.json", {"status": "complete", "n_tasks": 100, "tau": .5})
    patch(rollout.reuse, "sums", lambda _: {})
    patch(rollout.reuse, "load_task_inputs", lambda _: (corpus, tasks, {"fixture": "signed held-out"}))
    patch(rollout, "load_scores", lambda *_: {task_id: {0: .9, 1: .8}})

    # Keep the real RR replay and replace only the archived reference rollouts.
    baseline = tmp_path / "baseline.json"
    rollout.reuse.write_new(baseline, {})
    reference = SimpleNamespace(metrics=baseline, metrics_sha256=rollout.reuse.checksum(baseline),
                                steps=baseline, steps_sha256=rollout.reuse.checksum(baseline))
    patch(rollout.data, "eval_sources", lambda: {
        name: reference for name in ("no_compress", rollout.R2, rollout.V3, "bm25", "oracle")
    })
    replay_reference = rollout.reuse.replay
    finals = {(cell, task_id): [0, 1] for cell in rollout.reuse.CELLS}
    patch(rollout.reuse, "replay", lambda source, *args:
          (finals, {}) if source.metrics == baseline else replay_reference(source, *args))
    tokenizer = SimpleNamespace(apply_chat_template=lambda *_args, **_kw: [10, 11])
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *_args, **_kw: tokenizer)
    ))
    policies = (rollout.R2, rollout.V3)
    history = list(rollout.reuse.make_requests(
        [rollout.Strategy(name) for name in policies],
        {name: finals for name in policies}, tasks, corpus, tokenizer, False,
    ))
    patch(rollout, "historical_reader_parts", lambda: ({task_id: history}, {"fixture": "refs"}))
    rollout.prepare("heldout")

    prepared = rollout.read(out / "input_preflight.json")
    prompts = list(rollout.reuse.rows(out / "requests.jsonl"))
    steps = list(rollout.reuse.rows(tmp_path / "rollouts/heldout/RR-fill/steps.jsonl"))
    assert prepared["n_prompts"] == len(prompts) == 120
    assert prepared["rollouts"]["RR-τ"]["metrics"]["run"]["tau"] == .5
    assert len(steps) == 48
    assert {row["task"] for row in steps + prompts} == {task_id}
    assert "task" not in task_row

