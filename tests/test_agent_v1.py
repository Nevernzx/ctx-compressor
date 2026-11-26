"""CPU contracts for Task 20 sampling, transcripts, compression and metrics."""
import json
from pathlib import Path
import random
from types import SimpleNamespace

import pytest

from eval.cost import trajectory_cost
from scripts import agent_v1_data as data, agent_v1_engines as engines, agent_v1_smoke as smoke
from src import agent_v1 as agent, agent_v1_metrics as metrics
from src.build_traj import render_input
from src.reranker_v1 import require_new


class SimpleTokenizer:
    def __init__(self):
        self.calls = []

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return list(map(ord, text))

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        assert kwargs == {"tokenize": True, "add_generation_prompt": True, "enable_thinking": False}
        return self.encode("".join(f"<{x['role']}>{x['content']}</>" for x in messages) + "<assistant>")


def fixture(row, budget=8, *, max_model_len=agent.MAX_MODEL_LEN, body=None):
    task = {"dataset": "musique", "task": 1, "budget": budget, "questions": [
        {"id": str(i), "question": f"question {i}", "answer": str(i),
         "answer_aliases": [], "gold": [i - 1], "gold_hop": [i - 1]}
        for i in range(1, 4)
    ]}
    corpus = [{"pid": i, "title": f"title{i}", "paragraph_text": body or f"body{i}",
               "text": f"title{i}\n{body or f'body{i}'}", "ntok": 4}
              for i in range(3)]
    return agent.Episode(task, row, corpus, SimpleTokenizer(), max_model_len=max_model_len)


def reply(ep, text, results=()):
    return ep.accept_reply(ep.next_prompt(), text, 17, lambda q: list(results))


def search(ep, q, results):
    return reply(ep, f" find evidence.\nAction: search[{q}]", results)


def finalize(ep, result='{"1":"1","2":"2","3":"3"}'):
    return reply(ep, f" answer now.\nAction: finish[{result}]")


def test_parse():
    assert agent.parse_reply("Thought: Search now.\nAction: search[year [2006]]").query == "year [2006]"
    assert agent.parse_reply('Thought: answer.\nAction: finish[{"1":"yes","3":"Paris"}]').answers == ("yes", "", "Paris")
    assert agent.parse_reply('Thought: answer.\nAction: finish[{"1":null,"2":21,"3":"X"}]').answers == ("", "", "X")
    for invalid in ("Action: search[raw]", "Thought: x\nAction: search[]",
                    "Thought: x\nAction: search[q]\nAction: search[a]", "Thought: x\nAction: tool[q]"):
        assert agent.parse_reply(invalid).kind == "invalid"
    p = agent.parse_reply('Thought: finish.\nAction: finish[{"1":"broken",}]')
    assert p.kind == "finish" and p.answers == ("", "", "") and p.malformed_json


def test_observation_render_delete_unrepeat_and_empty():
    ep = fixture("latest", budget=4)
    a = search(ep, "a", [0])
    ep.apply_decision(a.turn, agent.baseline_compression(a))
    assert ep.observation(0, ep.kept, ep.origin, ep.views, None) == "[P1] title0: body0"
    a = search(ep, "b", [0, 1])
    ep.apply_decision(a.turn, agent.baseline_compression(a))
    assert ep.observation(0, ep.kept, ep.origin, ep.views, None) == agent.EMPTY_OBSERVATION
    assert ep.observation(1, ep.kept, ep.origin, ep.views, None) == "[P2] title1: body1"
    assert search(ep, "b", [1]) is None
    assert ep.turns[-1]["compression"] is None
    assert ep.turns[-1]["observation"] == agent.EMPTY_OBSERVATION


@pytest.mark.parametrize("row", agent.ROWS)
def test_budget_and_append(row):
    ep = fixture(row, budget=8)
    start = ep.next_prompt()
    prefix = ep.encode("Thought:")
    assert start[-len(prefix):] == prefix
    a = search(ep, "one", [0, 1])
    assert ep.turns[-1]["raw_output"] == "Thought: find evidence.\nAction: search[one]"
    assert not a.state["gold"] or row == "oracle"
    first = decision(ep, a)
    ep.apply_decision(a.turn, first)
    after_first = ep.next_prompt()
    assert after_first[-len(prefix):] == prefix
    a = search(ep, "two", [1, 2])
    ep.apply_decision(a.turn, decision(ep, a))
    if row == "no_compress":
        assert ep.kept == [0, 1, 2]
    else:
        assert sum(ep.passage(p)["ntok"] for p in ep.kept) <= ep.budget
    if ep.mode == "append":
        assert ep.kept[:len(first.kept)] == first.kept
        assert after_first[:len(start)] == start
        assert ep.next_prompt()[:len(after_first)] == after_first


def decision(ep, a):
    if ep.row == "LLMLingua-2":
        return agent.llmlingua_compression(a, lambda _: "short",
                                            lambda visible: 4 if visible.endswith("\nshort") else 100)
    if ep.row.startswith("RR-τ"):
        return agent.reranker_compression(a, {0: .4, 1: .9, 2: .7})
    if ep.row.startswith("r4"):
        if a.state["kept_ids"]:
            invalid = '{"add":[2],"drop":[]}'
            actual = agent.apply_compression(a, invalid)
            assert actual.details["invalid_reason"] == "over_budget" and actual.kept == ep.kept
            return actual
        return agent.apply_compression(a, '{"add":[0,1],"drop":[]}')
    return agent.baseline_compression(a)


def test_overlength_guard_preserves_old_context_and_retrieval():
    ep = fixture("no_compress", max_model_len=1700, body="x" * 2500)
    a = search(ep, "big", [0])
    ep.apply_decision(a.turn, agent.baseline_compression(a))
    assert ep.force_reason == "overlength"
    assert ep.kept == [] and ep.searched == {0}
    assert ep.turns[-1]["rejected_next_prompt_tokens"] > 1700 - 512
    assert agent.FORCE_MESSAGE in "".join(map(chr, ep.next_prompt()))
    finalize(ep)
    assert ep.completed()


def test_forced_search_limit_and_consecutive_invalid():
    ep = fixture("no_compress")
    a = search(ep, "same", [0])
    ep.apply_decision(a.turn, agent.baseline_compression(a))
    for _ in range(agent.MAX_SEARCHES - 1):
        assert search(ep, "same", [0]) is None
    assert ep.force_reason == "search_limit" and len(ep.queries) == 12
    assert agent.FORCE_MESSAGE in "".join(map(chr, ep.next_prompt()))
    finalize(ep)
    assert ep.answers == ("1", "2", "3")
    ep = fixture("latest")
    reply(ep, "garbage")
    assert "".join(map(chr, ep.next_prompt())).endswith(agent.INVALID_OBSERVATION + "\n\n" + agent.THOUGHT_PREFILL)
    reply(ep, "again")
    assert ep.force_reason == "invalid_format" and ep.queries == []
    reply(ep, " not following instruction.\nAction: search[q]")
    assert ep.answers == ("", "", "") and not ep.queries


def test_cost_matches_registered_rates():
    prior, current = [1, 2, 3], [1, 2, 4, 5]
    assert agent.prefix_cost(prior, current) == (2, 2.6)
    assert agent.prefix_cost([], prior) == (0, 3.75)
    assert (agent.prefix_cost([], prior)[1] + agent.prefix_cost(prior, current)[1]) == (
        trajectory_cost([prior, current], dict.fromkeys([1, 2, 3, 4, 5], 1))[1])


def test_task17_signed_five_state_adapter_parity():
    path = Path(__file__).parent / "fixtures/agent_v1_rr5.json"
    contract = json.loads(path.read_text())
    cases = contract["cases"]
    assert len(cases) == 5 and {x["mode"] for x in cases} == {"append", "reselect"}
    for row in cases:
        state = row["state"]
        for key in ("ntok", "pid_by_lid"):
            state[key] = {int(k): v for k, v in state[key].items()}
        state["gold"] = []
        scores = {int(k): v for k, v in row["scores"].items()}
        outcome = agent.reranker_compression(agent.SearchRequest(None, state, [], {}), scores)
        assert outcome.details["action"] == row["expected_action"]
        assert outcome.kept == row["expected_keep"]


def test_r4_uses_exact_training_renderer_and_chat_template():
    ep = fixture("r4 reselect")
    req = search(ep, "first", [0, 2])
    tok = SimpleTokenizer()
    out = agent.compressor_prompt(tok, req.state)
    signed_renderer_output = render_input(req.state)
    assert tok.calls[-1][0][0]["content"].encode("utf-8") == signed_renderer_output.encode("utf-8")
    assert out["prompt_token_ids"] == tok.apply_chat_template(
        [{"role": "user", "content": signed_renderer_output}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False)
    assert len(req.state["questions"]) == 3 and req.state["gold"] == []


def test_seeded_disjoint_sampling_and_budget():
    base = [{"id": f"{i:04d}", "gold": [0, 0]} for i in range(400)]
    locked_ids = {x["id"] for x in base}
    excluded = [{"id": f"excluded{i:03d}", "gold": [0]} for i in range(40)]
    for ds, seed in data.SEEDS.items():
        main, smoke_rows = data.select(list(reversed(base)), excluded if ds == "musique" else None, ds)
        flatten = lambda seq: [q["id"] for row in seq for q in row]
        assert flatten(main) == random.Random(seed).sample(sorted(locked_ids), 300)
        assert len(flatten(smoke_rows)) == 30
        assert set(flatten(main)).isdisjoint(flatten(smoke_rows))
        expected_pool = sorted(q["id"] for q in excluded) if ds == "musique" else sorted(locked_ids - set(flatten(main)))
        assert flatten(smoke_rows) == random.Random(seed + 100).sample(expected_pool, 30)
        if ds == "musique":
            assert locked_ids.isdisjoint(flatten(smoke_rows))
        tune, stats = data.budget_for(main, [{"ntok": 100}])
        assert tune == 150 and stats["n"] == 100 and stats["p90"] == 100


def test_no_overwrite(tmp_path):
    dest = tmp_path / "episode.jsonl"
    first = smoke.SmokeArchive(dest)
    first.file.close()
    with pytest.raises(FileExistsError):
        smoke.SmokeArchive(dest)
    with pytest.raises(FileExistsError):
        require_new(dest)
    assert dest.is_file()


def test_effective_cascade_and_task_order(tmp_path):
    for actual in (None, False, True):
        e = SimpleNamespace(vllm_config=SimpleNamespace(model_config=SimpleNamespace(disable_cascade_attn=actual)))
        if actual is True:
            assert engines.enforce_cascade(e, tmp_path/"log")["effective_disable_cascade_attn"]
        else:
            with pytest.raises(ValueError):
                engines.enforce_cascade(e, tmp_path/"log")
    status = tmp_path/"status.json"
    status.write_text('{"status":"running"}')
    with pytest.raises(ValueError):
        smoke.require_task19(status)
    status.write_text('{"status":"evaluation_complete"}')
    assert smoke.require_task19(status)["size"] == status.stat().st_size


def test_metrics_missing_answer_and_evidence():
    ep = fixture("latest", budget=4)
    a = search(ep, "lookup", [0, 1, 2])
    ep.apply_decision(a.turn, agent.baseline_compression(a))
    finalize(ep, '{"1":"1","2":"wrong"}')
    output, records = metrics.analyze_row([ep.record()])
    assert output["em"] == output["f1"] == 1 / 3
    assert output["evidence"]["retrieved"] == {"count": 3, "n": 3, "rate": 1.0}
    assert output["evidence"]["lost_given_retrieved"]["count"] == 2
    assert output["evidence"]["retained"]["count"] == 1
    paired = metrics.paired(records, records)
    assert paired["n"] == 3 and paired["f1_equal"] == 3
    assert paired["em_2x2_left_rows_right_columns"] == [[2, 0], [0, 1]]


def test_mock_search_handoff(monkeypatch, tmp_path):
    ep = fixture("r4 reselect")
    monkeypatch.setattr(agent, "search_bm25", lambda index, q: [0])
    def complete(_llm, _deployment, batch):
        for episode in batch:
            raw = " lookup.\nAction: search[q]" if not episode.turns else ' answer.\nAction: finish[{"1":"1"}]'
            yield episode.next_prompt(), raw, 15, 0
    monkeypatch.setattr(engines, "generate_agent", complete)
    deployment = {"agent": {"params": {"max_num_seqs": 1}}}
    indexes = {"musique": object()}
    sink = smoke.SmokeArchive(tmp_path/"trace.jsonl")
    try:
        pending = smoke.r4_search_boundary(None, deployment, [ep], indexes, sink)
        assert len(pending) == 1
        action = pending[0]
        ep.apply_decision(action.turn, agent.apply_compression(action, '{"add":[0],"drop":[]}'))
        assert smoke.r4_search_boundary(None, deployment, [ep], indexes, sink) == []
    finally:
        sink.file.close()
    assert json.loads((tmp_path/"trace.jsonl").read_text())["final_keep"] == [0]


def test_prefill_preserves_append_tokens_when_encoding_would_merge():
    class BoundaryTokenizer(SimpleTokenizer):
        def encode(self, text, add_special_tokens=False):
            if text.startswith("Thought: "):
                return [70000] + super().encode(text[len("Thought: "):], add_special_tokens)
            return super().encode(text, add_special_tokens)

    episode = fixture("r4 append")
    episode.tokenizer = BoundaryTokenizer()
    first = episode.next_prompt()
    request = search(episode, "evidence", [0])
    episode.apply_decision(request.turn, agent.apply_compression(request, '{"add":[0],"drop":[]}'))
    second = episode.next_prompt()
    rendered = "".join(map(chr, second))
    assert rendered.endswith("[P1] title0: body0\n\nThought:")
    assert 70000 not in second
    assert second[:len(first)] == first
    result = reply(episode, " Answer the questions.\nAction: finish[{}]")
    assert result is None and episode.completed()
    assert episode.turns[-1]["raw_output"].startswith("Thought: Answer")
    assert episode.turns[-1]["prefix_hit_tokens"] == len(first)
    assert episode.turns[-1]["output_tokens"] == 17


def test_prefill_covers_invalid_observation_and_forced_finish():
    episode = fixture("latest")
    reply(episode, "\n[P2] invented passage\nThought: Reasoning\nAction: search[q]")
    assert episode.turns[-1]["format_invalid"]
    assert "".join(map(chr, episode.next_prompt())).endswith("Invalid action format.\n\nThought:")
    reply(episode, " incomplete")
    forced = "".join(map(chr, episode.next_prompt()))
    assert agent.FORCE_MESSAGE in forced and forced.endswith("Thought:")
    finalize(episode)
    assert episode.turns[-1]["forced"] == "invalid_format"
    assert episode.turns[-1]["raw_output"].startswith("Thought: answer now.")
    assert episode.answers == ("1", "2", "3")
