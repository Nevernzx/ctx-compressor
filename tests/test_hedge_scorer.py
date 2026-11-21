from types import SimpleNamespace
import pytest

from scripts.hedge_scorer import continuation, raw_logprob, gate_gpu


class CodepointTokenizer:
    def encode(self, value, add_special_tokens=False):
        assert not add_special_tokens
        return list(map(ord,value))

    def decode(self, ids, skip_special_tokens=False):
        return "".join(map(chr,ids))


@pytest.mark.parametrize("candidate", [0,10,209])
def test_teacher_forcing_full_literal(candidate):
    tok = CodepointTokenizer()
    ids = continuation(tok, candidate)
    assert tok.decode(ids) == '{"add":[' + str(candidate) + '],"'


@pytest.mark.parametrize("candidate", [-1, True, "2", 1.5])
def test_invalid_candidate_id(candidate):
    with pytest.raises(ValueError):
        continuation(CodepointTokenizer(), candidate)


def test_logprob_uses_selected_token_even_if_not_argmax():
    out = SimpleNamespace(token_ids=[31,45,99],logprobs=[
        {31:SimpleNamespace(logprob=-7.0),4:SimpleNamespace(logprob=-0.1)},
        {45:SimpleNamespace(logprob=-0.4)}, {99:SimpleNamespace(logprob=-1.0)}])
    assert raw_logprob(out, [31,45]) == -7.4
    with pytest.raises(ValueError, match="absent"):
        raw_logprob(out, [4,45])


@pytest.mark.parametrize("val",[float("nan"), float("inf"), 0.2])
def test_rejects_invalid_logprob(val):
    out = SimpleNamespace(token_ids=[3], logprobs=[{3:SimpleNamespace(logprob=val)}])
    with pytest.raises(ValueError, match="raw logprob"):
        raw_logprob(out,[3])


def test_gpu_guard_checks_selection_before_invoking_smi(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    with pytest.raises(ValueError, match="physical GPU 1"):
        gate_gpu()



def test_score_candidates_sums_every_forced_token_and_keeps_state_alignment(monkeypatch):
    import sys
    from types import ModuleType
    from scripts.hedge_scorer import score_candidates

    vllm = ModuleType("vllm")
    vllm.SamplingParams = SimpleNamespace
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    class Engine:
        def get_tokenizer(self):
            return CodepointTokenizer()

        def generate(self, prompts, sampling_params, lora_request, use_tqdm):
            assert lora_request == "v2" and not use_tqdm
            params = sampling_params if isinstance(sampling_params, list) else [sampling_params] * len(prompts)
            out = []
            for p, sp in zip(prompts,params):
                ids = getattr(sp, "extra_args", {}).get("hedge_teacher_tokens",[3])
                tokens = SimpleNamespace(token_ids=ids,logprobs=[
                    {i:SimpleNamespace(logprob=-(i+1)/100)} for i in ids])
                out.append(SimpleNamespace(request_id=str(len(out)),prompt_token_ids=p["prompt_token_ids"],outputs=[tokens]))
            return out
    def capture(llm,adapter,prompts):
        # Simulate a worker returning raw logit and float64 normalization,
        # distinct from the native request-output log probabilities.
        return [(SimpleNamespace(request_id=str(i)),[(3,-0.15,2.0)]) for i,_ in enumerate(prompts)]
    monkeypatch.setattr("scripts.hedge_scorer.captured_natural_paths",capture)
    candidates = [[12,4],[],[111]]
    audit=[]
    out = score_candidates(Engine(),"v2",[[1],[2,3],[4,5]],candidates,audit=audit)
    assert len(audit)==3
    assert [list(branches) for _,_,branches in audit]==candidates
    assert all(tokens==[(3,-0.15,2.0)] for _,tokens,_ in audit)
    assert [list(row) for row in out] == candidates
    for row in out:
        for c, value in row.items():
            expected = sum(-(i+1)/100 for i in continuation(CodepointTokenizer(),c))
            assert value == pytest.approx(expected)



@pytest.mark.parametrize("ids,want", [( [5,1,9],-1.75 ), ([5,7,8],-0.75), ([1,2,3],-2.25)])
def test_common_prefix_uses_each_token_once(ids, want):
    from scripts.hedge_scorer import shared_score
    natural = ([5,7,8,24],[-0.25,-0.25,-0.25,-2.5])
    forced = SimpleNamespace(token_ids=ids, logprobs=[{token:SimpleNamespace(logprob=-0.75)} for token in ids])
    assert shared_score(ids, natural, forced) == want



def test_worker_token_reconciliation_and_offsets(tmp_path):
    from scripts.hedge_scorer import read_logit_observations
    path = tmp_path/"logits.jsonl"
    prefix = b"[9,0,12,-1,0]\n"
    path.write_bytes(prefix+b"[10,0,99,0,2]\n[10,0,10,1.0,3.0]\n[10,1,11,2.5,3]\n")
    records,offset=read_logit_observations(path,len(prefix),[10],[SimpleNamespace(token_ids=[10,11])])
    assert records==[[(10,1.0,3.0),(11,2.5,3.0)]]
    assert offset==path.stat().st_size
    with pytest.raises(ValueError,match="argmax"):
        read_logit_observations(path,len(prefix),[10],[SimpleNamespace(token_ids=[10,12])])
