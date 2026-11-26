"""Task 20 ReAct episode logic without GPU imports or network tools."""
from dataclasses import dataclass, field
from itertools import takewhile
import json
import re
import time

from src.build_traj import make_sample, oracle_action, render_paragraph
from src.heldout_eval import apply_model_output
from src.model_inference import compact, inference_prompt, sha256_json
from src.policies import latest_policy, no_compress_policy
from src.reader_heldout_inputs import need
from src.reranker_v1 import reranker_action

K = 3
TOP_K = 20
MAX_SEARCHES = 4 * K
MAX_INVALID = 2
MAX_MODEL_LEN = 40960
PROMPT_LIMIT = MAX_MODEL_LEN - 512
MAX_NEW = 256
TAU = 0.5
RATE = 0.33
FORCE_TOKENS = ("\n", "?")
FORCE_MESSAGE = "Search limit reached. Give your final answers now."
INVALID_OBSERVATION = "Invalid action format."
EMPTY_OBSERVATION = "(no passages retained)"
THOUGHT_PREFILL = "Thought:"
SYSTEM = (
    "Answer all three numbered questions using the passage collection. "
    "In each turn, write a Thought of one or two sentences and exactly one Action. "
    "To retrieve relevant passages, write your query in search brackets. "
    "Read the resulting Observation before deciding your next action. "
    "When you have your final answers, finish with a JSON object of strings for questions 1, 2, and 3. "
    "Use exactly one of these two formats:\n"
    "Thought: <one or two sentences>\nAction: search[<query>]\n"
    "Thought: <one or two sentences>\nAction: finish[{\"1\":\"<answer>\",\"2\":\"<answer>\",\"3\":\"<answer>\"}]"
)
ROWS = ("no_compress", "latest", "oracle", "RR-τ reselect", "RR-τ append",
        "LLMLingua-2", "r4 reselect", "r4 append")
PAIRS = (("r4 reselect", "RR-τ reselect"), ("r4 append", "RR-τ append"),
         ("r4 reselect", "no_compress"), ("r4 append", "no_compress"),
         ("r4 reselect", "LLMLingua-2"), ("r4 reselect", "latest"), ("oracle", "r4 reselect"))
REACT = re.compile(r"Thought:[ \t]*(?P<thought>.+?)\r?\nAction:[ \t]*(?P<kind>search|finish)\[(?P<payload>.*)\]", re.S)


@dataclass(frozen=True)
class Reply:
    kind: str
    query: str | None = None
    answers: tuple[str, ...] = ("",) * K
    malformed_json: bool = False


def unique_object(pairs):
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError("Duplicate JSON keys")
    return value


def parse_reply(raw):
    match = REACT.fullmatch(raw.strip())
    if not match:
        return Reply("invalid")
    thought = match["thought"].strip()
    payload = match["payload"].strip()
    if not thought or re.search(r"(?m)^(?:Thought|Action|Observation)\s*:", thought):
        return Reply("invalid")
    if match["kind"] == "search":
        return Reply("search", query=payload) if payload and "\n" not in payload and "\r" not in payload else Reply("invalid")
    try:
        decoded = json.loads(payload, object_pairs_hook=unique_object)
    except (ValueError, RecursionError):
        return Reply("finish", malformed_json=True)
    if not isinstance(decoded, dict):
        return Reply("finish", malformed_json=True)
    answers = [decoded.get(str(i)) for i in range(1, K + 1)]
    return Reply("finish", answers=tuple(v.strip() if isinstance(v, str) else "" for v in answers))


def prefix_cost(prior, current):
    """Prefix length and Task 02's cached token cost, applied to individual tokens."""
    shared = sum(1 for _ in takewhile(lambda pair: pair[0] == pair[1], zip(prior, current)))
    return shared, (shared + 25 * (len(current) - shared)) / 20


def compressor_prompt(tokenizer, state):
    """Use the signed training renderer and generation prefix."""
    return inference_prompt(tokenizer, state)


def search_bm25(index, query):
    import bm25s
    ids, _ = index.retrieve(bm25s.tokenize([query], show_progress=False), k=TOP_K, show_progress=False)
    return [int(pid) for pid in ids[0]]


@dataclass
class Compression:
    kept: list[int]
    views: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)


@dataclass
class SearchRequest:
    episode: object
    state: dict
    fresh: list[int]
    turn: dict


def apply_compression(request, raw, *, views=None):
    result = apply_model_output(request.state, raw)
    by_lid = request.state["pid_by_lid"]
    kept = [by_lid[i] for i in result["keep_after"]]
    return Compression(kept, views or {}, {
        "raw_output": raw, "action": result["action"], "invalid_reason": result["invalid_reason"],
        "used_tokens": result["used_after"],
    })


def baseline_compression(request):
    policy = {"no_compress": no_compress_policy, "latest": latest_policy(), "oracle": oracle_action}
    return apply_compression(request, compact(policy[request.episode.row](request.state)))


def reranker_compression(request, scores):
    return apply_compression(request, compact(reranker_action(request.state, scores, TAU)))


def compressed_view(passage, text, ntok):
    visible = render_paragraph(passage["title"], text)
    return {**passage, "paragraph_text": text, "text": visible, "ntok": ntok(visible)}


def llmlingua_compression(request, compress, ntok):
    ep = request.episode
    views = dict(ep.views)
    for pid in request.fresh:
        passage = ep.corpus[pid]
        text = compress(passage["paragraph_text"])
        need(isinstance(text, str), "LLMLingua-2 did not return compressed text")
        views[pid] = compressed_view(passage, text, ntok)
    state = make_sample(ep.question_text, ep.kept, request.fresh, (), views, ep.budget, ep.mode, ep.lids)
    applied = SearchRequest(ep, state, request.fresh, request.turn)
    decision = apply_compression(applied, compact(latest_policy()(state)), views=views)
    # Log each derived passage, including dropped and unselected results.
    decision.details["new_compressed"] = {str(pid): {"body": views[pid]["paragraph_text"],
                                                    "ntok": views[pid]["ntok"]}
                                           for pid in request.fresh}
    return decision


class Episode:
    """Own one task and one policy's ReAct history, evidence and action journal."""

    def __init__(self, task, row, corpus, tokenizer, *, max_model_len=MAX_MODEL_LEN):
        need(row in ROWS and len(task["questions"]) == K, "Unexpected task or strategy")
        self.task, self.row, self.corpus, self.tokenizer = task, row, corpus, tokenizer
        self.max_model_len = max_model_len
        self.mode = "append" if row.endswith(" append") else "reselect"
        self.budget = task["budget"]
        self.question_text = [q["question"] for q in task["questions"]]
        self.gold = {pid for q in task["questions"] for pid in q["gold"]}
        self.kept, self.views, self.lids, self.origin, self.searched = [], {}, {}, {}, set()
        self.blocks, self.turns, self.queries = [], [], []
        self.answers = None
        self.force_reason, self.invalid_streak = None, 0
        self.previous_tokens = []
        self.thought_tokens = self.encode(THOUGHT_PREFILL)
        self.prompt = self.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_text(self.question_text)}])
        self.force_tail = self.encode("<|im_end|>\n") + self.chat([{"role": "user", "content": FORCE_MESSAGE}])

    def encode(self, text):
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def chat(self, messages):
        return list(self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False))

    def passage(self, pid):
        return self.views.get(pid, self.corpus[pid])

    def observation(self, i, kept, origin, views, special):
        if special is not None:
            return special
        lines = []
        for pid in kept:
            if origin.get(pid) == i:
                p = views.get(pid, self.corpus[pid])
                lines.append(f"[P{self.lids[pid] + 1}] {p['title']}: {p['paragraph_text']}")
        return "\n".join(lines) if lines else EMPTY_OBSERVATION

    def prompt_tokens(self, *, kept=None, origin=None, views=None, force=None, history_size=None):
        kept = self.kept if kept is None else kept
        origin = self.origin if origin is None else origin
        views = self.views if views is None else views
        force = self.force_reason is not None if force is None else force
        result = list(self.prompt)
        for i, (raw, special) in enumerate(self.blocks[:history_size]):
            observation = self.observation(i, kept, origin, views, special)
            # Each round is a separate, stable token chunk so an appended round
            # cannot retokenize a previous newline at the boundary.
            result.extend(self.thought_tokens)
            completion = raw[len(THOUGHT_PREFILL):].rstrip()
            result.extend(self.encode(f"{completion}\nObservation:\n{observation}\n\n"))
        if force:
            result.extend(self.force_tail)
        result.extend(self.thought_tokens)
        return result

    def next_prompt(self):
        need(self.answers is None, "Already completed episode")
        tokens = self.prompt_tokens()
        if self.force_reason and len(tokens) + MAX_NEW > self.max_model_len:
            # The reserved span includes generated tokens. If role framing also
            # crosses the window, omit the triggering block from the final
            # prompt only and keep the full output in the episode archive.
            tokens = self.prompt_tokens(history_size=len(self.blocks) - 1)
            self.turns[-1]["force_prompt_omits_triggering_block"] = True
        bound = self.max_model_len - (MAX_NEW if self.force_reason else 512)
        need(len(tokens) <= bound, f"Agent prompt exceeds allowed context: {len(tokens)} > {bound}")
        return tokens

    def accept_reply(self, tokens, completion, output_tokens, search):
        need(self.answers is None and 0 <= output_tokens <= MAX_NEW, "Invalid generation state or output length")
        reply_text = THOUGHT_PREFILL + completion
        reply = parse_reply(reply_text)
        hit, charge = prefix_cost(self.previous_tokens, tokens)
        if self.mode == "append" and not self.force_reason:
            need(hit == len(self.previous_tokens), "Append prompt does not retain its token prefix")
        self.previous_tokens = tokens
        turn = {"round": len(self.turns), "raw_output": reply_text, "action": reply.kind, "query": reply.query,
                "prompt_tokens": len(tokens), "prompt_sha256": sha256_json(tokens), "prefix_hit_tokens": hit,
                "cached_input_token_cost": charge, "output_tokens": output_tokens,
                "forced": self.force_reason, "keep_before": self.kept.copy(), "keep_after": self.kept.copy(),
                "malformed_finish_json": reply.malformed_json, "format_invalid": reply.kind == "invalid"}
        self.turns.append(turn)
        if self.force_reason:
            turn["format_invalid"] = reply.kind != "finish" or reply.malformed_json
            self.answers = reply.answers
            return None
        if reply.kind == "finish":
            turn["format_invalid"] = reply.malformed_json
            self.answers = reply.answers
            return None
        if reply.kind == "invalid":
            self.invalid_streak += 1
            self.blocks.append((reply_text, INVALID_OBSERVATION))
            turn["observation"] = INVALID_OBSERVATION
            if self.invalid_streak == MAX_INVALID:
                self.force_reason = "invalid_format"
            elif len(self.prompt_tokens(force=False)) > self.max_model_len - 512:
                self.force_reason = "overlength"
            return None

        self.invalid_streak = 0
        self.queries.append(reply.query)
        returned = search(reply.query)
        need(len(returned) == len(set(returned)) and all(type(pid) is int and 0 <= pid < len(self.corpus) for pid in returned),
             "BM25 returned duplicate or unknown passages")
        for pid in returned:
            self.lids.setdefault(pid, len(self.lids))
        self.searched.update(returned)
        turn["retrieved"] = returned
        kept = set(self.kept)
        fresh = [pid for pid in returned if pid not in kept]
        self.blocks.append((reply_text, None))
        if not fresh:
            self.apply_decision(turn, Compression(self.kept.copy(), dict(self.views)))
            return None
        budget = self.budget
        if self.row == "no_compress":
            budget = max(budget, sum(self.corpus[pid]["ntok"] for pid in self.kept + fresh))
        # Only oracle can receive gold. The compressor prompt is label-blind too.
        state = make_sample(self.question_text, self.kept, fresh, self.gold if self.row == "oracle" else (),
                            self.corpus, budget, self.mode, self.lids)
        return SearchRequest(self, state, fresh, turn)

    def apply_decision(self, turn, decision):
        need(turn is self.turns[-1] and self.answers is None, "Stale compression request")
        need(len(decision.kept) == len(set(decision.kept)) and set(decision.kept) <= set(self.kept + turn["retrieved"]),
             "Compression selected an unavailable or duplicate passage")
        if self.mode == "append":
            need(decision.kept[:len(self.kept)] == self.kept, "Append policy altered its retained prefix")
        views = {pid: passage for pid, passage in decision.views.items() if pid in decision.kept}
        need(self.row == "no_compress" or sum(views.get(pid, self.corpus[pid])["ntok"] for pid in decision.kept) <= self.budget,
             "Compression exceeded its passage budget")
        origin = dict(self.origin)
        for pid in decision.kept:
            origin.setdefault(pid, len(self.blocks) - 1)
        candidate = self.prompt_tokens(kept=decision.kept, origin=origin, views=views, force=False)
        if len(candidate) > self.max_model_len - 512:
            self.force_reason = "overlength"
            turn["rejected_next_prompt_tokens"] = len(candidate)
        else:
            self.kept, self.views, self.origin = decision.kept, views, origin
            if len(self.queries) == MAX_SEARCHES:
                self.force_reason = "search_limit"
        turn["observation"] = self.observation(len(self.blocks)-1, self.kept, self.origin, self.views, None)
        turn["keep_after"] = self.kept.copy()
        turn["compression"] = decision.details or None

    def completed(self):
        return self.answers is not None

    def record(self):
        need(self.completed(), "Cannot archive an incomplete episode")
        return {
            "dataset": self.task["dataset"], "task": self.task["task"], "row": self.row, "budget": self.budget,
            "questions": self.task["questions"], "answers": self.answers, "turns": self.turns, "queries": self.queries,
            "force_reason": self.force_reason, "searched_pids": sorted(self.searched), "final_keep": self.kept,
            "episode_local_ids": {str(pid): lid+1 for pid, lid in self.lids.items()},
            "final_observations": [self.observation(i, self.kept, self.origin, self.views, special)
                                   for i, (_, special) in enumerate(self.blocks)],
            "compressed_final": self.views,
        }


def user_text(questions):
    return "\n".join(f"{i}. {q}" for i, q in enumerate(questions, start=1))


def execute_compression(request, compress):
    began = time.monotonic()
    decision = compress(request)
    decision.details["wall_seconds"] = time.monotonic() - began
    request.episode.apply_decision(request.turn, decision)
