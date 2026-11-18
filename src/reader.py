"""Exact TASK_03 reader messages and deterministic answer parsing.

This module uses only Python's standard library. Chat templating, token counts
and generation belong to the runner; no system message is added here.
"""
import re


_CONTEXT_TEMPLATE = (
    "Answer the question based on the passages below. Reply with only the final "
    "answer (a short phrase such as a name, date, place or number), with no explanation."
    "\n\nPassages:\n{passages}\n\nQuestion: {question}"
)
_NO_CONTEXT_TEMPLATE = (
    "Answer the question. Reply with only the final answer (a short phrase such "
    "as a name, date, place or number), with no explanation."
    "\n\nQuestion: {question}"
)
_COMPLETE_THINK = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
_ANSWER_PREFIX = re.compile(r"^Answer:\s*", flags=re.IGNORECASE)
_QUOTES = "\"'“”‘’"


def render_passages(corpus, pids):
    """Join original title/newline/body fragments, in the supplied PID order."""
    return "\n\n".join(
        corpus[pid]["title"] + "\n" + corpus[pid]["paragraph_text"]
        for pid in pids
    )


def render_user_message(passages, question, no_context=False):
    """Return the task's exact user text; do not strip text or append a newline."""
    if no_context:
        return _NO_CONTEXT_TEMPLATE.format(question=question)
    return _CONTEXT_TEMPLATE.format(passages=passages, question=question)


def parse_answer(raw_output):
    """Apply TASK_03's five parsing steps in the specified order.

    Only complete lowercase <think>...</think> blocks are removed. An unmatched
    block remains visible in the prediction for format auditing. Answer: must
    occur at the beginning after stripping whitespace and is case-insensitive.
    Boundary straight/curly quotes and final full stops are removed; internal
    punctuation, case and subsequent reasoning are not otherwise rewritten.
    """
    text = _COMPLETE_THINK.sub("", raw_output)
    text = text.strip()
    text = _ANSWER_PREFIX.sub("", text, count=1)
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    # A stop can be inside or outside closing quotation marks. Both are the
    # same final parsing step; do not remove leading dots or internal quotes.
    first_line = first_line.strip(_QUOTES).strip()
    first_line = first_line.rstrip(".。").rstrip()
    return first_line.strip(_QUOTES).strip()
