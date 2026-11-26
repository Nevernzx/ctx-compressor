"""Regression checks for the CPU preflight and the inherited Task 03 reader protocol."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from src.qa_metrics import score_answer
from src.reader import render_passages, render_user_message
from src.reader_heldout_inputs import (
    CELLS, Source, check_length, dedup_contexts, export_requests,
    make_requests, request_tokens, verify,
)


class Tokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize and add_generation_prompt and not enable_thinking
        assert len(messages) == 1 and messages[0]["role"] == "user"
        return list(messages[0]["content"].encode("utf-8"))


class HeldoutReaderInputsTest(unittest.TestCase):
    def test_render_retains_original_order_and_task03_chat_protocol(self):
        corpus = [{"title":"First","paragraph_text":"Original body."},
                  {"title":"Second","paragraph_text":"Body\nwith linebreak."}]
        passages = render_passages(corpus,[1,0])
        self.assertEqual(passages,"Second\nBody\nwith linebreak.\n\nFirst\nOriginal body.")
        ids, sig = request_tokens(Tokenizer(),passages,"Who?")
        expected = render_user_message(passages,"Who?").encode()
        self.assertEqual(bytes(ids),expected)
        self.assertEqual(sig,hashlib.sha1(expected).hexdigest())
        empty, _ = request_tokens(Tokenizer(), "", "Who?", True)
        self.assertEqual(bytes(empty).decode(), render_user_message("", "Who?", no_context=True))

    def test_alias_and_reference_protocol(self):
        self.assertEqual(score_answer("Nightwing","Dick Grayson",["Nightwing"]), {"em":1.0,"f1":1.0})

    def test_dedup_checks_task_and_pid_order_across_all_cells(self):
        finals = {(cell, t):[1,t+2] for cell in CELLS for t in (0,1)}
        self.assertTrue(dedup_contexts(finals, (0,1))["all_identical"])
        finals[CELLS[-1], 0] = [2,1]
        proof = dedup_contexts(finals,(0,1))
        self.assertFalse(proof["all_identical"])
        self.assertEqual((proof["differing_tasks"],proof["differing_cells"]), (1,1))

    def test_generation_order_and_individual_evidence_vs_trajectory(self):
        corpus = [{"title":str(i),"paragraph_text":"body"} for i in range(2)]
        src = Source("s", Path("metrics"),"",Path("steps"),"","test fixture")
        metadata = [{"question":str(i), "example_id":str(i), "answer":"canonical",
                     "answer_aliases":["alias"], "question_hops":2, "gold":{i%2}} for i in range(4)]
        tasks = {0:{"metadata":metadata,"all_gold":{0,1}}}
        finals = {"s":{(cell,0):[1] for cell in CELLS}}
        out = list(make_requests([src],finals,tasks,corpus,Tokenizer(),False))
        self.assertEqual(len(out),24)
        self.assertEqual([r["config"] for r in out[:4]],["s|reselect|tight"]*4)
        self.assertEqual([r["question_index"] for r in out[:4]], list(range(4)))
        self.assertEqual([r["evidence_complete"] for r in out[:4]], [False,True,False,True])
        self.assertTrue(all(r["trajectory_lost"] for r in out))
        self.assertEqual(out[0]["answer_aliases"],["alias"])

    def test_never_truncate_or_overwrite_and_check_signatures(self):
        check_length(40960-48, "okay")
        with self.assertRaisesRegex(ValueError, "overflow"):
            check_length(40960-47, "long")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"prompts.jsonl"
            source = {"config":"c","prompt_tokens":30}
            proof = export_requests(path,[source])
            verify(path,proof["sha256"])
            self.assertEqual(json.loads(path.read_text()),source)
            with self.assertRaises(FileExistsError):
                export_requests(path,[source])
            with self.assertRaisesRegex(ValueError,"mismatch"):
                verify(path,"c"*64)
