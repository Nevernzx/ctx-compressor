"""Independent selected-logit and float64 normalization for native vLLM greedy requests."""
import json
from pathlib import Path
import re

import torch
from vllm.v1.sample.logits_processor.builtin import process_dict_updates
from scripts.hedge_forced_tokens import ForcedCandidateTokens

CAPTURE_DIR = Path("/root/userfolder/ctx-compressor/hedge_v2/score_logits")


class CaptureCandidateLogits(ForcedCandidateTokens):
    """Observe unconstrained rows before sampling, then apply ordinary candidate forcing."""

    def __init__(self, vllm_config, device, is_pin_memory):
        super().__init__(vllm_config, device, is_pin_memory)
        self.natural = {}
        self.streams = {}

    @staticmethod
    def capture_request(params, prompt_ids, output_ids):
        extra = params.extra_args or {}
        session = extra.get("hedge_logit_session")
        if session is None:
            return None
        serial = extra["hedge_logit_request"]
        if (not re.fullmatch("[a-f0-9]{32}",session) or type(serial) is not int or serial<0
                or params.temperature != 0 or params.presence_penalty or params.frequency_penalty
                or params.repetition_penalty!=1 or params.min_tokens or params.logit_bias
                or params.allowed_token_ids or params.bad_words or extra.get("hedge_teacher_tokens")):
            raise ValueError("Logit capture must use natural, unconstrained greedy decoding")
        return session, serial, output_ids

    def update_state(self, batch_update):
        super().update_state(batch_update)
        process_dict_updates(self.natural, batch_update, self.capture_request)

    def apply(self, logits):
        if self.natural:
            indices = list(self.natural)
            rows = logits[indices].double()
            value, token = torch.max(rows, dim=-1)
            # Independent from the vLLM sampler's float32 log_softmax. No
            # vLLM sampled logprob is ever read by this worker-side observer.
            normalizer = torch.logsumexp(rows, dim=-1)
            captures = torch.stack((token.double(),value,normalizer),dim=1).tolist()
            writes = {}
            for index, (target, logit, log_z) in zip(indices,captures):
                session, serial, output_ids = self.natural[index]
                writes.setdefault(session,[]).append([serial,len(output_ids),int(target),logit,log_z])
            for session,records in writes.items():
                if session not in self.streams:
                    CAPTURE_DIR.mkdir(parents=True,exist_ok=True)
                    self.streams[session]=(CAPTURE_DIR/(session+".jsonl")).open("x",encoding="utf-8")
                stream=self.streams[session]
                stream.writelines(json.dumps(record,separators=(",",":"),allow_nan=False)+"\n" for record in records)
                stream.flush()
        return super().apply(logits)

