"""vLLM V1 request-local teacher forcing. Imported only by the GPU runtime."""
from vllm.v1.sample.logits_processor.builtin import process_dict_updates
from vllm.v1.sample.logits_processor.interface import LogitsProcessor


class ForcedCandidateTokens(LogitsProcessor):
    """Choose the supplied next token after vLLM has saved the raw log probabilities."""

    def __init__(self, vllm_config, device, is_pin_memory):
        if vllm_config.model_config.logprobs_mode != "raw_logprobs":
            raise ValueError("Teacher forcing requires pre-processor raw log probabilities")
        self.sequences = {}

    def is_argmax_invariant(self):
        return False

    @staticmethod
    def request(params, prompt_ids, output_ids):
        tokens = (params.extra_args or {}).get("hedge_teacher_tokens")
        if tokens is None:
            return None
        if not tokens or len(tokens) != params.max_tokens or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError("Invalid teacher forcing request")
        return tokens, output_ids

    def update_state(self, batch_update):
        process_dict_updates(self.sequences, batch_update, self.request)

    def apply(self, logits):
        if self.sequences:
            indices = []
            targets = []
            for index, (tokens, emitted) in self.sequences.items():
                indices.append(index)
                targets.append(tokens[len(emitted)])
            # vLLM Sampler.forward gathers from its separate unmodified log-softmax
            # buffer after it determines the sampled ids (logprobs_mode=raw_logprobs).
            logits[indices, targets] = float("inf")
        return logits

