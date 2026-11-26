"""Task 10 log capture routes Task 09's unchanged worker into the new run directory."""
from pathlib import Path
from scripts import hedge_raw_logits

CAPTURE_DIR = Path("/root/userfolder/ctx-compressor/dagger_v3/score_logits")


class CaptureCandidateLogits(hedge_raw_logits.CaptureCandidateLogits):
    def __init__(self, vllm_config, device, is_pin_memory):
        hedge_raw_logits.CAPTURE_DIR = CAPTURE_DIR
        super().__init__(vllm_config, device, is_pin_memory)

