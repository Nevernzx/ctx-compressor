"""Read-only GPU 1 admission for each Task 10 resumed GPU stage."""
import time

from scripts import run_formal_grpo_pipeline as controller


POLL_SECONDS = 5 * 60
TIMEOUT_SECONDS = 36 * 60 * 60
REQUIRED_FREE = 3
LIMIT_MIB = 2048


def wait_for_gpu(stage, publish, *, reader=controller.gpu, clock=time.monotonic, sleep=time.sleep):
    """Publish every physical GPU 1 reading, then return the third consecutive free reading."""
    start = clock()
    deadline = start + TIMEOUT_SECONDS
    consecutive = 0
    while clock() < deadline:
        sample = reader()
        observed = clock()
        if sample["index"] != 1 or sample["uuid"] != controller.GPU_UUID:
            raise ValueError(f"{stage}: expected physical GPU 1, got {sample}")
        consecutive = consecutive + 1 if sample["memory_used_mib"] < LIMIT_MIB else 0
        publish({"stage": stage, "status": "waiting_gpu", "utc": controller.now(), "gpu": sample,
                 "consecutive_free": consecutive, "required_free": REQUIRED_FREE, "limit_mib": LIMIT_MIB,
                 "poll_seconds": POLL_SECONDS, "elapsed_seconds": observed - start, "timeout_seconds": TIMEOUT_SECONDS})
        remaining = deadline - clock()
        if remaining <= 0:
            break
        if consecutive == REQUIRED_FREE:
            return sample
        sleep(min(POLL_SECONDS, remaining))
    raise TimeoutError(f"{stage}: timed out after 36 h waiting for physical GPU 1 below {LIMIT_MIB} MiB "
                       f"on {REQUIRED_FREE} consecutive readings")
