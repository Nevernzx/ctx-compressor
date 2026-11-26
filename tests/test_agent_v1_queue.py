"""The detached Task 20 queue admits the smoke only after Task 19 G3 and lock release."""
import pytest

from scripts import agent_v1_queue as queue


@pytest.mark.parametrize("stage,status,released", [
    ("G2-v3", "running", False),
    ("G3", "running", False),
    ("terminal", "failed", True),
    ("all", "evaluation_complete", False),
])
def test_smoke_stays_queued_for_incomplete_task19(stage,status,released):
    assert not queue.ready({"task":19,"stage":stage,"status":status,"lock_released":released})


def test_smoke_admission_recognizes_task19_terminal_state():
    proof = {"task":19,"stage":"all","status":"evaluation_complete","lock_released":True}
    assert queue.ready(proof)
    with pytest.raises(ValueError,match="unexpected Task 19"):
        queue.ready({**proof,"task":20})
