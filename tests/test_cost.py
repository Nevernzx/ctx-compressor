import pytest

from eval.cost import trajectory_cost


@pytest.mark.parametrize("size,reselect,expected", [
    (3000, False, (165000, 44250)),
    (2000, True, (20000, 25000)),
    (300, False, (16500, 4425)),
])
def test_plan_ten_step_costs(size, reselect, expected):
    contexts = [[i] if reselect else list(range(i + 1)) for i in range(10)]
    assert trajectory_cost(contexts, {i: size for i in range(10)}) == pytest.approx(expected)


def test_only_longest_common_prefix_hits_cache():
    # Fragment 2 reappears after a changed prefix, so it is written again.
    assert trajectory_cost([[0, 1, 2], [0, 3, 2]], {i: 100 for i in range(4)}) == (600, 630)


def test_cache_rates_are_parameters():
    assert trajectory_cost([[0], [0, 1]], {0: 100, 1: 200}, 0.1, 2) == (400, 610)


def test_empty_trajectory():
    assert trajectory_cost([], {}) == (0, 0)
