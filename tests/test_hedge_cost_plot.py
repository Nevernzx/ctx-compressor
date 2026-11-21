"""Guard the cost axis choice and the pooled trajectory denominator without a plotting runtime."""
import pytest

from scripts.plot_hedge_cost import pooled_point


def example():
    cells = {}
    for mode in ("append", "reselect"):
        for tier in ("tight", "mid", "loose"):
            v = {"n": 249, "mean": 5.0}
            cells[f"oracle|{mode}|{tier}"] = {"metrics": {"cost": {"cached": v},
                                                          "evidence_loss": {"n": 249, "count": 1}}}
    return {"strategy": "oracle", "configs": cells,
            "settings": {"split": "train-heldout", "n_tasks": 249, "n_trajectories": 1494},
            "combined": {"n_tasks": 1494, "evidence_loss": {"n": 1494, "count": 6, "rate": 6/1494},
                         "cost": {"plain": {"mean": 42}, "cached": {"mean": 5.0, "n": 1494}}}}


def test_cost_axis_uses_trajectory_mean_cached_cost():
    point = pooled_point(example(), "oracle")
    assert point == {"cost": 5.0, "loss_rate": 6/1494, "loss_count": 6, "n": 1494}


def test_inconsistent_pooled_cost_is_rejected():
    metrics = example()
    metrics["combined"]["cost"]["cached"]["mean"] = 42.0
    with pytest.raises(ValueError, match="six cells"):
        pooled_point(metrics, "oracle")
