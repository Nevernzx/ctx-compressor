"""Task 12 reader aggregation, pairing, costs and publication contracts."""
from copy import deepcopy

import pytest

from scripts import reader_heldout_report as report


def toy_rows(strategy, *, wins=0, cell_rows=4):
    sources, preds = [], []
    cells = ("shared",) if strategy in ("no_context", "no_compress") else report.CELLS
    for cell in cells:
        for i in range(cell_rows):
            mode, tier = (None, None) if cell == "shared" else cell.split("|")
            config = strategy if cell == "shared" else f"{strategy}|{cell}"
            answer, aliases = "New York City", ["NYC"]
            common = dict(config=config, task=700, question_index=i, example_id=f"example_{i}",
                          strategy=strategy, mode=mode, tier=tier, question_hops=2 if i<3 else 4,
                          trajectory_lost=strategy != "no_compress", evidence_complete=strategy == "no_compress",
                          prompt_sha1="a"*40, prompt_tokens=8+i, answer=answer, answer_aliases=aliases)
            ok = i < wins
            pred = {**common, "raw_output":"Answer: NYC." if ok else "London",
                    "pred":"NYC" if ok else "London","f1":float(ok),"em":float(ok),
                    "output_tokens":1,"output_token_ids":[4],"finish_reason":"stop","input_truncated":False}
            sources.append(common)
            preds.append(pred)
    return sources, preds


def test_alias_at_scoring_and_independent_pair_alignment():
    data = {}
    for s, wins in [(report.R2, 2), (report.V3, 1)]:
        req, preds = toy_rows(s, wins=wins)
        data.update(report.collect(reversed(preds), report.index_requests(req), [s]))
    pair = report.paired(report.R2, report.V3, data)
    assert pair["n"] == 24
    assert (pair["f1_difference_pp"], pair["em_difference_pp"]) == (25, 25)
    assert pair["f1_counts"] == {"left_higher":6, "equal":18, "right_higher":0}
    assert pair["em_matrix_correct_then_incorrect"] == [[6,6],[0,12]]
    assert pair["left_scores"]["em"] == .5


def test_incomplete_duplicate_or_mis_scored_predictions_rejected():
    req, pred = toy_rows(report.R2, wins=1)
    truth = report.index_requests(req)
    for bad in (pred[:-1], pred + pred[:1], [dict(pred[0], f1=0)] + pred[1:],
                [dict(pred[0], trajectory_lost=False)] + pred[1:]):
        with pytest.raises(ValueError):
            report.collect(bad, truth, [report.R2])


def test_repeated_controls_sample_sizes_and_true_empty_groups():
    req, pred, costs = [], [], {}
    for s in report.STRATEGIES:
        source, predictions = toy_rows(s, wins=2)
        req.extend(source)
        pred.extend(predictions)
        costs[s] = {cell:dict(n=1,cached_mean=123.,recomputed_cached_mean=123.) for cell in report.CELLS}
    pf = {"no_compress":{"all_identical":True, "reference_cell":"append|tight","tasks":1,
                          "differing_cells":0,"differing_tasks":0,"by_cell":dict.fromkeys(report.CELLS,0)}, "costs":costs}
    data = report.collect(pred, report.index_requests(req), report.STRATEGIES)
    result = report.make_results(data, pf, n_tasks=1)
    assert result["n_predictions"] == 8*24+2*4
    assert result["strategies"]["no_compress"]["pooled"]["n"] == 4
    assert result["strategies"]["no_context"]["cells"]["shared"]["cost"]["cached_mean"] == 0
    assert result["strategies"]["no_compress"]["cells"]["shared"]["cost"]["n"] == 6
    g = result["strategies"][report.R2]["pooled"]
    assert g["by_hops"]["3"] == {"n":0,"f1":None,"em":None}
    assert g["by_position"]["3"]["n"] == 6
    assert g["by_trajectory_lost"]["true"]["n"] == 24
    assert g["prompt_tokens"] == {"n":24,"mean":9.5,"p50":9.5,"p90":11.,"max":11}
    result["smoke"] = {"n":40,"raw_output_mismatches":0}
    md = report.render_markdown(result)
    assert "缓存成本" in md and "预算" in md and "逐题配对" in md
    broken = deepcopy(pf)
    broken["costs"][report.R2]["append|mid"]["cached_mean"] += 1
    with pytest.raises(ValueError, match="cached cost"):
        report.make_results(data, broken, n_tasks=1)


def test_wrong_trajectory_flag_and_missing_config_fail():
    req, pred = toy_rows(report.R2)
    data = report.collect(pred, report.index_requests(req), [report.R2])
    scores = data[report.R2]["append|tight"]
    from dataclasses import replace
    scores[700,1] = replace(scores[700,1], lost=False)
    with pytest.raises(ValueError, match="trajectory loss"):
        report.audit_matrix(data, None, n_tasks=1)
    del data[report.R2]["append|tight"]
    with pytest.raises(ValueError, match="incomplete cells"):
        report.audit_matrix(data, None, n_tasks=1)


def test_refuse_overwrite_before_publishing_any_file(tmp_path):
    existing = tmp_path / "second"
    existing.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        report.publish(tmp_path, {"first": b"new", "second": b"replacement"})
    assert existing.read_bytes() == b"original"
    assert not (tmp_path / "first").exists()

