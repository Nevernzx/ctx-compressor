"""The v2 report pairs signed instances exactly and groups duplicate prompts by tokens."""
from copy import deepcopy
import pytest

from scripts import reader_heldout_v2_report as report
from tests.test_reader_heldout_report import toy_rows


def test_registered_triplet_uses_unrounded_f1_and_matching_em_order():
    prev=report.previous
    counts={prev.R2:2,prev.V3:1,prev.H1:3,report.diag.R3:4}
    data={}
    for name, correct in counts.items():
        req, rows=toy_rows(name,wins=correct)
        data.update(prev.collect(rows,prev.index_requests(req),[name]))
    r3_r2=prev.paired(report.diag.R3,prev.R2,data)
    assert r3_r2["em_matrix_correct_then_incorrect"]==[[12,12],[0,0]]
    assert r3_r2["f1_difference_pp"]==r3_r2["em_difference_pp"]==50
    assert prev.paired(prev.R2,prev.H1,data)["f1_difference_pp"]==-25
    assert prev.paired(prev.R2,prev.V3,data)["f1_difference_pp"]==25


def test_token_identity_crosses_strategies_and_rejects_order_change():
    r3_req,_=toy_rows(report.diag.R3)
    r2_req,_=toy_rows(report.diag.R2)
    controls=[r for name in ("no_compress","no_context") for r in toy_rows(name)[0]]
    # Six other strategy batches complete the historical 200-row prefix.
    hist=controls+r2_req
    for name in report.previous.STRATEGIES:
        if name not in ("no_compress","no_context",report.diag.R2):
            hist+=toy_rows(name)[0]
    inputs=hist+r3_req
    for i,r in enumerate(inputs):
        r["token_ids"]=[123,456,i]
    # Text fingerprints can differ while exact token sequences still identify a duplicate.
    r3_req[0]["token_ids"]=r2_req[0]["token_ids"]
    r3_req[0]["prompt_sha1"]="different"
    sources,ids,order=report.index_inputs(inputs)
    assert order==list(sources)
    assert ids[report.key(r3_req[0])]==ids[report.key(r2_req[0])]
    assert len(set(ids.values()))==223
    drift=[dict(r,raw_output="answer") for r in reversed(inputs)]
    rows,raw=report.index_outputs(drift,order)
    with pytest.raises(ValueError,match="order"):
        list(rows)


def test_paired_historical_table_checks_coverage_and_metadata():
    name=report.diag.R3
    req,old=toy_rows(name,wins=1)
    for row in old:
        row.pop('token_ids',None)
    sources=report.previous.index_requests(req)
    raw={report.key(r):r["raw_output"] for r in old}
    raw[report.key(old[0])]=" raw spelling differs "
    summary={name:{"pooled":{"n":24,"f1":.25,"em":.25}}}
    per=report.historical_pair(reversed(old),sources,raw,[name],summary)[name]
    assert (per["raw_different"],per["n"],per["f1"])==(1,24,.25)
    with pytest.raises(ValueError,match="coverage"):
        report.historical_pair(old[1:],sources,raw,[name],summary)
    bad=deepcopy(old);bad[2]["prompt_sha1"]="other"
    with pytest.raises(ValueError,match="metadata"):
        report.historical_pair(bad,sources,raw,[name],summary)
    with pytest.raises(ValueError,match="historical key"):
        report.historical_pair(old+[old[0]],sources,raw,[name],summary)

