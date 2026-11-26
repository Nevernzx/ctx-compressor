"""Task 20 signs raw attempts without inspecting CPU-only test fixtures."""
from scripts.agent_v1_seal import attempt_files


def test_attempt_listing_prunes_cpu_fixture_subtrees(tmp_path):
    attempt = tmp_path.resolve()
    root_record = attempt / "status.json"
    root_record.write_text("failed")
    log = attempt / "logs" / "main.log"
    log.parent.mkdir()
    log.write_text("rule a")
    for name in ("closeout", "cpu_telemetry_check"):
        fixtures = attempt / name / "cases"
        fixtures.mkdir(parents=True)
        (fixtures / "alias").symlink_to(log)
    assert set(attempt_files(attempt)) == {root_record, log}

