from pathlib import Path

from reporting.defect_log import DefectLog
from schemas.test_procedure import Feature
from scenarios.aeb_scenarios import build_aeb_scenarios, run_aeb_scenario, run_all


def test_scenario_coverage_is_8_to_10():
    specs = build_aeb_scenarios()
    assert 8 <= len(specs) <= 10
    assert len({s.scenario_id for s in specs}) == len(specs)


def test_hard_brake_triggers_full_brake_without_collision():
    spec = next(s for s in build_aeb_scenarios() if s.scenario_id == "aeb_lead_hard_brake_cruise_speed")
    result = run_aeb_scenario(spec)
    assert result.passed
    assert result.procedure.feature is Feature.AEB
    assert result.observed_metrics["collided"] == 0.0
    assert result.observed_metrics["brake_response_latency_s"] >= 0.0


def test_moderate_brake_does_not_trigger_nuisance_alert():
    spec = next(
        s for s in build_aeb_scenarios() if s.scenario_id == "aeb_lead_moderate_brake_within_acc_authority"
    )
    result = run_aeb_scenario(spec)
    assert result.passed
    assert result.observed_metrics["brake_response_latency_s"] == -1.0  # never triggered


def test_aeb_covers_the_gap_acc_alone_could_not():
    spec = next(s for s in build_aeb_scenarios() if s.scenario_id == "aeb_covers_acc_authority_gap")
    result = run_aeb_scenario(spec)
    assert result.passed
    assert result.observed_metrics["collided"] == 0.0


def test_already_critical_start_is_unrecoverable():
    spec = next(s for s in build_aeb_scenarios() if s.scenario_id == "aeb_already_critical_at_start")
    result = run_aeb_scenario(spec)
    assert not result.passed
    assert result.observed_metrics["collided"] == 1.0


def test_run_all_produces_report_and_defect_log(tmp_path):
    output_dir = tmp_path / "reports"
    defect_log_path = tmp_path / "reports" / "aeb_defect_log.csv"

    report_path = run_all(output_dir=str(output_dir), defect_log_path=str(defect_log_path))

    assert Path(report_path).exists()
    content = Path(report_path).read_text()
    assert "AEB" in content

    defect_log = DefectLog(defect_log_path)
    defects = defect_log.read_all()
    assert len(defects) >= 1
    assert all(d["feature"] == "AEB" for d in defects)
