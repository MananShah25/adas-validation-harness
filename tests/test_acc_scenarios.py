from pathlib import Path

from reporting.defect_log import DefectLog
from schemas.test_procedure import Feature
from scenarios.acc_scenarios import build_acc_scenarios, run_acc_scenario, run_all


def test_scenario_coverage_is_8_to_10():
    specs = build_acc_scenarios()
    assert 8 <= len(specs) <= 10
    assert len({s.scenario_id for s in specs}) == len(specs)


def test_steady_matched_speed_scenario_passes_at_equilibrium():
    spec = next(s for s in build_acc_scenarios() if s.scenario_id == "acc_steady_matched_speed")
    result = run_acc_scenario(spec)
    assert result.passed
    assert result.procedure.feature is Feature.ACC
    assert result.observed_metrics["final_following_distance_error_m"] < 0.01
    assert len(result.sample_times) > 0
    assert result.time_series["gap_m"]


def test_hard_brake_exceeds_authority_scenario_fails_on_ttc():
    spec = next(
        s for s in build_acc_scenarios() if s.scenario_id == "acc_hard_brake_exceeds_authority"
    )
    result = run_acc_scenario(spec)
    assert not result.passed
    assert result.observed_metrics["min_ttc_s"] < spec.pass_criteria["min_time_to_collision_s"]


def test_run_all_produces_report_and_defect_log(tmp_path):
    output_dir = tmp_path / "reports"
    defect_log_path = tmp_path / "reports" / "defect_log.csv"

    report_path = run_all(output_dir=str(output_dir), defect_log_path=str(defect_log_path))

    assert Path(report_path).exists()
    content = Path(report_path).read_text()
    assert "ACC" in content
    assert "FAIL" in content
    assert "PASS" in content

    defect_log = DefectLog(defect_log_path)
    defects = defect_log.read_all()
    assert len(defects) >= 1
    assert all(d["feature"] == "ACC" for d in defects)
