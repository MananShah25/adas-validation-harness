from pathlib import Path

from reporting.defect_log import DefectLog
from schemas.test_procedure import Feature
from scenarios.lka_scenarios import build_lka_scenarios, run_lka_scenario, run_all


def test_scenario_coverage_is_8_to_10():
    specs = build_lka_scenarios()
    assert 8 <= len(specs) <= 10
    assert len({s.scenario_id for s in specs}) == len(specs)


def test_covers_both_straight_and_curve_road_types():
    road_types = {s.road_type for s in build_lka_scenarios()}
    assert road_types == {"straight", "curve"}


def test_centered_cruise_stays_perfectly_centered():
    spec = next(s for s in build_lka_scenarios() if s.scenario_id == "lka_straight_centered_cruise")
    result = run_lka_scenario(spec)
    assert result.passed
    assert result.procedure.feature is Feature.LKA
    assert result.observed_metrics["max_lane_deviation_m"] == 0.0


def test_large_offset_stress_scenario_fails():
    spec = next(s for s in build_lka_scenarios() if s.scenario_id == "lka_straight_large_offset_stress")
    result = run_lka_scenario(spec)
    assert not result.passed
    assert result.observed_metrics["max_lane_deviation_m"] > spec.pass_criteria["max_lane_deviation_m"]


def test_curve_centered_scenario_tracks_within_tolerance():
    spec = next(s for s in build_lka_scenarios() if s.scenario_id == "lka_curve_centered")
    result = run_lka_scenario(spec)
    assert result.passed
    assert len(result.sample_times) > 0
    assert result.time_series["lateral_offset_m"]


def test_run_all_produces_report_and_defect_log(tmp_path):
    output_dir = tmp_path / "reports"
    defect_log_path = tmp_path / "reports" / "lka_defect_log.csv"

    report_path = run_all(output_dir=str(output_dir), defect_log_path=str(defect_log_path))

    assert Path(report_path).exists()
    content = Path(report_path).read_text()
    assert "LKA" in content

    defect_log = DefectLog(defect_log_path)
    defects = defect_log.read_all()
    assert len(defects) >= 1
    assert all(d["feature"] == "LKA" for d in defects)
