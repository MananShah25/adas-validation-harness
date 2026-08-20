from pathlib import Path

from reporting.defect_log import DefectLog
from scenarios.acc_scenarios import build_acc_scenarios
from scenarios.aeb_scenarios import build_aeb_scenarios
from scenarios.lka_scenarios import build_lka_scenarios
from scenarios.track1_suite import run_full_suite


def test_full_suite_produces_one_consolidated_report(tmp_path):
    output_dir = tmp_path / "reports"
    defect_log_path = tmp_path / "reports" / "defect_log.csv"

    report_path = run_full_suite(output_dir=str(output_dir), defect_log_path=str(defect_log_path))

    assert Path(report_path).exists()
    content = Path(report_path).read_text()

    # One report, all three features present in the same summary table.
    assert "| ACC |" in content
    assert "| LKA |" in content
    assert "| AEB |" in content

    expected_total = len(build_acc_scenarios()) + len(build_lka_scenarios()) + len(build_aeb_scenarios())
    assert f"acc_steady_matched_speed" in content
    assert f"lka_straight_centered_cruise" in content
    assert f"aeb_lead_hard_brake_cruise_speed" in content

    # Scenario table has one row per scenario across all three features.
    # ("simulation" is the data_source column, unique to scenario rows --
    # the summary table's per-feature rows don't have it.)
    scenario_rows = [
        line for line in content.splitlines()
        if line.startswith("| ") and "simulation" in line
    ]
    assert len(scenario_rows) == expected_total


def test_full_suite_defect_log_spans_multiple_features(tmp_path):
    output_dir = tmp_path / "reports"
    defect_log_path = tmp_path / "reports" / "defect_log.csv"

    run_full_suite(output_dir=str(output_dir), defect_log_path=str(defect_log_path))

    defects = DefectLog(defect_log_path).read_all()
    features_with_defects = {d["feature"] for d in defects}
    assert len(defects) >= 1
    # No numpy scalar reprs (e.g. "np.float64(...)") should leak into the log.
    assert all("np.float64" not in d["observed"] for d in defects)
    # ACC and LKA are both known to have genuine failing scenarios.
    assert {"ACC", "LKA"}.issubset(features_with_defects)
