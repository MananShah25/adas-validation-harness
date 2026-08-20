"""Phase 9: confirms Track 1 (simulation) and Track 2 (OBD-II + vision
backbone) converge on the identical report/schema format, per the project
spec's requirement not to let the two tracks' schemas diverge. Does not
test a merged report -- that merge is explicitly out of scope (Option C).
"""

from pathlib import Path

from fusion.aeb_pipeline import build_fusion_scenarios, run_fusion_scenario
from reporting.defect_log import _FIELDNAMES as DEFECT_LOG_FIELDNAMES
from reporting.defect_log import DefectLog
from run_full_validation_suite import run_and_confirm_convergence
from schemas.test_procedure import DataSource, TestProcedure, TestResult
from scenarios.acc_scenarios import build_acc_scenarios, run_acc_scenario


def _report_sections(content: str) -> list[str]:
    return [line for line in content.splitlines() if line.startswith("## ")]


def _table_header(content: str, section: str) -> str:
    lines = content.splitlines()
    start = lines.index(f"## {section}")
    for line in lines[start:]:
        if line.startswith("| ") and "---" not in line:
            return line
    raise AssertionError(f"no table header found under ## {section}")


def test_track1_and_track2_results_are_literally_the_same_types():
    t1_spec = build_acc_scenarios()[0]
    t1_result = run_acc_scenario(t1_spec)

    t2_spec = build_fusion_scenarios()[0]
    t2_result = run_fusion_scenario(t2_spec)

    assert type(t1_result) is TestResult
    assert type(t2_result) is TestResult
    assert type(t1_result.procedure) is TestProcedure
    assert type(t2_result.procedure) is TestProcedure
    assert isinstance(t1_result.procedure.data_source, DataSource)
    assert isinstance(t2_result.procedure.data_source, DataSource)
    # Every TestResult, regardless of track, exposes the same fields.
    assert set(vars(t1_result).keys()) == set(vars(t2_result).keys())
    assert set(vars(t1_result.procedure).keys()) == set(vars(t2_result.procedure).keys())


def test_both_tracks_produce_reports_with_identical_structure(tmp_path):
    t1_report, t2_report = run_and_confirm_convergence(output_dir=str(tmp_path / "reports"))
    t1_content = Path(t1_report).read_text()
    t2_content = Path(t2_report).read_text()

    assert t1_content.startswith("# ADAS Validation Report")
    assert t2_content.startswith("# ADAS Validation Report")

    assert _report_sections(t1_content) == _report_sections(t2_content)
    assert _table_header(t1_content, "Summary") == _table_header(t2_content, "Summary")
    assert _table_header(t1_content, "Scenario results") == _table_header(t2_content, "Scenario results")


def test_both_tracks_defect_logs_share_identical_columns(tmp_path):
    log1 = DefectLog(tmp_path / "t1_defects.csv")
    log2 = DefectLog(tmp_path / "t2_defects.csv")

    header1 = log1.path.read_text().splitlines()[0]
    header2 = log2.path.read_text().splitlines()[0]
    assert header1 == header2 == ",".join(DEFECT_LOG_FIELDNAMES)
