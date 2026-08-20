import re

import pytest

from reporting.defect_log import Defect, DefectLog, Severity
from reporting.report_builder import ReportBuilder
from schemas.test_procedure import DataSource, Feature, TestProcedure, TestResult


def _result(scenario_id="aeb_demo", passed=False, data_source=DataSource.SIMULATION, with_series=True):
    procedure = TestProcedure(
        feature=Feature.AEB,
        scenario_id=scenario_id,
        description="Lead brakes hard",
        preconditions={"ego_speed_kmh": 50},
        pass_criteria={"brake_time_to_collision_s": 1.8},
        data_source=data_source,
    )
    return TestResult(
        procedure=procedure,
        passed=passed,
        observed_metrics={"min_ttc_s": 1.2, "triggered": 1.0},
        time_series={"ttc_s": [3.0, 2.5, 1.8, 1.2], "gap_m": [40, 30, 20, 10]} if with_series else {},
        sample_times=[0.0, 0.5, 1.0, 1.5] if with_series else [],
    )


def _page_count(pdf_bytes: bytes) -> int:
    return len(re.findall(rb"/Type\s*/Page[^s]", pdf_bytes))


def test_build_pdf_writes_a_valid_pdf(tmp_path):
    builder = ReportBuilder(output_dir=tmp_path / "reports")
    pdf_path = builder.build_pdf([_result()])

    assert pdf_path.exists()
    assert pdf_path.suffix == ".pdf"
    data = pdf_path.read_bytes()
    assert data.startswith(b"%PDF-")
    assert _page_count(data) >= 2  # at least one text page plus one plot page


def test_pdf_gains_a_page_per_scenario_with_series(tmp_path):
    builder = ReportBuilder(output_dir=tmp_path / "reports")
    one = _page_count(builder.build_pdf([_result("a")]).read_bytes())
    three = _page_count(builder.build_pdf([_result("a"), _result("b"), _result("c")]).read_bytes())
    assert three == one + 2


def test_pdf_handles_results_without_time_series(tmp_path):
    builder = ReportBuilder(output_dir=tmp_path / "reports")
    pdf_path = builder.build_pdf([_result(with_series=False)])
    assert pdf_path.read_bytes().startswith(b"%PDF-")


def test_pdf_handles_empty_defect_log(tmp_path):
    builder = ReportBuilder(output_dir=tmp_path / "reports")
    empty_log = DefectLog(tmp_path / "empty.csv")
    lines = builder._pdf_text_lines([_result()], empty_log)
    assert any("No defects logged" in line for line in lines)


def test_pdf_text_includes_defects(tmp_path):
    log = DefectLog(tmp_path / "defects.csv")
    log.log(
        Defect(
            feature=Feature.AEB,
            scenario_id="aeb_demo",
            expected="TTC stays above 1.8s",
            observed="TTC fell to 1.2s",
            severity=Severity.HIGH,
            reproduction_steps="run aeb_demo",
            data_source=DataSource.SIMULATION,
        )
    )
    builder = ReportBuilder(output_dir=tmp_path / "reports")
    text = "\n".join(builder._pdf_text_lines([_result()], log))

    assert "DEF-0001" in text
    assert "expected" in text and "observed" in text
    assert "HIGH" in text


def test_markdown_and_pdf_agree_on_the_summary(tmp_path):
    """Both formats must tally the same run identically -- they share
    _feature_counts precisely so they cannot drift apart."""
    results = [_result("a", passed=True), _result("b", passed=False), _result("c", passed=False)]
    builder = ReportBuilder(output_dir=tmp_path / "reports")

    markdown = builder.build(results).read_text()
    pdf_text = "\n".join(builder._pdf_text_lines(results, None))

    assert "| AEB | 1 | 2 | 3 | 33% |" in markdown
    assert any(
        line.split() == ["AEB", "1", "2", "3", "33%"] for line in pdf_text.splitlines()
    ), f"PDF summary row missing or disagrees:\n{pdf_text}"
