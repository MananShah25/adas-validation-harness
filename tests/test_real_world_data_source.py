"""DataSource.REAL_WORLD exists in the schema, but until real hardware
arrives no scenario emits it. These tests exercise that path deliberately,
so the day a live drive is recorded the reporting side is already known to
handle it -- rather than discovering a formatting bug with the car running.

They also pin the labelling rule: a run is only REAL_WORLD when every input
is real. Replayed or synthesized inputs stay SIMULATION.
"""

import math

import pytest

from fusion.aeb_pipeline import (
    AebFusionPipeline,
    FusionScenarioSpec,
    SyntheticGapSource,
    build_fusion_scenarios,
    run_fusion_scenario,
)
from ingestion.obd_reader import OBDReader
from reporting.defect_log import Defect, DefectLog, Severity
from reporting.report_builder import ReportBuilder
from schemas.test_procedure import DataSource, Feature, TestProcedure, TestResult


def _real_world_result(passed=False):
    procedure = TestProcedure(
        feature=Feature.AEB,
        scenario_id="aeb_live_drive_001",
        description="Logged drive, lead vehicle brakes on approach to a junction",
        preconditions={"ego_speed_source": "OBDReader(mode='live')", "gap_source": "dashcam"},
        pass_criteria={"brake_time_to_collision_s": 1.8},
        data_source=DataSource.REAL_WORLD,
    )
    return TestResult(
        procedure=procedure,
        passed=passed,
        observed_metrics={"min_ttc_s": 1.4, "triggered": 1.0},
        time_series={"ttc_s": [4.0, 3.1, 2.2, 1.4]},
        sample_times=[0.0, 0.4, 0.8, 1.2],
    )


def test_real_world_result_renders_in_markdown_report(tmp_path):
    builder = ReportBuilder(output_dir=tmp_path / "reports")
    content = builder.build([_real_world_result()]).read_text()

    assert "real_world" in content
    assert "aeb_live_drive_001" in content
    assert "FAIL" in content


def test_real_world_result_renders_in_pdf_report(tmp_path):
    builder = ReportBuilder(output_dir=tmp_path / "reports")
    text = "\n".join(builder._pdf_text_lines([_real_world_result()], None))
    assert "real_world" in text
    assert "aeb_live_drive_001" in text


def test_mixed_sources_in_one_report_stay_distinguishable(tmp_path):
    """Simulation and real-world results must remain separable in the same
    report -- that separation is the whole point of the field."""
    simulated = run_fusion_scenario(build_fusion_scenarios()[0])
    builder = ReportBuilder(output_dir=tmp_path / "reports")
    content = builder.build([simulated, _real_world_result()]).read_text()

    rows = [line for line in content.splitlines() if line.startswith("| ") and "AEB" in line]
    sources = {"real_world" in row for row in rows}
    assert sources == {True, False}, "report should contain both a real_world and a simulation row"


def test_real_world_defect_logs_with_correct_source(tmp_path):
    log = DefectLog(tmp_path / "defects.csv")
    log.log(
        Defect(
            feature=Feature.AEB,
            scenario_id="aeb_live_drive_001",
            expected="TTC stays above 1.8s",
            observed="TTC fell to 1.4s",
            severity=Severity.CRITICAL,
            reproduction_steps="replay logged drive 001",
            data_source=DataSource.REAL_WORLD,
        )
    )
    rows = log.read_all()
    assert rows[0]["data_source"] == "real_world"
    assert "real_world" in log.to_markdown()


def test_fusion_spec_can_declare_real_world():
    """The plumbing for a genuinely real run exists and is reachable today,
    even though no such run can be produced without hardware."""
    spec = FusionScenarioSpec(
        scenario_id="fusion_live_drive",
        description="placeholder for a real logged drive",
        gap_source=SyntheticGapSource(initial_gap_m=40.0, closing_rate_mps=6.0),
        expect_trigger=True,
        data_source=DataSource.REAL_WORLD,
        gap_source_label="vision (real dashcam footage)",
    )
    result = run_fusion_scenario(spec)
    assert result.procedure.data_source is DataSource.REAL_WORLD


def test_shipped_scenarios_all_declare_simulation():
    """No scenario in the repo may claim real_world while its inputs are
    mock, replayed or synthesized."""
    for spec in build_fusion_scenarios():
        result = run_fusion_scenario(spec)
        assert result.procedure.data_source is DataSource.SIMULATION, spec.scenario_id


def test_fusion_accepts_an_injected_reader_factory():
    """A live run swaps the reader without touching pipeline code -- the
    design bet this project rests on. Exercised here with a mock reader
    standing in for the live one."""
    built = {}

    def factory():
        built["called"] = True
        return OBDReader(mode="mock", seed=3, hz=10.0, initial_speed_kph=60.0)

    spec = FusionScenarioSpec(
        scenario_id="fusion_injected_reader",
        description="reader supplied by factory",
        gap_source=SyntheticGapSource(initial_gap_m=40.0, closing_rate_mps=6.0),
        expect_trigger=True,
        obd_reader_factory=factory,
        duration_s=5.0,
    )
    result = run_fusion_scenario(spec)
    assert built.get("called") is True
    assert result.observed_metrics["triggered"] == 1.0
