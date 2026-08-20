import math
from pathlib import Path

import pytest

from fusion.aeb_pipeline import (
    AebFusionPipeline,
    SyntheticGapSource,
    build_fusion_scenarios,
    run_all,
    run_fusion_scenario,
)
from ingestion.obd_reader import OBDReader
from reporting.defect_log import DefectLog
from schemas.test_procedure import Feature


def test_synthetic_gap_source_linear_and_clamped_at_zero():
    source = SyntheticGapSource(initial_gap_m=40.0, closing_rate_mps=6.0)
    assert source(0.0) == pytest.approx(40.0)
    assert source(5.0) == pytest.approx(10.0)
    assert source(100.0) == 0.0  # clamped, never negative


def test_first_pipeline_step_has_no_prior_gap_so_ttc_is_infinite():
    reader = OBDReader(mode="mock", seed=0, initial_speed_kph=50.0)
    pipeline = AebFusionPipeline(reader, SyntheticGapSource(initial_gap_m=40.0, closing_rate_mps=6.0))
    result = pipeline.step()
    assert math.isinf(result.ttc_s)
    assert result.aeb_state == "normal"


def test_pipeline_derives_closing_rate_from_consecutive_gap_readings():
    reader = OBDReader(mode="mock", seed=0, hz=10.0, initial_speed_kph=50.0)
    pipeline = AebFusionPipeline(reader, SyntheticGapSource(initial_gap_m=40.0, closing_rate_mps=6.0))
    pipeline.step()  # first step: establishes the baseline gap, ttc still inf
    result = pipeline.step()  # second step: a real closing rate can now be derived
    # gap closes at a fixed 6 m/s regardless of ego speed -> ttc = gap / 6
    assert result.ttc_s == pytest.approx(result.gap_m / 6.0, rel=1e-6)


def test_scenario_coverage_has_unique_ids():
    specs = build_fusion_scenarios()
    assert len(specs) >= 3
    assert len({s.scenario_id for s in specs}) == len(specs)


def test_closing_hazard_triggers_full_brake():
    spec = next(s for s in build_fusion_scenarios() if s.scenario_id == "fusion_closing_hazard")
    result = run_fusion_scenario(spec)
    assert result.passed
    assert result.procedure.feature is Feature.AEB
    assert result.observed_metrics["triggered"] == 1.0
    # brake_ttc=1.8s crossed when gap < 6*1.8=10.8m i.e. t > (40-10.8)/6 = 4.867s
    assert result.observed_metrics["brake_onset_time_s"] == pytest.approx(4.9, abs=0.15)


def test_steady_gap_does_not_trigger_nuisance_alert():
    spec = next(s for s in build_fusion_scenarios() if s.scenario_id == "fusion_steady_safe_gap")
    result = run_fusion_scenario(spec)
    assert result.passed
    assert result.observed_metrics["triggered"] == 0.0
    assert result.observed_metrics["min_ttc_s"] == 99.0  # capped infinite TTC, no closing


def test_run_all_produces_report_and_handles_zero_defects(tmp_path):
    output_dir = tmp_path / "reports"
    defect_log_path = tmp_path / "reports" / "fusion_defect_log.csv"

    report_path = run_all(output_dir=str(output_dir), defect_log_path=str(defect_log_path))

    assert Path(report_path).exists()
    content = Path(report_path).read_text()
    assert "AEB" in content

    # All scenarios are expected to pass, so the defect log should exist but be empty.
    defects = DefectLog(defect_log_path).read_all()
    assert defects == []
