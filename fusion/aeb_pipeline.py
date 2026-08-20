"""Fusion and scoring: combines OBD-II ego speed with a vision-estimated
gap distance into a single TTC-based AEB decision, scored against the same
AEB pass_criteria schema used by Track 1 (schemas.test_procedure.
DEFAULT_PASS_CRITERIA[Feature.AEB]).

This is the module that becomes the literal, real AEB validation test once
real OBD-II hardware and real dashcam video replace the mock/synthetic
sources fed into it here -- no pipeline code below should need to change,
only the sources (same swap-the-source philosophy as
ingestion/obd_reader.py's mode="mock"/"replay"/"live").

No dashcam footage is available yet (see vision/detector.py), so the gap
distance side is a GapSource: a callable of elapsed time standing in for
vision.detector.VehicleDetector.lead_vehicle(frame).estimated_distance_m
on a synchronized video frame. Every scenario built here is consequently
scored as DataSource.SIMULATION, even ones using OBDReader(mode="replay")
for genuinely real ego-speed data, since the gap side stays synthetic
until real footage lands.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from controllers.aeb import aeb_controller
from ingestion.obd_reader import OBDCommand, OBDReader
from metrics.ttc_lane_brake import time_to_collision
from reporting.defect_log import Defect, DefectLog, Severity
from reporting.report_builder import ReportBuilder
from schemas.test_procedure import DEFAULT_PASS_CRITERIA, DataSource, Feature, TestProcedure, TestResult

# Callable[[elapsed_time_s], gap_m_or_None]. In production this is a
# synchronized video frame's vision.detector output; here, until real
# dashcam footage is available, a scripted synthetic profile.
GapSource = Callable[[float], "float | None"]

TTC_CAP_S = 99.0


def _kph_to_mps(kph: float) -> float:
    return kph / 3.6


@dataclass
class FusionSampleResult:
    timestamp: float
    ego_speed_mps: float
    gap_m: float | None
    ttc_s: float
    aeb_state: str


class SyntheticGapSource:
    """Stands in for a real vision-estimated gap-distance stream until real
    dashcam footage is available. A pure function of elapsed time (not
    internal step count), so it stays correctly synchronized regardless of
    the OBD reader's sampling rate or irregular replay timestamps: gap(t)
    = max(0, initial_gap - closing_rate * t).
    """

    def __init__(self, initial_gap_m: float, closing_rate_mps: float = 0.0):
        self.initial_gap_m = initial_gap_m
        self.closing_rate_mps = closing_rate_mps

    def __call__(self, t: float) -> float | None:
        return max(0.0, self.initial_gap_m - self.closing_rate_mps * t)


class AebFusionPipeline:
    """Combines an OBDReader (ego speed) with a GapSource (vision-estimated
    gap distance) into a running AEB decision, reusing the same TTC metric
    and aeb_controller as Track 1's simulated AEB scenarios.

    Lead speed isn't observed directly -- vision gives a distance, not a
    velocity -- so it's derived from the closing rate between consecutive
    gap readings (lead_speed = ego_speed - d(gap)/dt), then fed through the
    same shared time_to_collision(ego_speed, lead_speed, gap) used
    everywhere else in this project, rather than inventing a second TTC
    formula for this track.
    """

    def __init__(
        self,
        obd_reader: OBDReader,
        gap_source: GapSource,
        pass_criteria: dict | None = None,
    ):
        self.obd_reader = obd_reader
        self.gap_source = gap_source
        self.pass_criteria = dict(pass_criteria or DEFAULT_PASS_CRITERIA[Feature.AEB])
        self._last_gap: float | None = None
        self._last_timestamp: float | None = None

    def step(self) -> FusionSampleResult:
        sample = self.obd_reader.next_sample()
        timestamp = sample[OBDCommand.SPEED].timestamp
        ego_speed_mps = _kph_to_mps(sample[OBDCommand.SPEED].value)
        gap = self.gap_source(timestamp)

        ttc = math.inf
        if gap is not None and self._last_gap is not None and self._last_timestamp is not None:
            dt = timestamp - self._last_timestamp
            if dt > 0:
                closing_rate_mps = (self._last_gap - gap) / dt
                lead_speed_mps = ego_speed_mps - closing_rate_mps
                ttc = time_to_collision(ego_speed=ego_speed_mps, lead_speed=lead_speed_mps, gap_distance=gap)

        aeb_state = aeb_controller(
            ttc,
            warning_ttc=self.pass_criteria.get("warning_time_to_collision_s", 2.0),
            brake_ttc=self.pass_criteria.get("brake_time_to_collision_s", 1.8),
        )

        self._last_gap = gap
        self._last_timestamp = timestamp

        return FusionSampleResult(
            timestamp=timestamp, ego_speed_mps=ego_speed_mps, gap_m=gap, ttc_s=ttc, aeb_state=aeb_state
        )


@dataclass
class FusionScenarioSpec:
    scenario_id: str
    description: str
    gap_source: GapSource
    expect_trigger: bool
    initial_ego_speed_kph: float = 50.0
    duration_s: float = 15.0
    obd_seed: int = 0
    obd_hz: float = 10.0
    pass_criteria: dict = field(default_factory=lambda: dict(DEFAULT_PASS_CRITERIA[Feature.AEB]))


def run_fusion_scenario(spec: FusionScenarioSpec) -> TestResult:
    reader = OBDReader(mode="mock", seed=spec.obd_seed, hz=spec.obd_hz, initial_speed_kph=spec.initial_ego_speed_kph)
    pipeline = AebFusionPipeline(obd_reader=reader, gap_source=spec.gap_source, pass_criteria=spec.pass_criteria)

    n_steps = int(spec.duration_s * spec.obd_hz)
    state_to_int = {"normal": 0, "warning": 1, "full_brake": 2}
    sample_times: list[float] = []
    ttc_series: list[float] = []
    gap_series: list[float] = []
    ego_speed_series: list[float] = []
    brake_state_series: list[float] = []
    min_ttc = math.inf
    brake_onset_time: float | None = None

    for _ in range(n_steps):
        result = pipeline.step()
        sample_times.append(result.timestamp)
        ttc_series.append(result.ttc_s if math.isfinite(result.ttc_s) else TTC_CAP_S)
        gap_series.append(result.gap_m if result.gap_m is not None else -1.0)
        ego_speed_series.append(result.ego_speed_mps)
        brake_state_series.append(float(state_to_int[result.aeb_state]))
        min_ttc = min(min_ttc, result.ttc_s)
        if result.aeb_state == "full_brake" and brake_onset_time is None:
            brake_onset_time = result.timestamp

    passed = brake_onset_time is not None if spec.expect_trigger else brake_onset_time is None

    procedure = TestProcedure(
        feature=Feature.AEB,
        scenario_id=spec.scenario_id,
        description=spec.description,
        preconditions={
            "initial_ego_speed_kph": spec.initial_ego_speed_kph,
            "expect_trigger": spec.expect_trigger,
            "ego_speed_source": "OBDReader(mode='mock')",
            "gap_source": "synthetic (no dashcam footage available yet)",
        },
        pass_criteria=spec.pass_criteria,
        data_source=DataSource.SIMULATION,
    )

    return TestResult(
        procedure=procedure,
        passed=passed,
        observed_metrics={
            "min_ttc_s": min_ttc if math.isfinite(min_ttc) else TTC_CAP_S,
            "triggered": float(brake_onset_time is not None),
            "brake_onset_time_s": brake_onset_time if brake_onset_time is not None else -1.0,
        },
        time_series={
            "ttc_s": ttc_series,
            "gap_m": gap_series,
            "ego_speed_mps": ego_speed_series,
            "brake_state": brake_state_series,
        },
        sample_times=sample_times,
    )


def build_fusion_scenarios() -> list[FusionScenarioSpec]:
    return [
        FusionScenarioSpec(
            scenario_id="fusion_closing_hazard",
            description=(
                "Ego cruises ~50 kph while the (synthetic) vision-estimated gap closes "
                "rapidly from 40m; fusion should derive a closing lead and trigger AEB."
            ),
            initial_ego_speed_kph=50.0,
            gap_source=SyntheticGapSource(initial_gap_m=40.0, closing_rate_mps=6.0),
            expect_trigger=True,
        ),
        FusionScenarioSpec(
            scenario_id="fusion_steady_safe_gap",
            description="Gap holds roughly steady; fusion must not raise a nuisance alert.",
            initial_ego_speed_kph=50.0,
            gap_source=SyntheticGapSource(initial_gap_m=40.0, closing_rate_mps=0.0),
            expect_trigger=False,
        ),
        FusionScenarioSpec(
            scenario_id="fusion_slow_closing_within_margin",
            description="Gap closes slowly enough that TTC should stay above the warning threshold.",
            initial_ego_speed_kph=40.0,
            gap_source=SyntheticGapSource(initial_gap_m=50.0, closing_rate_mps=1.0),
            expect_trigger=False,
        ),
        FusionScenarioSpec(
            scenario_id="fusion_close_range_fast_closing",
            description="A short initial gap closing fast at higher speed; should trigger AEB quickly.",
            initial_ego_speed_kph=70.0,
            gap_source=SyntheticGapSource(initial_gap_m=25.0, closing_rate_mps=8.0),
            expect_trigger=True,
            duration_s=10.0,
        ),
    ]


def run_all(output_dir: str = "reports", defect_log_path: str = "reports/fusion_defect_log.csv") -> Path:
    defect_log = DefectLog(defect_log_path)
    results = [run_fusion_scenario(spec) for spec in build_fusion_scenarios()]

    for result in results:
        if result.passed:
            continue
        defect_log.log(
            Defect(
                feature=Feature.AEB,
                scenario_id=result.procedure.scenario_id,
                expected=f"pass_criteria satisfied: {result.procedure.pass_criteria}",
                observed=f"observed_metrics: {result.observed_metrics}",
                severity=Severity.HIGH,
                reproduction_steps=(
                    "run_fusion_scenario(spec) for spec.scenario_id == "
                    f"'{result.procedure.scenario_id}' in fusion/aeb_pipeline.py"
                ),
                data_source=DataSource.SIMULATION,
            )
        )

    builder = ReportBuilder(output_dir=output_dir)
    return builder.build(results, defect_log=defect_log)


if __name__ == "__main__":
    report_path = run_all()
    print(f"Report written to {report_path}")
