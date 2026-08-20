"""Test procedure and result schema shared by both simulation and real-vehicle tracks.

Threshold rationale (researched Aug 2026, see report_builder output for
citations in generated reports):
- AEB warning TTC = 2.0s matches NHTSA's FCW criteria for decelerating/
  stopped lead-vehicle scenarios (TTC > 2.0-2.1s triggers a warning).
- AEB brake TTC = 1.8s follows NHTSA's warning-abort threshold (~1.8-1.9s),
  tighter than the original 1.0s placeholder.
- LKA lane deviation = 0.3m is a conservative proxy, not a direct Euro NCAP
  match: Euro NCAP's LSS protocol scores lane-keeping via Distance-To-Lane-
  Edge (DTLE) relative to the lane marking (-0.1m reference point), not a
  flat lateral offset from lane center.
- No max_brake_response_latency_s gate: an earlier draft included one at
  0.5s, but that number wasn't actually sourced from NHTSA/Euro NCAP (only
  the TTC thresholds above were). Simulation runs showed genuinely safe,
  correctly-behaving AEB scenarios taking 3-5s between the lead vehicle's
  braking onset and AEB engagement, since that gap is dominated by closing
  dynamics and ACC's own moderating response, not system reaction delay.
  scenarios/aeb_scenarios.py still computes and reports brake_response_
  latency_s per scenario as a diagnostic, just doesn't gate pass/fail on
  an invented threshold.
Sources: Euro NCAP AEB C2C Test Protocol v4.3, Euro NCAP Safe Driving
Protocol v1.2 (2026), NHTSA AEB Final Rule (2024), Euro NCAP LSS Test
Protocol v4.3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Feature(str, Enum):
    ACC = "ACC"
    LKA = "LKA"
    AEB = "AEB"


class DataSource(str, Enum):
    SIMULATION = "simulation"
    REAL_WORLD = "real_world"


DEFAULT_PASS_CRITERIA: dict[Feature, dict[str, float]] = {
    Feature.ACC: {
        "min_time_to_collision_s": 2.0,
        "max_following_distance_error_m": 1.0,
    },
    Feature.LKA: {
        "max_lane_deviation_m": 0.3,
    },
    Feature.AEB: {
        "warning_time_to_collision_s": 2.0,
        "brake_time_to_collision_s": 1.8,
    },
}


@dataclass
class TestProcedure:
    feature: Feature
    scenario_id: str
    description: str
    preconditions: dict[str, Any]
    pass_criteria: dict[str, float]
    data_source: DataSource

    def __post_init__(self) -> None:
        if isinstance(self.feature, str):
            self.feature = Feature(self.feature)
        if isinstance(self.data_source, str):
            self.data_source = DataSource(self.data_source)


@dataclass
class TestResult:
    """Outcome of running a TestProcedure once, from either track."""

    procedure: TestProcedure
    passed: bool
    observed_metrics: dict[str, float]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    time_series: dict[str, list[float]] = field(default_factory=dict)
    sample_times: list[float] = field(default_factory=list)
    notes: str = ""
