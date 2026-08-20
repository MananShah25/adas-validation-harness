"""LKA scenarios: a straight-segment test via highway-env's highway-v0, and a
winding-road test built directly on highway-env's Road/Vehicle/SineLane
physics, since highway-v0 only supports straight multi-lane roads.

acc_controller handles longitudinal speed as the spec asks, but with no real
lead vehicle: each step it's fed a synthetic lead exactly at the desired gap
for the ego's current speed and traveling at the target cruise speed, which
makes its gap term cancel out and leaves a simple proportional speed-hold.
That's enough to keep the vehicle moving at a reasonable pace for what is
fundamentally a steering (lka_controller) test, without needing a second
vehicle on the road.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium
import highway_env  # noqa: F401  (registers the highway-v0 env on import)
import numpy as np
from highway_env import utils
from highway_env.road.lane import SineLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.kinematics import Vehicle

from controllers.acc import acc_controller
from controllers.lka import lka_controller
from metrics.ttc_lane_brake import lane_deviation
from reporting.defect_log import Defect, DefectLog, Severity
from reporting.report_builder import ReportBuilder
from schemas.test_procedure import DEFAULT_PASS_CRITERIA, DataSource, Feature, TestProcedure, TestResult

ACCELERATION_RANGE = (-5.0, 5.0)
STEERING_RANGE = (-math.pi / 4, math.pi / 4)


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _cruise_accel(ego_speed: float, target_speed: float) -> float:
    desired_gap = 2.0 + 1.5 * ego_speed
    return acc_controller(ego_speed=ego_speed, lead_speed=target_speed, gap=desired_gap)


@dataclass
class LkaScenarioSpec:
    scenario_id: str
    description: str
    road_type: str  # "straight" or "curve"
    initial_speed: float
    target_speed: float
    initial_lateral_offset: float = 0.0
    initial_heading_error: float = 0.0
    duration_s: float = 20.0
    policy_frequency: int = 10
    pass_criteria: dict = field(default_factory=lambda: dict(DEFAULT_PASS_CRITERIA[Feature.LKA]))
    curve_amplitude: float = 3.0
    curve_period: float = 120.0
    curve_length: float = 400.0


def _make_straight_env(policy_frequency: int, duration_s: float):
    config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 1,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": True,
            "normalize": False,
        },
        "action": {
            "type": "ContinuousAction",
            "longitudinal": True,
            "lateral": True,
        },
        "lanes_count": 2,
        "vehicles_count": 0,
        "controlled_vehicles": 1,
        "initial_lane_id": 0,
        "duration": duration_s,
        "policy_frequency": policy_frequency,
        "simulation_frequency": 15,
        "ego_spacing": 2,
        "offroad_terminal": False,
    }
    return gymnasium.make("highway-v0", config=config)


def _build_curve_lane(amplitude: float, period: float, length: float) -> SineLane:
    pulsation = 2 * math.pi / period
    return SineLane(start=[0.0, 0.0], end=[length, 0.0], amplitude=amplitude, pulsation=pulsation, phase=0.0)


def _finish_lka_result(
    spec: LkaScenarioSpec,
    sample_times: list[float],
    offset_series: list[float],
    heading_err_series: list[float],
    speed_series: list[float],
    max_deviation: float,
) -> TestResult:
    passed = max_deviation <= spec.pass_criteria.get("max_lane_deviation_m", 0.3)

    procedure = TestProcedure(
        feature=Feature.LKA,
        scenario_id=spec.scenario_id,
        description=spec.description,
        preconditions={
            "road_type": spec.road_type,
            "initial_speed_mps": spec.initial_speed,
            "target_speed_mps": spec.target_speed,
            "initial_lateral_offset_m": spec.initial_lateral_offset,
            "initial_heading_error_rad": spec.initial_heading_error,
        },
        pass_criteria=spec.pass_criteria,
        data_source=DataSource.SIMULATION,
    )

    return TestResult(
        procedure=procedure,
        passed=passed,
        observed_metrics={"max_lane_deviation_m": max_deviation},
        time_series={
            "lateral_offset_m": offset_series,
            "heading_error_rad": heading_err_series,
            "speed_mps": speed_series,
        },
        sample_times=sample_times,
    )


def run_lka_straight_scenario(spec: LkaScenarioSpec, seed: int = 0) -> TestResult:
    env = _make_straight_env(spec.policy_frequency, spec.duration_s)
    env.reset(seed=seed)
    ego = env.unwrapped.controlled_vehicles[0]
    lane = env.unwrapped.road.network.get_lane(("0", "1", 0))

    ego.position = np.array(lane.position(0.0, spec.initial_lateral_offset))
    ego.heading = lane.heading_at(0.0) + spec.initial_heading_error
    ego.speed = spec.initial_speed

    dt = 1.0 / spec.policy_frequency
    steps = int(spec.duration_s * spec.policy_frequency)
    sample_times: list[float] = []
    offset_series: list[float] = []
    heading_err_series: list[float] = []
    speed_series: list[float] = []
    max_deviation = 0.0
    t = 0.0

    for _ in range(steps):
        s, lateral_offset = lane.local_coordinates(ego.position)
        s, lateral_offset = float(s), float(lateral_offset)
        heading_error = _wrap_angle(ego.heading - lane.heading_at(s))
        ego_speed = float(ego.speed)

        steering = lka_controller(lateral_offset=lateral_offset, heading_error=heading_error)
        accel = _cruise_accel(ego_speed, spec.target_speed)

        sample_times.append(t)
        offset_series.append(lateral_offset)
        heading_err_series.append(heading_error)
        speed_series.append(ego_speed)
        max_deviation = max(max_deviation, lane_deviation(lateral_offset))

        accel_norm = utils.lmap(accel, list(ACCELERATION_RANGE), [-1.0, 1.0])
        steer_norm = utils.lmap(steering, list(STEERING_RANGE), [-1.0, 1.0])
        action = np.array([_clip(accel_norm, -1.0, 1.0), _clip(steer_norm, -1.0, 1.0)], dtype=np.float32)

        _, _, terminated, truncated, _ = env.step(action)
        t += dt
        if terminated or truncated:
            break

    env.close()
    return _finish_lka_result(spec, sample_times, offset_series, heading_err_series, speed_series, max_deviation)


def run_lka_curve_scenario(spec: LkaScenarioSpec, seed: int = 0) -> TestResult:
    lane = _build_curve_lane(spec.curve_amplitude, spec.curve_period, spec.curve_length)
    net = RoadNetwork()
    net.add_lane("0", "1", lane)
    road = Road(network=net, np_random=np.random.RandomState(seed), record_history=False)

    ego = Vehicle(
        road,
        position=lane.position(0.0, spec.initial_lateral_offset),
        heading=lane.heading_at(0.0) + spec.initial_heading_error,
        speed=spec.initial_speed,
    )
    road.vehicles.append(ego)

    dt = 1.0 / spec.policy_frequency
    steps = int(spec.duration_s * spec.policy_frequency)
    sample_times: list[float] = []
    offset_series: list[float] = []
    heading_err_series: list[float] = []
    speed_series: list[float] = []
    max_deviation = 0.0
    t = 0.0

    for _ in range(steps):
        s, lateral_offset = lane.local_coordinates(ego.position)
        s, lateral_offset = float(s), float(lateral_offset)
        heading_error = _wrap_angle(ego.heading - lane.heading_at(s))
        ego_speed = float(ego.speed)

        steering = lka_controller(lateral_offset=lateral_offset, heading_error=heading_error)
        accel = _cruise_accel(ego_speed, spec.target_speed)

        sample_times.append(t)
        offset_series.append(lateral_offset)
        heading_err_series.append(heading_error)
        speed_series.append(ego_speed)
        max_deviation = max(max_deviation, lane_deviation(lateral_offset))

        ego.action = {"steering": steering, "acceleration": accel}
        ego.step(dt)
        t += dt

        if s >= spec.curve_length - 1.0:
            break

    return _finish_lka_result(spec, sample_times, offset_series, heading_err_series, speed_series, max_deviation)


def run_lka_scenario(spec: LkaScenarioSpec, seed: int = 0) -> TestResult:
    if spec.road_type == "straight":
        return run_lka_straight_scenario(spec, seed=seed)
    if spec.road_type == "curve":
        return run_lka_curve_scenario(spec, seed=seed)
    raise ValueError(f"Unknown road_type: {spec.road_type!r}")


def build_lka_scenarios() -> list[LkaScenarioSpec]:
    return [
        LkaScenarioSpec(
            scenario_id="lka_straight_centered_cruise",
            description="Straight road, starts perfectly centered; sanity check.",
            road_type="straight",
            initial_speed=22.0,
            target_speed=22.0,
        ),
        LkaScenarioSpec(
            scenario_id="lka_straight_small_offset_recovery",
            description="Straight road, starts 0.15m off-center; should recover within tolerance.",
            road_type="straight",
            initial_speed=22.0,
            target_speed=22.0,
            initial_lateral_offset=0.15,
        ),
        LkaScenarioSpec(
            scenario_id="lka_straight_heading_error_recovery",
            description="Straight road, centered but with a 0.1 rad initial heading error.",
            road_type="straight",
            initial_speed=20.0,
            target_speed=20.0,
            initial_heading_error=0.1,
        ),
        LkaScenarioSpec(
            scenario_id="lka_straight_large_offset_stress",
            description="Straight road, large 0.6m initial offset (stress test of controller authority).",
            road_type="straight",
            initial_speed=22.0,
            target_speed=22.0,
            initial_lateral_offset=0.6,
        ),
        LkaScenarioSpec(
            scenario_id="lka_straight_high_speed",
            description="Straight road, small offset at highway speed (30 m/s).",
            road_type="straight",
            initial_speed=30.0,
            target_speed=30.0,
            initial_lateral_offset=0.15,
        ),
        LkaScenarioSpec(
            scenario_id="lka_straight_low_speed",
            description="Straight road, small offset at low urban speed (10 m/s).",
            road_type="straight",
            initial_speed=10.0,
            target_speed=10.0,
            initial_lateral_offset=0.15,
        ),
        LkaScenarioSpec(
            scenario_id="lka_curve_centered",
            description="Winding road (SineLane), starts centered; the core 'through a curve' test.",
            road_type="curve",
            initial_speed=18.0,
            target_speed=18.0,
            curve_amplitude=3.0,
            curve_period=120.0,
            curve_length=400.0,
        ),
        LkaScenarioSpec(
            scenario_id="lka_curve_small_offset_recovery",
            description="Winding road, starts with a small initial offset while also tracking the curve.",
            road_type="curve",
            initial_speed=18.0,
            target_speed=18.0,
            initial_lateral_offset=0.15,
            curve_amplitude=3.0,
            curve_period=120.0,
            curve_length=400.0,
        ),
        LkaScenarioSpec(
            scenario_id="lka_curve_tighter_curve_stress",
            description="A tighter, more aggressive curve (shorter period) at moderate speed.",
            road_type="curve",
            initial_speed=18.0,
            target_speed=18.0,
            curve_amplitude=3.0,
            curve_period=60.0,
            curve_length=300.0,
        ),
        LkaScenarioSpec(
            scenario_id="lka_curve_high_speed",
            description="Same winding road as lka_curve_centered, but at highway speed (28 m/s).",
            road_type="curve",
            initial_speed=28.0,
            target_speed=28.0,
            curve_amplitude=3.0,
            curve_period=120.0,
            curve_length=450.0,
        ),
    ]


def run_all(output_dir: str = "reports", defect_log_path: str = "reports/lka_defect_log.csv") -> Path:
    defect_log = DefectLog(defect_log_path)
    results = [run_lka_scenario(spec) for spec in build_lka_scenarios()]

    for result in results:
        if result.passed:
            continue
        deviation = result.observed_metrics.get("max_lane_deviation_m", 0.0)
        threshold = result.procedure.pass_criteria.get("max_lane_deviation_m", 0.3)
        severity = Severity.HIGH if deviation > 2 * threshold else Severity.MEDIUM
        defect_log.log(
            Defect(
                feature=Feature.LKA,
                scenario_id=result.procedure.scenario_id,
                expected=f"pass_criteria satisfied: {result.procedure.pass_criteria}",
                observed=f"observed_metrics: {result.observed_metrics}",
                severity=severity,
                reproduction_steps=(
                    "run_lka_scenario(spec) for spec.scenario_id == "
                    f"'{result.procedure.scenario_id}' in scenarios/lka_scenarios.py"
                ),
                data_source=DataSource.SIMULATION,
            )
        )

    builder = ReportBuilder(output_dir=output_dir)
    return builder.build(results, defect_log=defect_log)


if __name__ == "__main__":
    report_path = run_all()
    print(f"Report written to {report_path}")
