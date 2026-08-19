"""ACC scenarios run against highway-env, scored through the shared validation core.

Environment setup: a single ego vehicle (ContinuousAction, longitudinal-only)
and one scripted "lead" vehicle we control directly frame-by-frame, bypassing
highway-env's default random traffic/IDM behavior so scenarios are fully
deterministic and repeatable. The lead vehicle's speed (and, for cut-in
scenarios, lane position) is driven by a small script function of elapsed
time, using the same kinematic bicycle model highway-env uses for the ego
vehicle.

Simplification: gap/TTC are computed as a simple longitudinal (x-axis)
distance between ego and lead regardless of which lane the lead is
currently in. Before a cut-in vehicle merges into ego's lane this number
isn't a "real" collision-relevant gap, but it becomes meaningful exactly
when the merge happens, which is the behavior under test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import gymnasium
import highway_env  # noqa: F401  (registers the highway-v0 env on import)
import numpy as np
from highway_env import utils
from highway_env.vehicle.kinematics import Vehicle

from controllers.acc import acc_controller
from metrics.ttc_lane_brake import following_distance_error, time_to_collision
from reporting.defect_log import Defect, DefectLog, Severity
from reporting.report_builder import ReportBuilder
from schemas.test_procedure import DEFAULT_PASS_CRITERIA, DataSource, Feature, TestProcedure, TestResult

LANE_WIDTH = 4.0  # matches highway_env.road.lane.StraightLane.DEFAULT_WIDTH
ACCELERATION_RANGE = (-5.0, 5.0)  # matches ContinuousAction.ACCELERATION_RANGE
TTC_CAP_S = 99.0  # cap used only for plotting/logging an otherwise-infinite TTC

LeadScript = Callable[[float, Vehicle], "tuple[float, float]"]


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def steady_script() -> LeadScript:
    def script(t: float, vehicle: Vehicle) -> tuple[float, float]:
        return 0.0, 0.0

    return script


def decel_script(decel_mps2: float, start_t: float, duration: float, min_speed: float = 0.0) -> LeadScript:
    def script(t: float, vehicle: Vehicle) -> tuple[float, float]:
        if start_t <= t < start_t + duration and vehicle.speed > min_speed:
            return -abs(decel_mps2), 0.0
        return 0.0, 0.0

    return script


def accel_script(accel_mps2: float, start_t: float, duration: float, max_speed: float = 35.0) -> LeadScript:
    def script(t: float, vehicle: Vehicle) -> tuple[float, float]:
        if start_t <= t < start_t + duration and vehicle.speed < max_speed:
            return abs(accel_mps2), 0.0
        return 0.0, 0.0

    return script


def stop_and_go_script(
    decel_mps2: float, decel_start: float, decel_duration: float, reaccel_mps2: float, reaccel_start: float, reaccel_duration: float
) -> LeadScript:
    def script(t: float, vehicle: Vehicle) -> tuple[float, float]:
        if decel_start <= t < decel_start + decel_duration and vehicle.speed > 0.0:
            return -abs(decel_mps2), 0.0
        if reaccel_start <= t < reaccel_start + reaccel_duration:
            return abs(reaccel_mps2), 0.0
        return 0.0, 0.0

    return script


def oscillating_script(amplitude_mps2: float, period_s: float) -> LeadScript:
    def script(t: float, vehicle: Vehicle) -> tuple[float, float]:
        return amplitude_mps2 * math.sin(2 * math.pi * t / period_s), 0.0

    return script


def cutin_script(start_t: float, target_lane_y: float = 0.0, gain_y: float = 0.6, gain_heading: float = 1.2) -> LeadScript:
    def script(t: float, vehicle: Vehicle) -> tuple[float, float]:
        if t < start_t:
            return 0.0, 0.0
        lateral_error = vehicle.position[1] - target_lane_y
        heading_error = vehicle.heading
        steering = _clip(-gain_y * lateral_error - gain_heading * heading_error, -0.5, 0.5)
        return 0.0, steering

    return script


@dataclass
class AccScenarioSpec:
    scenario_id: str
    description: str
    initial_ego_speed: float
    initial_lead_speed: float
    initial_gap: float
    lead_script: LeadScript
    lead_lane_id: int = 0
    duration_s: float = 20.0
    policy_frequency: int = 5
    pass_criteria: dict = field(default_factory=lambda: dict(DEFAULT_PASS_CRITERIA[Feature.ACC]))


def _make_env(policy_frequency: int, duration_s: float):
    config = {
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 2,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": True,
            "normalize": False,
            "see_behind": False,
            "order": "sorted",
        },
        "action": {
            "type": "ContinuousAction",
            "longitudinal": True,
            "lateral": False,
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


def run_acc_scenario(spec: AccScenarioSpec, seed: int = 0) -> TestResult:
    env = _make_env(spec.policy_frequency, spec.duration_s)
    env.reset(seed=seed)

    ego = env.unwrapped.controlled_vehicles[0]
    ego.position = np.array([0.0, 0.0])
    ego.heading = 0.0
    ego.speed = spec.initial_ego_speed

    lane_y = spec.lead_lane_id * LANE_WIDTH
    # Offset by one vehicle length so spec.initial_gap matches the bumper-to-bumper
    # gap the controller and metrics actually see at t=0 (see gap computation below).
    lead = Vehicle(
        env.unwrapped.road,
        position=np.array([spec.initial_gap + Vehicle.LENGTH, lane_y]),
        heading=0.0,
        speed=spec.initial_lead_speed,
    )
    env.unwrapped.road.vehicles.append(lead)

    dt = 1.0 / spec.policy_frequency
    sample_times: list[float] = []
    ttc_series: list[float] = []
    gap_series: list[float] = []
    gap_error_series: list[float] = []
    ego_speed_series: list[float] = []
    lead_speed_series: list[float] = []
    min_ttc = math.inf
    peak_gap_error = 0.0
    t = 0.0
    steps = int(spec.duration_s * spec.policy_frequency)

    for _ in range(steps):
        ego_speed = float(ego.speed)
        lead_speed = float(lead.speed)
        gap = float(lead.position[0] - ego.position[0]) - Vehicle.LENGTH

        accel_cmd = acc_controller(ego_speed=ego_speed, lead_speed=lead_speed, gap=gap)
        ttc = time_to_collision(ego_speed=ego_speed, lead_speed=lead_speed, gap_distance=gap)
        target_gap = 2.0 + 1.5 * ego_speed
        gap_err = following_distance_error(actual_gap=gap, target_gap=target_gap)

        sample_times.append(t)
        ttc_series.append(ttc if math.isfinite(ttc) else TTC_CAP_S)
        gap_series.append(gap)
        gap_error_series.append(gap_err)
        ego_speed_series.append(ego_speed)
        lead_speed_series.append(lead_speed)
        min_ttc = min(min_ttc, ttc)
        peak_gap_error = max(peak_gap_error, abs(gap_err))

        action_value = utils.lmap(accel_cmd, list(ACCELERATION_RANGE), [-1.0, 1.0])
        action = np.array([_clip(action_value, -1.0, 1.0)], dtype=np.float32)

        lead_accel, lead_steer = spec.lead_script(t, lead)
        lead.action = {"acceleration": lead_accel, "steering": lead_steer}

        _, _, terminated, truncated, _ = env.step(action)
        t += dt
        if terminated or truncated:
            break

    env.close()

    # Settled-state tracking error: average |error| over the last 20% of samples.
    # Transient error right after a scripted disturbance (decel/cut-in/etc.) is
    # expected and isn't itself a defect; what matters is whether ACC converges.
    settle_window = max(1, len(gap_error_series) // 5)
    settled_errors = gap_error_series[-settle_window:] if gap_error_series else [0.0]
    final_gap_error = sum(abs(e) for e in settled_errors) / len(settled_errors)

    passed = (
        min_ttc >= spec.pass_criteria.get("min_time_to_collision_s", 2.0)
        and final_gap_error <= spec.pass_criteria.get("max_following_distance_error_m", 1.0)
    )

    procedure = TestProcedure(
        feature=Feature.ACC,
        scenario_id=spec.scenario_id,
        description=spec.description,
        preconditions={
            "initial_ego_speed_mps": spec.initial_ego_speed,
            "initial_lead_speed_mps": spec.initial_lead_speed,
            "initial_gap_m": spec.initial_gap,
            "lead_lane_id": spec.lead_lane_id,
        },
        pass_criteria=spec.pass_criteria,
        data_source=DataSource.SIMULATION,
    )

    return TestResult(
        procedure=procedure,
        passed=passed,
        observed_metrics={
            "min_ttc_s": min_ttc if math.isfinite(min_ttc) else TTC_CAP_S,
            "final_following_distance_error_m": final_gap_error,
            "peak_following_distance_error_m": peak_gap_error,
        },
        time_series={
            "ttc_s": ttc_series,
            "gap_m": gap_series,
            "ego_speed_mps": ego_speed_series,
            "lead_speed_mps": lead_speed_series,
        },
        sample_times=sample_times,
    )


def build_acc_scenarios() -> list[AccScenarioSpec]:
    return [
        AccScenarioSpec(
            scenario_id="acc_steady_matched_speed",
            description="Lead holds a constant speed matching ego; gap should stay stable.",
            initial_ego_speed=25.0,
            initial_lead_speed=25.0,
            initial_gap=39.5,
            lead_script=steady_script(),
        ),
        AccScenarioSpec(
            scenario_id="acc_steady_faster_lead",
            description="Lead is steady and faster than ego; ego should speed up without overshooting.",
            initial_ego_speed=22.0,
            initial_lead_speed=28.0,
            initial_gap=35.0,
            lead_script=steady_script(),
        ),
        AccScenarioSpec(
            scenario_id="acc_steady_slower_lead",
            description="Lead is steady and slower than ego; ego must slow to match without collision.",
            initial_ego_speed=25.0,
            initial_lead_speed=18.0,
            initial_gap=35.0,
            lead_script=steady_script(),
        ),
        AccScenarioSpec(
            scenario_id="acc_moderate_decel",
            description="Lead decelerates moderately (2 m/s^2) from steady-state car-following.",
            initial_ego_speed=25.0,
            initial_lead_speed=25.0,
            initial_gap=39.5,
            lead_script=decel_script(decel_mps2=2.0, start_t=5.0, duration=3.0),
        ),
        AccScenarioSpec(
            scenario_id="acc_hard_brake_exceeds_authority",
            description=(
                "Lead brakes hard (6 m/s^2), exceeding ACC's -4 m/s^2 authority. "
                "Expected to fail ACC criteria; this gap is what AEB exists to cover."
            ),
            initial_ego_speed=25.0,
            initial_lead_speed=25.0,
            initial_gap=39.5,
            lead_script=decel_script(decel_mps2=6.0, start_t=5.0, duration=4.0),
        ),
        AccScenarioSpec(
            scenario_id="acc_lead_accelerates_away",
            description="Lead accelerates away from steady-state; gap should grow smoothly.",
            initial_ego_speed=22.0,
            initial_lead_speed=22.0,
            initial_gap=35.0,
            lead_script=accel_script(accel_mps2=1.5, start_t=5.0, duration=5.0),
        ),
        AccScenarioSpec(
            scenario_id="acc_lead_stop_and_go",
            description="Lead decelerates to a stop, holds, then re-accelerates.",
            initial_ego_speed=20.0,
            initial_lead_speed=20.0,
            initial_gap=32.0,
            lead_script=stop_and_go_script(
                decel_mps2=3.0, decel_start=5.0, decel_duration=7.0,
                reaccel_mps2=2.0, reaccel_start=15.0, reaccel_duration=5.0,
            ),
            duration_s=22.0,
        ),
        AccScenarioSpec(
            scenario_id="acc_lead_speed_oscillation",
            description="Lead speed oscillates sinusoidally; tests smoothness of the gap response.",
            initial_ego_speed=22.0,
            initial_lead_speed=22.0,
            initial_gap=35.0,
            lead_script=oscillating_script(amplitude_mps2=1.5, period_s=6.0),
        ),
        AccScenarioSpec(
            scenario_id="acc_cutin_moderate_gap",
            description="A vehicle in the adjacent lane merges into ego's lane with a comfortable gap.",
            initial_ego_speed=25.0,
            initial_lead_speed=25.0,
            initial_gap=30.0,
            lead_lane_id=1,
            lead_script=cutin_script(start_t=5.0),
        ),
        AccScenarioSpec(
            scenario_id="acc_cutin_close_gap",
            description="A vehicle in the adjacent lane merges into ego's lane at close range (stress test).",
            initial_ego_speed=25.0,
            initial_lead_speed=23.0,
            initial_gap=15.0,
            lead_lane_id=1,
            lead_script=cutin_script(start_t=3.0),
        ),
    ]


def run_all(output_dir: str = "reports", defect_log_path: str = "reports/defect_log.csv") -> Path:
    defect_log = DefectLog(defect_log_path)
    results = [run_acc_scenario(spec) for spec in build_acc_scenarios()]

    for result in results:
        if result.passed:
            continue
        min_ttc = result.observed_metrics.get("min_ttc_s", TTC_CAP_S)
        severity = Severity.HIGH if min_ttc < 1.0 else Severity.MEDIUM
        defect_log.log(
            Defect(
                feature=Feature.ACC,
                scenario_id=result.procedure.scenario_id,
                expected=f"pass_criteria satisfied: {result.procedure.pass_criteria}",
                observed=f"observed_metrics: {result.observed_metrics}",
                severity=severity,
                reproduction_steps=(
                    "run_acc_scenario(spec) for spec.scenario_id == "
                    f"'{result.procedure.scenario_id}' in scenarios/acc_scenarios.py"
                ),
                data_source=DataSource.SIMULATION,
            )
        )

    builder = ReportBuilder(output_dir=output_dir)
    return builder.build(results, defect_log=defect_log)


if __name__ == "__main__":
    report_path = run_all()
    print(f"Report written to {report_path}")
