"""AEB scenarios: acc_controller drives normally until aeb_controller's TTC
monitor overrides to full braking authority, independent of ACC.

Reuses the deterministic lead-vehicle scripting approach and environment
setup from scenarios/acc_scenarios.py, but with a wider acceleration range
(AEB needs more braking authority than ACC's own -4 m/s^2 self-limit) and a
finer policy frequency (brake response latency needs sub-0.5s resolution).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium
import highway_env  # noqa: F401  (registers the highway-v0 env on import)
import numpy as np
from highway_env import utils
from highway_env.vehicle.kinematics import Vehicle

from controllers.acc import acc_controller
from controllers.aeb import aeb_controller
from metrics.ttc_lane_brake import brake_response_latency, time_to_collision
from reporting.defect_log import Defect, DefectLog, Severity
from reporting.report_builder import ReportBuilder
from scenarios.acc_scenarios import LANE_WIDTH, LeadScript, decel_script, steady_script, stop_and_go_script
from schemas.test_procedure import DEFAULT_PASS_CRITERIA, DataSource, Feature, TestProcedure, TestResult

AEB_ACCELERATION_RANGE = (-9.0, 3.0)  # wider than ACC's own +-5 default: full AEB authority
FULL_BRAKE_ACCEL = -9.0
TTC_CAP_S = 99.0


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class AebScenarioSpec:
    scenario_id: str
    description: str
    initial_ego_speed: float
    initial_lead_speed: float
    initial_gap: float
    lead_script: LeadScript
    lead_event_time: float  # when the lead's braking begins, used as the latency reference event
    expect_trigger: bool = True
    lead_lane_id: int = 0
    duration_s: float = 15.0
    policy_frequency: int = 10
    pass_criteria: dict = field(default_factory=lambda: dict(DEFAULT_PASS_CRITERIA[Feature.AEB]))


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
            "acceleration_range": list(AEB_ACCELERATION_RANGE),
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


def run_aeb_scenario(spec: AebScenarioSpec, seed: int = 0) -> TestResult:
    env = _make_env(spec.policy_frequency, spec.duration_s)
    env.reset(seed=seed)

    ego = env.unwrapped.controlled_vehicles[0]
    ego.position = np.array([0.0, 0.0])
    ego.heading = 0.0
    ego.speed = spec.initial_ego_speed

    lane_y = spec.lead_lane_id * LANE_WIDTH
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
    ego_speed_series: list[float] = []
    brake_state_series: list[int] = []  # 0=normal, 1=warning, 2=full_brake
    min_ttc = math.inf
    brake_onset_time: float | None = None
    collided = False
    t = 0.0
    steps = int(spec.duration_s * spec.policy_frequency)
    state_to_int = {"normal": 0, "warning": 1, "full_brake": 2}

    for _ in range(steps):
        ego_speed = float(ego.speed)
        lead_speed = float(lead.speed)
        gap = float(lead.position[0] - ego.position[0]) - Vehicle.LENGTH

        ttc = time_to_collision(ego_speed=ego_speed, lead_speed=lead_speed, gap_distance=gap)
        state = aeb_controller(
            ttc,
            warning_ttc=spec.pass_criteria.get("warning_time_to_collision_s", 2.0),
            brake_ttc=spec.pass_criteria.get("brake_time_to_collision_s", 1.8),
        )
        if state == "full_brake":
            accel_cmd = FULL_BRAKE_ACCEL
            if brake_onset_time is None:
                brake_onset_time = t
        else:
            accel_cmd = acc_controller(ego_speed=ego_speed, lead_speed=lead_speed, gap=gap)

        sample_times.append(t)
        ttc_series.append(ttc if math.isfinite(ttc) else TTC_CAP_S)
        gap_series.append(gap)
        ego_speed_series.append(ego_speed)
        brake_state_series.append(state_to_int[state])
        min_ttc = min(min_ttc, ttc)

        action_value = utils.lmap(accel_cmd, list(AEB_ACCELERATION_RANGE), [-1.0, 1.0])
        action = np.array([_clip(action_value, -1.0, 1.0)], dtype=np.float32)

        lead_accel, lead_steer = spec.lead_script(t, lead)
        lead.action = {"acceleration": lead_accel, "steering": lead_steer}

        _, _, terminated, truncated, _ = env.step(action)
        if ego.crashed:
            collided = True
        t += dt
        if terminated or truncated:
            break

    env.close()

    brake_latency = None
    if brake_onset_time is not None:
        brake_latency = brake_response_latency(
            event_timestamp=spec.lead_event_time, brake_onset_timestamp=brake_onset_time
        )

    # Pass/fail is safety-outcome-based: never collide, and correctly decide
    # whether to intervene. brake_response_latency_s is still computed above
    # and reported in observed_metrics as a diagnostic (see schemas/
    # test_procedure.py for why it isn't gated against a fixed threshold).
    if spec.expect_trigger:
        passed = not collided and brake_onset_time is not None
    else:
        passed = not collided and brake_onset_time is None

    procedure = TestProcedure(
        feature=Feature.AEB,
        scenario_id=spec.scenario_id,
        description=spec.description,
        preconditions={
            "initial_ego_speed_mps": spec.initial_ego_speed,
            "initial_lead_speed_mps": spec.initial_lead_speed,
            "initial_gap_m": spec.initial_gap,
            "lead_event_time_s": spec.lead_event_time,
            "expect_trigger": spec.expect_trigger,
        },
        pass_criteria=spec.pass_criteria,
        data_source=DataSource.SIMULATION,
    )

    return TestResult(
        procedure=procedure,
        passed=passed,
        observed_metrics={
            "min_ttc_s": min_ttc if math.isfinite(min_ttc) else TTC_CAP_S,
            "brake_response_latency_s": brake_latency if brake_latency is not None else -1.0,
            "collided": float(collided),
        },
        time_series={
            "ttc_s": ttc_series,
            "gap_m": gap_series,
            "ego_speed_mps": ego_speed_series,
            "brake_state": [float(s) for s in brake_state_series],
        },
        sample_times=sample_times,
    )


def build_aeb_scenarios() -> list[AebScenarioSpec]:
    return [
        AebScenarioSpec(
            scenario_id="aeb_lead_hard_brake_cruise_speed",
            description="Lead brakes hard (-7 m/s^2) from steady cruise at 25 m/s.",
            initial_ego_speed=25.0,
            initial_lead_speed=25.0,
            initial_gap=39.5,
            lead_script=decel_script(decel_mps2=7.0, start_t=3.0, duration=4.0),
            lead_event_time=3.0,
        ),
        AebScenarioSpec(
            scenario_id="aeb_lead_sudden_stop",
            description="Lead comes to a near-instant full stop (-9 m/s^2 burst).",
            initial_ego_speed=22.0,
            initial_lead_speed=22.0,
            initial_gap=35.0,
            lead_script=decel_script(decel_mps2=9.0, start_t=3.0, duration=3.0),
            lead_event_time=3.0,
        ),
        AebScenarioSpec(
            scenario_id="aeb_lead_hard_brake_high_speed",
            description="Highway-speed (30 m/s) hard brake event.",
            initial_ego_speed=30.0,
            initial_lead_speed=30.0,
            initial_gap=47.0,
            lead_script=decel_script(decel_mps2=7.0, start_t=3.0, duration=4.5),
            lead_event_time=3.0,
        ),
        AebScenarioSpec(
            scenario_id="aeb_lead_hard_brake_low_speed",
            description="Urban-speed (15 m/s) hard brake event.",
            initial_ego_speed=15.0,
            initial_lead_speed=15.0,
            initial_gap=24.5,
            lead_script=decel_script(decel_mps2=6.0, start_t=3.0, duration=3.0),
            lead_event_time=3.0,
        ),
        AebScenarioSpec(
            scenario_id="aeb_covers_acc_authority_gap",
            description=(
                "Same conditions AND control-loop frequency as Phase 2's "
                "acc_hard_brake_exceeds_authority (which crashed under ACC alone): "
                "lead brakes at -6 m/s^2, beyond ACC's -4 m/s^2 authority. Isolates "
                "whether AEB's wider braking authority alone covers the gap, holding "
                "reaction frequency fixed so it isn't a confound."
            ),
            initial_ego_speed=25.0,
            initial_lead_speed=25.0,
            initial_gap=39.5,
            lead_script=decel_script(decel_mps2=6.0, start_t=5.0, duration=4.0),
            lead_event_time=5.0,
            policy_frequency=5,
        ),
        AebScenarioSpec(
            scenario_id="aeb_lead_moderate_brake_within_acc_authority",
            description="Moderate brake (-3 m/s^2), within ACC's own -4 m/s^2 authority.",
            initial_ego_speed=25.0,
            initial_lead_speed=25.0,
            initial_gap=39.5,
            lead_script=decel_script(decel_mps2=3.0, start_t=3.0, duration=5.0),
            lead_event_time=3.0,
            expect_trigger=False,
        ),
        AebScenarioSpec(
            scenario_id="aeb_repeated_braking_events",
            description="Lead brakes hard, recovers, then brakes hard again; latency measured on the first event.",
            initial_ego_speed=22.0,
            initial_lead_speed=22.0,
            initial_gap=35.0,
            lead_script=stop_and_go_script(
                decel_mps2=7.5, decel_start=3.0, decel_duration=5.0,
                reaccel_mps2=1.5, reaccel_start=9.0, reaccel_duration=3.0,
            ),
            lead_event_time=3.0,
            duration_s=16.0,
        ),
        AebScenarioSpec(
            scenario_id="aeb_no_nuisance_trigger_steady_cruise",
            description="Lead holds steady speed and gap; AEB must not intervene unnecessarily.",
            initial_ego_speed=25.0,
            initial_lead_speed=25.0,
            initial_gap=39.5,
            lead_script=steady_script(),
            lead_event_time=0.0,
            expect_trigger=False,
        ),
        AebScenarioSpec(
            scenario_id="aeb_already_critical_at_start",
            description=(
                "Lead already close and much slower when the scenario begins "
                "(simulating stopped traffic just revealed), probing whether even "
                "AEB's full authority can recover from an already-critical initial gap."
            ),
            initial_ego_speed=25.0,
            initial_lead_speed=5.0,
            initial_gap=10.0,
            lead_script=steady_script(),
            lead_event_time=0.0,
            duration_s=8.0,
        ),
    ]


def run_all(output_dir: str = "reports", defect_log_path: str = "reports/aeb_defect_log.csv") -> Path:
    defect_log = DefectLog(defect_log_path)
    results = [run_aeb_scenario(spec) for spec in build_aeb_scenarios()]

    for result in results:
        if result.passed:
            continue
        collided = bool(result.observed_metrics.get("collided", 0.0))
        severity = Severity.CRITICAL if collided else Severity.HIGH
        defect_log.log(
            Defect(
                feature=Feature.AEB,
                scenario_id=result.procedure.scenario_id,
                expected=f"pass_criteria satisfied: {result.procedure.pass_criteria}",
                observed=f"observed_metrics: {result.observed_metrics}",
                severity=severity,
                reproduction_steps=(
                    "run_aeb_scenario(spec) for spec.scenario_id == "
                    f"'{result.procedure.scenario_id}' in scenarios/aeb_scenarios.py"
                ),
                data_source=DataSource.SIMULATION,
            )
        )

    builder = ReportBuilder(output_dir=output_dir)
    return builder.build(results, defect_log=defect_log)


if __name__ == "__main__":
    report_path = run_all()
    print(f"Report written to {report_path}")
