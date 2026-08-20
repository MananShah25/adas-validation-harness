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
from ingestion.obd_reader import OBDCommand, OBDReader, ReplayExhausted
from metrics.ttc_lane_brake import time_to_collision
from reporting.defect_log import Defect, DefectLog, Severity
from reporting.report_builder import ReportBuilder
from schemas.test_procedure import DEFAULT_PASS_CRITERIA, DataSource, Feature, TestProcedure, TestResult
from vision.frames import FrameSequence

# Callable[[elapsed_time_s], gap_m_or_None]. In production this is a
# synchronized video frame's vision.detector output; here, until real
# dashcam footage is available, a scripted synthetic profile.
@dataclass
class GapReading:
    """A gap measurement and the time it was actually taken.

    `measured_at` matters whenever the gap sensor updates slower than the
    fusion loop samples it. A 5 fps camera queried at 10 Hz returns the same
    frame twice; differencing by *query* time then attributes a two-frame
    change to a one-step interval and doubles the apparent closing rate,
    which feeds straight into TTC. Carrying the measurement time makes the
    rate estimate independent of how often it happens to be asked.
    """

    distance_m: float
    measured_at: float


# A GapSource may return a bare float (measurement time == query time, which
# is right for an analytic profile) or a GapReading (for a real sensor whose
# samples land on their own clock). None means no reading available.
GapSource = Callable[[float], "float | GapReading | None"]

TTC_CAP_S = 99.0


def _kph_to_mps(kph: float) -> float:
    return kph / 3.6


def _normalize_reading(raw, query_time: float) -> "GapReading | None":
    if raw is None:
        return None
    if isinstance(raw, GapReading):
        return raw
    return GapReading(distance_m=float(raw), measured_at=query_time)


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


class DetectorGapSource:
    """The real vision path: a VehicleDetector over a timed FrameSequence,
    adapted to the GapSource contract the fusion pipeline consumes.

    This is the adapter that makes Track 2 genuinely end to end -- ego speed
    from OBD-II, gap distance from actual YOLO inference on actual frames --
    rather than two halves proven only in isolation.

    Two behaviours worth knowing:

    * Detections are cached per frame index. Fusion usually samples faster
      than the clip's frame rate (10 Hz against, say, 5 fps), and running
      the detector repeatedly on a frame already processed would cost real
      time for an identical answer.
    * A frame with no vehicle in it, and any time past the end of the clip,
      both return None -- not a stale distance. The pipeline treats None as
      "no reading", so a dropout correctly suspends the TTC calculation
      instead of silently extrapolating from an old frame.
    """

    def __init__(self, detector, frames: FrameSequence):
        self.detector = detector
        self.frames = frames
        self._cache: dict[int, float | None] = {}
        self.frames_processed = 0
        self.frames_without_detection = 0

    def __call__(self, t: float) -> GapReading | None:
        index = self.frames.index_at(t)
        if index is None:
            return None  # past the end of the clip

        if index in self._cache:
            distance = self._cache[index]
        else:
            lead = self.detector.lead_vehicle(self.frames.frames[index])
            distance = lead.estimated_distance_m if lead is not None else None
            self.frames_processed += 1
            if distance is None:
                self.frames_without_detection += 1
            self._cache[index] = distance

        if distance is None:
            return None  # frame contained no vehicle
        # Stamped with the frame's own time, not the query time, so the
        # pipeline derives the closing rate over the real interval between
        # frames however often it samples us.
        return GapReading(distance_m=distance, measured_at=index / self.frames.fps)

    @property
    def detection_rate(self) -> float:
        """Fraction of processed frames that yielded a lead vehicle."""
        if self.frames_processed == 0:
            return 0.0
        return 1.0 - (self.frames_without_detection / self.frames_processed)


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
        # Tracked at the last *change* in the gap reading, not at the last
        # step. A camera updating slower than the control loop repeats a
        # reading across consecutive steps; differencing per-step would
        # alternate between a rate of 0 and double the true rate, which
        # aliases straight into TTC. Holding the last real measurement
        # interval is what a rate estimator on real sensor data has to do.
        self._last_gap: float | None = None
        self._last_timestamp: float | None = None
        self._closing_rate_mps: float | None = None

    def step(self) -> FusionSampleResult:
        sample = self.obd_reader.next_sample()
        timestamp = sample[OBDCommand.SPEED].timestamp
        ego_speed_mps = _kph_to_mps(sample[OBDCommand.SPEED].value)
        reading = _normalize_reading(self.gap_source(timestamp), timestamp)
        gap = reading.distance_m if reading is not None else None

        if reading is None:
            # No reading (dropout, or past the end of the clip). Discard the
            # rate estimate rather than carrying it across the gap.
            self._last_gap = None
            self._last_timestamp = None
            self._closing_rate_mps = None
        elif self._last_gap is None or self._last_timestamp is None:
            self._last_gap = reading.distance_m
            self._last_timestamp = reading.measured_at
        elif reading.measured_at != self._last_timestamp:
            # A genuinely new measurement. Keyed on measurement time rather
            # than on the value, since two consecutive frames can legitimately
            # produce an identical distance.
            dt = reading.measured_at - self._last_timestamp
            if dt > 0:
                self._closing_rate_mps = (self._last_gap - reading.distance_m) / dt
            self._last_gap = reading.distance_m
            self._last_timestamp = reading.measured_at

        ttc = math.inf
        if gap is not None and self._closing_rate_mps is not None:
            lead_speed_mps = ego_speed_mps - self._closing_rate_mps
            ttc = time_to_collision(ego_speed=ego_speed_mps, lead_speed=lead_speed_mps, gap_distance=gap)

        aeb_state = aeb_controller(
            ttc,
            warning_ttc=self.pass_criteria.get("warning_time_to_collision_s", 2.0),
            brake_ttc=self.pass_criteria.get("brake_time_to_collision_s", 1.8),
        )

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
    # Stays SIMULATION while any input is synthesized. A run is only
    # REAL_WORLD once BOTH sides are real: OBDReader(mode="live") on a real
    # drive and a VideoFrameSequence over footage recorded on that same
    # drive. Replayed OBD-II paired with a synthesized clip is not a real
    # run, and labelling it one would put unearned weight on the results.
    data_source: DataSource = DataSource.SIMULATION
    obd_reader_factory: "Callable[[], OBDReader] | None" = None
    gap_source_label: str = "synthetic (scripted closing profile)"


def run_fusion_scenario(spec: FusionScenarioSpec) -> TestResult:
    if spec.obd_reader_factory is not None:
        reader = spec.obd_reader_factory()
    else:
        reader = OBDReader(
            mode="mock", seed=spec.obd_seed, hz=spec.obd_hz, initial_speed_kph=spec.initial_ego_speed_kph
        )
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
        try:
            result = pipeline.step()
        except ReplayExhausted:
            break  # a finite recording simply ended; score what was observed
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
            "ego_speed_source": f"OBDReader(mode='{reader.mode}')",
            "gap_source": spec.gap_source_label,
        },
        pass_criteria=spec.pass_criteria,
        data_source=spec.data_source,
    )

    # Count AEB state changes. A clean event goes normal -> warning ->
    # full_brake and stays there; a high count means the decision is
    # chattering, which on real detector output signals a rate estimate too
    # noisy to act on. Worth measuring rather than eyeballing.
    state_transitions = sum(
        1 for a, b in zip(brake_state_series, brake_state_series[1:]) if a != b
    )

    observed = {
        "min_ttc_s": min_ttc if math.isfinite(min_ttc) else TTC_CAP_S,
        "triggered": float(brake_onset_time is not None),
        "brake_onset_time_s": brake_onset_time if brake_onset_time is not None else -1.0,
        "aeb_state_transitions": float(state_transitions),
    }
    # A vision-driven run reports how much of the clip actually produced a
    # detection. A gap metric computed from frames that mostly saw nothing
    # would otherwise look identical to one computed from a clean track.
    if isinstance(spec.gap_source, DetectorGapSource):
        observed["vision_detection_rate"] = spec.gap_source.detection_rate
        observed["frames_detected_on"] = float(spec.gap_source.frames_processed)

    return TestResult(
        procedure=procedure,
        passed=passed,
        observed_metrics=observed,
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


def build_vision_fusion_scenario(
    clip_image_path: str | Path | None = None,
    detector=None,
    frames: FrameSequence | None = None,
    obd_hz: float = 10.0,
) -> FusionScenarioSpec:
    """A fusion scenario whose gap distance comes from real YOLO inference.

    This is the end-to-end path: OBD-II ego speed on one side, an actual
    VehicleDetector reading actual frames on the other, meeting at the same
    time_to_collision() and aeb_controller() Track 1 uses.

    With no dashcam footage available, the default clip is a synthesized
    approach built from a real photograph (see vision/synthetic_clip.py for
    exactly what that does and does not establish). Pass `frames` --
    typically a VideoFrameSequence over a real clip -- to run it on real
    footage instead; nothing else here changes.

    Constructing the detector is deferred to call time because loading YOLO
    weights is slow and shouldn't happen just from importing this module.
    """
    from vision.detector import VehicleDetector
    from vision.synthetic_clip import make_approach_sequence

    if frames is None:
        if clip_image_path is None:
            import ultralytics

            clip_image_path = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
        frames = make_approach_sequence(clip_image_path, fps=5.0, n_frames=24)
        label = "vision (synthesized approach clip, real YOLOv8n inference)"
    else:
        label = "vision (supplied frame sequence, real YOLOv8n inference)"

    detector = detector if detector is not None else VehicleDetector()

    return FusionScenarioSpec(
        scenario_id="fusion_vision_closing_hazard",
        description=(
            "End-to-end vision path: gap distance comes from real YOLOv8n detections on a "
            "closing sequence, ego speed from the OBD-II stream. Should trigger AEB as the "
            "detected vehicle fills more of the frame and the estimated gap collapses."
        ),
        gap_source=DetectorGapSource(detector, frames),
        gap_source_label=label,
        expect_trigger=True,
        initial_ego_speed_kph=45.0,
        duration_s=frames.duration_s,
        obd_hz=obd_hz,
    )


def run_all(
    output_dir: str = "reports",
    defect_log_path: str = "reports/fusion_defect_log.csv",
    emit_pdf: bool = False,
) -> Path:
    defect_log = DefectLog(defect_log_path)
    results = [run_fusion_scenario(spec) for spec in build_fusion_scenarios()]
    results.append(run_fusion_scenario(build_vision_fusion_scenario()))

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
    if emit_pdf:
        builder.build_pdf(results, defect_log=defect_log)
    return builder.build(results, defect_log=defect_log)


if __name__ == "__main__":
    report_path = run_all()
    print(f"Report written to {report_path}")
