"""Tests for the vision -> fusion path: the DetectorGapSource adapter, the
measurement-time-based closing-rate derivation, and one genuinely end-to-end
run with real YOLO inference driving the AEB decision.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import ultralytics

from fusion.aeb_pipeline import (
    AebFusionPipeline,
    DetectorGapSource,
    GapReading,
    build_vision_fusion_scenario,
    run_fusion_scenario,
)
from ingestion.obd_reader import OBDReader
from schemas.test_procedure import DataSource, Feature
from vision.frames import FrameSequence

BUS_JPG = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"


class StubDetection:
    def __init__(self, distance):
        self.estimated_distance_m = distance


class StubDetector:
    """Returns a scripted distance per frame, so adapter behaviour can be
    tested without paying for YOLO inference."""

    def __init__(self, distances):
        self.distances = distances
        self.calls = 0

    def lead_vehicle(self, frame):
        self.calls += 1
        index = int(frame[0, 0, 0])
        distance = self.distances[index]
        return None if distance is None else StubDetection(distance)


def _frames(n):
    return FrameSequence([np.full((4, 4, 3), i, dtype=np.uint8) for i in range(n)], fps=5.0)


# ---------- DetectorGapSource ----------

def test_adapter_returns_reading_stamped_with_frame_time():
    source = DetectorGapSource(StubDetector([30.0, 25.0, 20.0]), _frames(3))
    reading = source(0.0)
    assert isinstance(reading, GapReading)
    assert reading.distance_m == pytest.approx(30.0)
    assert reading.measured_at == pytest.approx(0.0)

    # queried mid-frame: the stamp is the FRAME's time, not the query time
    reading = source(0.25)
    assert reading.distance_m == pytest.approx(25.0)
    assert reading.measured_at == pytest.approx(0.2)


def test_adapter_caches_one_inference_per_frame():
    detector = StubDetector([30.0, 25.0, 20.0])
    source = DetectorGapSource(detector, _frames(3))
    for t in (0.0, 0.05, 0.1, 0.15, 0.19):
        source(t)
    assert detector.calls == 1, "repeated queries within one frame must not re-run the detector"
    source(0.2)
    assert detector.calls == 2


def test_adapter_returns_none_past_end_of_clip():
    source = DetectorGapSource(StubDetector([30.0, 25.0]), _frames(2))
    assert source(0.4) is None
    assert source(99.0) is None


def test_adapter_returns_none_when_no_vehicle_detected():
    source = DetectorGapSource(StubDetector([30.0, None, 20.0]), _frames(3))
    assert source(0.0) is not None
    assert source(0.2) is None
    assert source(0.4) is not None


def test_adapter_tracks_detection_rate():
    source = DetectorGapSource(StubDetector([30.0, None, 20.0, None]), _frames(4))
    for i in range(4):
        source(i / 5.0)
    assert source.frames_processed == 4
    assert source.frames_without_detection == 2
    assert source.detection_rate == pytest.approx(0.5)


def test_detection_rate_is_zero_before_any_frame_processed():
    source = DetectorGapSource(StubDetector([30.0]), _frames(1))
    assert source.detection_rate == 0.0


# ---------- closing-rate derivation ----------

def test_closing_rate_uses_measurement_time_not_query_rate():
    """A 5 fps gap source queried at 10 Hz must not double the closing rate.

    Differencing by query time would attribute a two-frame change to a
    one-step interval; the reading's own timestamp prevents that.
    """
    true_rate = 6.0

    def gap_source(t):
        index = math.floor(t * 5)
        measured_at = index / 5.0
        return GapReading(distance_m=max(0.0, 40.0 - true_rate * measured_at), measured_at=measured_at)

    reader = OBDReader(mode="mock", seed=0, hz=10.0, initial_speed_kph=50.0)
    pipeline = AebFusionPipeline(reader, gap_source)

    checked = 0
    for _ in range(12):
        result = pipeline.step()
        if math.isinf(result.ttc_s):
            continue  # first sample, no rate yet
        assert result.ttc_s == pytest.approx(result.gap_m / true_rate, rel=1e-6)
        checked += 1
    assert checked >= 8


def test_ttc_is_held_across_repeated_readings_not_reset_to_infinity():
    def gap_source(t):
        index = math.floor(t * 5)
        return GapReading(distance_m=40.0 - 6.0 * (index / 5.0), measured_at=index / 5.0)

    pipeline = AebFusionPipeline(OBDReader(mode="mock", seed=0, hz=10.0, initial_speed_kph=50.0), gap_source)
    ttcs = [pipeline.step().ttc_s for _ in range(8)]
    settled = ttcs[2:]
    assert all(math.isfinite(v) for v in settled), f"TTC dropped to inf on a repeated frame: {ttcs}"


def test_dropout_discards_stale_rate_estimate():
    """After a frame with no detection, TTC must not resume from a stale rate.

    A steady closing profile with one hole punched in it: TTC should be
    finite once primed, go infinite at the dropout, and stay infinite on the
    very next reading (re-priming) rather than reusing the pre-dropout rate.
    """
    dropout_index = 4

    def gap_source(t):
        index = math.floor(t * 5)
        if index == dropout_index:
            return None
        return GapReading(distance_m=max(0.0, 40.0 - 6.0 * (index / 5.0)), measured_at=index / 5.0)

    pipeline = AebFusionPipeline(OBDReader(mode="mock", seed=0, hz=5.0, initial_speed_kph=50.0), gap_source)
    results = [pipeline.step() for _ in range(8)]

    dropouts = [i for i, r in enumerate(results) if r.gap_m is None]
    assert dropouts, "the scripted dropout should appear in the run"
    d = dropouts[0]

    assert any(math.isfinite(r.ttc_s) for r in results[:d]), "should have primed before the dropout"
    assert math.isinf(results[d].ttc_s), "no reading means no TTC"
    assert math.isinf(results[d + 1].ttc_s), "must re-prime, not reuse the pre-dropout rate"
    assert math.isfinite(results[d + 2].ttc_s), "should recover once two readings are available again"


def test_plain_float_gap_source_still_supported():
    """Analytic sources returning a bare float keep working -- measurement
    time is simply the query time."""
    pipeline = AebFusionPipeline(
        OBDReader(mode="mock", seed=0, hz=10.0, initial_speed_kph=50.0),
        lambda t: max(0.0, 40.0 - 6.0 * t),
    )
    pipeline.step()
    result = pipeline.step()
    assert result.ttc_s == pytest.approx(result.gap_m / 6.0, rel=1e-6)


# ---------- end to end, real inference ----------

@pytest.mark.skipif(not BUS_JPG.exists(), reason="bundled ultralytics sample image not found")
def test_end_to_end_vision_fusion_run():
    """The whole Track 2 path with nothing stubbed: real YOLOv8n inference on
    a closing sequence, real OBD-II ego speed, real shared TTC and AEB logic."""
    spec = build_vision_fusion_scenario()
    result = run_fusion_scenario(spec)

    assert result.procedure.feature is Feature.AEB
    assert result.passed, "a closing vehicle should trigger AEB"
    assert result.observed_metrics["triggered"] == 1.0
    assert result.observed_metrics["vision_detection_rate"] > 0.9
    assert result.observed_metrics["frames_detected_on"] > 0

    gaps = [g for g in result.time_series["gap_m"] if g >= 0]
    assert gaps[0] > gaps[-1], "the detected gap must actually close"

    states = set(result.time_series["brake_state"])
    assert 2.0 in states, "full_brake state must be reached"


@pytest.mark.skipif(not BUS_JPG.exists(), reason="bundled ultralytics sample image not found")
def test_vision_run_is_labelled_simulation_not_real_world():
    """A synthesized clip is not a real drive, and must not claim to be."""
    spec = build_vision_fusion_scenario()
    assert spec.data_source is DataSource.SIMULATION
    assert "synthes" in spec.gap_source_label.lower()
