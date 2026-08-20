from pathlib import Path

import numpy as np
import pytest
import ultralytics

from vision.detector import VEHICLE_CLASS_IDS, VehicleDetector, estimate_distance_m

BUS_JPG = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"


def test_estimate_distance_basic():
    # 1.8m real width, 1000px focal length, 900px wide bbox -> 2.0m away
    assert estimate_distance_m(bbox_width_px=900.0, focal_length_px=1000.0, real_width_m=1.8) == pytest.approx(2.0)


def test_estimate_distance_scales_inversely_with_bbox_width():
    near = estimate_distance_m(bbox_width_px=900.0, focal_length_px=1000.0)
    far = estimate_distance_m(bbox_width_px=450.0, focal_length_px=1000.0)
    assert far == pytest.approx(near * 2)


def test_estimate_distance_zero_width_returns_none():
    assert estimate_distance_m(bbox_width_px=0.0, focal_length_px=1000.0) is None


def test_estimate_distance_negative_width_returns_none():
    assert estimate_distance_m(bbox_width_px=-5.0, focal_length_px=1000.0) is None


@pytest.fixture(scope="module")
def detector():
    return VehicleDetector()


@pytest.mark.skipif(not BUS_JPG.exists(), reason="bundled ultralytics sample image not found")
class TestRealImageDetection:
    """Uses the real photo bundled with the installed ultralytics package
    (not synthetic, not externally downloaded) since no dashcam footage is
    available yet -- deferred per project decision.
    """

    def test_detects_a_vehicle_class(self, detector):
        detections = detector.detect(str(BUS_JPG))
        assert len(detections) >= 1
        assert all(d.class_id in VEHICLE_CLASS_IDS for d in detections)
        assert all(0.0 <= d.confidence <= 1.0 for d in detections)

    def test_lead_vehicle_has_positive_distance_estimate(self, detector):
        lead = detector.lead_vehicle(str(BUS_JPG))
        assert lead is not None
        assert lead.estimated_distance_m > 0.0

    def test_bbox_width_matches_xyxy(self, detector):
        lead = detector.lead_vehicle(str(BUS_JPG))
        x1, _, x2, _ = lead.bbox_xyxy
        assert lead.bbox_width_px == pytest.approx(x2 - x1)


def test_blank_frame_yields_no_detections(detector):
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    assert detector.detect(blank) == []
    assert detector.lead_vehicle(blank) is None
