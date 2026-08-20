"""Forward vision module: YOLOv8n object detection for the lead vehicle,
plus a monocular distance estimate from bounding-box width.

Weights auto-download on first use (via ultralytics) into vision/weights/,
which is gitignored -- nothing here depends on that file being committed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

# COCO class ids for vehicle-like objects YOLOv8n is pretrained on.
VEHICLE_CLASS_IDS: frozenset[int] = frozenset({2, 3, 5, 7})  # car, motorcycle, bus, truck

AVERAGE_VEHICLE_WIDTH_M = 1.8  # typical passenger car width, used as the distance-heuristic reference
DEFAULT_WEIGHTS_PATH = Path(__file__).parent / "weights" / "yolov8n.pt"


def estimate_distance_m(
    bbox_width_px: float,
    focal_length_px: float,
    real_width_m: float = AVERAGE_VEHICLE_WIDTH_M,
) -> float | None:
    """Monocular distance heuristic: distance = (real_width * focal_length) / pixel_width.

    Returns None when the bounding box has no width (nothing to estimate from).
    """
    if bbox_width_px <= 0:
        return None
    return (real_width_m * focal_length_px) / bbox_width_px


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]  # pixel coordinates: (x1, y1, x2, y2)
    estimated_distance_m: float | None

    @property
    def bbox_width_px(self) -> float:
        x1, _, x2, _ = self.bbox_xyxy
        return x2 - x1


class VehicleDetector:
    """Wraps a YOLOv8n model, filtered to vehicle classes, with distance estimation."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_WEIGHTS_PATH,
        focal_length_px: float = 1000.0,
        confidence_threshold: float = 0.4,
    ):
        self.model = YOLO(str(model_path))
        self.focal_length_px = focal_length_px
        self.confidence_threshold = confidence_threshold

    def detect(self, frame: np.ndarray | str | Path) -> list[Detection]:
        """Runs detection on `frame` (an array, or a path to an image file) and
        returns every detected vehicle as a Detection, largest-bbox-first."""
        results = self.model.predict(frame, verbose=False, conf=self.confidence_threshold)
        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id not in VEHICLE_CLASS_IDS:
                    continue
                bbox_xyxy = tuple(float(v) for v in box.xyxy[0])
                width_px = bbox_xyxy[2] - bbox_xyxy[0]
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=result.names[class_id],
                        confidence=float(box.conf[0]),
                        bbox_xyxy=bbox_xyxy,
                        estimated_distance_m=estimate_distance_m(width_px, self.focal_length_px),
                    )
                )
        detections.sort(key=lambda d: d.bbox_width_px, reverse=True)
        return detections

    def lead_vehicle(self, frame: np.ndarray | str | Path) -> Detection | None:
        """The nearest detected vehicle (largest bounding-box width), or None."""
        detections = self.detect(frame)
        return detections[0] if detections else None
