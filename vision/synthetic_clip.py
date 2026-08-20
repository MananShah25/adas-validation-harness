"""Synthesized approach clips, for exercising the vision->fusion path until
real dashcam footage is available.

WHAT THIS IS: a real photograph of a real vehicle, composited at increasing
scale onto a fixed-size canvas so the vehicle grows in frame the way it
would if you were closing on it. YOLO runs on every frame for real, so the
bounding boxes, the monocular distance estimates, and everything downstream
are genuine detector output -- not hand-written numbers.

WHAT THIS IS NOT: real dashcam footage. There is no real ego motion, no
road scene changing, no motion blur or lighting variation, and the
"approach" is a scale ramp rather than translation through a 3D scene. It
is enough to prove the pipeline is wired correctly end to end. It is NOT
enough to characterise real-world detection accuracy, and no accuracy claim
should be made from it.

Replace with VideoFrameSequence over a real clip when footage exists; the
fusion code above it does not change.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vision.frames import FrameSequence

# Mid-grey surround. Keeps the composite neutral and avoids introducing
# edge artefacts that the detector might latch onto.
CANVAS_FILL = 128


def make_approach_frames(
    image_path: str | Path,
    n_frames: int = 24,
    start_scale: float = 0.22,
    end_scale: float = 0.90,
) -> list[np.ndarray]:
    """Frames in which the subject grows, simulating an approach.

    `start_scale` is the subject's apparent size at the start (small = far),
    `end_scale` at the end (large = near). Canvas size is constant across the
    sequence, which keeps focal length constant -- that is what makes the
    monocular distance heuristic comparable frame to frame. end_scale stays
    below 0.95 so the subject never touches the frame edge; a clipped bounding
    box would stop growing and silently flatten the distance curve.

    Apparent size is ramped on a reciprocal schedule, not linearly. Estimated
    distance goes as 1/bbox_width, so a linear size ramp produces a distance
    curve that collapses hyperbolically -- a lead vehicle that appears to brake
    violently as it nears, which is not the constant-closing-speed approach an
    AEB test wants. Interpolating 1/scale linearly makes distance fall linearly
    instead, i.e. a steady closing speed.
    """
    import cv2

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")
    if n_frames < 2:
        raise ValueError("n_frames must be at least 2 to form an approach")
    if not (0 < start_scale < end_scale <= 0.95):
        raise ValueError(
            f"require 0 < start_scale < end_scale <= 0.95, got {start_scale}, {end_scale}"
        )

    source = cv2.imread(str(path))
    if source is None:
        raise ValueError(f"could not decode image: {path}")

    height, width = source.shape[:2]

    inv_start, inv_end = 1.0 / start_scale, 1.0 / end_scale

    frames: list[np.ndarray] = []
    for i in range(n_frames):
        inv_scale = inv_start + (inv_end - inv_start) * (i / (n_frames - 1))
        scale = 1.0 / inv_scale
        pasted_w, pasted_h = max(1, int(width * scale)), max(1, int(height * scale))
        resized = cv2.resize(source, (pasted_w, pasted_h), interpolation=cv2.INTER_AREA)

        canvas = np.full((height, width, 3), CANVAS_FILL, dtype=np.uint8)
        x0 = (width - pasted_w) // 2
        y0 = (height - pasted_h) // 2
        canvas[y0:y0 + pasted_h, x0:x0 + pasted_w] = resized
        frames.append(canvas)

    return frames


def make_approach_sequence(
    image_path: str | Path,
    fps: float = 5.0,
    n_frames: int = 24,
    start_scale: float = 0.22,
    end_scale: float = 0.90,
) -> FrameSequence:
    """make_approach_frames(...) wrapped as a time-addressable FrameSequence."""
    frames = make_approach_frames(
        image_path, n_frames=n_frames, start_scale=start_scale, end_scale=end_scale
    )
    return FrameSequence(frames, fps=fps)
