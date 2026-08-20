"""Timed frame sequences for the forward vision module.

The fusion pipeline asks for a gap distance at a given elapsed time, so the
vision side needs frames addressable by time rather than by index. This
module provides that in two forms: an in-memory sequence (a list of arrays
or image paths at a fixed rate) and a video-file sequence (a dashcam clip).

Both expose the same frame_at(t) contract, so swapping a synthesized test
clip for real dashcam footage later changes only which one is constructed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Decoding a whole clip into memory keeps frame_at(t) exact and cheap;
# random access via cv2's CAP_PROP_POS_FRAMES is unreliable across codecs.
# Dashcam test clips are short, so this is the right tradeoff -- but it is
# capped so a large file fails loudly instead of exhausting memory.
DEFAULT_MAX_FRAMES = 900


class FrameSequence:
    """A fixed-rate sequence of frames addressable by elapsed time.

    Frames may be numpy arrays or paths to image files -- VehicleDetector
    accepts either, so no decoding happens here.
    """

    def __init__(self, frames: list, fps: float = 10.0):
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        if not frames:
            raise ValueError("FrameSequence requires at least one frame")
        self.frames = list(frames)
        self.fps = fps

    @property
    def duration_s(self) -> float:
        return len(self.frames) / self.fps

    def index_at(self, t: float) -> int | None:
        """Frame index covering elapsed time `t`, or None past the end."""
        if t < 0:
            return None
        index = int(t * self.fps)
        return index if index < len(self.frames) else None

    def frame_at(self, t: float):
        """The frame covering elapsed time `t`, or None past the end."""
        index = self.index_at(t)
        return None if index is None else self.frames[index]

    def __len__(self) -> int:
        return len(self.frames)


class VideoFrameSequence(FrameSequence):
    """A FrameSequence decoded from a video file (e.g. a dashcam clip).

    Requires opencv. Uses the file's own frame rate unless one is given.
    """

    def __init__(self, video_path: str | Path, fps: float | None = None, max_frames: int = DEFAULT_MAX_FRAMES):
        import cv2

        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"video not found: {path}")

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"could not open video: {path}")
        try:
            source_fps = capture.get(cv2.CAP_PROP_FPS)
            frames = []
            while len(frames) < max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            capture.release()

        if not frames:
            raise ValueError(f"no decodable frames in video: {path}")
        if len(frames) == max_frames:
            raise ValueError(
                f"video {path} exceeds max_frames={max_frames}; trim the clip or raise the cap "
                "rather than silently validating against a truncated recording"
            )

        resolved_fps = fps if fps is not None else (source_fps if source_fps and source_fps > 0 else 10.0)
        super().__init__(frames, fps=resolved_fps)
        self.video_path = path


def frames_from_images(paths: list[str | Path], fps: float = 10.0) -> FrameSequence:
    """A FrameSequence over image files on disk, played at `fps`."""
    resolved = [Path(p) for p in paths]
    missing = [str(p) for p in resolved if not p.exists()]
    if missing:
        raise FileNotFoundError(f"image(s) not found: {missing}")
    return FrameSequence(resolved, fps=fps)


def frames_from_arrays(arrays: list[np.ndarray], fps: float = 10.0) -> FrameSequence:
    """A FrameSequence over in-memory frames."""
    return FrameSequence(arrays, fps=fps)
