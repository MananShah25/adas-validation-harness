from pathlib import Path

import numpy as np
import pytest
import ultralytics

from vision.frames import FrameSequence, VideoFrameSequence, frames_from_arrays, frames_from_images
from vision.synthetic_clip import make_approach_frames, make_approach_sequence

BUS_JPG = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"


def _arrays(n=5):
    return [np.full((8, 8, 3), i, dtype=np.uint8) for i in range(n)]


# ---------- FrameSequence ----------

def test_index_at_maps_time_to_frame():
    seq = FrameSequence(_arrays(5), fps=10.0)
    assert seq.index_at(0.0) == 0
    assert seq.index_at(0.05) == 0   # within the first frame's interval
    assert seq.index_at(0.1) == 1
    assert seq.index_at(0.45) == 4


def test_index_and_frame_past_end_return_none():
    seq = FrameSequence(_arrays(3), fps=10.0)
    assert seq.index_at(0.3) is None
    assert seq.frame_at(0.3) is None
    assert seq.frame_at(99.0) is None


def test_negative_time_returns_none():
    seq = FrameSequence(_arrays(3), fps=10.0)
    assert seq.index_at(-0.1) is None


def test_duration_and_len():
    seq = FrameSequence(_arrays(20), fps=5.0)
    assert len(seq) == 20
    assert seq.duration_s == pytest.approx(4.0)


def test_rejects_empty_or_invalid_fps():
    with pytest.raises(ValueError):
        FrameSequence([], fps=10.0)
    with pytest.raises(ValueError):
        FrameSequence(_arrays(2), fps=0.0)
    with pytest.raises(ValueError):
        FrameSequence(_arrays(2), fps=-5.0)


def test_frames_from_images_rejects_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        frames_from_images([tmp_path / "nope.jpg"])


def test_frames_from_images_accepts_real_file():
    seq = frames_from_images([BUS_JPG, BUS_JPG], fps=2.0)
    assert len(seq) == 2
    assert seq.duration_s == pytest.approx(1.0)


def test_frames_from_arrays_roundtrip():
    seq = frames_from_arrays(_arrays(4), fps=8.0)
    assert len(seq) == 4
    assert seq.frame_at(0.0).shape == (8, 8, 3)


def test_video_sequence_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        VideoFrameSequence("definitely_not_a_video.mp4")


# ---------- synthetic approach clip ----------

def test_approach_frames_keep_constant_resolution():
    frames = make_approach_frames(BUS_JPG, n_frames=6)
    shapes = {f.shape for f in frames}
    assert len(shapes) == 1, "canvas size must stay constant to hold focal length fixed"


def test_approach_sequence_length_and_timing():
    seq = make_approach_sequence(BUS_JPG, fps=5.0, n_frames=10)
    assert len(seq) == 10
    assert seq.duration_s == pytest.approx(2.0)


def test_approach_rejects_degenerate_parameters():
    with pytest.raises(ValueError):
        make_approach_frames(BUS_JPG, n_frames=1)
    with pytest.raises(ValueError):
        make_approach_frames(BUS_JPG, start_scale=0.9, end_scale=0.2)  # not increasing
    with pytest.raises(ValueError):
        make_approach_frames(BUS_JPG, start_scale=0.2, end_scale=0.99)  # would clip the frame


def test_approach_missing_image_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        make_approach_frames(tmp_path / "nope.jpg")


def test_subject_grows_across_the_approach():
    """The composited subject must occupy progressively more of the canvas --
    this is what makes the detected bounding box grow."""
    frames = make_approach_frames(BUS_JPG, n_frames=6)
    # Non-background pixels stand in for subject area; the canvas fill is flat grey.
    areas = [int(np.count_nonzero(np.any(f != 128, axis=2))) for f in frames]
    assert areas == sorted(areas), f"subject area must be non-decreasing, got {areas}"
    assert areas[-1] > areas[0] * 2, "subject should grow substantially over the approach"
