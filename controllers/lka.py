"""Proportional-derivative lane centering controller for Lane Keeping Assist."""

from __future__ import annotations


def lka_controller(
    lateral_offset: float,
    heading_error: float,
    max_steer: float = 0.5,
) -> float:
    """Commanded steering angle in radians, clipped to [-max_steer, max_steer]."""
    steering = -0.3 * lateral_offset - 0.5 * heading_error
    return max(-max_steer, min(max_steer, steering))
