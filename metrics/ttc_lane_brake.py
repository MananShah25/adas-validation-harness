"""Core ADAS metrics shared by simulation and real-vehicle tracks.

Keeping these calculations in one place means a pass/fail threshold means
the same thing regardless of whether the numbers came from highway-env or
a replayed OBD-II log.
"""

from __future__ import annotations

import math


def time_to_collision(ego_speed: float, lead_speed: float, gap_distance: float) -> float:
    """Seconds until ego collides with lead vehicle, assuming constant speeds.

    Returns math.inf if the gap is not closing (lead as fast or faster than
    ego), since there is no collision to predict under constant speeds.
    """
    closing_speed = ego_speed - lead_speed
    if closing_speed <= 0:
        return math.inf
    if gap_distance <= 0:
        return 0.0
    return gap_distance / closing_speed


def lane_deviation(lateral_offset_from_center: float) -> float:
    """Absolute lateral deviation from lane center, in meters."""
    return abs(lateral_offset_from_center)


def following_distance_error(actual_gap: float, target_gap: float) -> float:
    """Signed error between actual and target following distance, in meters.

    Positive means following farther than the target gap, negative means
    closer.
    """
    return actual_gap - target_gap


def brake_response_latency(event_timestamp: float, brake_onset_timestamp: float) -> float:
    """Seconds between the triggering event and brake onset.

    Raises ValueError if brake onset precedes the triggering event, since
    that indicates a timestamp/ordering bug in the caller rather than a
    valid (even if bad) latency measurement.
    """
    latency = brake_onset_timestamp - event_timestamp
    if latency < 0:
        raise ValueError(
            f"brake_onset_timestamp ({brake_onset_timestamp}) is before "
            f"event_timestamp ({event_timestamp})"
        )
    return latency
