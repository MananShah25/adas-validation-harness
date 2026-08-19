"""Proportional gap controller for Adaptive Cruise Control."""

from __future__ import annotations


def acc_controller(
    ego_speed: float,
    lead_speed: float,
    gap: float,
    target_gap: float = 2.0,
    time_gap: float = 1.5,
    min_accel: float = -4.0,
    max_accel: float = 2.0,
) -> float:
    """Commanded longitudinal acceleration in m/s^2, clipped to [min_accel, max_accel]."""
    desired_gap = target_gap + time_gap * ego_speed
    gap_error = gap - desired_gap
    speed_error = lead_speed - ego_speed
    accel = 0.5 * gap_error + 0.3 * speed_error
    return max(min_accel, min(max_accel, accel))
