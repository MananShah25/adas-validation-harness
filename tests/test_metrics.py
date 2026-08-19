import math

import pytest

from metrics.ttc_lane_brake import (
    brake_response_latency,
    following_distance_error,
    lane_deviation,
    time_to_collision,
)


def test_ttc_basic_closing():
    assert time_to_collision(ego_speed=20, lead_speed=10, gap_distance=50) == pytest.approx(5.0)


def test_ttc_lead_faster_returns_inf():
    assert time_to_collision(ego_speed=20, lead_speed=25, gap_distance=50) == math.inf


def test_ttc_equal_speeds_returns_inf():
    assert time_to_collision(ego_speed=20, lead_speed=20, gap_distance=50) == math.inf


def test_ttc_zero_gap_returns_zero():
    assert time_to_collision(ego_speed=20, lead_speed=0, gap_distance=0) == 0.0


def test_ttc_negative_gap_returns_zero():
    assert time_to_collision(ego_speed=20, lead_speed=0, gap_distance=-5) == 0.0


def test_lane_deviation_positive_offset():
    assert lane_deviation(0.25) == pytest.approx(0.25)


def test_lane_deviation_negative_offset():
    assert lane_deviation(-0.25) == pytest.approx(0.25)


def test_following_distance_error_farther_than_target():
    assert following_distance_error(actual_gap=30, target_gap=20) == pytest.approx(10)


def test_following_distance_error_closer_than_target():
    assert following_distance_error(actual_gap=15, target_gap=20) == pytest.approx(-5)


def test_brake_response_latency_basic():
    assert brake_response_latency(
        event_timestamp=10.0, brake_onset_timestamp=10.4
    ) == pytest.approx(0.4)


def test_brake_response_latency_negative_raises():
    with pytest.raises(ValueError):
        brake_response_latency(event_timestamp=10.0, brake_onset_timestamp=9.5)
