import pytest

from controllers.lka import lka_controller


def test_zero_error_yields_zero_steering():
    assert lka_controller(lateral_offset=0.0, heading_error=0.0) == pytest.approx(0.0)


def test_positive_offset_steers_negative():
    # offset to the right of center (positive) should steer left (negative)
    assert lka_controller(lateral_offset=0.5, heading_error=0.0) < 0


def test_negative_offset_steers_positive():
    assert lka_controller(lateral_offset=-0.5, heading_error=0.0) > 0


def test_output_clipped_to_max_steer():
    steering = lka_controller(lateral_offset=10.0, heading_error=10.0, max_steer=0.4)
    assert steering == pytest.approx(-0.4)


def test_output_clipped_to_min_steer():
    steering = lka_controller(lateral_offset=-10.0, heading_error=-10.0, max_steer=0.4)
    assert steering == pytest.approx(0.4)
