import pytest

from controllers.acc import acc_controller


def test_zero_error_yields_zero_accel():
    # gap exactly at desired_gap, speeds matched -> no correction needed
    ego_speed = 20.0
    target_gap, time_gap = 2.0, 1.5
    desired_gap = target_gap + time_gap * ego_speed
    accel = acc_controller(ego_speed=ego_speed, lead_speed=ego_speed, gap=desired_gap)
    assert accel == pytest.approx(0.0, abs=1e-9)


def test_gap_too_small_decelerates():
    accel = acc_controller(ego_speed=20.0, lead_speed=20.0, gap=5.0)
    assert accel < 0


def test_gap_too_large_accelerates():
    accel = acc_controller(ego_speed=20.0, lead_speed=20.0, gap=200.0)
    assert accel > 0


def test_output_clipped_to_max_accel():
    accel = acc_controller(ego_speed=10.0, lead_speed=10.0, gap=1000.0)
    assert accel == pytest.approx(2.0)


def test_output_clipped_to_min_decel():
    accel = acc_controller(ego_speed=20.0, lead_speed=0.0, gap=0.0)
    assert accel == pytest.approx(-4.0)
