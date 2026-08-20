from controllers.aeb import aeb_controller


def test_ttc_above_warning_is_normal():
    assert aeb_controller(ttc=5.0) == "normal"


def test_ttc_between_thresholds_is_warning():
    assert aeb_controller(ttc=1.9, warning_ttc=2.0, brake_ttc=1.8) == "warning"


def test_ttc_below_brake_threshold_is_full_brake():
    assert aeb_controller(ttc=1.0, warning_ttc=2.0, brake_ttc=1.8) == "full_brake"


def test_ttc_at_warning_boundary_is_warning():
    # ttc < warning_ttc is the trigger condition; exactly-equal stays "normal"
    assert aeb_controller(ttc=2.0, warning_ttc=2.0, brake_ttc=1.8) == "normal"
    assert aeb_controller(ttc=1.999, warning_ttc=2.0, brake_ttc=1.8) == "warning"


def test_ttc_at_brake_boundary_is_warning_not_full_brake():
    assert aeb_controller(ttc=1.8, warning_ttc=2.0, brake_ttc=1.8) == "warning"
    assert aeb_controller(ttc=1.799, warning_ttc=2.0, brake_ttc=1.8) == "full_brake"
