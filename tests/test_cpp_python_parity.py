"""Deterministic parity between the C++ control-loop core and the pure-Python
reference implementation.

The Python implementations in metrics/ and controllers/ are the reference
oracle; the C++ port in cpp/ must agree with them exactly. "Exactly" here means
bit-identical for the metrics and controllers, not merely close: both sides do
the same IEEE-754 double arithmetic in the same order, so any drift would
indicate a genuine porting error rather than acceptable floating-point noise.

Skips cleanly if the extension module hasn't been built, so the suite still
passes on a fresh clone with no C++ toolchain. Build it with:

    cd cpp && cmake -S . -B build && cmake --build build
"""

from __future__ import annotations

import math

import pytest

adas_core = pytest.importorskip(
    "adas_core",
    reason="C++ extension not built; run: cd cpp && cmake -S . -B build && cmake --build build",
)

from controllers.acc import acc_controller as py_acc
from controllers.aeb import aeb_controller as py_aeb
from controllers.lka import lka_controller as py_lka
from metrics.ttc_lane_brake import brake_response_latency as py_latency
from metrics.ttc_lane_brake import following_distance_error as py_gap_err
from metrics.ttc_lane_brake import lane_deviation as py_dev
from metrics.ttc_lane_brake import time_to_collision as py_ttc

# Deterministic sweeps -- fixed literals, no RNG, so a failure is always
# reproducible from the test name alone.
SPEEDS = [0.0, 1.0, 5.5, 10.0, 15.0, 20.0, 22.5, 25.0, 30.0, 40.0]
GAPS = [-5.0, 0.0, 0.5, 2.0, 10.0, 25.0, 39.5, 50.0, 100.0, 250.0]
OFFSETS = [-2.0, -0.6, -0.3, -0.05, 0.0, 0.05, 0.3, 0.6, 2.0]
HEADINGS = [-0.5, -0.15, -0.01, 0.0, 0.01, 0.15, 0.5]
TTCS = [0.0, 0.5, 1.0, 1.5, 1.799, 1.8, 1.801, 1.999, 2.0, 2.001, 5.0, 50.0, math.inf]


def _identical(a: float, b: float) -> bool:
    """Bit-identical, treating inf/inf and nan/nan as equal."""
    if math.isnan(a) and math.isnan(b):
        return True
    if math.isinf(a) or math.isinf(b):
        return a == b
    return a == b


# ---------------------------------------------------------------- metrics


def test_time_to_collision_parity_across_grid():
    checked = 0
    for ego in SPEEDS:
        for lead in SPEEDS:
            for gap in GAPS:
                got = adas_core.time_to_collision(ego, lead, gap)
                want = py_ttc(ego, lead, gap)
                assert _identical(got, want), f"ttc({ego},{lead},{gap}): C++={got} Python={want}"
                checked += 1
    assert checked == len(SPEEDS) * len(SPEEDS) * len(GAPS)


def test_lane_deviation_parity():
    for offset in OFFSETS:
        assert _identical(adas_core.lane_deviation(offset), py_dev(offset))


def test_following_distance_error_parity():
    for actual in GAPS:
        for target in GAPS:
            got = adas_core.following_distance_error(actual, target)
            want = py_gap_err(actual, target)
            assert _identical(got, want), f"gap_err({actual},{target}): C++={got} Python={want}"


def test_brake_response_latency_parity():
    for event in [0.0, 1.0, 3.0, 10.0]:
        for onset in [0.0, 1.0, 3.0, 10.0, 10.4]:
            if onset >= event:
                assert _identical(
                    adas_core.brake_response_latency(event, onset), py_latency(event, onset)
                )
            else:
                # Both sides must reject the same inputs with the same type.
                with pytest.raises(ValueError):
                    adas_core.brake_response_latency(event, onset)
                with pytest.raises(ValueError):
                    py_latency(event, onset)


# ------------------------------------------------------------ controllers


def test_acc_controller_parity_across_grid():
    checked = 0
    for ego in SPEEDS:
        for lead in SPEEDS:
            for gap in GAPS:
                got = adas_core.acc_controller(ego, lead, gap)
                want = py_acc(ego, lead, gap)
                assert _identical(got, want), f"acc({ego},{lead},{gap}): C++={got} Python={want}"
                checked += 1
    assert checked == len(SPEEDS) * len(SPEEDS) * len(GAPS)


def test_acc_controller_parity_with_custom_limits():
    for min_a, max_a in [(-4.0, 2.0), (-1.0, 0.5), (-9.0, 3.0)]:
        for ego in SPEEDS:
            for gap in GAPS:
                got = adas_core.acc_controller(ego, ego, gap, 2.0, 1.5, min_a, max_a)
                want = py_acc(ego, ego, gap, 2.0, 1.5, min_a, max_a)
                assert _identical(got, want)


def test_lka_controller_parity_across_grid():
    for offset in OFFSETS:
        for heading in HEADINGS:
            got = adas_core.lka_controller(offset, heading)
            want = py_lka(offset, heading)
            assert _identical(got, want), f"lka({offset},{heading}): C++={got} Python={want}"


def test_aeb_controller_parity_including_threshold_boundaries():
    for ttc in TTCS:
        got = adas_core.aeb_controller(ttc)
        want = py_aeb(ttc)
        assert got == want, f"aeb({ttc}): C++={got!r} Python={want!r}"


def test_aeb_controller_parity_with_custom_thresholds():
    for warn, brake in [(2.0, 1.8), (3.0, 1.0), (1.5, 1.5)]:
        for ttc in TTCS:
            assert adas_core.aeb_controller(ttc, warn, brake) == py_aeb(ttc, warn, brake)


# ------------------------------------------------- interop surface details


def test_cpp_module_accepts_the_same_keyword_arguments():
    """Drop-in replacement means keyword names must match, not just positions."""
    assert _identical(
        adas_core.acc_controller(ego_speed=20.0, lead_speed=18.0, gap=30.0),
        py_acc(ego_speed=20.0, lead_speed=18.0, gap=30.0),
    )
    assert _identical(
        adas_core.lka_controller(lateral_offset=0.2, heading_error=0.05),
        py_lka(lateral_offset=0.2, heading_error=0.05),
    )
    assert adas_core.aeb_controller(ttc=1.5) == py_aeb(ttc=1.5)


def test_cpp_defaults_match_python_defaults():
    """Defaults live in two places now; this pins them together."""
    assert _identical(adas_core.acc_controller(20.0, 20.0, 40.0), py_acc(20.0, 20.0, 40.0))
    assert _identical(adas_core.lka_controller(0.3, 0.1), py_lka(0.3, 0.1))
    assert adas_core.aeb_controller(1.9) == py_aeb(1.9)


def test_infinity_crosses_the_language_boundary_intact():
    result = adas_core.time_to_collision(20.0, 25.0, 50.0)
    assert math.isinf(result) and result > 0
    assert adas_core.aeb_controller(math.inf) == "normal"
