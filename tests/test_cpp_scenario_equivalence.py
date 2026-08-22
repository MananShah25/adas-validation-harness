"""Scenario-level equivalence: the existing Python test scenarios, driven by
the C++ control-loop core, must produce identical results.

The grid parity test (test_cpp_python_parity.py) proves the functions agree in
isolation. This proves something stronger and more useful: swap the C++ core in
underneath the *unmodified* scenario suites and every observed metric across a
full simulation run comes out identical. That is what substantiates "the
control loop core is implemented in C++, driven from Python" -- the harness
above it is genuinely unchanged.

The swap is done by monkeypatching the already-imported module attributes the
scenario modules call, which is exactly how a runtime backend switch would
work; no scenario code is edited.
"""

from __future__ import annotations

import pytest

adas_core = pytest.importorskip(
    "adas_core",
    reason="C++ extension not built; run: cd cpp && cmake -S . -B build && cmake --build build",
)

import scenarios.acc_scenarios as acc_mod
import scenarios.aeb_scenarios as aeb_mod
import scenarios.lka_scenarios as lka_mod


@pytest.fixture
def cpp_backend(monkeypatch):
    """Swap every per-timestep function the scenario modules call for its C++
    equivalent, for the duration of one test."""
    # ACC scenarios use acc_controller + the two metrics.
    monkeypatch.setattr(acc_mod, "acc_controller", adas_core.acc_controller)
    monkeypatch.setattr(acc_mod, "time_to_collision", adas_core.time_to_collision)
    monkeypatch.setattr(acc_mod, "following_distance_error", adas_core.following_distance_error)

    # AEB scenarios use acc_controller as the baseline plus aeb_controller.
    monkeypatch.setattr(aeb_mod, "acc_controller", adas_core.acc_controller)
    monkeypatch.setattr(aeb_mod, "aeb_controller", adas_core.aeb_controller)
    monkeypatch.setattr(aeb_mod, "time_to_collision", adas_core.time_to_collision)
    monkeypatch.setattr(aeb_mod, "brake_response_latency", adas_core.brake_response_latency)

    # LKA scenarios use lka_controller + acc_controller (speed hold) + deviation.
    monkeypatch.setattr(lka_mod, "lka_controller", adas_core.lka_controller)
    monkeypatch.setattr(lka_mod, "acc_controller", adas_core.acc_controller)
    monkeypatch.setattr(lka_mod, "lane_deviation", adas_core.lane_deviation)
    return adas_core


def _run_suite(specs, runner):
    out = {}
    for spec in specs:
        result = runner(spec)
        out[spec.scenario_id] = (result.passed, dict(result.observed_metrics))
    return out


def _assert_identical(python_results: dict, cpp_results: dict, label: str):
    assert set(python_results) == set(cpp_results), f"{label}: scenario sets differ"
    for scenario_id in python_results:
        py_passed, py_metrics = python_results[scenario_id]
        cpp_passed, cpp_metrics = cpp_results[scenario_id]
        assert py_passed == cpp_passed, (
            f"{label}/{scenario_id}: pass/fail differs -- Python={py_passed} C++={cpp_passed}"
        )
        assert set(py_metrics) == set(cpp_metrics), f"{label}/{scenario_id}: metric keys differ"
        for key in py_metrics:
            assert py_metrics[key] == cpp_metrics[key], (
                f"{label}/{scenario_id}.{key}: Python={py_metrics[key]} C++={cpp_metrics[key]}"
            )


def test_acc_suite_identical_under_cpp_core(cpp_backend, monkeypatch):
    specs = acc_mod.build_acc_scenarios()
    # Baseline with pure Python first (fixture not yet applied to a fresh import
    # is impossible, so capture Python results by undoing the patch).
    monkeypatch.undo()
    python_results = _run_suite(specs, acc_mod.run_acc_scenario)

    monkeypatch.setattr(acc_mod, "acc_controller", adas_core.acc_controller)
    monkeypatch.setattr(acc_mod, "time_to_collision", adas_core.time_to_collision)
    monkeypatch.setattr(acc_mod, "following_distance_error", adas_core.following_distance_error)
    cpp_results = _run_suite(specs, acc_mod.run_acc_scenario)

    _assert_identical(python_results, cpp_results, "ACC")
    assert len(cpp_results) == 10


def test_lka_suite_identical_under_cpp_core(monkeypatch):
    specs = lka_mod.build_lka_scenarios()
    python_results = _run_suite(specs, lka_mod.run_lka_scenario)

    monkeypatch.setattr(lka_mod, "lka_controller", adas_core.lka_controller)
    monkeypatch.setattr(lka_mod, "acc_controller", adas_core.acc_controller)
    monkeypatch.setattr(lka_mod, "lane_deviation", adas_core.lane_deviation)
    cpp_results = _run_suite(specs, lka_mod.run_lka_scenario)

    _assert_identical(python_results, cpp_results, "LKA")
    assert len(cpp_results) == 10


def test_aeb_suite_identical_under_cpp_core(monkeypatch):
    specs = aeb_mod.build_aeb_scenarios()
    python_results = _run_suite(specs, aeb_mod.run_aeb_scenario)

    monkeypatch.setattr(aeb_mod, "acc_controller", adas_core.acc_controller)
    monkeypatch.setattr(aeb_mod, "aeb_controller", adas_core.aeb_controller)
    monkeypatch.setattr(aeb_mod, "time_to_collision", adas_core.time_to_collision)
    monkeypatch.setattr(aeb_mod, "brake_response_latency", adas_core.brake_response_latency)
    cpp_results = _run_suite(specs, aeb_mod.run_aeb_scenario)

    _assert_identical(python_results, cpp_results, "AEB")
    assert len(cpp_results) == 9


def test_full_track1_pass_rates_unchanged_under_cpp_core(monkeypatch):
    """The headline numbers (6/10, 8/10, 8/9) must hold with the C++ core."""
    monkeypatch.setattr(acc_mod, "acc_controller", adas_core.acc_controller)
    monkeypatch.setattr(acc_mod, "time_to_collision", adas_core.time_to_collision)
    monkeypatch.setattr(acc_mod, "following_distance_error", adas_core.following_distance_error)
    monkeypatch.setattr(lka_mod, "lka_controller", adas_core.lka_controller)
    monkeypatch.setattr(lka_mod, "acc_controller", adas_core.acc_controller)
    monkeypatch.setattr(lka_mod, "lane_deviation", adas_core.lane_deviation)
    monkeypatch.setattr(aeb_mod, "acc_controller", adas_core.acc_controller)
    monkeypatch.setattr(aeb_mod, "aeb_controller", adas_core.aeb_controller)
    monkeypatch.setattr(aeb_mod, "time_to_collision", adas_core.time_to_collision)
    monkeypatch.setattr(aeb_mod, "brake_response_latency", adas_core.brake_response_latency)

    acc = [acc_mod.run_acc_scenario(s) for s in acc_mod.build_acc_scenarios()]
    lka = [lka_mod.run_lka_scenario(s) for s in lka_mod.build_lka_scenarios()]
    aeb = [aeb_mod.run_aeb_scenario(s) for s in aeb_mod.build_aeb_scenarios()]

    assert sum(r.passed for r in acc) == 6, "ACC pass rate changed under the C++ core"
    assert sum(r.passed for r in lka) == 8, "LKA pass rate changed under the C++ core"
    assert sum(r.passed for r in aeb) == 8, "AEB pass rate changed under the C++ core"
