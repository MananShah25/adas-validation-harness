"""Staged time-to-collision braking controller for Automatic Emergency Braking.

Thresholds default to the NHTSA-sourced values documented in
schemas/test_procedure.py (DEFAULT_PASS_CRITERIA[Feature.AEB]): a 2.0s
warning TTC and a 1.8s brake TTC, tighter than an unsourced placeholder.
"""

from __future__ import annotations


def aeb_controller(ttc: float, warning_ttc: float = 2.0, brake_ttc: float = 1.8) -> str:
    """Returns one of "full_brake", "warning", "normal" based on time-to-collision."""
    if ttc < brake_ttc:
        return "full_brake"
    elif ttc < warning_ttc:
        return "warning"
    return "normal"
