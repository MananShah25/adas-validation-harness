"""OBD-II ingestion layer, run in two modes so nothing here depends on
hardware: "mock" (this module, a synthetic drive-cycle generator) and
"replay" (Phase 6, replays a downloaded public dataset through the same
interface). A future "live" mode, once real Bluetooth OBD-II hardware is
attached, is meant to be a one-line change: OBDReader(mode="mock") ->
OBDReader(mode="live"). No pipeline code downstream of OBDReader should
need to change when that happens.

Response shape loosely mirrors python-obd's OBDResponse (a command, a
value, and a unit) without taking a hard dependency on the python-obd
package itself, which assumes a live serial connection.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Iterator


class OBDCommand(str, Enum):
    SPEED = "SPEED"
    RPM = "RPM"
    THROTTLE_POS = "THROTTLE_POS"
    # Not a standard SAE Mode 01 PID (real vehicles expose it via a separate
    # signal or a manufacturer-specific PID), but the validation harness
    # needs it for brake_response_latency, so it's modeled here alongside
    # the standard PIDs.
    BRAKE = "BRAKE"


SUPPORTED_COMMANDS: tuple[OBDCommand, ...] = (
    OBDCommand.SPEED,
    OBDCommand.RPM,
    OBDCommand.THROTTLE_POS,
    OBDCommand.BRAKE,
)

_UNITS: dict[OBDCommand, str] = {
    OBDCommand.SPEED: "kph",
    OBDCommand.RPM: "rpm",
    OBDCommand.THROTTLE_POS: "percent",
    OBDCommand.BRAKE: "boolean",
}


@dataclass
class OBDResponse:
    command: OBDCommand
    value: float | None
    unit: str
    timestamp: float

    def is_null(self) -> bool:
        return self.value is None


class MockOBDSource:
    """Synthetic OBD-II data generator.

    Simulates a plausible drive cycle -- cruising speed with gentle random
    throttle variation, punctuated by occasional braking events (throttle
    drops to 0, brake engages, speed decreases) -- matching the shape of a
    real drive log closely enough to exercise the ingestion/parsing logic
    before real hardware or a replayed dataset is available.
    """

    def __init__(self, seed: int = 0, hz: float = 10.0):
        self._rng = random.Random(seed)
        self.hz = hz
        self.timestamp = 0.0
        self._speed_kph = 40.0
        self._rpm = 800.0 + 40.0 * 35.0
        self._throttle = 20.0
        self._braking = False
        self._brake_timer = 0.0

    def advance(self) -> None:
        """Step the synthetic drive cycle forward by one sample interval."""
        dt = 1.0 / self.hz
        self.timestamp += dt

        if not self._braking and self._rng.random() < 0.01:
            self._braking = True
            self._brake_timer = self._rng.uniform(1.0, 3.0)

        if self._braking:
            self._brake_timer -= dt
            self._throttle = 0.0
            self._speed_kph = max(0.0, self._speed_kph - 15.0 * dt)
            if self._brake_timer <= 0.0:
                self._braking = False
        else:
            self._throttle = max(0.0, min(100.0, self._throttle + self._rng.uniform(-2.0, 2.0)))
            accel = (self._throttle - 20.0) * 0.05
            self._speed_kph = max(0.0, min(140.0, self._speed_kph + accel * dt * 10.0))

        self._rpm = max(0.0, 800.0 + self._speed_kph * 35.0 + self._rng.uniform(-50.0, 50.0))

    def current_value(self, command: OBDCommand) -> float:
        """Value for `command` at the current (already-advanced) sample."""
        if command is OBDCommand.SPEED:
            return self._speed_kph
        if command is OBDCommand.RPM:
            return self._rpm
        if command is OBDCommand.THROTTLE_POS:
            return self._throttle
        if command is OBDCommand.BRAKE:
            return 1.0 if self._braking else 0.0
        raise ValueError(f"Unsupported command: {command!r}")


class OBDReader:
    """Unified OBD-II ingestion interface.

    OBDReader(mode="mock") requires no hardware and is safe to unit test.
    OBDReader(mode="live") is reserved for real Bluetooth OBD-II hardware
    and isn't implemented yet (see project spec: "What NOT to build yet").
    """

    def __init__(self, mode: str = "mock", seed: int = 0, hz: float = 10.0):
        self.mode = mode
        self.hz = hz
        if mode == "mock":
            self._source = MockOBDSource(seed=seed, hz=hz)
        elif mode == "live":
            raise NotImplementedError(
                "OBDReader(mode='live') requires real Bluetooth OBD-II hardware, "
                "which hasn't been attached yet -- see project spec."
            )
        else:
            raise ValueError(f"Unknown OBDReader mode: {mode!r}")
        self._source.advance()  # so a query() before the first next_sample() has valid data

    def next_sample(self) -> dict[OBDCommand, OBDResponse]:
        """Advance one time step and return readings for every supported command."""
        self._source.advance()
        return {command: self.query(command) for command in SUPPORTED_COMMANDS}

    def query(self, command: OBDCommand) -> OBDResponse:
        """Read a single command from the current (already-advanced) sample.

        Mirrors python-obd's connection.query(command) -- a single-PID read
        against whatever the vehicle's state currently is, without itself
        advancing that state.
        """
        value = self._source.current_value(command)
        return OBDResponse(command=command, value=value, unit=_UNITS[command], timestamp=self._source.timestamp)

    def stream(self, n_samples: int) -> Iterator[dict[OBDCommand, OBDResponse]]:
        """Yield `n_samples` consecutive full samples, one per call to next_sample()."""
        for _ in range(n_samples):
            yield self.next_sample()
