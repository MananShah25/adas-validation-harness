"""OBD-II ingestion layer, run in three modes so nothing here depends on
hardware: "mock" (a synthetic drive-cycle generator), "replay" (replays a
downloaded public dataset through the same interface), and "live" (a
deliberate NotImplementedError -- real Bluetooth OBD-II hardware isn't
attached yet, see project spec). Swapping to "live" later is meant to be a
one-line change; no pipeline code downstream of OBDReader should need to
change when that happens.

Response shape loosely mirrors python-obd's OBDResponse (a command, a
value, and a unit) without taking a hard dependency on the python-obd
package itself, which assumes a live serial connection.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
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


class ReplayExhausted(RuntimeError):
    """Raised by a replay source's advance() when no recorded samples remain."""


def _parse_numeric(raw: str) -> float | None:
    """Parses fields like "48,60%" or "1,4" (comma-decimal, optional '%')."""
    raw = raw.strip()
    if not raw:
        return None
    raw = raw.rstrip("%").replace(",", ".")
    try:
        return float(raw)
    except ValueError:
        return None


@dataclass
class _ReplayRow:
    timestamp_s: float
    speed_kph: float
    rpm: float
    throttle_pct: float
    brake: float


class ReplayOBDSource:
    """Replays a downloaded public OBD-II dataset through the same
    advance()/current_value() interface as MockOBDSource.

    The source CSV (a public Kaggle OBD-II vehicle diagnostics dataset,
    exp1_14drivers_14cars_dailyRoutes.csv) logs many separate trips per
    vehicle at an irregular sampling rate, spanning weeks, with ENGINE_RPM
    and THROTTLE_POS missing on the large majority of rows overall. Rather
    than resample or fabricate data, this splits one vehicle's rows into
    trips (a gap over trip_gap_s starts a new trip), picks the single trip
    with the best coverage of the fields this harness needs, forward-fills
    the rare remaining gaps within that trip, and replays its real recorded
    timestamps as-is -- irregular spacing included -- so the parsing logic
    gets exercised against real-shaped data, per the project spec.

    The dataset has no brake signal at all. The "brake" value this class
    reports is a documented heuristic -- throttle below the trip's own
    brake_throttle_percentile (throttle position sensor baselines vary a
    lot between vehicles, so a fixed absolute cutoff isn't meaningful; this
    dataset's car1, for example, never reports throttle below 15% even at
    a dead stop) while decelerating faster than brake_decel_rate_kph_s --
    not a real measurement, and it should never be treated as ground truth.
    """

    REQUIRED_FIELDS = ("SPEED", "ENGINE_RPM", "THROTTLE_POS")

    def __init__(
        self,
        csv_path: str | Path,
        vehicle_id: str = "car1",
        trip_gap_s: float = 60.0,
        brake_throttle_percentile: float = 0.4,
        brake_decel_rate_kph_s: float = 1.0,
    ):
        self.csv_path = Path(csv_path)
        self.vehicle_id = vehicle_id
        self._rows = self._load_best_trip(trip_gap_s, brake_throttle_percentile, brake_decel_rate_kph_s)
        if not self._rows:
            raise ValueError(
                f"No usable trip found for vehicle_id={vehicle_id!r} in {self.csv_path}"
            )
        self._index = -1
        self.timestamp = 0.0

    def _load_best_trip(
        self, trip_gap_s: float, brake_throttle_percentile: float, brake_decel_rate_kph_s: float
    ) -> list[_ReplayRow]:
        raw_rows: list[dict] = []
        with self.csv_path.open(newline="", encoding="utf-8", errors="replace") as f:
            for record in csv.DictReader(f):
                if record.get("VEHICLE_ID") != self.vehicle_id:
                    continue
                if not record.get("TIMESTAMP", "").strip():
                    continue
                raw_rows.append(record)
        raw_rows.sort(key=lambda r: int(r["TIMESTAMP"]))

        trips: list[list[dict]] = []
        current: list[dict] = []
        prev_ts: float | None = None
        for record in raw_rows:
            ts = int(record["TIMESTAMP"]) / 1000.0
            if prev_ts is not None and ts - prev_ts > trip_gap_s:
                trips.append(current)
                current = []
            current.append(record)
            prev_ts = ts
        if current:
            trips.append(current)
        if not trips:
            return []

        def coverage(trip: list[dict]) -> int:
            return sum(
                1
                for r in trip
                if all(_parse_numeric(r.get(field, "")) is not None for field in self.REQUIRED_FIELDS)
            )

        best_trip = max(trips, key=coverage)

        # Pass 1: parse speed/rpm/throttle, forward-filling the rare gaps.
        timestamps: list[float] = []
        speeds: list[float] = []
        rpms: list[float] = []
        throttles: list[float] = []
        last_rpm: float | None = None
        last_throttle: float | None = None
        for record in best_trip:
            speed = _parse_numeric(record.get("SPEED", ""))
            rpm = _parse_numeric(record.get("ENGINE_RPM", ""))
            throttle = _parse_numeric(record.get("THROTTLE_POS", ""))
            if speed is None:
                continue
            rpm = rpm if rpm is not None else last_rpm
            throttle = throttle if throttle is not None else last_throttle
            if rpm is None or throttle is None:
                continue  # can't forward-fill before the first real reading
            last_rpm, last_throttle = rpm, throttle
            timestamps.append(int(record["TIMESTAMP"]) / 1000.0)
            speeds.append(speed)
            rpms.append(rpm)
            throttles.append(throttle)

        if not speeds:
            return []

        # Pass 2: derive the brake heuristic from this trip's own throttle
        # distribution and each sample's actual (irregular) dt.
        sorted_throttles = sorted(throttles)
        idx = min(len(sorted_throttles) - 1, int(brake_throttle_percentile * len(sorted_throttles)))
        throttle_cutoff = sorted_throttles[idx]

        parsed: list[_ReplayRow] = []
        for i in range(len(speeds)):
            braking = 0.0
            if i > 0:
                dt = timestamps[i] - timestamps[i - 1]
                decel_rate = (speeds[i - 1] - speeds[i]) / dt if dt > 0 else 0.0
                if throttles[i] <= throttle_cutoff and decel_rate >= brake_decel_rate_kph_s:
                    braking = 1.0
            parsed.append(
                _ReplayRow(
                    timestamp_s=timestamps[i], speed_kph=speeds[i], rpm=rpms[i], throttle_pct=throttles[i], brake=braking
                )
            )

        if parsed:
            t0 = parsed[0].timestamp_s
            for row in parsed:
                row.timestamp_s -= t0

        return parsed

    def advance(self) -> None:
        if self._index + 1 >= len(self._rows):
            raise ReplayExhausted(
                f"replay of vehicle_id={self.vehicle_id!r} exhausted after {len(self._rows)} samples"
            )
        self._index += 1
        self.timestamp = self._rows[self._index].timestamp_s

    def current_value(self, command: OBDCommand) -> float:
        row = self._rows[self._index]
        if command is OBDCommand.SPEED:
            return row.speed_kph
        if command is OBDCommand.RPM:
            return row.rpm
        if command is OBDCommand.THROTTLE_POS:
            return row.throttle_pct
        if command is OBDCommand.BRAKE:
            return row.brake
        raise ValueError(f"Unsupported command: {command!r}")

    def __len__(self) -> int:
        return len(self._rows)


class OBDReader:
    """Unified OBD-II ingestion interface.

    OBDReader(mode="mock") requires no hardware and is safe to unit test.
    OBDReader(mode="replay") replays a downloaded public dataset (pass
    csv_path, optionally vehicle_id) through the identical interface.
    OBDReader(mode="live") is reserved for real Bluetooth OBD-II hardware
    and isn't implemented yet (see project spec: "What NOT to build yet").
    """

    def __init__(
        self,
        mode: str = "mock",
        seed: int = 0,
        hz: float = 10.0,
        csv_path: str | Path | None = None,
        vehicle_id: str = "car1",
    ):
        self.mode = mode
        self.hz = hz
        if mode == "mock":
            self._source = MockOBDSource(seed=seed, hz=hz)
        elif mode == "replay":
            if csv_path is None:
                raise ValueError("OBDReader(mode='replay') requires csv_path")
            self._source = ReplayOBDSource(csv_path=csv_path, vehicle_id=vehicle_id)
        elif mode == "live":
            raise NotImplementedError(
                "OBDReader(mode='live') requires real Bluetooth OBD-II hardware, "
                "which hasn't been attached yet -- see project spec."
            )
        else:
            raise ValueError(f"Unknown OBDReader mode: {mode!r}")
        # Lazily advanced on first use (by next_sample() or a standalone
        # query()), not here -- advancing eagerly in __init__ *and* in
        # next_sample() would silently skip the very first sample.
        self._started = False

    @property
    def total_samples(self) -> int | None:
        """Number of samples available, or None for an unbounded source (mock)."""
        try:
            return len(self._source)
        except TypeError:
            return None

    def next_sample(self) -> dict[OBDCommand, OBDResponse]:
        """Advance one time step and return readings for every supported command."""
        self._source.advance()
        self._started = True
        return {command: self.query(command) for command in SUPPORTED_COMMANDS}

    def query(self, command: OBDCommand) -> OBDResponse:
        """Read a single command from the current sample.

        Mirrors python-obd's connection.query(command) -- a single-PID read
        against whatever the vehicle's state currently is, without itself
        advancing that state. If called before any next_sample(), advances
        once to load the first sample.
        """
        if not self._started:
            self._source.advance()
            self._started = True
        value = self._source.current_value(command)
        return OBDResponse(command=command, value=value, unit=_UNITS[command], timestamp=self._source.timestamp)

    def stream(self, n_samples: int) -> Iterator[dict[OBDCommand, OBDResponse]]:
        """Yield up to `n_samples` consecutive full samples.

        A finite source (replay) stops early, yielding fewer than
        n_samples, once its recorded data is exhausted -- it does not
        raise, matching what happens when a real recording simply ends.
        """
        for _ in range(n_samples):
            try:
                yield self.next_sample()
            except ReplayExhausted:
                return
