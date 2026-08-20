from pathlib import Path

import pytest

from ingestion.obd_reader import OBDCommand, OBDReader, ReplayExhausted, ReplayOBDSource

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "exp1_14drivers_14cars_dailyRoutes.csv"

pytestmark = pytest.mark.skipif(not DATASET_PATH.exists(), reason="dataset not present in data/")


def test_replay_reader_yields_all_available_samples_no_off_by_one():
    reader = OBDReader(mode="replay", csv_path=DATASET_PATH, vehicle_id="car1")
    total = reader.total_samples
    assert total is not None and total > 100

    samples = list(reader.stream(total + 10))
    assert len(samples) == total  # stops cleanly, no error, no dropped first sample


def test_replay_first_sample_matches_source_first_row_directly():
    source = ReplayOBDSource(csv_path=DATASET_PATH, vehicle_id="car1")
    first_row = source._rows[0]

    reader = OBDReader(mode="replay", csv_path=DATASET_PATH, vehicle_id="car1")
    first_sample = reader.next_sample()
    assert first_sample[OBDCommand.SPEED].value == pytest.approx(first_row.speed_kph)
    assert first_sample[OBDCommand.SPEED].timestamp == pytest.approx(first_row.timestamp_s)


def test_replay_values_stay_within_plausible_ranges():
    reader = OBDReader(mode="replay", csv_path=DATASET_PATH, vehicle_id="car1")
    for sample in reader.stream(reader.total_samples):
        assert 0.0 <= sample[OBDCommand.SPEED].value <= 200.0
        assert sample[OBDCommand.RPM].value >= 0.0
        assert 0.0 <= sample[OBDCommand.THROTTLE_POS].value <= 100.0
        assert sample[OBDCommand.BRAKE].value in (0.0, 1.0)


def test_replay_timestamps_are_monotonic_and_start_at_zero():
    reader = OBDReader(mode="replay", csv_path=DATASET_PATH, vehicle_id="car1")
    timestamps = [s[OBDCommand.SPEED].timestamp for s in reader.stream(reader.total_samples)]
    assert timestamps[0] == pytest.approx(0.0)
    assert timestamps == sorted(timestamps)


def test_replay_brake_heuristic_fires_a_plausible_fraction_of_the_time():
    reader = OBDReader(mode="replay", csv_path=DATASET_PATH, vehicle_id="car1")
    samples = list(reader.stream(reader.total_samples))
    brake_rate = sum(1 for s in samples if s[OBDCommand.BRAKE].value == 1.0) / len(samples)
    # Not a real measurement (documented heuristic) -- just check it isn't
    # degenerate (never firing, or firing on nearly everything).
    assert 0.01 < brake_rate < 0.5


def test_source_advance_raises_replay_exhausted_past_the_end():
    source = ReplayOBDSource(csv_path=DATASET_PATH, vehicle_id="car1")
    for _ in range(len(source)):  # consumes every row, index -1 -> len-1
        source.advance()
    with pytest.raises(ReplayExhausted):
        source.advance()


def test_unknown_vehicle_id_raises_value_error():
    with pytest.raises(ValueError):
        ReplayOBDSource(csv_path=DATASET_PATH, vehicle_id="not_a_real_car")


def test_replay_mode_requires_csv_path():
    with pytest.raises(ValueError):
        OBDReader(mode="replay")


def test_parses_comma_decimal_and_percent_fields():
    from ingestion.obd_reader import _parse_numeric

    assert _parse_numeric("48,60%") == pytest.approx(48.60)
    assert _parse_numeric("25%") == pytest.approx(25.0)
    assert _parse_numeric("1,4") == pytest.approx(1.4)
    assert _parse_numeric("") is None
    assert _parse_numeric("   ") is None
