import pytest

from ingestion.obd_reader import SUPPORTED_COMMANDS, OBDCommand, OBDReader, OBDResponse


def test_next_sample_returns_all_supported_commands():
    reader = OBDReader(mode="mock", seed=1)
    sample = reader.next_sample()
    assert set(sample.keys()) == set(SUPPORTED_COMMANDS)
    for command, response in sample.items():
        assert response.command is command
        assert not response.is_null()


def test_query_matches_python_obd_style_single_command_read():
    reader = OBDReader(mode="mock", seed=1)
    response = reader.query(OBDCommand.RPM)
    assert response.command is OBDCommand.RPM
    assert response.unit == "rpm"
    assert isinstance(response.value, float)


def test_values_stay_within_plausible_ranges_over_many_samples():
    reader = OBDReader(mode="mock", seed=7, hz=10.0)
    for sample in reader.stream(500):
        assert 0.0 <= sample[OBDCommand.SPEED].value <= 140.0
        assert sample[OBDCommand.RPM].value >= 0.0
        assert 0.0 <= sample[OBDCommand.THROTTLE_POS].value <= 100.0
        assert sample[OBDCommand.BRAKE].value in (0.0, 1.0)


def test_timestamps_increase_monotonically():
    reader = OBDReader(mode="mock", seed=3, hz=10.0)
    timestamps = [sample[OBDCommand.SPEED].timestamp for sample in reader.stream(50)]
    assert timestamps == sorted(timestamps)
    assert len(set(timestamps)) == len(timestamps)
    assert timestamps[1] - timestamps[0] == pytest.approx(0.1)


def test_same_seed_is_deterministic():
    reader_a = OBDReader(mode="mock", seed=99)
    reader_b = OBDReader(mode="mock", seed=99)
    samples_a = [s[OBDCommand.SPEED].value for s in reader_a.stream(30)]
    samples_b = [s[OBDCommand.SPEED].value for s in reader_b.stream(30)]
    assert samples_a == samples_b


def test_different_seeds_diverge():
    reader_a = OBDReader(mode="mock", seed=1)
    reader_b = OBDReader(mode="mock", seed=2)
    samples_a = [s[OBDCommand.SPEED].value for s in reader_a.stream(30)]
    samples_b = [s[OBDCommand.SPEED].value for s in reader_b.stream(30)]
    assert samples_a != samples_b


def test_stream_yields_exactly_n_samples():
    reader = OBDReader(mode="mock", seed=0)
    samples = list(reader.stream(15))
    assert len(samples) == 15


def test_live_mode_not_implemented():
    with pytest.raises(NotImplementedError):
        OBDReader(mode="live")


def test_unknown_mode_raises_value_error():
    with pytest.raises(ValueError):
        OBDReader(mode="bogus")


def test_response_is_null_when_value_is_none():
    response = OBDResponse(command=OBDCommand.RPM, value=None, unit="rpm", timestamp=0.0)
    assert response.is_null()
