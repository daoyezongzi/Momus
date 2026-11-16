import pytest

from critic.experiment import freeze_config, reserve_sealed_output, validate_phase


def test_dev_phase_cannot_enable_2025_labels():
    with pytest.raises(RuntimeError, match="development phase"):
        validate_phase("dev", allow_2025_labels=True, frozen_config=None)


def test_sealed_phase_requires_matching_frozen_hash(tmp_path):
    config = tmp_path / "frozen.json"
    digest = freeze_config({"pred_len": 10, "lags": [1, 3, 5]}, config)
    assert validate_phase("sealed-2025", True, config, expected_hash=digest) == digest


def test_sealed_result_cannot_be_overwritten(tmp_path):
    output = tmp_path / "sealed"
    reserve_sealed_output(output, "abc")
    with pytest.raises(RuntimeError, match="already exists"):
        reserve_sealed_output(output, "abc")


def test_sealed_phase_requires_explicit_permission_and_config(tmp_path):
    with pytest.raises(RuntimeError, match="requires explicit"):
        validate_phase("sealed-2025", False, tmp_path / "missing.json", expected_hash="abc")
