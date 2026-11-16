import pandas as pd
import pytest

from critic.leakage import (
    assert_label_seal,
    assert_no_lookahead,
    assert_source_consistency,
)


def test_future_input_is_rejected():
    cutoff = pd.Timestamp("2024-12-13")
    with pytest.raises(ValueError, match="input timestamp exceeds cutoff"):
        assert_no_lookahead(
            [pd.Timestamp("2024-12-12"), pd.Timestamp("2024-12-16")],
            [pd.Timestamp("2024-12-17")],
            cutoff,
        )


def test_label_at_or_before_cutoff_is_rejected():
    cutoff = pd.Timestamp("2024-12-13")
    with pytest.raises(ValueError, match="label timestamp must be after cutoff"):
        assert_no_lookahead(
            [pd.Timestamp("2024-12-12"), pd.Timestamp("2024-12-13")],
            [pd.Timestamp("2024-12-13")],
            cutoff,
        )


def test_2025_label_is_sealed_during_development():
    with pytest.raises(RuntimeError, match="2025 labels are sealed"):
        assert_label_seal(pd.Timestamp("2025-01-02"), allow_2025_labels=False)


def test_2025_label_allowed_only_when_explicitly_unsealed():
    assert_label_seal(pd.Timestamp("2025-01-02"), allow_2025_labels=True)


def test_formal_run_rejects_mixed_sources():
    with pytest.raises(ValueError, match="single Tushare source"):
        assert_source_consistency(["tushare", "eastmoney"])
