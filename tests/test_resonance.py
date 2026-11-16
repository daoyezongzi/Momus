import numpy as np
import pandas as pd
import pytest

from critic.resonance import build_resonance_features


def frame(close, volume):
    index = pd.date_range("2024-01-01", periods=len(close), freq="B")
    return pd.DataFrame({"close": close, "volume": volume}, index=index)


def test_price_and_volume_features_are_separate_and_finite():
    target = frame([10, 11, 12, 11, 12, 13], [100, 110, 120, 100, 130, 150])
    related = {
        "a": frame([20, 21, 22, 21, 22, 23], [200, 210, 230, 220, 250, 280]),
        "b": frame([30, 29, 30, 31, 32, 33], [300, 290, 310, 320, 340, 360]),
    }
    features = build_resonance_features(
        target, related, target.index[-1], lookback=5, lags=(1, 3, 5)
    )
    assert features.price.shape == (6,)
    assert features.volume.shape == (5,)
    assert features.synchronous.shape == (3,)
    assert features.lead_lag.shape == (3,)
    assert np.isfinite(features.vector("price_volume")).all()


def test_zero_volume_is_missing_not_zero_change():
    target = frame([10, 11, 12], [100, 0, 120])
    related = {"a": frame([20, 21, 22], [200, 0, 220])}
    features = build_resonance_features(
        target, related, target.index[-1], lookback=3, lags=(1,)
    )
    assert not features.mask.all()


def test_negative_lag_is_rejected():
    target = frame([10, 11, 12], [100, 110, 120])
    with pytest.raises(ValueError, match="positive"):
        build_resonance_features(target, {}, target.index[-1], 3, lags=(-1,))


def test_future_rows_are_ignored_before_feature_calculation():
    target = frame([10, 11, 12, 13], [100, 110, 120, 130])
    related = {"a": frame([20, 21, 22, 999], [200, 210, 220, 999])}
    features_with_future = build_resonance_features(
        target, related, target.index[-2], lookback=3, lags=(1,)
    )
    related_truncated = {"a": related["a"].iloc[:-1]}
    features_without_future = build_resonance_features(
        target.iloc[:-1], related_truncated, target.index[-2], lookback=3, lags=(1,)
    )
    assert np.allclose(features_with_future.vector("price_volume"), features_without_future.vector("price_volume"))
