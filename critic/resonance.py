import numpy as np
import pandas as pd

from .schemas import ResonanceFeatures


_PRICE_DIM = 6
_VOLUME_DIM = 5
_SYNC_DIM = 3


def _truncate(df: pd.DataFrame, cutoff_T) -> pd.DataFrame:
    return df.loc[df.index <= pd.Timestamp(cutoff_T)].copy()


def _log_returns(series: pd.Series) -> pd.Series:
    valid = series.astype(float).where(series.astype(float) > 0)
    return np.log(valid).diff()


def _aligned_returns(frames: dict[str, pd.DataFrame], column: str, cutoff_T, lookback: int):
    values = {}
    missing = {}
    for code, df in frames.items():
        sliced = _truncate(df, cutoff_T)[column].tail(lookback + 1)
        values[code] = _log_returns(sliced)
        missing[code] = sliced.tail(lookback).isna() | (sliced.tail(lookback).astype(float) <= 0)
    returns = pd.DataFrame(values).tail(lookback)
    missing_mask = pd.DataFrame(missing).reindex(returns.index).fillna(True)
    return returns, missing_mask


def _safe_mean(value) -> float:
    if value is None:
        return 0.0
    try:
        if pd.isna(value):
            return 0.0
    except TypeError:
        pass
    return float(value)


def _safe_corr(left: pd.Series, right: pd.Series) -> float:
    paired = pd.concat([left, right], axis=1).dropna()
    if len(paired) < 3:
        return 0.0
    corr = paired.corr().iloc[0, 1]
    return _safe_mean(corr)


def build_resonance_features(
    target_df: pd.DataFrame,
    related_dfs: dict[str, pd.DataFrame],
    cutoff_T,
    lookback: int,
    lags=(1, 3, 5),
) -> ResonanceFeatures:
    if any(lag <= 0 for lag in lags):
        raise ValueError("lead-lag values must be positive")

    cutoff = pd.Timestamp(cutoff_T)
    target = _truncate(target_df, cutoff)
    related = {code: _truncate(df, cutoff) for code, df in related_dfs.items()}

    close_returns, _ = _aligned_returns(related, "close", cutoff, lookback)
    volume_returns, volume_missing = _aligned_returns(related, "volume", cutoff, lookback)
    target_close = _log_returns(target["close"]).tail(lookback)

    direction = np.sign(close_returns)
    up_ratio = _safe_mean((direction > 0).mean(axis=1).mean()) if not close_returns.empty else 0.0
    down_ratio = _safe_mean((direction < 0).mean(axis=1).mean()) if not close_returns.empty else 0.0
    chain_return = _safe_mean(close_returns.mean(axis=1).mean()) if not close_returns.empty else 0.0
    dispersion = _safe_mean(close_returns.std(axis=1).mean()) if close_returns.shape[1] > 1 else 0.0
    agreement = (
        _safe_mean(direction.eq(np.sign(target_close), axis=0).mean().mean())
        if not close_returns.empty
        else 0.0
    )
    same_direction = max(up_ratio, down_ratio)

    volume_direction = np.sign(volume_returns)
    volume_features = np.array(
        [
            _safe_mean((volume_direction > 0).mean().mean()) if not volume_returns.empty else 0.0,
            _safe_mean(volume_direction.nunique(axis=1).eq(1).mean()) if not volume_returns.empty else 0.0,
            _safe_mean(((direction > 0) & (volume_direction > 0)).mean().mean()) if not volume_returns.empty else 0.0,
            _safe_mean(volume_returns.mean(axis=1).mean()) if not volume_returns.empty else 0.0,
            _safe_mean(volume_returns.std(axis=1).mean()) if volume_returns.shape[1] > 1 else 0.0,
        ],
        dtype=np.float32,
    )

    chain_series = close_returns.mean(axis=1) if not close_returns.empty else pd.Series(dtype=float)
    lead_lag = []
    for lag in lags:
        lead_lag.append(_safe_corr(chain_series.shift(lag), target_close))

    volume_valid = not bool(volume_missing.any().any()) if not volume_missing.empty else False
    mask = np.concatenate(
        [
            np.ones(_PRICE_DIM, dtype=bool),
            np.full(_VOLUME_DIM, volume_valid, dtype=bool),
            np.ones(_SYNC_DIM + len(lags), dtype=bool),
        ]
    )

    return ResonanceFeatures(
        price=np.nan_to_num(
            np.array(
                [up_ratio, down_ratio, chain_return, dispersion, agreement, same_direction],
                dtype=np.float32,
            )
        ),
        volume=np.nan_to_num(volume_features),
        synchronous=np.nan_to_num(np.array([agreement, up_ratio, down_ratio], dtype=np.float32)),
        lead_lag=np.nan_to_num(np.asarray(lead_lag, dtype=np.float32)),
        mask=mask,
    )
