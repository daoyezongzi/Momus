"""Event-style holding-period return and signal utilities."""

import numpy as np


CLOSE_COL = 3
_NORMALIZE_EPS = 1e-5


def compute_signal(final_norm, x_mean, x_std, entry_price):
    """Return the sign of the inverse-transformed final predicted close."""
    try:
        final_close = final_norm[-1, CLOSE_COL]
        predicted_close = final_close * (x_std[CLOSE_COL] + _NORMALIZE_EPS) + x_mean[CLOSE_COL]
    except (IndexError, TypeError):
        return 0

    if not np.isfinite(predicted_close) or not np.isfinite(entry_price) or entry_price <= 0:
        return 0
    return int(np.sign(predicted_close - entry_price))


def holding_period_return(x_df, y_true, entry_mode, cost):
    """Calculate an uncosted holding-period return from entry through exit close."""
    if cost < 0:
        raise ValueError("cost must be non-negative")
    if entry_mode == "close":
        entry = x_df["close"].iloc[-1]
    elif entry_mode == "next_open":
        entry = y_true[0, 0]
    else:
        raise ValueError("entry_mode must be 'close' or 'next_open'")

    exit_price = y_true[-1, CLOSE_COL]
    if not np.isfinite(entry) or entry <= 0 or not np.isfinite(exit_price) or exit_price <= 0:
        return np.nan
    return (exit_price - entry) / entry


def trade_pnl(direction, holding_return, long_only, cost):
    """Apply a directional trading decision and deduct cost only for actual trades."""
    if direction == 0 or (long_only and direction < 0):
        return 0.0
    if not np.isfinite(holding_return):
        return np.nan
    return direction * holding_return - cost


def heuristic_candidate_index(paths, mode="most_bullish", entry_price=None, seed=0):
    """Return a candidate index from a non-learned rule baseline."""
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 3 or len(paths) == 0:
        raise ValueError("paths must be a non-empty (N, T, F) array")

    if mode == "random":
        return int(np.random.default_rng(seed).integers(0, len(paths)))
    if mode == "most_bullish":
        if entry_price is None:
            entry_price = 0.0
        return int(np.nanargmax(paths[:, -1, CLOSE_COL] - entry_price))
    if mode == "lowest_volatility":
        ranges = np.nanmax(paths[:, :, 1], axis=1) - np.nanmin(paths[:, :, 2], axis=1)
        return int(np.nanargmin(ranges))
    raise ValueError("mode must be 'random', 'most_bullish', or 'lowest_volatility'")


def oracle_candidate_index(
    paths,
    y_true,
    mode="path",
    entry_price=None,
    holding_return=None,
    long_only=True,
    cost=0.0,
):
    """Return the best candidate index under an ex-post oracle criterion.

    This is an upper-bound diagnostic, not a tradable strategy: it may use y_true
    or realized holding_return to measure the fixed candidate set's ceiling.
    """
    paths = np.asarray(paths, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    if paths.ndim != 3 or len(paths) == 0:
        raise ValueError("paths must be a non-empty (N, T, F) array")

    if mode == "path":
        price_cols = [0, 1, 2, 3]
        errors = np.mean((paths[:, :, price_cols] - y_true[None, :, price_cols]) ** 2, axis=(1, 2))
        return int(np.nanargmin(errors))

    if entry_price is None or not np.isfinite(entry_price):
        raise ValueError("entry_price is required for direction/trading oracle")
    predicted_direction = np.sign(paths[:, -1, CLOSE_COL] - entry_price).astype(int)

    if mode == "direction":
        true_direction = int(np.sign(y_true[-1, CLOSE_COL] - entry_price))
        mismatch = (predicted_direction != true_direction).astype(int)
        magnitude = np.abs(paths[:, -1, CLOSE_COL] - entry_price)
        order_key = mismatch * 1e12 + magnitude
        return int(np.nanargmin(order_key))

    if mode == "trading":
        if holding_return is None or not np.isfinite(holding_return):
            raise ValueError("holding_return is required for trading oracle")
        pnl = np.array([
            trade_pnl(int(direction), holding_return, long_only=long_only, cost=cost)
            for direction in predicted_direction
        ], dtype=float)
        return int(np.nanargmax(pnl))

    raise ValueError("mode must be 'path', 'direction', or 'trading'")


REGIME_LABELS = {
    "gross_long_short": "毛·多空对称",
    "gross_long_only": "毛·只做多",
    "net_long_short": "净·多空对称（理论参考：A股裸卖空不现实）",
    "net_long_only": "净·只做多",
}


def _finite_values(values):
    """Return only finite values as a float array."""
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def backtest_metrics(pnl_array, pred_len, baseline_pnl=None):
    """Summarize finite holding-period PnL values and optional excess returns."""
    if pred_len <= 0:
        raise ValueError("pred_len must be positive")

    pnl = np.asarray(pnl_array, dtype=float)
    valid = _finite_values(pnl)
    nan = float("nan")
    metrics = {
        "count": len(valid),
        "mean_return": nan,
        "win_rate": nan,
        "std_return": nan,
        "sharpe": nan,
        "information_ratio": nan,
        "cumulative_return": nan,
    }
    if len(valid):
        mean_return = valid.mean()
        std_return = valid.std(ddof=0)
        metrics.update(
            mean_return=mean_return,
            win_rate=(valid > 0).mean(),
            std_return=std_return,
            sharpe=(mean_return / std_return * np.sqrt(252 / pred_len)) if std_return > 0 else nan,
            cumulative_return=valid.sum(),
        )

    if baseline_pnl is not None:
        baseline = np.asarray(baseline_pnl, dtype=float)
        if pnl.shape != baseline.shape:
            raise ValueError("baseline_pnl must have the same shape as pnl_array")
        valid_pair = np.isfinite(pnl) & np.isfinite(baseline)
        excess = pnl[valid_pair] - baseline[valid_pair]
        if len(excess):
            excess_std = excess.std(ddof=0)
            if np.allclose(excess, excess[0]):
                if np.allclose(excess, 0.0):
                    metrics["information_ratio"] = 0.0
            else:
                metrics["information_ratio"] = excess.mean() / excess_std

    return metrics


def _number(value, percentage=False):
    """Format a numeric metric with a stable representation for NaN."""
    if not np.isfinite(value):
        return "nan"
    return f"{value:.4%}" if percentage else f"{value:.4f}"


def format_backtest_table(results):
    """Format backtest results grouped by accounting regime and strategy."""
    lines = [
        "regime | label | strategy | N | 平均收益 | 胜率 | 标准差 | Sharpe | IR | 累计收益 | invalid",
        "--- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---:",
    ]
    for regime, strategies in results.items():
        label = REGIME_LABELS.get(regime, regime)
        for strategy, item in strategies.items():
            metrics = item["metrics"]
            invalid_count = item.get("invalid_count", 0)
            lines.append(
                f"{regime} | {label} | {strategy} | {metrics['count']} | "
                f"{_number(metrics['mean_return'], percentage=True)} | "
                f"{_number(metrics['win_rate'], percentage=True)} | "
                f"{_number(metrics['std_return'], percentage=True)} | "
                f"{_number(metrics['sharpe'])} | "
                f"{_number(metrics['information_ratio'])} | "
                f"{_number(metrics['cumulative_return'], percentage=True)} | "
                f"{invalid_count}"
            )
    return "\n".join(lines)
