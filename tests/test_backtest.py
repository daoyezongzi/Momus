import numpy as np
import pandas as pd
import pytest

from critic.backtest import (
    backtest_metrics,
    compute_signal,
    format_backtest_table,
    heuristic_candidate_index,
    holding_period_return,
    oracle_candidate_index,
    trade_pnl,
)


def _prices(t_close=100.0, next_open=102.0, exit_close=110.0):
    x_df = pd.DataFrame({"close": [95.0, t_close]})
    y_true = np.zeros((2, 6), dtype=np.float64)
    y_true[0, 0] = next_open
    y_true[-1, 3] = exit_close
    return x_df, y_true


def test_compute_signal_inverse_transforms_final_close():
    final_norm = np.zeros((2, 6), dtype=np.float64)
    x_mean = np.array([0, 0, 0, 100, 0, 0], dtype=np.float64)
    x_std = np.array([1, 1, 1, 10, 1, 1], dtype=np.float64)
    final_norm[-1, 3] = 0.5
    assert compute_signal(final_norm, x_mean, x_std, entry_price=100.0) == 1
    final_norm[-1, 3] = -0.5
    assert compute_signal(final_norm, x_mean, x_std, entry_price=100.0) == -1
    final_norm[-1, 3] = 0.0
    assert compute_signal(final_norm, x_mean, x_std, entry_price=100.0) == 0
    assert compute_signal(final_norm, x_mean, x_std, entry_price=0.0) == 0
    assert compute_signal(final_norm, x_mean, x_std, entry_price=np.nan) == 0


def test_holding_period_return_uses_close_or_uncosted_next_open_entry():
    x_df, y_true = _prices()
    assert np.isclose(holding_period_return(x_df, y_true, "close", 0.0), 0.10)
    # Cost validation belongs here, but the returned holding return itself is uncosted.
    assert np.isclose(holding_period_return(x_df, y_true, "next_open", 0.00075), (110.0 - 102.0) / 102.0)


def test_holding_period_return_rejects_negative_cost_and_invalid_entry_mode():
    x_df, y_true = _prices()
    with pytest.raises(ValueError, match="cost must be non-negative"):
        holding_period_return(x_df, y_true, "close", -0.00075)
    with pytest.raises(ValueError, match="entry_mode must be 'close' or 'next_open'"):
        holding_period_return(x_df, y_true, "invalid", 0.0)


def test_holding_period_return_rejects_invalid_entry_or_exit():
    x_df, y_true = _prices(t_close=0.0)
    assert np.isnan(holding_period_return(x_df, y_true, "close", 0.0))
    x_df, y_true = _prices(next_open=np.nan)
    assert np.isnan(holding_period_return(x_df, y_true, "next_open", 0.0))
    x_df, y_true = _prices(exit_close=np.inf)
    assert np.isnan(holding_period_return(x_df, y_true, "close", 0.0))


def test_trade_pnl_handles_direction_modes_and_cost():
    assert np.isclose(trade_pnl(1, 0.10, long_only=False, cost=0.0), 0.10)
    assert np.isclose(trade_pnl(-1, 0.10, long_only=False, cost=0.0), -0.10)
    assert np.isclose(trade_pnl(1, 0.10, long_only=True, cost=0.00075), 0.09925)
    assert trade_pnl(-1, 0.10, long_only=True, cost=0.00075) == 0.0
    assert trade_pnl(0, 0.10, long_only=False, cost=0.00075) == 0.0


def test_backtest_metrics_matches_known_values_and_filters_nan():
    pnl = np.array([0.10, -0.05, 0.0, np.nan])
    baseline = np.array([0.05, -0.02, 0.0, 0.01])
    metrics = backtest_metrics(pnl, pred_len=10, baseline_pnl=baseline)
    valid = np.array([0.10, -0.05, 0.0])
    excess = valid - np.array([0.05, -0.02, 0.0])
    assert metrics["count"] == 3
    assert np.isclose(metrics["mean_return"], valid.mean())
    assert np.isclose(metrics["win_rate"], 1 / 3)
    assert np.isclose(metrics["std_return"], valid.std(ddof=0))
    assert np.isclose(metrics["sharpe"], valid.mean() / valid.std(ddof=0) * np.sqrt(252 / 10))
    assert np.isclose(metrics["information_ratio"], excess.mean() / excess.std(ddof=0))
    assert np.isclose(metrics["cumulative_return"], valid.sum())


def test_backtest_metrics_handles_empty_and_zero_variance():
    empty = backtest_metrics(np.array([np.nan]), pred_len=10)
    assert empty["count"] == 0
    assert np.isnan(empty["mean_return"])
    assert np.isnan(empty["sharpe"])
    constant = backtest_metrics(np.array([0.01, 0.01]), pred_len=10)
    assert np.isnan(constant["sharpe"])


def test_backtest_metrics_validates_baseline_shape_and_zero_excess():
    with pytest.raises(ValueError):
        backtest_metrics(np.array([0.1, 0.2]), 10, np.array([0.1]))
    same = backtest_metrics(np.array([0.1, 0.2]), 10, np.array([0.1, 0.2]))
    assert same["information_ratio"] == 0.0


def test_backtest_metrics_returns_nan_for_nonzero_constant_excess():
    metrics = backtest_metrics(
        np.array([0.2, 0.3]), pred_len=10, baseline_pnl=np.array([0.1, 0.2])
    )
    assert np.isnan(metrics["information_ratio"])


def test_backtest_metrics_rejects_non_positive_prediction_length():
    with pytest.raises(ValueError, match="pred_len must be positive"):
        backtest_metrics(np.array([0.1]), pred_len=0)






def test_heuristic_candidate_index_most_bullish_uses_largest_final_close():
    paths = np.zeros((3, 2, 6), dtype=np.float32)
    paths[:, -1, 3] = [1.1, 1.4, 0.9]

    assert heuristic_candidate_index(paths, mode="most_bullish", entry_price=1.0) == 1


def test_heuristic_candidate_index_lowest_volatility_uses_tightest_range():
    paths = np.zeros((3, 3, 6), dtype=np.float32)
    paths[0, :, 1] = [1.2, 1.3, 1.2]
    paths[0, :, 2] = [0.8, 0.7, 0.8]
    paths[1, :, 1] = [1.05, 1.06, 1.05]
    paths[1, :, 2] = [0.95, 0.94, 0.95]
    paths[2, :, 1] = [1.4, 1.5, 1.4]
    paths[2, :, 2] = [0.6, 0.5, 0.6]

    assert heuristic_candidate_index(paths, mode="lowest_volatility") == 1


def test_heuristic_candidate_index_random_is_seeded():
    paths = np.zeros((5, 2, 6), dtype=np.float32)

    assert heuristic_candidate_index(paths, mode="random", seed=7) == heuristic_candidate_index(paths, mode="random", seed=7)

    paths = np.zeros((3, 2, 6), dtype=np.float32)
    y_true = np.zeros((2, 6), dtype=np.float32)
    paths[0, :, 3] = [0.0, 0.5]
    paths[1, :, 3] = [0.0, 0.1]
    paths[2, :, 3] = [0.0, 0.0]
    y_true[:, 3] = [0.0, 0.1]

    assert oracle_candidate_index(paths, y_true, mode="path") == 1


def test_oracle_candidate_index_direction_mode_prefers_matching_direction():
    paths = np.zeros((3, 2, 6), dtype=np.float32)
    paths[:, -1, 3] = [0.8, 1.2, 1.5]
    y_true = np.zeros((2, 6), dtype=np.float32)
    y_true[-1, 3] = 1.1

    assert oracle_candidate_index(paths, y_true, mode="direction", entry_price=1.0) == 1


def test_oracle_candidate_index_trading_mode_prefers_best_long_only_pnl():
    paths = np.zeros((3, 2, 6), dtype=np.float32)
    paths[:, -1, 3] = [0.8, 1.1, 1.4]
    y_true = np.zeros((2, 6), dtype=np.float32)
    y_true[0, 0] = 1.0
    y_true[-1, 3] = 1.2

    assert oracle_candidate_index(
        paths,
        y_true,
        mode="trading",
        entry_price=1.0,
        holding_return=0.2,
        long_only=True,
        cost=0.0,
    ) == 1


def test_format_backtest_table_marks_theoretical_short_result():
    metrics = {
        "count": 2,
        "mean_return": 0.01,
        "win_rate": 0.5,
        "std_return": 0.02,
        "sharpe": 2.51,
        "information_ratio": 0.3,
        "cumulative_return": 0.02,
    }
    results = {
        "net_long_short": {
            "simple_mean": {
                "pnl": np.array([0.02, 0.0]),
                "metrics": metrics,
                "invalid_count": 1,
            },
            "critic_weighted": {
                "pnl": np.array([0.03, np.nan]),
                "metrics": metrics,
                "invalid_count": 7,
            },
        }
    }
    text = format_backtest_table(results)
    assert "net_long_short" in text
    assert "理论参考" in text
    assert "simple_mean" in text
    assert "1.0000%" in text
    assert "invalid" in text.splitlines()[0]
    assert text.splitlines()[2].endswith("| 1")
    assert text.splitlines()[3].endswith("| 7")
