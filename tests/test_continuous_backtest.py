"""单元测试：连续资金曲线回测核心函数。"""
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from critic.continuous_backtest import (
    build_non_overlapping_windows,
    calculate_drawdown_series,
    calculate_mdd,
    calculate_calmar,
    continuous_metrics,
    summarize_strategy_metrics,
)


@pytest.fixture
def sample_dataframe():
    """构造测试用的 DataFrame（100 天数据）。"""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    df = pd.DataFrame({
        'open': np.random.uniform(10, 20, 100),
        'high': np.random.uniform(20, 25, 100),
        'low': np.random.uniform(5, 10, 100),
        'close': np.random.uniform(10, 20, 100),
        'volume': np.random.randint(1000, 10000, 100),
        'amount': np.random.uniform(10000, 50000, 100),
    }, index=dates)
    return df


def test_build_non_overlapping_windows_basic(sample_dataframe):
    """测试基本的不重叠窗口构建。"""
    lookback = 30
    pred_len = 10
    windows = build_non_overlapping_windows(sample_dataframe, lookback, pred_len)

    # 100 天数据，lookback=30, pred_len=10
    # 第一个窗口: [0:30] hist + [30:40] future
    # 第二个窗口: [10:40] hist + [40:50] future
    # ...
    # 预期窗口数 = (100 - 30) // 10 = 7
    assert len(windows) == 7

    # 检查第一个窗口
    x_df, x_ts, y_ts, y_true = windows[0]
    assert len(x_df) == lookback
    assert len(y_ts) == pred_len
    assert y_true.shape[0] == pred_len
    assert y_true.shape[1] >= 4  # 至少有 OHLC

    # 检查窗口首尾相接（不重叠）
    assert x_ts.iloc[-1] < y_ts.iloc[0]  # 历史结束 < 未来开始

    # 第二个窗口的历史起点应该从第一个窗口的未来开始（步长=pred_len）
    if len(windows) > 1:
        x_df_2, x_ts_2, _, _ = windows[1]
        # 第二个窗口从索引 pred_len 开始，不是从第一个窗口的未来结束点
        # 实际上是固定步长滚动，不是首尾相接
        assert len(x_df_2) == lookback


def test_build_non_overlapping_windows_with_date_range(sample_dataframe):
    """测试带日期范围筛选的窗口构建。"""
    lookback = 30
    pred_len = 10
    windows = build_non_overlapping_windows(
        sample_dataframe,
        lookback,
        pred_len,
        start_date='2024-01-15',
        end_date='2024-03-01'
    )

    # 筛选后数据量减少，窗口数应该变少
    assert len(windows) < 7
    assert len(windows) > 0

    # 检查所有窗口在日期范围内
    for x_df, _, _, _ in windows:
        assert x_df.index[0] >= pd.to_datetime('2024-01-15')


def test_build_non_overlapping_windows_insufficient_data():
    """测试数据不足时返回空列表。"""
    dates = pd.date_range('2024-01-01', periods=20, freq='D')
    df = pd.DataFrame({
        'open': np.ones(20),
        'high': np.ones(20),
        'low': np.ones(20),
        'close': np.ones(20),
    }, index=dates)

    lookback = 30
    pred_len = 10
    windows = build_non_overlapping_windows(df, lookback, pred_len)

    # 总数据 20 < lookback + pred_len = 40
    assert len(windows) == 0


def test_calculate_drawdown_series_monotonic_increase():
    """测试单调上涨的净值序列（无回撤）。"""
    equity = np.array([100, 110, 120, 130, 140])
    drawdown = calculate_drawdown_series(equity)

    # 单调上涨，所有回撤应为 0
    assert np.allclose(drawdown, 0.0)


def test_calculate_drawdown_series_with_drawdown():
    """测试有回撤的净值序列。"""
    equity = np.array([100, 120, 110, 105, 115, 125])
    drawdown = calculate_drawdown_series(equity)

    # 峰值 120，后续跌到 105
    # 回撤 = (105 - 120) / 120 = -0.125
    assert drawdown[0] == 0.0  # 初始无回撤
    assert drawdown[1] == 0.0  # 新高
    assert drawdown[2] < 0.0   # 开始回撤
    assert drawdown[3] < drawdown[2]  # 回撤加深
    assert drawdown[4] > drawdown[3]  # 回撤收窄
    assert drawdown[5] == 0.0  # 创新高，回撤归零


def test_calculate_mdd():
    """测试最大回撤计算。"""
    # 手工构造已知 MDD 的序列
    equity = np.array([100, 120, 100, 80, 90, 110, 130])
    mdd = calculate_mdd(equity)

    # 峰值 120 → 谷底 80，回撤 (80-120)/120 = -33.33%
    expected_mdd = (80 - 120) / 120
    assert np.isclose(mdd, expected_mdd)


def test_calculate_mdd_no_drawdown():
    """测试无回撤情况。"""
    equity = np.array([100, 110, 120, 130])
    mdd = calculate_mdd(equity)
    assert mdd == 0.0


def test_calculate_calmar_normal():
    """测试正常情况下的 Calmar 计算。"""
    annual_return = 0.15
    mdd = -0.10
    calmar = calculate_calmar(annual_return, mdd)

    expected = 0.15 / 0.10
    assert np.isclose(calmar, expected)


def test_calculate_calmar_edge_cases():
    """测试 Calmar 边界情况。"""
    # MDD = 0（无回撤）
    calmar = calculate_calmar(0.15, 0.0)
    assert np.isnan(calmar)

    # MDD 为正（异常）
    calmar = calculate_calmar(0.15, 0.05)
    assert np.isnan(calmar)

    # 负收益
    calmar = calculate_calmar(-0.05, -0.10)
    assert np.isclose(calmar, -0.5)


def test_continuous_metrics_basic():
    """测试完整指标计算。"""
    # 构造简单净值序列
    equity = np.array([100, 110, 115, 120, 125])
    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    initial_capital = 100.0

    metrics = continuous_metrics(equity, dates.tolist(), initial_capital)

    # 基础检查
    assert metrics['final_equity'] == 125.0
    assert np.isclose(metrics['total_return'], 0.25)
    assert metrics['days'] == 4

    # 年化收益: (1.25)^(252/4) - 1
    expected_annual = (1.25) ** (252 / 4) - 1
    assert np.isclose(metrics['annual_return'], expected_annual)

    # MDD = 0（单调上涨）
    assert metrics['mdd'] == 0.0

    # Calmar 应为 NaN（MDD=0）
    assert np.isnan(metrics['calmar'])


def test_continuous_metrics_with_drawdown():
    """测试有回撤的指标计算。"""
    # 峰值 120 → 谷底 100 → 恢复到 130
    equity = np.array([100, 120, 110, 100, 115, 130])
    dates = pd.date_range('2024-01-01', periods=6, freq='D')
    initial_capital = 100.0

    metrics = continuous_metrics(equity, dates.tolist(), initial_capital)

    # 总收益 30%
    assert np.isclose(metrics['total_return'], 0.30)

    # MDD = (100-120)/120 = -16.67%
    expected_mdd = (100 - 120) / 120
    assert np.isclose(metrics['mdd'], expected_mdd)

    # Calmar = 年化收益 / abs(MDD)
    assert np.isfinite(metrics['calmar'])
    assert metrics['calmar'] > 0  # 正收益 + 负回撤


def test_continuous_metrics_insufficient_data():
    """测试数据不足情况。"""
    equity = np.array([100])
    dates = [datetime(2024, 1, 1)]
    initial_capital = 100.0

    metrics = continuous_metrics(equity, dates, initial_capital)

    # 单点数据，大部分指标应为 NaN
    assert metrics['final_equity'] == 100.0
    assert np.isnan(metrics['total_return'])
    assert np.isnan(metrics['annual_return'])
    assert metrics['days'] == 0


def test_continuous_metrics_sharpe():
    """测试 Sharpe 比率计算。"""
    # 构造有波动的净值序列
    equity = np.array([100, 105, 103, 108, 110, 115, 112, 118, 120])
    dates = pd.date_range('2024-01-01', periods=9, freq='D')
    initial_capital = 100.0
    risk_free_rate = 0.03

    metrics = continuous_metrics(equity, dates, initial_capital, risk_free_rate)

    # 年化 Sharpe = (年化收益 - 无风险利率) / 年化波动率
    assert np.isfinite(metrics['annual_sharpe'])
    assert np.isfinite(metrics['annual_volatility'])
    assert metrics['annual_volatility'] > 0


def test_continuous_metrics_pred_len_annualization():
    """pred_len>1 时年化必须按真实交易日数换算，而非窗口数。"""
    # 18 个窗口，每窗口持有 10 交易日 → 真实 180 交易日
    equity = np.linspace(100.0, 130.0, 19)
    dates = pd.date_range('2024-01-01', periods=19, freq='D')
    initial_capital = 100.0

    m = continuous_metrics(equity, dates.tolist(), initial_capital, pred_len=10)

    # days 应为窗口数 × pred_len = 18 × 10 = 180
    assert m['days'] == 180

    # 年化按 252/180 换算，而非旧 bug 的 252/18
    total_return = 0.30
    expected_annual = (1 + total_return) ** (252 / 180) - 1
    assert np.isclose(m['annual_return'], expected_annual)

    # 波动率按每年 252/10 个窗口年化
    period_returns = np.diff(equity) / equity[:-1]
    expected_vol = np.std(period_returns, ddof=1) * np.sqrt(252 / 10)
    assert np.isclose(m['annual_volatility'], expected_vol)


def test_continuous_metrics_pred_len_default_backward_compat():
    """pred_len 默认为 1 时，行为与旧版一致（每步=1交易日）。"""
    equity = np.array([100, 110, 115, 120, 125])
    dates = pd.date_range('2024-01-01', periods=5, freq='D')

    m = continuous_metrics(equity, dates.tolist(), 100.0)

    assert m['days'] == 4
    assert np.isclose(m['annual_return'], (1.25) ** (252 / 4) - 1)

def test_continuous_metrics_negative_return():
    """测试负收益情况。"""
    equity = np.array([100, 95, 90, 85, 80])
    dates = pd.date_range('2024-01-01', periods=5, freq='D')
    initial_capital = 100.0

    metrics = continuous_metrics(equity, dates, initial_capital)

    # 总收益 -20%
    assert np.isclose(metrics['total_return'], -0.20)

    # 年化收益为负
    assert metrics['annual_return'] < 0

    # MDD 应该是从峰值到谷底的最大跌幅
    expected_mdd = (80 - 100) / 100
    assert np.isclose(metrics['mdd'], expected_mdd)

    # Calmar = 负收益 / abs(MDD)，应为负
    assert metrics['calmar'] < 0



def test_summarize_strategy_metrics_separates_mean_calmar_from_ratio_of_means():
    rows = [
        {"code": "A", "strategy": "critic_weighted", "annual_return": 0.10, "mdd": -0.10, "calmar": 1.0},
        {"code": "B", "strategy": "critic_weighted", "annual_return": 0.10, "mdd": -0.001, "calmar": 100.0},
    ]

    summary = summarize_strategy_metrics(rows, "critic_weighted")

    assert np.isclose(summary["mean_annual_return"], 0.10)
    assert np.isclose(summary["mean_mdd"], -0.0505)
    assert np.isclose(summary["mean_calmar"], 50.5)
    assert np.isclose(summary["median_calmar"], 50.5)
    assert np.isclose(summary["ratio_of_means_calmar"], 0.10 / 0.0505)
    assert summary["ratio_of_means_calmar"] < summary["mean_calmar"]



def test_format_summary_shows_separate_calmar_aggregation_columns():
    from types import SimpleNamespace
    from examples.run_continuous_backtest import format_summary

    args = SimpleNamespace(
        start="20240101",
        end="20241231",
        initial_capital=100000.0,
        pred_len=10,
        regime="net_long_only",
    )
    all_results = {
        "A": {
            "net_long_only": {
                "simple_mean": {"annual_return": 0.10, "mdd": -0.10, "calmar": 1.0},
                "critic_weighted": {"annual_return": 0.10, "mdd": -0.10, "calmar": 1.0},
                "critic_top1": {"annual_return": 0.10, "mdd": -0.10, "calmar": 1.0},
            }
        },
        "B": {
            "net_long_only": {
                "simple_mean": {"annual_return": 0.10, "mdd": -0.001, "calmar": 100.0},
                "critic_weighted": {"annual_return": 0.10, "mdd": -0.001, "calmar": 100.0},
                "critic_top1": {"annual_return": 0.10, "mdd": -0.001, "calmar": 100.0},
            }
        },
    }

    text = format_summary(all_results, args)

    assert "均值Calmar" in text
    assert "中位Calmar" in text
    assert "均值比Calmar" in text
    assert "50.50" in text
    assert "1.98" in text



def test_summarize_strategy_metrics_handles_missing_strategy():
    summary = summarize_strategy_metrics([], "simple_mean")

    assert np.isnan(summary["mean_annual_return"])
    assert np.isnan(summary["mean_mdd"])
    assert np.isnan(summary["mean_calmar"])
    assert np.isnan(summary["median_calmar"])
    assert np.isnan(summary["ratio_of_means_calmar"])
    assert np.isnan(summary["win_rate"])
