"""连续资金曲线回测 — 单资产复利净值模拟。

扩展 E2 事件式回测，支持：
- 不重叠窗口序列（避免重复计仓）
- 复利净值累积
- 最大回撤 (MDD)、Calmar 比率、年化收益、年化 Sharpe
"""
import numpy as np
import pandas as pd
from collections import namedtuple
from typing import List, Dict, Tuple, Optional, Callable

from critic.backtest import (
    compute_signal,
    holding_period_return,
    trade_pnl,
)


EquityCurve = namedtuple('EquityCurve', ['dates', 'equity', 'positions', 'trades'])


def build_non_overlapping_windows(
    df: pd.DataFrame,
    lookback: int,
    pred_len: int,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> List[Tuple[pd.DataFrame, pd.Series, pd.Series, np.ndarray]]:
    """构建不重叠的连续窗口序列。

    Args:
        df: 完整历史数据，必须包含 ['open', 'high', 'low', 'close'] 列
            index 为 DatetimeIndex 或包含 'timestamps' 列
        lookback: 回看天数
        pred_len: 预测天数（持有期）
        start_date: 回测起始日期（可选）
        end_date: 回测结束日期（可选）

    Returns:
        List[(x_df, x_ts, y_ts, y_true)]
        - x_df: 历史 OHLCV 数据（lookback 天）
        - x_ts: 历史时间戳
        - y_ts: 未来时间戳（pred_len 天）
        - y_true: 未来真实 OHLCV（pred_len 天，numpy 数组）
    """
    # 确保有时间索引
    if 'timestamps' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index('timestamps')

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df must have DatetimeIndex or 'timestamps' column")

    # 筛选日期范围
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]

    if len(df) < lookback + pred_len:
        return []

    windows = []
    start_idx = 0

    while start_idx + lookback + pred_len <= len(df):
        # 历史窗口
        x_df = df.iloc[start_idx : start_idx + lookback].copy()
        x_ts = pd.Series(x_df.index, index=x_df.index)

        # 未来窗口
        y_df = df.iloc[start_idx + lookback : start_idx + lookback + pred_len]
        y_ts = pd.Series(y_df.index, index=y_df.index)

        # 转为 numpy 数组（OHLCV 顺序）
        cols = ['open', 'high', 'low', 'close']
        optional_cols = []
        if 'volume' in y_df.columns:
            optional_cols.append('volume')
        if 'amount' in y_df.columns:
            optional_cols.append('amount')

        y_true = y_df[cols + optional_cols].values

        windows.append((x_df, x_ts, y_ts, y_true))

        # 不重叠：步长 = pred_len
        start_idx += pred_len

    return windows


def simulate_equity_curve(
    windows: List[Tuple],
    predictor,
    strategies: Dict[str, Callable],
    regime_settings: Dict,
    initial_capital: float = 100000.0,
    sample_count: int = 16,
    target_code: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, EquityCurve]:
    """模拟连续净值曲线。

    Args:
        windows: 不重叠窗口序列（来自 build_non_overlapping_windows）
        predictor: CriticKronosPredictor 实例
        strategies: dict[str, callable] - 策略名 -> 策略函数
        regime_settings: dict - 口径配置 {'entry_mode': str, 'long_only': bool, 'cost': float}
        initial_capital: 初始资金
        sample_count: Kronos 采样数
        target_code: 股票代码（可选，传给 predictor）
        verbose: 是否打印进度

    Returns:
        dict[strategy_name, EquityCurve]
    """
    if not windows:
        raise ValueError("windows list is empty")

    # 初始化每个策略的净值序列
    results = {}
    for strategy_name in strategies.keys():
        results[strategy_name] = {
            'dates': [],
            'equity': [],
            'positions': [],
            'trades': [],
        }

    # 设置初始净值
    for strategy_name in results:
        results[strategy_name]['equity'].append(initial_capital)
        results[strategy_name]['dates'].append(windows[0][0].index[-1])  # T 日
        results[strategy_name]['positions'].append(0)

    # 逐窗口模拟
    for i, (x_df, x_ts, y_ts, y_true) in enumerate(windows):
        if verbose:
            print(f"  窗口 {i+1}/{len(windows)}: {x_df.index[-1].date()} → {y_ts.iloc[-1].date()}")

        # 生成候选路径
        try:
            paths, hist_norm, corr_feat, (x_mean, x_std) = predictor.generate_paths(
                x_df, x_ts, y_ts, len(y_ts),
                sample_count=sample_count,
                verbose=False,
                target_code=target_code,
            )
        except Exception as e:
            if verbose:
                print(f"    警告: 窗口 {i+1} 生成路径失败 ({e})，跳过")
            # 净值保持不变
            for strategy_name in results:
                current_equity = results[strategy_name]['equity'][-1]
                results[strategy_name]['equity'].append(current_equity)
                results[strategy_name]['dates'].append(y_ts.iloc[-1])
                results[strategy_name]['positions'].append(0)
            continue

        # 计算真实持有期收益
        try:
            realized_return = holding_period_return(
                x_df, y_true, regime_settings['entry_mode'], regime_settings['cost']
            )
        except Exception as e:
            if verbose:
                print(f"    警告: 窗口 {i+1} 计算收益失败 ({e})，跳过")
            for strategy_name in results:
                current_equity = results[strategy_name]['equity'][-1]
                results[strategy_name]['equity'].append(current_equity)
                results[strategy_name]['dates'].append(y_ts.iloc[-1])
                results[strategy_name]['positions'].append(0)
            continue

        # 方向基准价
        direction_base = x_df['close'].iloc[-1]

        # 对每个策略计算信号和更新净值
        for strategy_name, strategy_fn in strategies.items():
            # 策略组合路径
            final_norm = strategy_fn(paths, hist=hist_norm, corr_feat=corr_feat)

            # 计算信号
            direction = compute_signal(final_norm, x_mean, x_std, direction_base)

            # 计算策略收益
            pnl = trade_pnl(
                direction,
                realized_return,
                regime_settings['long_only'],
                regime_settings['cost']
            )

            # 更新净值
            current_equity = results[strategy_name]['equity'][-1]
            if np.isfinite(pnl):
                new_equity = current_equity * (1 + pnl)
            else:
                new_equity = current_equity
                if verbose:
                    print(f"    警告: {strategy_name} 窗口 {i+1} pnl 非有限值，净值保持不变")

            results[strategy_name]['equity'].append(new_equity)
            results[strategy_name]['dates'].append(y_ts.iloc[-1])
            results[strategy_name]['positions'].append(direction)

            # 记录交易
            if direction != 0:
                results[strategy_name]['trades'].append({
                    'date': y_ts.iloc[-1],
                    'direction': direction,
                    'pnl': pnl,
                    'equity_before': current_equity,
                    'equity_after': new_equity,
                })

    # 转为 EquityCurve namedtuple
    equity_curves = {}
    for strategy_name, data in results.items():
        equity_curves[strategy_name] = EquityCurve(
            dates=data['dates'],
            equity=np.array(data['equity']),
            positions=data['positions'],
            trades=data['trades'],
        )

    return equity_curves


def calculate_drawdown_series(equity_curve: np.ndarray) -> np.ndarray:
    """计算回撤序列。

    Args:
        equity_curve: 净值序列

    Returns:
        drawdown: 回撤序列（负值表示回撤）
    """
    cumulative_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - cumulative_max) / cumulative_max
    return drawdown


def calculate_mdd(equity_curve: np.ndarray) -> float:
    """计算最大回撤。

    Args:
        equity_curve: 净值序列

    Returns:
        mdd: 最大回撤（负值）
    """
    drawdown = calculate_drawdown_series(equity_curve)
    return float(drawdown.min())


def calculate_calmar(annual_return: float, mdd: float) -> float:
    """计算 Calmar 比率。

    Args:
        annual_return: 年化收益率
        mdd: 最大回撤（负值）

    Returns:
        calmar: Calmar 比率（年化收益 / abs(MDD)）
    """
    if mdd >= 0 or not np.isfinite(mdd):
        return np.nan
    return annual_return / abs(mdd)


def continuous_metrics(
    equity_curve: np.ndarray,
    dates: List,
    initial_capital: float,
    risk_free_rate: float = 0.03,
    pred_len: int = 1,
) -> Dict[str, float]:
    """计算完整指标集。

    Args:
        equity_curve: 净值序列（每个点对应一个持有期窗口的收盘）
        dates: 日期序列
        initial_capital: 初始资金
        risk_free_rate: 无风险利率（默认 3%）
        pred_len: 单个窗口的持有期交易日数。净值序列每步跨越 pred_len 个
            交易日，年化时必须按真实交易日数换算，否则会把窗口数当成交易日数
            导致年化收益指数级虚高。

    Returns:
        dict: {
            'final_equity': 最终净值,
            'total_return': 总收益率,
            'annual_return': 年化收益率,
            'mdd': 最大回撤,
            'calmar': Calmar 比率,
            'annual_sharpe': 年化 Sharpe,
            'annual_volatility': 年化波动率,
            'days': 交易日数,
        }
    """
    if len(equity_curve) < 2:
        return {
            'final_equity': equity_curve[-1] if len(equity_curve) > 0 else np.nan,
            'total_return': np.nan,
            'annual_return': np.nan,
            'mdd': np.nan,
            'calmar': np.nan,
            'annual_sharpe': np.nan,
            'annual_volatility': np.nan,
            'days': len(equity_curve) - 1,
        }

    # 基础指标
    final_equity = equity_curve[-1]
    total_return = (final_equity / initial_capital) - 1.0
    n_periods = len(equity_curve) - 1
    # 真实交易日数 = 窗口数 × 每窗口持有期
    days = n_periods * pred_len

    # 年化收益（按真实交易日数换算，252 交易日/年）
    if days > 0:
        annual_return = (1 + total_return) ** (252 / days) - 1.0
    else:
        annual_return = np.nan

    # 最大回撤
    mdd = calculate_mdd(equity_curve)

    # Calmar
    calmar = calculate_calmar(annual_return, mdd)

    # 每窗口期收益序列（每步跨越 pred_len 个交易日）
    period_returns = np.diff(equity_curve) / equity_curve[:-1]

    # 年化波动率：每年约 252/pred_len 个不重叠窗口
    if len(period_returns) > 1:
        periods_per_year = 252 / pred_len
        annual_volatility = np.std(period_returns, ddof=1) * np.sqrt(periods_per_year)
    else:
        annual_volatility = np.nan

    # 年化 Sharpe
    if annual_volatility > 0:
        annual_sharpe = (annual_return - risk_free_rate) / annual_volatility
    else:
        annual_sharpe = np.nan

    return {
        'final_equity': float(final_equity),
        'total_return': float(total_return),
        'annual_return': float(annual_return),
        'mdd': float(mdd),
        'calmar': float(calmar),
        'annual_sharpe': float(annual_sharpe),
        'annual_volatility': float(annual_volatility),
        'days': int(days),
    }


def summarize_strategy_metrics(rows, strategy_name: str) -> dict[str, float]:
    """Aggregate per-stock continuous metrics without hiding Calmar outliers."""
    selected = [r for r in rows if r.get("strategy") == strategy_name]
    if not selected:
        return {
            "mean_annual_return": np.nan,
            "mean_mdd": np.nan,
            "mean_calmar": np.nan,
            "median_calmar": np.nan,
            "ratio_of_means_calmar": np.nan,
            "win_rate": np.nan,
        }

    annual = np.array([r.get("annual_return", np.nan) for r in selected], dtype=float)
    mdd = np.array([r.get("mdd", np.nan) for r in selected], dtype=float)
    calmar = np.array([r.get("calmar", np.nan) for r in selected], dtype=float)

    mean_annual = float(np.nanmean(annual))
    mean_mdd = float(np.nanmean(mdd))
    ratio_of_means = calculate_calmar(mean_annual, mean_mdd)

    return {
        "mean_annual_return": mean_annual,
        "mean_mdd": mean_mdd,
        "mean_calmar": float(np.nanmean(calmar)),
        "median_calmar": float(np.nanmedian(calmar)),
        "ratio_of_means_calmar": float(ratio_of_means),
        "win_rate": float(np.nanmean(annual > 0)),
    }



def format_continuous_results(
    metrics_by_strategy: Dict[str, Dict[str, float]],
    code: str,
    regime_label: str,
    start_date: str,
    end_date: str,
    initial_capital: float,
    pred_len: int,
) -> str:
    """格式化单只股票的连续回测结果。

    Args:
        metrics_by_strategy: dict[strategy_name, metrics_dict]
        code: 股票代码
        regime_label: 口径标签
        start_date, end_date: 时间范围
        initial_capital: 初始资金
        pred_len: 持有期天数

    Returns:
        str: 格式化的结果表
    """
    lines = [
        "━" * 80,
        "连续回测结果",
        "━" * 80,
        f"股票: {code}",
        f"口径: {regime_label}",
        f"时间: {start_date} 至 {end_date}",
        f"初始资金: {initial_capital:,.0f}",
        f"持有期: {pred_len} 天",
        "━" * 80,
        "",
        f"{'策略':<16} | {'最终净值':>10} | {'总收益':>8} | {'年化收益':>8} | {'最大回撤':>8} | {'Calmar':>7} | {'年化Sharpe':>10} | {'年化波动率':>10}",
        "-" * 120,
    ]

    for strategy_name, m in metrics_by_strategy.items():
        lines.append(
            f"{strategy_name:<16} | "
            f"{m['final_equity']:>10,.0f} | "
            f"{m['total_return']:>7.2%} | "
            f"{m['annual_return']:>7.2%} | "
            f"{m['mdd']:>7.2%} | "
            f"{m['calmar']:>7.2f} | "
            f"{m['annual_sharpe']:>10.2f} | "
            f"{m['annual_volatility']:>9.2%}"
        )

    # 改善幅度（相对 simple_mean）
    if 'simple_mean' in metrics_by_strategy:
        baseline = metrics_by_strategy['simple_mean']
        lines.append("")
        lines.append("改善幅度（相对 simple_mean）:")
        for strategy_name, m in metrics_by_strategy.items():
            if strategy_name == 'simple_mean':
                continue
            total_return_diff = m['total_return'] - baseline['total_return']
            mdd_improve = (baseline['mdd'] - m['mdd']) / abs(baseline['mdd']) if baseline['mdd'] < 0 else np.nan
            calmar_change = ((m['calmar'] - baseline['calmar']) / abs(baseline['calmar'])) if np.isfinite(baseline['calmar']) and baseline['calmar'] != 0 else np.nan

            lines.append(
                f"  {strategy_name}: "
                f"总收益 {total_return_diff:+.2%}, "
                f"MDD 改善 {mdd_improve:.1%}, "
                f"Calmar 提升 {calmar_change:.1%}"
            )

    return "\n".join(lines)
