"""连续资金曲线回测驱动脚本。

基于 E2 框架扩展，支持：
- 不重叠窗口序列
- 复利净值累积
- MDD、Calmar、年化收益、年化 Sharpe
- 四口径全覆盖
- 多股票并行
"""
import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def infer_critic_arch(state_dict):
    """Identify the critic architecture represented by a checkpoint."""
    has_final = any(key.startswith("final_head.") for key in state_dict)
    has_fine = any(key.startswith("fine_head.") for key in state_dict)
    has_coarse = any(key.startswith("coarse_head.") for key in state_dict)
    dual_heads = (has_final, has_fine, has_coarse)
    if all(dual_heads):
        return "dual_head"
    if any(dual_heads):
        raise ValueError("dual_head checkpoint must include final_head, fine_head, and coarse_head")
    return "fundamental"


def _critic_kwargs_from_state_dict(state_dict):
    """从权重推断 Critic 构造参数。"""
    architecture = infer_critic_arch(state_dict)
    if architecture == "dual_head":
        kwargs = {}
        fine_weight = state_dict.get("fine_head.0.weight")
        final_weight = state_dict.get("final_head.0.weight")
        if fine_weight is not None:
            feature_width = fine_weight.shape[1]
            if feature_width == 96:
                kwargs["path_corr_interaction"] = False
            elif feature_width == 128:
                kwargs["path_corr_interaction"] = True
        if fine_weight is not None and final_weight is not None:
            feature_width = fine_weight.shape[1]
            final_width = final_weight.shape[1]
            has_base = any(key.startswith("base_head.") for key in state_dict)
            has_residual_weights = {
                "residual_fine_weight",
                "residual_coarse_weight",
            }.issubset(state_dict)
            if has_residual_weights:
                kwargs["final_fusion"] = True
                kwargs["final_combiner"] = "residual"
            elif has_base and final_width == 3:
                kwargs["final_fusion"] = True
                kwargs["final_combiner"] = "calibrated"
            else:
                kwargs["final_fusion"] = final_width == feature_width + 2
                kwargs["final_combiner"] = "explicit"
        return kwargs

    head_weight = state_dict.get("head.0.weight")
    if head_weight is None:
        return {}
    head_in = head_weight.shape[1]
    if head_in == 96:
        return {"path_corr_interaction": False}
    if head_in == 128:
        return {"path_corr_interaction": True}
    return {}


def build_critic_from_state_dict(state_dict, critic_classes):
    """Build a critic matching the checkpoint architecture."""
    architecture = infer_critic_arch(state_dict)
    critic_cls = critic_classes[architecture]
    return critic_cls(in_dim=6, **_critic_kwargs_from_state_dict(state_dict))

# 交易成本
STAMP_DUTY = 0.0005
ROUND_TRIP_COMMISSION = 0.00025
NET_COST = STAMP_DUTY + ROUND_TRIP_COMMISSION
SEED = 0

# 四套口径
REGIMES = {
    "gross_long_short": {"entry_mode": "close", "long_only": False, "cost": 0.0},
    "gross_long_only": {"entry_mode": "close", "long_only": True, "cost": 0.0},
    "net_long_short": {"entry_mode": "next_open", "long_only": False, "cost": NET_COST},
    "net_long_only": {"entry_mode": "next_open", "long_only": True, "cost": NET_COST},
}

REGIME_LABELS = {
    "gross_long_short": "毛·多空对称",
    "gross_long_only": "毛·只做多",
    "net_long_short": "净·多空对称（理论参考：A股裸卖空不现实）",
    "net_long_only": "净·只做多",
}


def make_local_corr_fetch(local_data_dir):
    """返回一个与 _fetch_daily_kline 同签名的本地 CSV 读取函数。

    关联标的从本地 CSV 读取，消除联网抓取与请求间隔，让 GPU 成为真正瓶颈。
    本地缺失的标的返回 None（provider 会用零特征占位，保持张量形状）。
    """
    base = Path(local_data_dir)

    def _fetch(stock_code, start_date, end_date, session=None, **kwargs):
        f = base / f"{stock_code}.csv"
        if not f.exists():
            return None
        df = pd.read_csv(f)
        ts_col = 'timestamps' if 'timestamps' in df.columns else 'trade_date'
        df[ts_col] = pd.to_datetime(df[ts_col])
        df = df.set_index(ts_col)
        df = df[(df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))]
        return df if len(df) else None

    return _fetch


def set_seed(seed):
    """设置随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_runtime_dependencies():
    """加载运行时依赖。"""
    from critic import (
        CorrelatedDataProvider,
        CriticKronosPredictor,
        CriticReranker,
        DualHeadCritic,
        FundamentalCritic,
        make_critic_strategy,
        make_tushare_fetch,
        strategy_simple_mean,
    )
    from critic.continuous_backtest import (
        build_non_overlapping_windows,
        simulate_equity_curve,
        continuous_metrics,
        format_continuous_results,
    )
    from model import Kronos, KronosTokenizer
    import torch

    return {
        "CorrelatedDataProvider": CorrelatedDataProvider,
        "CriticKronosPredictor": CriticKronosPredictor,
        "CriticReranker": CriticReranker,
        "DualHeadCritic": DualHeadCritic,
        "FundamentalCritic": FundamentalCritic,
        "Kronos": Kronos,
        "KronosTokenizer": KronosTokenizer,
        "make_critic_strategy": make_critic_strategy,
        "make_tushare_fetch": make_tushare_fetch,
        "strategy_simple_mean": strategy_simple_mean,
        "build_non_overlapping_windows": build_non_overlapping_windows,
        "simulate_equity_curve": simulate_equity_curve,
        "continuous_metrics": continuous_metrics,
        "format_continuous_results": format_continuous_results,
        "torch": torch,
    }


def load_stock_data(code, fetch_fn, start_date, end_date, lookback, pred_len, local_data_dir=None):
    """加载单只股票的历史数据。

    Args:
        code: 股票代码
        fetch_fn: 数据获取函数
        start_date, end_date: 日期范围
        lookback, pred_len: 窗口参数
        local_data_dir: 本地数据目录（优先使用）

    Returns:
        df: DataFrame 或 None（失败时）
    """
    try:
        # 优先尝试本地数据
        if local_data_dir:
            local_file = Path(local_data_dir) / f"{code}.csv"
            if local_file.exists():
                print(f"  从本地加载: {local_file}")
                df = pd.read_csv(local_file)

                # 处理时间列
                if 'timestamps' in df.columns:
                    df['timestamps'] = pd.to_datetime(df['timestamps'])
                    df = df.set_index('timestamps')
                elif 'trade_date' in df.columns:
                    df['trade_date'] = pd.to_datetime(df['trade_date'])
                    df = df.set_index('trade_date')

                # 筛选日期范围
                if start_date:
                    df = df[df.index >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df.index <= pd.to_datetime(end_date)]

                if len(df) >= lookback + pred_len:
                    return df
                else:
                    print(f"  本地数据不足（需要 {lookback + pred_len}，实际 {len(df)}）")

        # 需要额外的历史数据用于构建第一个窗口
        extended_start = pd.to_datetime(start_date) - pd.Timedelta(days=lookback * 2)

        if fetch_fn:
            # Tushare
            df = fetch_fn(code, extended_start.strftime('%Y%m%d'), end_date)
        else:
            # 东方财富
            import akshare as ak
            df = ak.stock_zh_a_hist(
                symbol=code,
                start_date=extended_start.strftime('%Y%m%d'),
                end_date=end_date,
                adjust="qfq"
            )
            # 重命名列
            df = df.rename(columns={
                '日期': 'trade_date',
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume',
                '成交额': 'amount',
            })

        if df is None or len(df) < lookback + pred_len:
            return None

        # 确保有时间索引
        if 'trade_date' in df.columns:
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.set_index('trade_date')

        return df

    except Exception as e:
        print(f"  加载 {code} 数据失败: {e}")
        return None


def run_single_stock(code, df, args, deps, predictor, strategies):
    """运行单只股票的连续回测。

    Args:
        code: 股票代码
        df: 历史数据
        args: 命令行参数
        deps: 运行时依赖
        predictor: CriticKronosPredictor 实例
        strategies: 策略字典

    Returns:
        dict[regime, dict[strategy, metrics]]
    """
    print(f"\n{'='*80}")
    print(f"股票: {code}")
    print(f"{'='*80}")

    results = {}

    # 对每个口径运行回测
    regimes_to_run = [args.regime] if args.regime != 'all' else REGIMES.keys()

    for regime in regimes_to_run:
        print(f"\n口径: {REGIME_LABELS[regime]}")

        # 构建不重叠窗口
        windows = deps["build_non_overlapping_windows"](
            df,
            lookback=args.lookback,
            pred_len=args.pred_len,
            start_date=args.start,
            end_date=args.end,
        )

        if not windows:
            print(f"  警告: 无法构建窗口，跳过")
            continue

        print(f"  窗口数: {len(windows)}")

        # 模拟净值曲线
        try:
            equity_curves = deps["simulate_equity_curve"](
                windows=windows,
                predictor=predictor,
                strategies=strategies,
                regime_settings=REGIMES[regime],
                initial_capital=args.initial_capital,
                sample_count=args.sample_count,
                target_code=code,
                verbose=args.verbose,
            )
        except Exception as e:
            print(f"  模拟净值曲线失败: {e}")
            import traceback
            traceback.print_exc()
            continue

        # 计算指标
        metrics_by_strategy = {}
        for strategy_name, equity_curve in equity_curves.items():
            metrics = deps["continuous_metrics"](
                equity_curve.equity,
                equity_curve.dates,
                args.initial_capital,
                risk_free_rate=args.risk_free_rate,
                pred_len=args.pred_len,
            )
            metrics['trade_count'] = len(equity_curve.trades)
            metrics_by_strategy[strategy_name] = metrics

        results[regime] = metrics_by_strategy

        # 保存净值曲线 CSV
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for strategy_name, equity_curve in equity_curves.items():
            csv_file = output_dir / f"equity_{code}_{regime}_{strategy_name}.csv"
            df_equity = pd.DataFrame({
                'date': equity_curve.dates,
                'equity': equity_curve.equity,
                'position': equity_curve.positions,
            })
            df_equity.to_csv(csv_file, index=False)

        print(f"  净值曲线已保存: {output_dir / f'equity_{code}_{regime}_*.csv'}")

        # 打印结果
        result_text = deps["format_continuous_results"](
            metrics_by_strategy,
            code,
            REGIME_LABELS[regime],
            args.start,
            args.end,
            args.initial_capital,
            args.pred_len,
        )
        print("\n" + result_text)

        # 保存结果文本
        txt_file = output_dir / f"result_{code}_{regime}.txt"
        txt_file.write_text(result_text, encoding='utf-8')

    return results


def format_summary(all_results, args):
    """格式化多股票汇总表。

    Args:
        all_results: dict[code, dict[regime, dict[strategy, metrics]]]
        args: 命令行参数

    Returns:
        str: 汇总表文本
    """
    lines = [
        "━" * 120,
        "连续回测汇总",
        "━" * 120,
        f"时间: {args.start} 至 {args.end}",
        f"股票数: {len(all_results)}",
        f"初始资金: {args.initial_capital:,.0f}",
        f"持有期: {args.pred_len} 天",
        "━" * 120,
        "",
    ]

    regimes_to_show = [args.regime] if args.regime != 'all' else REGIMES.keys()

    for regime in regimes_to_show:
        lines.append(f"### {REGIME_LABELS[regime]}")
        lines.append("")

        # 收集所有股票在该口径下的指标
        regime_data = []
        for code, code_results in all_results.items():
            if regime not in code_results:
                continue

            for strategy_name, metrics in code_results[regime].items():
                regime_data.append({
                    'code': code,
                    'strategy': strategy_name,
                    **metrics
                })

        if not regime_data:
            lines.append("(无数据)")
            lines.append("")
            continue

        df = pd.DataFrame(regime_data)

        # 按策略聚合
        lines.append(f"{'策略':<16} | {'平均年化收益':>12} | {'平均MDD':>10} | {'均值Calmar':>10} | {'中位Calmar':>11} | {'均值比Calmar':>12} | {'胜率(年化>0)':>14}")
        lines.append("-" * 110)

        from critic.continuous_backtest import summarize_strategy_metrics

        for strategy_name in ['simple_mean', 'critic_weighted', 'critic_top1']:
            strategy_df = df[df['strategy'] == strategy_name]
            if len(strategy_df) == 0:
                continue

            summary = summarize_strategy_metrics(regime_data, strategy_name)

            lines.append(
                f"{strategy_name:<16} | "
                f"{summary['mean_annual_return']:>11.2%} | "
                f"{summary['mean_mdd']:>9.2%} | "
                f"{summary['mean_calmar']:>10.2f} | "
                f"{summary['median_calmar']:>11.2f} | "
                f"{summary['ratio_of_means_calmar']:>12.2f} | "
                f"{summary['win_rate']:>13.2%}"
            )

        lines.append("")
        lines.append("按股票明细:")
        lines.append(f"{'代码':<8} | {'simple_mean':>13} | {'critic_weighted':>16} | {'critic_top1':>13} | {'critic_top1 Calmar':>18}")
        lines.append("-" * 80)

        for code in sorted(all_results.keys()):
            if regime not in all_results[code]:
                continue

            code_metrics = all_results[code][regime]
            sm = code_metrics.get('simple_mean', {})
            cw = code_metrics.get('critic_weighted', {})
            ct = code_metrics.get('critic_top1', {})

            lines.append(
                f"{code:<8} | "
                f"{sm.get('annual_return', np.nan):>12.2%} | "
                f"{cw.get('annual_return', np.nan):>15.2%} | "
                f"{ct.get('annual_return', np.nan):>12.2%} | "
                f"{ct.get('calmar', np.nan):>18.2f}"
            )

        lines.append("")

    return "\n".join(lines)


def run(args):
    """主函数。"""
    set_seed(args.seed)
    deps = load_runtime_dependencies()

    print("加载 Kronos 模型...")
    tokenizer = deps["KronosTokenizer"].from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = deps["Kronos"].from_pretrained("NeoQuasar/Kronos-small")

    print(f"加载 Critic 权重: {args.critic_ckpt}")
    state_dict = deps["torch"].load(args.critic_ckpt, map_location=args.device)
    critic = build_critic_from_state_dict(
        state_dict,
        {
            "dual_head": deps["DualHeadCritic"],
            "fundamental": deps["FundamentalCritic"],
        },
    ).to(args.device)
    critic.load_state_dict(state_dict)
    critic.eval()

    # 数据源
    fetch_fn = None
    corr_provider = None
    if args.tushare_token:
        fetch_fn = deps["make_tushare_fetch"](args.tushare_token)
        print("数据源: Tushare")
    else:
        print("数据源: 东方财富 (akshare)")

    # 关联标的数据源：优先本地 CSV（无网络 I/O，无请求间隔），回退到 fetch_fn/东方财富
    corr_fetch_fn = fetch_fn
    corr_delay = 0 if fetch_fn else 1.5
    if args.local_data_dir and Path(args.local_data_dir).exists():
        corr_fetch_fn = make_local_corr_fetch(args.local_data_dir)
        corr_delay = 0
        print(f"关联标的数据源: 本地 CSV ({args.local_data_dir})")

    corr_provider = deps["CorrelatedDataProvider"](
        lookback=args.lookback,
        fetch_fn=corr_fetch_fn,
        request_delay=corr_delay,
    )

    reranker = deps["CriticReranker"](mode="weighted", temperature=1.0)
    predictor = deps["CriticKronosPredictor"](
        model,
        tokenizer,
        critic=critic,
        reranker=reranker,
        corr_provider=corr_provider,
        device=args.device,
    )

    strategies = {
        "simple_mean": deps["strategy_simple_mean"],
        "critic_weighted": deps["make_critic_strategy"](
            critic, deps["CriticReranker"](mode="weighted", temperature=1.0)
        ),
        "critic_top1": deps["make_critic_strategy"](
            critic, deps["CriticReranker"](mode="top1")
        ),
    }

    # 读取股票代码
    if args.codes_file:
        codes = Path(args.codes_file).read_text().strip().split('\n')
        codes = [c.strip() for c in codes if c.strip()]
    else:
        codes = args.codes.split(',')

    print(f"\n待回测股票: {codes}")
    print(f"回测区间: {args.start} 至 {args.end}")
    print(f"初始资金: {args.initial_capital:,.0f}")
    print(f"持有期: {args.pred_len} 天")

    # 逐只股票运行
    all_results = {}
    for i, code in enumerate(codes):
        print(f"\n进度: {i+1}/{len(codes)}")

        # 加载数据
        df = load_stock_data(
            code, fetch_fn, args.start, args.end, args.lookback, args.pred_len,
            local_data_dir=args.local_data_dir
        )

        if df is None:
            print(f"  跳过 {code}（数据加载失败）")
            continue

        # 运行回测
        results = run_single_stock(code, df, args, deps, predictor, strategies)
        if results:
            all_results[code] = results

    # 生成汇总
    if all_results:
        summary = format_summary(all_results, args)
        print("\n" + summary)

        summary_file = Path(args.output_dir) / "summary.txt"
        summary_file.write_text(summary, encoding='utf-8')
        print(f"\n汇总已保存: {summary_file}")
    else:
        print("\n警告: 没有任何股票成功完成回测")


def parse_args():
    parser = argparse.ArgumentParser(description="连续资金曲线回测")
    parser.add_argument("--codes", default="300750,002594", help="股票代码列表（逗号分隔）")
    parser.add_argument("--codes-file", default=None, help="股票代码文件路径")
    parser.add_argument("--start", default="20240101", help="回测起始日期")
    parser.add_argument("--end", default="20241231", help="回测结束日期")
    parser.add_argument("--lookback", type=int, default=90, help="回看天数")
    parser.add_argument("--pred-len", type=int, default=10, help="预测天数（持有期）")
    parser.add_argument("--initial-capital", type=float, default=100000, help="初始资金")
    parser.add_argument("--risk-free-rate", type=float, default=0.03, help="无风险利率")
    parser.add_argument("--regime", default="net_long_only",
                        choices=list(REGIMES.keys()) + ['all'], help="口径")
    parser.add_argument("--critic-ckpt", required=True, help="Critic 权重路径")
    parser.add_argument("--tushare-token", default=None, help="Tushare token")
    parser.add_argument("--sample-count", type=int, default=16, help="Kronos 采样数")
    parser.add_argument("--device", default="cpu", help="计算设备")
    parser.add_argument("--seed", type=int, default=SEED, help="随机种子")
    parser.add_argument("--output-dir", default="output/results/continuous/", help="输出目录")
    parser.add_argument("--local-data-dir", default="data/market/test", help="本地数据目录")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
