"""E2 event-style holding-period backtest runner."""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

STAMP_DUTY = 0.0005
ROUND_TRIP_COMMISSION = 0.00025
NET_COST = STAMP_DUTY + ROUND_TRIP_COMMISSION
SEED = 0

REGIMES = {
    "gross_long_short": {"entry_mode": "close", "long_only": False, "cost": 0.0},
    "gross_long_only": {"entry_mode": "close", "long_only": True, "cost": 0.0},
    "net_long_short": {"entry_mode": "next_open", "long_only": False, "cost": NET_COST},
    "net_long_only": {"entry_mode": "next_open", "long_only": True, "cost": NET_COST},
}


_BACKTEST_MODULE = None


def _backtest():
    """Load critic/backtest.py without importing critic.__init__."""
    global _BACKTEST_MODULE
    if _BACKTEST_MODULE is None:
        import importlib.util

        module_path = Path(__file__).resolve().parents[1] / "critic" / "backtest.py"
        spec = importlib.util.spec_from_file_location("_e2_backtest_core", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _BACKTEST_MODULE = module
    return _BACKTEST_MODULE


def set_seed(seed):
    """Seed Python, NumPy, and PyTorch generators when PyTorch is installed."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_temperatures(value):
    """Parse comma-separated positive temperatures for sensitivity sweeps."""
    temperatures = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not temperatures:
        raise ValueError("at least one temperature is required")
    if any(t <= 0 for t in temperatures):
        raise ValueError("temperatures must be positive")
    return temperatures


def _format_temperature(value):
    return f"{value:g}"


def render_temperature_report(per_window_by_temperature, pred_len, sample_count, seed):
    """Render weighted-strategy sensitivity across softmax temperatures."""
    lines = [
        "E2 温度 τ 敏感性",
        f"temperatures={','.join(_format_temperature(t) for t in per_window_by_temperature)}, pred_len={pred_len}, sample_count={sample_count}, seed={seed}",
        "simple_mean 固定；仅改变 critic_weighted 的 softmax 温度 τ。",
        "累计收益是各窗口收益相加，不是复利资金曲线。",
        "",
        "| τ | regime | N | 平均收益 | 胜率 | 标准差 | Sharpe | IR | 累计收益 | invalid |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for temperature, per_window in per_window_by_temperature.items():
        results = aggregate_pnls(per_window, pred_len)
        for regime, strategies in results.items():
            item = strategies["critic_weighted"]
            metrics = item["metrics"]
            lines.append(
                "| "
                + " | ".join([
                    _format_temperature(temperature),
                    regime,
                    str(metrics["count"]),
                    f"{metrics['mean_return']:.4%}",
                    f"{metrics['win_rate']:.4%}",
                    f"{metrics['std_return']:.4%}",
                    f"{metrics['sharpe']:.4f}",
                    f"{metrics['information_ratio']:.4f}",
                    f"{metrics['cumulative_return']:.4%}",
                    str(item["invalid_count"]),
                ])
                + " |"
            )
    return "\n".join(lines)


def render_report(results, window_count, pred_len, sample_count, seed):
    """Render a reproducible E2 accounting report."""
    return "\n".join([
        "E2 事件式持有期收益回测",
        f"windows={window_count}, pred_len={pred_len}, sample_count={sample_count}, seed={seed}",
        f"net_cost={NET_COST:.6f} (stamp={STAMP_DUTY:.6f}, commission={ROUND_TRIP_COMMISSION:.6f})",
        "Sharpe/IR用于相对比较；累计收益是各窗口收益相加，不是复利资金曲线。",
        "注意：窗口可能重叠，结果应视为事件样本统计而非独立交易序列。",
        "净多空口径为理论参考，A股裸卖空并不现实。",
        "",
        _backtest().format_backtest_table(results),
    ])


def evaluate_window(x_df, y_true, final_paths, x_mean, x_std, net_cost=NET_COST):
    """Evaluate all strategies under each fixed accounting regime."""
    direction_base = x_df["close"].iloc[-1]
    results = {}
    module = _backtest()
    for regime, settings in REGIMES.items():
        cost = settings["cost"] if settings["cost"] == 0.0 else net_cost
        realized_return = module.holding_period_return(
            x_df, y_true, settings["entry_mode"], cost
        )
        results[regime] = {}
        for strategy, final_path in final_paths.items():
            direction = module.compute_signal(final_path, x_mean, x_std, direction_base)
            results[regime][strategy] = float(
                module.trade_pnl(direction, realized_return, settings["long_only"], cost)
            )
    return results


def aggregate_pnls(per_window, pred_len):
    """Aggregate per-window PnLs and calculate regime-relative metrics."""
    if not per_window:
        return {}

    results = {}
    first_window = per_window[0]
    for regime, strategies in first_window.items():
        strategy_arrays = {
            strategy: np.asarray(
                [window[regime][strategy] for window in per_window], dtype=float
            )
            for strategy in strategies
        }
        baseline = strategy_arrays["simple_mean"]
        results[regime] = {}
        for strategy, pnl in strategy_arrays.items():
            results[regime][strategy] = {
                "pnl": pnl,
                "metrics": _backtest().backtest_metrics(pnl, pred_len, baseline_pnl=baseline),
                "invalid_count": int(np.count_nonzero(~np.isfinite(pnl))),
            }
    return results


def _load_runtime_dependencies():
    from critic import (
        CorrelatedDataProvider,
        CriticKronosPredictor,
        CriticReranker,
        FundamentalCritic,
        make_critic_strategy,
        make_prob_top1_strategy,
        make_tushare_fetch,
        strategy_median,
        strategy_simple_mean,
    )
    from examples.run_critic_comparison import build_real_windows, build_synth_windows
    from examples.train_critic_pipeline import _load_dotenv
    from model import Kronos, KronosTokenizer
    import torch

    return {
        "CorrelatedDataProvider": CorrelatedDataProvider,
        "CriticKronosPredictor": CriticKronosPredictor,
        "CriticReranker": CriticReranker,
        "FundamentalCritic": FundamentalCritic,
        "Kronos": Kronos,
        "KronosTokenizer": KronosTokenizer,
        "_load_dotenv": _load_dotenv,
        "build_real_windows": build_real_windows,
        "build_synth_windows": build_synth_windows,
        "make_critic_strategy": make_critic_strategy,
        "make_prob_top1_strategy": make_prob_top1_strategy,
        "make_tushare_fetch": make_tushare_fetch,
        "strategy_median": strategy_median,
        "strategy_simple_mean": strategy_simple_mean,
        "torch": torch,
    }


def _critic_kwargs_from_state_dict(state_dict):
    """Infer FundamentalCritic construction flags from a saved checkpoint."""
    head_weight = state_dict.get("head.0.weight")
    if head_weight is None:
        return {}
    head_in = head_weight.shape[1]
    if head_in == 96:
        return {"path_corr_interaction": False}
    if head_in == 128:
        return {"path_corr_interaction": True}
    return {}

def _maybe_plot(results, figures_dir):
    try:
        from examples.plot_e2_backtest import plot_backtest_results
    except ModuleNotFoundError as exc:
        if exc.name == "examples.plot_e2_backtest":
            print("绘图模块尚未实现，跳过图表输出。")
            return
        raise
    plot_backtest_results(results, figures_dir)


def run(args):
    """Run E2 with one shared candidate set per evaluation window."""
    if not args.critic_ckpt:
        raise ValueError("E2 requires --critic-ckpt")
    set_seed(args.seed)
    deps = _load_runtime_dependencies()

    fetch_fn = None
    corr_provider = None
    if args.real:
        deps["_load_dotenv"]()
        token = args.tushare_token or os.environ.get("TUSHARE_TOKEN")
        if token:
            fetch_fn = deps["make_tushare_fetch"](token)
            print("数据源: tushare")
        else:
            print("数据源: 东方财富")
        corr_provider = deps["CorrelatedDataProvider"](
            lookback=args.lookback,
            fetch_fn=fetch_fn,
            request_delay=0 if fetch_fn else 1.5,
        )

    print("加载 Kronos 模型...")
    tokenizer = deps["KronosTokenizer"].from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = deps["Kronos"].from_pretrained("NeoQuasar/Kronos-small")

    state_dict = deps["torch"].load(args.critic_ckpt, map_location=args.device)
    critic = deps["FundamentalCritic"](
        in_dim=6, **_critic_kwargs_from_state_dict(state_dict)
    )
    critic.load_state_dict(state_dict)
    critic.eval()

    reranker = deps["CriticReranker"](mode="weighted", temperature=args.temperature)
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
        "median": deps["strategy_median"],
        "critic_weighted": deps["make_critic_strategy"](
            critic, deps["CriticReranker"](mode="weighted", temperature=args.temperature)
        ),
        "critic_top1": deps["make_critic_strategy"](
            critic, deps["CriticReranker"](mode="top1")
        ),
    }

    windows = (
        deps["build_real_windows"](args, fetch_fn)
        if args.real
        else deps["build_synth_windows"](args.n_samples, args.lookback, args.pred_len)
    )
    if not windows:
        raise RuntimeError("未构造出任何评估窗口，终止。")

    per_window = []
    per_window_by_temperature = {
        temperature: [] for temperature in parse_temperatures(args.temperatures)
    }
    for i, (code, x_df, x_ts, y_ts, y_true) in enumerate(windows):
        paths, hist_norm, corr_feat, (x_mean, x_std), log_probs = predictor.generate_paths(
            x_df, x_ts, y_ts, args.pred_len,
            sample_count=args.sample_count, verbose=False, target_code=code,
            return_log_probs=True,
        )
        # prob_top1 按窗口 log_probs 构建（生成器最自信路径，与 critic_top1 配对）
        window_strategies = dict(strategies)
        window_strategies["prob_top1"] = deps["make_prob_top1_strategy"](log_probs)
        final_paths = {
            name: strategy(paths, hist=hist_norm, corr_feat=corr_feat)
            for name, strategy in window_strategies.items()
        }
        window_result = evaluate_window(x_df, y_true, final_paths, x_mean, x_std, NET_COST)
        per_window.append(window_result)
        for temperature in per_window_by_temperature:
            temperature_strategy = deps["make_critic_strategy"](
                critic, deps["CriticReranker"](mode="weighted", temperature=temperature)
            )
            temperature_paths = dict(final_paths)
            temperature_paths["critic_weighted"] = temperature_strategy(
                paths, hist=hist_norm, corr_feat=corr_feat
            )
            per_window_by_temperature[temperature].append(
                evaluate_window(x_df, y_true, temperature_paths, x_mean, x_std, NET_COST)
            )
        print(f"窗口 {i + 1}/{len(windows)} 完成" + (f"（{code}）" if code else ""))

    results = aggregate_pnls(per_window, pred_len=args.pred_len)
    report = render_report(
        results, len(windows), args.pred_len, args.sample_count, args.seed
    )
    Path(args.output).write_text(report, encoding="utf-8")
    temperature_report = render_temperature_report(
        per_window_by_temperature, args.pred_len, args.sample_count, args.seed
    )
    Path(args.temperature_output).write_text(temperature_report, encoding="utf-8")
    print("\n" + report)
    print(f"报告已写入: {args.output}")
    print(f"温度敏感性报告已写入: {args.temperature_output}")
    _maybe_plot(results, args.figures_dir)
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--critic-ckpt", required=True)
    parser.add_argument("--codes", default="300750,002594")
    parser.add_argument("--codes-file", default=None)
    parser.add_argument("--tushare-token", default=None)
    parser.add_argument("--start", default="20240101")
    parser.add_argument("--end", default="20241231")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--n-windows", type=int, default=5)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--lookback", type=int, default=90)
    parser.add_argument("--pred-len", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperatures", default="0.25,0.5,1.0,2.0,5.0")
    parser.add_argument("--temperature-output", default="results_e2_temperature_sensitivity.txt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", default="results_e2_backtest.txt")
    parser.add_argument("--figures-dir", default="figures")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
