"""Plot E2 event-style backtest results without opening a GUI."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from critic.backtest import REGIME_LABELS


STRATEGY_LABELS = {
    "simple_mean": "简单均值",
    "critic_weighted": "Critic 加权",
    "critic_top1": "Critic Top-1",
}


def _strategy_items(strategies):
    names = list(STRATEGY_LABELS)
    names.extend(name for name in strategies if name not in STRATEGY_LABELS)
    return names


def _regime_label(regime):
    return REGIME_LABELS.get(regime, regime)


def _strategy_label(strategy):
    return STRATEGY_LABELS.get(strategy, strategy)


def _plot_regime_subplots(results, plotter):
    regimes = list(results)
    if not regimes:
        regimes = [None]
    figure, axes = plt.subplots(1, len(regimes), squeeze=False, figsize=(4 * len(regimes), 3.5))
    axes = axes[0]
    for axis, regime in zip(axes, regimes):
        strategies = results.get(regime, {}) if regime is not None else {}
        plotter(axis, regime, strategies)
        axis.set_title(_regime_label(regime) if regime is not None else "无结果")
        axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    return figure


def plot_backtest_results(results, output_dir):
    """Write PnL distribution, win-rate, and mean-return figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    def plot_pnl(axis, _regime, strategies):
        for strategy in _strategy_items(strategies):
            if strategy not in strategies:
                continue
            pnl = np.asarray(strategies[strategy].get("pnl", []), dtype=float)
            pnl = pnl[np.isfinite(pnl)]
            axis.hist(pnl, bins=10, alpha=0.5, label=_strategy_label(strategy))
        axis.set_xlabel("PnL")
        axis.set_ylabel("Count")
        axis.legend()

    def plot_win_rate(axis, _regime, strategies):
        names = _strategy_items(strategies)
        values = [strategies[name].get("metrics", {}).get("win_rate", np.nan) for name in names]
        axis.bar(range(len(names)), values)
        axis.set_xticks(range(len(names)), [_strategy_label(name) for name in names])
        axis.set_ylim(0, 1)
        axis.set_ylabel("Win rate")

    def plot_mean_return(axis, _regime, strategies):
        names = _strategy_items(strategies)
        values = [strategies[name].get("metrics", {}).get("mean_return", np.nan) for name in names]
        axis.bar(range(len(names)), values)
        axis.set_xticks(range(len(names)), [_strategy_label(name) for name in names])
        axis.set_ylabel("Mean return")

    plotters = (
        ("e2_pnl_distribution.png", plot_pnl),
        ("e2_win_rate.png", plot_win_rate),
        ("e2_mean_return.png", plot_mean_return),
    )
    paths = []
    for filename, plotter in plotters:
        figure = _plot_regime_subplots(results, plotter)
        path = output_dir / filename
        figure.savefig(path, dpi=160)
        plt.close(figure)
        paths.append(path)
    return paths
