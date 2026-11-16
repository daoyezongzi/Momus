import shutil
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from examples.plot_e2_backtest import plot_backtest_results


def _local_tmp(name):
    path = Path(".pytest_local_tmp") / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


def test_plot_backtest_results_writes_three_figures():
    strategies = {}
    for name, pnl in {
        "simple_mean": [0.01, -0.01],
        "critic_weighted": [0.02, 0.00],
        "critic_top1": [0.03, -0.02],
    }.items():
        arr = np.asarray(pnl, dtype=float)
        strategies[name] = {
            "pnl": arr,
            "metrics": {
                "mean_return": float(arr.mean()),
                "win_rate": float(np.mean(arr > 0)),
            },
            "invalid_count": 0,
        }
    results = {"gross_long_short": strategies}
    output_dir = _local_tmp("figures")
    try:
        paths = plot_backtest_results(results, output_dir)
        assert {path.name for path in paths} == {
            "e2_pnl_distribution.png",
            "e2_win_rate.png",
            "e2_mean_return.png",
        }
        assert all(path.exists() and path.stat().st_size > 0 for path in paths)
        assert plt.get_fignums() == []
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_plot_backtest_results_handles_nonfinite_pnl_and_nan_metrics():
    strategies = {}
    for name in ("simple_mean", "critic_weighted", "critic_top1"):
        strategies[name] = {
            "pnl": np.array([np.nan, np.inf], dtype=float),
            "metrics": {"mean_return": np.nan, "win_rate": np.nan},
            "invalid_count": 2,
        }
    results = {"net_long_only": strategies}

    output_dir = _local_tmp("nan_figures")
    try:
        paths = plot_backtest_results(results, output_dir)
        assert len(paths) == 3
        assert all(path.exists() and path.stat().st_size > 0 for path in paths)
        assert plt.get_fignums() == []
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
