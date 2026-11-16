"""Kronos-aligned, fixed-cache evaluation for Momus.

This script is deliberately read-only with respect to model inputs.  It does
not train a critic, generate candidates, choose a checkpoint, or change the
candidate order.  It evaluates the existing new_pool_II caches with the five
existing calibrated Momus checkpoints.

Price IC and Price RankIC follow the Price Series axis described by Kronos:
for each window and each OHLC channel, correlate the predicted and observed
forecast series along the forecast horizon, then average the four channels.
The source materials label the two coefficients IC and RankIC but do not give
an executable coefficient definition; this implementation records the
standard Pearson/Spearman operationalization explicitly in protocol.json.

Run from the repository root:

    python examples/evaluate_kronos_alignment.py

Outputs are written to output/results/kronos_alignment_20260907/.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "output/results/kronos_alignment_20260907"
DEFAULT_CACHES = {
    "2024": ROOT / "output/candidates/new_pool_II_2024_N16.pkl",
    "2025": ROOT / "output/candidates/new_pool_II_2025_N16.pkl",
    "2026H1": ROOT / "output/candidates/new_pool_II_2026H1_N16.pkl",
}
DEFAULT_CHECKPOINTS = {
    str(seed): ROOT / "output/results/regime_generalization/multiregime_train"
    / f"seed_{seed}/fusion_best.pt"
    for seed in range(5)
}
OHLC = (0, 1, 2, 3)
METHODS = ("simple_mean", "median", "momus_weighted", "momus_top1")
METHOD_LABELS = {
    "simple_mean": "Kronos simple mean",
    "median": "pointwise median",
    "momus_weighted": "Momus weighted",
    "momus_top1": "Momus top-1 candidate",
}
SCALING_METHODS = ("simple_mean", "momus_weighted")
METRICS = ("ohlc_mse", "ohlc_mae", "price_ic", "price_rankic")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_numpy_critic_engine():
    """Load the already validated torch-free implementation without mutation."""
    path = ROOT / "output/results/new_pool_II_calibrated_reeval/np_engine.py"
    spec = importlib.util.spec_from_file_location("momus_np_engine_alignment", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load calibrated numpy engine: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalized_history_and_target(x_df, y_true, clip: float = 5.0):
    columns = ["open", "high", "low", "close", "volume", "amount"]
    hist = x_df[columns].to_numpy(dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.float32)
    mean = hist.mean(axis=0, keepdims=True)
    scale = hist.std(axis=0, keepdims=True) + 1e-5
    hist_norm = np.clip((hist - mean) / scale, -clip, clip).astype(np.float32)
    y_norm = ((y_true - mean) / scale).astype(np.float32)
    return hist_norm, y_norm, mean, scale


def _finite_pair(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def _pearson(a, b) -> float:
    a, b = _finite_pair(a, b)
    if len(a) < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denominator = math.sqrt(float(np.dot(a, a) * np.dot(b, b)))
    if denominator <= 0.0:
        return float("nan")
    return float(np.dot(a, b) / denominator)


def _spearman(a, b) -> float:
    a, b = _finite_pair(a, b)
    if len(a) < 2:
        return float("nan")
    return _pearson(rankdata(a, method="average"), rankdata(b, method="average"))


def _nanmean_or_nan(values) -> float:
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def kronos_price_correlations(pred, truth):
    """Return Price IC/RankIC over horizon, separately for the four OHLC axes."""
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if pred.ndim != 2 or truth.ndim != 2 or pred.shape != truth.shape:
        raise ValueError(f"prediction/target shape mismatch: {pred.shape} vs {truth.shape}")
    ic = [_pearson(pred[:, channel], truth[:, channel]) for channel in OHLC]
    rank_ic = [_spearman(pred[:, channel], truth[:, channel]) for channel in OHLC]
    return {
        "price_ic": _nanmean_or_nan(ic),
        "price_rankic": _nanmean_or_nan(rank_ic),
        "price_ic_valid_channels": int(np.isfinite(ic).sum()),
        "price_rankic_valid_channels": int(np.isfinite(rank_ic).sum()),
        "price_ic_open": ic[0],
        "price_ic_high": ic[1],
        "price_ic_low": ic[2],
        "price_ic_close": ic[3],
        "price_rankic_open": rank_ic[0],
        "price_rankic_high": rank_ic[1],
        "price_rankic_low": rank_ic[2],
        "price_rankic_close": rank_ic[3],
    }


def metric_row(pred, truth):
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    if not np.isfinite(pred[:, OHLC]).all() or not np.isfinite(truth[:, OHLC]).all():
        raise ValueError("non-finite OHLC prediction or target")
    error = pred[:, OHLC] - truth[:, OHLC]
    row = {
        "ohlc_mse": float(np.mean(error ** 2)),
        "ohlc_mae": float(np.mean(np.abs(error))),
    }
    row.update(kronos_price_correlations(pred[:, OHLC], truth[:, OHLC]))
    return row


def aggregate_metric_rows(rows):
    result = {}
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        result[metric] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
    result["windows"] = len(rows)
    result["valid_price_ic_channels"] = int(sum(row["price_ic_valid_channels"] for row in rows))
    result["valid_price_rankic_channels"] = int(
        sum(row["price_rankic_valid_channels"] for row in rows)
    )
    return result


def _stable_softmax(scores, temperature: float = 1.0):
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Momus critic returned a non-finite score")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = scores / temperature
    logits -= np.max(logits)
    weights = np.exp(logits)
    weights /= weights.sum()
    return weights


def strategy_paths(paths, hist_norm, critic, temperature: float):
    """Return the four requested aggregators for one fixed candidate subset."""
    paths = np.asarray(paths, dtype=np.float32)
    scores = np.asarray(critic.score(paths, hist_norm), dtype=float)
    weights = _stable_softmax(scores, temperature)
    return {
        "simple_mean": np.mean(paths, axis=0),
        "median": np.median(paths, axis=0),
        "momus_weighted": np.sum(weights[:, None, None] * paths, axis=0),
        "momus_top1": paths[int(np.argmax(scores))],
    }


def _date_text(value):
    if hasattr(value, "iloc"):
        value = value.iloc[-1]
    if hasattr(value, "date"):
        return str(value.date())
    return str(value)


def _new_accumulator():
    return defaultdict(lambda: defaultdict(list))


def _append_metrics(accumulator, key, metrics):
    for metric in METRICS:
        accumulator[key][metric].append(metrics[metric])
    for metric in ("price_ic_valid_channels", "price_rankic_valid_channels"):
        accumulator[key][metric].append(metrics[metric])


def _mean_values(values):
    return float(np.nanmean(np.asarray(values, dtype=float)))


def _std_values(values):
    values = np.asarray(values, dtype=float)
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def evaluate_period(label, cache_path: Path, critics, n_values, temperature, clip):
    with cache_path.open("rb") as handle:
        cache = pickle.load(handle)
    metadata = cache.get("metadata", {})
    windows = cache["windows"]
    candidates = cache["candidates"]
    if metadata.get("sample_count") != 16 or len(candidates) != len(windows):
        raise ValueError(f"{label}: cache is not the expected fixed N=16 schema")
    for n in n_values:
        if n < 1 or n > 16:
            raise ValueError(f"invalid candidate count N={n}")

    main_values = _new_accumulator()
    scaling_values = _new_accumulator()
    window_rows = []
    n1_checks = []

    for seed, critic in critics.items():
        for window_index, (window, path_array) in enumerate(zip(windows, candidates)):
            code, x_df, x_ts, y_ts, y_true = window
            hist_norm, truth_norm, mean, scale = normalized_history_and_target(
                x_df, y_true, clip=clip
            )
            paths = np.asarray(path_array, dtype=np.float32)
            if paths.shape[0] != 16 or paths.shape[1:] != truth_norm.shape:
                raise ValueError(
                    f"{label}/{code}/{window_index}: paths {paths.shape}, target {truth_norm.shape}"
                )

            # Every N is a deterministic prefix of the existing 16 candidates.
            for n in n_values:
                subset = paths[:n]
                predictions = strategy_paths(subset, hist_norm, critic, temperature)
                for method in SCALING_METHODS:
                    metrics = metric_row(predictions[method], truth_norm)
                    _append_metrics(scaling_values, (label, seed, n, method), metrics)
                    if n == 16:
                        _append_metrics(main_values, (label, seed, method), metrics)

                if n == 1:
                    simple = metric_row(predictions["simple_mean"], truth_norm)
                    weighted = metric_row(predictions["momus_weighted"], truth_norm)
                    n1_checks.append(
                        max(abs(simple[m] - weighted[m]) for m in METRICS if np.isfinite(simple[m]))
                    )

                if n == 16:
                    # The main table includes all four methods.  The scoring
                    # call above already produced the two learned methods; the
                    # deterministic methods are included in the same sample.
                    for method in ("median", "momus_top1"):
                        metrics = metric_row(predictions[method], truth_norm)
                        _append_metrics(main_values, (label, seed, method), metrics)
                    for method in METHODS:
                        metrics = metric_row(predictions[method], truth_norm)
                        window_rows.append(
                            {
                                "period": label,
                                "seed": int(seed),
                                "code": str(code),
                                "window_index": int(window_index),
                                "origin": _date_text(x_ts),
                                "entry": _date_text(y_ts.iloc[0] if hasattr(y_ts, "iloc") else y_ts[0]),
                                "horizon_end": _date_text(y_ts),
                                "candidate_count": 16,
                                "method": method,
                                **metrics,
                            }
                        )

            # Check that inverse affine transformation would not change Price
            # IC/RankIC.  This is a local invariant, not a second metric path.
            if window_index == 0 and seed == min(critics):
                raw_pred = predictions["simple_mean"] * scale + mean
                raw_truth = truth_norm * scale + mean
                normalized_corr = kronos_price_correlations(
                    predictions["simple_mean"][:, OHLC], truth_norm[:, OHLC]
                )
                raw_corr = kronos_price_correlations(
                    raw_pred[:, OHLC], raw_truth[:, OHLC]
                )
                for metric in ("price_ic", "price_rankic"):
                    if not np.isclose(normalized_corr[metric], raw_corr[metric], atol=1e-6):
                        raise AssertionError(f"de-normalization changed {metric}")

    return {
        "label": label,
        "metadata": metadata,
        "cache_path": cache_path,
        "main_values": main_values,
        "scaling_values": scaling_values,
        "window_rows": window_rows,
        "n1_max_abs_metric_difference": float(max(n1_checks) if n1_checks else 0.0),
    }


def build_per_seed_rows(period_results):
    rows = []
    for result in period_results:
        for (period, seed, method), values in sorted(result["main_values"].items()):
            base = _mean_values(result["main_values"][(period, seed, "simple_mean")]["ohlc_mse"])
            row = {
                "period": period,
                "seed": int(seed),
                "method": method,
                "method_label": METHOD_LABELS[method],
                "candidate_count": 16,
                "windows": len(values["ohlc_mse"]),
            }
            for metric in METRICS:
                row[metric] = _mean_values(values[metric])
            row["mse_improvement_pct"] = (base - row["ohlc_mse"]) / base * 100.0
            row["price_ic_valid_channels"] = int(sum(values.get("price_ic_valid_channels", [])))
            row["price_rankic_valid_channels"] = int(
                sum(values.get("price_rankic_valid_channels", []))
            )
            rows.append(row)
    return rows


def build_scaling_per_seed_rows(period_results):
    rows = []
    for result in period_results:
        for (period, seed, n, method), values in sorted(result["scaling_values"].items()):
            base = _mean_values(result["scaling_values"][(period, seed, n, "simple_mean")]["ohlc_mse"])
            row = {
                "period": period,
                "seed": int(seed),
                "candidate_count": int(n),
                "method": method,
                "method_label": METHOD_LABELS[method],
                "windows": len(values["ohlc_mse"]),
            }
            for metric in METRICS:
                row[metric] = _mean_values(values[metric])
            row["mse_improvement_pct"] = (base - row["ohlc_mse"]) / base * 100.0
            rows.append(row)
    return rows


def summarize_seed_rows(rows, group_columns):
    frame = pd.DataFrame(rows)
    output = []
    for keys, group in frame.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["seed_count"] = int(group["seed"].nunique())
        row["windows"] = int(group["windows"].iloc[0])
        for metric in METRICS + ("mse_improvement_pct",):
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_seed_std"] = _std_values(group[metric].to_numpy())
        output.append(row)
    return pd.DataFrame(output)


def build_main_table(summary):
    table = summary.copy()
    table["Method"] = table["method_label"]
    table["OHLC-MSE"] = table["ohlc_mse_mean"]
    table["OHLC-MAE"] = table["ohlc_mae_mean"]
    table["Price IC"] = table["price_ic_mean"]
    table["Price RankIC"] = table["price_rankic_mean"]
    table["MSE improvement (%)"] = table["mse_improvement_pct_mean"]
    columns = [
        "period", "Method", "OHLC-MSE", "OHLC-MAE", "Price IC", "Price RankIC",
        "MSE improvement (%)", "seed_count", "windows",
    ]
    table["_method_order"] = pd.Categorical(table["method"], categories=METHODS, ordered=True)
    return table[columns + ["_method_order"]].sort_values(
        ["period", "_method_order"]
    ).drop(columns=["_method_order"])


def build_scaling_crossovers(scaling_summary, n_values):
    rows = []
    for period in sorted(scaling_summary.period.unique()):
        part = scaling_summary[scaling_summary.period == period]
        simple = part[part.method == "simple_mean"].set_index("candidate_count")
        momus = part[part.method == "momus_weighted"].set_index("candidate_count")
        for momus_n in n_values:
            for simple_n in n_values:
                if momus_n >= simple_n:
                    continue
                comparisons = {
                    "ohlc_mse": momus.loc[momus_n, "ohlc_mse_mean"] <= simple.loc[simple_n, "ohlc_mse_mean"],
                    "price_ic": momus.loc[momus_n, "price_ic_mean"] >= simple.loc[simple_n, "price_ic_mean"],
                    "price_rankic": momus.loc[momus_n, "price_rankic_mean"] >= simple.loc[simple_n, "price_rankic_mean"],
                }
                for metric, matches in comparisons.items():
                    rows.append(
                        {
                            "period": period,
                            "momus_candidate_count": int(momus_n),
                            "simple_mean_candidate_count": int(simple_n),
                            "metric": metric,
                            "momus_value": float(momus.loc[momus_n, f"{metric}_mean"]),
                            "simple_mean_value": float(simple.loc[simple_n, f"{metric}_mean"]),
                            "momus_matches_or_beats": bool(matches),
                        }
                    )
    return pd.DataFrame(rows)


def plot_scaling(scaling_summary: pd.DataFrame, output_path: Path, n_values):
    colors = {"2024": "#1f77b4", "2025": "#ff7f0e", "2026H1": "#2ca02c"}
    panels = [
        ("ohlc_mse", "OHLC-MSE (lower is better)"),
        ("price_ic", "Price IC (higher is better)"),
        ("price_rankic", "Price RankIC (higher is better)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.0), constrained_layout=False)
    for ax, (metric, title) in zip(axes, panels):
        for period in ("2024", "2025", "2026H1"):
            color = colors[period]
            for method, linestyle, marker in (
                ("simple_mean", "--", "o"),
                ("momus_weighted", "-", "s"),
            ):
                part = scaling_summary[
                    (scaling_summary.period == period) & (scaling_summary.method == method)
                ].sort_values("candidate_count")
                ax.plot(
                    part.candidate_count,
                    part[f"{metric}_mean"],
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    linewidth=1.5,
                    markersize=4,
                    label=f"{period} {'simple' if method == 'simple_mean' else 'Momus'}",
                )
        ax.set_xticks(n_values)
        ax.set_xlabel("candidate count N")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("mean across five seeds")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=False)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.25, wspace=0.25)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_protocol(output_dir, caches, checkpoints, n_values, temperature, clip, period_results):
    protocol = {
        "scope": "Kronos-aligned fixed-cache evaluation; no training or candidate generation",
        "candidate_subset_rule": "For each cached N=16 window, use candidates[:N] for N in the declared order.",
        "candidate_counts": [int(n) for n in n_values],
        "temperature": float(temperature),
        "history_normalization": "hist_norm=clip((hist-mean)/(std+1e-5), -5, 5); target uses same mean/std without future statistics",
        "error_space": "existing Momus normalized-space OHLC-MSE and OHLC-MAE",
        "price_metric_space": "window-standardized OHLC; inverse affine transformation is checked to leave correlations unchanged",
        "price_ic_axis": "within each window, each OHLC channel, forecast-horizon axis; Pearson correlation; average four channels, then average windows",
        "price_rankic_axis": "within each window, each OHLC channel, forecast-horizon axis; average-rank Spearman correlation; average four channels, then average windows",
        "coefficient_status": "The Kronos materials identify IC/RankIC and the Price Series axis but do not expose an executable coefficient definition; Pearson/Spearman is the explicit conventional operationalization used here.",
        "main_methods": {method: METHOD_LABELS[method] for method in METHODS},
        "scaling_methods": {method: METHOD_LABELS[method] for method in SCALING_METHODS},
        "caches": {
            label: {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "metadata": result["metadata"],
                "n1_common_start_max_abs_metric_difference": result["n1_max_abs_metric_difference"],
            }
            for label, path, result in zip(caches, [caches[k] for k in caches], period_results)
        },
        "checkpoints": {
            str(seed): {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for seed, path in checkpoints.items()
        },
        "period_results": {
            result["label"]: {
                "cache_windows": int(result["metadata"].get("total_windows", 0)),
                "n1_common_start_max_abs_metric_difference": result["n1_max_abs_metric_difference"],
            }
            for result in period_results
        },
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_report(output_dir, main_table, scaling_summary, crossover, n1_diffs):
    lines = [
        "# Momus–Kronos 对齐评价（固定缓存）",
        "",
        "本目录只包含固定 `new_pool_II` 候选缓存和既有 calibrated Momus 权重的再评价；未重训、未重新生成候选、未按新增指标选择 checkpoint。",
        "",
        "## 价格评价实现",
        "",
        "- Price IC：每个窗口、每个 OHLC 通道，在 10 步预测 horizon 上计算 Pearson 相关，再平均四个通道和窗口。",
        "- Price RankIC：同一轴上先使用 average rank，再计算 Spearman 相关；同样平均四个通道和窗口。",
        "- 源材料没有给出可执行的相关系数函数；因此代码将 Pearson/Spearman 作为明确的 conventional operationalization，并在 `protocol.json` 标注，不能把这部分表述成官方代码中存在的函数复现。",
        "- 价格相关指标不需要反标准化：每个窗口每个通道只做正尺度仿射变换，相关系数不变；代码对首个窗口做了反标准化不变性检查。",
        "",
        "## 主结果（五个 seed 的均值）",
        "",
        "| period | Method | OHLC-MSE | OHLC-MAE | Price IC | Price RankIC | MSE improvement (%) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in main_table.iterrows():
        lines.append(
            f"| {row['period']} | {row['Method']} | {row['OHLC-MSE']:.6f} | {row['OHLC-MAE']:.6f} | "
            f"{row['Price IC']:.6f} | {row['Price RankIC']:.6f} | {row['MSE improvement (%)']:.3f}% |"
        )
    lines += [
        "",
        "## Test-Time Scaling",
        "",
        "候选数使用固定缓存的前 N 条，N=" + ", ".join(map(str, sorted(scaling_summary.candidate_count.unique()))) + "；实线是 Momus weighted，虚线是 simple mean。图中均为五个 seed 的均值，seed 离散度保存在 `scaling_summary.csv`。",
        "",
        "![Test-Time Scaling](test_time_scaling.png)",
        "",
        "逐指标的跨 N 对照见 `scaling_crossovers.csv`。这些比较只说明评价指标上的数值匹配，不等同于计算效率或推理成本节省。",
        "",
        "## N=1",
        "",
        f"N=1 时 Momus weighted 与 simple mean 的最大指标差为 {max(n1_diffs):.3e}；作为共同起点，不解释为 Momus 的加权优势。",
        "",
        "## 投资评价",
        "",
        "本轮没有重新运行投资回测。Kronos 投资协议与当前 Momus 回测的差异及后续实现清单见 `investment_protocol_gap.md`。",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_investment_gap(output_dir):
    text = """# 投资协议对齐准备（本轮不运行回测）

## 后续需要实现的 Kronos 口径

| 项目 | Kronos 论文/扩展材料 | 当前 Momus 已有回测 | 对齐状态 |
|---|---|---|---|
| predictive signal | `R=(mean_{h=1..H} predicted close_h - current close) / current close`；日频 H=10 | `raw_close.mean()/last_close-1` | 公式已对齐论文；仍需在最终实现中固定字段和反标准化位置 |
| 股票池 | 沪深300或中证800；Kronos repo 默认 csi300 | 固定主题股票池 `new_pool_II`，86 只 | 不一致；本轮不引入新股票池 |
| 调仓/执行 | 日频产生 signal；Top-k/drop-n；Qlib daily executor，延迟到下一执行日 | 仅固定预测 origin 形成 signal；下一交易日开盘执行 | 执行时序相近，但信号日期覆盖规则不同 |
| top-k | CSI300: k=50, n_drop=5；CSI800: k=200, n_drop=10 | k=50, 每次最多替换 5 只 | 参数只在 k=50 这一档相同；股票池不同 |
| long-only | 论文描述等权持有 top-k；当前 Momus 不卖空 | 当前 Momus long-only | 对齐 |
| minimum holding | Kronos repo/config `hold_thresh=5` | `min_hold=5` | 对齐 |
| transaction cost | 论文写每次交易 0.15%；官方 Qlib demo 配置 open_cost=.001、close_cost=.0015、min_cost=5 | 当前 Momus 买卖均按 0.15% 扣除，分数股，未设 min cost | 论文层面接近；代码层面仍有成本模型差异，不能静默混用 |
| benchmark | Kronos repo 映射 CSI300/800/1000 指数；AER/IR 对主动收益计算 | 当前 Momus 用研究股票池同池等权组合 | 不一致；需明确 benchmark 后再计算 AER/IR |
| AER / IR | Qlib `risk_analysis` 的主动收益序列 `strategy - benchmark`；当前 repo 默认 `mode="sum"`，实现日尺度因版本需锁定 | `AER=252*mean(active)`，`IR=sqrt(252)*mean(active)/sample_std(active)` | 公式形式接近，但 Qlib 版本、日尺度常数、成本口径需先冻结 |

## 不能在本轮声称

- 不能把现有 Momus 同池等权 benchmark 的 AER/IR 直接命名为 Kronos 官方投资结果。
- 不能把官方 Qlib demo 的标准化价格差 signal 与论文中的百分比 signal 静默视为同一个定义；后续应选择并记录单一主口径。
- 本轮不新增股票池、不修改 k/drop/holding/cost，不为得到更好收益而改回测规则。
"""
    (output_dir / "investment_protocol_gap.md").write_text(text, encoding="utf-8")


def parse_key_value(items, defaults):
    if not items:
        return dict(defaults)
    result = {}
    for item in items:
        key, value = item.split("=", 1)
        result[key] = Path(value)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", action="append", help="period=cache.pkl; repeatable")
    parser.add_argument("--checkpoint", action="append", help="seed=checkpoint.pt; repeatable")
    parser.add_argument("--n-values", default="1,4,8,16")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--clip", type=float, default=5.0)
    args = parser.parse_args()

    caches = parse_key_value(args.cache, DEFAULT_CACHES)
    checkpoints = parse_key_value(args.checkpoint, DEFAULT_CHECKPOINTS)
    n_values = tuple(sorted({int(item) for item in args.n_values.split(",") if item.strip()}))
    if not caches or not checkpoints:
        raise ValueError("at least one cache and one checkpoint are required")
    if 16 not in n_values:
        raise ValueError("N=16 is required for the main fixed-budget table")
    if any(not path.exists() for path in list(caches.values()) + list(checkpoints.values())):
        missing = [str(path) for path in list(caches.values()) + list(checkpoints.values()) if not path.exists()]
        raise FileNotFoundError("missing fixed input(s): " + ", ".join(missing))

    engine = load_numpy_critic_engine()
    critics = {int(seed): engine.load_dual_calibrated(path) for seed, path in checkpoints.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    period_results = []
    all_window_rows = []
    for label, cache_path in caches.items():
        result = evaluate_period(
            label,
            cache_path,
            critics,
            n_values,
            args.temperature,
            args.clip,
        )
        period_results.append(result)
        all_window_rows.extend(result["window_rows"])
        print(f"evaluated {label}: {len(result['window_rows'])} main window rows", flush=True)

    per_seed = pd.DataFrame(build_per_seed_rows(period_results))
    per_seed.to_csv(args.output_dir / "forecast_per_seed.csv", index=False, float_format="%.10g")
    forecast_summary = summarize_seed_rows(per_seed.to_dict("records"), ["period", "method", "method_label", "candidate_count"])
    forecast_summary.to_csv(args.output_dir / "forecast_summary.csv", index=False, float_format="%.10g")
    main_table = build_main_table(forecast_summary)
    main_table.to_csv(args.output_dir / "main_table.csv", index=False, float_format="%.10g")

    scaling_per_seed = pd.DataFrame(build_scaling_per_seed_rows(period_results))
    scaling_per_seed.to_csv(args.output_dir / "scaling_per_seed.csv", index=False, float_format="%.10g")
    scaling_summary = summarize_seed_rows(
        scaling_per_seed.to_dict("records"),
        ["period", "candidate_count", "method", "method_label"],
    )
    scaling_summary.to_csv(args.output_dir / "scaling_summary.csv", index=False, float_format="%.10g")

    crossover = build_scaling_crossovers(scaling_summary, n_values)
    crossover.to_csv(args.output_dir / "scaling_crossovers.csv", index=False, float_format="%.10g")
    all_three = crossover.groupby(
        ["period", "momus_candidate_count", "simple_mean_candidate_count"], as_index=False
    ).agg(
        metrics=("metric", "count"),
        all_three_metrics=("momus_matches_or_beats", "all"),
    )
    all_three.to_csv(args.output_dir / "scaling_crossovers_all_three.csv", index=False)

    pd.DataFrame(all_window_rows).to_csv(
        args.output_dir / "forecast_window_metrics.csv", index=False, float_format="%.10g"
    )
    n1_diffs = [result["n1_max_abs_metric_difference"] for result in period_results]
    checks = {
        "fixed_cache_only": True,
        "candidate_counts": list(n_values),
        "main_candidate_count": 16,
        "n1_common_start_max_abs_metric_difference": float(max(n1_diffs)),
        "n1_common_start_pass": bool(max(n1_diffs) <= 1e-6),
        "main_window_rows": len(all_window_rows),
        "per_seed_rows": len(per_seed),
        "scaling_per_seed_rows": len(scaling_per_seed),
    }
    (args.output_dir / "checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_protocol(args.output_dir, caches, checkpoints, n_values, args.temperature, args.clip, period_results)
    write_investment_gap(args.output_dir)
    plot_scaling(scaling_summary, args.output_dir / "test_time_scaling.png", n_values)
    write_report(args.output_dir, main_table, scaling_summary, crossover, n1_diffs)

    print(f"output={args.output_dir}")
    print(main_table.to_string(index=False))


if __name__ == "__main__":
    main()
