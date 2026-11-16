"""Evaluate the frozen HS300 candidate caches without resampling.

Compared with the earlier generic alignment evaluator, this runner computes
the Momus scores once per ``seed x window`` and reuses them for every prefix
N.  This changes only execution time, not the fixed-cache protocol or any
metric definition.
"""
from __future__ import annotations

import argparse
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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "output/results/hs300_main_20260907"
DEFAULT_CACHES = {
    "2024": ROOT / "output/candidates/hs300_2024_N16.pkl",
    "2025": ROOT / "output/candidates/hs300_2025_N16.pkl",
    "2026H1": ROOT / "output/candidates/hs300_2026H1_N16.pkl",
}
DEFAULT_CHECKPOINTS = {
    str(seed): ROOT / "output/results/regime_generalization/multiregime_train"
    / f"seed_{seed}/fusion_best.pt"
    for seed in range(5)
}
OHLC = (0, 1, 2, 3)
METHODS = ("simple_mean", "median", "momus_weighted", "momus_top1")
SCALING_METHODS = ("simple_mean", "momus_weighted")
METRICS = ("ohlc_mse", "ohlc_mae", "price_ic", "price_rankic")
METHOD_LABELS = {
    "simple_mean": "Kronos simple mean",
    "median": "pointwise median",
    "momus_weighted": "Momus weighted",
    "momus_top1": "Momus top-1 candidate",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_numpy_engine():
    path = ROOT / "output/results/new_pool_II_calibrated_reeval/np_engine.py"
    spec = importlib.util.spec_from_file_location("momus_np_engine_hs300", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def normalized_history_and_target(x_df, y_true, clip=5.0):
    columns = ["open", "high", "low", "close", "volume", "amount"]
    hist = x_df[columns].to_numpy(dtype=np.float32)
    target = np.asarray(y_true, dtype=np.float32)
    mean = hist.mean(axis=0, keepdims=True)
    scale = hist.std(axis=0, keepdims=True) + 1e-5
    hist_norm = np.clip((hist - mean) / scale, -clip, clip).astype(np.float32)
    target_norm = ((target - mean) / scale).astype(np.float32)
    return hist_norm, target_norm, mean, scale


def average_rank(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def correlation(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float(np.dot(a, a) * np.dot(b, b)))
    if denom <= 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def price_metrics(pred, truth):
    pred = np.asarray(pred, dtype=float)
    truth = np.asarray(truth, dtype=float)
    error = pred - truth
    ic = []
    rank_ic = []
    for channel in range(4):
        ic.append(correlation(pred[:, channel], truth[:, channel]))
        rank_ic.append(correlation(average_rank(pred[:, channel]), average_rank(truth[:, channel])))
    finite_ic = [x for x in ic if np.isfinite(x)]
    finite_rank = [x for x in rank_ic if np.isfinite(x)]
    return {
        "ohlc_mse": float(np.mean(error ** 2)),
        "ohlc_mae": float(np.mean(np.abs(error))),
        "price_ic": float(np.mean(finite_ic)) if finite_ic else float("nan"),
        "price_rankic": float(np.mean(finite_rank)) if finite_rank else float("nan"),
        "price_ic_valid_channels": len(finite_ic),
        "price_rankic_valid_channels": len(finite_rank),
    }


def stable_softmax(scores):
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("non-finite Momus score")
    logits = scores - np.max(scores)
    weights = np.exp(logits)
    return weights / weights.sum()


def aggregate_paths(paths, scores, n):
    subset = np.asarray(paths[:n], dtype=np.float32)
    subset_scores = np.asarray(scores[:n], dtype=float)
    weights = stable_softmax(subset_scores)
    return {
        "simple_mean": np.mean(subset, axis=0),
        "median": np.median(subset, axis=0),
        "momus_weighted": np.sum(weights[:, None, None] * subset, axis=0),
        "momus_top1": subset[int(np.argmax(subset_scores))],
    }


def date_text(value):
    if hasattr(value, "date"):
        return str(value.date())
    return str(value)


def append_metric(accumulator, key, metrics):
    for metric in METRICS:
        accumulator[key][metric].append(metrics[metric])


def mean_value(values):
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def std_value(values):
    values = np.asarray(values, dtype=float)
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def evaluate_period(label, cache_path, critics, n_values, clip):
    with cache_path.open("rb") as handle:
        cache = pickle.load(handle)
    metadata = cache["metadata"]
    windows = cache["windows"]
    candidates = cache["candidates"]
    if metadata.get("sample_count") != 16 or metadata.get("lookback") != 90 or metadata.get("pred_len") != 10:
        raise ValueError(f"{label}: cache metadata is not the frozen HS300 protocol")
    if len(windows) != len(candidates):
        raise ValueError(f"{label}: window/candidate count mismatch")

    main_values = defaultdict(lambda: defaultdict(list))
    scaling_values = defaultdict(lambda: defaultdict(list))
    window_rows = []
    n1_diffs = []

    for seed, critic in critics.items():
        for window_index, (window, path_array) in enumerate(zip(windows, candidates)):
            code, x_df, x_ts, y_ts, y_true = window
            hist_norm, truth_norm, mean, scale = normalized_history_and_target(x_df, y_true, clip)
            paths = np.asarray(path_array, dtype=np.float32)
            if paths.shape != (16, 10, 6) or truth_norm.shape != (10, 6):
                raise ValueError(f"{label}/{code}/{window_index}: {paths.shape} / {truth_norm.shape}")

            # The expensive Momus forward pass is intentionally done once.
            scores = np.asarray(critic.score(paths, hist_norm), dtype=float)
            for n in n_values:
                predictions = aggregate_paths(paths, scores, n)
                simple_metrics = price_metrics(predictions["simple_mean"][:, OHLC], truth_norm[:, OHLC])
                weighted_metrics = price_metrics(predictions["momus_weighted"][:, OHLC], truth_norm[:, OHLC])
                append_metric(scaling_values, (label, seed, n, "simple_mean"), simple_metrics)
                append_metric(scaling_values, (label, seed, n, "momus_weighted"), weighted_metrics)
                if n == 1:
                    n1_diffs.append(
                        max(abs(simple_metrics[m] - weighted_metrics[m])
                            for m in METRICS
                            if np.isfinite(simple_metrics[m]) and np.isfinite(weighted_metrics[m]))
                    )
                if n == 16:
                    for method in METHODS:
                        metrics = price_metrics(predictions[method][:, OHLC], truth_norm[:, OHLC])
                        append_metric(main_values, (label, seed, method), metrics)
                        window_rows.append({
                            "period": label,
                            "seed": int(seed),
                            "code": str(code),
                            "window_index": int(window_index),
                            "origin": date_text(x_ts.iloc[-1] if hasattr(x_ts, "iloc") else x_ts[-1]),
                            "entry": date_text(y_ts.iloc[0] if hasattr(y_ts, "iloc") else y_ts[0]),
                            "horizon_end": date_text(y_ts.iloc[-1] if hasattr(y_ts, "iloc") else y_ts[-1]),
                            "candidate_count": 16,
                            "method": method,
                            **metrics,
                        })

            # Correlation is invariant to the positive per-window affine inverse transform.
            first = aggregate_paths(paths, scores, 16)["simple_mean"]
            # Cast before the inverse affine transform.  The cached paths
            # and normalized targets are float32; doing the round-trip in
            # float32 can reorder near-tied values and make a rank
            # correlation look different even though the mathematical
            # positive-affine invariance still holds.
            raw_pred = (
                first.astype(np.float64) * scale.astype(np.float64)
                + mean.astype(np.float64)
            )
            raw_truth = (
                truth_norm.astype(np.float64) * scale.astype(np.float64)
                + mean.astype(np.float64)
            )
            for metric, raw_metric in (("price_ic", "price_ic"), ("price_rankic", "price_rankic")):
                lhs = price_metrics(first[:, OHLC], truth_norm[:, OHLC])[metric]
                rhs = price_metrics(raw_pred[:, OHLC], raw_truth[:, OHLC])[raw_metric]
                # The remaining tolerance is only for floating point
                # arithmetic; it does not alter the reported metric.
                if not np.isclose(lhs, rhs, atol=1e-12):
                    raise AssertionError(f"{label}: inverse transform changed {metric}")

    return {
        "label": label,
        "metadata": metadata,
        "cache_path": cache_path,
        "main_values": main_values,
        "scaling_values": scaling_values,
        "window_rows": window_rows,
        "n1_max_abs_metric_difference": float(max(n1_diffs) if n1_diffs else 0.0),
    }


def rows_from_values(results, values_key, methods):
    rows = []
    for result in results:
        values = result[values_key]
        for key, metrics_values in sorted(values.items(), key=str):
            if values_key == "main_values":
                period, seed, method = key
                candidate_count = 16
            else:
                period, seed, candidate_count, method = key
            base_key = (period, seed, "simple_mean") if values_key == "main_values" else (period, seed, candidate_count, "simple_mean")
            base = mean_value(values[base_key]["ohlc_mse"])
            row = {
                "period": period,
                "seed": int(seed),
                "candidate_count": int(candidate_count),
                "method": method,
                "method_label": METHOD_LABELS[method],
                "windows": len(metrics_values["ohlc_mse"]),
            }
            for metric in METRICS:
                row[metric] = mean_value(metrics_values[metric])
            row["mse_improvement_pct"] = (base - row["ohlc_mse"]) / base * 100.0
            rows.append(row)
    return rows


def summarize(rows, group_columns):
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
            row[f"{metric}_seed_std"] = std_value(group[metric].to_numpy())
        output.append(row)
    return pd.DataFrame(output)


def write_scaling_plot(summary, path, n_values):
    panels = [("ohlc_mse", "OHLC-MSE (lower is better)"), ("price_ic", "Price IC"), ("price_rankic", "Price RankIC")]
    colors = {"2024": "#1f77b4", "2025": "#ff7f0e", "2026H1": "#2ca02c"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for axis, (metric, title) in zip(axes, panels):
        for period in ("2024", "2025", "2026H1"):
            for method, style, marker in (("simple_mean", "--", "o"), ("momus_weighted", "-", "s")):
                part = summary[(summary.period == period) & (summary.method == method)].sort_values("candidate_count")
                axis.plot(part.candidate_count, part[f"{metric}_mean"], color=colors[period], linestyle=style,
                           marker=marker, linewidth=1.5, markersize=4,
                           label=f"{period} {'Mean' if method == 'simple_mean' else 'Momus'}")
        axis.set_xticks(n_values)
        axis.set_title(title)
        axis.set_xlabel("candidate count N")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("mean across five Momus checkpoints")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0.18, 1, 1])
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=False)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_outputs(output_dir, results, checkpoints, n_values, protocol_path):
    output_dir.mkdir(parents=True, exist_ok=True)
    main_rows = rows_from_values(results, "main_values", METHODS)
    scaling_rows = rows_from_values(results, "scaling_values", SCALING_METHODS)
    main_df = pd.DataFrame(main_rows)
    scaling_df = pd.DataFrame(scaling_rows)
    main_df.to_csv(output_dir / "forecast_per_seed.csv", index=False, float_format="%.10g")
    scaling_df.to_csv(output_dir / "scaling_per_seed.csv", index=False, float_format="%.10g")
    main_summary = summarize(main_rows, ["period", "method", "method_label", "candidate_count"])
    scaling_summary = summarize(scaling_rows, ["period", "candidate_count", "method", "method_label"])
    main_summary.to_csv(output_dir / "forecast_summary.csv", index=False, float_format="%.10g")
    scaling_summary.to_csv(output_dir / "scaling_summary.csv", index=False, float_format="%.10g")

    main_table = main_summary.copy()
    main_table["Method"] = main_table["method_label"]
    main_table["OHLC-MSE"] = main_table["ohlc_mse_mean"]
    main_table["OHLC-MAE"] = main_table["ohlc_mae_mean"]
    main_table["Price IC"] = main_table["price_ic_mean"]
    main_table["Price RankIC"] = main_table["price_rankic_mean"]
    main_table["MSE improvement (%)"] = main_table["mse_improvement_pct_mean"]
    main_table["_order"] = pd.Categorical(main_table["method"], categories=METHODS, ordered=True)
    main_table = main_table.sort_values(["period", "_order"]).drop(columns=["_order"])
    main_table[["period", "Method", "OHLC-MSE", "OHLC-MAE", "Price IC", "Price RankIC", "MSE improvement (%)", "seed_count", "windows"]].to_csv(
        output_dir / "main_table.csv", index=False, float_format="%.10g"
    )

    window_rows = [row for result in results for row in result["window_rows"]]
    pd.DataFrame(window_rows).to_csv(output_dir / "forecast_window_metrics.csv", index=False, float_format="%.10g")

    crossover_rows = []
    for period in sorted(scaling_summary.period.unique()):
        part = scaling_summary[scaling_summary.period == period]
        simple = part[part.method == "simple_mean"].set_index("candidate_count")
        momus = part[part.method == "momus_weighted"].set_index("candidate_count")
        for momus_n in n_values:
            for simple_n in n_values:
                if momus_n >= simple_n:
                    continue
                for metric, higher in (("ohlc_mse", False), ("price_ic", True), ("price_rankic", True)):
                    lhs = momus.loc[momus_n, f"{metric}_mean"]
                    rhs = simple.loc[simple_n, f"{metric}_mean"]
                    crossover_rows.append({
                        "period": period,
                        "momus_candidate_count": int(momus_n),
                        "simple_mean_candidate_count": int(simple_n),
                        "metric": metric,
                        "momus_value": float(lhs),
                        "simple_mean_value": float(rhs),
                        "momus_matches_or_beats": bool(lhs <= rhs if not higher else lhs >= rhs),
                    })
    crossover = pd.DataFrame(crossover_rows)
    crossover.to_csv(output_dir / "scaling_crossovers.csv", index=False, float_format="%.10g")
    if len(crossover):
        all_three = crossover.groupby(["period", "momus_candidate_count", "simple_mean_candidate_count"], as_index=False).agg(
            metrics=("metric", "count"), all_three_metrics=("momus_matches_or_beats", "all")
        )
    else:
        all_three = pd.DataFrame()
    all_three.to_csv(output_dir / "scaling_crossovers_all_three.csv", index=False)
    write_scaling_plot(scaling_summary, output_dir / "test_time_scaling.png", n_values)

    n1 = [result["n1_max_abs_metric_difference"] for result in results]
    checks = {
        "fixed_hs300_cache_only": True,
        "candidate_counts": list(n_values),
        "main_candidate_count": 16,
        "n1_common_start_max_abs_metric_difference": float(max(n1)),
        "n1_common_start_pass": bool(max(n1) <= 1e-6),
        "main_window_rows": len(window_rows),
        "per_seed_rows": len(main_df),
        "scaling_per_seed_rows": len(scaling_df),
    }
    (output_dir / "checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# HS300 主预测与 Test-Time Scaling",
        "",
        "本报告只使用三份冻结 HS300 N=16 候选缓存和五个既有 calibrated Momus checkpoint；没有重训、重新采样或按结果选 checkpoint。",
        "Price IC/RankIC 沿 Kronos Price Series 的 horizon × OHLC 轴计算；Kronos 材料没有暴露可执行的系数 estimator，因此这里明确采用 Pearson / average-rank Spearman operationalization，而不声称存在未公开的官方函数。",
        "",
        "## 主预测表（五 seed 均值）",
        "",
        "| period | Method | OHLC-MSE | OHLC-MAE | Price IC | Price RankIC | MSE improvement (%) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in main_table.iterrows():
        lines.append(f"| {row['period']} | {row['Method']} | {row['OHLC-MSE']:.6f} | {row['OHLC-MAE']:.6f} | {row['Price IC']:.6f} | {row['Price RankIC']:.6f} | {row['MSE improvement (%)']:.3f}% |")
    lines += [
        "",
        "## Test-Time Scaling",
        "",
        "N 使用固定缓存的前缀 `{1,4,8,16}`；实线为 Momus weighted，虚线为 simple mean。",
        "",
        "![Test-Time Scaling](test_time_scaling.png)",
        "",
        f"N=1 的五 seed/三期最大逐指标差为 `{max(n1):.3e}`，按协议作为共同起点。",
        "",
        "较小 N 与较大 N 的数值交叉比较见 `scaling_crossovers.csv`；没有推理成本测量，因此不把数值匹配解释为计算效率。",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    protocol = {
        "scope": "HS300 fixed-cache main forecast and test-time scaling",
        "frozen_protocol_path": str(protocol_path.relative_to(ROOT)),
        "frozen_protocol_sha256": sha256(protocol_path),
        "candidate_subset_rule": "candidates[:N] in original generated order",
        "candidate_counts": list(map(int, n_values)),
        "normalization": "history mean/std + 1e-5, clip [-5,5]; target uses same history affine transform",
        "price_metric_axis": "per window, per OHLC channel, over 10-step horizon; average four channels then windows",
        "price_metric_operationalization": "Pearson and average-rank Spearman as recorded in the current Momus alignment implementation",
        "caches": {
            result["label"]: {
                "path": str(result["cache_path"].relative_to(ROOT)),
                "sha256": sha256(result["cache_path"]),
                "metadata": result["metadata"],
                "n1_max_abs_metric_difference": result["n1_max_abs_metric_difference"],
            }
            for result in results
        },
        "checkpoints": {
            str(seed): {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for seed, path in checkpoints.items()
        },
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_key_values(items, defaults):
    if not items:
        return dict(defaults)
    output = {}
    for item in items:
        key, value = item.split("=", 1)
        output[key] = Path(value)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache", action="append")
    parser.add_argument("--checkpoint", action="append")
    parser.add_argument("--n-values", default="1,4,8,16")
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--frozen-protocol", type=Path, default=ROOT / "docs/development/protocols/2026-09-07_hs300_main_preregistration.md")
    args = parser.parse_args(argv)
    caches = parse_key_values(args.cache, DEFAULT_CACHES)
    checkpoints = parse_key_values(args.checkpoint, DEFAULT_CHECKPOINTS)
    n_values = tuple(sorted({int(x) for x in args.n_values.split(",") if x.strip()}))
    if 16 not in n_values or any(n < 1 or n > 16 for n in n_values):
        raise ValueError("n-values must be between 1 and 16 and include 16")
    paths = list(caches.values()) + list(checkpoints.values()) + [args.frozen_protocol]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    for path in args.output_dir.glob("*"):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {args.output_dir}")

    engine = load_numpy_engine()
    critics = {int(seed): engine.load_dual_calibrated(path) for seed, path in checkpoints.items()}
    results = []
    for label, cache_path in caches.items():
        result = evaluate_period(label, cache_path, critics, n_values, args.clip)
        results.append(result)
        print(f"evaluated {label}: windows={len(result['window_rows']) // (len(METHODS) * len(critics))}", flush=True)
    write_outputs(args.output_dir, results, checkpoints, n_values, args.frozen_protocol)
    print(f"output={args.output_dir}", flush=True)
    print(pd.read_csv(args.output_dir / "main_table.csv").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
