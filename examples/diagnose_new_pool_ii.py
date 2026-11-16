"""只读诊断 new_pool_II 上 Critic 的候选排序、融合与 oracle gap。"""

import argparse
import csv
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")

from critic.diagnostics import evaluate_candidate_reranking
from critic.evaluation import direction_accuracy, mae, strategy_median, strategy_simple_mean
from critic.reranker import CriticReranker
from examples.evaluate_cache import load_critic


def _rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(scores, distances):
    score_ranks = _rankdata(scores)
    distance_ranks = _rankdata(-np.asarray(distances, dtype=float))
    if np.allclose(score_ranks, score_ranks[0]) or np.allclose(distance_ranks, distance_ranks[0]):
        return float("nan")
    return float(np.corrcoef(score_ranks, distance_ranks)[0, 1])


def _mse(path, y_true):
    return float(np.mean((path - y_true) ** 2))


def summarize_window_heads(final_scores, fine_scores, coarse_scores,
                           hybrid_distances, fine_distances, coarse_distances):
    return {
        "final_hybrid_spearman": _spearman(final_scores, hybrid_distances),
        "final_fine_spearman": _spearman(final_scores, fine_distances),
        "final_coarse_spearman": _spearman(final_scores, coarse_distances),
        "fine_fine_spearman": _spearman(fine_scores, fine_distances),
        "coarse_coarse_spearman": _spearman(coarse_scores, coarse_distances),
    }


def _strategy_metrics(path, y_true):
    return {
        "ohlc_mse": float(np.mean((path[:, :4] - y_true[:, :4]) ** 2)),
        "ohlc_mae": mae(path, y_true),
        "six_feature_mse": _mse(path, y_true),
        "direction_accuracy": direction_accuracy(path, y_true),
    }


def baseline_rows_for_window(candidates, y_true, hist, dual_critic, small_mlp, temperature):
    dual_scores = dual_critic.score(candidates, hist)
    small_scores = small_mlp.score(candidates, hist)
    weighted = CriticReranker(mode="weighted", temperature=temperature)
    paths = {
        "simple_mean": strategy_simple_mean(candidates),
        "median": strategy_median(candidates),
        "small_mlp_weighted": weighted.combine(candidates, small_scores),
        "small_mlp_top1": candidates[int(np.argmax(small_scores))],
        "dual_weighted": weighted.combine(candidates, dual_scores),
        "dual_top1": candidates[int(np.argmax(dual_scores))],
    }
    ohlc_distances = np.mean((candidates[:, :, :4] - y_true[None, :, :4]) ** 2, axis=(1, 2))
    paths["oracle_ohlc"] = candidates[int(np.argmin(ohlc_distances))]
    return [{"strategy": name, **_strategy_metrics(path, y_true)} for name, path in paths.items()]


def summarize_window(code, window_index, candidates, y_true_norm, scores, temperature):
    distances = np.mean((candidates - y_true_norm) ** 2, axis=(1, 2))
    scaled = np.asarray(scores, dtype=float) / temperature
    scaled -= np.max(scaled)
    weights = np.exp(scaled)
    weights /= weights.sum()
    weighted_path = np.sum(candidates * weights[:, None, None], axis=0)
    simple_path = strategy_simple_mean(candidates)
    top1_index = int(np.argmax(scores))
    oracle_index = int(np.argmin(distances))
    score_order = np.argsort(-np.asarray(scores), kind="mergesort")
    oracle_rank = int(np.where(score_order == oracle_index)[0][0]) + 1
    entropy = float(-(weights * np.log(weights + 1e-12)).sum())

    return {
        "code": str(code),
        "window_index": int(window_index),
        "candidate_count": int(len(candidates)),
        "oracle_index": oracle_index,
        "top1_index": top1_index,
        "oracle_rank": oracle_rank,
        "spearman": _spearman(scores, distances),
        "score_entropy": entropy,
        "effective_n": float(math.exp(entropy)),
        "max_weight": float(np.max(weights)),
        "simple_mean_mse": _mse(simple_path, y_true_norm),
        "weighted_mse": _mse(weighted_path, y_true_norm),
        "simple_mean_ohlc_mse": _mse(simple_path[:, :4], y_true_norm[:, :4]),
        "weighted_ohlc_mse": _mse(weighted_path[:, :4], y_true_norm[:, :4]),
        "top1_mse": float(distances[top1_index]),
        "oracle_mse": float(distances[oracle_index]),
    }


def normalized_history(hist, clip=5.0):
    hist = np.asarray(hist, dtype=np.float32)
    hist_mean = hist.mean(axis=0, keepdims=True)
    hist_std = hist.std(axis=0, keepdims=True) + 1e-5
    return np.clip((hist - hist_mean) / hist_std, -clip, clip).astype(np.float32)


def _normalized_target(x_df, y_true, clip=5.0):
    hist = x_df[["open", "high", "low", "close", "volume", "amount"]].values.astype(np.float32)
    hist_mean = hist.mean(axis=0, keepdims=True)
    hist_std = hist.std(axis=0, keepdims=True) + 1e-5
    hist_norm = normalized_history(hist, clip=clip)
    y_true_norm = ((y_true - hist_mean) / hist_std).astype(np.float32)
    return hist_norm, y_true_norm


def diagnose_samples(samples, critic, temperature):
    """诊断带有既有 label 的 CriticSample，不重算训练标签。"""
    rows = []
    score_rows = []
    distance_rows = []
    for window_index, sample in enumerate(samples):
        scores = critic.score(sample.paths, sample.hist)
        distances = np.asarray(sample.distances, dtype=np.float32)
        y_true = sample.y_true
        if y_true is None:
            raise ValueError("CriticSample diagnostics requires y_true for MSE decomposition")
        row = summarize_window(sample.code, window_index, sample.paths, y_true, scores, temperature)
        row["oracle_index"] = int(np.argmin(distances))
        score_order = np.argsort(-np.asarray(scores), kind="mergesort")
        row["oracle_rank"] = int(np.where(score_order == row["oracle_index"])[0][0]) + 1
        row["spearman"] = _spearman(scores, distances)
        rows.append(row)
        score_rows.append(np.asarray(scores, dtype=np.float32))
        distance_rows.append(distances)
    return rows, evaluate_candidate_reranking(score_rows, distance_rows, temperature=temperature)


def diagnose_cache(cache_path, critic, temperature, clip=5.0):
    with open(cache_path, "rb") as handle:
        cache = pickle.load(handle)
    rows = []
    score_rows = []
    distance_rows = []
    for window_index, (window, candidates) in enumerate(zip(cache["windows"], cache["candidates"])):
        code, x_df, _, _, y_true = window
        hist_norm, y_true_norm = _normalized_target(x_df, y_true, clip=clip)
        scores = critic.score(candidates, hist_norm)
        distances = np.mean(
            (candidates[:, :, :4] - y_true_norm[None, :, :4]) ** 2,
            axis=(1, 2),
        ).astype(np.float32)
        row = summarize_window(code, window_index, candidates, y_true_norm, scores, temperature)
        row["oracle_index"] = int(np.argmin(distances))
        score_order = np.argsort(-np.asarray(scores), kind="mergesort")
        row["oracle_rank"] = int(np.where(score_order == row["oracle_index"])[0][0]) + 1
        row["spearman"] = _spearman(scores, distances)
        row["top1_ohlc_mse"] = float(distances[row["top1_index"]])
        row["oracle_ohlc_mse"] = float(distances[row["oracle_index"]])
        rows.append(row)
        score_rows.append(np.asarray(scores, dtype=np.float32))
        distance_rows.append(distances)
    return rows, evaluate_candidate_reranking(score_rows, distance_rows, temperature=temperature)


def _mean(rows, key):
    return float(np.nanmean([row[key] for row in rows]))


def _write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_baselines_cache(cache_path, dual_critic, small_mlp, temperature, clip):
    with open(cache_path, "rb") as handle:
        cache = pickle.load(handle)
    rows = []
    for window_index, (window, candidates) in enumerate(zip(cache["windows"], cache["candidates"])):
        code, x_df, _, _, y_true = window
        hist_norm, y_true_norm = _normalized_target(x_df, y_true, clip=clip)
        for row in baseline_rows_for_window(
            candidates, y_true_norm, hist_norm, dual_critic, small_mlp, temperature
        ):
            rows.append({"code": str(code), "window_index": window_index, **row})
    return rows


def summarize_seed_metrics(rows, dual_strategy, small_mlp_strategy):
    summary = []
    for strategy in sorted({row["strategy"] for row in rows}):
        values = np.asarray([row["ohlc_mse"] for row in rows if row["strategy"] == strategy], dtype=float)
        summary.append({
            "strategy": strategy,
            "seed_count": len(values),
            "mean_ohlc_mse": float(values.mean()),
            "std_ohlc_mse": float(values.std(ddof=0)),
            "best_ohlc_mse": float(values.min()),
            "worst_ohlc_mse": float(values.max()),
        })
    dual = {row["seed"]: row["ohlc_mse"] for row in rows if row["strategy"] == dual_strategy}
    small = {row["seed"]: row["ohlc_mse"] for row in rows if row["strategy"] == small_mlp_strategy}
    common = sorted(set(dual) & set(small))
    deltas = np.asarray([dual[seed] - small[seed] for seed in common], dtype=float)
    paired = {
        "dual_strategy": dual_strategy,
        "small_mlp_strategy": small_mlp_strategy,
        "paired_seed_count": len(common),
        "mean_dual_minus_small_mlp": float(deltas.mean()),
        "dual_wins": int((deltas < 0).sum()),
    }
    return summary, paired


def summarize_baselines(label, rows):
    summary = []
    for strategy in sorted({row["strategy"] for row in rows}):
        strategy_rows = [row for row in rows if row["strategy"] == strategy]
        summary.append({
            "label": label,
            "strategy": strategy,
            "windows": len(strategy_rows),
            "ohlc_mse": _mean(strategy_rows, "ohlc_mse"),
            "ohlc_mae": _mean(strategy_rows, "ohlc_mae"),
            "six_feature_mse": _mean(strategy_rows, "six_feature_mse"),
            "direction_accuracy": _mean(strategy_rows, "direction_accuracy"),
            "prob_top1_status": "unavailable_missing_log_probs",
        })
    simple_mse = next(row["ohlc_mse"] for row in summary if row["strategy"] == "simple_mean")
    for row in summary:
        row["ohlc_mse_improvement_pct"] = (simple_mse - row["ohlc_mse"]) / simple_mse * 100
    return summary


def _summary_row(label, rows, ranking):
    simple_mse = _mean(rows, "simple_mean_mse")
    weighted_mse = _mean(rows, "weighted_mse")
    return {
        "label": label,
        "windows": len(rows),
        "simple_mean_mse": simple_mse,
        "weighted_mse": weighted_mse,
        "simple_mean_ohlc_mse": _mean(rows, "simple_mean_ohlc_mse"),
        "weighted_ohlc_mse": _mean(rows, "weighted_ohlc_mse"),
        "top1_6feature_mse": _mean(rows, "top1_mse"),
        "oracle_6feature_mse": _mean(rows, "oracle_mse"),
        "top1_ohlc_mse": _mean(rows, "top1_ohlc_mse"),
        "oracle_ohlc_mse": _mean(rows, "oracle_ohlc_mse"),
        "weighted_vs_simple_pct": (simple_mse - weighted_mse) / simple_mse * 100,
        "mean_spearman": ranking["mean_spearman"],
        "ndcg": ranking["ndcg"],
        "mrr": ranking["mrr"],
        "top1_oracle_hit_rate": ranking["top1_oracle_hit_rate"],
        "top3_oracle_hit_rate": ranking["top3_oracle_hit_rate"],
        "mean_oracle_rank": ranking["mean_oracle_rank"],
        "mean_score_entropy": ranking["mean_score_entropy"],
        "mean_effective_n": _mean(rows, "effective_n"),
        "mean_max_weight": _mean(rows, "max_weight"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--small-mlp-checkpoint")
    parser.add_argument("--cache", action="append", required=True, help="label=path，可重复")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--clip", type=float, default=5.0,
                        help="必须与候选生成/训练时 predictor.clip 一致")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    critic, architecture = load_critic(args.checkpoint, "auto", args.device)
    small_mlp = None
    if args.small_mlp_checkpoint:
        small_mlp, _ = load_critic(
            args.small_mlp_checkpoint, "fundamental", args.device,
            path_corr_interaction=False,
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    baseline_summaries = []
    for item in args.cache:
        label, cache_path = item.split("=", 1)
        rows, ranking = diagnose_cache(cache_path, critic, args.temperature, clip=args.clip)
        _write_csv(output_dir / f"{label}_windows.csv", rows)
        summaries.append(_summary_row(label, rows, ranking))
        if small_mlp is not None:
            baseline_rows = evaluate_baselines_cache(
                cache_path, critic, small_mlp, args.temperature, args.clip
            )
            _write_csv(output_dir / f"{label}_strategies.csv", baseline_rows)
            baseline_summaries.extend(summarize_baselines(label, baseline_rows))
    _write_csv(output_dir / "summary.csv", summaries)
    _write_csv(output_dir / "baseline_summary.csv", baseline_summaries)
    print(f"architecture={architecture} clip={args.clip}")
    for row in summaries:
        print(row)
    for row in baseline_summaries:
        print(row)


if __name__ == "__main__":
    main()
