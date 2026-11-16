"""语义分解归因消融的评估与聚合。

预注册协议：docs/development/protocols/2026-07-28_semantic_decomposition_ablation_preregistration.md

对每个 (臂, seed) 的 best.pt 在固定的 new_pool_II 候选 cache 上评估：
- 主指标：weighted 策略 OHLC MSE（协议 §4.1）
- 次要：top1、以及与权重无关的 simple_mean / median / oracle_ohlc（用于 §3.1 自检门）
- H5 机制证据：fine/coarse head 与 fine/coarse 目标的 2x2 对应矩阵

评估期的 fine/coarse 目标用与训练同一套函数重算（compute_distances /
compute_coarse_trading_distances），保证语义定义一致。
"""
import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from critic.reranker import CriticReranker
from critic.train_critic import compute_coarse_trading_distances, compute_distances
from examples.diagnose_new_pool_ii import _normalized_target, _spearman
from examples.evaluate_cache import load_critic
from examples.run_semantic_decomposition_ablation import ARMS

DEFAULT_CACHES = {
    "2024": "output/candidates/new_pool_II_2024_N16.pkl",
    "2025": "output/candidates/new_pool_II_2025_N16.pkl",
    "2026H1": "output/candidates/new_pool_II_2026H1_N16.pkl",
}
ABLATION_DIR = "output/results/semantic_decomposition_ablation"


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {path}", flush=True)


def ohlc_mse(path, y_true):
    return float(np.mean((path[:, :4] - y_true[:, :4]) ** 2))


def head_correlation_matrix(critic, candidates, hist, fine_target, coarse_target):
    """H5 的 2x2 对应矩阵：含 off-diagonal，用于判定对角占优。"""
    fine_scores = critic.score(candidates, hist, head="fine")
    coarse_scores = critic.score(candidates, hist, head="coarse")
    return {
        "fine_vs_fine": _spearman(fine_scores, fine_target),
        "fine_vs_coarse": _spearman(fine_scores, coarse_target),
        "coarse_vs_coarse": _spearman(coarse_scores, coarse_target),
        "coarse_vs_fine": _spearman(coarse_scores, fine_target),
    }


def evaluate_window(critic, is_dual, candidates, y_true_norm, hist_norm, temperature):
    """单窗口：策略 MSE + （dual 臂）head 对应相关性。"""
    scores = critic.score(candidates, hist_norm)
    weighted = CriticReranker(mode="weighted", temperature=temperature)
    ohlc_distances = np.mean(
        (candidates[:, :, :4] - y_true_norm[None, :, :4]) ** 2, axis=(1, 2))
    row = {
        "weighted": ohlc_mse(weighted.combine(candidates, scores), y_true_norm),
        "top1": ohlc_mse(candidates[int(np.argmax(scores))], y_true_norm),
        "simple_mean": ohlc_mse(candidates.mean(axis=0), y_true_norm),
        "median": ohlc_mse(np.median(candidates, axis=0), y_true_norm),
        "oracle_ohlc": ohlc_mse(candidates[int(np.argmin(ohlc_distances))], y_true_norm),
    }
    heads = {}
    if is_dual:
        fine_target = compute_distances(candidates, y_true_norm)
        coarse_target = compute_coarse_trading_distances(candidates, hist_norm, y_true_norm)
        heads = head_correlation_matrix(critic, candidates, hist_norm, fine_target, coarse_target)
    return row, heads


def evaluate_arm_seed(arm, seed, args):
    ckpt = Path(args.output_dir) / "checkpoints" / arm / f"seed_{seed}" / "best.pt"
    if not ckpt.exists():
        print(f"missing checkpoint, skip: {ckpt}", flush=True)
        return [], []
    meta_path = ckpt.parent / "metadata.json"
    head_hint = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    critic, arch = load_critic(str(ckpt), "auto", args.device, path_corr_interaction=False)
    is_dual = arch == "dual_head"

    window_rows, head_rows = [], []
    for period, cache_path in sorted(args.caches.items()):
        with open(cache_path, "rb") as handle:
            cache = pickle.load(handle)
        for window_index, (window, candidates) in enumerate(
                zip(cache["windows"], cache["candidates"])):
            code, x_df, _, _, y_true = window
            hist_norm, y_true_norm = _normalized_target(x_df, y_true, clip=args.clip)
            metrics, heads = evaluate_window(
                critic, is_dual, candidates, y_true_norm, hist_norm, args.temperature)
            base = {"arm": arm, "seed": seed, "period": period,
                    "code": str(code), "window_index": window_index}
            for strategy, value in metrics.items():
                window_rows.append({**base, "strategy": strategy, "ohlc_mse": value})
            if heads:
                head_rows.append({**base, **heads})
    print(f"evaluated arm={arm} seed={seed} arch={arch} "
          f"params={head_hint.get('param_count')} windows={len(head_rows) or 'n/a'}", flush=True)
    return window_rows, head_rows


def aggregate_per_seed(window_rows):
    """窗口级 -> (臂, seed, 期, 策略) 均值。"""
    buckets = {}
    for row in window_rows:
        key = (row["arm"], row["seed"], row["period"], row["strategy"])
        buckets.setdefault(key, []).append(row["ohlc_mse"])
    return [
        {"arm": a, "seed": s, "period": p, "strategy": st,
         "mean_ohlc_mse": float(np.mean(v)), "window_count": len(v)}
        for (a, s, p, st), v in sorted(buckets.items())
    ]


def selfcheck_weight_independent(per_seed_rows):
    """§3.1 自检门：simple_mean/median/oracle_ohlc 与权重无关，
    必须在所有臂与所有 seed 上完全一致。"""
    rows = []
    for strategy in ("simple_mean", "median", "oracle_ohlc"):
        by_period = {}
        for row in per_seed_rows:
            if row["strategy"] != strategy:
                continue
            by_period.setdefault(row["period"], []).append(row["mean_ohlc_mse"])
        for period, values in sorted(by_period.items()):
            spread = float(max(values) - min(values)) if values else float("nan")
            rows.append({
                "strategy": strategy, "period": period, "n": len(values),
                "value": values[0] if values else float("nan"),
                "max_minus_min": spread,
                "verdict": "PASS" if spread < 1e-9 else "FAIL",
            })
    return rows


def bootstrap_ci(values, n_boot=5000, seed=20260724):
    """配对差的 bootstrap 95% CI（seed 沿用既有管线）。"""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(n_boot, values.size))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_deltas(per_seed_rows, reference="B0_dual_head"):
    """同 seed 配对差：reference - arm。正值 = reference 更好（MSE 更小）。"""
    index = {(r["arm"], r["seed"], r["period"], r["strategy"]): r["mean_ohlc_mse"]
             for r in per_seed_rows}
    arms = sorted({r["arm"] for r in per_seed_rows} - {reference})
    periods = sorted({r["period"] for r in per_seed_rows})
    seeds = sorted({r["seed"] for r in per_seed_rows})
    out = []
    for arm in arms:
        for period in periods:
            for strategy in ("weighted", "top1"):
                deltas, pcts = [], []
                for seed in seeds:
                    ref = index.get((reference, seed, period, strategy))
                    cur = index.get((arm, seed, period, strategy))
                    if ref is None or cur is None:
                        continue
                    deltas.append(cur - ref)
                    pcts.append((cur - ref) / cur * 100)
                if not deltas:
                    continue
                low, high = bootstrap_ci(deltas)
                out.append({
                    "reference": reference, "arm": arm, "period": period, "strategy": strategy,
                    "seed_count": len(deltas),
                    "mean_delta_arm_minus_ref": float(np.mean(deltas)),
                    "mean_improvement_pct_of_ref": float(np.mean(pcts)),
                    "ci95_low": low, "ci95_high": high,
                    "ci_excludes_zero": bool(low > 0 or high < 0),
                    "ref_better_seed_count": int(sum(1 for d in deltas if d > 0)),
                })
    return out


def aggregate_head_rows(head_rows):
    """H5：按 (臂, seed, 期) 聚合 2x2 矩阵，并判定对角是否占优。"""
    keys = ("fine_vs_fine", "fine_vs_coarse", "coarse_vs_coarse", "coarse_vs_fine")
    buckets = {}
    for row in head_rows:
        buckets.setdefault((row["arm"], row["seed"], row["period"]), []).append(row)
    out = []
    for (arm, seed, period), rows in sorted(buckets.items()):
        means = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
        fine_margin = means["fine_vs_fine"] - means["fine_vs_coarse"]
        coarse_margin = means["coarse_vs_coarse"] - means["coarse_vs_fine"]
        out.append({
            "arm": arm, "seed": seed, "period": period, "window_count": len(rows), **means,
            "fine_diagonal_margin": fine_margin,
            "coarse_diagonal_margin": coarse_margin,
            "diagonal_dominant": bool(fine_margin > 0 and coarse_margin > 0),
        })
    return out


def summarize(per_seed_rows):
    """跨 seed 汇总，含全胜计数。"""
    buckets = {}
    for row in per_seed_rows:
        buckets.setdefault((row["arm"], row["period"], row["strategy"]), []).append(
            row["mean_ohlc_mse"])
    return [
        {"arm": a, "period": p, "strategy": st, "seed_count": len(v),
         "mean_ohlc_mse": float(np.mean(v)), "std_ohlc_mse": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
         "min_ohlc_mse": float(np.min(v)), "max_ohlc_mse": float(np.max(v))}
        for (a, p, st), v in sorted(buckets.items())
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=ABLATION_DIR)
    parser.add_argument("--arm", action="append", choices=sorted(ARMS), default=None)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--period", action="append", choices=sorted(DEFAULT_CACHES), default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reference", default="B0_dual_head")
    args = parser.parse_args()

    periods = args.period or sorted(DEFAULT_CACHES)
    args.caches = {p: DEFAULT_CACHES[p] for p in periods}
    arms = args.arm or list(ARMS)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    window_rows, head_rows = [], []
    for arm in arms:
        for seed in seeds:
            w, h = evaluate_arm_seed(arm, seed, args)
            window_rows.extend(w)
            head_rows.extend(h)
    if not window_rows:
        raise SystemExit("no checkpoints evaluated; run training first")

    out = Path(args.output_dir)
    write_csv(out / "window_metrics.csv", window_rows)
    per_seed = aggregate_per_seed(window_rows)
    write_csv(out / "per_seed.csv", per_seed)
    write_csv(out / "summary.csv", summarize(per_seed))

    selfcheck = selfcheck_weight_independent(per_seed)
    write_csv(out / "pipeline_selfcheck.csv", selfcheck)
    failures = [r for r in selfcheck if r["verdict"] == "FAIL"]
    if failures:
        print("SELFCHECK FAIL: weight-independent strategies diverge across arms/seeds")
        for row in failures:
            print(f"  {row['strategy']} {row['period']} spread={row['max_minus_min']:.3e}")

    deltas = paired_deltas(per_seed, reference=args.reference)
    if deltas:
        write_csv(out / "paired_delta_ci.csv", deltas)
    if head_rows:
        write_csv(out / "head_correlation.csv", aggregate_head_rows(head_rows))
    print("SELFCHECK PASS" if not failures else "SELFCHECK FAILED — results void per protocol 3.1")


if __name__ == "__main__":
    main()
