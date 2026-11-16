"""消融聚合逻辑测试：判定结论直接建在这些纯函数上。"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from examples.evaluate_semantic_decomposition_ablation import (
    aggregate_head_rows,
    aggregate_per_seed,
    bootstrap_ci,
    paired_deltas,
    selfcheck_weight_independent,
    summarize,
)


def window_row(arm, seed, period, strategy, mse, index=0):
    return {"arm": arm, "seed": seed, "period": period, "code": "000001",
            "window_index": index, "strategy": strategy, "ohlc_mse": mse}


def test_aggregate_per_seed_averages_windows():
    rows = [window_row("B0", 0, "2024", "weighted", 1.0, 0),
            window_row("B0", 0, "2024", "weighted", 2.0, 1)]
    out = aggregate_per_seed(rows)
    assert len(out) == 1
    assert out[0]["mean_ohlc_mse"] == pytest.approx(1.5)
    assert out[0]["window_count"] == 2


def test_selfcheck_passes_when_weight_independent_metrics_identical():
    rows = [{"arm": a, "seed": s, "period": "2024", "strategy": "simple_mean",
             "mean_ohlc_mse": 1.75, "window_count": 10}
            for a in ("B0", "A0", "C3") for s in (0, 1)]
    verdicts = selfcheck_weight_independent(rows)
    simple = [r for r in verdicts if r["strategy"] == "simple_mean"]
    assert simple and all(r["verdict"] == "PASS" for r in simple)
    assert simple[0]["n"] == 6


def test_selfcheck_fails_when_an_arm_diverges():
    rows = [{"arm": "B0", "seed": 0, "period": "2024", "strategy": "oracle_ohlc",
             "mean_ohlc_mse": 1.0, "window_count": 10},
            {"arm": "A0", "seed": 0, "period": "2024", "strategy": "oracle_ohlc",
             "mean_ohlc_mse": 1.2, "window_count": 10}]
    verdicts = selfcheck_weight_independent(rows)
    oracle = [r for r in verdicts if r["strategy"] == "oracle_ohlc"]
    assert oracle[0]["verdict"] == "FAIL"


def test_paired_delta_sign_reference_better_is_positive():
    # B0 的 MSE 更小 => B0 更好 => delta = arm - ref > 0
    rows = []
    for seed in (0, 1, 2):
        rows.append({"arm": "B0_dual_head", "seed": seed, "period": "2024",
                     "strategy": "weighted", "mean_ohlc_mse": 1.50, "window_count": 5})
        rows.append({"arm": "C3_dual_head_swapped", "seed": seed, "period": "2024",
                     "strategy": "weighted", "mean_ohlc_mse": 1.56, "window_count": 5})
    out = paired_deltas(rows, reference="B0_dual_head")
    row = next(r for r in out if r["arm"] == "C3_dual_head_swapped")
    assert row["mean_delta_arm_minus_ref"] == pytest.approx(0.06)
    assert row["ref_better_seed_count"] == 3
    assert row["mean_improvement_pct_of_ref"] == pytest.approx(0.06 / 1.56 * 100)


def test_paired_delta_ci_excludes_zero_only_when_consistent():
    consistent = []
    mixed = []
    for seed, (a, b) in enumerate([(1.50, 1.56), (1.51, 1.57), (1.49, 1.55)]):
        consistent += [
            {"arm": "B0_dual_head", "seed": seed, "period": "2024", "strategy": "weighted",
             "mean_ohlc_mse": a, "window_count": 5},
            {"arm": "X", "seed": seed, "period": "2024", "strategy": "weighted",
             "mean_ohlc_mse": b, "window_count": 5}]
    for seed, (a, b) in enumerate([(1.50, 1.56), (1.51, 1.40), (1.49, 1.60)]):
        mixed += [
            {"arm": "B0_dual_head", "seed": seed, "period": "2024", "strategy": "weighted",
             "mean_ohlc_mse": a, "window_count": 5},
            {"arm": "X", "seed": seed, "period": "2024", "strategy": "weighted",
             "mean_ohlc_mse": b, "window_count": 5}]
    assert paired_deltas(consistent)[0]["ci_excludes_zero"] is True
    assert paired_deltas(mixed)[0]["ci_excludes_zero"] is False


def test_bootstrap_ci_is_deterministic_and_brackets_mean():
    values = [0.05, 0.06, 0.04, 0.055, 0.045]
    first = bootstrap_ci(values)
    assert first == bootstrap_ci(values)
    low, high = first
    assert low < float(np.mean(values)) < high


def test_diagonal_dominance_detected_and_rejected():
    strong = [{"arm": "B0", "seed": 0, "period": "2024",
               "fine_vs_fine": 0.62, "fine_vs_coarse": 0.41,
               "coarse_vs_coarse": 0.58, "coarse_vs_fine": 0.33}]
    inverted = [{"arm": "C3", "seed": 0, "period": "2024",
                 "fine_vs_fine": 0.35, "fine_vs_coarse": 0.60,
                 "coarse_vs_coarse": 0.30, "coarse_vs_fine": 0.55}]
    assert aggregate_head_rows(strong)[0]["diagonal_dominant"] is True
    row = aggregate_head_rows(inverted)[0]
    assert row["diagonal_dominant"] is False
    assert row["fine_diagonal_margin"] < 0


def test_summarize_reports_spread_across_seeds():
    rows = [{"arm": "B0", "seed": s, "period": "2024", "strategy": "weighted",
             "mean_ohlc_mse": v, "window_count": 5}
            for s, v in enumerate([1.50, 1.52, 1.54])]
    out = summarize(rows)[0]
    assert out["seed_count"] == 3
    assert out["mean_ohlc_mse"] == pytest.approx(1.52)
    assert out["min_ohlc_mse"] == pytest.approx(1.50)
    assert out["max_ohlc_mse"] == pytest.approx(1.54)
