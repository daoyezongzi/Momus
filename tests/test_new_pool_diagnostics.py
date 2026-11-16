import numpy as np

from critic import CriticSample
from examples.diagnose_new_pool_ii import (
    summarize_seed_metrics,
    baseline_rows_for_window,
    diagnose_samples,
    normalized_history,
    summarize_window,
    summarize_window_heads,
)
from critic.reranker import CriticReranker


class ZeroCritic:
    def score(self, paths, hist, corr_feat=None):
        return np.zeros(len(paths), dtype=np.float32)




class CloseMeanCritic:
    def score(self, paths, hist):
        return paths[:, :, 3].mean(axis=1)


def test_summarize_seed_metrics_reports_seed_stability_and_paired_delta():
    rows = [
        {"seed": 0, "strategy": "dual_top1", "ohlc_mse": 0.8},
        {"seed": 1, "strategy": "dual_top1", "ohlc_mse": 0.9},
        {"seed": 0, "strategy": "small_mlp_top1", "ohlc_mse": 1.0},
        {"seed": 1, "strategy": "small_mlp_top1", "ohlc_mse": 0.85},
    ]

    summary, paired = summarize_seed_metrics(rows, "dual_top1", "small_mlp_top1")

    dual = next(row for row in summary if row["strategy"] == "dual_top1")
    assert dual["seed_count"] == 2
    assert np.isclose(dual["mean_ohlc_mse"], 0.85)
    assert np.isclose(dual["best_ohlc_mse"], 0.8)
    assert paired["mean_dual_minus_small_mlp"] < 0
    assert paired["dual_wins"] == 1


def test_normalized_target_clips_history_but_not_future():
    import pandas as pd
    from examples.diagnose_new_pool_ii import _normalized_target

    x_df = pd.DataFrame({
        name: [0.0, 1.0] for name in ["open", "high", "low", "close", "volume", "amount"]
    })
    y_true = np.full((1, 6), 100.0, dtype=np.float32)

    hist_norm, y_norm = _normalized_target(x_df, y_true, clip=0.5)

    assert np.max(np.abs(hist_norm)) <= 0.5
    assert np.max(y_norm) > 0.5


def test_baseline_summary_includes_median_and_small_mlp():
    rows = baseline_rows_for_window(
        candidates=np.zeros((2, 2, 6), dtype=np.float32),
        y_true=np.zeros((2, 6), dtype=np.float32),
        hist=np.zeros((4, 6), dtype=np.float32),
        dual_critic=ZeroCritic(),
        small_mlp=ZeroCritic(),
        temperature=1.0,
    )

    assert {row["strategy"] for row in rows} == {
        "simple_mean", "median", "small_mlp_weighted", "small_mlp_top1",
        "dual_weighted", "dual_top1", "oracle_ohlc",
    }


def test_normalized_history_clips_like_training_input():
    hist = np.array([[0.0] * 6, [100.0] * 6], dtype=np.float32)

    normalized = normalized_history(hist, clip=0.5)

    assert normalized.shape == hist.shape
    assert np.all(np.abs(normalized) <= 0.5)


def test_summarize_window_reports_rank_weight_and_mse_metrics():
    candidates = np.zeros((3, 2, 6), dtype=np.float32)
    candidates[0, :, 3] = [0.0, 0.2]
    candidates[1, :, 3] = [0.0, 0.8]
    candidates[2, :, 3] = [0.0, 0.4]
    y_true = np.zeros((2, 6), dtype=np.float32)
    y_true[:, 3] = [0.0, 0.8]

    row = summarize_window(
        code="000001",
        window_index=7,
        candidates=candidates,
        y_true_norm=y_true,
        scores=np.array([0.2, 1.0, 0.5], dtype=np.float32),
        temperature=1.0,
    )

    assert row["code"] == "000001"
    assert row["window_index"] == 7
    assert row["candidate_count"] == 3
    assert row["oracle_index"] == 1
    assert row["oracle_rank"] == 1
    assert row["top1_mse"] == row["oracle_mse"]
    assert row["weighted_mse"] < row["simple_mean_mse"]
    assert row["weighted_ohlc_mse"] < row["simple_mean_ohlc_mse"]
    assert 1.0 <= row["effective_n"] <= 3.0
    assert 0.0 <= row["spearman"] <= 1.0


def test_diagnose_samples_uses_stored_distances_without_recomputing_labels():
    paths = np.zeros((3, 2, 6), dtype=np.float32)
    paths[:, :, 3] = [[0.0, 0.2], [0.0, 0.8], [0.0, 0.4]]
    sample = CriticSample(
        paths=paths,
        hist=np.zeros((4, 6), dtype=np.float32),
        corr_feat=None,
        distances=np.array([0.4, 0.1, 0.2], dtype=np.float32),
        code="000002",
        y_true=np.zeros((2, 6), dtype=np.float32),
    )

    rows, ranking = diagnose_samples([sample], CloseMeanCritic(), temperature=1.0)

    assert rows[0]["oracle_index"] == 1
    assert rows[0]["oracle_rank"] == 1
    assert ranking["top1_oracle_hit_rate"] == 1.0


def test_summarize_window_heads_separates_final_fine_and_coarse_alignment():
    final_scores = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    fine_scores = np.array([1.0, 0.5, 0.0], dtype=np.float32)
    coarse_scores = np.array([0.0, 1.0, 0.5], dtype=np.float32)
    fine_distances = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    coarse_distances = np.array([0.3, 0.1, 0.2], dtype=np.float32)
    hybrid_distances = np.array([0.3, 0.2, 0.1], dtype=np.float32)

    result = summarize_window_heads(
        final_scores,
        fine_scores,
        coarse_scores,
        hybrid_distances,
        fine_distances,
        coarse_distances,
    )

    assert result["final_hybrid_spearman"] == 1.0
    assert result["fine_fine_spearman"] == 1.0
    assert result["coarse_coarse_spearman"] == 1.0
