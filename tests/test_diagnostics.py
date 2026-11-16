"""Diagnostics for critic/oracle gap on fixed candidate samples."""

import numpy as np

from critic import CriticSample
from critic.diagnostics import evaluate_candidate_reranking, score_samples_for_diagnostics


def test_evaluate_candidate_reranking_reports_oracle_gap_metrics():
    scores = [
        np.array([0.1, 0.9, 0.2, 0.0], dtype=np.float32),
        np.array([0.6, 0.5, 0.4, 0.3], dtype=np.float32),
    ]
    distances = [
        np.array([0.3, 0.1, 0.2, 0.4], dtype=np.float32),
        np.array([0.4, 0.1, 0.2, 0.3], dtype=np.float32),
    ]

    metrics = evaluate_candidate_reranking(scores, distances, top_k=2, temperature=1.0)

    assert metrics["sample_count"] == 2
    assert metrics["top1_oracle_hit_rate"] == 0.5
    assert metrics["top2_oracle_hit_rate"] == 1.0
    assert metrics["mean_oracle_rank"] == 1.5
    assert metrics["mrr"] == 0.75
    assert 0.0 < metrics["ndcg"] <= 1.0
    assert metrics["mean_spearman"] > 0.0
    assert metrics["mean_score_entropy"] > 0.0


def test_evaluate_candidate_reranking_handles_invalid_samples():
    metrics = evaluate_candidate_reranking(
        [np.array([np.nan, np.nan], dtype=np.float32), np.array([0.2, 0.1], dtype=np.float32)],
        [np.array([0.1, 0.2], dtype=np.float32), np.array([np.nan, np.nan], dtype=np.float32)],
        top_k=1,
    )

    assert metrics["sample_count"] == 0
    assert np.isnan(metrics["top1_oracle_hit_rate"])
    assert np.isnan(metrics["mean_spearman"])


class ScoreByCloseMeanCritic:
    def score(self, paths, hist, corr_feat=None):
        return paths[:, :, 3].mean(axis=1)


def test_score_samples_for_diagnostics_uses_critic_scores_and_sample_distances():
    paths = np.zeros((3, 2, 6), dtype=np.float32)
    paths[:, :, 3] = np.array([
        [0.0, 0.1],
        [0.0, 0.9],
        [0.0, 0.3],
    ], dtype=np.float32)
    sample = CriticSample(
        paths=paths,
        hist=np.zeros((4, 6), dtype=np.float32),
        corr_feat=None,
        distances=np.array([0.3, 0.1, 0.2], dtype=np.float32),
    )

    score_rows, distance_rows = score_samples_for_diagnostics(ScoreByCloseMeanCritic(), [sample])

    assert len(score_rows) == 1
    assert len(distance_rows) == 1
    np.testing.assert_allclose(score_rows[0], np.array([0.05, 0.45, 0.15], dtype=np.float32))
    np.testing.assert_allclose(distance_rows[0], sample.distances)

    metrics = evaluate_candidate_reranking(score_rows, distance_rows, top_k=1)
    assert metrics["top1_oracle_hit_rate"] == 1.0
