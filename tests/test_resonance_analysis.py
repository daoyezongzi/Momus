import numpy as np

from critic.resonance_analysis import (
    CachedResonanceScore,
    assign_bins,
    compute_cached_resonance_score,
    summarize_distances_by_bin,
)
from critic.schemas import CriticSample


def sample_with_directions(target_moves, chain_moves, distances):
    hist = np.zeros((len(target_moves) + 1, 6), dtype=np.float32)
    hist[1:, 3] = np.cumsum(target_moves)
    corr = np.zeros((2, len(target_moves), 6), dtype=np.float32)
    corr[:, :, 3] = np.asarray(chain_moves, dtype=np.float32)
    corr[:, :, 4] = np.asarray(chain_moves, dtype=np.float32)
    return CriticSample(
        paths=np.zeros((len(distances), 10, 6), dtype=np.float32),
        hist=hist,
        corr_feat=corr,
        distances=np.asarray(distances, dtype=np.float32),
    )


def test_cached_resonance_score_is_high_for_aligned_chain():
    sample = sample_with_directions(
        target_moves=[1, 1, -1, -1],
        chain_moves=[1, 1, -1, -1],
        distances=[1, 2, 3],
    )
    score = compute_cached_resonance_score(sample)
    assert isinstance(score, CachedResonanceScore)
    assert score.price >= 0.9
    assert score.volume >= 0.9


def test_assign_bins_uses_training_quantiles():
    train_scores = [CachedResonanceScore(price=x, volume=x, price_volume=x) for x in [0.1, 0.2, 0.8, 0.9]]
    eval_scores = [CachedResonanceScore(price=x, volume=x, price_volume=x) for x in [0.15, 0.5, 0.95]]
    bins = assign_bins(train_scores, eval_scores, field="price")
    assert bins == ["low", "mid", "high"]


def test_summarize_distances_by_bin_reports_expected_distance():
    samples = [
        sample_with_directions([1], [1], [1, 3]),
        sample_with_directions([1], [1], [2, 4]),
        sample_with_directions([1], [1], [10, 20]),
    ]
    summary = summarize_distances_by_bin(samples, ["low", "low", "high"])
    assert summary["low"]["n"] == 2
    assert summary["low"]["simple_mean_exp_distance"] == 2.5
    assert summary["high"]["simple_mean_exp_distance"] == 15.0
