import json

import numpy as np
import pandas as pd

from examples.data_scaling_robustness import (
    aggregate_stage_b_summary,
    bootstrap_mean_ci,
    build_stage_b_model,
    build_stage_b_pools,
    cache_to_critic_samples,
    compute_stage_b_paired_deltas,
    direction_accuracy,
    evaluate_prediction_metrics,
    normalize_history,
    pool_slices,
    samples_from_cache,
    split_train_val,
    stage_b_checkpoint_jobs,
    write_stage_b_checkpoint_sidecars,
)


def tiny_cache():
    x_df = pd.DataFrame({
        "open": [1.0, 2.0, 3.0],
        "high": [2.0, 3.0, 4.0],
        "low": [0.5, 1.5, 2.5],
        "close": [1.5, 2.5, 3.5],
        "volume": [10.0, 20.0, 30.0],
        "amount": [100.0, 200.0, 300.0],
    })
    y_true = np.ones((2, 6), dtype=np.float32)
    candidates = np.stack([
        np.zeros((2, 6), dtype=np.float32),
        np.ones((2, 6), dtype=np.float32),
    ])
    return {
        "windows": [("000001", x_df, None, None, y_true)],
        "candidates": [candidates],
        "metadata": {"sample_count": 2},
    }


def test_normalize_history_clips_training_input():
    hist = np.array([[0.0] * 6, [100.0] * 6], dtype=np.float32)

    normalized = normalize_history(hist, clip=0.5)

    assert normalized.dtype == np.float32
    assert normalized.shape == hist.shape
    assert np.max(np.abs(normalized)) <= 0.5


def test_pool_slices_match_preregistered_scaling_counts():
    full_count = 465
    new_pool_count = 1257

    groups = pool_slices(full_count, new_pool_count)

    assert [group["label"] for group in groups] == ["G1_31", "G2_61", "G3_91", "G4_117"]
    assert [len(group["indices"]) for group in groups] == [465, 915, 1365, 1722]


def test_samples_from_cache_returns_stage_a_rows_with_hist_norm():
    rows = samples_from_cache(tiny_cache())

    assert len(rows) == 1
    assert rows[0]["paths"].shape == (2, 2, 6)
    assert rows[0]["hist"].shape == (3, 6)
    assert rows[0]["distances"].shape == (2,)
    assert np.max(np.abs(rows[0]["hist"])) <= 5.0


def test_cache_to_critic_samples_preserves_dual_granularity_labels():
    samples = cache_to_critic_samples(tiny_cache(), pool_label="G_test")

    assert len(samples) == 1
    sample = samples[0]
    assert sample.code == "000001"
    assert sample.paths.shape == (2, 2, 6)
    assert sample.hist.shape == (3, 6)
    assert sample.distances.shape == (2,)
    assert sample.fine_distances.shape == (2,)
    assert sample.coarse_distances.shape == (2,)




def test_stage_b_checkpoint_jobs_freeze_two_pools_two_models_three_seeds():
    jobs = stage_b_checkpoint_jobs(["G1_31", "G4_117"], seeds=[0, 1, 2])

    assert len(jobs) == 12
    assert jobs[0] == {"pool": "G1_31", "model": "dualhead_calibrated", "seed": 0}
    assert jobs[-1] == {"pool": "G4_117", "model": "small_mlp_no_interaction", "seed": 2}




def test_split_train_val_is_seeded_and_non_empty():
    samples = list(range(10))

    train_a, val_a = split_train_val(samples, seed=7, val_fraction=0.2)
    train_b, val_b = split_train_val(samples, seed=7, val_fraction=0.2)

    assert train_a == train_b
    assert val_a == val_b
    assert len(train_a) == 8
    assert len(val_a) == 2
    assert sorted(train_a + val_a) == samples




def test_evaluate_prediction_metrics_reports_ohlc_and_six_feature_errors():
    pred = np.zeros((2, 6), dtype=np.float32)
    y_true = np.ones((2, 6), dtype=np.float32)

    metrics = evaluate_prediction_metrics(pred, y_true)

    assert metrics["ohlc_mse"] == 1.0
    assert metrics["ohlc_mae"] == 1.0
    assert metrics["six_feature_mse"] == 1.0




def test_aggregate_stage_b_summary_counts_positive_seed_wins():
    rows = [
        {"period": "2024", "pool": "G4_117", "model": "dualhead_calibrated", "strategy": "weighted", "seed": 0, "ohlc_mse": 1.0, "improvement_pct_vs_simple_mean": 10.0},
        {"period": "2024", "pool": "G4_117", "model": "dualhead_calibrated", "strategy": "weighted", "seed": 1, "ohlc_mse": 2.0, "improvement_pct_vs_simple_mean": -1.0},
    ]

    summary = aggregate_stage_b_summary(rows)

    assert summary[0]["mean_ohlc_mse"] == 1.5
    assert summary[0]["win_count"] == 1
    assert summary[0]["positive_seed_ratio"] == 0.5




def test_bootstrap_mean_ci_is_deterministic_and_ordered():
    lo, hi = bootstrap_mean_ci(np.array([1.0, 2.0, 3.0]), seed=0, n_boot=100)

    assert lo <= hi
    assert (lo, hi) == bootstrap_mean_ci(np.array([1.0, 2.0, 3.0]), seed=0, n_boot=100)
