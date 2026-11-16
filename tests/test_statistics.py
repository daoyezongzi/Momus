import pandas as pd

from critic.statistics import holm_adjust, paired_effect_table


def test_holm_adjustment_matches_known_values():
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.04})
    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.06}


def test_paired_effect_joins_by_sample_not_row_order():
    records = pd.DataFrame(
        [
            {"sample_id": "s1", "stock": "a", "cutoff_T": "2024-01-01", "strategy": "base", "metric": 2.0},
            {"sample_id": "s2", "stock": "b", "cutoff_T": "2024-01-02", "strategy": "base", "metric": 4.0},
            {"sample_id": "s2", "stock": "b", "cutoff_T": "2024-01-02", "strategy": "corr", "metric": 3.0},
            {"sample_id": "s1", "stock": "a", "cutoff_T": "2024-01-01", "strategy": "corr", "metric": 1.0},
        ]
    )
    table = paired_effect_table(
        records,
        "corr",
        "base",
        lower_is_better=True,
        n_boot=50,
        block_len=2,
        seed=0,
    )
    assert table["effect"] == -1.0
    assert table["ci_low"] <= table["effect"] <= table["ci_high"]


def test_paired_effect_rejects_unpaired_data():
    records = pd.DataFrame(
        [
            {"sample_id": "s1", "stock": "a", "cutoff_T": "2024-01-01", "strategy": "base", "metric": 2.0},
            {"sample_id": "s2", "stock": "b", "cutoff_T": "2024-01-02", "strategy": "corr", "metric": 3.0},
        ]
    )
    try:
        paired_effect_table(records, "corr", "base", lower_is_better=True, n_boot=10, seed=0)
    except ValueError as exc:
        assert "no paired samples" in str(exc)
    else:
        raise AssertionError("expected unpaired data to fail")
