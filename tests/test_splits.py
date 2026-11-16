import pandas as pd
import pytest

from critic.splits import (
    FoldSpec,
    assert_purge_embargo,
    assert_unseen_not_in_training,
    make_default_folds,
    make_seen_unseen_split,
    split_hash,
)


def test_default_folds_leave_2025_out_of_development():
    folds = make_default_folds(pred_len=10)
    assert folds[0].validation_end == pd.Timestamp("2023-12-31")
    assert folds[1].validation_end == pd.Timestamp("2024-12-31")
    assert all(fold.test_start is None for fold in folds)


def test_seen_unseen_split_is_deterministic_and_chain_stratified():
    chain_by_code = {
        "a": "chain1",
        "b": "chain1",
        "c": "chain1",
        "d": "chain2",
        "e": "chain2",
        "f": "chain2",
    }
    first = make_seen_unseen_split(chain_by_code, unseen_ratio=1 / 3, seed=7)
    second = make_seen_unseen_split(chain_by_code, unseen_ratio=1 / 3, seed=7)
    assert first == second
    seen, unseen = first
    assert {chain_by_code[code] for code in unseen} == {"chain1", "chain2"}
    assert seen.isdisjoint(unseen)
    assert split_hash(seen, unseen) == split_hash(set(seen), set(unseen))


def test_unseen_code_cannot_appear_as_training_relation():
    with pytest.raises(ValueError, match="unseen code leaked"):
        assert_unseen_not_in_training(
            train_targets={"a"}, mapping={"a": ["b", "c"]}, unseen={"c"}
        )


def test_purge_embargo_rejects_overlapping_boundary():
    fold = FoldSpec(
        fold_id="fold_bad",
        train_end=pd.Timestamp("2023-12-31"),
        validation_start=pd.Timestamp("2024-01-05"),
        validation_end=pd.Timestamp("2024-12-31"),
        pred_len=10,
        embargo=10,
    )
    with pytest.raises(ValueError, match="embargo"):
        assert_purge_embargo(fold)
