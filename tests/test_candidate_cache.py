import numpy as np
import pandas as pd
import pytest

from critic.candidate_cache import (
    CandidateBundle,
    CandidateCacheKey,
    assert_same_candidates,
    candidate_config_hash,
)


def key(seed=7):
    return CandidateCacheKey(
        "300750",
        pd.Timestamp("2024-01-31"),
        10,
        4,
        seed,
        "Kronos-small",
        "Kronos-Tokenizer-base",
        "abc",
    )


def bundle(paths, seed=7):
    return CandidateBundle(
        key(seed),
        paths,
        np.zeros((90, 6), dtype=np.float32),
        np.zeros(6, dtype=np.float32),
        np.ones(6, dtype=np.float32),
        np.zeros((10, 6), dtype=np.float32),
        10.0,
        None,
    )


def test_candidate_hash_is_order_independent():
    assert candidate_config_hash({"b": 2, "a": 1}) == candidate_config_hash({"a": 1, "b": 2})


def test_identical_candidates_pass():
    paths = np.zeros((4, 10, 6), dtype=np.float32)
    assert_same_candidates(bundle(paths), bundle(paths.copy()))


def test_any_candidate_difference_is_rejected():
    original = bundle(np.zeros((4, 10, 6), dtype=np.float32))
    changed = np.zeros((4, 10, 6), dtype=np.float32)
    changed[0, 0, 0] = 1
    with pytest.raises(ValueError, match="candidate paths differ"):
        assert_same_candidates(original, bundle(changed))


def test_candidate_key_difference_is_rejected_even_if_paths_match():
    paths = np.zeros((4, 10, 6), dtype=np.float32)
    with pytest.raises(ValueError, match="candidate paths differ"):
        assert_same_candidates(bundle(paths, seed=1), bundle(paths, seed=2))
