"""语义分解归因消融的承重逻辑测试。

覆盖三件事：
1. head_hidden_dim 解耦后默认参数量零回归（否则历史 checkpoint 读不进来）
2. 置换 cache 的不变量（distances 不变、边缘分布不变、语义确实被破坏）
3. load_critic 对加宽头部的宽度推断
"""
import os
import sys
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from critic.model import DualHeadCritic, FundamentalCritic
from critic.schemas import CriticSample
from examples.build_ablation_label_caches import (
    assert_arm_semantics,
    assert_untouched,
    make_c1_duplicate_final,
    make_c2_shuffled,
    make_c3_swapped,
)


def count(model):
    return sum(p.numel() for p in model.parameters())


@pytest.mark.parametrize("pci,expected", [(True, 40194), (False, 38146)])
def test_fundamental_default_param_count_unchanged(pci, expected):
    assert count(FundamentalCritic(path_corr_interaction=pci)) == expected


@pytest.mark.parametrize("pci,expected", [(True, 57157), (False, 51013)])
def test_dual_calibrated_default_param_count_unchanged(pci, expected):
    model = DualHeadCritic(path_corr_interaction=pci, final_combiner="calibrated")
    assert count(model) == expected


def test_head_hidden_dim_widens_head_but_not_encoder():
    narrow = FundamentalCritic(path_corr_interaction=False)
    wide = FundamentalCritic(path_corr_interaction=False, head_hidden_dim=195)
    assert count(wide) > count(narrow)
    # encoder 必须逐参数不变
    for name in ("seq_encoder.gru.weight_ih_l0", "seq_encoder.proj.weight"):
        assert dict(narrow.named_parameters())[name].shape == dict(wide.named_parameters())[name].shape
    assert wide.head[0].out_features == 195


def test_capacity_matched_arm_is_within_one_percent_of_dual():
    dual = count(DualHeadCritic(path_corr_interaction=False, final_combiner="calibrated"))
    wide = count(FundamentalCritic(path_corr_interaction=False, head_hidden_dim=195))
    assert abs(wide - dual) / dual < 0.01


def make_sample(seed):
    rng = np.random.default_rng(seed)
    fine = rng.random(8).astype(np.float32)
    coarse = rng.random(8).astype(np.float32)
    distances = ((fine - fine.mean()) / fine.std() + coarse).astype(np.float32)
    return CriticSample(
        paths=rng.random((8, 10, 6)).astype(np.float32),
        hist=rng.random((60, 6)).astype(np.float32),
        distances=distances,
        corr_feat=None,
        code="000001",
        fine_distances=fine,
        coarse_distances=coarse,
    )


@pytest.fixture
def samples():
    return [make_sample(i) for i in range(4)]


def test_c3_swap_is_exact_and_leaves_final_target_untouched(samples):
    out = make_c3_swapped(samples)
    assert_untouched(samples, out, "C3_swapped")
    assert_arm_semantics(samples, out, "C3_swapped")
    for src, dst in zip(samples, out):
        assert np.array_equal(dst.fine_distances, src.coarse_distances)
        assert np.array_equal(dst.distances, src.distances)


def test_c2_shuffle_preserves_marginals_but_breaks_pairing(samples):
    out = make_c2_shuffled(samples)
    assert_untouched(samples, out, "C2_shuffled")
    assert_arm_semantics(samples, out, "C2_shuffled")
    changed = sum(
        1 for src, dst in zip(samples, out)
        if not np.array_equal(src.fine_distances, dst.fine_distances)
    )
    assert changed == len(samples)


def test_c1_duplicates_final_target_into_both_aux_heads(samples):
    out = make_c1_duplicate_final(samples)
    assert_untouched(samples, out, "C1_duplicate_final")
    assert_arm_semantics(samples, out, "C1_duplicate_final")
    for src, dst in zip(samples, out):
        assert np.array_equal(dst.fine_distances, src.distances)
        assert np.array_equal(dst.coarse_distances, src.distances)


def test_builders_do_not_mutate_source(samples):
    before = [s.fine_distances.copy() for s in samples]
    make_c1_duplicate_final(samples)
    make_c2_shuffled(samples)
    make_c3_swapped(samples)
    for original, sample in zip(before, samples):
        assert np.array_equal(original, sample.fine_distances)


def test_assert_untouched_detects_tampered_distances(samples):
    out = make_c3_swapped(samples)
    out[0].distances = out[0].distances + 1.0
    with pytest.raises(AssertionError, match="distances"):
        assert_untouched(samples, out, "C3_swapped")


@pytest.mark.parametrize("head_hidden_dim", [64, 195])
def test_load_critic_infers_head_width(head_hidden_dim):
    from examples.evaluate_cache import load_critic

    model = FundamentalCritic(path_corr_interaction=False, head_hidden_dim=head_hidden_dim)
    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    try:
        torch.save(model.state_dict(), path)
        loaded, arch = load_critic(path, "auto", "cpu", path_corr_interaction=False)
        assert arch == "fundamental"
        assert count(loaded) == count(model)
    finally:
        os.remove(path)
