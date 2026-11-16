"""步骤 2 原型验证：不依赖大模型下载，验证 Critic 管道核心逻辑。

覆盖：
- CriticReranker 的 weighted / top1 融合正确性与退化性质
- RandomCritic / FundamentalCritic 的打分接口与形状
"""
import numpy as np
import torch
import pytest

from critic import CriticReranker, RandomCritic, FundamentalCritic


def test_reranker_top1_selects_argmax():
    rr = CriticReranker(mode="top1")
    paths = np.stack([
        np.full((3, 6), 1.0),
        np.full((3, 6), 2.0),
        np.full((3, 6), 3.0),
    ])  # (3, 3, 6)
    scores = np.array([0.1, 0.9, 0.5])
    out = rr.combine(paths, scores)
    assert out.shape == (3, 6)
    assert np.allclose(out, 2.0)  # 选中 scores 最大的第 1 条（值全 2.0）


def test_reranker_weighted_high_temperature_approx_mean():
    """τ 很大时 weighted 退化为简单平均（含 baseline 性质）。"""
    rr = CriticReranker(mode="weighted", temperature=1e6)
    rng = np.random.default_rng(0)
    paths = rng.standard_normal((5, 4, 6)).astype(np.float32)
    scores = rng.standard_normal(5).astype(np.float32)
    out = rr.combine(paths, scores)
    assert np.allclose(out, paths.mean(axis=0), atol=1e-3)


def test_reranker_weighted_low_temperature_approx_top1():
    """τ 很小时 weighted 趋近 top1。"""
    rr = CriticReranker(mode="weighted", temperature=1e-3)
    paths = np.stack([np.full((3, 6), 1.0), np.full((3, 6), 9.0)])
    scores = np.array([0.0, 1.0])
    out = rr.combine(paths, scores)
    assert np.allclose(out, 9.0, atol=1e-2)


def test_reranker_weights_sum_to_one():
    rr = CriticReranker(mode="weighted", temperature=1.0)
    paths = np.ones((4, 2, 6), dtype=np.float32)
    scores = np.array([1.0, 2.0, 3.0, 4.0])
    out = rr.combine(paths, scores)
    # 所有路径值相同(全1)，任何权重组合结果仍为全1
    assert np.allclose(out, 1.0)


def test_reranker_rejects_bad_shape():
    rr = CriticReranker()
    with pytest.raises(ValueError):
        rr.combine(np.ones((3, 6)), np.ones(3))  # 缺一维


def test_random_critic_shape():
    critic = RandomCritic(seed=42)
    paths = np.zeros((16, 10, 6), dtype=np.float32)
    scores = critic.score(paths)
    assert scores.shape == (16,)


def test_fundamental_critic_forward_shape():
    critic = FundamentalCritic(in_dim=6)
    path = torch.zeros(16, 10, 6)
    hist = torch.zeros(16, 400, 6)
    out = critic(path, hist)
    assert out.shape == (16,)


def test_fundamental_critic_score_interface():
    """真 Critic 的 score 接口与 RandomCritic 对齐，hist 自动广播到 N 条。"""
    critic = FundamentalCritic(in_dim=6)
    paths = np.zeros((16, 10, 6), dtype=np.float32)
    hist = np.zeros((400, 6), dtype=np.float32)
    scores = critic.score(paths, hist=hist)
    assert scores.shape == (16,)


def test_fundamental_critic_param_count_under_budget():
    """参数量应远小于 5M（设计文档 §5.2）。"""
    critic = FundamentalCritic()
    n_params = sum(p.numel() for p in critic.parameters())
    assert n_params < 5_000_000


def test_fundamental_critic_with_corr_feat():
    """接入关联特征：forward 与 score 都应支持 corr 输入。"""
    critic = FundamentalCritic(in_dim=6)
    # forward 张量形式
    path = torch.zeros(16, 10, 6)
    hist = torch.zeros(16, 90, 6)
    corr = torch.zeros(16, 4, 90, 6)  # 4 个关联标的
    out = critic(path, hist, corr)
    assert out.shape == (16,)


def test_fundamental_critic_score_with_corr_feat():
    """score 接口传入 (n_corr, lookback, feat) 关联特征，自动广播到 N 条。"""
    critic = FundamentalCritic(in_dim=6)
    paths = np.zeros((16, 10, 6), dtype=np.float32)
    hist = np.zeros((90, 6), dtype=np.float32)
    corr_feat = np.zeros((4, 90, 6), dtype=np.float32)
    scores = critic.score(paths, hist=hist, corr_feat=corr_feat)
    assert scores.shape == (16,)


def test_fundamental_critic_corr_none_still_works():
    """corr_feat=None 时退化为仅用 path+hist，不报错（向后兼容）。"""
    critic = FundamentalCritic(in_dim=6)
    paths = np.zeros((8, 10, 6), dtype=np.float32)
    hist = np.zeros((90, 6), dtype=np.float32)
    scores = critic.score(paths, hist=hist, corr_feat=None)
    assert scores.shape == (8,)


def test_corr_attention_pooling_varies_with_input():
    """关联特征不同应导致分数不同（注意力池化确实在起作用）。"""
    torch.manual_seed(0)
    critic = FundamentalCritic(in_dim=6)
    path = torch.randn(2, 10, 6)
    hist = torch.randn(2, 90, 6)
    corr_a = torch.randn(2, 4, 90, 6)
    corr_b = torch.randn(2, 4, 90, 6)
    out_a = critic(path, hist, corr_a)
    out_b = critic(path, hist, corr_b)
    assert not torch.allclose(out_a, out_b)  # 关联输入影响输出
