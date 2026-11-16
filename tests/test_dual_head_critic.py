import numpy as np
import pytest
import torch

from critic import DualHeadCritic
from critic.model import CriticScoreOutput


class _ConstantHead(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)

    def forward(self, x):
        return torch.full((x.shape[0], 1), self.value, dtype=x.dtype, device=x.device)


class _ConstantBaseHead(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = float(value)

    def forward(self, x):
        return torch.full((x.shape[0], 1), self.value, dtype=x.dtype, device=x.device)


def test_dual_head_critic_forward_returns_three_scores():
    torch.manual_seed(0)
    critic = DualHeadCritic(in_dim=6)
    path = torch.randn(4, 10, 6)
    hist = torch.randn(4, 60, 6)

    out = critic(path, hist)

    assert isinstance(out, CriticScoreOutput)
    assert out.final.shape == (4,)
    assert out.fine.shape == (4,)
    assert out.coarse.shape == (4,)


def test_dual_head_critic_score_uses_final_score_by_default():
    torch.manual_seed(0)
    critic = DualHeadCritic(in_dim=6)
    paths = np.random.default_rng(0).standard_normal((3, 10, 6)).astype(np.float32)
    hist = np.random.default_rng(1).standard_normal((60, 6)).astype(np.float32)

    scores = critic.score(paths, hist)

    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))


def test_dual_head_critic_score_supports_each_head():
    torch.manual_seed(0)
    critic = DualHeadCritic(in_dim=6)
    paths = np.random.default_rng(2).standard_normal((3, 10, 6)).astype(np.float32)
    hist = np.random.default_rng(3).standard_normal((60, 6)).astype(np.float32)

    fine_scores = critic.score(paths, hist, head="fine")
    coarse_scores = critic.score(paths, hist, head="coarse")

    assert fine_scores.shape == (3,)
    assert coarse_scores.shape == (3,)
    assert np.all(np.isfinite(fine_scores))
    assert np.all(np.isfinite(coarse_scores))


def test_dual_head_critic_score_rejects_unknown_head():
    critic = DualHeadCritic(in_dim=6)
    paths = np.zeros((2, 10, 6), dtype=np.float32)
    hist = np.zeros((60, 6), dtype=np.float32)

    with pytest.raises(ValueError, match="head must be"):
        critic.score(paths, hist, head="bad")


def test_dual_head_critic_score_accepts_corr_and_resonance_features():
    torch.manual_seed(0)
    critic = DualHeadCritic(in_dim=6, resonance_dim=4)
    paths = np.random.default_rng(4).standard_normal((3, 10, 6)).astype(np.float32)
    hist = np.random.default_rng(5).standard_normal((60, 6)).astype(np.float32)
    corr = np.random.default_rng(6).standard_normal((2, 60, 6)).astype(np.float32)
    resonance_1d = np.random.default_rng(7).standard_normal(4).astype(np.float32)
    resonance_2d = np.random.default_rng(8).standard_normal((3, 4)).astype(np.float32)

    corr_scores = critic.score(paths, hist, corr_feat=corr)
    resonance_1d_scores = critic.score(paths, hist, resonance_feat=resonance_1d)
    resonance_2d_scores = critic.score(paths, hist, resonance_feat=resonance_2d)

    assert corr_scores.shape == (3,)
    assert resonance_1d_scores.shape == (3,)
    assert resonance_2d_scores.shape == (3,)
    assert np.all(np.isfinite(corr_scores))
    assert np.all(np.isfinite(resonance_1d_scores))
    assert np.all(np.isfinite(resonance_2d_scores))




def test_dual_head_critic_reuses_shared_encoders():
    critic = DualHeadCritic(in_dim=6)

    assert hasattr(critic, "seq_encoder")
    assert hasattr(critic, "corr_encoder")
    assert not hasattr(critic, "fine_encoder")
    assert not hasattr(critic, "coarse_encoder")
    assert not hasattr(critic, "final_encoder")



def test_dual_head_critic_final_score_depends_on_fine_and_coarse_scores():
    torch.manual_seed(0)
    path = torch.randn(4, 10, 6)
    hist = torch.randn(4, 60, 6)
    critic = DualHeadCritic(in_dim=6)

    critic.fine_head = _ConstantHead(0.0)
    critic.coarse_head = _ConstantHead(0.0)
    base_final = critic(path, hist).final

    critic.fine_head = _ConstantHead(10.0)
    critic.coarse_head = _ConstantHead(-5.0)
    changed_final = critic(path, hist).final

    assert not torch.allclose(base_final, changed_final)


def test_dual_head_critic_residual_combiner_exposes_fine_and_coarse_weights():
    critic = DualHeadCritic(in_dim=6, final_combiner="residual")

    assert critic.final_combiner == "residual"
    assert hasattr(critic, "base_head")
    assert critic.residual_fine_weight.shape == torch.Size([])
    assert critic.residual_coarse_weight.shape == torch.Size([])


def test_dual_head_critic_residual_combiner_uses_explicit_score_terms():
    torch.manual_seed(0)
    path = torch.randn(4, 10, 6)
    hist = torch.randn(4, 60, 6)
    critic = DualHeadCritic(in_dim=6, final_combiner="residual")

    critic.base_head = _ConstantBaseHead(1.0)
    critic.fine_head = _ConstantHead(2.0)
    critic.coarse_head = _ConstantHead(3.0)
    critic.residual_fine_weight.data.fill_(0.5)
    critic.residual_coarse_weight.data.fill_(-0.25)

    out = critic(path, hist)

    assert torch.allclose(out.final, torch.full((4,), 1.25))


def test_dual_head_critic_calibrated_combiner_uses_score_only_final_head():
    critic = DualHeadCritic(in_dim=6, final_combiner="calibrated")

    assert critic.final_combiner == "calibrated"
    assert hasattr(critic, "base_head")
    assert critic.final_head[0].in_features == 3


def test_dual_head_critic_rejects_unknown_final_combiner():
    with pytest.raises(ValueError, match="final_combiner"):
        DualHeadCritic(in_dim=6, final_combiner="gate")


def test_setwise_off_by_default_no_mixer():
    critic = DualHeadCritic(in_dim=6)
    assert critic.setwise is False
    assert critic.set_mixer is None


def test_setwise_on_builds_mixer_and_forward_shapes():
    torch.manual_seed(0)
    critic = DualHeadCritic(in_dim=6, setwise=True)
    assert critic.set_mixer is not None
    out = critic(torch.randn(5, 10, 6), torch.randn(5, 60, 6))
    assert out.final.shape == (5,)
    assert out.fine.shape == (5,)
    assert out.coarse.shape == (5,)


def test_setwise_zero_init_equals_pointwise():
    """末层零初始化 => 训练起点 setwise 恒等 pointwise(干净消融基线)。"""
    torch.manual_seed(0)
    pw = DualHeadCritic(in_dim=6, setwise=False)
    torch.manual_seed(0)
    sw = DualHeadCritic(in_dim=6, setwise=True)
    path, hist = torch.randn(6, 10, 6), torch.randn(6, 60, 6)
    with torch.no_grad():
        assert torch.allclose(pw(path, hist).final, sw(path, hist).final, atol=1e-6)


def test_setwise_permutation_equivariance():
    """重排候选 => 输出同序重排(置换等变)。"""
    torch.manual_seed(1)
    critic = DualHeadCritic(in_dim=6, setwise=True)
    # 打破零初始化,让 mixer 真正起作用
    for p in critic.set_mixer.proj[-1].parameters():
        torch.nn.init.normal_(p, std=0.5)
    path, hist = torch.randn(7, 10, 6), torch.randn(7, 60, 6)
    perm = torch.randperm(7)
    with torch.no_grad():
        base = critic(path, hist).final
        permuted = critic(path[perm], hist[perm]).final
    assert torch.allclose(base[perm], permuted, atol=1e-5)


def test_setwise_score_depends_on_other_candidates():
    """setwise 核心:某候选的分数应随组内其他候选变化(pointwise 则不会)。"""
    torch.manual_seed(2)
    critic = DualHeadCritic(in_dim=6, setwise=True)
    for p in critic.set_mixer.proj[-1].parameters():
        torch.nn.init.normal_(p, std=0.5)
    hist = torch.randn(4, 60, 6)
    path_a = torch.randn(4, 10, 6)
    path_b = path_a.clone()
    path_b[1:] = torch.randn(3, 10, 6)  # 保持第 0 条不变,改其余候选
    with torch.no_grad():
        s_a = critic(path_a, hist).final[0]
        s_b = critic(path_b, hist).final[0]
    assert not torch.allclose(s_a, s_b, atol=1e-4)
