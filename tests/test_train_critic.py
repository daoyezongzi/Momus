"""训练逻辑测试（不依赖 Kronos，纯张量）。

重点验证：
- compute_distances 正确性
- pairwise_ranking_loss 的排序语义与边界
- 训练能收敛，且训练后 Critic 学会把"低距离候选"排前面（端到端学习验证）
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from critic import (
    CriticSample, compute_distances, pairwise_ranking_loss,
    train_critic, FundamentalCritic,
)
from critic.train_critic import (
    compute_directional_trading_distances,
    compute_hybrid_trading_distances,
    compute_return_weighted_distances,
    compute_coarse_trading_distances,
)


def test_compute_distances_zero_for_exact_match():
    y_true = np.random.randn(10, 6).astype(np.float32)
    paths = np.stack([y_true, y_true + 1.0])  # 第0条完全匹配
    d = compute_distances(paths, y_true)
    assert d.shape == (2,)
    assert d[0] < 1e-6           # 完全匹配距离≈0
    assert d[1] > d[0]           # 偏离的更大


def test_ranking_loss_lower_when_correctly_ordered():
    """分数与距离负相关（好路径高分）时，损失应更小。"""
    distances = torch.tensor([0.1, 0.5, 0.9])
    good_scores = torch.tensor([2.0, 0.0, -2.0])   # 距离小→分高，排序正确
    bad_scores = torch.tensor([-2.0, 0.0, 2.0])    # 完全反向
    l_good = pairwise_ranking_loss(good_scores, distances)
    l_bad = pairwise_ranking_loss(bad_scores, distances)
    assert l_good < l_bad


def test_ranking_loss_zero_when_all_distances_equal():
    """无有序对（距离全相等）时损失为 0，且保持可反传。"""
    scores = torch.randn(5, requires_grad=True)
    distances = torch.ones(5)
    loss = pairwise_ranking_loss(scores, distances)
    assert torch.allclose(loss, torch.zeros(()))
    loss.backward()
    assert scores.grad is not None
    assert torch.allclose(scores.grad, torch.zeros(5))


def test_ranking_loss_is_differentiable():
    scores = torch.randn(4, requires_grad=True)
    distances = torch.tensor([0.1, 0.2, 0.3, 0.4])
    loss = pairwise_ranking_loss(scores, distances)
    loss.backward()
    assert scores.grad is not None
    assert not torch.allclose(scores.grad, torch.zeros(4))


def _make_learnable_samples(n_samples=12, n_cand=8, pred_len=8, lookback=40, seed=0):
    """构造可学习样本：候选路径的"均值幅度"与距离强相关，Critic 应能学到。

    让距离标签直接由路径的一个可观测统计量决定，确保存在可学习的信号。
    """
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_samples):
        paths = rng.standard_normal((n_cand, pred_len, 6)).astype(np.float32)
        hist = rng.standard_normal((lookback, 6)).astype(np.float32)
        # 距离 = 路径 close 列均值的平方（可从路径本身观测到的信号）
        signal = paths[:, :, 3].mean(axis=1)
        distances = (signal ** 2).astype(np.float32)
        samples.append(CriticSample(paths=paths, hist=hist, corr_feat=None,
                                    distances=distances))
    return samples


def test_directional_trading_distances_prefer_correct_long_direction():
    hist = np.zeros((3, 6), dtype=np.float32)
    hist[-1, 3] = 1.0
    paths = np.zeros((3, 2, 6), dtype=np.float32)
    paths[:, :, 3] = np.array([
        [1.0, 1.2],  # bullish, correct when future is up
        [1.0, 0.8],  # bearish, wrong when future is up
        [1.0, 1.0],  # flat / no-trade
    ], dtype=np.float32)
    y_true_norm = np.zeros((2, 6), dtype=np.float32)
    y_true_norm[-1, 3] = 1.1

    distances = compute_directional_trading_distances(paths, hist, y_true_norm, long_only=True)

    assert distances[0] < distances[1]
    assert distances[0] < distances[2]
    assert distances.shape == (3,)


def test_return_weighted_distances_reward_correct_magnitude():
    hist = np.zeros((3, 6), dtype=np.float32)
    hist[-1, 3] = 1.0
    paths = np.zeros((4, 2, 6), dtype=np.float32)
    paths[:, :, 3] = np.array([
        [1.0, 1.3],  # strong bullish, best when future is up
        [1.0, 1.1],  # weak bullish
        [1.0, 0.8],  # bearish, wrong
        [1.0, 1.0],  # flat
    ], dtype=np.float32)
    y_true_norm = np.zeros((2, 6), dtype=np.float32)
    y_true_norm[-1, 3] = 1.2

    distances = compute_return_weighted_distances(paths, hist, y_true_norm, long_only=True)

    assert distances[0] < distances[1]
    assert distances[1] < distances[2]
    assert distances[1] < distances[3]
    assert distances.shape == (4,)


def test_coarse_trading_distances_respect_component_weights():
    hist = np.zeros((3, 6), dtype=np.float32)
    hist[-1, 3] = 1.0
    paths = np.zeros((3, 2, 6), dtype=np.float32)
    paths[:, :, 3] = np.array([
        [1.0, 1.2],
        [1.0, 0.8],
        [1.0, 1.5],
    ], dtype=np.float32)
    y_true_norm = np.zeros((2, 6), dtype=np.float32)
    y_true_norm[-1, 3] = 1.2

    direction_only = compute_coarse_trading_distances(
        paths, hist, y_true_norm, direction_weight=1.0, endpoint_weight=0.0
    )
    endpoint_only = compute_coarse_trading_distances(
        paths, hist, y_true_norm, direction_weight=0.0, endpoint_weight=1.0
    )

    assert direction_only[0] == direction_only[2]
    assert direction_only[1] > direction_only[0]
    assert endpoint_only[0] < endpoint_only[2]
    assert endpoint_only[0] < endpoint_only[1]


def test_coarse_trading_distances_treat_flat_realized_move_as_zero_direction():
    hist = np.zeros((3, 6), dtype=np.float32)
    hist[-1, 3] = 1.0
    paths = np.zeros((3, 2, 6), dtype=np.float32)
    paths[:, :, 3] = np.array([
        [1.0, 1.0],
        [1.0, 1.1],
        [1.0, 0.9],
    ], dtype=np.float32)
    y_true_norm = np.zeros((2, 6), dtype=np.float32)
    y_true_norm[-1, 3] = 1.0

    distances = compute_coarse_trading_distances(
        paths, hist, y_true_norm, direction_weight=1.0, endpoint_weight=0.0
    )

    assert distances[0] < distances[1]
    assert distances[0] < distances[2]


def test_train_critic_dual_head_threads_multitask_weights_through_validation():
    from critic import DualHeadCritic

    torch.manual_seed(0)
    train_samples = _make_learnable_samples(n_samples=2, n_cand=4, seed=13)
    val_samples = _make_learnable_samples(n_samples=2, n_cand=4, seed=14)
    for sample in train_samples + val_samples:
        sample.fine_distances = sample.distances.copy()
        sample.coarse_distances = sample.distances[::-1].copy()
    critic = DualHeadCritic(in_dim=6)

    history = train_critic(
        critic,
        train_samples,
        epochs=2,
        lr=1e-3,
        device="cpu",
        verbose=False,
        val_samples=val_samples,
        restore_best=True,
        fine_loss_weight=0.5,
        coarse_loss_weight=0.5,
    )

    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2
    assert "restored_val_loss" in history
    assert np.isfinite(history["restored_val_loss"])


def test_hybrid_trading_distances_keep_path_quality_and_penalize_wrong_direction():
    hist = np.zeros((3, 6), dtype=np.float32)
    hist[-1, 3] = 1.0
    paths = np.zeros((3, 2, 6), dtype=np.float32)
    paths[:, :, 3] = np.array([
        [1.0, 1.18],  # close to true and correct direction
        [1.0, 1.5],   # correct direction but worse endpoint error
        [1.0, 0.9],   # wrong direction
    ], dtype=np.float32)
    y_true_norm = np.zeros((2, 6), dtype=np.float32)
    y_true_norm[-1, 3] = 1.2

    distances = compute_hybrid_trading_distances(
        paths,
        hist,
        y_true_norm,
        direction_weight=1.0,
        endpoint_weight=1.0,
    )

    assert distances[0] < distances[1]
    assert distances[0] < distances[2]
    assert distances.shape == (3,)



def test_train_critic_full_build_samples_supports_hybrid_reward(monkeypatch):
    import examples.train_critic_full as train_full

    hist = pd.DataFrame(
        {
            "timestamps": pd.date_range("2022-01-01", periods=3),
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1.0, 1.0, 1.0],
            "amount": [1.0, 1.0, 1.0],
        }
    )
    fut = pd.DataFrame(
        {
            "timestamps": pd.date_range("2022-01-04", periods=2),
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.1, 1.2],
            "volume": [1.0, 1.0],
            "amount": [1.0, 1.0],
        }
    )
    preds = np.zeros((1, 3, 2, 6), dtype=np.float32)
    preds[0, :, :, 3] = np.array([[1.1, 1.2], [1.1, 0.9], [1.1, 1.5]], dtype=np.float32)

    monkeypatch.setattr(train_full, "auto_regressive_inference", lambda *args, **kwargs: preds)
    monkeypatch.setattr(train_full, "corr_feat_local", lambda *args, **kwargs: None)

    samples = train_full.build_samples_batched(
        model=None,
        tokenizer=None,
        code="000001",
        windows=[(hist, fut)],
        pred_len=2,
        dev="cpu",
        sample_count=3,
        max_context=512,
        clip=5,
        data_dir="unused",
        batch_windows=1,
        reward_mode="hybrid",
    )

    assert len(samples) == 1
    assert samples[0].distances.shape == (3,)
    assert samples[0].distances[0] < samples[0].distances[1]


def test_training_reduces_loss():
    """训练若干 epoch 后损失应下降。"""
    torch.manual_seed(0)
    samples = _make_learnable_samples()
    critic = FundamentalCritic(in_dim=6)
    history = train_critic(critic, samples, epochs=15, lr=1e-3,
                           device="cpu", verbose=False)
    assert history[-1] < history[0]  # 末轮损失低于首轮


def test_train_critic_reports_validation_history():
    torch.manual_seed(0)
    train_samples = _make_learnable_samples(n_samples=8, seed=1)
    val_samples = _make_learnable_samples(n_samples=4, seed=2)
    critic = FundamentalCritic(in_dim=6)

    history = train_critic(
        critic,
        train_samples,
        epochs=3,
        lr=1e-3,
        device="cpu",
        verbose=False,
        val_samples=val_samples,
    )

    assert {"train_loss", "val_loss", "best_epoch", "best_val_loss"}.issubset(history)
    assert len(history["train_loss"]) == 3
    assert len(history["val_loss"]) == 3
    assert all(np.isfinite(history["train_loss"]))
    assert all(np.isfinite(history["val_loss"]))


def test_train_critic_restores_best_validation_checkpoint():
    torch.manual_seed(0)
    train_samples = _make_learnable_samples(n_samples=10, seed=3)
    val_samples = _make_learnable_samples(n_samples=5, seed=4)
    critic = FundamentalCritic(in_dim=6)

    history = train_critic(
        critic,
        train_samples,
        epochs=8,
        lr=1e-3,
        device="cpu",
        verbose=False,
        val_samples=val_samples,
        restore_best=True,
    )

    best_epoch = int(np.argmin(history["val_loss"])) + 1
    assert history["best_epoch"] == best_epoch
    assert np.isclose(history["best_val_loss"], min(history["val_loss"]))
    restored_val_loss = history["restored_val_loss"]
    assert np.isclose(restored_val_loss, history["best_val_loss"], atol=1e-6)


def test_train_critic_saves_best_checkpoint_file():
    torch.manual_seed(0)
    train_samples = _make_learnable_samples(n_samples=10, seed=5)
    val_samples = _make_learnable_samples(n_samples=5, seed=6)
    best_path = Path("output/results/test_best_checkpoint.pt")
    best_path.parent.mkdir(parents=True, exist_ok=True)
    if best_path.exists():
        best_path.unlink()
    critic = FundamentalCritic(in_dim=6)

    history = train_critic(
        critic,
        train_samples,
        epochs=4,
        lr=1e-3,
        device="cpu",
        verbose=False,
        val_samples=val_samples,
        best_path=str(best_path),
    )

    try:
        assert best_path.exists()
        assert history["best_path"] == str(best_path)
        saved_state = torch.load(best_path, map_location="cpu")
        assert set(saved_state) == set(critic.state_dict())
    finally:
        if best_path.exists():
            best_path.unlink()


def test_train_critic_saves_requested_epoch_checkpoints():
    torch.manual_seed(0)
    train_samples = _make_learnable_samples(n_samples=8, seed=7)
    checkpoint_dir = Path("output/results/test_epoch_checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for path in checkpoint_dir.glob("critic_e*.pt"):
        path.unlink()
    critic = FundamentalCritic(in_dim=6)

    history = train_critic(
        critic,
        train_samples,
        epochs=4,
        lr=1e-3,
        device="cpu",
        verbose=False,
        save_epochs={1, 3},
        checkpoint_dir=str(checkpoint_dir),
    )

    try:
        assert (checkpoint_dir / "critic_e1.pt").exists()
        assert (checkpoint_dir / "critic_e3.pt").exists()
        assert not (checkpoint_dir / "critic_e2.pt").exists()
        assert history["checkpoint_paths"] == [
            str(checkpoint_dir / "critic_e1.pt"),
            str(checkpoint_dir / "critic_e3.pt"),
        ]
    finally:
        for path in checkpoint_dir.glob("critic_e*.pt"):
            path.unlink()


def test_train_critic_full_parser_accepts_dual_head_args():
    import examples.train_critic_full as train_full

    args = train_full.parse_args([
        "--critic-arch", "dual_head",
        "--fine-loss-weight", "0.25",
        "--coarse-loss-weight", "0.75",
        "--final-combiner", "residual",
    ])

    assert args.critic_arch == "dual_head"
    assert args.fine_loss_weight == 0.25
    assert args.coarse_loss_weight == 0.75
    assert args.final_combiner == "residual"


def test_train_critic_full_build_samples_populates_dual_head_distances(monkeypatch):
    import examples.train_critic_full as train_full

    hist = pd.DataFrame(
        {
            "timestamps": pd.date_range("2022-01-01", periods=3),
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1.0, 1.0, 1.0],
            "amount": [1.0, 1.0, 1.0],
        }
    )
    fut = pd.DataFrame(
        {
            "timestamps": pd.date_range("2022-01-04", periods=2),
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.1, 1.2],
            "volume": [1.0, 1.0],
            "amount": [1.0, 1.0],
        }
    )
    preds = np.zeros((1, 3, 2, 6), dtype=np.float32)
    preds[0, :, :, 3] = np.array([[1.1, 1.2], [1.1, 0.9], [1.1, 1.5]], dtype=np.float32)

    monkeypatch.setattr(train_full, "auto_regressive_inference", lambda *args, **kwargs: preds)
    monkeypatch.setattr(train_full, "corr_feat_local", lambda *args, **kwargs: None)

    samples = train_full.build_samples_batched(
        model=None,
        tokenizer=None,
        code="000001",
        windows=[(hist, fut)],
        pred_len=2,
        dev="cpu",
        sample_count=3,
        max_context=512,
        clip=5,
        data_dir="unused",
        batch_windows=1,
        reward_mode="hybrid",
        include_granularity_distances=True,
    )

    assert len(samples) == 1
    assert samples[0].fine_distances is not None
    assert samples[0].coarse_distances is not None
    assert samples[0].fine_distances.shape == samples[0].distances.shape
    assert samples[0].coarse_distances.shape == samples[0].distances.shape


def test_train_critic_full_parser_accepts_no_path_corr_interaction():
    import examples.train_critic_full as train_full
    # 默认：不带开关 -> 保持原行为
    default = train_full.parse_args(["--samples-cache", "x.pkl", "--critic-out", "o.pt"])
    assert default.no_path_corr_interaction is False
    # 带开关
    on = train_full.parse_args(
        ["--samples-cache", "x.pkl", "--critic-out", "o.pt", "--no-path-corr-interaction"])
    assert on.no_path_corr_interaction is True


def test_small_mlp_single_head_trains_without_path_corr_interaction():
    """small_mlp baseline：单头 FundamentalCritic(path_corr_interaction=False) 可训练收敛。"""
    torch.manual_seed(0)
    samples = _make_learnable_samples(n_samples=12, seed=1)
    critic = FundamentalCritic(in_dim=6, setwise=False, path_corr_interaction=False)
    assert critic.path_corr_interaction is False
    history = train_critic(critic, samples, epochs=15, lr=1e-3,
                           device="cpu", verbose=False)
    assert np.isfinite(history[0]) and np.isfinite(history[-1])
    assert history[-1] < history[0]  # 单头 MLP 也能学到排序信号


def test_train_critic_full_parser_accepts_best_checkpoint_args():
    import examples.train_critic_full as train_full

    args = train_full.parse_args([
        "--val-ratio", "0.2",
        "--restore-best",
        "--best-critic-out", "best.pt",
        "--best-metric", "val_loss",
    ])

    assert args.restore_best is True
    assert args.best_critic_out == "best.pt"
    assert args.best_metric == "val_loss"


def test_train_critic_full_parser_accepts_epoch_checkpoint_args():
    import examples.train_critic_full as train_full

    args = train_full.parse_args([
        "--save-epochs", "1,3,5,7",
        "--checkpoint-dir", "output/results/epoch_sweep",
    ])

    assert args.save_epochs == "1,3,5,7"
    assert args.checkpoint_dir == "output/results/epoch_sweep"


def test_trained_critic_ranks_better_than_random():
    """训练后 Critic 对候选的排序应优于随机（与真实距离的秩相关为负）。"""
    torch.manual_seed(0)
    np.random.seed(0)
    samples = _make_learnable_samples(n_samples=20, seed=1)
    critic = FundamentalCritic(in_dim=6)
    train_critic(critic, samples, epochs=30, lr=1e-3, device="cpu", verbose=False)

    # 在新样本上评估：高分候选是否确实距离更小
    test_samples = _make_learnable_samples(n_samples=5, seed=99)
    agree = 0
    total = 0
    for s in test_samples:
        scores = critic.score(s.paths, hist=s.hist, corr_feat=None)
        # 分数最高的候选，其距离应偏小（低于该样本中位数）
        best = int(np.argmax(scores))
        if s.distances[best] <= np.median(s.distances):
            agree += 1
        total += 1
    assert agree >= 3  # 5 个样本中至少 3 个命中（远优于随机 ~2.5）


def test_compute_coarse_trading_distances_combines_direction_and_endpoint():
    hist = np.zeros((3, 6), dtype=np.float32)
    hist[-1, 3] = 1.0
    paths = np.zeros((3, 2, 6), dtype=np.float32)
    paths[:, :, 3] = np.array([
        [1.0, 1.2],
        [1.0, 0.8],
        [1.0, 1.5],
    ], dtype=np.float32)
    y_true_norm = np.zeros((2, 6), dtype=np.float32)
    y_true_norm[-1, 3] = 1.2

    distances = compute_coarse_trading_distances(paths, hist, y_true_norm)

    assert distances[0] < distances[1]
    assert distances[0] < distances[2]


def test_dual_head_sample_loss_uses_fine_and_coarse_labels():
    from critic import DualHeadCritic
    from critic.train_critic import _sample_ranking_loss

    torch.manual_seed(0)
    sample = _make_learnable_samples(n_samples=1, n_cand=4, seed=11)[0]
    sample.fine_distances = sample.distances.copy()
    sample.coarse_distances = sample.distances[::-1].copy()
    critic = DualHeadCritic(in_dim=6)

    total_loss = _sample_ranking_loss(
        critic,
        sample,
        device="cpu",
        use_corr=True,
        fine_loss_weight=0.5,
        coarse_loss_weight=0.5,
    )

    n = sample.paths.shape[0]
    path_t = torch.from_numpy(sample.paths.astype(np.float32))
    hist_t = torch.from_numpy(sample.hist.astype(np.float32)).unsqueeze(0).expand(n, -1, -1)
    outputs = critic(path_t, hist_t, None)
    expected = (
        pairwise_ranking_loss(outputs.final, torch.from_numpy(sample.distances.astype(np.float32)))
        + 0.5 * pairwise_ranking_loss(outputs.fine, torch.from_numpy(sample.fine_distances.astype(np.float32)))
        + 0.5 * pairwise_ranking_loss(outputs.coarse, torch.from_numpy(sample.coarse_distances.astype(np.float32)))
    )

    total_loss.backward()
    assert torch.allclose(total_loss.detach(), expected.detach())


def test_dual_head_sample_loss_requires_enabled_granularity_labels():
    from critic import DualHeadCritic
    from critic.train_critic import _sample_ranking_loss

    sample = _make_learnable_samples(n_samples=1, n_cand=4, seed=12)[0]
    critic = DualHeadCritic(in_dim=6)

    with pytest.raises(ValueError, match="fine_distances"):
        _sample_ranking_loss(
            critic,
            sample,
            device="cpu",
            use_corr=True,
            fine_loss_weight=0.5,
        )

    sample.fine_distances = sample.distances.copy()
    with pytest.raises(ValueError, match="coarse_distances"):
        _sample_ranking_loss(
            critic,
            sample,
            device="cpu",
            use_corr=True,
            coarse_loss_weight=0.5,
        )
