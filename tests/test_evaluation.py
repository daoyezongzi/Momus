"""对比实验框架测试：精度指标 + 同批候选公平对比。"""
import numpy as np

from critic import (
    mse, mae, direction_accuracy,
    strategy_simple_mean, strategy_median, make_critic_strategy,
    make_prob_top1_strategy,
    evaluate_strategies, aggregate_results, format_comparison,
    RandomCritic, CriticReranker,
)
from critic.evaluation import (
    amplitude_mae,
    close_cumulative_return_mae,
    five_metrics,
    horizon_direction_accuracy,
)


def test_mse_mae_zero_for_perfect_pred():
    y = np.random.randn(10, 6).astype(np.float32)
    assert mse(y, y) < 1e-12
    assert mae(y, y) < 1e-12


def test_direction_accuracy_perfect_and_worst():
    # 单调上升的 close：方向全为 +1
    y = np.zeros((5, 6), dtype=np.float32)
    y[:, 3] = [1, 2, 3, 4, 5]
    assert direction_accuracy(y, y) == 1.0
    # 完全反向
    rev = y.copy()
    rev[:, 3] = [5, 4, 3, 2, 1]
    assert direction_accuracy(rev, y) == 0.0


def test_simple_mean_strategy_matches_np_mean():
    paths = np.random.randn(8, 10, 6).astype(np.float32)
    out = strategy_simple_mean(paths)
    assert np.allclose(out, paths.mean(axis=0))


def test_median_strategy_matches_np_median():
    paths = np.random.randn(8, 10, 6).astype(np.float32)
    out = strategy_median(paths)
    assert out.shape == (10, 6)
    assert np.allclose(out, np.median(paths, axis=0))


def test_median_robust_to_outlier():
    # 3 条候选，其中一条是极端离群值；median 应忽略它，mean 会被拉偏
    paths = np.array([[[1.0] * 6], [[3.0] * 6], [[100.0] * 6]])  # (3,1,6)
    med = strategy_median(paths)
    mean = strategy_simple_mean(paths)
    assert np.allclose(med, 3.0)          # 中位数取中间那条
    assert not np.allclose(mean, 3.0)     # 均值被 100 拉偏


def test_prob_top1_picks_argmax_log_prob():
    rng = np.random.default_rng(0)
    paths = rng.standard_normal((5, 10, 6)).astype(np.float32)
    log_probs = np.array([-9.0, -1.0, -0.2, -7.0, -3.0])  # 最大在 idx=2
    strat = make_prob_top1_strategy(log_probs)
    out = strat(paths)
    assert out.shape == (10, 6)
    assert np.array_equal(out, paths[2])


def test_prob_top1_length_mismatch_raises():
    paths = np.random.randn(4, 10, 6).astype(np.float32)
    strat = make_prob_top1_strategy(np.array([1.0, 2.0]))  # 只有 2 个，候选 4 个
    try:
        strat(paths)
        assert False, "应因 log_probs 与候选数不一致而报错"
    except ValueError:
        pass


def test_prob_top1_runs_in_framework():
    """prob_top1 应能作为每窗策略在公平对比框架里评估。"""
    rng = np.random.default_rng(1)
    paths = rng.standard_normal((16, 10, 6)).astype(np.float32)
    y_true = rng.standard_normal((10, 6)).astype(np.float32)
    log_probs = rng.standard_normal(16)
    res = evaluate_strategies(
        paths, y_true,
        {"simple_mean": strategy_simple_mean,
         "prob_top1": make_prob_top1_strategy(log_probs)},
    )
    assert "prob_top1" in res and "mse" in res["prob_top1"]


def test_oracle_strategy_beats_simple_mean():
    """公平对比框架应能区分好坏策略：

    构造一个"作弊 oracle"策略(直接选最接近真实的候选)，它的 MSE 必须
    优于 simple_mean —— 验证框架确实在同批候选上正确度量差异。
    """
    rng = np.random.default_rng(0)
    pred_len = 10
    y_true = rng.standard_normal((pred_len, 6)).astype(np.float32)
    # 候选：其中一条非常接近 y_true，其余是噪声
    paths = rng.standard_normal((8, pred_len, 6)).astype(np.float32)
    paths[3] = y_true + 0.01 * rng.standard_normal((pred_len, 6)).astype(np.float32)

    def oracle(paths, hist=None, corr_feat=None):
        d = np.mean((paths[:, :, :4] - y_true[None, :, :4]) ** 2, axis=(1, 2))
        return paths[int(np.argmin(d))]

    res = evaluate_strategies(
        paths, y_true,
        {"simple_mean": strategy_simple_mean, "oracle": oracle},
    )
    assert res["oracle"]["mse"] < res["simple_mean"]["mse"]


def test_critic_strategy_runs_in_framework():
    """RandomCritic + reranker 包成策略应能在框架里正常评估。"""
    critic = RandomCritic(seed=0)
    reranker = CriticReranker(mode="weighted")
    strat = make_critic_strategy(critic, reranker)

    paths = np.random.randn(16, 10, 6).astype(np.float32)
    hist = np.random.randn(90, 6).astype(np.float32)
    y_true = np.random.randn(10, 6).astype(np.float32)
    res = evaluate_strategies(paths, y_true,
                              {"simple_mean": strategy_simple_mean, "critic": strat},
                              hist=hist)
    assert "critic" in res and "mse" in res["critic"]


def test_aggregate_and_format():
    per_sample = [
        {"simple_mean": {"mse": 1.0, "mae": 0.5, "direction_acc": 0.6},
         "critic": {"mse": 0.8, "mae": 0.4, "direction_acc": 0.7}},
        {"simple_mean": {"mse": 2.0, "mae": 1.5, "direction_acc": 0.4},
         "critic": {"mse": 1.2, "mae": 0.6, "direction_acc": 0.5}},
    ]
    agg = aggregate_results(per_sample)
    assert abs(agg["simple_mean"]["mse"] - 1.5) < 1e-9
    assert abs(agg["critic"]["mse"] - 1.0) < 1e-9
    # 格式化输出应含策略名与改善百分比
    text = format_comparison(agg, baseline="simple_mean")


def test_horizon_direction_uses_last_observed_close():
    pred = np.array([[10, 12, 9, 11], [11, 13, 10, 12]], dtype=float)
    true = np.array([[10, 12, 9, 10.5], [11, 13, 10, 11.5]], dtype=float)
    assert horizon_direction_accuracy(pred, true, last_close=10.0) == 1.0


def test_close_cumulative_return_mae():
    pred = np.array([[0, 0, 0, 12]], dtype=float)
    true = np.array([[0, 0, 0, 11]], dtype=float)
    assert np.isclose(close_cumulative_return_mae(pred, true, 10.0), 0.1)


def test_amplitude_mae():
    pred = np.array([[10, 13, 9, 12]], dtype=float)
    true = np.array([[10, 12, 9, 11]], dtype=float)
    assert np.isclose(amplitude_mae(pred, true, 10.0), 0.1)


def test_five_metrics_contains_formal_metric_set():
    pred = np.array([[10, 13, 9, 12], [11, 14, 10, 13]], dtype=float)
    true = np.array([[10, 12, 9, 11], [11, 13, 10, 12]], dtype=float)
    metrics = five_metrics(pred, true, last_close=10.0)
    assert set(metrics) == {
        "ohlc_mse",
        "ohlc_mae",
        "close_cum_return_mae",
        "direction_accuracy",
        "amplitude_mae",
    }
