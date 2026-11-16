"""对比实验框架（设计文档 §8）：加 Critic vs 不加的公平对比。

核心设计：在**同一批候选路径**上套用不同重排序策略，消除采样随机性干扰，
保证对比公平。这是消融实验（E1 精度）能得出可信结论的前提。

策略（strategy）统一签名：strategy(paths, hist, corr_feat) -> final_path
  - simple_mean: baseline，简单平均（等价原 Kronos 行为）
  - critic_weighted: Critic softmax 加权
  - critic_top1: Critic 选最高分单条

精度指标（在归一化或原始空间均可，需保持 y_true 同空间）：
  - MSE / MAE：逐点误差
  - 方向准确率：预测的涨跌方向与真实是否一致（金融最关心）
"""
import numpy as np


# ============ 精度指标（E1） ============

def mse(pred: np.ndarray, y_true: np.ndarray, cols=(0, 1, 2, 3)) -> float:
    p, t = pred[:, list(cols)], y_true[:, list(cols)]
    return float(np.mean((p - t) ** 2))


def mae(pred: np.ndarray, y_true: np.ndarray, cols=(0, 1, 2, 3)) -> float:
    p, t = pred[:, list(cols)], y_true[:, list(cols)]
    return float(np.mean(np.abs(p - t)))


def direction_accuracy(pred: np.ndarray, y_true: np.ndarray, close_col=3) -> float:
    """方向准确率：预测 close 的逐日涨跌方向与真实一致的比例。

    金融预测中方向往往比绝对值更重要。返回 [0,1]。
    """
    p = np.diff(pred[:, close_col])
    t = np.diff(y_true[:, close_col])
    if len(p) == 0:
        return 0.0
    return float(np.mean(np.sign(p) == np.sign(t)))



def close_cumulative_return_mae(pred: np.ndarray, y_true: np.ndarray, last_close: float,
                                close_col=3) -> float:
    pred_return = pred[-1, close_col] / last_close - 1.0
    true_return = y_true[-1, close_col] / last_close - 1.0
    return float(abs(pred_return - true_return))


def horizon_direction_accuracy(pred: np.ndarray, y_true: np.ndarray, last_close: float,
                                close_col=3) -> float:
    pred_direction = np.sign(pred[-1, close_col] - last_close)
    true_direction = np.sign(y_true[-1, close_col] - last_close)
    return float(pred_direction == true_direction)


def amplitude_mae(pred: np.ndarray, y_true: np.ndarray, last_close: float,
                  high_col=1, low_col=2) -> float:
    pred_amplitude = (np.max(pred[:, high_col]) - np.min(pred[:, low_col])) / last_close
    true_amplitude = (np.max(y_true[:, high_col]) - np.min(y_true[:, low_col])) / last_close
    return float(abs(pred_amplitude - true_amplitude))


def five_metrics(pred: np.ndarray, y_true: np.ndarray, last_close: float) -> dict[str, float]:
    return {
        "ohlc_mse": mse(pred, y_true),
        "ohlc_mae": mae(pred, y_true),
        "close_cum_return_mae": close_cumulative_return_mae(pred, y_true, last_close),
        "direction_accuracy": horizon_direction_accuracy(pred, y_true, last_close),
        "amplitude_mae": amplitude_mae(pred, y_true, last_close),
    }


# ============ 重排序策略（同一批候选上对比） ============

def strategy_simple_mean(paths, hist=None, corr_feat=None):
    """baseline：简单平均，等价原 Kronos 的 np.mean。"""
    return np.mean(paths, axis=0)


def strategy_median(paths, hist=None, corr_feat=None):
    """baseline：逐点中位数。对极端候选路径鲁棒的统计聚合。

    与 simple_mean 同签名，融合在归一化空间进行（predictor 负责反归一化）。
    """
    return np.median(paths, axis=0)


def make_critic_strategy(critic, reranker):
    """把 (critic, reranker) 包成统一签名的策略函数。"""
    def _strategy(paths, hist=None, corr_feat=None):
        scores = critic.score(paths, hist=hist, corr_feat=corr_feat)
        return reranker.combine(paths, scores)
    return _strategy


def make_prob_top1_strategy(log_probs):
    """baseline：取生成对数似然最大的单条候选（生成器最自信路径）。

    log_probs 由 predictor.generate_paths(return_log_probs=True) 提供，是每条候选
    路径在 temperature-scaled 采样分布上的 Σ_t[logp(s1)+logp(s2)]。每个评估窗口的
    候选各自一组，故用工厂按窗口捕获对应 log_probs。

    与 critic_top1 严格配对（同为取单条最优，只换排序信号）：证明"见过真值的
    critic 排序"优于"生成器似然排序"。
    """
    log_probs = np.asarray(log_probs)

    def _strategy(paths, hist=None, corr_feat=None):
        if log_probs.shape[0] != paths.shape[0]:
            raise ValueError(
                f"log_probs 长度 {log_probs.shape[0]} 与候选数 {paths.shape[0]} 不一致")
        return paths[int(np.argmax(log_probs))]
    return _strategy


# ============ 对比评估 ============

def evaluate_strategies(paths, y_true, strategies, hist=None, corr_feat=None):
    """在同一批候选上评估多种策略的精度。

    Args:
        paths: (N, pred_len, feat) 候选路径。
        y_true: (pred_len, feat) 真实未来（与 paths 同一空间）。
        strategies: dict[str, callable]，策略名 -> 策略函数。
        hist, corr_feat: 传给 Critic 类策略。

    Returns:
        dict[str, dict]：每个策略的 {mse, mae, direction_acc}。
    """
    results = {}
    for name, strat in strategies.items():
        final = strat(paths, hist=hist, corr_feat=corr_feat)
        results[name] = {
            "mse": mse(final, y_true),
            "mae": mae(final, y_true),
            "direction_acc": direction_accuracy(final, y_true),
        }
    return results


def aggregate_results(per_sample_results):
    """把多个样本的评估结果按策略聚合成均值。

    Args:
        per_sample_results: List[dict[str, dict]]，evaluate_strategies 的输出列表。

    Returns:
        dict[str, dict]：每策略各指标的均值。
    """
    if not per_sample_results:
        return {}
    strategies = per_sample_results[0].keys()
    agg = {}
    for name in strategies:
        metrics = per_sample_results[0][name].keys()
        agg[name] = {
            m: float(np.mean([r[name][m] for r in per_sample_results]))
            for m in metrics
        }
    return agg


def format_comparison(agg_results, baseline="simple_mean") -> str:
    """把聚合结果格式化成可读对比表，含相对 baseline 的提升。"""
    if not agg_results:
        return "(无结果)"
    lines = [f"{'策略':<18}{'MSE':>10}{'MAE':>10}{'方向准确率':>12}{'MSE改善':>10}"]
    lines.append("-" * 62)
    base_mse = agg_results.get(baseline, {}).get("mse", None)
    for name, m in agg_results.items():
        improve = ""
        if base_mse is not None and base_mse > 0 and name != baseline:
            improve = f"{(base_mse - m['mse']) / base_mse * 100:+.1f}%"
        lines.append(f"{name:<18}{m['mse']:>10.4f}{m['mae']:>10.4f}"
                     f"{m['direction_acc']:>12.2%}{improve:>10}")
    return "\n".join(lines)
