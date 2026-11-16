"""Critic 训练脚本（设计文档 §5.3）。

训练思路：历史数据里"未来"已知，用它当标准答案教 Critic 认出好路径。

分两阶段（解耦，因样本构造需跑 Kronos 推理，很慢）：
  阶段1 build_samples_for_series: 对 (股票, 历史时点 T) 用 Kronos 生成 N 条候选，
        算每条与真实未来的距离，存为样本。可缓存到磁盘。
  阶段2 train_critic: 用 pairwise ranking loss (Bradley-Terry) 训练。

关键红线：真实未来只用于阶段1 算距离标签，绝不作为 Critic 输入（推理期同样不输入）。
这是 reward-model 式训练，非数据泄漏（详见设计文档 §5.4、§6 讨论）。
"""
import copy
import os
import pickle
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from .schemas import CriticSample


# ============ 阶段1：样本构造（重，依赖 Kronos） ============


def compute_distances(paths: np.ndarray, y_true: np.ndarray,
                      price_cols=(0, 1, 2, 3)) -> np.ndarray:
    """算每条候选与真实未来路径的距离（默认对 OHLC 求 MSE）。

    Args:
        paths: (N, pred_len, feat) 候选路径（与 y_true 同一归一化空间）。
        y_true: (pred_len, feat) 真实未来路径。
        price_cols: 参与距离计算的列索引（默认 open/high/low/close）。
    Returns:
        (N,) 距离，越小表示越接近真实。
    """
    p = paths[:, :, list(price_cols)]
    t = y_true[None, :, list(price_cols)]
    return np.mean((p - t) ** 2, axis=(1, 2)).astype(np.float32)


def build_samples_for_series(predictor, df, x_timestamp, y_timestamp, pred_len,
                             target_code=None, sample_count=16, T=1.0,
                             top_k=0, top_p=0.9):
    """对单个 (股票, T) 生成候选并构造训练样本。依赖 Kronos（重）。

    利用 CriticKronosPredictor 底层的 return_individual 能力拿到 N 条候选，
    同时要求调用方提供该时点的真实未来 y_true（在归一化空间）。

    返回 CriticSample。注意：本函数按 predictor 的归一化口径处理，保证
    候选路径与 y_true 在同一空间可比。
    """
    import pandas as pd
    from model.kronos import auto_regressive_inference, calc_time_stamps

    price_cols = predictor.price_cols + [predictor.vol_col, predictor.amt_vol]
    x = df[price_cols].values.astype(np.float32)
    x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
    x_norm = np.clip((x - x_mean) / (x_std + 1e-5), -predictor.clip, predictor.clip)

    x_stamp = calc_time_stamps(x_timestamp).values.astype(np.float32)
    y_stamp = calc_time_stamps(y_timestamp).values.astype(np.float32)

    dev = predictor.device
    preds_all = auto_regressive_inference(
        predictor.tokenizer, predictor.model,
        torch.from_numpy(x_norm[None]).to(dev),
        torch.from_numpy(x_stamp[None]).to(dev),
        torch.from_numpy(y_stamp[None]).to(dev),
        predictor.max_context, pred_len, predictor.clip, T, top_k, top_p,
        sample_count, False, return_individual=True,
    )
    paths = preds_all[0][:, -pred_len:, :]  # (N, pred_len, feat)

    corr_feat = None
    if predictor.corr_provider is not None and target_code is not None:
        corr_feat = predictor.corr_provider.get(target_code, x_timestamp)

    return paths, x_norm, corr_feat, (x_mean, x_std)


def save_samples(samples, path):
    with open(path, "wb") as f:
        pickle.dump(samples, f)


def load_samples(path):
    with open(path, "rb") as f:
        return pickle.load(f)




def compute_directional_trading_distances(
    paths: np.ndarray,
    hist: np.ndarray,
    y_true_norm: np.ndarray,
    long_only: bool = True,
    min_return: float = 0.0,
) -> np.ndarray:
    """Trading-direction pseudo distance: lower is better for ranking.

    This converts path forecasts into a direction/reward label instead of an OHLC
    reconstruction label. For long-only experiments, bullish forecasts are good
    only when the realized normalized close move is above min_return; bearish or
    flat forecasts are treated as no-trade quality.
    """
    last_close = float(hist[-1, 3])
    realized = float(y_true_norm[-1, 3] - last_close)
    predicted = paths[:, -1, 3].astype(np.float32) - last_close

    if long_only:
        rewards = np.where(predicted > 0, realized - min_return, 0.0)
    else:
        rewards = np.sign(predicted) * realized
    return (-rewards).astype(np.float32)


def compute_return_weighted_distances(
    paths: np.ndarray,
    hist: np.ndarray,
    y_true_norm: np.ndarray,
    long_only: bool = True,
) -> np.ndarray:
    """Return-weighted pseudo distance: correct larger moves rank better."""
    last_close = float(hist[-1, 3])
    realized = float(y_true_norm[-1, 3] - last_close)
    predicted = paths[:, -1, 3].astype(np.float32) - last_close

    if long_only:
        rewards = np.where(predicted > 0, realized * np.maximum(predicted, 0.0), 0.0)
    else:
        rewards = np.sign(predicted) * realized * np.abs(predicted)
    return (-rewards).astype(np.float32)


def compute_coarse_trading_distances(
    paths: np.ndarray,
    hist: np.ndarray,
    y_true_norm: np.ndarray,
    direction_weight: float = 1.0,
    endpoint_weight: float = 1.0,
) -> np.ndarray:
    """Coarse pseudo distance: direction mismatch plus endpoint error."""
    last_close = float(hist[-1, 3])
    predicted = paths[:, -1, 3].astype(np.float32) - last_close
    realized = float(y_true_norm[-1, 3] - last_close)
    direction_penalty = (np.sign(predicted) != np.sign(realized)).astype(np.float32)
    endpoint_error = np.abs(predicted - realized).astype(np.float32)
    endpoint_scale = float(np.nanstd(endpoint_error))
    if endpoint_scale > 0:
        endpoint_component = (endpoint_error - float(np.nanmean(endpoint_error))) / endpoint_scale
    else:
        endpoint_component = np.zeros_like(endpoint_error, dtype=np.float32)
    return (direction_weight * direction_penalty + endpoint_weight * endpoint_component).astype(np.float32)


def compute_hybrid_trading_distances(
    paths: np.ndarray,
    hist: np.ndarray,
    y_true_norm: np.ndarray,
    direction_weight: float = 1.0,
    endpoint_weight: float = 1.0,
    path_weight: float = 1.0,
) -> np.ndarray:
    """Coarse+fine pseudo distance for candidate verification.

    Fine grain: normalized OHLC path MSE.
    Coarse grain: horizon direction mismatch penalty.
    Endpoint: final close return error.
    """
    path_dist = compute_distances(paths, y_true_norm).astype(np.float32)
    scale = float(np.nanstd(path_dist))
    if scale > 0:
        path_component = (path_dist - float(np.nanmean(path_dist))) / scale
    else:
        path_component = np.zeros_like(path_dist, dtype=np.float32)

    coarse = compute_coarse_trading_distances(
        paths,
        hist,
        y_true_norm,
        direction_weight=direction_weight,
        endpoint_weight=endpoint_weight,
    )
    hybrid = path_weight * path_component + coarse
    return hybrid.astype(np.float32)


def pairwise_ranking_loss(scores: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry pairwise 排序损失（设计文档 §5.3）。

    对每对 (i, j)，若 d_i < d_j（i 更接近真实），则希望 score_i > score_j：
        L_ij = -log σ(score_i - score_j)
    只统计 d_i < d_j 的有序对，遍历求平均。

    Args:
        scores: (N,) Critic 给每条候选的分数。
        distances: (N,) 每条候选与真实未来的距离（越小越好）。
    Returns:
        标量损失。
    """
    n = scores.shape[0]
    # 构造所有对的分数差与距离差
    s_diff = scores[:, None] - scores[None, :]       # (N, N): score_i - score_j
    d_i = distances[:, None].expand(n, n)
    d_j = distances[None, :].expand(n, n)
    # 掩码：i 严格优于 j（距离更小）
    better = (d_i < d_j).float()
    n_pairs = better.sum()
    if n_pairs < 1:
        return scores.sum() * 0.0  # 保持 autograd 图不断开，训练循环可安全 backward
    # -log σ(s_i - s_j)，仅对 i 优于 j 的对求和
    loss = -(better * F.logsigmoid(s_diff)).sum() / n_pairs
    return loss


def _sample_ranking_loss(critic, sample, device, use_corr=True,
                         fine_loss_weight=0.0, coarse_loss_weight=0.0):
    """Compute pairwise loss for one CriticSample without mutating optimizer state."""
    n = sample.paths.shape[0]
    path_t = torch.from_numpy(sample.paths.astype(np.float32)).to(device)
    hist_t = torch.from_numpy(sample.hist.astype(np.float32)).to(device)
    hist_t = hist_t.unsqueeze(0).expand(n, -1, -1)
    corr_t = None
    if use_corr and sample.corr_feat is not None:
        corr_t = torch.from_numpy(sample.corr_feat.astype(np.float32)).to(device)
        corr_t = corr_t.unsqueeze(0).expand(n, -1, -1, -1)
    dist_t = torch.from_numpy(sample.distances.astype(np.float32)).to(device)
    outputs = critic(path_t, hist_t, corr_t)
    output_heads = ("final", "fine", "coarse")
    has_head_output = [hasattr(outputs, name) for name in output_heads]
    if any(has_head_output) and not all(has_head_output):
        raise TypeError("critic output must include final, fine, and coarse scores")
    if all(has_head_output):
        total_loss = pairwise_ranking_loss(outputs.final, dist_t)
        if fine_loss_weight > 0:
            if sample.fine_distances is None:
                raise ValueError("fine_distances are required when fine_loss_weight > 0")
            fine_t = torch.from_numpy(sample.fine_distances.astype(np.float32)).to(device)
            total_loss = total_loss + fine_loss_weight * pairwise_ranking_loss(outputs.fine, fine_t)
        if coarse_loss_weight > 0:
            if sample.coarse_distances is None:
                raise ValueError("coarse_distances are required when coarse_loss_weight > 0")
            coarse_t = torch.from_numpy(sample.coarse_distances.astype(np.float32)).to(device)
            total_loss = total_loss + coarse_loss_weight * pairwise_ranking_loss(outputs.coarse, coarse_t)
        return total_loss
    return pairwise_ranking_loss(outputs, dist_t)


def _average_loss(critic, samples, device, use_corr=True,
                  fine_loss_weight=0.0, coarse_loss_weight=0.0):
    """Evaluate average ranking loss on samples."""
    if not samples:
        return float("nan")
    was_training = critic.training
    critic.eval()
    total = 0.0
    with torch.no_grad():
        for sample in samples:
            total += float(_sample_ranking_loss(
                critic,
                sample,
                device,
                use_corr=use_corr,
                fine_loss_weight=fine_loss_weight,
                coarse_loss_weight=coarse_loss_weight,
            ).item())
    if was_training:
        critic.train()
    return total / len(samples)


def train_critic(critic, samples, epochs=20, lr=1e-3, device="cpu", verbose=True,
                 use_corr=True, val_samples=None, restore_best=False, best_path=None,
                 save_epochs=None, checkpoint_dir=None,
                 fine_loss_weight=0.0, coarse_loss_weight=0.0):
    """训练循环。每个样本内部做 pairwise 排序。

    Args:
        critic: FundamentalCritic 实例。
        samples: List[CriticSample]。
        epochs, lr: 训练超参。
        use_corr: False 时忽略关联特征（消融用），强制 Critic 只用 path+hist。
        val_samples: 验证样本；提供后返回 train/val history。
        restore_best: True 时训练结束恢复验证集 loss 最低的权重。
        best_path: 提供时，每次验证集刷新最优即原子保存 best-val 权重。
        save_epochs: 指定 epoch 集合；训练到这些 epoch 时保存 checkpoint。
        checkpoint_dir: save_epochs 的输出目录。
    Returns:
        无验证集时返回每个 epoch 的平均损失列表；有验证集时返回 history dict。
    """
    if best_path is not None and val_samples is None:
        raise ValueError("best_path requires val_samples")
    if save_epochs is not None and checkpoint_dir is None:
        raise ValueError("checkpoint_dir requires save_epochs")
    save_epochs = set(save_epochs or [])
    checkpoint_paths = []
    critic = critic.to(device)
    critic.train()
    opt = torch.optim.Adam(critic.parameters(), lr=lr)

    history = []
    train_history = []
    val_history = []
    best_state = None
    best_epoch = None
    best_val_loss = float("inf")
    order = np.arange(len(samples))
    for ep in range(epochs):
        ep_loss = 0.0
        np.random.shuffle(order)  # 打乱索引，不就地修改调用方的 samples 列表
        for i in order:
            s = samples[i]
            loss = _sample_ranking_loss(
                critic,
                s,
                device,
                use_corr=use_corr,
                fine_loss_weight=fine_loss_weight,
                coarse_loss_weight=coarse_loss_weight,
            )

            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()

        avg = ep_loss / max(len(samples), 1)
        history.append(avg)
        train_history.append(avg)
        if val_samples is not None:
            val_avg = _average_loss(
                critic,
                val_samples,
                device,
                use_corr=use_corr,
                fine_loss_weight=fine_loss_weight,
                coarse_loss_weight=coarse_loss_weight,
            )
            val_history.append(val_avg)
            if np.isfinite(val_avg) and val_avg < best_val_loss:
                best_val_loss = val_avg
                best_epoch = ep + 1
                best_state = copy.deepcopy(critic.state_dict())
                if best_path is not None:
                    save_state_dict_atomic(best_state, best_path)
        current_epoch = ep + 1
        if current_epoch in save_epochs:
            ckpt_path = os.path.join(checkpoint_dir, f"critic_e{current_epoch}.pt")
            save_state_dict_atomic(critic.state_dict(), ckpt_path)
            checkpoint_paths.append(ckpt_path)
        if verbose:
            if val_samples is None:
                print(f"[epoch {current_epoch}/{epochs}] loss={avg:.4f}")
            else:
                print(f"[epoch {current_epoch}/{epochs}] loss={avg:.4f} val_loss={val_history[-1]:.4f}")
    if val_samples is not None:
        result = {"train_loss": train_history, "val_loss": val_history}
        if best_epoch is not None:
            result["best_epoch"] = best_epoch
            result["best_val_loss"] = best_val_loss
            if best_path is not None:
                result["best_path"] = best_path
        if checkpoint_paths:
            result["checkpoint_paths"] = checkpoint_paths
        if restore_best and best_state is not None:
            critic.load_state_dict(best_state)
            result["restored_val_loss"] = _average_loss(
                critic,
                val_samples,
                device,
                use_corr=use_corr,
                fine_loss_weight=fine_loss_weight,
                coarse_loss_weight=coarse_loss_weight,
            )
        return result
    if checkpoint_paths:
        return {"train_loss": train_history, "checkpoint_paths": checkpoint_paths}
    return history


def save_state_dict_atomic(state_dict, path):
    """Atomically save a state_dict to avoid partially written checkpoints."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".pt", dir=directory)
    os.close(fd)
    try:
        torch.save(state_dict, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def save_critic(critic, path):
    torch.save(critic.state_dict(), path)


def load_critic(critic, path, device="cpu"):
    critic.load_state_dict(torch.load(path, map_location=device))
    return critic
