"""扩池 scaling 实验：梯度扩大训练股票池，测试跨股票泛化趋势。

训练池梯度（2024 数据）：
  G1: full86 (31 只)
  G2: full86 + new_pool[:30] (61 只)
  G3: full86 + new_pool[:60] (91 只)
  G4: full86 + new_pool[:86] (117 只)

Holdout: holdout_30 (30 只, 2025 数据)，不参与训练

用法：
    python examples/scaling_experiment.py --device cuda
"""
import argparse
import pickle
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, ".")

from critic.model import FundamentalCritic
from critic.train_critic import pairwise_ranking_loss
from critic.evaluation import strategy_simple_mean


def build_training_samples(cache_path):
    """从候选缓存构建简单训练样本：(paths, y_true_norm) → distances."""
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    windows = cache["windows"]
    candidates = cache["candidates"]

    all_paths = []
    all_distances = []

    for (code, x_df, x_ts, y_ts, y_true), cands in zip(windows, candidates):
        hist_arr = x_df[["open", "high", "low", "close", "volume", "amount"]].values.astype(np.float32)
        hist_mean = hist_arr.mean(axis=0, keepdims=True)
        hist_std = hist_arr.std(axis=0, keepdims=True) + 1e-5
        y_true_norm = (y_true - hist_mean) / hist_std

        # 逐候选计算 MSE distance（越小越好）
        dists = np.array([np.mean((cands[i] - y_true_norm) ** 2) for i in range(len(cands))], dtype=np.float32)

        all_paths.append(cands)
        all_distances.append(dists)

    return all_paths, all_distances


def build_holdout_data(cache_path):
    """从 holdout 缓存提取 paths, y_true_norm, hist 用于评估."""
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    windows = cache["windows"]
    candidates = cache["candidates"]

    data = []
    for (code, x_df, x_ts, y_ts, y_true), cands in zip(windows, candidates):
        hist_arr = x_df[["open", "high", "low", "close", "volume", "amount"]].values.astype(np.float32)
        hist_mean = hist_arr.mean(axis=0, keepdims=True)
        hist_std = hist_arr.std(axis=0, keepdims=True) + 1e-5
        y_true_norm = (y_true - hist_mean) / hist_std
        data.append((cands, y_true_norm, hist_arr))

    return data


def train_one_epoch(critic, optimizer, paths_list, dists_list, device, batch_size=32):
    """训练一个 epoch，使用 pairwise ranking loss。"""
    critic.train()
    total_loss = 0.0
    n_samples = len(paths_list)

    indices = np.random.permutation(n_samples)
    for start in range(0, n_samples, batch_size):
        batch_idx = indices[start:start + batch_size]

        optimizer.zero_grad()
        batch_loss = 0.0

        for idx in batch_idx:
            paths_np = paths_list[idx]
            dists_np = dists_list[idx]
            hist_np = np.zeros((60, 6), dtype=np.float32)

            path_t = torch.from_numpy(np.ascontiguousarray(paths_np)).to(device)
            hist_t = torch.from_numpy(np.ascontiguousarray(hist_np)).to(device)
            hist_t = hist_t.unsqueeze(0).expand(path_t.size(0), -1, -1)
            dists_t = torch.from_numpy(np.ascontiguousarray(dists_np)).to(device)

            out = critic.forward(path_t, hist_t, corr=None, resonance=None)
            if hasattr(out, 'final'):
                scores_t = out.final
            else:
                scores_t = out

            loss = pairwise_ranking_loss(scores_t, dists_t)
            batch_loss += loss

        batch_loss = batch_loss / len(batch_idx)
        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item() * len(batch_idx)

    return total_loss / n_samples


@torch.no_grad()
def evaluate_holdout_mse(critic, holdout_data, device):
    """在 holdout 上评估 weighted MSE vs simple_mean."""
    simple_mse_list = []
    critic_mse_list = []

    for cands, y_true_norm, hist_arr in holdout_data:
        N = len(cands)

        # simple_mean
        mean_path = cands.mean(axis=0)
        simple_mse = float(np.mean((mean_path - y_true_norm) ** 2))
        simple_mse_list.append(simple_mse)

        # critic_weighted
        scores = critic.score(cands, hist_arr)  # 传实际 hist
        scores_t = torch.from_numpy(scores).float()
        weights = F.softmax(scores_t, dim=0).numpy()
        weighted_path = np.sum(cands * weights[:, None, None], axis=0)
        critic_mse = float(np.mean((weighted_path - y_true_norm) ** 2))
        critic_mse_list.append(critic_mse)

    simple_avg = np.mean(simple_mse_list)
    critic_avg = np.mean(critic_mse_list)
    improvement = (simple_avg - critic_avg) / simple_avg * 100

    return simple_avg, critic_avg, improvement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = args.device

    # ---- 加载数据 ----
    print("加载训练缓存...")
    full86_paths, full86_dists = build_training_samples("output/candidates/full86_2024_N16.pkl")
    new_pool_paths, new_pool_dists = build_training_samples("output/candidates/new_pool_2024_N16.pkl")

    print(f"  full86: {len(full86_paths)} 样本")
    print(f"  new_pool_86: {len(new_pool_paths)} 样本")

    print("加载 holdout 缓存...")
    holdout_data = build_holdout_data("output/candidates/holdout_2025_N16.pkl")
    print(f"  holdout: {len(holdout_data)} 窗口")

    # ---- 评估 simple_mean baseline on holdout ----
    simple_mses = []
    for cands, y_true_norm, _ in holdout_data:
        mean_path = cands.mean(axis=0)
        simple_mses.append(float(np.mean((mean_path - y_true_norm) ** 2)))
    baseline_mse = np.mean(simple_mses)
    print(f"\nHoldout simple_mean MSE: {baseline_mse:.6f}")

    # ---- 梯度实验 ----
    pool_configs = [
        ("G1: 31 stocks (full86)", list(range(len(full86_paths))), list(range(len(full86_dists)))),
        ("G2: 61 stocks (+30)",
         list(range(len(full86_paths))) + list(range(len(full86_paths), len(full86_paths) + len(new_pool_paths[:30*15]))),
         list(range(len(full86_dists))) + list(range(len(full86_dists), len(full86_dists) + len(new_pool_dists[:30*15])))),
        ("G3: 91 stocks (+60)",
         list(range(len(full86_paths))) + list(range(len(full86_paths), len(full86_paths) + len(new_pool_paths[:60*15]))),
         list(range(len(full86_dists))) + list(range(len(full86_dists), len(full86_dists) + len(new_pool_dists[:60*15])))),
        ("G4: 117 stocks (full86 + new_pool_86)",
         list(range(len(full86_paths) + len(new_pool_paths))),
         list(range(len(full86_dists) + len(new_pool_dists)))),
    ]

    # 合并所有训练数据
    all_paths = full86_paths + new_pool_paths
    all_dists = full86_dists + new_pool_dists

    print(f"\n{'='*70}")
    print(f"Scaling 实验：训练池大小 vs holdout 泛化")
    print(f"{'='*70}")
    print(f"{'配置':<30} {'样本数':>8} {'TrainLoss':>10} {'HoldoutMSE':>12} {'Improvement':>12}")
    print(f"{'-'*70}")

    results = []
    for name, path_idx, dist_idx in pool_configs:
        pool_paths = [all_paths[i] for i in path_idx]
        pool_dists = [all_dists[i] for i in dist_idx]
        n_samples = len(pool_paths)

        # 训练
        critic = FundamentalCritic(in_dim=6, setwise=False, path_corr_interaction=True)
        critic = critic.to(device)
        optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr)

        final_loss = 0.0
        for epoch in range(args.epochs):
            loss = train_one_epoch(critic, optimizer, pool_paths, pool_dists, device)
            final_loss = loss

        # 评估
        simple_avg, critic_avg, improvement = evaluate_holdout_mse(critic, holdout_data, device)

        print(f"{name:<30} {n_samples:>8} {final_loss:>10.4f} {critic_avg:>12.6f} {improvement:>+11.2f}%")
        results.append((name, n_samples, final_loss, critic_avg, improvement))

    # ---- 汇总 ----
    print(f"\n{'='*70}")
    print(f"结论: holdout simple_mean MSE = {baseline_mse:.6f}")
    for name, n, loss, mse, imp in results:
        direction = "↑" if imp > 0 else "↓"
        print(f"  {name}: critic MSE={mse:.6f}, vs simple_mean {imp:+.2f}% {direction}")

    if results[-1][4] > results[0][4]:
        print(f"\n>>> 扩池改善了泛化: G1={results[0][4]:+.2f}% → G4={results[-1][4]:+.2f}%")
    else:
        print(f"\n>>> 扩池未改善泛化: 曲线平坦或下降，需检查架构")


if __name__ == "__main__":
    main()
