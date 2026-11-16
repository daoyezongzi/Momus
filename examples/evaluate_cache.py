"""预注册基线评估：从候选缓存加载，跑全策略 E1 精度对比。

用法：
    python examples/evaluate_cache.py \
        --cache output/candidates/new_pool_2024_N16.pkl \
        --critic-ckpt output/results/dual_head_combiner_screen/calibrated/seed_0/fusion_best.pt \
        --small-mlp-ckpt checkpoints/small_mlp_30ep.pt \
        --period 2024
"""
import argparse
import pickle
import sys
import numpy as np
import torch

sys.path.insert(0, ".")

from critic import (
    FundamentalCritic,
    DualHeadCritic,
    CriticReranker,
    strategy_simple_mean,
    strategy_median,
    make_critic_strategy,
    make_prob_top1_strategy,
)


def load_critic(ckpt_path, arch, device, setwise=False, path_corr_interaction=True):
    """加载 critic checkpoint，自动识别 FundamentalCritic vs DualHeadCritic。"""
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)

    if arch == "auto":
        # 根据 key 数量推断: FundamentalCritic ~19 keys, DualHeadCritic ~31 keys
        has_fine = any("fine_head" in k for k in sd)
        arch = "dual_head" if has_fine else "fundamental"

    # 推断打分头宽度（head_hidden_dim 可与 encoder hidden_dim 不同，见参数量匹配对照臂）
    head_key = next((k for k in ("head.0.weight", "fine_head.0.weight", "base_head.0.weight")
                     if k in sd), None)
    head_hidden_dim = int(sd[head_key].shape[0]) if head_key is not None else None

    if arch == "dual_head":
        # 从 state_dict 推断 final_combiner
        has_base_head = any("base_head" in k for k in sd)
        has_final_fusion = any("fine_head" in k for k in sd) and any("final_head" in k for k in sd)
        if has_base_head:
            # base_head + final_head(3,) → calibrated or residual
            # calibrated: final_head input dim = 3 (base+fine+coarse)
            final_combiner = "calibrated"
        else:
            final_combiner = "explicit"
        critic = DualHeadCritic(
            in_dim=6,
            final_combiner=final_combiner,
            setwise=setwise,
            path_corr_interaction=path_corr_interaction,
            head_hidden_dim=head_hidden_dim,
        )
    else:
        critic = FundamentalCritic(
            in_dim=6,
            setwise=setwise,
            path_corr_interaction=path_corr_interaction,
            head_hidden_dim=head_hidden_dim,
        )

    critic.load_state_dict(sd)
    critic = critic.to(device).eval()
    return critic, arch


def compute_mse(path, y_true):
    """逐点归一化 MSE（与原有 evaluate_strategies 一致）。"""
    return float(np.mean((path - y_true) ** 2))


def compute_mae(path, y_true):
    return float(np.mean(np.abs(path - y_true)))


def compute_direction_accuracy(path, y_true):
    """OHLC 日间方向准确率。"""
    path_close = path[:, 3]
    true_close = y_true[:, 3]
    if len(path_close) < 2:
        return 0.0
    pred_dir = np.diff(path_close) >= 0
    true_dir = np.diff(true_close) >= 0
    return float(np.mean(pred_dir == true_dir))


def evaluate_cache(args):
    # ---- 加载缓存 ----
    print(f"加载缓存: {args.cache}")
    with open(args.cache, "rb") as f:
        cache = pickle.load(f)

    windows = cache["windows"]
    candidates = cache["candidates"]
    metadata = cache["metadata"]
    print(f"  {len(windows)} 窗口, 每窗口 {metadata['sample_count']} 候选")

    # ---- 加载 critics ----
    device = args.device
    critics = {}

    if args.critic_ckpt:
        print(f"加载主 critic: {args.critic_ckpt}")
        critic, arch = load_critic(
            args.critic_ckpt, arch=args.critic_arch or "auto",
            device=device, setwise=False, path_corr_interaction=True,
        )
        critics["main"] = critic
        print(f"  架构: {arch}")

    if args.small_mlp_ckpt:
        print(f"加载 small_mlp: {args.small_mlp_ckpt}")
        small_mlp, _ = load_critic(
            args.small_mlp_ckpt, arch="fundamental",
            device=device, setwise=False, path_corr_interaction=False,
        )
        critics["small_mlp"] = small_mlp

    # ---- 构建策略 ----
    strategies = {
        "simple_mean": strategy_simple_mean,
        "median": strategy_median,
    }

    if "main" in critics:
        strategies["critic_weighted"] = make_critic_strategy(
            critics["main"], CriticReranker(mode="weighted", temperature=args.temperature))
        strategies["critic_top1"] = make_critic_strategy(
            critics["main"], CriticReranker(mode="top1"))

    if "small_mlp" in critics:
        strategies["small_mlp_weighted"] = make_critic_strategy(
            critics["small_mlp"], CriticReranker(mode="weighted", temperature=args.temperature))
        strategies["small_mlp_top1"] = make_critic_strategy(
            critics["small_mlp"], CriticReranker(mode="top1"))

    # prob_top1 需要每窗口重新构建（因为 log_probs 是 per-window 的）
    # 但缓存中没有 log_probs！用概率最高的候选代替——这里用简单的 argmax over log_prob
    # 如果缓存没有 log_probs，prob_top1 退化为可用策略

    # ---- 评估 ----
    results = {name: {"mse": [], "mae": [], "dir_acc": []} for name in strategies}

    # 计算 oracle 天花板
    oracle_mse = []
    oracle_mae = []

    for wi, (win, cands) in enumerate(zip(windows, candidates)):
        code, x_df, x_ts, y_ts, y_true = win
        N = len(cands)

        # 从 x_df 提取 hist features（标准化 K 线数据）
        hist_arr = x_df[["open", "high", "low", "close", "volume", "amount"]].values.astype(np.float32)
        # 用 hist 的 mean/std 归一化 y_true 到候选空间
        hist_mean = hist_arr.mean(axis=0, keepdims=True)
        hist_std = hist_arr.std(axis=0, keepdims=True) + 1e-5
        y_true_norm = (y_true - hist_mean) / hist_std

        # oracle: 选距离真值最近的候选（在归一化空间比较）
        mse_per_cand = [compute_mse(cands[i], y_true_norm) for i in range(N)]
        best_idx = int(np.argmin(mse_per_cand))
        oracle_mse.append(mse_per_cand[best_idx])
        oracle_mae.append(compute_mae(cands[best_idx], y_true_norm))

        # 逐策略评估
        for strat_name, strat_fn in strategies.items():
            if strat_name == "prob_top1":
                # prob_top1 需要 log_probs，缓存中没有则跳过
                continue
            try:
                selected = strat_fn(cands, hist=hist_arr)
                mse = compute_mse(selected, y_true_norm)
                mae = compute_mae(selected, y_true_norm)
                da = compute_direction_accuracy(selected, y_true_norm)
                results[strat_name]["mse"].append(mse)
                results[strat_name]["mae"].append(mae)
                results[strat_name]["dir_acc"].append(da)
            except Exception as e:
                print(f"  [{strat_name}] 窗口 {wi} 出错: {e}")

        if (wi + 1) % 100 == 0:
            print(f"  进度: {wi + 1}/{len(windows)}")

    # ---- 输出 ----
    print(f"\n{'='*70}")
    print(f"E1 评估: {args.period} ({len(windows)} 窗口)")
    print(f"{'='*70}")
    print(f"{'策略':<25} {'MSE':>10} {'MAE':>10} {'DirAcc':>8} {'vsOracle':>10}")
    print(f"{'-'*70}")

    oracle_mse_avg = np.mean(oracle_mse)
    oracle_mae_avg = np.mean(oracle_mae)
    print(f"{'oracle (天花板)':<25} {oracle_mse_avg:>10.6f} {oracle_mae_avg:>10.6f} {'-':>8} {'-':>10}")

    report = {}
    for name in strategies:
        if name == "prob_top1":
            continue
        r = results[name]
        if len(r["mse"]) == 0:
            continue
        mse_avg = np.mean(r["mse"])
        mae_avg = np.mean(r["mae"])
        da_avg = np.mean(r["dir_acc"])
        mse_ratio = mse_avg / oracle_mse_avg if oracle_mse_avg > 0 else float("inf")
        print(f"{name:<25} {mse_avg:>10.6f} {mae_avg:>10.6f} {da_avg:>8.4f} {mse_ratio:>10.4f}")
        report[name] = {
            "mse": mse_avg, "mae": mae_avg, "dir_acc": da_avg,
            "mse_vs_oracle": mse_ratio,
        }

    return report


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, help="候选缓存 .pkl")
    p.add_argument("--critic-ckpt", default=None, help="主 critic checkpoint")
    p.add_argument("--small-mlp-ckpt", default=None, help="small_mlp checkpoint")
    p.add_argument("--critic-arch", default=None, choices=["fundamental", "dual_head", "auto"],
                   help="critic 架构（默认自动检测）")
    p.add_argument("--temperature", type=float, default=1.0, help="softmax 温度")
    p.add_argument("--period", default="?", help="时段标签")
    p.add_argument("--device", default="cpu", help="cuda / cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_cache(args)
