"""稳健性验证（设计文档 §8 E4）：多随机种子下重排序改善的稳定性。

方向已定（选项A）：核心方法是**纯 path+hist 重排序**（E3 证明 corr 特征无泛化增益
甚至过拟合，故核心 Critic 不用 corr）。本脚本对多个随机种子各训练一个纯 path+hist
Critic，在同一批 2024 样本外样本上评估重排序改善，输出均值±标准差。

回答审稿人的核心质疑："+20~24% 会不会只是运气？" —— 若多种子下改善稳定为正、
标准差小、最坏种子也明显优于 baseline，则结论稳健。

随机性来源：种子影响 Critic 权重初始化 + 训练 shuffle 顺序（候选路径已在缓存中固定，
不含 Kronos 采样随机性——那需重跑 Kronos，另做）。

用法：
  python examples/run_robustness.py --seeds 0,1,2,3,4 --epochs 30 --device cuda
"""
import argparse
import sys

import numpy as np

sys.path.append("../")
sys.path.append(".")

from critic import FundamentalCritic, CriticReranker, train_critic
from critic.train_critic import load_samples
from examples.run_corr_ablation import eval_strategy, mean_distance_baseline


def run(args):
    import torch

    train_samples = load_samples(args.train_cache)
    eval_samples = load_samples(args.eval_cache)
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"训练样本: {len(train_samples)} | 评估样本: {len(eval_samples)}（样本外）")
    print(f"随机种子: {seeds} | 核心方法: 纯 path+hist 重排序\n")

    base = mean_distance_baseline(eval_samples)
    rerank_w = CriticReranker(mode="weighted", temperature=args.temperature)
    rerank_t = CriticReranker(mode="top1")

    w_imps, t_imps = [], []
    for sd in seeds:
        np.random.seed(sd)
        torch.manual_seed(sd)
        # 纯 path+hist：关掉交互项、训练不喂 corr（方向A的核心 Critic）
        critic = FundamentalCritic(in_dim=6, path_corr_interaction=False)
        train_critic(critic, train_samples, epochs=args.epochs, lr=args.lr,
                     device=args.device, verbose=False, use_corr=False)
        dw = eval_strategy(critic, eval_samples, rerank_w, use_corr=False)
        dt = eval_strategy(critic, eval_samples, rerank_t, use_corr=False)
        iw = (base - dw) / base * 100
        it = (base - dt) / base * 100
        w_imps.append(iw)
        t_imps.append(it)
        print(f"[seed {sd}] weighted {dw:.4f} ({iw:+.1f}%) | top1 {dt:.4f} ({it:+.1f}%)")

    w = np.array(w_imps)
    t = np.array(t_imps)
    print("\n" + "=" * 60)
    print(f"稳健性汇总（{len(seeds)} 个随机种子，2024 样本外）")
    print("=" * 60)
    print(f"baseline (simple_mean) 期望距离: {base:.4f}")
    print(f"{'策略':<16}{'改善均值':>10}{'标准差':>10}{'最差':>10}{'最好':>10}")
    print("-" * 60)
    print(f"{'weighted':<16}{w.mean():>9.1f}%{w.std():>9.2f}%{w.min():>9.1f}%{w.max():>9.1f}%")
    print(f"{'top1':<16}{t.mean():>9.1f}%{t.std():>9.2f}%{t.min():>9.1f}%{t.max():>9.1f}%")
    print(f"\n结论：{len(seeds)} 个种子全部为正改善"
          if (w.min() > 0 and t.min() > 0) else "\n注意：存在非正改善的种子")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", type=str, default="critic_samples_full.pkl")
    p.add_argument("--eval-cache", type=str, default="critic_samples_2024.pkl")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cpu")
    run(p.parse_args())
