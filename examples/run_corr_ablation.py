"""E3 关联特征消融（设计文档 §8 E3）：验证"关联标的特征"这个核心创新点是否有效。

对比三者在同一批样本外候选上的重排序质量：
  - simple_mean   : baseline，等权平均（等价原 Kronos）
  - critic_no_corr: Critic 只用 path+hist（去掉关联特征）
  - critic_corr   : Critic 用 path+hist+关联特征（完整版）

关键：带关联 vs 不带关联，用**完全相同的训练样本、相同的候选路径**，
唯一变量是关联特征 —— 这才是干净的消融，能说明关联特征本身的增量。

评估指标：重排序后的**期望距离**（用 reranker 权重加权样本的 distances）。
distances 是每条候选到真实未来的 MSE，越小越好。不必重跑 Kronos 指标——
消融只关心"哪版 Critic 挑的候选更接近真实"，加权 distances 直接回答这点。

两阶段：
  1. 训练样本（2022~2023，缓存 critic_samples_full.pkl）训两版 Critic。
  2. 评估样本（2024 样本外，需先 build-only 生成）上对比。

用法：
  # 先生成 2024 样本外评估样本（复用训练管道的 build-only）
  python examples/train_critic_pipeline.py --codes-file data/references/codes.txt \
      --start 20240101 --end 20241231 --n-windows 5 --build-only \
      --samples-cache critic_samples_2024.pkl
  # 再跑消融
  python examples/run_corr_ablation.py \
      --train-cache critic_samples_full.pkl \
      --eval-cache critic_samples_2024.pkl --epochs 30
"""
import argparse
import sys

import numpy as np

sys.path.append("../")
sys.path.append(".")

from critic import FundamentalCritic, CriticReranker, train_critic
from critic.train_critic import load_samples


def weighted_distance(critic, sample, reranker, use_corr):
    """一个样本上：Critic 打分 -> reranker 权重 -> 加权其 distances。

    返回该样本重排序后的期望距离（越小越好）。use_corr=False 时屏蔽关联特征，
    与无关联版 Critic 的训练口径一致。
    """
    corr = sample.corr_feat if use_corr else None
    scores = critic.score(sample.paths, hist=sample.hist, corr_feat=corr)
    if reranker.mode == "top1":
        return float(sample.distances[int(np.argmax(scores))])
    # weighted：与 reranker 相同的数值稳定 softmax
    logits = scores / reranker.temperature
    logits = logits - np.max(logits)
    w = np.exp(logits)
    w = w / np.sum(w)
    return float(np.sum(w * sample.distances))


def eval_strategy(critic, samples, reranker, use_corr):
    """整个评估集上的平均期望距离。"""
    return float(np.mean([weighted_distance(critic, s, reranker, use_corr)
                          for s in samples]))


def mean_distance_baseline(samples):
    """simple_mean 的等价：等权所有候选的 distances 平均。

    注：这是对"简单平均预测"距离的一致近似——等权融合候选，等价于等权其距离期望
    （二次项差异极小，作为 baseline 参照足够）。
    """
    return float(np.mean([s.distances.mean() for s in samples]))


def run(args):
    train_samples = load_samples(args.train_cache)
    eval_samples = load_samples(args.eval_cache)
    print(f"训练样本: {len(train_samples)}（{args.train_cache}）")
    print(f"评估样本: {len(eval_samples)}（{args.eval_cache}，样本外）")

    rerank_w = CriticReranker(mode="weighted", temperature=args.temperature)
    rerank_t = CriticReranker(mode="top1")

    # ---- 训练两版 Critic：唯一变量是 use_corr ----
    np.random.seed(args.seed)
    import torch
    torch.manual_seed(args.seed)
    critic_corr = FundamentalCritic(in_dim=6, path_corr_interaction=True)
    print(f"\n[1/2] 训练完整版 Critic（含关联特征+候选级交互）{args.epochs} epochs...")
    h1 = train_critic(critic_corr, train_samples, epochs=args.epochs, lr=args.lr,
                      device=args.device, verbose=True, use_corr=True)
    print(f"      loss {h1[0]:.4f} -> {h1[-1]:.4f}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # 无关联版关掉交互项 -> 纯 path+hist 干净基线（否则 corr_missing 的交互项会污染对照）
    critic_nocorr = FundamentalCritic(in_dim=6, path_corr_interaction=False)
    print(f"[2/2] 训练消融版 Critic（无关联特征，纯 path+hist）{args.epochs} epochs...")
    h2 = train_critic(critic_nocorr, train_samples, epochs=args.epochs, lr=args.lr,
                      device=args.device, verbose=True, use_corr=False)
    print(f"      loss {h2[0]:.4f} -> {h2[-1]:.4f}")

    # ---- 在 2024 样本外评估集上对比 ----
    base = mean_distance_baseline(eval_samples)
    rows = [
        ("simple_mean", base, base),
        ("critic_no_corr(weighted)",
         eval_strategy(critic_nocorr, eval_samples, rerank_w, use_corr=False), base),
        ("critic_corr(weighted)",
         eval_strategy(critic_corr, eval_samples, rerank_w, use_corr=True), base),
        ("critic_no_corr(top1)",
         eval_strategy(critic_nocorr, eval_samples, rerank_t, use_corr=False), base),
        ("critic_corr(top1)",
         eval_strategy(critic_corr, eval_samples, rerank_t, use_corr=True), base),
    ]

    print("\n" + "=" * 60)
    print(f"E3 关联特征消融（{len(eval_samples)} 样本外样本）")
    print("=" * 60)
    print(f"{'策略':<28}{'期望距离':>12}{'改善':>12}")
    print("-" * 60)
    for name, dist, b in rows:
        imp = "" if name == "simple_mean" else f"{(b - dist) / b * 100:+.1f}%"
        print(f"{name:<28}{dist:>12.4f}{imp:>12}")
    print("\n距离=每条候选到真实未来的MSE按重排序权重加权，越小越好。")
    print("看点：critic_corr 是否明显优于 critic_no_corr —— 差值即关联特征的增量。")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-cache", type=str, default="critic_samples_full.pkl")
    p.add_argument("--eval-cache", type=str, default="critic_samples_2024.pkl")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    run(p.parse_args())
