"""产业链分层分析：检查 Critic 重排序在各产业链上的表现差异。

目标：
1. 检查是否某些产业链过度拟合（训练表现好但测试表现差）
2. 检查是否某些产业链表现异常（显著好于或差于平均水平）
3. 分析训练集和测试集在各产业链上的一致性

产业链分组（基于 critic/config.py）：
- 新能源车/锂电池：300750, 002594, 002460, 002466, 300014, 601127
- 白酒：600519, 000858, 000568, 600809, 002304
- 半导体：688981, 002371, 688012, 603986, 002049
- 光伏：601012, 002129, 300274, 688599, 600438
- 医药/CXO：600276, 300760, 603259, 300347, 002821
- 银行/金融：600036, 601166, 600000, 601318, 601601

用法：
  python examples/run_industry_chain_analysis.py --train-cache critic_samples_full.pkl --eval-cache critic_samples_2024.pkl
"""
import argparse
import sys
from collections import defaultdict

import numpy as np

sys.path.append("../")
sys.path.append(".")

from critic import FundamentalCritic, CriticReranker
from critic.train_critic import load_samples


# 产业链分组（从 config.py 提取）
INDUSTRY_CHAINS = {
    "新能源车/锂电池": ["300750", "002594", "002460", "002466", "300014", "601127"],
    "白酒": ["600519", "000858", "000568", "600809", "002304"],
    "半导体": ["688981", "002371", "688012", "603986", "002049"],
    "光伏": ["601012", "002129", "300274", "688599", "600438"],
    "医药/CXO": ["600276", "300760", "603259", "300347", "002821"],
    "银行/金融": ["600036", "601166", "600000", "601318", "601601"],
}


def extract_stock_code(sample):
    """从样本中提取股票代码"""
    # CriticSample 是 dataclass，直接访问 code 属性
    if hasattr(sample, 'code'):
        return sample.code
    # 兼容旧版本样本（字典格式）
    if isinstance(sample, dict) and 'code' in sample:
        return sample['code']
    return None


def group_samples_by_industry(samples):
    """按产业链分组样本"""
    industry_samples = defaultdict(list)
    unmatched = []

    # 构建反向索引：股票代码 -> 产业链
    code_to_industry = {}
    for industry, codes in INDUSTRY_CHAINS.items():
        for code in codes:
            code_to_industry[code] = industry

    for sample in samples:
        code = extract_stock_code(sample)
        if code and code in code_to_industry:
            industry = code_to_industry[code]
            industry_samples[industry].append(sample)
        else:
            unmatched.append(sample)

    return industry_samples, unmatched


def compute_expected_distance(samples, critic, reranker, use_corr=False):
    """计算期望距离（与 run_corr_ablation.py 中的逻辑一致）"""
    distances = []
    for sample in samples:
        paths = sample["paths"]
        true_path = sample["y_true"]
        hist = sample["hist"]
        corr_feat = sample.get("corr_feat") if use_corr else None

        # Critic 打分 + 重排序
        scores = critic.score(paths, hist=hist, corr_feat=corr_feat)
        final = reranker.combine(paths, scores)

        # 计算距离（MSE on OHLC）
        dist = np.mean((final[:, :4] - true_path[:, :4]) ** 2)
        distances.append(dist)

    return float(np.mean(distances))


def baseline_distance(samples):
    """计算 baseline（简单平均）的期望距离"""
    distances = []
    for sample in samples:
        paths = sample["paths"]
        true_path = sample["y_true"]
        final = np.mean(paths, axis=0)
        dist = np.mean((final[:, :4] - true_path[:, :4]) ** 2)
        distances.append(dist)
    return float(np.mean(distances))


def run(args):
    import torch

    print("=" * 80)
    print("产业链分层分析 — 检查过拟合与表现差异")
    print("=" * 80)

    # 加载数据
    train_samples = load_samples(args.train_cache)
    eval_samples = load_samples(args.eval_cache)

    print(f"\n总样本数：训练集 {len(train_samples)} | 测试集 {len(eval_samples)}")

    # 按产业链分组
    train_by_industry, train_unmatched = group_samples_by_industry(train_samples)
    eval_by_industry, eval_unmatched = group_samples_by_industry(eval_samples)

    print(f"\n训练集分布：")
    for industry, samples in sorted(train_by_industry.items()):
        print(f"  {industry:<18}: {len(samples):>3} 样本")
    if train_unmatched:
        print(f"  {'未匹配':<18}: {len(train_unmatched):>3} 样本")

    print(f"\n测试集分布：")
    for industry, samples in sorted(eval_by_industry.items()):
        print(f"  {industry:<18}: {len(samples):>3} 样本")
    if eval_unmatched:
        print(f"  {'未匹配':<18}: {len(eval_unmatched):>3} 样本")

    # 训练 Critic（核心方法：纯 path+hist）
    print(f"\n训练 Critic（纯 path+hist 重排序，{args.epochs} epochs）...")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    critic = FundamentalCritic(in_dim=6, path_corr_interaction=False)

    # 简化训练（复用 train_critic 逻辑）
    from critic.train_critic import train_critic
    train_critic(critic, train_samples, epochs=args.epochs, lr=args.lr,
                 device=args.device, verbose=True, use_corr=False)

    # 创建重排序器
    reranker_weighted = CriticReranker(mode="weighted", temperature=1.0)
    reranker_top1 = CriticReranker(mode="top1")

    print("\n" + "=" * 80)
    print("分产业链评估结果")
    print("=" * 80)

    # 总体 baseline
    train_base_all = baseline_distance(train_samples)
    eval_base_all = baseline_distance(eval_samples)

    print(f"\n【总体 Baseline (simple_mean)】")
    print(f"  训练集: {train_base_all:.4f}")
    print(f"  测试集: {eval_base_all:.4f}")

    # 分产业链评估
    results = []

    for industry in sorted(INDUSTRY_CHAINS.keys()):
        if industry not in train_by_industry or industry not in eval_by_industry:
            continue

        train_samps = train_by_industry[industry]
        eval_samps = eval_by_industry[industry]

        # Baseline
        train_base = baseline_distance(train_samps)
        eval_base = baseline_distance(eval_samps)

        # Critic weighted
        train_weighted = compute_expected_distance(train_samps, critic, reranker_weighted, use_corr=False)
        eval_weighted = compute_expected_distance(eval_samps, critic, reranker_weighted, use_corr=False)

        # Critic top1
        train_top1 = compute_expected_distance(train_samps, critic, reranker_top1, use_corr=False)
        eval_top1 = compute_expected_distance(eval_samps, critic, reranker_top1, use_corr=False)

        # 计算改善率
        train_imp_w = (train_base - train_weighted) / train_base * 100 if train_base > 0 else 0
        eval_imp_w = (eval_base - eval_weighted) / eval_base * 100 if eval_base > 0 else 0

        train_imp_t = (train_base - train_top1) / train_base * 100 if train_base > 0 else 0
        eval_imp_t = (eval_base - eval_top1) / eval_base * 100 if eval_base > 0 else 0

        # 过拟合指标：训练改善 - 测试改善
        overfit_w = train_imp_w - eval_imp_w
        overfit_t = train_imp_t - eval_imp_t

        results.append({
            "industry": industry,
            "train_n": len(train_samps),
            "eval_n": len(eval_samps),
            "train_base": train_base,
            "eval_base": eval_base,
            "train_imp_w": train_imp_w,
            "eval_imp_w": eval_imp_w,
            "train_imp_t": train_imp_t,
            "eval_imp_t": eval_imp_t,
            "overfit_w": overfit_w,
            "overfit_t": overfit_t,
        })

    # 打印表格
    print(f"\n{'产业链':<18}{'训练样本':>8}{'测试样本':>8}{'训练改善(W)':>12}{'测试改善(W)':>12}{'过拟合(W)':>12}")
    print("-" * 80)
    for r in results:
        print(f"{r['industry']:<18}{r['train_n']:>8}{r['eval_n']:>8}"
              f"{r['train_imp_w']:>11.1f}%{r['eval_imp_w']:>11.1f}%{r['overfit_w']:>11.1f}%")

    print(f"\n{'产业链':<18}{'训练样本':>8}{'测试样本':>8}{'训练改善(T)':>12}{'测试改善(T)':>12}{'过拟合(T)':>12}")
    print("-" * 80)
    for r in results:
        print(f"{r['industry']:<18}{r['train_n']:>8}{r['eval_n']:>8}"
              f"{r['train_imp_t']:>11.1f}%{r['eval_imp_t']:>11.1f}%{r['overfit_t']:>11.1f}%")

    # 统计分析
    print("\n" + "=" * 80)
    print("诊断与结论")
    print("=" * 80)

    eval_imps_w = [r['eval_imp_w'] for r in results]
    eval_imps_t = [r['eval_imp_t'] for r in results]
    overfits_w = [r['overfit_w'] for r in results]
    overfits_t = [r['overfit_t'] for r in results]

    print(f"\n测试集改善（weighted）：")
    print(f"  均值 = {np.mean(eval_imps_w):.1f}%  标准差 = {np.std(eval_imps_w):.1f}%")
    print(f"  最小 = {np.min(eval_imps_w):.1f}%  最大 = {np.max(eval_imps_w):.1f}%")

    print(f"\n测试集改善（top1）：")
    print(f"  均值 = {np.mean(eval_imps_t):.1f}%  标准差 = {np.std(eval_imps_t):.1f}%")
    print(f"  最小 = {np.min(eval_imps_t):.1f}%  最大 = {np.max(eval_imps_t):.1f}%")

    print(f"\n过拟合指标（训练改善 - 测试改善）：")
    print(f"  weighted: 均值 = {np.mean(overfits_w):.1f}%  标准差 = {np.std(overfits_w):.1f}%")
    print(f"  top1:     均值 = {np.mean(overfits_t):.1f}%  标准差 = {np.std(overfits_t):.1f}%")

    # 标记异常
    print(f"\n异常检测（|改善差| > 5% 或 过拟合 > 10%）：")
    anomalies = []
    for r in results:
        issues = []
        if abs(r['eval_imp_w'] - np.mean(eval_imps_w)) > 5:
            issues.append(f"weighted测试改善偏离均值 {r['eval_imp_w'] - np.mean(eval_imps_w):+.1f}%")
        if abs(r['eval_imp_t'] - np.mean(eval_imps_t)) > 5:
            issues.append(f"top1测试改善偏离均值 {r['eval_imp_t'] - np.mean(eval_imps_t):+.1f}%")
        if r['overfit_w'] > 10:
            issues.append(f"weighted过拟合 {r['overfit_w']:.1f}%")
        if r['overfit_t'] > 10:
            issues.append(f"top1过拟合 {r['overfit_t']:.1f}%")

        if issues:
            anomalies.append((r['industry'], issues))

    if anomalies:
        for industry, issues in anomalies:
            print(f"  ⚠️  {industry}:")
            for issue in issues:
                print(f"      - {issue}")
    else:
        print("  ✅ 未检测到显著异常，各产业链表现一致")

    # 保存结果
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write("产业链分层分析结果\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"总体 Baseline (simple_mean):\n")
            f.write(f"  训练集: {train_base_all:.4f}\n")
            f.write(f"  测试集: {eval_base_all:.4f}\n\n")

            f.write(f"{'产业链':<18}{'训练样本':>8}{'测试样本':>8}{'训练改善(W)':>12}{'测试改善(W)':>12}{'过拟合(W)':>12}\n")
            f.write("-" * 80 + "\n")
            for r in results:
                f.write(f"{r['industry']:<18}{r['train_n']:>8}{r['eval_n']:>8}"
                       f"{r['train_imp_w']:>11.1f}%{r['eval_imp_w']:>11.1f}%{r['overfit_w']:>11.1f}%\n")

            f.write(f"\n{'产业链':<18}{'训练样本':>8}{'测试样本':>8}{'训练改善(T)':>12}{'测试改善(T)':>12}{'过拟合(T)':>12}\n")
            f.write("-" * 80 + "\n")
            for r in results:
                f.write(f"{r['industry']:<18}{r['train_n']:>8}{r['eval_n']:>8}"
                       f"{r['train_imp_t']:>11.1f}%{r['eval_imp_t']:>11.1f}%{r['overfit_t']:>11.1f}%\n")

            f.write("\n诊断与结论\n")
            f.write("=" * 80 + "\n")
            f.write(f"\n测试集改善（weighted）：均值 {np.mean(eval_imps_w):.1f}%，标准差 {np.std(eval_imps_w):.1f}%\n")
            f.write(f"测试集改善（top1）：均值 {np.mean(eval_imps_t):.1f}%，标准差 {np.std(eval_imps_t):.1f}%\n")
            f.write(f"\n过拟合指标：weighted 均值 {np.mean(overfits_w):.1f}%，top1 均值 {np.mean(overfits_t):.1f}%\n")

            if anomalies:
                f.write("\n异常检测：\n")
                for industry, issues in anomalies:
                    f.write(f"  {industry}:\n")
                    for issue in issues:
                        f.write(f"    - {issue}\n")
            else:
                f.write("\n未检测到显著异常，各产业链表现一致。\n")

        print(f"\n结果已保存至：{args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="产业链分层分析脚本")
    p.add_argument("--train-cache", type=str, default="critic_samples_full.pkl",
                   help="训练样本缓存文件")
    p.add_argument("--eval-cache", type=str, default="critic_samples_2024.pkl",
                   help="测试样本缓存文件")
    p.add_argument("--seed", type=int, default=0,
                   help="随机种子")
    p.add_argument("--epochs", type=int, default=30,
                   help="训练轮数")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="学习率")
    p.add_argument("--device", type=str, default="cuda",
                   help="设备 (cuda/cpu)")
    p.add_argument("--output", type=str, default="results_industry_chain_analysis.txt",
                   help="输出文件路径")

    args = p.parse_args()
    run(args)
