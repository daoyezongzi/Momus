"""Critic Reranking 评估

量化评估 Critic 对候选集的重排序效果

评估维度：
1. 排序质量：Spearman/Kendall 相关性
2. Top-K 精度：重排后 Top-K 的收益率分布
3. 收益提升：相对于 baseline 的收益改善
4. 产业链共振验证：高分样本是否确实具有产业链特征
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr, kendalltau
from collections import defaultdict
import pandas as pd
import sys

# 导入训练脚本中的模型和数据集
sys.path.append(str(Path(__file__).parent))
from train_critic import CriticNetwork, CriticDataset, KlineWindow


class CriticEvaluator:
    """Critic 评估器"""

    def __init__(self, model_path, device='cpu'):
        self.device = device
        self.model = CriticNetwork().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def score_samples(self, samples):
        """对样本打分"""
        dataset = CriticDataset(samples)
        scores = []

        with torch.no_grad():
            for i in range(len(dataset)):
                main_window, related_windows, _ = dataset[i]
                main_window = main_window.unsqueeze(0).to(self.device)
                related_windows = related_windows.unsqueeze(0).to(self.device)

                score = self.model(main_window, related_windows)
                scores.append(score.item())

        return np.array(scores)


def evaluate_ranking_quality(scores, returns):
    """评估排序质量

    指标：
    - Spearman 相关性：预测分数与真实收益的排序一致性
    - Kendall Tau：考虑并列情况的排序相关性
    - NDCG@K：归一化折损累积增益
    """
    # Spearman 相关性
    spearman_corr, spearman_p = spearmanr(scores, returns)

    # Kendall Tau
    kendall_corr, kendall_p = kendalltau(scores, returns)

    # NDCG@K
    def dcg_at_k(r, k):
        r = np.asfarray(r)[:k]
        if r.size:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        return 0.

    def ndcg_at_k(scores, returns, k):
        # 按分数排序
        ranked_indices = np.argsort(scores)[::-1]
        ranked_returns = returns[ranked_indices]

        # 理想排序（按真实收益排序）
        ideal_returns = np.sort(returns)[::-1]

        dcg = dcg_at_k(ranked_returns, k)
        idcg = dcg_at_k(ideal_returns, k)

        if idcg == 0:
            return 0.
        return dcg / idcg

    ndcg_5 = ndcg_at_k(scores, returns, 5)
    ndcg_10 = ndcg_at_k(scores, returns, 10)
    ndcg_20 = ndcg_at_k(scores, returns, 20)

    return {
        'spearman_corr': spearman_corr,
        'spearman_p': spearman_p,
        'kendall_corr': kendall_corr,
        'kendall_p': kendall_p,
        'ndcg@5': ndcg_5,
        'ndcg@10': ndcg_10,
        'ndcg@20': ndcg_20,
    }


def evaluate_topk_precision(scores, returns, k_list=[5, 10, 20, 50]):
    """评估 Top-K 精度

    指标：
    - Top-K 平均收益率
    - Top-K 胜率（正收益比例）
    - Top-K 相对提升（vs 随机选择）
    """
    results = {}

    for k in k_list:
        # 按分数排序取 Top-K
        ranked_indices = np.argsort(scores)[::-1][:k]
        topk_returns = returns[ranked_indices]

        # 随机选择 K 个（baseline）
        random_returns = returns

        results[f'top{k}'] = {
            'mean_return': float(np.mean(topk_returns)),
            'median_return': float(np.median(topk_returns)),
            'win_rate': float(np.mean(topk_returns > 0)),
            'std_return': float(np.std(topk_returns)),
            'max_return': float(np.max(topk_returns)),
            'min_return': float(np.min(topk_returns)),
            # 相对提升
            'vs_random_mean': float(np.mean(topk_returns) - np.mean(random_returns)),
            'vs_random_median': float(np.median(topk_returns) - np.median(random_returns)),
        }

    return results


def evaluate_gain_improvement(scores, returns, baseline_scores=None):
    """评估收益改善

    对比：
    - Critic reranking vs Random
    - Critic reranking vs Baseline (如果有)
    """
    results = {}

    # Critic Top-K
    for k in [5, 10, 20]:
        critic_topk = returns[np.argsort(scores)[::-1][:k]]
        random_topk = returns[:k]  # 假设 baseline 是随机排序

        results[f'gain_top{k}'] = {
            'critic_mean': float(np.mean(critic_topk)),
            'random_mean': float(np.mean(random_topk)),
            'absolute_gain': float(np.mean(critic_topk) - np.mean(random_topk)),
            'relative_gain': float((np.mean(critic_topk) - np.mean(random_topk)) / (abs(np.mean(random_topk)) + 1e-8)),
        }

        # 如果有 baseline 分数
        if baseline_scores is not None:
            baseline_topk = returns[np.argsort(baseline_scores)[::-1][:k]]
            results[f'gain_top{k}']['baseline_mean'] = float(np.mean(baseline_topk))
            results[f'gain_top{k}']['vs_baseline_gain'] = float(np.mean(critic_topk) - np.mean(baseline_topk))

    return results


def analyze_resonance_features(samples, scores, returns, top_k=20):
    """分析产业链共振特征

    验证：高分样本是否确实具有更强的产业链共振
    """
    # 按分数排序
    ranked_indices = np.argsort(scores)[::-1]

    # Top-K 高分样本
    topk_indices = ranked_indices[:top_k]
    topk_samples = [samples[i] for i in topk_indices]

    # 统计产业链特征
    topk_related_counts = [len(s['related_windows']) for s in topk_samples]
    topk_returns = returns[topk_indices]

    # Bottom-K 低分样本（对比）
    bottomk_indices = ranked_indices[-top_k:]
    bottomk_samples = [samples[i] for i in bottomk_indices]
    bottomk_related_counts = [len(s['related_windows']) for s in bottomk_samples]
    bottomk_returns = returns[bottomk_indices]

    # 全局统计
    all_related_counts = [len(s['related_windows']) for s in samples]

    return {
        'topk_avg_related': float(np.mean(topk_related_counts)),
        'topk_avg_return': float(np.mean(topk_returns)),
        'bottomk_avg_related': float(np.mean(bottomk_related_counts)),
        'bottomk_avg_return': float(np.mean(bottomk_returns)),
        'global_avg_related': float(np.mean(all_related_counts)),
        'global_avg_return': float(np.mean(returns)),
        # 验证：高分样本是否有更多关联标的
        'related_count_lift': float(np.mean(topk_related_counts) - np.mean(bottomk_related_counts)),
        # 验证：高分样本是否有更高收益
        'return_lift': float(np.mean(topk_returns) - np.mean(bottomk_returns)),
    }


def generate_evaluation_report(results, output_path):
    """生成评估报告"""
    report = []
    report.append("=" * 80)
    report.append("Critic Reranking 评估报告")
    report.append("=" * 80)

    # 1. 排序质量
    report.append("\n【1. 排序质量】")
    report.append(f"  Spearman 相关性: {results['ranking_quality']['spearman_corr']:.4f} (p={results['ranking_quality']['spearman_p']:.4e})")
    report.append(f"  Kendall Tau:     {results['ranking_quality']['kendall_corr']:.4f} (p={results['ranking_quality']['kendall_p']:.4e})")
    report.append(f"  NDCG@5:          {results['ranking_quality']['ndcg@5']:.4f}")
    report.append(f"  NDCG@10:         {results['ranking_quality']['ndcg@10']:.4f}")
    report.append(f"  NDCG@20:         {results['ranking_quality']['ndcg@20']:.4f}")

    # 2. Top-K 精度
    report.append("\n【2. Top-K 精度】")
    for k in [5, 10, 20]:
        metrics = results['topk_precision'][f'top{k}']
        report.append(f"\n  Top-{k}:")
        report.append(f"    平均收益率:   {metrics['mean_return']*100:.2f}%")
        report.append(f"    中位数收益率: {metrics['median_return']*100:.2f}%")
        report.append(f"    胜率:         {metrics['win_rate']*100:.1f}%")
        report.append(f"    标准差:       {metrics['std_return']*100:.2f}%")
        report.append(f"    最大收益:     {metrics['max_return']*100:.2f}%")
        report.append(f"    最小收益:     {metrics['min_return']*100:.2f}%")
        report.append(f"    vs Random (均值): +{metrics['vs_random_mean']*100:.2f}%")

    # 3. 收益改善
    report.append("\n【3. 收益改善（Critic vs Random）】")
    for k in [5, 10, 20]:
        gain = results['gain_improvement'][f'gain_top{k}']
        report.append(f"\n  Top-{k}:")
        report.append(f"    Critic 平均:  {gain['critic_mean']*100:.2f}%")
        report.append(f"    Random 平均:  {gain['random_mean']*100:.2f}%")
        report.append(f"    绝对提升:     +{gain['absolute_gain']*100:.2f}%")
        report.append(f"    相对提升:     +{gain['relative_gain']*100:.1f}%")

    # 4. 产业链共振验证
    report.append("\n【4. 产业链共振特征验证】")
    resonance = results['resonance_analysis']
    report.append(f"  Top-20 平均关联标的数: {resonance['topk_avg_related']:.1f}")
    report.append(f"  Bottom-20 平均关联标的数: {resonance['bottomk_avg_related']:.1f}")
    report.append(f"  关联数差异: +{resonance['related_count_lift']:.1f}")
    report.append(f"\n  Top-20 平均收益: {resonance['topk_avg_return']*100:.2f}%")
    report.append(f"  Bottom-20 平均收益: {resonance['bottomk_avg_return']*100:.2f}%")
    report.append(f"  收益差异: +{resonance['return_lift']*100:.2f}%")

    # 保存报告
    report_text = "\n".join(report)
    print(report_text)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"\n报告已保存: {output_path}")

    return report_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Critic 模型路径')
    parser.add_argument('--test-data', type=str, required=True, help='测试样本文件')
    parser.add_argument('--output', type=str, default=None, help='评估报告输出路径')
    parser.add_argument('--baseline-scores', type=str, default=None, help='Baseline 分数文件（可选）')
    args = parser.parse_args()

    print("=" * 60)
    print("Critic Reranking 评估")
    print("=" * 60)

    # 加载模型
    print(f"\n加载模型: {args.model}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = CriticEvaluator(args.model, device)

    # 加载测试数据
    print(f"加载测试数据: {args.test_data}")
    with open(args.test_data, 'rb') as f:
        test_samples = pickle.load(f)
    print(f"  测试样本数: {len(test_samples)}")

    # 提取真实收益
    true_returns = np.array([s['future_return'] for s in test_samples])

    # Critic 打分
    print("\nCritic 打分中...")
    critic_scores = evaluator.score_samples(test_samples)
    print(f"  分数范围: [{np.min(critic_scores):.4f}, {np.max(critic_scores):.4f}]")

    # 加载 baseline 分数（如果有）
    baseline_scores = None
    if args.baseline_scores:
        with open(args.baseline_scores, 'rb') as f:
            baseline_scores = pickle.load(f)

    # 评估
    print("\n开始评估...")
    results = {}

    # 1. 排序质量
    results['ranking_quality'] = evaluate_ranking_quality(critic_scores, true_returns)

    # 2. Top-K 精度
    results['topk_precision'] = evaluate_topk_precision(critic_scores, true_returns)

    # 3. 收益改善
    results['gain_improvement'] = evaluate_gain_improvement(critic_scores, true_returns, baseline_scores)

    # 4. 产业链共振验证
    results['resonance_analysis'] = analyze_resonance_features(test_samples, critic_scores, true_returns)

    # 生成报告
    print("\n" + "=" * 60)
    generate_evaluation_report(results, args.output)

    # 保存完整结果
    if args.output:
        results_path = Path(args.output).with_suffix('.pkl')
        with open(results_path, 'wb') as f:
            pickle.dump({
                'results': results,
                'critic_scores': critic_scores,
                'true_returns': true_returns,
            }, f)
        print(f"完整结果已保存: {results_path}")


if __name__ == '__main__':
    main()
