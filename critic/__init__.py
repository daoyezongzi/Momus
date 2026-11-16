"""Critic 矫正模块：对 Kronos 生成的 N 条候选 K 线路径重排序。

方案 A（路径级重排序 / Best-of-N Critic Reranking）。详见 docs/critic_kline_design.md。
"""
from .reranker import CriticReranker
from .model import RandomCritic, FundamentalCritic, DualHeadCritic
from .predictor import CriticKronosPredictor
from .data_provider import CorrelatedDataProvider, make_tushare_fetch
from .config import get_correlated_symbols
from .train_critic import (
    compute_distances,
    pairwise_ranking_loss,
    train_critic,
)
from .schemas import CriticSample, FeatureVariant, ResonanceFeatures, SampleAuditMeta
from .evaluation import (
    mse, mae, direction_accuracy,
    strategy_simple_mean, strategy_median, make_critic_strategy,
    make_prob_top1_strategy,
    evaluate_strategies, aggregate_results, format_comparison,
)

__all__ = [
    "CriticReranker",
    "RandomCritic",
    "FundamentalCritic",
    "DualHeadCritic",
    "CriticKronosPredictor",
    "CorrelatedDataProvider",
    "make_tushare_fetch",
    "get_correlated_symbols",
    "CriticSample",
    "compute_distances",
    "pairwise_ranking_loss",
    "train_critic",
    "mse",
    "mae",
    "direction_accuracy",
    "strategy_simple_mean",
    "strategy_median",
    "make_critic_strategy",
    "make_prob_top1_strategy",
    "evaluate_strategies",
    "aggregate_results",
    "format_comparison",
    "FeatureVariant",
    "ResonanceFeatures",
    "SampleAuditMeta",
]
