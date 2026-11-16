"""Candidate reranking diagnostics for critic/oracle gap analysis."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def _as_finite_pair(scores, distances):
    score_arr = np.asarray(scores, dtype=float)
    dist_arr = np.asarray(distances, dtype=float)
    if score_arr.shape != dist_arr.shape or score_arr.ndim != 1 or len(score_arr) == 0:
        return None, None
    mask = np.isfinite(score_arr) & np.isfinite(dist_arr)
    if mask.sum() < 2:
        return None, None
    return score_arr[mask], dist_arr[mask]


def _rankdata(values):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def _spearman(x, y):
    rx = _rankdata(x)
    ry = _rankdata(y)
    if np.allclose(rx, rx[0]) or np.allclose(ry, ry[0]):
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _softmax_entropy(scores, temperature):
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = np.asarray(scores, dtype=float) / temperature
    scaled = scaled - np.max(scaled)
    weights = np.exp(scaled)
    weights = weights / weights.sum()
    return float(-(weights * np.log(weights + 1e-12)).sum())


def _ndcg_for_order(order, distances):
    relevance = -np.asarray(distances, dtype=float)
    gains = relevance[order] - relevance.min()
    discounts = 1.0 / np.log2(np.arange(2, len(order) + 2))
    dcg = float((gains * discounts).sum())
    ideal = np.argsort(distances)
    ideal_gains = relevance[ideal] - relevance.min()
    ideal_dcg = float((ideal_gains * discounts).sum())
    if ideal_dcg <= 0:
        return float("nan")
    return dcg / ideal_dcg


def _nanmean(values):
    finite = [v for v in values if np.isfinite(v)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def score_samples_for_diagnostics(critic, samples, use_corr=True):
    """Score CriticSample rows for fixed-candidate oracle-gap diagnostics."""
    score_rows = []
    distance_rows = []
    for sample in samples:
        corr_feat = sample.corr_feat if use_corr else None
        scores = critic.score(sample.paths, sample.hist, corr_feat=corr_feat)
        score_rows.append(np.asarray(scores, dtype=np.float32))
        distance_rows.append(np.asarray(sample.distances, dtype=np.float32))
    return score_rows, distance_rows


def evaluate_candidate_reranking(
    score_rows: Iterable[np.ndarray],
    distance_rows: Iterable[np.ndarray],
    top_k: int = 3,
    temperature: float = 1.0,
) -> dict[str, float]:
    """Summarize how closely critic scores recover fixed-candidate oracle order.

    Lower distance means oracle-better; higher critic score means model-better.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    top1_hits = []
    topk_hits = []
    oracle_ranks = []
    reciprocal_ranks = []
    ndcgs = []
    spearmans = []
    entropies = []

    for scores, distances in zip(score_rows, distance_rows):
        scores, distances = _as_finite_pair(scores, distances)
        if scores is None:
            continue

        score_order = np.argsort(-scores, kind="mergesort")
        oracle_index = int(np.argmin(distances))
        rank_positions = np.where(score_order == oracle_index)[0]
        if len(rank_positions) == 0:
            continue
        oracle_rank = int(rank_positions[0]) + 1
        capped_k = min(top_k, len(score_order))

        top1_hits.append(float(oracle_rank == 1))
        topk_hits.append(float(oracle_rank <= capped_k))
        oracle_ranks.append(float(oracle_rank))
        reciprocal_ranks.append(1.0 / oracle_rank)
        ndcgs.append(_ndcg_for_order(score_order, distances))
        spearmans.append(_spearman(scores, -distances))
        entropies.append(_softmax_entropy(scores, temperature))

    sample_count = len(oracle_ranks)
    if sample_count == 0:
        nan = float("nan")
        return {
            "sample_count": 0,
            "top1_oracle_hit_rate": nan,
            f"top{top_k}_oracle_hit_rate": nan,
            "mean_oracle_rank": nan,
            "mrr": nan,
            "ndcg": nan,
            "mean_spearman": nan,
            "mean_score_entropy": nan,
        }

    return {
        "sample_count": sample_count,
        "top1_oracle_hit_rate": float(np.mean(top1_hits)),
        f"top{top_k}_oracle_hit_rate": float(np.mean(topk_hits)),
        "mean_oracle_rank": float(np.mean(oracle_ranks)),
        "mrr": float(np.mean(reciprocal_ranks)),
        "ndcg": _nanmean(ndcgs),
        "mean_spearman": _nanmean(spearmans),
        "mean_score_entropy": _nanmean(entropies),
    }
