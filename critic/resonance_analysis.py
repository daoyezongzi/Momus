from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CachedResonanceScore:
    price: float
    volume: float
    price_volume: float


def _direction_agreement(target: np.ndarray, related: np.ndarray) -> float:
    if target.size == 0 or related.size == 0:
        return 0.0
    target_sign = np.sign(target)
    related_sign = np.sign(related)
    if related_sign.ndim == 1:
        related_sign = related_sign[None, :]
    target_sign = target_sign[-related_sign.shape[1]:]
    related_sign = related_sign[:, -target_sign.shape[0]:]
    return float(np.mean(related_sign == target_sign[None, :]))


def _internal_direction_consensus(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    signs = np.sign(values)
    if signs.ndim == 1:
        signs = signs[None, :]
    up = np.mean(signs > 0, axis=0)
    down = np.mean(signs < 0, axis=0)
    return float(np.mean(np.maximum(up, down)))


def compute_cached_resonance_score(sample) -> CachedResonanceScore:
    target_close_moves = np.diff(sample.hist[:, 3])
    if sample.corr_feat is None:
        return CachedResonanceScore(0.0, 0.0, 0.0)
    corr_close_moves = sample.corr_feat[:, :, 3]
    corr_volume_moves = sample.corr_feat[:, :, 4] if sample.corr_feat.shape[2] > 4 else corr_close_moves
    price = _direction_agreement(target_close_moves, corr_close_moves)
    volume = _internal_direction_consensus(corr_volume_moves)
    return CachedResonanceScore(price=price, volume=volume, price_volume=(price + volume) / 2.0)


def assign_bins(train_scores, eval_scores, field: str):
    train_values = np.asarray([getattr(score, field) for score in train_scores], dtype=float)
    low_cut, high_cut = np.quantile(train_values, [1 / 3, 2 / 3])
    bins = []
    for score in eval_scores:
        value = getattr(score, field)
        if value <= low_cut:
            bins.append("low")
        elif value >= high_cut:
            bins.append("high")
        else:
            bins.append("mid")
    return bins


def summarize_distances_by_bin(samples, bins):
    summary = {}
    for sample, bin_name in zip(samples, bins):
        row = summary.setdefault(bin_name, {"n": 0, "_values": []})
        row["n"] += 1
        row["_values"].append(float(np.mean(sample.distances)))
    for row in summary.values():
        row["simple_mean_exp_distance"] = float(np.mean(row.pop("_values")))
    return summary


def bootstrap_bin_difference(low_values, high_values, n_boot=2000, seed=0):
    low = np.asarray(low_values, dtype=float)
    high = np.asarray(high_values, dtype=float)
    if len(low) == 0 or len(high) == 0:
        raise ValueError("low and high bins must both be non-empty")
    effect = float(np.mean(high) - np.mean(low))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        low_sample = rng.choice(low, size=len(low), replace=True)
        high_sample = rng.choice(high, size=len(high), replace=True)
        boot.append(float(np.mean(high_sample) - np.mean(low_sample)))
    boot_arr = np.asarray(boot, dtype=float)
    ci_low, ci_high = np.quantile(boot_arr, [0.025, 0.975])
    p_value = 2 * min(np.mean(boot_arr <= 0), np.mean(boot_arr >= 0))
    return {
        "effect_high_minus_low": effect,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(min(1.0, p_value)),
    }
