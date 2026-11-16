import numpy as np
import pandas as pd


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    m = len(ordered)
    adjusted = {}
    running = 0.0
    for rank, key in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p_values[key]))
        adjusted[key] = running
    return {key: adjusted[key] for key in p_values}


def _paired(records: pd.DataFrame, strategy: str, baseline: str) -> pd.DataFrame:
    left = records[records.strategy == strategy].set_index("sample_id")
    right = records[records.strategy == baseline].set_index("sample_id")
    common = left.index.intersection(right.index)
    if common.empty:
        raise ValueError("no paired samples")
    return pd.DataFrame(
        {
            "stock": left.loc[common, "stock"].to_numpy(),
            "cutoff_T": pd.to_datetime(left.loc[common, "cutoff_T"]).to_numpy(),
            "delta": (left.loc[common, "metric"] - right.loc[common, "metric"]).to_numpy(dtype=float),
        }
    )


def paired_effect_table(
    records: pd.DataFrame,
    strategy: str,
    baseline: str,
    lower_is_better: bool,
    n_boot: int = 2000,
    block_len: int = 10,
    seed: int = 0,
) -> dict[str, float | bool]:
    paired = _paired(records, strategy, baseline)
    rng = np.random.default_rng(seed)
    stocks = paired.stock.unique()
    boot = []
    for _ in range(n_boot):
        sampled_stocks = rng.choice(stocks, size=len(stocks), replace=True)
        deltas = []
        for stock in sampled_stocks:
            values = paired.loc[paired.stock == stock].sort_values("cutoff_T").delta.to_numpy()
            if len(values) == 0:
                continue
            max_start = max(1, len(values) - block_len + 1)
            start = int(rng.integers(0, max_start))
            deltas.extend(values[start : start + block_len])
        if deltas:
            boot.append(float(np.mean(deltas)))

    effect = float(paired.delta.mean())
    if not boot:
        boot = [effect]
    boot_arr = np.asarray(boot, dtype=float)
    ci_low, ci_high = np.quantile(boot_arr, [0.025, 0.975])
    p_value = 2 * min(np.mean(boot_arr <= 0), np.mean(boot_arr >= 0))
    return {
        "effect": effect,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(min(1.0, p_value)),
        "lower_is_better": lower_is_better,
    }
