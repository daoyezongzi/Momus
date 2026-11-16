"""Evaluate no-training baselines on the frozen low-frequency candidate cache.

The metric follows the manuscript's OHLC-MSE convention: both prediction and
realized future are represented in each stock's frozen history-normalized
space, and MSE is computed over the four OHLC columns only.  This script does
not regenerate candidates, train weights, or modify the manuscript.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common_lowfreq import BASE, DATA, OUT, ROOT, calendar, digest, identity
from evaluate_lowfreq import load_shards, verify_provider
from prepare import bridge_mergers


RANDOM_SEEDS = tuple(range(5))
RANDOM_BASE_SEED = 20260916
OHLC = slice(0, 4)
OUTPUT = ROOT / "output/hs300_lowfreq_baseline_mse_20260916"


def _market_frames(members: pd.DataFrame, cal: pd.DatetimeIndex):
    raw = {
        code: pd.read_csv(DATA / "market" / f"{code}.csv", parse_dates=["date"]).set_index("date")
        for code in members.instrument.unique()
    }
    merger = json.loads((BASE / "merger_events.json").read_text(encoding="utf-8"))
    bridged, _ = bridge_mergers(raw, cal, merger["events"])
    return raw, bridged


def _truth_normalized(shard, markets: dict[str, pd.DataFrame], future: pd.DatetimeIndex):
    values = []
    for code in shard["codes"].tolist():
        frame = markets[str(code)].reindex(future)
        raw = frame[["open", "high", "low", "close", "volume", "amount"]].to_numpy(dtype=np.float64)
        values.append(raw)
    truth = np.stack(values)
    valid = np.isfinite(truth[:, :, :4]).all(axis=2)
    normalized = (truth - shard["mean"].astype(np.float64)[:, None, :]) / shard["scale"].astype(np.float64)[:, None, :]
    normalized[~valid] = np.nan
    return normalized, valid


def _mse_rows(
    predictions: dict[str, np.ndarray],
    truth: np.ndarray,
    valid: np.ndarray,
    period: str,
    execution: str,
    origin: str,
):
    rows = []
    stock_mask = valid.all(axis=1)
    valid_stocks = int(stock_mask.sum())
    excluded_stocks = int(len(stock_mask) - valid_stocks)
    for method, pred in predictions.items():
        error = pred[stock_mask, :, OHLC] - truth[stock_mask, :, OHLC]
        squared_error_sum = float(np.sum(error**2))
        absolute_error_sum = float(np.sum(np.abs(error)))
        denominator = int(error.shape[0] * error.shape[1] * error.shape[2])
        rows.append(
            {
                "period": period,
                "execution": execution,
                "origin": origin,
                "method": method,
                "stocks": int(pred.shape[0]),
                "valid_stocks": valid_stocks,
                "excluded_stocks": excluded_stocks,
                "valid_ohlc_values": denominator,
                "squared_error_sum": squared_error_sum,
                "absolute_error_sum": absolute_error_sum,
                "ohlc_mse": squared_error_sum / denominator if denominator else np.nan,
                "ohlc_mae": absolute_error_sum / denominator if denominator else np.nan,
            }
        )
    return rows


def _summary(per_window: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "simple_mean": ["simple_mean"],
        "median": ["median"],
        "random_candidate": [f"random_candidate_seed_{seed}" for seed in RANDOM_SEEDS],
        "most_bullish": ["most_bullish"],
        "lowest_volatility": ["lowest_volatility"],
        "momus_weighted": [f"momus_weighted_seed_{seed}" for seed in RANDOM_SEEDS],
        "momus_top1": [f"momus_top1_seed_{seed}" for seed in RANDOM_SEEDS],
        "oracle_ohlc": ["oracle_ohlc"],
    }
    labels = {
        "simple_mean": "Kronos等权平均",
        "median": "逐点中位数",
        "random_candidate": "随机候选（五种子均值）",
        "most_bullish": "最高终点涨幅候选",
        "lowest_volatility": "最低预测区间候选",
        "momus_weighted": "Momus加权聚合（五种子均值）",
        "momus_top1": "Momus首位候选（五种子均值）",
        "oracle_ohlc": "Oracle OHLC（未来信息上界）",
    }
    out = []
    for period in ["2025", "2026H1", "all"]:
        source = per_window if period == "all" else per_window[per_window.period == period]
        base_source = source[source.method == "simple_mean"]
        base = base_source.squared_error_sum.sum() / base_source.valid_ohlc_values.sum()
        for method, members in groups.items():
            selected = source[source.method.isin(members)]
            if selected.empty:
                continue
            denominator = selected.valid_ohlc_values.sum()
            mse = float(selected.squared_error_sum.sum() / denominator)
            mae = float(selected.absolute_error_sum.sum() / denominator)
            out.append(
                {
                    "period": period,
                    "method": method,
                    "method_label": labels[method],
                    "seed_count": len(members),
                    "windows": int(selected.execution.nunique()),
                    "valid_stock_horizons": int(selected.valid_stocks.sum()),
                    "excluded_stock_horizons": int(selected.excluded_stocks.sum()),
                    "ohlc_mse": mse,
                    "ohlc_mae": mae,
                    "mse_improvement_vs_simple_pct": float((base - mse) / base * 100.0),
                    "future_information": method == "oracle_ohlc",
                }
            )
    return pd.DataFrame(out)


def _write_report(root: Path, summary: pd.DataFrame, target_nonfinite: int) -> None:
    lines = [
        "# HS300低频 baseline OHLC-MSE 补充评估",
        "",
        "本评估复用冻结的37个决策日期、每期300只成分股和16条候选路径，没有重新生成候选路径或训练任何权重。真实未来仅用于事后评价，不进入候选选择。",
        "",
        "MSE定义为每只股票的历史窗口归一化空间中，预测路径与真实未来路径在OHLC四列上的逐点均方误差；数值越低越好。只有未来10个交易日OHLC全部可观测的股票路径纳入MSE，停牌或缺失路径不以前向填充代替。报告的改善率为相对Kronos等权平均的下降比例。",
        "",
        "| 时期 | 方法 | OHLC-MSE | OHLC-MAE | 相对等权平均改善 | 窗口数 | 有效股票路径 | 排除股票路径 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.period} | {row.method_label} | {row.ohlc_mse:.6f} | {row.ohlc_mae:.6f} | {row.mse_improvement_vs_simple_pct:+.2f}% | {row.windows} | {row.valid_stock_horizons} | {row.excluded_stock_horizons} |"
        )
    lines += [
        "",
        "Oracle OHLC使用未来真实路径选择候选，仅表示固定候选池的理论上界，不是可部署baseline。随机候选为五个固定种子的均值；Momus为五个既有checkpoint的均值。",
        f"",
        f"真实未来目标中的非有限OHLC日观测数：{target_nonfinite}；表中按完整10日路径统计，未使用填充值。",
        "",
        "该结果只说明预测层面的OHLC误差，不自动推出横截面收益排序或投资组合收益优势。",
    ]
    (root / "MSE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)

    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    generation = json.loads((OUT / "generation_manifest.json").read_text(encoding="utf-8"))
    cal = calendar()
    members = pd.read_csv(BASE / "membership.csv", parse_dates=["start_date", "end_date"])
    input_markets, markets = _market_frames(members, cal)

    random_generators = {
        seed: np.random.default_rng(RANDOM_BASE_SEED + seed) for seed in RANDOM_SEEDS
    }
    per_window = []
    target_nonfinite = 0
    target_invalid_stock_horizons = 0
    replay_checks = []

    for period, (start, end) in protocol["periods"].items():
        selected = [row for row in protocol["schedule"] if row["period"] == period]
        _, _, shards, checks, _ = load_shards(
            selected, protocol, cal, input_markets,
        )
        replay_checks.extend(checks)
        for meta in shards:
            with np.load(meta["path"], allow_pickle=False) as shard:
                paths = shard["paths"].astype(np.float64)
                scores = shard["scores"].astype(np.float64)
                hist = shard["hist"].astype(np.float64)
                closes = paths[:, :, :, 3]
                future = pd.DatetimeIndex(shard["future"])
                truth, valid = _truth_normalized(shard, markets, future)
                target_nonfinite += int((~valid).sum())
                target_invalid_stock_horizons += int((~valid.all(axis=1)).sum())

                stocks = np.arange(paths.shape[0])
                predictions = {
                    "simple_mean": paths.mean(axis=1),
                    "median": np.median(paths, axis=1),
                }
                for seed, rng in random_generators.items():
                    chosen = rng.integers(0, paths.shape[1], size=paths.shape[0])
                    predictions[f"random_candidate_seed_{seed}"] = paths[stocks, chosen]

                bullish = np.nanargmax(closes[:, :, -1] - hist[:, -1, 3][:, None], axis=1)
                predictions["most_bullish"] = paths[stocks, bullish]

                predicted_range = np.nanmax(paths[:, :, :, 1], axis=2) - np.nanmin(paths[:, :, :, 2], axis=2)
                lowvol = np.nanargmin(predicted_range, axis=1)
                predictions["lowest_volatility"] = paths[stocks, lowvol]

                for seed in RANDOM_SEEDS:
                    score = scores[seed]
                    shifted = score - score.max(axis=1, keepdims=True)
                    weights = np.exp(shifted)
                    weights /= weights.sum(axis=1, keepdims=True)
                    predictions[f"momus_weighted_seed_{seed}"] = np.einsum(
                        "bn,bnhf->bhf", weights, paths
                    )
                    top1 = np.argmax(score, axis=1)
                    predictions[f"momus_top1_seed_{seed}"] = paths[stocks, top1]

                candidate_error = paths[:, :, :, OHLC] - truth[:, None, :, OHLC]
                candidate_error = np.where(valid[:, None, :, None], candidate_error, np.nan)
                candidate_mse = np.nanmean(candidate_error**2, axis=(2, 3))
                oracle = np.argmin(candidate_mse, axis=1)
                predictions["oracle_ohlc"] = paths[stocks, oracle]

                per_window.extend(
                    _mse_rows(
                        predictions,
                        truth,
                        valid,
                        period,
                        str(shard["execution"]),
                        str(shard["origin"]),
                    )
                )

    per_window_df = pd.DataFrame(per_window)
    summary = _summary(per_window_df)
    per_window_df.to_csv(root / "mse_per_window.csv", index=False, float_format="%.12g")
    summary.to_csv(root / "mse_summary.csv", index=False, float_format="%.12g")
    pd.DataFrame(replay_checks).to_csv(root / "cache_replay_checks.csv", index=False, float_format="%.12g")

    source_files = [
        Path(__file__),
        ROOT / "examples/hs300_lowfreq_investment/evaluate_lowfreq.py",
        ROOT / "examples/hs300_lowfreq_investment/common_lowfreq.py",
        ROOT / "examples/hs300_daily_investment/prepare.py",
        ROOT / "critic/evaluation.py",
    ]
    manifest = {
        "status": "COMPLETE",
        "base_output": str(OUT),
        "base_protocol_identity": identity(protocol),
        "base_generation_identity": identity(generation),
        "candidate_generation_reused": True,
        "model_training": False,
        "manuscript_modified": False,
        "periods": list(protocol["periods"]),
        "forecast_dates": len(protocol["schedule"]),
        "mse_space": "per-stock history-normalized OHLC columns 0:4",
        "source_sha256": {str(path): digest(path) for path in source_files},
        "target_invalid_ohlc_day_count": target_nonfinite,
        "target_invalid_stock_horizons": target_invalid_stock_horizons,
    }
    (root / "evaluation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_report(root, summary, target_nonfinite)
    print(json.dumps({"status": "COMPLETE", "output": str(root), "rows": len(per_window_df)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
