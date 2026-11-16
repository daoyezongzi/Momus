"""Replay no-training baselines on the frozen low-frequency cache.

This extension deliberately does not regenerate candidates or train weights.
It reuses the sealed 37-date shards and evaluates deterministic heuristics,
five seeded random selections, and five already-trained Small MLP checkpoints.
The original Kronos/simple-mean, median, and Momus accounts remain in their
existing output directories and are combined only in the summary inputs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common_lowfreq import BASE, DATA, OUT, ROOT, digest, identity, calendar
from evaluate_lowfreq import load_shards, verify_provider
from prepare import bridge_mergers
from qlib_runner import run_backtest
from validation import audit_backtest, audit_independent_ledger, validate_signal_schedule
from evaluate import measures

sys.path.insert(0, str(ROOT / "examples"))
from evaluate_cache import load_critic


EXTENSION_OUT = ROOT / "output/hs300_lowfreq_baseline_extension_no_training_20260916"
PREVIOUS_EXTENSION_OUT = ROOT / "output/hs300_lowfreq_baseline_extension_20260916"
RANDOM_SEEDS = tuple(range(5))
SMALL_MLP_ROOT = ROOT / "output/results/new_pool_II_multiseed_robustness/small_mlp"
METHODS = (
    [f"random_candidate_seed_{i}" for i in RANDOM_SEEDS]
    + ["most_bullish", "lowest_volatility"]
    + [f"small_mlp_weighted_seed_{i}" for i in RANDOM_SEEDS]
    + [f"small_mlp_top1_seed_{i}" for i in RANDOM_SEEDS]
)
METHOD_LABELS = {
    "simple_mean": "Kronos等权平均",
    "median": "逐点中位数",
    "random_candidate": "随机候选（五种子均值）",
    "most_bullish": "最高终点涨幅候选",
    "lowest_volatility": "最低预测区间候选",
    "small_mlp_weighted": "Small MLP加权（五种子均值）",
    "small_mlp_top1": "Small MLP首位候选（五种子均值）",
    "momus_weighted": "Momus加权聚合（五种子均值）",
    "momus_top1": "Momus首位候选（五种子均值）",
}


def _load_small_mlp_models(device: str) -> list[torch.nn.Module]:
    """Load the five existing Small MLP checkpoints without retraining."""
    models = []
    for seed in RANDOM_SEEDS:
        checkpoint = SMALL_MLP_ROOT / f"seed_{seed}" / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing frozen Small MLP checkpoint: {checkpoint}")
        model, arch = load_critic(
            str(checkpoint),
            arch="fundamental",
            device=device,
            setwise=False,
            path_corr_interaction=False,
        )
        if arch != "fundamental":
            raise ValueError(f"Unexpected Small MLP architecture for {checkpoint}: {arch}")
        models.append(model)
    return models


def _batch_small_mlp_scores(model: torch.nn.Module, paths: np.ndarray, hist: np.ndarray) -> np.ndarray:
    """Score all stock-candidate pairs in batches with a frozen Small MLP."""
    stocks, candidates, horizon, features = paths.shape
    flat_paths = np.ascontiguousarray(paths.astype(np.float32).reshape(stocks * candidates, horizon, features))
    flat_hist = np.ascontiguousarray(
        np.repeat(hist.astype(np.float32)[:, None, :, :], candidates, axis=1).reshape(
            stocks * candidates, hist.shape[1], hist.shape[2]
        )
    )
    device = next(model.parameters()).device
    outputs = []
    with torch.no_grad():
        for start in range(0, len(flat_paths), 2048):
            end = min(start + 2048, len(flat_paths))
            path_t = torch.from_numpy(flat_paths[start:end]).to(device)
            hist_t = torch.from_numpy(flat_hist[start:end]).to(device)
            outputs.append(model(path_t, hist_t).detach().cpu().numpy())
    return np.concatenate(outputs).reshape(stocks, candidates)


def _signal_from_shards(shards: list[dict], small_mlps: list[torch.nn.Module]) -> dict[str, pd.Series]:
    """Build no-training signals from cached paths and frozen checkpoints."""
    pieces = {method: [] for method in METHODS}
    random_generators = {
        seed: np.random.default_rng(20260916 + seed) for seed in RANDOM_SEEDS
    }
    replay_rows = []
    for meta in shards:
        with np.load(meta["path"], allow_pickle=False) as shard:
            paths = shard["paths"].astype(float)
            hist = shard["hist"].astype(float)
            closes = paths[:, :, :, 3]
            codes = shard["codes"].tolist()
            origin = pd.Timestamp(str(shard["origin"]))
            index = pd.MultiIndex.from_product([[origin], codes], names=["datetime", "instrument"])
            stock_index = np.arange(len(codes))

            def to_signal(predicted_close: np.ndarray) -> np.ndarray:
                return (
                    predicted_close * shard["scale"][:, 3] + shard["mean"][:, 3]
                ) / shard["last_close"] - 1

            random_finite = True
            for seed, rng in random_generators.items():
                selected = rng.integers(0, paths.shape[1], size=len(codes))
                average_close = closes[stock_index, selected, :].mean(axis=1)
                signal = to_signal(average_close)
                pieces[f"random_candidate_seed_{seed}"].append(
                    pd.Series(signal, index=index, dtype=float)
                )
                random_finite = random_finite and bool(np.isfinite(signal).all())

            most_bullish = np.nanargmax(
                closes[:, :, -1] - hist[:, -1, 3][:, None], axis=1
            )
            bullish_signal = to_signal(closes[stock_index, most_bullish, :].mean(axis=1))
            pieces["most_bullish"].append(pd.Series(bullish_signal, index=index, dtype=float))

            predicted_range = (
                np.nanmax(paths[:, :, :, 1], axis=2)
                - np.nanmin(paths[:, :, :, 2], axis=2)
            )
            lowest_volatility = np.nanargmin(predicted_range, axis=1)
            lowvol_signal = to_signal(closes[stock_index, lowest_volatility, :].mean(axis=1))
            pieces["lowest_volatility"].append(pd.Series(lowvol_signal, index=index, dtype=float))

            small_mlp_finite = True
            candidate_close_means = closes.mean(axis=2)
            for seed, model in zip(RANDOM_SEEDS, small_mlps):
                scores = _batch_small_mlp_scores(model, paths, hist)
                shifted = scores - scores.max(axis=1, keepdims=True)
                weights = np.exp(shifted)
                weighted_close = (weights * candidate_close_means).sum(axis=1) / weights.sum(axis=1)
                weighted_signal = to_signal(weighted_close)
                pieces[f"small_mlp_weighted_seed_{seed}"].append(
                    pd.Series(weighted_signal, index=index, dtype=float)
                )

                selected = np.argmax(scores, axis=1)
                top1_signal = to_signal(closes[stock_index, selected, :].mean(axis=1))
                pieces[f"small_mlp_top1_seed_{seed}"].append(
                    pd.Series(top1_signal, index=index, dtype=float)
                )
                small_mlp_finite = small_mlp_finite and bool(
                    np.isfinite(scores).all() and np.isfinite(weighted_signal).all() and np.isfinite(top1_signal).all()
                )

            replay_rows.append(
                {
                    "execution": str(shard["execution"]),
                    "origin": str(shard["origin"]),
                    "stocks": len(codes),
                    "candidates": int(paths.shape[1]),
                    "random_signal_finite": random_finite,
                    "most_bullish_signal_finite": bool(np.isfinite(bullish_signal).all()),
                    "lowest_volatility_signal_finite": bool(np.isfinite(lowvol_signal).all()),
                    "small_mlp_signal_finite": small_mlp_finite,
                }
            )

    signals = {method: pd.concat(part).sort_index() for method, part in pieces.items()}
    return signals, pd.DataFrame(replay_rows)


def _extra_return_ic(shards: list[dict], markets: dict[str, pd.DataFrame], cal: pd.DatetimeIndex,
                     signals: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for meta in shards:
        with np.load(meta["path"], allow_pickle=False) as shard:
            future = pd.DatetimeIndex(shard["future"])
            realized = []
            for code, last in zip(shard["codes"], shard["last_close"]):
                marks = markets[str(code)].close.reindex(cal).ffill().reindex(future)
                realized.append(float(marks.mean() / last - 1) if marks.notna().all() else np.nan)
            realized = np.asarray(realized, dtype=float)
            origin = pd.Timestamp(str(meta["origin"]))
            for method, signal in signals.items():
                predicted = signal.xs(origin, level="datetime").reindex(shard["codes"]).to_numpy(dtype=float)
                mask = np.isfinite(realized) & np.isfinite(predicted)
                p, y = pd.Series(predicted[mask]), pd.Series(realized[mask])
                ic = p.corr(y) if p.std() > 0 and y.std() > 0 else np.nan
                rank_ic = p.rank().corr(y.rank()) if p.std() > 0 and y.std() > 0 else np.nan
                rows.append(
                    {
                        "origin": str(origin.date()),
                        "method": method,
                        "stocks": int(mask.sum()),
                        "return_ic": ic,
                        "return_rank_ic": rank_ic,
                        "label": "mean next 10 market-day adjusted closing marks / latest positive-volume observed adjusted close at or before origin - 1",
                    }
                )
    return pd.DataFrame(rows)


def _write_positions(result, path: Path) -> None:
    holdings = []
    for day, position in result.positions.items():
        for code in position.get_stock_list():
            holdings.append(
                {
                    "date": str(day.date()),
                    "instrument": code,
                    "synthetic_amount": position.get_stock_amount(code),
                    "adjusted_price": position.get_stock_price(code),
                    "weight": position.get_stock_weight(code),
                    "cash": position.get_cash(),
                }
            )
    pd.DataFrame(holdings).to_csv(path, index=False, float_format="%.12g")


def _aggregate_summary(rows: pd.DataFrame) -> pd.DataFrame:
    output = []
    for period in sorted(rows.period.unique()):
        part = rows[rows.period == period]
        groups = [
            ("simple_mean", part[part.method == "simple_mean"]),
            ("median", part[part.method == "median"]),
            ("random_candidate", part[part.method.str.startswith("random_candidate_seed_")]),
            ("most_bullish", part[part.method == "most_bullish"]),
            ("lowest_volatility", part[part.method == "lowest_volatility"]),
            ("small_mlp_weighted", part[part.method.str.startswith("small_mlp_weighted_seed_")]),
            ("small_mlp_top1", part[part.method.str.startswith("small_mlp_top1_seed_")]),
            ("momus_weighted", part[part.method.str.startswith("momus_seed_")]),
            ("momus_top1", part[part.method.str.startswith("momus_top1_seed_")]),
        ]
        for method, selected in groups:
            if selected.empty:
                continue
            numeric = selected.select_dtypes(include="number").mean()
            output.append(
                {
                    "period": period,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    **{key: float(value) for key, value in numeric.items()},
                    "accounts": int(len(selected)),
                }
            )
    return pd.DataFrame(output)


def _risk_metrics(report: pd.DataFrame) -> dict[str, float]:
    returns = report.net_return.to_numpy(dtype=float)
    volatility = float(np.std(returns, ddof=1) * np.sqrt(252)) if len(returns) > 1 else np.nan
    sharpe = float(np.mean(returns) / np.std(returns, ddof=1) * np.sqrt(252)) if len(returns) > 1 and np.std(returns, ddof=1) > 0 else np.nan
    return {"annualized_volatility": volatility, "sharpe": sharpe}


def _enrich_existing_risk_metrics(existing: pd.DataFrame) -> pd.DataFrame:
    existing = existing.copy()
    volatility, sharpe = [], []
    for row in existing.itertuples(index=False):
        daily = pd.read_csv(OUT / "results" / str(row.period) / f"{row.method}_daily.csv")
        metrics = _risk_metrics(daily)
        volatility.append(metrics["annualized_volatility"])
        sharpe.append(metrics["sharpe"])
    existing["annualized_volatility"] = volatility
    existing["sharpe"] = sharpe
    return existing


def _write_report(root: Path, combined: pd.DataFrame, summary: pd.DataFrame) -> None:
    lines = [
        "# HS300低频收益 baseline 扩展回放",
        "",
        "本目录只对冻结的37个决策日期、每期300只成分股和16条候选路径进行新增方法回放；没有重新生成候选路径或重新训练模型。新增方法包括五组随机候选、最高终点涨幅、最低预测区间，以及复用既有五个Small MLP checkpoint的加权和首位选择。Kronos等权平均、逐点中位数、Momus加权和Momus首位候选账户引用既有评估结果。",
        "",
        "所有账户共享原协议：Top50、drop5、最短持有5个交易日、下一期开盘成交、95%目标仓位、双边费率各0.15%、最低费用5元和100股取整。随机候选使用固定种子20260916至20260920；Small MLP使用既有seed 0至4 checkpoint，汇总均值不是五账户合成后的可交易组合。",
        "",
        "| 时期 | 方法 | 账户数 | 扣费累计收益 | 同期指数收益 | 最大回撤 | 年化波动率 | 夏普比率 | 收益信号IC | 收益信号RankIC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row.period} | {row.method_label} | {int(row.accounts)} | {row.cumulative_net_return:.2%} | {row.benchmark_cumulative_return:.2%} | {row.max_drawdown:.2%} | {row.annualized_volatility:.2%} | {row.sharpe:.4f} | {row.mean_return_ic:.4f} | {row.mean_return_rank_ic:.4f} |"
        )
    lines += [
        "",
        "该扩展仍属于补充性低频投资组合回放；37个决策截面、固定模型和固定候选缓存不足以支持普遍性的投资优越性结论。原始数据的复权、公司行为和点时成员覆盖限制仍然有效。",
        "",
        "`investment_per_run_all_methods.csv`保留逐账户结果；`investment_summary_all_methods.csv`为各方法的分组汇总。",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=EXTENSION_OUT)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)

    small_mlps = _load_small_mlp_models(args.device)

    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    generation = json.loads((OUT / "generation_manifest.json").read_text(encoding="utf-8"))
    provider = verify_provider()
    cal = calendar()
    members = pd.read_csv(BASE / "membership.csv", parse_dates=["start_date", "end_date"])
    input_markets = {
        code: pd.read_csv(DATA / "market" / f"{code}.csv", parse_dates=["date"]).set_index("date")
        for code in members.instrument.unique()
    }
    markets, _ = bridge_mergers(
        input_markets,
        cal,
        json.loads((BASE / "merger_events.json").read_text(encoding="utf-8"))["events"],
    )

    source_files = [
        Path(__file__),
        ROOT / "examples/hs300_lowfreq_investment/evaluate_lowfreq.py",
        ROOT / "examples/hs300_lowfreq_investment/common_lowfreq.py",
        ROOT / "examples/hs300_lowfreq_investment/validation.py",
        ROOT / "examples/hs300_lowfreq_investment/../hs300_daily_investment/qlib_runner.py",
        ROOT / "examples/evaluate_cache.py",
        ROOT / "critic/model.py",
    ]
    manifest = {
        "status": "RUNNING",
        "created": "2026-09-16",
        "base_output": str(OUT),
        "base_protocol_identity": identity(protocol),
        "base_generation_identity": identity(generation),
        "new_methods": METHODS,
        "small_mlp_checkpoints": [
            str(SMALL_MLP_ROOT / f"seed_{seed}" / "best.pt") for seed in RANDOM_SEEDS
        ],
        "source_sha256": {str(path): digest(path) for path in source_files},
        "candidate_generation_reused": True,
        "model_training": False,
        "manuscript_modified": False,
    }
    (root / "evaluation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = []
    replay = []
    ledger = []
    ic_parts = []
    started = time.time()
    for period, (start, end) in protocol["periods"].items():
        period_rows = [row for row in protocol["schedule"] if row["period"] == period]
        days = cal[(cal >= start) & (cal <= end)]
        _, expected, shards, checks, _ = load_shards(period_rows, protocol, cal, input_markets)
        signals, replay_df = _signal_from_shards(shards, small_mlps)
        for method, signal in signals.items():
            validate_signal_schedule(signal, expected)
        replay.append(replay_df)
        ic_parts.append(_extra_return_ic(shards, markets, cal, signals))

        period_dir = root / period
        period_dir.mkdir(parents=True, exist_ok=True)
        execution_dates = [row["execution"] for row in period_rows]
        for method, signal in signals.items():
            print(f"backtest {period} {method}", flush=True)
            result = run_backtest(
                provider,
                signal,
                days[0],
                days[-1],
                expected_signals=expected,
                allow_missing_signal_dates=True,
            )
            audit = audit_backtest(result, execution_dates, days)
            audit["independent_ledger"] = audit_independent_ledger(result, markets, cal)
            for order in result.trades.itertuples():
                day = str(pd.Timestamp(order.datetime).date())
                if order.direction == "buy" and order.instrument not in protocol["universes"][day]:
                    raise ValueError("Purchased a name outside the scheduled constituent universe")
            ledger.append({"period": period, "method": method, **audit})
            result.report.to_csv(period_dir / f"{method}_daily.csv", float_format="%.12g")
            result.trades.to_csv(period_dir / f"{method}_orders.csv", index=False, float_format="%.12g")
            _write_positions(result, period_dir / f"{method}_positions.csv")
            result.metadata.update(scheduled_execution_dates=execution_dates, no_orders_between_decisions=True)
            (period_dir / f"{method}_protocol.json").write_text(
                json.dumps(result.metadata, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
            )
            row = {
                "period": period,
                "method": method,
                "decision_dates": len(period_rows),
                **measures(result.report, result.trades),
                **_risk_metrics(result.report),
            }
            rows.append(row)

        print(f"validated {period}: shards={len(shards)} checks={len(checks)}", flush=True)

    extra_rows = pd.DataFrame(rows)
    existing_frames = [
        _enrich_existing_risk_metrics(pd.read_csv(OUT / "results" / "investment_per_run.csv"))
    ]
    previous_rows_path = PREVIOUS_EXTENSION_OUT / "investment_per_run_all_methods.csv"
    if previous_rows_path.exists():
        previous_rows = pd.read_csv(previous_rows_path)
        previous_rows = previous_rows[
            (previous_rows.method == "median")
            | previous_rows.method.astype(str).str.startswith("momus_top1_seed_")
        ]
        existing_frames.append(previous_rows)
    existing_rows = pd.concat(existing_frames, ignore_index=True, sort=False)
    combined = pd.concat([existing_rows, extra_rows], ignore_index=True, sort=False)
    ic = pd.concat(ic_parts, ignore_index=True)
    # The first 2025 execution date uses the preceding 2024-12-31 origin;
    # map by frozen origin rather than by the period's execution dates.
    origin_lookup = {str(item["origin"]): item["period"] for item in protocol["schedule"]}
    ic["period"] = ic.origin.map(lambda value: origin_lookup[pd.Timestamp(value).strftime("%Y-%m-%d")])
    for row in rows:
        part = ic[(ic.period == row["period"]) & (ic.method == row["method"])]
        row["mean_return_ic"] = float(part.return_ic.mean())
        row["mean_return_rank_ic"] = float(part.return_rank_ic.mean())
    extra_rows = pd.DataFrame(rows)
    combined = pd.concat([existing_rows, extra_rows], ignore_index=True, sort=False)
    combined.to_csv(root / "investment_per_run_all_methods.csv", index=False, float_format="%.12g")
    summary = _aggregate_summary(combined)
    summary.to_csv(root / "investment_summary_all_methods.csv", index=False, float_format="%.12g")
    ic.to_csv(root / "decision_return_ic_new_methods.csv", index=False, float_format="%.12g")
    pd.concat(replay, ignore_index=True).to_csv(root / "signal_replay_checks.csv", index=False, float_format="%.12g")
    (root / "ledger_checks.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_report(root, combined, summary)
    manifest.update(
        {
            "status": "COMPLETE",
            "elapsed_seconds": time.time() - started,
            "periods": list(protocol["periods"]),
            "new_portfolios": len(rows),
            "new_methods_per_period": len(METHODS),
            "forecast_dates": len(protocol["schedule"]),
            "manuscript_modified": False,
        }
    )
    (root / "evaluation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "output": str(root), "new_portfolios": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
