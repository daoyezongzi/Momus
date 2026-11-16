"""Paired Qlib portfolios and date-axis investment statistics from daily shards."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from common import OUT, DATA, PERIODS, digest, identity, write_json
from qlib_runner import run_backtest
from prepare import bridge_mergers

LABELS = ["simple_mean"] + [f"momus_seed_{i}" for i in range(5)]


def measures(report, trades):
    r = report.net_return.to_numpy()
    b = report.bench.to_numpy()
    a = r-b
    sd = np.std(a, ddof=1) if len(a) > 1 else np.nan
    nav = float(report.net_nav.iloc[-1])
    benchmark = float(np.prod(1+b))
    return dict(days=len(r), cumulative_net_return=nav-1, benchmark_cumulative_return=benchmark-1,
                cumulative_return_difference=nav-benchmark, compound_annual_return=nav**(252/len(r))-1,
                max_drawdown=float(report.net_drawdown.min()), aer=252*float(a.mean()),
                information_ratio=float(np.sqrt(252)*a.mean()/sd) if sd > 0 else None,
                average_daily_turnover=float(report.turnover.mean()), fees_currency=float(trades.cost.sum()),
                mean_positions=float(report.holding_count.mean()), max_positions=int(report.holding_count.max()))


def block_intervals(difference, block, rng, repeats=5000):
    """Circular moving blocks on paired dates, conditional on frozen candidates."""
    difference = np.asarray(difference, dtype=float)
    n = len(difference)
    if n < 3*block:
        return {"block_days": block, "available_blocks": n/block, "interval": None, "reason": "fewer than three blocks"}
    starts = rng.integers(0, n, size=(repeats, int(np.ceil(n/block))))
    indices = (starts[..., None] + np.arange(block)) % n
    sample = difference[indices.reshape(repeats, -1)[:, :n]].mean(axis=1)
    return {"block_days": block, "available_blocks": n/block, "interval": np.quantile(sample, [.025, .975]).tolist(), "unit": "mean daily net return difference"}


def load_shards(start, end, calendar):
    days = calendar[(calendar >= start) & (calendar <= end)]
    signals = [[] for _ in LABELS]
    expected, shard_meta = {}, []
    generation = json.loads((OUT / "generation_manifest.json").read_text(encoding="utf-8"))
    for day in days:
        path = OUT / "candidates" / f"{day:%Y%m%d}.npz"
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if meta["sha256"] != digest(path) or meta["manifest"] != identity(generation):
            raise ValueError(f"Invalid shard identity: {path}")
        with np.load(path, allow_pickle=False) as s:
            origin = pd.Timestamp(str(s["origin"]))
            if calendar[calendar.get_loc(day)-1] != origin or str(s["execution"]) != str(day.date()):
                raise ValueError("Shard signal/execution lag mismatch")
            codes = s["codes"].tolist()
            expected[origin] = codes
            idx = pd.MultiIndex.from_product([[origin], codes], names=["datetime", "instrument"])
            for method, values in zip(signals, s["signals"]):
                method.append(pd.Series(values, index=idx))
            shard_meta.append(dict(path=path, origin=origin, day=day, codes=codes))
    return [pd.concat(x) for x in signals], expected, shard_meta


def cross_section_metrics(shards, market, calendar):
    rows = []
    for meta in shards:
        with np.load(meta["path"], allow_pickle=False) as s:
            future = pd.DatetimeIndex(s["future"])
            realized = []
            for code, last in zip(s["codes"], s["last_close"]):
                # Economic marks carry through suspension; merger claims are
                # bridged by the same disclosed valuation convention as Qlib.
                marks = market[str(code)].close.reindex(calendar).ffill().reindex(future)
                realized.append(float(marks.mean()/last-1) if marks.notna().all() else np.nan)
            realized = np.array(realized)
            for label, signal in zip(LABELS, s["signals"]):
                mask = np.isfinite(realized) & np.isfinite(signal)
                p, y = pd.Series(signal[mask]), pd.Series(realized[mask])
                ic = p.corr(y) if p.std() > 0 and y.std() > 0 else np.nan
                rank = p.rank().corr(y.rank()) if p.std() > 0 and y.std() > 0 else np.nan
                rows.append({"origin": str(meta["origin"].date()), "method": label, "stocks": int(mask.sum()), "return_ic": ic, "return_rank_ic": rank, "label": "mean next 10 market-day adjusted closing marks / origin close - 1"})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = OUT / ("smoke_results" if args.smoke else "results")
    root.mkdir(exist_ok=True)
    provider_identity = json.loads((OUT / "provider_identity.json").read_text(encoding="utf-8"))
    provider = OUT / provider_identity["path"]
    for name, key in [("manifest.json", "manifest_sha256"), ("calendars/day.txt", "calendar_sha256"), ("instruments/all.txt", "instruments_sha256")]:
        if digest(provider / name) != provider_identity[key]:
            raise ValueError(f"Provider metadata changed: {name}")
    provider_manifest = json.loads((provider / "manifest.json").read_text(encoding="utf-8"))
    for name, sha in provider_manifest["feature_sha256"].items():
        if digest(provider / name) != sha:
            raise ValueError(f"Provider binary changed: {name}")
    calendar = pd.DatetimeIndex(pd.read_csv(DATA / "market/SH000300.csv", parse_dates=["date"]).date)
    members = pd.read_csv(OUT / "membership.csv")
    market = {code: pd.read_csv(DATA / "market" / f"{code}.csv", parse_dates=["date"]).set_index("date") for code in members.instrument.unique()}
    market, _ = bridge_mergers(market, calendar, json.loads((OUT / "merger_events.json").read_text(encoding="utf-8"))["events"])
    periods = {"smoke": ("2024-01-01", "2024-01-03")} if args.smoke else PERIODS
    summaries, inference = [], []
    started = time.time()
    for period, (start, end) in periods.items():
        days = calendar[(calendar >= start) & (calendar <= end)]
        signals, expected, shards = load_shards(start, end, calendar)
        reports = []
        directory = root / period; directory.mkdir(exist_ok=True)
        for label, signal in zip(LABELS, signals):
            print(f"backtest {period} {label}", flush=True)
            result = run_backtest(provider, signal, days[0], days[-1], expected_signals=expected)
            result.report.to_csv(directory / f"{label}_daily.csv", float_format="%.12g")
            result.trades.to_csv(directory / f"{label}_orders.csv", index=False, float_format="%.12g")
            holdings = []
            for day, position in result.positions.items():
                for code in position.get_stock_list():
                    holdings.append(dict(date=str(day.date()), instrument=code, synthetic_amount=position.get_stock_amount(code), adjusted_price=position.get_stock_price(code), weight=position.get_stock_weight(code), cash=position.get_cash()))
            pd.DataFrame(holdings).to_csv(directory / f"{label}_positions.csv", index=False, float_format="%.12g")
            write_json(directory / f"{label}_protocol.json", result.metadata)
            summaries.append({"period": period, "method": label, **measures(result.report, result.trades)})
            reports.append(result.report)
        ic = cross_section_metrics(shards, market, calendar)
        ic.to_csv(directory / "daily_return_ic.csv", index=False, float_format="%.12g")
        for row in summaries:
            if row["period"] == period:
                part = ic[ic.method == row["method"]]
                row.update(mean_return_ic=float(part.return_ic.mean()), mean_return_rank_ic=float(part.return_rank_ic.mean()))
        delta = np.mean([r.net_return.to_numpy() for r in reports[1:]], axis=0) - reports[0].net_return.to_numpy()
        rng = np.random.default_rng(20260908)
        inference.append({"period": period, "estimand": "five-Momus-seed mean daily net return minus common-candidate simple mean", "daily_mean_difference": float(delta.mean()), "candidate_repeats": 1, "model_seeds": 5, "blocks": [block_intervals(delta, b, rng) for b in (10,20)], "interpretation": "supplementary conditional analysis; training seeds are not independent candidate generations or markets"})
        pd.DataFrame(summaries).to_csv(root / "investment_per_run.csv", index=False, float_format="%.12g")
        write_json(root / "paired_date_intervals.json", inference)
    write_json(root / "evaluation_status.json", {"status": "SMOKE_PASSED" if args.smoke else "COMPLETE", "portfolios": len(summaries), "periods": list(periods), "elapsed_seconds": time.time()-started, "claim": "research simulation under disclosed execution/adjusted-unit assumptions; not deployment proof"})


if __name__ == "__main__":
    main()
