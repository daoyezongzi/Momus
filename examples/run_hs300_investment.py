"""Run the frozen HS300 Top-50 investment comparison on fixed N=16 caches.

This script deliberately keeps the investment experiment separate from the
forecast evaluator.  It reuses the same candidate caches, Momus checkpoints,
normalization, and N=16 aggregation, then applies the frozen
Top-50/drop-5/hold-5 long-only protocol.  No signal, checkpoint, or subset is
selected using future returns or investment results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/results/hs300_main_20260907"
CACHE_DIR = ROOT / "output/candidates"
MARKET_DIRS = {
    "2024": ROOT / "data/market_2024_hs300",
    "2025": ROOT / "data/market_2025_hs300",
    "2026H1": ROOT / "data/market_2026H1_hs300",
}
CACHE_PATHS = {
    label: CACHE_DIR / f"hs300_{label}_N16.pkl" for label in MARKET_DIRS
}
CHECKPOINTS = {
    str(seed): ROOT / "output/results/regime_generalization/multiregime_train"
    / f"seed_{seed}/fusion_best.pt"
    for seed in range(5)
}
FROZEN_PROTOCOL = ROOT / "docs/development/protocols/2026-09-07_hs300_main_preregistration.md"
BENCHMARK_FILE = OUTPUT / "benchmark_sh000300.csv"
BENCHMARK_META = OUTPUT / "benchmark_sh000300_manifest.json"

TOPK = 50
N_DROP = 5
HOLD_THRESH = 5
OPEN_COST = 0.001
CLOSE_COST = 0.0015
MIN_COST = 5.0
ACCOUNT = 100_000_000.0
METHODS = ("simple_mean", "momus_weighted")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_forecast_module():
    sys.path.insert(0, str(ROOT / "examples"))
    import evaluate_hs300_main as evaluator

    return evaluator


def fetch_benchmark(output_file: Path) -> dict:
    """Fetch SH000300 daily K-lines from the public Tencent endpoint.

    The repository's configured Tushare credential was invalid during the
    run.  The public fallback is recorded explicitly instead of substituting
    an equal-weight stock-pool benchmark.
    """
    params = {"param": "sh000300,day,2024-01-01,2026-06-30,640,qfq"}
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    rows = ((payload.get("data") or {}).get("sh000300") or {}).get("day") or []
    if not rows:
        raise RuntimeError("Tencent returned no SH000300 daily K-lines")
    columns = [
        "date",
        "close",
        "open",
        "high",
        "low",
        "volume",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    for column in columns[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame = frame.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if not frame["date"].is_monotonic_increasing:
        raise ValueError("benchmark dates are not monotonic")
    frame.to_csv(output_file, index=False, float_format="%.10g")
    metadata = {
        "symbol": "SH000300",
        "source": "Tencent public historical K-line endpoint",
        "url": url,
        "params": params,
        "fallback_reason": "configured Tushare token was rejected during the run",
        "rows": int(len(frame)),
        "first_date": str(frame["date"].iloc[0].date()),
        "last_date": str(frame["date"].iloc[-1].date()),
        "sha256": sha256(output_file),
    }
    BENCHMARK_META.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def load_benchmark() -> tuple[pd.DataFrame, dict]:
    if BENCHMARK_FILE.exists():
        frame = pd.read_csv(BENCHMARK_FILE, parse_dates=["date"])
        metadata = json.loads(BENCHMARK_META.read_text(encoding="utf-8")) if BENCHMARK_META.exists() else {
            "symbol": "SH000300",
            "source": "existing output benchmark file",
            "sha256": sha256(BENCHMARK_FILE),
        }
    else:
        metadata = fetch_benchmark(BENCHMARK_FILE)
        frame = pd.read_csv(BENCHMARK_FILE, parse_dates=["date"])
    required = {"date", "open", "close"}
    if not required.issubset(frame.columns):
        raise ValueError(f"benchmark missing columns: {required - set(frame.columns)}")
    frame = frame.sort_values("date").drop_duplicates("date").set_index("date")
    if not np.isfinite(frame[["open", "close"]].to_numpy(dtype=float)).all():
        raise ValueError("benchmark contains non-finite prices")
    return frame, metadata


def load_market(label: str, codes: set[str]) -> dict[str, pd.DataFrame]:
    market = {}
    root = MARKET_DIRS[label]
    for code in sorted(codes):
        path = root / f"{code}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, parse_dates=["timestamps"])
        frame = frame.sort_values("timestamps").drop_duplicates("timestamps").set_index("timestamps")
        required = {"open", "close", "volume"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path}: missing {required - set(frame.columns)}")
        if not np.isfinite(frame[["open", "close", "volume"]].to_numpy(dtype=float)).all():
            raise ValueError(f"{path}: non-finite market fields")
        market[code] = frame[["open", "close", "volume"]]
    return market


def compute_signals(label, cache_path, critics, evaluator):
    with cache_path.open("rb") as handle:
        cache = pickle.load(handle)
    metadata = cache["metadata"]
    if metadata.get("sample_count") != 16 or metadata.get("lookback") != 90 or metadata.get("pred_len") != 10:
        raise ValueError(f"{label}: cache metadata is not frozen N=16 HS300 protocol")
    rows = []
    codes = set()
    for seed, critic in critics.items():
        for window_index, (window, path_array) in enumerate(zip(cache["windows"], cache["candidates"])):
            code, x_df, x_ts, y_ts, y_true = window
            code = str(code)
            codes.add(code)
            hist_norm, _, mean, scale = evaluator.normalized_history_and_target(x_df, y_true, clip=5.0)
            paths = np.asarray(path_array, dtype=np.float32)
            if paths.shape != (16, 10, 6):
                raise ValueError(f"{label}/{code}/{window_index}: {paths.shape}")
            scores = np.asarray(critic.score(paths, hist_norm), dtype=float)
            predictions = evaluator.aggregate_paths(paths, scores, 16)
            last_close = float(x_df["close"].iloc[-1])
            origin = pd.Timestamp(x_ts.iloc[-1] if hasattr(x_ts, "iloc") else x_ts[-1]).normalize()
            entry = pd.Timestamp(y_ts.iloc[0] if hasattr(y_ts, "iloc") else y_ts[0]).normalize()
            horizon_end = pd.Timestamp(y_ts.iloc[-1] if hasattr(y_ts, "iloc") else y_ts[-1]).normalize()
            for method in METHODS:
                pred = predictions[method]
                raw = pred.astype(np.float64) * scale.astype(np.float64) + mean.astype(np.float64)
                predicted_close = float(np.mean(raw[:, 3]))
                rows.append({
                    "period": label,
                    "seed": int(seed),
                    "code": code,
                    "window_index": int(window_index),
                    "origin": origin,
                    "entry": entry,
                    "horizon_end": horizon_end,
                    "candidate_count": 16,
                    "method": method,
                    "last_close": last_close,
                    "predicted_close_mean": predicted_close,
                    "signal": predicted_close / last_close - 1.0,
                })
    return pd.DataFrame(rows), codes, metadata


def _trade_cost(notional: float, rate: float) -> float:
    return max(abs(float(notional)) * rate, MIN_COST)


def simulate_portfolio(market, snapshots, benchmark, k=TOPK, drop=N_DROP, min_hold=HOLD_THRESH):
    codes = sorted(market)
    calendar = pd.DatetimeIndex(sorted(set().union(*(set(frame.index) for frame in market.values()))))
    last_horizon = pd.to_datetime(snapshots["horizon_end"]).max()
    calendar = calendar[calendar <= last_horizon]
    opens = pd.DataFrame({code: frame["open"] for code, frame in market.items()}).reindex(calendar)
    closes = pd.DataFrame({code: frame["close"] for code, frame in market.items()}).reindex(calendar)
    volumes = pd.DataFrame({code: frame["volume"] for code, frame in market.items()}).reindex(calendar)
    valid = opens.gt(0) & volumes.gt(0) & opens.notna() & closes.gt(0) & closes.notna()
    marks = closes.ffill()
    events = {
        pd.Timestamp(origin): {
            str(row.code): float(row.signal)
            for row in group.itertuples()
            if np.isfinite(float(row.signal))
        }
        for origin, group in snapshots.groupby("origin")
    }
    events = {date: values for date, values in events.items() if date in calendar and values}
    if not events:
        raise ValueError("no dated cross-sectional signals")
    first_origin = min(events)
    first_origin_idx = int(calendar.get_indexer([first_origin])[0])
    if first_origin_idx < 0 or first_origin_idx + 1 >= len(calendar):
        raise ValueError(f"first origin has no next trading day: {first_origin}")

    holdings: dict[str, float] = {}
    acquired: dict[str, int] = {}
    cash = ACCOUNT
    fees_paid = 0.0
    daily = []
    trades = []
    audits = []

    for day_idx in range(first_origin_idx + 1, len(calendar)):
        day = calendar[day_idx]
        prev = calendar[day_idx - 1]
        signals = events.get(prev)
        if signals is not None:
            ranked = sorted(
                (code for code in signals if code in market and np.isfinite(signals[code])),
                key=lambda code: (-float(signals[code]), code),
            )
            wish = set(ranked[:k])
            if not holdings:
                target = set(ranked[:k])
            else:
                target = set(holdings)
                removable = sorted(
                    (
                        code
                        for code in holdings
                        if code not in wish and day_idx - acquired[code] >= min_hold
                    ),
                    key=lambda code: (-ranked.index(code) if code in ranked else 1, code),
                )
                removable = removable[:drop]
                for code in removable:
                    target.discard(code)
                for code in ranked:
                    if len(target) >= k:
                        break
                    target.add(code)

            open_marks = {
                code: (float(opens.at[day, code]) if bool(valid.at[day, code]) else float(marks.at[prev, code]))
                for code in holdings
                if np.isfinite(marks.at[prev, code])
            }
            nav_before = cash + sum(holdings[code] * open_marks[code] for code in open_marks)
            if not np.isfinite(nav_before) or nav_before <= 0:
                raise ValueError(f"invalid pre-trade NAV on {day}")
            target_size = max(len(target), 1)
            target_value = nav_before / target_size
            desired = {}
            for code in sorted(set(holdings) | target):
                old = holdings.get(code, 0.0)
                if not bool(valid.at[day, code]):
                    desired[code] = old
                    continue
                desired[code] = target_value / float(opens.at[day, code]) if code in target else 0.0
                if old > 0 and day_idx - acquired[code] < min_hold:
                    desired[code] = max(desired[code], old)

            # Sell first so the buys can use the released cash.
            for code in sorted(list(holdings)):
                old = holdings[code]
                qty = max(0.0, old - desired.get(code, 0.0))
                if qty <= 1e-15 or not bool(valid.at[day, code]):
                    continue
                holding_days = day_idx - acquired[code]
                price = float(opens.at[day, code])
                notional = qty * price
                fee = _trade_cost(notional, CLOSE_COST)
                cash += notional - fee
                fees_paid += fee
                holdings[code] -= qty
                if holdings[code] <= 1e-12:
                    del holdings[code]
                    acquired.pop(code, None)
                trades.append({
                    "date": str(day.date()),
                    "signal_date": str(prev.date()),
                    "code": code,
                    "side": "sell",
                    "quantity": qty,
                    "price": price,
                    "notional": notional,
                    "fee": fee,
                    "holding_days": holding_days,
                })

            wants = {
                code: max(0.0, desired.get(code, 0.0) - holdings.get(code, 0.0))
                for code in desired
                if desired.get(code, 0.0) > holdings.get(code, 0.0)
                and bool(valid.at[day, code])
            }
            total_notional = sum(qty * float(opens.at[day, code]) for code, qty in wants.items())
            if total_notional > 0:
                def buy_cost(scale):
                    return sum(
                        scale * qty * float(opens.at[day, code])
                        + _trade_cost(scale * qty * float(opens.at[day, code]), OPEN_COST)
                        for code, qty in wants.items()
                    )

                available_cash = max(cash, 0.0)
                if buy_cost(1.0) <= available_cash:
                    scale = 1.0
                else:
                    # min_cost makes the cash requirement piecewise linear;
                    # solve against the actual per-trade cost function so the
                    # account never goes negative from a sizing approximation.
                    low, high = 0.0, 1.0
                    for _ in range(60):
                        mid = (low + high) / 2.0
                        if buy_cost(mid) <= available_cash:
                            low = mid
                        else:
                            high = mid
                    scale = low
            else:
                scale = 0.0
            for code in sorted(wants):
                qty = wants[code] * scale
                if qty <= 1e-15:
                    continue
                price = float(opens.at[day, code])
                notional = qty * price
                fee = _trade_cost(notional, OPEN_COST)
                cash -= notional + fee
                fees_paid += fee
                holdings[code] = holdings.get(code, 0.0) + qty
                acquired.setdefault(code, day_idx)
                trades.append({
                    "date": str(day.date()),
                    "signal_date": str(prev.date()),
                    "code": code,
                    "side": "buy",
                    "quantity": qty,
                    "price": price,
                    "notional": notional,
                    "fee": fee,
                    "holding_days": 0,
                })
            if cash < -1e-5:
                raise AssertionError(f"negative cash after trade on {day}: {cash}")
            audits.append({
                "date": str(day.date()),
                "signal_date": str(prev.date()),
                "signals": len(signals),
                "target_size": len(target),
                "holdings": len(holdings),
                "turnover_notional": float(sum(t["notional"] for t in trades if t["date"] == str(day.date()))),
            })

        equity = cash + sum(
            quantity * float(marks.at[day, code])
            for code, quantity in holdings.items()
            if np.isfinite(marks.at[day, code])
        )
        if not np.isfinite(equity) or equity <= 0:
            raise ValueError(f"invalid equity on {day}: {equity}")
        daily.append({
            "date": day,
            "equity": equity,
            "gross_equity": equity + fees_paid,
            "cash": cash,
            "holdings": len(holdings),
            "fees_paid": fees_paid,
        })

    daily_frame = pd.DataFrame(daily)
    if daily_frame.empty:
        raise ValueError("portfolio produced no daily rows")
    daily_frame["date"] = pd.to_datetime(daily_frame["date"])
    bench = benchmark.reindex(daily_frame["date"])["close"]
    if bench.isna().any():
        missing = [str(x.date()) for x in bench[bench.isna()].index[:5]]
        raise ValueError(f"benchmark missing portfolio dates, examples={missing}")
    daily_frame["benchmark"] = bench.to_numpy(dtype=float) / float(bench.iloc[0])
    strategy_returns = daily_frame["equity"].pct_change().dropna().to_numpy(dtype=float)
    benchmark_returns = daily_frame["benchmark"].pct_change().dropna().to_numpy(dtype=float)
    active = strategy_returns - benchmark_returns
    valid_days = int(len(active))
    mean_active = float(np.mean(active)) if valid_days else float("nan")
    active_std = float(np.std(active, ddof=1)) if valid_days > 1 else float("nan")
    aer = 252.0 * mean_active if valid_days else float("nan")
    ir = float(np.sqrt(252.0) * mean_active / active_std) if np.isfinite(active_std) and active_std > 0 else float("nan")
    start_equity = float(daily_frame["equity"].iloc[0])
    end_equity = float(daily_frame["equity"].iloc[-1])
    gross_start = float(daily_frame["gross_equity"].iloc[0])
    gross_end = float(daily_frame["gross_equity"].iloc[-1])
    annual_return = (end_equity / start_equity) ** (252.0 / valid_days) - 1.0 if valid_days else float("nan")
    mdd = float(np.min(daily_frame["equity"] / daily_frame["equity"].cummax() - 1.0))
    metrics = {
        "aer": aer,
        "ir": ir,
        "annual_return": annual_return,
        "mdd": mdd,
        "total_return_net": end_equity / start_equity - 1.0,
        "total_return_gross": gross_end / gross_start - 1.0,
        "mean_active_daily": mean_active,
        "std_active_daily": active_std,
        "valid_days": valid_days,
        "event_count": len(audits),
        "final_holdings": int(daily_frame["holdings"].iloc[-1]),
        "average_holdings": float(daily_frame["holdings"].mean()),
        "fees_total": float(daily_frame["fees_paid"].iloc[-1]),
        "start_date": str(daily_frame["date"].iloc[0].date()),
        "end_date": str(daily_frame["date"].iloc[-1].date()),
    }
    daily_frame["strategy_return"] = daily_frame["equity"].pct_change()
    daily_frame["benchmark_return"] = daily_frame["benchmark"].pct_change()
    daily_frame["active_return"] = daily_frame["strategy_return"] - daily_frame["benchmark_return"]
    return metrics, daily_frame, pd.DataFrame(trades), pd.DataFrame(audits)


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    metric_columns = ["aer", "ir", "annual_return", "mdd", "total_return_net", "total_return_gross", "fees_total"]
    grouped = per_seed.groupby(["period", "method"], as_index=False)[metric_columns].agg(["mean", "std"])
    grouped.columns = [
        "_".join(str(part) for part in column if str(part) not in {"", "None"}).rstrip("_")
        if isinstance(column, tuple) else str(column)
        for column in grouped.columns
    ]
    return grouped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    required = list(CACHE_PATHS.values()) + list(CHECKPOINTS.values()) + [FROZEN_PROTOCOL]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(missing))

    evaluator = load_forecast_module()
    engine = evaluator.load_numpy_engine()
    critics = {int(seed): engine.load_dual_calibrated(path) for seed, path in CHECKPOINTS.items()}
    benchmark, benchmark_metadata = load_benchmark()
    per_seed_rows = []
    signal_frames = []
    daily_frames = []
    trade_frames = []
    audit_frames = []

    for label, cache_path in CACHE_PATHS.items():
        signal_frame, codes, cache_metadata = compute_signals(label, cache_path, critics, evaluator)
        signal_frames.append(signal_frame)
        market = load_market(label, codes)
        for seed in sorted(signal_frame["seed"].unique()):
            for method in METHODS:
                part = signal_frame[(signal_frame["seed"] == seed) & (signal_frame["method"] == method)].copy()
                metrics, daily, trades, audits = simulate_portfolio(
                    market,
                    part[["origin", "code", "signal", "horizon_end"]],
                    benchmark,
                )
                metrics.update({"period": label, "seed": int(seed), "method": method, "candidate_count": 16})
                per_seed_rows.append(metrics)
                daily["period"] = label
                daily["seed"] = int(seed)
                daily["method"] = method
                daily_frames.append(daily)
                if not trades.empty:
                    trades["period"] = label
                    trades["seed"] = int(seed)
                    trades["method"] = method
                    trade_frames.append(trades)
                if not audits.empty:
                    audits["period"] = label
                    audits["seed"] = int(seed)
                    audits["method"] = method
                    audit_frames.append(audits)
                print(f"simulated investment {label} seed={seed} method={method}", flush=True)

    per_seed = pd.DataFrame(per_seed_rows)
    signals = pd.concat(signal_frames, ignore_index=True)
    daily = pd.concat(daily_frames, ignore_index=True)
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    audits = pd.concat(audit_frames, ignore_index=True) if audit_frames else pd.DataFrame()
    per_seed.to_csv(args.output_dir / "investment_per_seed.csv", index=False, float_format="%.10g")
    signals.to_csv(args.output_dir / "investment_signals.csv", index=False, float_format="%.10g")
    daily.to_csv(args.output_dir / "investment_daily.csv", index=False, float_format="%.10g")
    trades.to_csv(args.output_dir / "investment_trades.csv", index=False, float_format="%.10g")
    audits.to_csv(args.output_dir / "investment_events.csv", index=False, float_format="%.10g")
    summary = summarize(per_seed)
    summary.to_csv(args.output_dir / "investment_summary.csv", index=False, float_format="%.10g")

    protocol = {
        "frozen_protocol_path": str(FROZEN_PROTOCOL.relative_to(ROOT)),
        "frozen_protocol_sha256": sha256(FROZEN_PROTOCOL),
        "candidate_count": 16,
        "methods": list(METHODS),
        "stock_pool": "HS300 list data/references/hs300_2026H1.txt, current-period valid files only",
        "signal": "mean of inverse-transformed predicted close over 10 horizon steps / last historical close - 1",
        "portfolio": {
            "topk": TOPK,
            "n_drop": N_DROP,
            "hold_thresh": HOLD_THRESH,
            "long_only": True,
            "weighting": "equal-weight target; fractional shares for executable accounting",
            "rebalance": "each fixed cache origin; execute at next available market-day open",
        },
        "costs": {
            "open_cost": OPEN_COST,
            "close_cost": CLOSE_COST,
            "min_cost": MIN_COST,
            "account": ACCOUNT,
        },
        "benchmark": benchmark_metadata,
        "aer": "252 * mean(daily strategy return - daily SH000300 return)",
        "ir": "sqrt(252) * mean(active daily return) / sample_std(active daily return, ddof=1)",
        "annual_return": "(equity_end/equity_start) ** (252/valid_days) - 1",
        "mdd": "min(equity / cumulative_peak - 1)",
        "metric_frequency": "daily marked-to-close portfolio; first entry day is baseline and subsequent valid days are used for AER/IR",
        "checkpoint_hashes": {
            str(seed): {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for seed, path in CHECKPOINTS.items()
        },
        "cache_hashes": {
            label: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for label, path in CACHE_PATHS.items()
        },
    }
    (args.output_dir / "investment_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# HS300 Investment Simulation",
        "",
        "本报告使用冻结的三份 N=16 候选缓存、五个已有 Momus checkpoint 和 HS300 Top-50/drop-5/hold-5 long-only 等权协议；没有重训或按回测结果选模型。",
        "",
        f"Benchmark: SH000300，来源 `{benchmark_metadata.get('source')}`。Tushare token 在运行时被拒绝，因此使用已记录的 Tencent 公共历史 K 线回退，未用股票池等权组合替代 benchmark。",
        "",
        "AER 与 IR 在每日收盘标记收益上计算，主动收益为策略日收益减 SH000300 日收益；IR 的 `std` 为样本标准差（ddof=1）。",
        "",
        "| period | method | AER | IR | annual return | MDD | net total | gross total | fees | valid days |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in per_seed.groupby(["period", "method"], as_index=False)[
        ["aer", "ir", "annual_return", "mdd", "total_return_net", "total_return_gross", "fees_total", "valid_days"]
    ].mean().itertuples(index=False):
        lines.append(
            f"| {row.period} | {row.method} | {row.aer:.6f} | {row.ir:.6f} | {row.annual_return:.6f} | {row.mdd:.6f} | {row.total_return_net:.6f} | {row.total_return_gross:.6f} | {row.fees_total:.2f} | {row.valid_days:.0f} |"
        )
    lines.extend([
        "",
        "每个 period/method 的五 seed 明细见 `investment_per_seed.csv`，均值和标准差见 `investment_summary.csv`。",
        "",
        "Benchmark 原始响应解析后的文件和请求元数据见 `benchmark_sh000300.csv` 与 `benchmark_sh000300_manifest.json`。",
    ])
    (args.output_dir / "INVESTMENT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"output={args.output_dir}")
    print(per_seed.groupby(["period", "method"])[["aer", "ir", "annual_return", "mdd"]].mean().to_string(index=False))


if __name__ == "__main__":
    main()
