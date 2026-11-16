"""Pinned Qlib 0.9.7 adapter; prices/amounts use adjusted synthetic units.

The provider accepts already adjusted OHLC and volume, together with the factor
mapping a synthetic amount into physical shares.  Limit flags must be computed
from information available at execution (normally raw opening price/limit price).
No corporate action cash ledger or signal imputation is performed here.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FIELDS = ("open", "high", "low", "close", "volume", "factor", "limit_buy", "limit_sell")
BENCHMARK = "SH000300"


def _dates(values, label):
    dates = pd.DatetimeIndex(values)
    if dates.tz is not None or dates.hasnans or dates.has_duplicates:
        raise ValueError(f"{label}: dates must be unique, finite and timezone naive")
    if not dates.equals(dates.normalize()) or not dates.is_monotonic_increasing:
        raise ValueError(f"{label}: dates must be sorted daily timestamps")
    return dates


def validate_frames(frames):
    """Reject Qlib's silent close-price fallback and factor/lot-size downgrade."""
    if BENCHMARK not in frames:
        raise ValueError(f"Missing benchmark {BENCHMARK}")
    clean = {}
    for code, frame in sorted(frames.items()):
        if not re.fullmatch(r"(?:SH|SZ|BJ)\d{6}", code):
            raise ValueError(f"Invalid instrument {code!r}")
        if frame.empty or not set(FIELDS).issubset(frame.columns):
            raise ValueError(f"{code}: empty data or missing provider fields")
        dates = _dates(frame.index, code)
        x = frame.loc[:, list(FIELDS)].astype(float).copy()
        x.index = dates
        for flag in ("limit_buy", "limit_sell"):
            if not x[flag].isin([0.0, 1.0]).all():
                raise ValueError(f"{code}: {flag} must contain explicit 0/1 flags")
        quoted = x.close.notna()
        for field in ("open", "high", "low", "close", "factor"):
            v = x.loc[quoted, field]
            if not (np.isfinite(v) & (v > 0)).all():
                raise ValueError(f"{code}: invalid {field}; Qlib fallback prohibited")
        if ((x.loc[quoted, "high"] < x.loc[quoted, ["open", "close", "low"]].max(axis=1)) |
                (x.loc[quoted, "low"] > x.loc[quoted, ["open", "close", "high"]].min(axis=1))).any():
            raise ValueError(f"{code}: inconsistent OHLC")
        tradable = (x.limit_buy == 0) | (x.limit_sell == 0)
        if (~quoted & tradable).any():
            raise ValueError(f"{code}: missing close must forbid both directions")
        if not (np.isfinite(x.loc[tradable, "volume"]) & (x.loc[tradable, "volume"] > 0)).all():
            raise ValueError(f"{code}: tradable rows require positive finite volume")
        if (np.isinf(x.to_numpy())).any():
            raise ValueError(f"{code}: infinite provider values")
        clean[code] = x
    return clean


def write_provider(frames, provider_dir, calendar=None):
    """Write a new Qlib binary provider, never overwrite an existing directory.

    calendar is the market calendar spanning all rows.  One future sentinel date
    is appended for Qlib's right endpoint lookup, with no fabricated quotations.
    Pass the full market calendar when all stocks are suspended on a market day.
    """
    frames = validate_frames(frames)
    observed = pd.DatetimeIndex(sorted(set().union(*(set(x.index) for x in frames.values()))))
    cal = _dates(observed if calendar is None else calendar, "calendar")
    if not observed.isin(cal).all():
        raise ValueError("Provider dates fall outside calendar")
    if len(cal) < 2:
        raise ValueError("Provider needs at least two market dates")
    if not np.isfinite(frames[BENCHMARK].reindex(cal).close).all():
        raise ValueError("Benchmark needs a finite close on every market calendar date")
    provider_dir = Path(provider_dir).resolve()
    if provider_dir.exists() and any(provider_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite provider: {provider_dir}")
    provider_dir.mkdir(parents=True, exist_ok=True)
    (provider_dir / "calendars").mkdir(exist_ok=True)
    (provider_dir / "instruments").mkdir(exist_ok=True)
    # Sentinel marks the interval endpoint only; it is never an execution date.
    extended = cal.append(pd.DatetimeIndex([cal[-1] + pd.Timedelta(days=1)]))
    calendar_text = "\n".join(extended.strftime("%Y-%m-%d")) + "\n"
    for name in ("day.txt", "day_future.txt"):
        (provider_dir / "calendars" / name).write_text(calendar_text, encoding="utf-8")
    listing, hashes = [], {}
    for code, frame in frames.items():
        start, end = cal.get_indexer([frame.index[0], frame.index[-1]])
        x = frame.reindex(cal[start:end + 1])
        # Calendar gaps represent unavailable/suspended quotes, never trades.
        x.loc[x.close.isna(), ["limit_buy", "limit_sell"]] = 1.0
        x["change"] = x.close / x.close.ffill().shift(1) - 1.0
        directory = provider_dir / "features" / code.lower()
        directory.mkdir(parents=True, exist_ok=True)
        for field in (*FIELDS, "change"):
            payload = np.r_[float(start), x[field].to_numpy()].astype("<f4").tobytes()
            target = directory / f"{field}.day.bin"
            target.write_bytes(payload)
            hashes[str(target.relative_to(provider_dir))] = hashlib.sha256(payload).hexdigest()
        listing.append(f"{code}\t{frame.index[0]:%Y-%m-%d}\t{frame.index[-1]:%Y-%m-%d}")
    (provider_dir / "instruments" / "all.txt").write_text("\n".join(listing) + "\n", encoding="utf-8")
    manifest = {"format": 1, "qlib_version": "0.9.7", "instruments": list(frames),
                "calendar": cal.strftime("%Y-%m-%d").tolist(), "feature_sha256": hashes,
                "valuation": "adjusted synthetic units; no separate cash dividend ledger"}
    (provider_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


@dataclass
class BacktestResult:
    report: pd.DataFrame
    positions: dict
    trades: pd.DataFrame
    metadata: dict


def run_backtest(provider_dir, signal, start_time, end_time, *, topk=50, n_drop=5,
                 hold_thresh=5, account=100_000_000, expected_signals=None,
                 allow_missing_signal_dates=False):
    """Run the official strategy unchanged; record all submitted orders.

    Signals are dated at the end of their input window.  The strategy applies
    the t-1 shift. expected_signals maps origin date to its exact expected set;
    unexpected missing scores are never replaced with stale or invented values.
    """
    import qlib
    from qlib.backtest import backtest
    from qlib.backtest.executor import SimulatorExecutor
    from qlib.constant import REG_CN

    if qlib.__version__ != "0.9.7":
        raise RuntimeError(f"Expected Qlib 0.9.7, found {qlib.__version__}")
    provider_dir = Path(provider_dir).resolve()
    manifest = json.loads((provider_dir / "manifest.json").read_text(encoding="utf-8"))
    cal = pd.DatetimeIndex(manifest["calendar"])
    start_time, end_time = pd.Timestamp(start_time), pd.Timestamp(end_time)
    if start_time not in cal or end_time not in cal or start_time > end_time or start_time == cal[0]:
        raise ValueError("Execution bounds require market dates and a preceding signal date")
    if not isinstance(signal, pd.Series) or not isinstance(signal.index, pd.MultiIndex):
        raise ValueError("signal must be a MultiIndex Series")
    if set(signal.index.names) != {"datetime", "instrument"} or signal.index.has_duplicates:
        raise ValueError("signal index requires unique datetime/instrument pairs")
    signal = signal.reorder_levels(["datetime", "instrument"]).sort_index().astype(float)
    if not np.isfinite(signal.to_numpy()).all():
        raise ValueError("signal contains missing/nonfinite scores")
    signal_dates = pd.DatetimeIndex(signal.index.get_level_values("datetime"))
    if signal_dates.tz is not None or not signal_dates.equals(signal_dates.normalize()):
        raise ValueError("signal timestamps must be timezone-naive market dates")
    codes = set(signal.index.get_level_values("instrument"))
    if BENCHMARK in codes or not codes.issubset(manifest["instruments"]):
        raise ValueError("signal contains benchmark or unknown instruments")
    required = cal[cal.get_loc(start_time) - 1:cal.get_loc(end_time)]
    for date in required:
        if date not in signal_dates:
            if allow_missing_signal_dates:
                continue
            raise ValueError(f"Missing signal date {date:%Y-%m-%d}")
        actual = set(signal.xs(date, level="datetime").index)
        if len(actual) < topk:
            raise ValueError(f"Incomplete cross-section on {date:%Y-%m-%d}: {len(actual)} < {topk}")
        if expected_signals is not None:
            expected = expected_signals.get(date, expected_signals.get(str(date.date())))
            if expected is None or actual != set(expected):
                raise ValueError(f"Signal universe mismatch on {date:%Y-%m-%d}")
    # Avoid Redis dependency and process-based worker creation on Windows.
    qlib.init(provider_uri=str(provider_dir), region=REG_CN, expression_cache=None,
              dataset_cache=None, kernels=1, joblib_backend="threading")
    records = []

    class RecordingExecutor(SimulatorExecutor):
        def _collect_data(self, trade_decision, level=0):
            result, extra = super()._collect_data(trade_decision, level=level)
            for order, value, cost, price in result:
                records.append({"datetime": order.start_time, "instrument": order.stock_id,
                                "direction": "buy" if int(order.direction) == 1 else "sell",
                                "requested_amount": order.amount, "deal_amount": order.deal_amount,
                                "factor": order.factor, "raw_shares": order.deal_amount * order.factor
                                if order.factor is not None else np.nan,
                                "price": price, "value": value, "cost": cost})
            return result, extra

    executor = RecordingExecutor(time_per_step="day", generate_portfolio_metrics=True,
                                 trade_type="serial")
    strategy = {"class": "TopkDropoutStrategy", "module_path": "qlib.contrib.strategy.signal_strategy",
                "kwargs": {"signal": signal, "topk": topk, "n_drop": n_drop,
                           "hold_thresh": hold_thresh, "risk_degree": 0.95,
                           "method_buy": "top", "method_sell": "bottom", "only_tradable": False,
                           "forbid_all_trade_at_limit": True}}
    exchange = {"freq": "day", "codes": [x for x in manifest["instruments"] if x != BENCHMARK],
                "deal_price": "$open", "open_cost": 0.0015, "close_cost": 0.0015,
                "min_cost": 5, "trade_unit": 100, "impact_cost": 0.0,
                "limit_threshold": ("$limit_buy", "$limit_sell")}
    portfolios, _ = backtest(start_time=start_time, end_time=end_time, strategy=strategy,
                            executor=executor, account=account, benchmark=BENCHMARK,
                            exchange_kwargs=exchange)
    report, positions = portfolios["1day"]
    report = report.copy()
    report["net_return"] = report["return"] - report["cost"]
    report["net_nav"] = report["account"] / account
    report["holding_count"] = [len(positions[d].get_stock_list()) for d in report.index]
    # Includes initial cash NAV=1, essential for first-day losses.
    peak = np.maximum.accumulate(np.r_[1.0, report.net_nav.to_numpy()])[1:]
    report["net_drawdown"] = report.net_nav.to_numpy() / peak - 1.0
    np.testing.assert_allclose((1 + report.net_return).cumprod(), report.net_nav, rtol=1e-10, atol=1e-10)
    columns = ["datetime", "instrument", "direction", "requested_amount", "deal_amount", "factor",
               "raw_shares", "price", "value", "cost"]
    metadata = {"qlib_version": qlib.__version__, "topk": topk, "n_drop": n_drop,
                "hold_thresh": hold_thresh, "risk_degree": 0.95, "open_cost": 0.0015,
                "close_cost": 0.0015, "min_cost": 5, "trade_unit": 100,
                "signal_shift": 1, "rebalance_retained_holdings": False,
                "account": account, "benchmark": BENCHMARK,
                "execution_start": str(start_time.date()), "execution_end": str(end_time.date()),
                "dates_above_topk": report.index[report.holding_count > topk].strftime("%Y-%m-%d").tolist(),
                "valuation": manifest["valuation"]}
    return BacktestResult(report, positions, pd.DataFrame(records, columns=columns), metadata)
