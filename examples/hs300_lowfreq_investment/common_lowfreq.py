"""Frozen schedule and shared inputs for the supplementary low-frequency run."""
from pathlib import Path
import json
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE_CODE = ROOT / "examples/hs300_daily_investment"
sys.path.insert(0, str(BASE_CODE))
from common import OUT as BASE, DATA, CHECKPOINTS, digest, identity, write_json

OUT = ROOT / "output/hs300_lowfreq_investment_20260908"
PERIODS = {"2025": ("2025-01-01", "2025-12-31"), "2026H1": ("2026-01-01", "2026-06-30")}


def calendar():
    return pd.DatetimeIndex(pd.read_csv(DATA / "market/SH000300.csv", parse_dates=["date"]).date)


def schedule(cal):
    rows = []
    for period, (start, end) in PERIODS.items():
        days = cal[(cal >= start) & (cal <= end)]
        for day in days[::10]:
            i = cal.get_loc(day)
            if i < 1 or len(cal[i:i+10]) != 10:
                raise ValueError("Incomplete history/calendar horizon")
            rows.append(dict(period=period, execution=str(day.date()), origin=str(cal[i-1].date()),
                             label_end=str(cal[i+9].date())))
    return rows


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    cal = calendar()
    gate = json.loads((BASE / "data_ready.json").read_text(encoding="utf-8"))
    if not gate["generation_ready"]:
        raise ValueError("Original data audit did not pass")
    for relative, sha in gate["input_sha256"].items():
        if digest(BASE / relative) != sha:
            raise ValueError(f"Changed input: {relative}")
    members = pd.read_csv(BASE / "membership.csv", parse_dates=["start_date", "end_date"])
    rows = schedule(cal)
    universes = {}
    for row in rows:
        day = pd.Timestamp(row["execution"])
        codes = sorted(members.loc[(members.start_date <= day) & (members.end_date >= day), "instrument"])
        if len(codes) != 300 or len(set(codes)) != 300:
            raise ValueError(f"Incomplete universe on {day}")
        universes[row["execution"]] = codes
    protocol = dict(version=1, periods=PERIODS, schedule=rows, universes=universes,
        interval_market_days=10, reset_schedule_each_period=True, separate_initial_accounts=True,
        data_identity=identity(gate), topk=50, n_drop=5, hold_thresh_market_days=5,
        retained_holdings_rebalanced=False, risk_degree=.95, cash=100_000_000,
        fees_each_side=.0015, minimum_fee=5, physical_trade_unit=100,
        signal="mean next 10 predicted adjusted closes / origin adjusted close - 1",
        trade_timing="origin close information; next market open; scheduled execution dates only",
        absent_forecasts="no orders between scheduled dates; no stale-score forward fill",
        membership_exit="existing nonmembers remain valued; original Qlib dropout ranks missing held names last on the next scheduled decision; no forced immediate liquidation",
        candidates=16, lookback=90, horizon=10, momus_seeds=list(range(5)),
        candidate_repeats=1, comparison="shared candidate arithmetic mean vs five fixed Momus weighted models",
        statistics="per-period descriptive results; paired circular daily blocks 10/20 trading days, 5000 draws, seed 20260908; no independent-generation or long-run profitability claim",
        incomplete_final_holding_interval="mark to period-end close without forced liquidation; full 10-day signal labels may extend beyond period end",
        limitations=gate["limits"] + ["supplementary low-frequency strategy, not original Kronos daily replication", "37 decision dates total; limited market-time evidence", "existing Momus model selection and frozen candidates; not a newly preregistered confirmatory trial"])
    path = OUT / "protocol.json"
    if path.exists() and identity(json.loads(path.read_text(encoding="utf-8"))) != identity(protocol):
        raise ValueError("Frozen protocol changed")
    write_json(path, protocol)
    pd.DataFrame(rows).to_csv(OUT / "schedule.csv", index=False)
    return protocol


def protect(stage):
    original = json.loads((BASE / "protected_inputs.json").read_text(encoding="utf-8"))
    # Also preserve the paused 20-date daily candidate cache and its manifest.
    paths = list(original)
    paths += [str(p) for p in (BASE / "candidates").glob("*") if p.suffix in (".npz", ".json")]
    paths += [str(BASE / "generation_manifest.json"), str(ROOT / "examples/hs300_daily_investment/generate.py")]
    current = {p: digest(p) for p in paths}
    target = OUT / "protected_inputs.json"
    if target.exists():
        before = json.loads(target.read_text(encoding="utf-8"))
        if before != current:
            raise ValueError("Protected original input or paused cache changed")
    else:
        if any(current[p] != sha for p, sha in original.items()):
            raise ValueError("Original audit protection no longer matches")
        write_json(target, current)
    write_json(OUT / f"protection_{stage}.json", dict(status="PASSED", files=len(current)))
    return len(current)
