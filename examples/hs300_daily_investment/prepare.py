"""Audit new market inputs and build the pinned Qlib research provider."""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from common import ROOT, OUT, DATA, COLS, digest, identity, write_json
from data import validate
from qlib_runner import FIELDS, write_provider


def bridge_mergers(frames, calendar, events):
    """Value non-electing public share entitlements; do not fabricate trading.

    The legacy instrument remains a separately identified claim in Qlib. At
    delisting it is valued as ratio * successor price. Both trade directions
    remain blocked until the verified new-share listing date. Synthetic units
    avoid loss/double-counting of value; registry fractional-share allocation
    is approximated proportionally and is disclosed.
    """
    bridged, records = {k: v.copy() for k, v in frames.items()}, []
    for event in events:
        old, new = event["absorbed_instrument"], event["successor_instrument"]
        ratio = event["exchange_ratio_new_raw_shares_per_old_raw_share"]
        last = pd.Timestamp(event["last_trading_date"])
        convert = pd.Timestamp(event["delisting_date"])
        release = pd.Timestamp(event["first_unrestricted_public_holder_sell_date"])
        if old not in frames or new not in frames:
            raise ValueError(f"Missing merger history {old}/{new}")
        old_frame, new_frame = frames[old], frames[new]
        if old_frame.index.max() != last:
            raise ValueError(f"Absorbed ticker last quotation differs from issuer: {old}")
        fold = float(old_frame.factor.iloc[-1])
        # Successor factor known at the conversion; subsequent ex-date changes
        # preserve total-return price representation without a second payment.
        fnew = float(new_frame.loc[:convert].factor.iloc[-1])
        dates = calendar[calendar >= convert]
        proxy = new_frame.reindex(calendar).copy()
        quoted = proxy.close.notna()
        for field in ["raw_open", "raw_high", "raw_low", "raw_close", "factor"]:
            proxy[field] = proxy[field].ffill()
        proxy = proxy.loc[dates].copy()
        proxy_factor = fold * ratio * proxy.factor / fnew
        for field in COLS[:4]:
            proxy[field] = proxy["raw_"+field] * proxy_factor
        proxy["volume"] = proxy.raw_volume.fillna(0) / proxy_factor
        proxy["factor"] = proxy_factor
        blocked = (dates < release) | (~quoted.loc[dates].to_numpy())
        proxy.loc[blocked, ["limit_buy", "limit_sell"]] = 1.
        proxy["economic_successor"] = new
        bridged[old] = pd.concat([old_frame, proxy]).sort_index()
        np.testing.assert_allclose(proxy.close, proxy.raw_close * fold * ratio * new_frame.factor.reindex(calendar).ffill().loc[dates] / fnew, rtol=1e-12)
        records.append({"old": old, "successor": new, "ratio": ratio, "conversion": str(convert.date()), "first_sell": str(release.date()), "old_synthetic_factor": fold, "successor_factor_at_conversion": fnew, "claim_rows": len(proxy), "policy": "separate legacy claim marked to known successor quote from delisting; no sale before listing; proportional residual-share approximation"})
    return bridged, records


def provider_frames(frames, calendar):
    result = {}
    for code, frame in frames.items():
        data = frame[list(FIELDS)].reindex(calendar).copy()
        # Retain only from the first actual quotation; later missing observations
        # are suspended/unavailable, with both directions forbidden.
        data = data.loc[frame.index.min():]
        missing = data.close.isna() | data.volume.fillna(0).le(0)
        data.loc[missing, ["limit_buy", "limit_sell"]] = 1.
        # Real quotes with zero volume may mark value, but cannot execute.
        data.loc[data.close.isna(), ["open", "high", "low", "factor", "volume"]] = np.nan
        result[code] = data
    return result


def main():
    status = json.loads((DATA / "collection_status.json").read_text(encoding="utf-8"))
    if not status["all_complete"] or status["failures"]:
        raise ValueError("Market collection incomplete")
    members = pd.read_csv(OUT / "membership.csv", parse_dates=["start_date", "end_date"])
    market = {}
    hashes = {"membership.csv": digest(OUT / "membership.csv"), "membership_events.json": digest(OUT / "membership_events.json"), "merger_events.json": digest(OUT / "merger_events.json")}
    audit = []
    for code in sorted({"SH000300", *members.instrument.unique()}):
        path = DATA / "market" / f"{code}.csv"
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        if digest(path) != meta["sha256"]:
            raise ValueError(f"Market hash mismatch: {code}")
        frame = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
        validate(frame, code)
        market[code] = frame
        hashes[str(path.relative_to(OUT))] = digest(path)
        audit.append({"instrument": code, "rows": len(frame), "start": str(frame.index.min().date()), "end": str(frame.index.max().date()), "zero_volume": int(frame.volume.eq(0).sum()), "factor_changes": meta["factor_changes"]})
    calendar = market["SH000300"].index
    coverage, exclusions = [], []
    for ix, day in enumerate(calendar):
        if day < pd.Timestamp("2024-01-01") or day > pd.Timestamp("2026-06-30"):
            continue
        origin = calendar[ix-1]
        codes = members.loc[(members.start_date <= day) & (members.end_date >= day), "instrument"]
        if len(codes) != 300 or codes.duplicated().any():
            raise ValueError(f"Membership does not equal 300: {day}")
        eligible, stale = 0, 0
        for code in codes:
            h = market[code].loc[:origin]; h = h[h.volume > 0]
            if len(h) < 90:
                exclusions.append({"date": str(day.date()), "instrument": code, "observations": len(h), "reason": "insufficient observable history"})
            else:
                eligible += 1
                stale += h.index[-1] < origin
        if eligible < 50:
            raise ValueError(f"Too few observable stocks: {day}")
        coverage.append({"execution": str(day.date()), "origin": str(origin.date()), "members": 300, "eligible_forecasts": eligible, "histories_last_trade_before_origin": int(stale)})
    # All additions must have been publicly announced before the first new
    # member's prediction origin. Do not leak future membership into signals.
    events = json.loads((OUT / "membership_events.json").read_text(encoding="utf-8"))
    for event in events:
        effective = pd.Timestamp(event["effective_date"])
        before = calendar[calendar < effective][-1]
        if pd.Timestamp(event["publish_date"]) > before:
            raise ValueError("Membership became known after signal origin")
    mergers = json.loads((OUT / "merger_events.json").read_text(encoding="utf-8"))
    bridged, bridges = bridge_mergers(market, calendar, mergers["events"])
    provider_input_identity = identity(hashes)
    code_hashes = {"prepare": digest(__file__), "qlib_runner": digest(ROOT / "examples/hs300_daily_investment/qlib_runner.py")}
    provider = OUT / ("qlib_provider_" + identity({"inputs": provider_input_identity, "code": code_hashes})[:12])
    if provider.exists():
        frozen = json.loads((provider / "manifest.json").read_text(encoding="utf-8"))
        for relative, sha in frozen["feature_sha256"].items():
            if digest(provider / relative) != sha:
                raise ValueError(f"Provider feature changed: {relative}")
    else:
        write_provider(provider_frames(bridged, calendar), provider, calendar)
    write_json(OUT / "provider_identity.json", {"path": provider.name, "input_identity": provider_input_identity, "code_hashes": code_hashes, "manifest_sha256": digest(provider / "manifest.json"), "calendar_sha256": digest(provider / "calendars/day.txt"), "instruments_sha256": digest(provider / "instruments/all.txt")})
    pd.DataFrame(audit).to_csv(OUT / "market_audit.csv", index=False)
    pd.DataFrame(coverage).to_csv(OUT / "daily_coverage.csv", index=False)
    pd.DataFrame(exclusions, columns=["date", "instrument", "observations", "reason"]).to_csv(OUT / "history_exclusions.csv", index=False)
    write_json(OUT / "merger_bridge_audit.json", bridges)
    cross = json.loads((OUT / "cross_source_checks.json").read_text(encoding="utf-8"))
    # Cross-source mismatches must be reviewed explicitly; arithmetic checks do
    # not by themselves certify external source accuracy.
    passed = (cross["status"] == "COMPLETE" and not cross["changed_during_check"] and cross["coverage_issues"] == 0 and cross["factor_reference_mismatches"] == 0 and all(x["mismatched_pairs"] == 0 for x in cross["comparison_totals"]))
    if not passed:
        raise ValueError("Cross-source checks require investigation before generation")
    write_json(OUT / "data_ready.json", {"generation_ready": True, "large_run_requires": "real-provider paired smoke test after two pilot days", "input_sha256": hashes, "market_instruments": len(market)-1, "execution_days": len(coverage), "min_forecasts_per_day": min(x["eligible_forecasts"] for x in coverage), "excluded_stock_dates": len(exclusions), "cross_source_status": cross["status"], "limits": ["uniform open +/-9.5% research trading band, not full statutory exchange/ST limits", "adjusted synthetic returns, not cash dividend payment/tax ledger", "legacy merger claims represented separately until sold; proportional fractional allocation", "membership archive not independently vendor-confirmed", "2024 overlaps foundation-model pretraining/development"]})
    print({"provider_created": True, "days": len(coverage), "exclusions": len(exclusions), "generation_ready": True, "large_run_requires_smoke": True}, flush=True)


if __name__ == "__main__":
    main()
