"""Read-only, cross-source price and factor/reference audit; never filters stocks."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .common import ROOT, OUT, DATA, write_json
except ImportError:
    from common import ROOT, OUT, DATA, write_json

PERIODS = ("2024", "2025", "2026H1")
PRICE_TOLERANCE = .005 + 1e-8


def _number(value):
    return float(value) if np.isfinite(value) else None


def _read_frame(payload, date_column):
    frame = pd.read_csv(io.BytesIO(payload))
    frame[date_column] = pd.to_datetime(frame[date_column], errors="raise")
    if frame[date_column].duplicated().any():
        raise ValueError(f"duplicate {date_column}")
    return frame.set_index(date_column).sort_index()


def compare(old, new, field, *, instrument, period, old_multiplier=1., tolerance=PRICE_TOLERANCE,
            activity=False):
    """Align exact dates, retain every price anomaly and top activity examples."""
    dates = old.index.intersection(new.index)
    a, b = old.reindex(dates).to_numpy(float) * old_multiplier, new.reindex(dates).to_numpy(float)
    good = np.isfinite(a) & np.isfinite(b)
    difference = np.abs(b - a)
    bad = ~good | (difference > tolerance)
    exact = good & (difference <= 1e-8)
    relative = np.divide(difference, np.abs(a), out=np.full(len(a), np.nan), where=good & (a != 0))
    finite_relative = relative[np.isfinite(relative)]
    ratios = np.divide(b, a, out=np.full(len(a), np.nan), where=good & (a != 0))
    ratios = ratios[np.isfinite(ratios)]
    summary = {
        "instrument": instrument, "period": period, "field": field,
        "kind": "activity" if activity else "price", "old_multiplier": old_multiplier,
        "absolute_tolerance": tolerance, "old_rows": len(old), "new_rows": len(new),
        "shared_rows": len(dates), "old_dates_missing_new": len(old.index.difference(new.index)),
        "finite_pairs": int(good.sum()), "exact_pairs": int(exact.sum()),
        "within_tolerance_pairs": int((good & ~bad).sum()), "mismatched_pairs": int(bad.sum()),
        "max_absolute_difference": _number(np.max(difference[good])) if good.any() else None,
        "median_relative_difference": _number(np.median(finite_relative)) if len(finite_relative) else None,
        "p99_relative_difference": _number(np.quantile(finite_relative, .99)) if len(finite_relative) else None,
        "median_new_old_ratio": _number(np.median(ratios)) if len(ratios) else None,
    }
    indices = np.flatnonzero(bad)
    if activity and len(indices) > 3:
        indices = indices[np.argsort(np.nan_to_num(difference[indices], nan=np.inf))[-3:]]
    anomalies = [{"kind": summary["kind"], "instrument": instrument, "period": period,
                  "date": str(dates[i].date()), "field": field, "old_converted": _number(a[i]),
                  "new_value": _number(b[i]), "absolute_difference": _number(difference[i]),
                  "tolerance": tolerance, "listing": "largest_three_examples" if activity else "complete"}
                 for i in indices]
    return summary, anomalies


def factor_checks(instrument, frame):
    """Check ex-reference against factor ratio; derived references are not independent evidence."""
    previous_factor = frame.factor.shift(1)
    ratio = frame.factor / previous_factor
    expected = frame.previous_observed_close / ratio
    rounded = np.floor(expected * 100 + .5) / 100
    changed = ratio.notna() & (ratio - 1).abs().gt(1e-10)
    discrepancy = (frame.reference_close - expected).abs()
    derived = frame.get("factor_derived_reference", pd.Series(0, index=frame.index)).astype(bool)
    explicit = frame.explicit_reference.astype(bool)
    # Include unexpected references even when no factor change was detected.
    event = changed | explicit | derived | (discrepancy > PRICE_TOLERANCE)
    rows = []
    for date in frame.index[event]:
        independent = bool(explicit.loc[date] and not derived.loc[date])
        row = {"instrument": instrument, "date": str(date.date()),
               "previous_observed_close": _number(frame.loc[date, "previous_observed_close"]),
               "reference_close": _number(frame.loc[date, "reference_close"]),
               "previous_factor": _number(previous_factor.loc[date]), "factor": _number(frame.loc[date, "factor"]),
               "factor_ratio": _number(ratio.loc[date]), "factor_changed": bool(changed.loc[date]),
               "explicit_reference": bool(explicit.loc[date]), "factor_derived_reference": bool(derived.loc[date]),
               "independent_reference": independent, "expected_reference_unrounded": _number(expected.loc[date]),
               "expected_reference_tick_rounded": _number(rounded.loc[date]),
               "absolute_difference": _number(discrepancy.loc[date]),
               "testable": bool(np.isfinite(discrepancy.loc[date])),
               "within_half_tick": bool(np.isfinite(discrepancy.loc[date]) and discrepancy.loc[date] <= PRICE_TOLERANCE),
               "interpretation": "independent_source_reference" if independent else "derived_or_ordinary_reference_consistency"}
        rows.append(row)
    return rows


def run():
    market = DATA / "market"
    inventory, not_ready, frames = [], [], {}
    for path in sorted(market.glob("*.csv")):
        meta_path = path.with_suffix(".json")
        try:
            before = meta_path.read_bytes()
            meta = json.loads(before)
            payload = path.read_bytes()
            checksum = hashlib.sha256(payload).hexdigest()
            if meta.get("transform_version") != 2 or checksum != meta.get("sha256") or before != meta_path.read_bytes():
                not_ready.append(path.stem)
                continue
            frames[path.stem] = _read_frame(payload, "date")
            inventory.append({"path": str(path.relative_to(ROOT)), "sha256": checksum, "transform_version": 2})
        except (FileNotFoundError, ValueError) as error:
            not_ready.append({"instrument": path.stem, "reason": str(error)})
    if not_ready:
        write_json(OUT / "cross_source_checks.json", {"status": "INPUT_TRANSFORM_IN_PROGRESS",
                   "ready_files": len(frames), "not_ready": not_ready, "generation_ready_decision": "not made"})
        print(json.dumps({"status": "INPUT_TRANSFORM_IN_PROGRESS", "ready": len(frames), "pending": len(not_ready)}))
        return 2
    summaries, anomalies, coverage, factor_rows = [], [], [], []
    for period in PERIODS:
        for path in sorted((ROOT / f"data/market_{period}_hs300").glob("*.csv")):
            number = path.stem
            instrument = ("SH" if number.startswith("6") else "SZ") + number
            if instrument not in frames:
                coverage.append({"instrument": instrument, "period": period, "kind": "missing_new_instrument"})
                continue
            payload = path.read_bytes()
            old = _read_frame(payload, "timestamps")
            new = frames[instrument]
            inventory.append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(payload).hexdigest()})
            for date in old.index.difference(new.index):
                coverage.append({"instrument": instrument, "period": period, "date": str(date.date()), "kind": "old_date_missing_new"})
            for field in ("open", "high", "low", "close", "volume", "amount"):
                multiplier = {"volume": 100., "amount": 1000.}.get(field, 1.)
                new_field = "raw_" + field if field != "amount" else "amount"
                summary, examples = compare(old[field], new[new_field], field, instrument=instrument, period=period,
                                            old_multiplier=multiplier, tolerance=1.000001 if field in ("volume", "amount") else PRICE_TOLERANCE,
                                            activity=field in ("volume", "amount"))
                summaries.append(summary)
                anomalies.extend(examples)
    benchmark_path = ROOT / "output/results/hs300_evidence_audit_20260908/sources/benchmark_raw.json"
    payload = benchmark_path.read_bytes()
    raw = json.loads(payload)["data"]["sh000300"]["day"]
    # Tencent day positions are date, OPEN, CLOSE, HIGH, LOW, VOLUME.
    benchmark = pd.DataFrame(raw, columns=["date", "open", "close", "high", "low", "volume"])
    benchmark["date"] = pd.to_datetime(benchmark.date)
    benchmark = benchmark.set_index("date").astype(float)
    inventory.append({"path": str(benchmark_path.relative_to(ROOT)), "sha256": hashlib.sha256(payload).hexdigest()})
    for field in ("open", "high", "low", "close"):
        summary, examples = compare(benchmark[field], frames["SH000300"]["raw_" + field], field,
                                    instrument="SH000300", period="tencent_benchmark")
        summaries.append(summary)
        anomalies.extend(examples)
    for instrument, frame in frames.items():
        factor_rows.extend(factor_checks(instrument, frame))
    summary_df = pd.DataFrame(summaries)
    factor_df = pd.DataFrame(factor_rows)
    summary_df.to_csv(OUT / "cross_source_checks.csv", index=False)
    pd.DataFrame(anomalies, columns=["kind", "instrument", "period", "date", "field", "old_converted",
                                    "new_value", "absolute_difference", "tolerance", "listing"]).to_csv(
                                        OUT / "cross_source_anomalies.csv", index=False)
    pd.DataFrame(coverage, columns=["instrument", "period", "date", "kind"]).to_csv(OUT / "cross_source_coverage.csv", index=False)
    factor_df.to_csv(OUT / "factor_reference_checks.csv", index=False)
    totals = summary_df.groupby(["period", "kind"])[["shared_rows", "exact_pairs", "within_tolerance_pairs", "mismatched_pairs"]].sum().reset_index().to_dict("records")
    bad_factor = factor_df.loc[factor_df.testable & ~factor_df.within_half_tick] if len(factor_df) else factor_df
    # Recheck input identities after the full audit; never certify a moving snapshot.
    changed_after = [item["path"] for item in inventory
                     if hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() != item["sha256"]]
    report = {"status": "COMPLETE" if not changed_after else "INPUT_CHANGED_DURING_CHECK",
              "generation_ready_decision": "not made; anomalies require review; no stocks filtered",
              "new_instruments": len(frames), "source_snapshot": inventory, "changed_during_check": changed_after,
              "price_tolerance": PRICE_TOLERANCE, "activity_tolerance_converted_units": 1.000001,
              "unit_conversions": {"old_tushare_volume_to_shares": 100, "old_tushare_amount_to_yuan": 1000,
                                   "new_sina_volume": "shares", "new_sina_amount": "yuan"},
              "comparison_totals": totals, "coverage_issues": len(coverage),
              "factor_reference_events": len(factor_rows),
              "factor_changes": int(factor_df.factor_changed.sum()) if len(factor_df) else 0,
              "independent_reference_events": int(factor_df.independent_reference.sum()) if len(factor_df) else 0,
              "untestable_reference_events": int((~factor_df.testable).sum()) if len(factor_df) else 0,
              "factor_reference_mismatches": len(bad_factor),
              "factor_reference_mismatch_records": bad_factor.to_dict("records"),
              "anomaly_listing": "all price mismatches; largest three differences per activity field/instrument/period",
              "large_price_moves": "not filtered or winsorized"}
    write_json(OUT / "cross_source_checks.json", report)
    print(json.dumps({k: report[k] for k in ("status", "new_instruments", "comparison_totals", "coverage_issues",
                                            "factor_reference_events", "factor_reference_mismatches")}, ensure_ascii=False))
    return 0 if not changed_after else 2


if __name__ == "__main__":
    raise SystemExit(run())
