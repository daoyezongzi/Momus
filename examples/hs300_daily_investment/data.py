"""Resumable public-source market collection; never changes existing market CSVs.

Sina's KLC decoder is obtained from AKShare's published source and frozen by hash.
Remote response scripts are parsed as data, never evaluated. The audited decoder
only receives the encoded price string. Source responses remain available.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from common import ROOT, OUT, DATA, COLS, digest, write_json

NODE = Path("C:/Users/Soyo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe")
START, END = "2023-06-01", "2026-07-15"
TRANSFORM_VERSION = 2
SOURCE = "https://raw.githubusercontent.com/akfamily/akshare/main/akshare/stock/cons.py"
HIST = "https://finance.sina.com.cn/realstock/company/{}/hisdata_klc2/klc_kl.js"
FACTOR = "https://finance.sina.com.cn/realstock/company/{}/hfq.js"


def fetch_bytes(url, path):
    path = Path(path)
    if path.exists():
        return path.read_bytes()
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=(10, 30), headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            payload = response.content
            if not payload:
                raise ValueError("empty HTTP response")
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(path.suffix + ".tmp")
            temp.write_bytes(payload)
            temp.replace(path)
            return payload
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(1 + attempt)


def prepare_decoder():
    folder = DATA / "sources"
    source_path = folder / "akshare_sina_cons.py"
    source = fetch_bytes(SOURCE, source_path).decode("utf-8")
    decoder = None
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "hk_js_decode" for t in node.targets):
            decoder = ast.literal_eval(node.value)
    if not isinstance(decoder, str):
        raise ValueError("AKShare decoder assignment absent")
    path = folder / "decode_sina.cjs"
    code = decoder + '\nconst fs=require("fs"); const x=fs.readFileSync(0,"utf8"); process.stdout.write(JSON.stringify(d(JSON.parse(x))));\n'
    if path.exists() and path.read_text(encoding="utf-8") != code:
        raise ValueError("frozen decoder differs")
    path.write_text(code, encoding="utf-8")
    write_json(folder / "decoder_manifest.json", {"source": SOURCE, "source_sha256": digest(source_path), "decoder_sha256": digest(path), "license": "AKShare MIT; https://github.com/akfamily/akshare/blob/main/LICENSE"})
    return path


def decode_history(payload, decoder):
    text = payload.decode("utf-8-sig")
    match = re.search(r'=\s*("[^"\r\n]*")', text)
    if match is None:
        raise ValueError("encoded KLC string absent")
    # JSON literal only; never run the remote JavaScript assignment.
    encoded = json.loads(match.group(1))
    result = subprocess.run([str(NODE), str(decoder)], input=json.dumps(encoded), text=True, capture_output=True, timeout=30, check=True)
    records = json.loads(result.stdout)
    if not isinstance(records, list) or not records:
        raise ValueError("empty decoded history")
    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
    if df.date.duplicated().any():
        raise ValueError("duplicate source dates")
    df = df.sort_values("date").set_index("date")
    for name in COLS:
        df[name] = pd.to_numeric(df[name], errors="raise")
    # Explicit prevclose on ex-dates is the source's price reference; otherwise
    # previous observed raw close. Preserve both for inspection.
    df["previous_observed_close"] = df.close.shift(1)
    reported = pd.to_numeric(df.get("prevclose", pd.Series(np.nan, index=df.index)), errors="raise")
    df["reference_close"] = reported.fillna(df.previous_observed_close)
    df["explicit_reference"] = reported.notna().astype(int)
    return df.loc[START:END].copy()


def parse_factors(payload):
    text = payload.decode("utf-8-sig")
    value = text[text.index("=") + 1:].strip().rstrip(";")
    # Source adds another harmless JS assignment after the JSON object.
    # Decode exactly the first JSON value; do not execute trailing script.
    obj, _ = json.JSONDecoder().raw_decode(value)
    df = pd.DataFrame(obj["data"]).rename(columns={"d": "date", "f": "factor"})
    if not {"date", "factor"}.issubset(df.columns) or df.empty:
        raise ValueError("missing adjustment factor history")
    df["date"] = pd.to_datetime(df.date)
    df["factor"] = pd.to_numeric(df.factor, errors="raise")
    if df.date.duplicated().any() or not np.isfinite(df.factor).all() or (df.factor <= 0).any():
        raise ValueError("invalid factor history")
    return df.sort_values("date").set_index("date")[["factor"]]


def adjusted_frame(raw, factors):
    """Multiplicative adjusted synthetic units; no separate cash-dividend ledger."""
    result = raw.copy()
    aligned = factors.reindex(factors.index.union(raw.index)).sort_index().ffill().reindex(raw.index).factor
    if aligned.isna().any():
        raise ValueError("factor unknown at beginning; do not backfill from future")
    # Scale constant known at the first retained observation, not at sample end.
    factor = aligned / aligned.iloc[0]
    for name in COLS[:4]:
        result["raw_" + name] = raw[name]
        result[name] = raw[name] * factor
    result["raw_volume"] = raw.volume
    result["factor"] = factor
    result["volume"] = raw.volume / factor
    # Actual currency turnover invariant to synthetic share conversion.
    result["amount"] = raw.amount
    # Research execution constraint adapted from official demo's 9.5% band:
    # use the OPEN and ex-date reference known at that time, not today's CLOSE.
    # This is a uniform conservative band, NOT complete board/ST legal limits.
    reference = raw.reference_close.copy()
    ratio = aligned / aligned.shift(1)
    infer_reference = raw.explicit_reference.eq(0) & ratio.notna() & ratio.sub(1).abs().gt(1e-10)
    # Ex-dividend while suspended may not appear as an explicit KLC prevclose.
    # Factor ratio contains that historical adjustment; round reference to ticks.
    reference.loc[infer_reference] = np.floor(raw.previous_observed_close.loc[infer_reference] / ratio.loc[infer_reference] * 100 + .5) / 100
    result["reference_close"] = reference
    result["factor_derived_reference"] = infer_reference.astype(int)
    move = raw.open / reference - 1
    result["limit_buy"] = (move >= .095).astype(float)
    result["limit_sell"] = (move <= -.095).astype(float)
    if reference.isna().any():
        raise ValueError("missing opening price reference")
    return result


def validate(df, instrument):
    if df.empty:
        raise ValueError(f"{instrument}: no retained market data")
    if not np.isfinite(df[COLS + ["factor", "reference_close"]].to_numpy(float)).all():
        raise ValueError(f"{instrument}: nonfinite required fields")
    if (df[COLS[:4] + ["factor", "reference_close"]] <= 0).any().any():
        raise ValueError(f"{instrument}: nonpositive price/factor")
    if (df[["volume", "amount"]] < 0).any().any():
        raise ValueError(f"{instrument}: negative trading activity")
    invalid = (df.high + 1e-7 < df[["open", "close", "low"]].max(axis=1)) | (df.low - 1e-7 > df[["open", "close", "high"]].min(axis=1))
    if invalid.any():
        raise ValueError(f"{instrument}: invalid OHLC ordering")
    # Derived CSVs retain 12 significant digits; allow that serialization error.
    np.testing.assert_allclose(df.close / df.factor, df.raw_close, rtol=2e-11)
    np.testing.assert_allclose(df.volume * df.factor, df.raw_volume, rtol=2e-11)


def collect(instrument, decoder):
    instrument = instrument.upper()
    if re.fullmatch(r"(?:SH|SZ)\d{6}", instrument) is None:
        raise ValueError(instrument)
    destination = DATA / "market" / f"{instrument}.csv"
    meta_path = DATA / "market" / f"{instrument}.json"
    if destination.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if digest(destination) != meta["sha256"]:
            raise ValueError(f"changed saved data: {instrument}")
        if meta.get("transform_version") == TRANSFORM_VERSION:
            return meta
    symbol = instrument.lower()
    source_path = DATA / "raw" / f"{instrument}_klc.js"
    raw = decode_history(fetch_bytes(HIST.format(symbol), source_path), decoder)
    if instrument == "SH000300":
        factors = pd.DataFrame({"factor": [1.]}, index=pd.to_datetime(["1990-01-01"]))
        factor_meta = None
    else:
        factor_path = DATA / "raw" / f"{instrument}_hfq.js"
        factors = parse_factors(fetch_bytes(FACTOR.format(symbol), factor_path))
        factor_meta = {"url": FACTOR.format(symbol), "sha256": digest(factor_path)}
    df = adjusted_frame(raw, factors)
    validate(df, instrument)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, float_format="%.12g")
    meta = {"transform_version": TRANSFORM_VERSION, "instrument": instrument, "rows": len(df), "first": str(df.index.min().date()), "last": str(df.index.max().date()), "sha256": digest(destination), "source": HIST.format(symbol), "source_sha256": digest(source_path), "factor_source": factor_meta, "factor_changes": int(df.factor.diff().fillna(0).ne(0).sum()), "zero_volume_dates": int(df.volume.eq(0).sum()), "price_representation": "raw OHLC * multiplicative historical hfq factor / first retained factor; volume divided by same factor; actual amount retained", "execution_limits": "uniform +/-9.5% open/reference research band; not complete board/ST exchange limits", "explicit_ex_references": int(df.explicit_reference.sum()), "factor_derived_references": int(df.factor_derived_reference.sum())}
    write_json(meta_path, meta)
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", nargs="*")
    parser.add_argument("--membership", type=Path)
    parser.add_argument("--initial-pool", action="store_true")
    args = parser.parse_args()
    codes = list(args.codes or [])
    if args.membership:
        codes += pd.read_csv(args.membership).instrument.unique().tolist()
    if args.initial_pool:
        for code in (ROOT / "data/references/hs300_2026H1.txt").read_text().splitlines():
            if code.strip():
                number, exchange = code.strip().split(".")
                codes.append(exchange + number)
    codes = sorted(set(["SH000300", *codes]))
    decoder = prepare_decoder()
    successes, failures = [], []
    started = time.time()
    for i, code in enumerate(codes):
        try:
            meta = collect(code, decoder)
            successes.append(meta)
            print(f"market {i+1}/{len(codes)} {code} rows={meta['rows']} factors={meta['factor_changes']}", flush=True)
        except Exception as exc:
            failures.append({"instrument": code, "error": f"{type(exc).__name__}: {exc}"})
            print(f"market FAILED {code}: {type(exc).__name__}: {exc}", flush=True)
        write_json(DATA / "collection_status.json", {"requested": len(codes), "completed": len(successes), "failures": failures, "elapsed_seconds": time.time() - started, "all_complete": i + 1 == len(codes) and not failures})
        time.sleep(.25)
    write_json(DATA / "market_manifest.json", successes)
    if failures:
        raise SystemExit(f"{len(failures)} failed instruments; inspect collection_status.json")


if __name__ == "__main__":
    main()
