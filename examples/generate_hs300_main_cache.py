"""Generate the frozen HS300 N=16 candidate caches.

This is the current ``generate_candidate_cache.py`` protocol with two
reproducibility/performance additions only: the project seed is fixed before
sampling, and windows for one stock are batched before calling the same
Kronos ``auto_regressive_inference`` implementation.  Window construction,
normalization, sampling parameters, candidate order, and cache schema remain
the current fixed-cache semantics.

Run from the repository root with the torch5060 environment, for example:

    C:/Users/Soyo/anaconda3/envs/torch5060/python.exe \
        examples/generate_hs300_main_cache.py \
        --data-dir data/market_2024_hs300 \
        --codes-file data/references/hs300_2026H1.txt \
        --output output/candidates/hs300_2024_N16.pkl \
        --device cuda
"""
from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.kronos import auto_regressive_inference, calc_time_stamps


COLS = ["open", "high", "low", "close", "volume", "amount"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_kline(data_dir: Path, code: str) -> pd.DataFrame | None:
    path = data_dir / f"{code}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    required = ["timestamps", *COLS]
    if any(column not in df.columns for column in required):
        return None
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    return df.sort_values("timestamps").reset_index(drop=True)


def make_walk_forward_windows(df: pd.DataFrame, lookback: int, pred_len: int):
    """Match the existing generator: origin starts at lookback, stride=pred_len."""
    windows = []
    start_idx = lookback
    while start_idx + pred_len <= len(df):
        hist = df.iloc[start_idx - lookback:start_idx]
        future = df.iloc[start_idx:start_idx + pred_len]
        windows.append(
            {
                "x_df": hist[COLS].copy(),
                "x_ts": pd.Series(hist["timestamps"].values),
                "y_ts": pd.Series(future["timestamps"].values),
                "y_true": future[COLS].to_numpy(dtype=np.float32),
            }
        )
        start_idx += pred_len
    return windows


def generate_batch(model, tokenizer, chunk, args):
    x_norms, x_stamps, y_stamps = [], [], []
    metadata = []
    for window in chunk:
        x = window["x_df"][COLS].to_numpy(dtype=np.float32)
        mean = x.mean(axis=0)
        std = x.std(axis=0)
        x_norm = np.clip((x - mean) / (std + 1e-5), -args.clip, args.clip)
        x_norms.append(x_norm.astype(np.float32))
        x_stamps.append(calc_time_stamps(window["x_ts"]).to_numpy(dtype=np.float32))
        y_stamps.append(calc_time_stamps(window["y_ts"]).to_numpy(dtype=np.float32))
        metadata.append((mean, std))

    xb = torch.from_numpy(np.stack(x_norms)).to(args.device)
    xsb = torch.from_numpy(np.stack(x_stamps)).to(args.device)
    ysb = torch.from_numpy(np.stack(y_stamps)).to(args.device)
    preds = auto_regressive_inference(
        tokenizer,
        model,
        xb,
        xsb,
        ysb,
        args.max_context,
        args.pred_len,
        args.clip,
        args.temperature,
        args.top_k,
        args.top_p,
        args.sample_count,
        False,
        return_individual=True,
    )
    return preds[:, :, -args.pred_len:, :].astype(np.float32), metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--codes-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--lookback", type=int, default=90)
    parser.add_argument("--pred-len", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-context", type=int, default=512)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--batch-windows", type=int, default=16)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-codes", type=int, default=None, help="debug-only input limit")
    parser.add_argument("--max-windows", type=int, default=None, help="debug-only window limit per code")
    args = parser.parse_args(argv)

    if args.sample_count != 16 or args.lookback != 90 or args.pred_len != 10:
        raise ValueError("HS300 main cache is frozen to sample_count=16, lookback=90, pred_len=10")
    if args.temperature != 1.0 or args.top_k != 0 or args.top_p != 0.9:
        raise ValueError("HS300 main sampling is frozen to T=1.0, top_k=0, top_p=0.9")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing cache: {args.output}")
    if not args.data_dir.exists():
        raise FileNotFoundError(args.data_dir)
    if not args.codes_file.exists():
        raise FileNotFoundError(args.codes_file)
    if args.batch_windows <= 0:
        raise ValueError("batch-windows must be positive")

    set_seed(args.seed)
    from model import Kronos, KronosTokenizer

    print(f"loading Kronos on {args.device}", flush=True)
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    tokenizer = tokenizer.to(args.device).eval()
    model = model.to(args.device).eval()

    codes = [line.strip() for line in args.codes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.max_codes is not None:
        if args.max_codes <= 0:
            raise ValueError("max-codes must be positive")
        codes = codes[:args.max_codes]
    all_windows = []
    all_candidates = []
    eligible_codes = []
    skipped_codes = []

    for code_index, code in enumerate(codes, start=1):
        df = load_kline(args.data_dir, code)
        if df is None:
            skipped_codes.append({"code": code, "reason": "missing_or_invalid_csv"})
            print(f"[{code_index}/{len(codes)}] {code}: skipped", flush=True)
            continue
        windows = make_walk_forward_windows(df, args.lookback, args.pred_len)
        if args.max_windows is not None:
            if args.max_windows <= 0:
                raise ValueError("max-windows must be positive")
            windows = windows[:args.max_windows]
        if not windows:
            skipped_codes.append({"code": code, "reason": "insufficient_rows", "rows": len(df)})
            print(f"[{code_index}/{len(codes)}] {code}: skipped rows={len(df)}", flush=True)
            continue
        eligible_codes.append(code)
        print(f"[{code_index}/{len(codes)}] {code}: {len(windows)} windows", flush=True)
        for start in range(0, len(windows), args.batch_windows):
            chunk = windows[start:start + args.batch_windows]
            paths, _ = generate_batch(model, tokenizer, chunk, args)
            if paths.shape != (len(chunk), args.sample_count, args.pred_len, len(COLS)):
                raise RuntimeError(f"unexpected candidate shape: {paths.shape}")
            for local_index, window in enumerate(chunk):
                all_windows.append(
                    (code, window["x_df"], window["x_ts"], window["y_ts"], window["y_true"])
                )
                all_candidates.append(paths[local_index])
            if (start + len(chunk)) % max(args.batch_windows * 4, 1) == 0:
                print(
                    f"  progress {start + len(chunk)}/{len(windows)} windows; total={len(all_windows)}",
                    flush=True,
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache = {
        "windows": all_windows,
        "candidates": all_candidates,
        "metadata": {
            "sample_count": args.sample_count,
            "lookback": args.lookback,
            "pred_len": args.pred_len,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "max_context": args.max_context,
            "clip": args.clip,
            "seed": args.seed,
            "data_dir": str(args.data_dir),
            "codes_file": str(args.codes_file),
            "nominal_codes": len(codes),
            "eligible_codes": eligible_codes,
            "skipped_codes": skipped_codes,
            "total_windows": len(all_windows),
            "generator": "examples/generate_hs300_main_cache.py",
        },
    }
    with args.output.open("xb") as handle:
        pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved {args.output} windows={len(all_windows)}", flush=True)


if __name__ == "__main__":
    main()
