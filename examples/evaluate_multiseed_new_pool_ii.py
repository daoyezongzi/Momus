"""固定 new_pool_II 缓存的多 seed learned-verifier 只读评估。"""

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from examples.diagnose_new_pool_ii import _normalized_target, baseline_rows_for_window
from examples.evaluate_cache import load_critic


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_period(label, cache_path, dual_critic, small_mlp, seed, temperature, clip):
    with open(cache_path, "rb") as handle:
        cache = pickle.load(handle)
    rows = []
    for window_index, (window, candidates) in enumerate(zip(cache["windows"], cache["candidates"])):
        code, x_df, _, _, y_true = window
        hist_norm, y_true_norm = _normalized_target(x_df, y_true, clip=clip)
        for row in baseline_rows_for_window(
            candidates, y_true_norm, hist_norm, dual_critic, small_mlp, temperature
        ):
            rows.append({"period": label, "seed": seed, "code": str(code),
                         "window_index": window_index, **row})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dual-checkpoint", required=True)
    parser.add_argument("--small-mlp-checkpoint", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cache", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    dual_critic, _ = load_critic(args.dual_checkpoint, "dual_head", args.device)
    small_mlp, _ = load_critic(
        args.small_mlp_checkpoint, "fundamental", args.device,
        path_corr_interaction=False,
    )
    rows = []
    for item in args.cache:
        label, cache_path = item.split("=", 1)
        rows.extend(evaluate_period(
            label, cache_path, dual_critic, small_mlp, args.seed,
            args.temperature, args.clip,
        ))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output, rows)
    print(f"seed={args.seed} rows={len(rows)} output={output}")


if __name__ == "__main__":
    main()
