import json
import pickle
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from critic.resonance_analysis import (
    assign_bins,
    bootstrap_bin_difference,
    compute_cached_resonance_score,
    summarize_distances_by_bin,
)


def load_samples(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def run(train_cache, eval_cache):
    train_samples = load_samples(train_cache)
    eval_samples = load_samples(eval_cache)
    train_scores = [compute_cached_resonance_score(sample) for sample in train_samples]
    eval_scores = [compute_cached_resonance_score(sample) for sample in eval_samples]
    result = {
        "train_samples": len(train_samples),
        "eval_samples": len(eval_samples),
        "by_resonance": {},
    }
    for field in ["price", "volume", "price_volume"]:
        bins = assign_bins(train_scores, eval_scores, field)
        summary = summarize_distances_by_bin(eval_samples, bins)
        values_by_bin = {"low": [], "mid": [], "high": []}
        for sample, bin_name in zip(eval_samples, bins):
            values_by_bin[bin_name].append(float(sample.distances.mean()))
        if values_by_bin["low"] and values_by_bin["high"]:
            summary["high_vs_low"] = bootstrap_bin_difference(
                values_by_bin["low"], values_by_bin["high"], n_boot=2000, seed=0
            )
        result["by_resonance"][field] = summary
    return result


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--eval-cache", required=True)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    result = run(args.train_cache, args.eval_cache)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    return result


if __name__ == "__main__":
    main()
