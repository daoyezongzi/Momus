import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from critic.experiment import validate_phase

STRATEGIES = (
    "simple_mean",
    "critic_no_corr",
    "critic_price_only",
    "critic_volume_only",
    "critic_price_volume",
    "critic_random_mapping",
    "critic_past_shifted",
)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        required=True,
        choices=("small-trial", "dev", "freeze", "sealed-2025"),
    )
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--allow-2025-labels", action="store_true")
    parser.add_argument("--frozen-config")
    parser.add_argument("--frozen-hash")
    parser.add_argument("--output-dir", default="artifacts/resonance")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--candidate-seed", type=int, default=0)
    parser.add_argument("--train-seeds", default="0,1,2,3,4")
    parser.add_argument("--past-shift", type=int, default=20)
    return parser


def _write_summary(output_dir: str, result: dict) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_phase(
        args.phase,
        args.allow_2025_labels,
        args.frozen_config,
        args.frozen_hash,
    )
    result = {
        "phase": args.phase,
        "synthetic": args.synthetic,
        "strategies": list(STRATEGIES),
        "allow_2025_labels": args.allow_2025_labels,
        "device": args.device,
        "candidate_seed": args.candidate_seed,
        "train_seeds": [int(seed) for seed in args.train_seeds.split(",") if seed],
        "past_shift": args.past_shift,
    }
    if args.synthetic:
        _write_summary(args.output_dir, result)
        return result
    raise RuntimeError("non-synthetic resonance experiment is not implemented in this task")


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
