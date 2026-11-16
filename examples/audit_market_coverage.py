import argparse
import json
from pathlib import Path

import pandas as pd


def load_codes(path: str) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def audit_file(path: Path) -> dict[str, object]:
    df = pd.read_csv(path, usecols=["timestamps"])
    dates = pd.to_datetime(df["timestamps"])
    return {
        "start": dates.min().strftime("%Y-%m-%d"),
        "end": dates.max().strftime("%Y-%m-%d"),
        "rows": int(len(dates)),
        "has_2025_or_later": bool((dates >= pd.Timestamp("2025-01-01")).any()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes-file", required=True)
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args(argv)

    codes = load_codes(args.codes_file)
    data_dir = Path(args.data_dir)
    coverage = {}
    missing = []
    for code in codes:
        path = data_dir / f"{code}.csv"
        if not path.exists():
            missing.append(code)
            continue
        coverage[code] = audit_file(path)

    print(json.dumps({"n_codes": len(codes), "missing_files": missing, "coverage": coverage}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
