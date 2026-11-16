import argparse
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from critic.config import CORRELATED_UNIVERSE
from critic.mapping import mapping_coverage, snapshot_mapping


def load_codes(path: str) -> list[str]:
    return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def infer_chain_by_code(mapping: dict[str, list[str]], stock_pool: list[str]) -> dict[str, str]:
    chain_by_code: dict[str, str] = {}
    for index, code in enumerate(sorted(stock_pool)):
        chain_by_code[code] = f"unassigned_{index}"
    for target, related in mapping.items():
        if target in chain_by_code:
            chain_id = f"manual_{target}"
            chain_by_code[target] = chain_id
            for code in related:
                if code in chain_by_code:
                    chain_by_code[code] = chain_id
    return chain_by_code


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes-file", required=True)
    parser.add_argument("--version", default="manual-v1")
    parser.add_argument("--mapped-only", action="store_true")
    args = parser.parse_args(argv)

    raw_codes = load_codes(args.codes_file)
    excluded_unmapped = []
    if args.mapped_only:
        codes = [code for code in raw_codes if code in CORRELATED_UNIVERSE]
        excluded_unmapped = [code for code in raw_codes if code not in CORRELATED_UNIVERSE]
    else:
        codes = raw_codes

    snapshot = snapshot_mapping(
        {code: CORRELATED_UNIVERSE[code] for code in codes if code in CORRELATED_UNIVERSE},
        infer_chain_by_code(CORRELATED_UNIVERSE, codes),
        codes,
        args.version,
    )
    payload = {"sha256": snapshot.sha256, **mapping_coverage(snapshot)}
    if args.mapped_only:
        payload["excluded_unmapped"] = excluded_unmapped
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
