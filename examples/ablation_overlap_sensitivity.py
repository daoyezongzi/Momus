"""消融的重叠敏感性再聚合（协议 §4.6 / runbook 步骤 7）。

codes_expanded.txt（sealed 线 86 股训练池）与 new_pool_II 共享 9 只股票，
每只贡献 40/3380 训练样本。本脚本对已有 window_metrics.csv 做纯描述性
再聚合，比较"全部 86 股" vs "排除 9 只重叠股（77 股）"下的配对差与 CI。

不重跑评估、不改任何模型行为。结论不一致时以 77 股为准（协议 §4.6）。
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, ".")

from examples.evaluate_semantic_decomposition_ablation import (
    ABLATION_DIR,
    aggregate_per_seed,
    paired_deltas,
    write_csv,
)

OVERLAP_CODES = {"300308", "300433", "300782", "600016", "600019",
                 "600048", "600050", "600104", "600111"}


def load_window_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return [{**r, "seed": int(r["seed"]), "ohlc_mse": float(r["ohlc_mse"])}
                for r in csv.DictReader(handle)]


def normalize_code(code):
    """候选 cache 里的 code 可能带交易所后缀或前导零差异。"""
    digits = "".join(ch for ch in str(code) if ch.isdigit())
    return digits[-6:] if len(digits) >= 6 else digits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=ABLATION_DIR)
    parser.add_argument("--reference", default="B0_dual_head")
    args = parser.parse_args()

    out = Path(args.output_dir)
    rows = load_window_rows(out / "window_metrics.csv")
    codes = {normalize_code(r["code"]) for r in rows}
    present_overlap = codes & OVERLAP_CODES
    kept = [r for r in rows if normalize_code(r["code"]) not in OVERLAP_CODES]

    print(f"total codes={len(codes)} overlap present={len(present_overlap)} "
          f"{sorted(present_overlap)}")
    print(f"windows: all={len(rows)} excluding_overlap={len(kept)} "
          f"dropped={len(rows) - len(kept)}")
    if not present_overlap:
        print("WARNING: none of the 9 overlap codes found; check code formatting")

    combined = []
    for label, subset in (("all_86", rows), ("excl_overlap_77", kept)):
        if not subset:
            continue
        per_seed = aggregate_per_seed(subset)
        for row in paired_deltas(per_seed, reference=args.reference):
            combined.append({"code_subset": label, **row})
    if not combined:
        raise SystemExit("no paired deltas computed; need >=2 arms evaluated")
    write_csv(out / "overlap_sensitivity.csv", combined)

    # 并排打印主指标（2024 / weighted），便于直接判读一致性
    print("\n2024 / weighted 主指标并排对比：")
    print(f"{'arm':26} {'all_86':>12} {'excl_77':>12}  一致?")
    by_arm = {}
    for row in combined:
        if row["period"] == "2024" and row["strategy"] == "weighted":
            by_arm.setdefault(row["arm"], {})[row["code_subset"]] = row
    for arm, pair in sorted(by_arm.items()):
        a = pair.get("all_86")
        b = pair.get("excl_overlap_77")
        if not (a and b):
            continue
        same = a["ci_excludes_zero"] == b["ci_excludes_zero"]
        print(f"{arm:26} {a['mean_improvement_pct_of_ref']:+11.3f}% "
              f"{b['mean_improvement_pct_of_ref']:+11.3f}%  "
              f"{'yes' if same else 'NO — 以 77 股为准'}")


if __name__ == "__main__":
    main()
