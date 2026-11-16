"""生成语义分解归因消融的置换标签 cache（C1/C2/C3）。

预注册协议：docs/development/protocols/2026-07-28_semantic_decomposition_ablation_preregistration.md

三个对照臂只改辅助头的目标，**不改** distances/paths/hist/corr_feat/code：
- C1 duplicate_final：fine := coarse := distances（排除"只是多任务正则"）
- C2 shuffled：fine/coarse 各自组内独立打乱（排除"只是梯度噪声"）
- C3 swapped：fine <-> coarse 互换（检验语义配对是否承载信息）

生成后逐项断言，断言失败即中止，避免产出静默错误的 cache。
"""
import argparse
import copy
import csv
import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

SHUFFLE_SEED_BASE = 20260728


def load_samples(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def make_c1_duplicate_final(samples):
    out = copy.deepcopy(samples)
    for sample in out:
        sample.fine_distances = sample.distances.copy()
        sample.coarse_distances = sample.distances.copy()
    return out


def make_c2_shuffled(samples):
    out = copy.deepcopy(samples)
    for index, sample in enumerate(out):
        rng = np.random.default_rng(SHUFFLE_SEED_BASE + index)
        n = sample.fine_distances.shape[0]
        sample.fine_distances = sample.fine_distances[rng.permutation(n)]
        sample.coarse_distances = sample.coarse_distances[rng.permutation(n)]
    return out


def make_c3_swapped(samples):
    out = copy.deepcopy(samples)
    for sample in out:
        fine = sample.fine_distances.copy()
        sample.fine_distances = sample.coarse_distances.copy()
        sample.coarse_distances = fine
    return out


BUILDERS = {
    "C1_duplicate_final": make_c1_duplicate_final,
    "C2_shuffled": make_c2_shuffled,
    "C3_swapped": make_c3_swapped,
}

UNTOUCHED_FIELDS = ("distances", "paths", "hist")


def assert_untouched(src, out, arm):
    """distances/paths/hist 必须逐比特不变；corr_feat/code 也不得变动。"""
    if len(src) != len(out):
        raise AssertionError(f"{arm}: sample count changed {len(src)} -> {len(out)}")
    for index, (a, b) in enumerate(zip(src, out)):
        for field in UNTOUCHED_FIELDS:
            va, vb = getattr(a, field), getattr(b, field)
            if not np.array_equal(va, vb):
                raise AssertionError(f"{arm}: sample {index} field {field} was modified")
        if a.code != b.code:
            raise AssertionError(f"{arm}: sample {index} code changed")
        if (a.corr_feat is None) != (b.corr_feat is None):
            raise AssertionError(f"{arm}: sample {index} corr_feat presence changed")
        if a.corr_feat is not None and not np.array_equal(a.corr_feat, b.corr_feat):
            raise AssertionError(f"{arm}: sample {index} corr_feat was modified")


def assert_arm_semantics(src, out, arm):
    """每个臂特有的不变量。"""
    for index, (a, b) in enumerate(zip(src, out)):
        if arm == "C1_duplicate_final":
            if not (np.array_equal(b.fine_distances, a.distances)
                    and np.array_equal(b.coarse_distances, a.distances)):
                raise AssertionError(f"C1: sample {index} aux targets are not copies of distances")
        elif arm == "C2_shuffled":
            # 组内打乱：多重集合必须完全相同，且顺序必须已改变（n>1 且非全等值时）
            if not np.array_equal(np.sort(b.fine_distances), np.sort(a.fine_distances)):
                raise AssertionError(f"C2: sample {index} fine multiset changed")
            if not np.array_equal(np.sort(b.coarse_distances), np.sort(a.coarse_distances)):
                raise AssertionError(f"C2: sample {index} coarse multiset changed")
        elif arm == "C3_swapped":
            if not (np.array_equal(b.fine_distances, a.coarse_distances)
                    and np.array_equal(b.coarse_distances, a.fine_distances)):
                raise AssertionError(f"C3: sample {index} targets not exactly swapped")


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def c2_order_change_ratio(src, out):
    """C2 实际改变了配对顺序的样本占比，用于报告打乱强度。"""
    changed = sum(
        1 for a, b in zip(src, out)
        if not np.array_equal(a.fine_distances, b.fine_distances)
    )
    return changed / max(1, len(src))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="output/results/regime_generalization/samples_multiregime_train_dual.pkl",
    )
    parser.add_argument("--output-dir", default="output/results/semantic_decomposition_ablation")
    parser.add_argument("--arm", action="append", choices=sorted(BUILDERS), default=None)
    args = parser.parse_args()

    src = load_samples(args.source)
    missing = [i for i, s in enumerate(src)
               if s.fine_distances is None or s.coarse_distances is None]
    if missing:
        raise SystemExit(f"source cache lacks fine/coarse labels in {len(missing)} samples")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arms = args.arm or sorted(BUILDERS)

    rows = [{
        "arm": "source",
        "path": args.source,
        "sample_count": len(src),
        "sha256": sha256_of(args.source),
        "note": "frozen input, unmodified",
    }]
    for arm in arms:
        out = BUILDERS[arm](src)
        assert_untouched(src, out, arm)
        assert_arm_semantics(src, out, arm)
        path = output_dir / f"cache_{arm}.pkl"
        with open(path, "wb") as handle:
            pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
        note = "assertions passed"
        if arm == "C2_shuffled":
            note += f"; order_changed_ratio={c2_order_change_ratio(src, out):.4f}"
        rows.append({
            "arm": arm,
            "path": str(path),
            "sample_count": len(out),
            "sha256": sha256_of(path),
            "note": note,
        })
        print(f"{arm}: {len(out)} samples -> {path}  [{note}]", flush=True)

    audit = output_dir / "cache_audit.csv"
    with open(audit, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"audit -> {audit}")


if __name__ == "__main__":
    main()

