from dataclasses import dataclass
import hashlib
import random

import pandas as pd


@dataclass(frozen=True)
class FoldSpec:
    fold_id: str
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    pred_len: int
    embargo: int
    test_start: pd.Timestamp | None = None
    test_end: pd.Timestamp | None = None


def make_default_folds(pred_len: int = 10) -> list[FoldSpec]:
    return [
        FoldSpec(
            "fold_2023",
            pd.Timestamp("2022-12-31"),
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2023-12-31"),
            pred_len,
            pred_len,
        ),
        FoldSpec(
            "fold_2024",
            pd.Timestamp("2023-12-31"),
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-12-31"),
            pred_len,
            pred_len,
        ),
    ]


def make_seen_unseen_split(chain_by_code, unseen_ratio: float, seed: int):
    rng = random.Random(seed)
    by_chain: dict[str, list[str]] = {}
    for code, chain in chain_by_code.items():
        by_chain.setdefault(chain, []).append(code)

    unseen: set[str] = set()
    for codes in by_chain.values():
        ordered = sorted(codes)
        rng.shuffle(ordered)
        count = max(1, round(len(ordered) * unseen_ratio))
        unseen.update(ordered[:count])

    seen = set(chain_by_code) - unseen
    return seen, unseen


def assert_unseen_not_in_training(train_targets, mapping, unseen) -> None:
    leaked = set(train_targets) & set(unseen)
    for target in train_targets:
        leaked.update(set(mapping.get(target, ())) & set(unseen))
    if leaked:
        raise ValueError(f"unseen code leaked into training: {sorted(leaked)}")


def assert_purge_embargo(fold: FoldSpec) -> None:
    min_validation_start = fold.train_end + pd.offsets.BDay(fold.embargo + 1)
    if fold.validation_start < min_validation_start:
        raise ValueError(
            f"validation_start violates embargo: {fold.validation_start} < {min_validation_start}"
        )
    if fold.test_start is not None:
        min_test_start = fold.validation_end + pd.offsets.BDay(fold.embargo + 1)
        if fold.test_start < min_test_start:
            raise ValueError(
                f"test_start violates embargo: {fold.test_start} < {min_test_start}"
            )


def split_hash(seen: set[str], unseen: set[str]) -> str:
    payload = "seen=" + ",".join(sorted(seen)) + ";unseen=" + ",".join(sorted(unseen))
    return hashlib.sha256(payload.encode()).hexdigest()
