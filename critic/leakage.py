from collections.abc import Iterable

import pandas as pd

SEALED_LABEL_START = pd.Timestamp("2025-01-01")


def assert_no_lookahead(input_timestamps, label_timestamps, cutoff_T) -> None:
    cutoff = pd.Timestamp(cutoff_T)
    inputs = pd.DatetimeIndex(input_timestamps)
    labels = pd.DatetimeIndex(label_timestamps)
    if inputs.empty or labels.empty:
        raise ValueError("input and label timestamps must be non-empty")
    if inputs.max() > cutoff:
        raise ValueError("input timestamp exceeds cutoff T")
    if labels.min() <= cutoff:
        raise ValueError("label timestamp must be after cutoff T")


def assert_label_seal(label_start_ts, allow_2025_labels: bool) -> None:
    if pd.Timestamp(label_start_ts) >= SEALED_LABEL_START and not allow_2025_labels:
        raise RuntimeError("2025 labels are sealed until configuration is frozen")


def assert_source_consistency(sources: Iterable[str]) -> None:
    values = set(sources)
    if values != {"tushare"}:
        raise ValueError("formal experiment requires a single Tushare source")
