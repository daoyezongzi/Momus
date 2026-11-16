from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd

from .schemas import SampleAuditMeta


@dataclass(frozen=True)
class CandidateCacheKey:
    target_code: str
    cutoff_T: pd.Timestamp
    pred_len: int
    sample_count: int
    seed: int
    kronos_model_id: str
    tokenizer_id: str
    generation_config_hash: str


@dataclass
class CandidateBundle:
    key: CandidateCacheKey
    paths: np.ndarray
    hist_norm: np.ndarray
    x_mean: np.ndarray
    x_std: np.ndarray
    y_true_norm: np.ndarray
    last_close: float
    meta: SampleAuditMeta | None


def candidate_config_hash(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def assert_same_candidates(first: CandidateBundle, second: CandidateBundle) -> None:
    if first.key != second.key or not np.array_equal(first.paths, second.paths):
        raise ValueError("candidate paths differ across ablation variants")
