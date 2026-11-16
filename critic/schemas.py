from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

SplitName = Literal["train", "validation", "test"]
SeenFlag = Literal["seen", "unseen"]
FeatureVariant = Literal[
    "no_corr",
    "price_only",
    "volume_only",
    "price_volume",
    "random_mapping",
    "past_shifted",
]


@dataclass(frozen=True)
class SampleAuditMeta:
    target_code: str
    cutoff_T: pd.Timestamp
    input_max_ts: pd.Timestamp
    label_start_ts: pd.Timestamp
    label_end_ts: pd.Timestamp
    fold: str
    split: SplitName
    seen_flag: SeenFlag
    mapping_version: str
    data_source: Literal["tushare"]
    adjustment: str
    candidate_seed: int
    candidate_config_hash: str
    stock_pool_hash: str
    mapping_hash: str


@dataclass(frozen=True)
class ResonanceFeatures:
    price: np.ndarray
    volume: np.ndarray
    synchronous: np.ndarray
    lead_lag: np.ndarray
    mask: np.ndarray

    def vector(self, variant: FeatureVariant) -> np.ndarray:
        if variant == "no_corr":
            return np.empty(0, dtype=np.float32)
        if variant == "price_only":
            return np.concatenate([self.price, self.synchronous, self.lead_lag]).astype(
                np.float32, copy=False
            )
        if variant == "volume_only":
            return self.volume.astype(np.float32, copy=True)
        return np.concatenate(
            [self.price, self.volume, self.synchronous, self.lead_lag]
        ).astype(np.float32, copy=False)


@dataclass
class CriticSample:
    paths: np.ndarray
    hist: np.ndarray
    corr_feat: np.ndarray | None
    distances: np.ndarray
    code: str | None = None
    schema_version: int = 1
    y_true: np.ndarray | None = None
    resonance: ResonanceFeatures | None = None
    last_close: float | None = None
    meta: SampleAuditMeta | None = None
    fine_distances: np.ndarray | None = None
    coarse_distances: np.ndarray | None = None
