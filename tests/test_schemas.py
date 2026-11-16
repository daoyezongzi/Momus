import numpy as np
import pandas as pd

from critic.schemas import CriticSample, ResonanceFeatures, SampleAuditMeta


def make_meta(**overrides):
    values = dict(
        target_code="300750",
        cutoff_T=pd.Timestamp("2024-12-13"),
        input_max_ts=pd.Timestamp("2024-12-13"),
        label_start_ts=pd.Timestamp("2024-12-16"),
        label_end_ts=pd.Timestamp("2024-12-27"),
        fold="fold_2024",
        split="validation",
        seen_flag="seen",
        mapping_version="manual-v1",
        data_source="tushare",
        adjustment="qfq",
        candidate_seed=0,
        candidate_config_hash="candidate-hash",
        stock_pool_hash="pool-hash",
        mapping_hash="mapping-hash",
    )
    values.update(overrides)
    return SampleAuditMeta(**values)


def test_sample_keeps_predictions_labels_and_audit_meta():
    sample = CriticSample(
        schema_version=2,
        paths=np.zeros((4, 10, 6), dtype=np.float32),
        hist=np.zeros((90, 6), dtype=np.float32),
        y_true=np.zeros((10, 6), dtype=np.float32),
        corr_feat=None,
        resonance=ResonanceFeatures(
            price=np.ones(6, dtype=np.float32),
            volume=np.ones(5, dtype=np.float32),
            synchronous=np.ones(3, dtype=np.float32),
            lead_lag=np.ones(3, dtype=np.float32),
            mask=np.ones(17, dtype=bool),
        ),
        distances=np.zeros(4, dtype=np.float32),
        last_close=10.0,
        meta=make_meta(),
    )
    assert sample.paths.shape == (4, 10, 6)
    assert sample.meta.input_max_ts <= sample.meta.cutoff_T
    assert sample.meta.cutoff_T < sample.meta.label_start_ts
