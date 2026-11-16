"""predictor.generate_paths 的 return_log_probs 接线测试（不依赖真实 Kronos）。

用 monkeypatch 替换 auto_regressive_inference，验证：
- return_log_probs=False（默认）返回 4 元组（向后兼容）
- return_log_probs=True 返回 5 元组，log_probs 形状 (N,) 且值正确解包
"""
import numpy as np
import pandas as pd

import critic.predictor as predictor_mod
from critic.predictor import CriticKronosPredictor


class _Dummy:
    """最小 model/tokenizer 替身，仅支持 .to(device)。"""
    def to(self, device):
        return self


def _make_predictor():
    # critic/reranker 在 generate_paths 中不使用，置 None
    return CriticKronosPredictor(
        model=_Dummy(), tokenizer=_Dummy(), critic=None, reranker=None,
        device="cpu",
    )


def _make_df(lookback=12):
    idx = pd.date_range("2024-01-01", periods=lookback, freq="D")
    rng = np.random.default_rng(0)
    data = {c: rng.uniform(1, 2, lookback) for c in
            ["open", "high", "low", "close", "volume", "amount"]}
    return pd.DataFrame(data, index=idx), pd.Series(idx)


def test_generate_paths_default_returns_4_tuple(monkeypatch):
    N, pred_len, feat = 5, 4, 6
    preds = np.random.randn(1, N, pred_len, feat).astype(np.float32)
    monkeypatch.setattr(predictor_mod, "auto_regressive_inference",
                        lambda *a, **k: preds)
    pred = _make_predictor()
    df, ts = _make_df()
    y_ts = pd.Series(pd.date_range("2024-02-01", periods=pred_len, freq="D"))
    out = pred.generate_paths(df, ts, y_ts, pred_len, sample_count=N)
    assert len(out) == 4  # 向后兼容
    paths, hist, corr, norm = out
    assert paths.shape == (N, pred_len, feat)


def test_generate_paths_return_log_probs_unpacks_shape_and_values(monkeypatch):
    N, pred_len, feat = 5, 4, 6
    preds = np.random.randn(1, N, pred_len, feat).astype(np.float32)
    log_probs = np.array([[-1.0, -2.0, -0.5, -3.0, -4.0]])  # (B=1, N)

    def _mock(*a, **k):
        assert k.get("return_log_probs") is True
        return preds, log_probs
    monkeypatch.setattr(predictor_mod, "auto_regressive_inference", _mock)

    pred = _make_predictor()
    df, ts = _make_df()
    y_ts = pd.Series(pd.date_range("2024-02-01", periods=pred_len, freq="D"))
    out = pred.generate_paths(df, ts, y_ts, pred_len, sample_count=N,
                              return_log_probs=True)
    assert len(out) == 5  # 追加 log_probs
    paths, hist, corr, norm, lp = out
    assert lp.shape == (N,)  # (1,N) -> (N,)
    assert np.allclose(lp, log_probs[0])
    assert int(np.argmax(lp)) == 2  # 最大对数似然在 idx=2
