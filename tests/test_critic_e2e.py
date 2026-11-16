"""端到端冒烟测试：加载真实 Kronos 模型，跑通 CriticKronosPredictor 管道。

用合成 K 线数据（随机游走），验证：
1. return_individual=True 时 auto_regressive_inference 返回 N 条路径
2. CriticKronosPredictor（RandomCritic）端到端产出合法 pred_df
3. use_critic=False 时退回原行为，形状与列一致

需要 HF 缓存中的 Kronos-small + Tokenizer-base。无网络/无缓存时会 skip。
"""
import numpy as np
import pandas as pd
import pytest

LOOKBACK = 90
PRED_LEN = 10
N = 8


def _make_synth_kline(n_rows, seed=0):
    """生成合成 K 线：几何随机游走 + 合理的 OHLC 关系。"""
    rng = np.random.default_rng(seed)
    rets = rng.standard_normal(n_rows) * 0.01
    close = 100 * np.exp(np.cumsum(rets))
    high = close * (1 + np.abs(rng.standard_normal(n_rows)) * 0.005)
    low = close * (1 - np.abs(rng.standard_normal(n_rows)) * 0.005)
    open_ = np.concatenate([[close[0]], close[:-1]])
    volume = rng.integers(1e5, 1e6, n_rows).astype(float)
    amount = volume * close
    ts = pd.date_range("2024-01-01", periods=n_rows, freq="D")
    return pd.DataFrame({
        "timestamps": ts, "open": open_, "high": high, "low": low,
        "close": close, "volume": volume, "amount": amount,
    })


@pytest.fixture(scope="module")
def kronos_models():
    try:
        from model import Kronos, KronosTokenizer
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        return model, tokenizer
    except Exception as e:
        pytest.skip(f"无法加载 Kronos 模型（无缓存/无网络）: {e}")


def test_return_individual_returns_n_paths(kronos_models):
    """决策-A：return_individual=True 应返回 (B, N, seq, feat)。"""
    import torch
    from model.kronos import auto_regressive_inference
    model, tokenizer = kronos_models

    df = _make_synth_kline(LOOKBACK + PRED_LEN)
    x = df.loc[:LOOKBACK - 1, ["open", "high", "low", "close", "volume", "amount"]].values.astype(np.float32)
    x = (x - x.mean(0)) / (x.std(0) + 1e-5)
    x = np.clip(x, -5, 5)[np.newaxis, :]

    # 时间戳特征 (minute,hour,weekday,day,month)
    from model.kronos import calc_time_stamps
    x_stamp = calc_time_stamps(df.loc[:LOOKBACK - 1, "timestamps"]).values.astype(np.float32)[np.newaxis, :]
    y_stamp = calc_time_stamps(df.loc[LOOKBACK:LOOKBACK + PRED_LEN - 1, "timestamps"]).values.astype(np.float32)[np.newaxis, :]

    dev = "cpu"
    out = auto_regressive_inference(
        tokenizer.to(dev), model.to(dev),
        torch.from_numpy(x).to(dev), torch.from_numpy(x_stamp).to(dev),
        torch.from_numpy(y_stamp).to(dev), 512, PRED_LEN,
        sample_count=N, return_individual=True,
    )
    assert out.ndim == 4, f"期望 4 维 (B,N,seq,feat)，得到 {out.shape}"
    assert out.shape[0] == 1 and out.shape[1] == N, f"batch/N 维不符: {out.shape}"


def test_critic_predictor_e2e(kronos_models):
    """CriticKronosPredictor 端到端产出合法 pred_df。"""
    from critic import CriticKronosPredictor, RandomCritic, CriticReranker
    model, tokenizer = kronos_models

    df = _make_synth_kline(LOOKBACK + PRED_LEN)
    x_df = df.loc[:LOOKBACK - 1, ["open", "high", "low", "close", "volume", "amount"]]
    x_ts = df.loc[:LOOKBACK - 1, "timestamps"]
    y_ts = df.loc[LOOKBACK:LOOKBACK + PRED_LEN - 1, "timestamps"]

    predictor = CriticKronosPredictor(
        model, tokenizer,
        critic=RandomCritic(seed=1),
        reranker=CriticReranker(mode="weighted", temperature=1.0),
        device="cpu",
    )
    pred_df = predictor.predict(x_df, x_ts, y_ts, pred_len=PRED_LEN,
                                sample_count=N, verbose=False)
    assert pred_df.shape == (PRED_LEN, 6), f"pred_df 形状不符: {pred_df.shape}"
    assert list(pred_df.columns) == ["open", "high", "low", "close", "volume", "amount"]
    assert not pred_df.isnull().values.any(), "预测含 NaN"


def test_use_critic_false_falls_back(kronos_models):
    """消融开关：use_critic=False 退回原行为，产出合法结果。"""
    from critic import CriticKronosPredictor, RandomCritic, CriticReranker
    model, tokenizer = kronos_models

    df = _make_synth_kline(LOOKBACK + PRED_LEN, seed=3)
    x_df = df.loc[:LOOKBACK - 1, ["open", "high", "low", "close", "volume", "amount"]]
    x_ts = df.loc[:LOOKBACK - 1, "timestamps"]
    y_ts = df.loc[LOOKBACK:LOOKBACK + PRED_LEN - 1, "timestamps"]

    predictor = CriticKronosPredictor(
        model, tokenizer, critic=RandomCritic(), reranker=CriticReranker(), device="cpu",
    )
    predictor.use_critic = False
    pred_df = predictor.predict(x_df, x_ts, y_ts, pred_len=PRED_LEN,
                                sample_count=N, verbose=False)
    assert pred_df.shape == (PRED_LEN, 6)
    assert not pred_df.isnull().values.any()
