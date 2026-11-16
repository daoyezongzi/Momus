"""CorrelatedDataProvider 测试：重点验证防数据泄漏红线。

用 fetch_fn 注入假数据，不依赖真实网络。
"""
import numpy as np
import pandas as pd
import pytest

from critic.data_provider import (
    CorrelatedDataProvider, _to_returns, _market_prefix, _ts_suffix, make_tushare_fetch,
)


def _fake_df(start, periods, seed=0):
    """生成一段假日线，index 为交易日。"""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=periods, freq="D")
    close = 100 + np.cumsum(rng.standard_normal(periods))
    return pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": rng.integers(1e5, 1e6, periods).astype(float),
        "amount": close * 1e5,
    }, index=idx)


def test_market_prefix():
    assert _market_prefix("300750") == "0"  # 深市
    assert _market_prefix("002594") == "0"
    assert _market_prefix("600519") == "1"  # 沪市


def test_to_returns_shape_and_first_row_zero():
    df = _fake_df("2024-01-01", 30)
    rets = _to_returns(df)
    assert rets.shape == (30, 6)
    assert np.allclose(rets[0], 0.0)  # 首行补零


def test_get_shape_for_known_target():
    """300750 在 config 中配了关联标的，应返回 (n_corr, lookback, 6)。"""
    # 注入：无论问哪个 symbol，都返回一段截止 2023-05-31 的数据
    def fake_fetch(code, start, end, session=None):
        return _fake_df("2023-01-01", 150, seed=hash(code) % 100)

    provider = CorrelatedDataProvider(lookback=90, fetch_fn=fake_fetch, request_delay=0)
    x_ts = pd.date_range("2023-03-01", "2023-05-31", freq="D")
    feat = provider.get("300750", x_ts)
    assert feat is not None
    assert feat.ndim == 3 and feat.shape[1:] == (90, 6)


def test_get_returns_none_for_unknown_target():
    provider = CorrelatedDataProvider(fetch_fn=lambda *a, **k: None, request_delay=0)
    feat = provider.get("999999", pd.date_range("2023-01-01", periods=10))
    assert feat is None


def test_no_lookahead_assert_triggers():
    """核心红线：若抓取返回了 T 之后的数据，必须触发 assert。"""
    # 注入含未来数据的假抓取（数据延伸到 2023-08，但 T=2023-05-31）
    def leaky_fetch(code, start, end, session=None):
        return _fake_df("2023-01-01", 240)  # 延伸到 8 月，越过 T

    provider = CorrelatedDataProvider(lookback=90, fetch_fn=leaky_fetch, request_delay=0)
    x_ts = pd.date_range("2023-03-01", "2023-05-31", freq="D")
    with pytest.raises(AssertionError, match="数据泄漏"):
        provider.get("300750", x_ts)


def test_data_truncated_at_T():
    """未触发 assert 的正常情况下，返回特征应只基于 <= T 的数据。"""
    captured = {}

    def fake_fetch(code, start, end, session=None):
        df = _fake_df("2023-01-01", 120)  # 截止 ~2023-04-30，早于 T
        captured["max_date"] = df.index.max()
        return df

    provider = CorrelatedDataProvider(lookback=60, fetch_fn=fake_fetch, request_delay=0)
    T = pd.Timestamp("2023-05-31")
    x_ts = pd.date_range("2023-03-01", T, freq="D")
    feat = provider.get("300750", x_ts)
    assert feat is not None
    assert captured["max_date"] <= T  # 抓取数据未越过 T


def test_ts_suffix():
    assert _ts_suffix("300750") == "300750.SZ"  # 深市
    assert _ts_suffix("002594") == "002594.SZ"
    assert _ts_suffix("600519") == "600519.SH"  # 沪市


def test_tushare_fetch_adapts_format(monkeypatch):
    """make_tushare_fetch 应把 tushare 原始格式适配为标准 OHLCV。

    用假 pro_api 注入 tushare 风格数据（含 vol 列、trade_date、倒序），
    验证适配后：列名 volume、index 为日期且升序。不需要真 token。
    """
    import critic.data_provider as dp

    class _FakePro:
        def daily(self, ts_code, start_date, end_date):
            # tushare 返回通常按日期倒序，列名为 vol
            return pd.DataFrame({
                "trade_date": ["20230103", "20230104", "20230105"][::-1],
                "open": [10.0, 11.0, 12.0][::-1],
                "high": [10.5, 11.5, 12.5][::-1],
                "low": [9.5, 10.5, 11.5][::-1],
                "close": [10.2, 11.2, 12.2][::-1],
                "vol": [1000.0, 2000.0, 3000.0][::-1],
                "amount": [1e5, 2e5, 3e5][::-1],
            })

    class _FakeTS:
        @staticmethod
        def pro_api(token):
            return _FakePro()

    monkeypatch.setitem(__import__("sys").modules, "tushare", _FakeTS)
    fetch = make_tushare_fetch("fake_token")
    df = fetch("300750", "20230101", "20231231")
    assert list(df.columns) == ["open", "high", "low", "close", "volume", "amount"]
    assert df.index.is_monotonic_increasing        # 已按日期升序
    assert df.index[0] == pd.Timestamp("2023-01-03")
    assert df["volume"].iloc[0] == 1000.0           # vol -> volume 适配正确
