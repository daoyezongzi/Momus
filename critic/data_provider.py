"""CorrelatedDataProvider：抓取并对齐关联标的截止预测时点 T 的历史行情。

核心职责（设计文档 §3、§5.4）：
1. 按 config 中的关联标的清单，抓取各标的的日线历史。
2. **防数据泄漏**：强制所有返回数据的时间戳 <= 预测时点 T。这是 Critic 不作弊的
   代码层保障（ENFORCE_NO_LOOKAHEAD）。
3. 把关联标的历史转成收益率特征，对齐到统一的 lookback 窗口。

数据源复用 examples/get_akshare_date_2024-2025_x.py 的东方财富 API 方式。
财报特征首版留桩（设计文档 §11 决策③），仅用关联价格特征。
"""
import time
import json
import random

import numpy as np
import pandas as pd

from .config import get_correlated_symbols, ENFORCE_NO_LOOKAHEAD

EASTMONEY_KLINE_URL = "http://push2his.eastmoney.com/api/qt/stock/kline/get"


def _market_prefix(stock_code: str) -> str:
    """判断市场：'0'=深交所, '1'=上交所（与 examples 抓取脚本一致）。"""
    if stock_code.startswith(("0", "2", "3")):
        return "0"
    return "1"  # 6/9 开头沪市，其余默认沪市


def _ts_suffix(stock_code: str) -> str:
    """tushare 代码后缀：深市 .SZ / 沪市 .SH（北交所 .BJ 暂不覆盖）。"""
    if stock_code.startswith(("0", "2", "3")):
        return f"{stock_code}.SZ"
    return f"{stock_code}.SH"


def make_tushare_fetch(token: str):
    """返回一个与 _fetch_daily_kline 同签名的 tushare 抓取函数。

    tushare 用 token 授权，限流为积分制（非 IP 封禁），比东方财富稳定。
    输出对齐为 index=日期、列=open/high/low/close/volume/amount，故可直接
    通过 CorrelatedDataProvider 的 fetch_fn 注入，核心逻辑零改动。

    Args:
        token: tushare API token（从环境变量注入，勿硬编码/提交）。
    """
    import tushare as ts
    pro = ts.pro_api(token)

    def _fetch(stock_code, start_date, end_date, session=None,
               max_retries=3, retry_backoff=1.0):
        ts_code = _ts_suffix(stock_code)
        last_err = None
        for attempt in range(max_retries):
            try:
                df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
                if df is None or len(df) == 0:
                    return None
                # tushare 列: trade_date, open, high, low, close, vol, amount
                df = df.rename(columns={"vol": "volume"})
                df["date"] = pd.to_datetime(df["trade_date"])
                df = df.set_index("date").sort_index()
                return df[["open", "high", "low", "close", "volume", "amount"]]
            except Exception as e:  # 积分限流/网络抖动 -> 退避重试
                last_err = e
                time.sleep(retry_backoff * (attempt + 1) + random.uniform(0, 0.5))
        return None

    return _fetch


def _fetch_daily_kline(stock_code: str, start_date: str, end_date: str,
                       session=None, max_retries=3, retry_backoff=1.0) -> pd.DataFrame | None:
    """从东方财富抓日线。返回 index 为日期、含 open/high/low/close/volume/amount 的 DataFrame。

    注意东方财富 klines 字段顺序为 日期,开,收,高,低,量,额，此处已重排为标准 OHLC。
    """
    import requests

    secid = f"{_market_prefix(stock_code)}.{stock_code}"
    params = {
        "secid": secid,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",   # 日线
        "fqt": "1",     # 前复权
        "beg": start_date,
        "end": end_date,
        "lmt": "10000",
        "ut": "fa5fd1943c7b386f172d6893dbfba10b",
        "cb": f"jQuery{random.randint(1000000, 9999999)}_{int(time.time() * 1000)}",
    }
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"),
        "Referer": "https://quote.eastmoney.com/",
        "Accept": "*/*",
    }

    req = session or requests
    # 带指数退避的重试，缓解东方财富限流（连接被断/超时）
    resp = None
    for attempt in range(max_retries):
        try:
            resp = req.get(EASTMONEY_KLINE_URL, params=params, headers=headers, timeout=15)
            if resp.status_code == 200:
                break
        except requests.exceptions.RequestException:
            resp = None
        time.sleep(retry_backoff * (attempt + 1) + random.uniform(0, 0.5))
    if resp is None or resp.status_code != 200:
        return None

    text = resp.text
    if text.startswith("/**/"):
        text = text[4:]
    l, r = text.find("("), text.rfind(")")
    if l == -1 or r == -1:
        return None
    data = json.loads(text[l + 1:r])
    if not data or data.get("data") is None:
        return None
    klines = data["data"].get("klines", [])
    if not klines:
        return None

    rows = []
    for k in klines:
        it = k.split(",")
        if len(it) >= 6:
            rows.append({
                "date": it[0],
                "open": float(it[1]), "close": float(it[2]),
                "high": float(it[3]), "low": float(it[4]),
                "volume": float(it[5]),
                "amount": float(it[6]) if len(it) > 6 else 0.0,
            })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df[["open", "high", "low", "close", "volume", "amount"]]


def _to_returns(df: pd.DataFrame) -> np.ndarray:
    """转对数收益率特征，消除量纲、增强跨标的可比性（设计文档 §5.1）。

    对 OHLC 用对数收益，对 volume/amount 用对数差分。首行补 0。
    返回 (T, 6) 数组。
    """
    vals = df[["open", "high", "low", "close", "volume", "amount"]].values.astype(np.float64)
    vals = np.where(vals <= 0, 1e-8, vals)
    log_vals = np.log(vals)
    rets = np.diff(log_vals, axis=0)
    rets = np.vstack([np.zeros((1, rets.shape[1])), rets])
    return rets.astype(np.float32)


class CorrelatedDataProvider:
    """提供关联标的截止 T 的历史特征，供 FundamentalCritic 使用。"""

    def __init__(self, lookback=90, session=None, fetch_fn=None, request_delay=1.0):
        """
        Args:
            lookback: 关联特征回看窗口长度。
            session: 可选的 requests.Session（复用连接/测试注入）。
            fetch_fn: 可选的抓取函数覆盖（默认 _fetch_daily_kline），便于测试注入假数据。
            request_delay: 多标的抓取间的请求间隔秒数，缓解东方财富限流。
        """
        self.lookback = lookback
        self.session = session
        self._fetch = fetch_fn or _fetch_daily_kline
        self.request_delay = request_delay

    def get(self, target_code: str, x_timestamp) -> np.ndarray | None:
        """返回目标股关联标的的收益率特征，形状 (n_corr, lookback, 6)。

        Args:
            target_code: 目标股代码。
            x_timestamp: 目标股历史时间戳序列，其最大值定义预测时点 T。

        Returns:
            (n_corr, lookback, 6) 特征张量；无关联标的时返回 None。
            对齐规则：只取每个关联标的 <= T 的最后 lookback 天。
        """
        symbols = get_correlated_symbols(target_code)
        if not symbols:
            return None

        T = pd.to_datetime(pd.Series(x_timestamp)).max()
        # 抓取窗口：T 往前留足够天数（含非交易日冗余）
        start = (T - pd.DateOffset(days=int(self.lookback * 3))).strftime("%Y%m%d")
        end = T.strftime("%Y%m%d")

        feats = []
        for i, sym in enumerate(symbols):
            if i > 0 and self.request_delay > 0:
                time.sleep(self.request_delay)  # 请求间隔，缓解限流
            df = self._fetch(sym, start, end, session=self.session)
            if df is None or len(df) == 0:
                # 抓取失败或无数据 → 用零特征占位，保持张量形状稳定
                feats.append(np.zeros((self.lookback, 6), dtype=np.float32))
                continue

            # ---- 数据泄漏红线：强制 <= T ----
            if ENFORCE_NO_LOOKAHEAD:
                assert df.index.max() <= T, (
                    f"数据泄漏: 关联标的 {sym} 含 T={T} 之后的数据 "
                    f"(max={df.index.max()})")
            df = df[df.index <= T]

            rets = _to_returns(df)  # (len, 6)
            # 取最后 lookback 天；不足则前部补零
            if len(rets) >= self.lookback:
                rets = rets[-self.lookback:]
            else:
                pad = np.zeros((self.lookback - len(rets), 6), dtype=np.float32)
                rets = np.vstack([pad, rets])
            feats.append(rets)

        return np.stack(feats, axis=0)  # (n_corr, lookback, 6)
