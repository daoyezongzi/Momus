"""端到端 Critic 训练编排（真实数据）。

完整流程（设计文档 §5.3、§11 决策⑤）：
  1. 抓多只目标股历史（东方财富）
  2. walk-forward 切多个预测时点 T（训练/测试时段分离，防泄漏）
  3. 每个时点用 Kronos 生成 N 条候选，算每条与真实未来的距离 -> CriticSample
  4. 样本缓存到磁盘（样本构造慢，只做一次）
  5. 用 pairwise ranking loss 训练 Critic，存权重

两阶段解耦：--build-only 只构样本并缓存；--train-only 从缓存训练。
默认两者都做。

用法：
  # 小规模冒烟（先验证流程，2只股票各3个时点）
  python examples/train_critic_pipeline.py --codes 300750,002594 --n-windows 3 --epochs 5
  # 正式（设计文档建议 ~50 只，跑通后扩）
  python examples/train_critic_pipeline.py --codes-file data/references/codes.txt --n-windows 20 --epochs 30

注意：真实未来仅用于算距离标签，绝不作为 Critic 输入（reward-model 式，非泄漏）。
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append("../")
sys.path.append(".")

from critic import (
    CriticKronosPredictor, FundamentalCritic, CriticReranker,
    CorrelatedDataProvider, CriticSample, compute_distances,
    train_critic, make_tushare_fetch,
)
from critic.data_provider import _fetch_daily_kline
from critic.train_critic import build_samples_for_series, save_samples, load_samples, save_critic


def fetch_target_history(code, start="20220101", end="20231231", delay=1.5, fetch_fn=None):
    """抓目标股历史日线，返回带 timestamps 列的 DataFrame。

    fetch_fn 可注入 tushare 抓取（make_tushare_fetch 的返回）；默认用东方财富。
    """
    import time
    fn = fetch_fn or _fetch_daily_kline
    df = fn(code, start, end)
    if delay > 0:
        time.sleep(delay)
    if df is None or len(df) == 0:
        return None
    df = df.reset_index().rename(columns={"date": "timestamps"})
    return df


def make_walk_forward_windows(df, lookback, pred_len, n_windows):
    """从一只股票的历史里按时间切 n_windows 个 (历史, 未来) 窗口。

    窗口按时间等距分布，靠后的窗口 T 更晚。真实实验应保证训练窗口的 T
    早于测试窗口（时段分离），此处返回全部窗口，时段划分交由调用方。
    """
    total = len(df)
    min_needed = lookback + pred_len
    if total < min_needed:
        return []
    # 可放置窗口的起点范围
    last_start = total - min_needed
    if n_windows == 1:
        starts = [last_start]
    else:
        starts = np.linspace(0, last_start, n_windows).astype(int)
    windows = []
    cols = ["open", "high", "low", "close", "volume", "amount"]
    for s in starts:
        hist = df.iloc[s:s + lookback]
        fut = df.iloc[s + lookback:s + lookback + pred_len]
        windows.append({
            "x_df": hist[cols], "x_ts": hist["timestamps"],
            "y_ts": fut["timestamps"],
            "y_true": fut[cols].values.astype(np.float32),
        })
    return windows


def build_samples(args, predictor, fetch_fn=None):
    """阶段1：抓数据、切窗、生成候选、算距离 -> List[CriticSample]。"""
    codes = _load_codes(args)
    samples = []
    for ci, code in enumerate(codes):
        print(f"[{ci+1}/{len(codes)}] 抓取 {code} 历史...")
        df = fetch_target_history(code, args.start, args.end, fetch_fn=fetch_fn)
        if df is None:
            print(f"  跳过 {code}（抓取失败）")
            continue
        windows = make_walk_forward_windows(df, args.lookback, args.pred_len, args.n_windows)
        print(f"  切出 {len(windows)} 个窗口，逐个生成候选...")
        for wi, w in enumerate(windows):
            paths, x_norm, corr_feat, (mean, std) = build_samples_for_series(
                predictor, w["x_df"], w["x_ts"], w["y_ts"], args.pred_len,
                target_code=code, sample_count=args.sample_count,
            )
            # 真实未来归一化到候选同一空间后算距离
            y_norm = (w["y_true"] - mean) / (std + 1e-5)
            distances = compute_distances(paths, y_norm)
            samples.append(CriticSample(paths=paths, hist=x_norm,
                                        corr_feat=corr_feat, distances=distances,
                                        code=code))
            print(f"    窗口 {wi+1}/{len(windows)}: 距离范围 "
                  f"[{distances.min():.3f}, {distances.max():.3f}]")
    return samples


def _load_codes(args):
    if args.codes_file:
        with open(args.codes_file) as f:
            return [ln.strip() for ln in f if ln.strip()]
    return [c.strip() for c in args.codes.split(",") if c.strip()]


def _load_dotenv(path=".env"):
    """轻量加载 .env 到环境变量（不覆盖已存在的），避免引入额外依赖。"""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            os.environ.setdefault(key, val)


def main():
    _load_dotenv()  # 先加载 .env，使 --tushare-token 的默认值能读到环境变量
    p = argparse.ArgumentParser()
    p.add_argument("--codes", type=str, default="300750,002594")
    p.add_argument("--codes-file", type=str, default=None)
    p.add_argument("--start", type=str, default="20220101")
    p.add_argument("--end", type=str, default="20231231")
    p.add_argument("--lookback", type=int, default=90)
    p.add_argument("--pred-len", type=int, default=10)
    p.add_argument("--n-windows", type=int, default=3)
    p.add_argument("--sample-count", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--samples-cache", type=str, default="critic_samples.pkl")
    p.add_argument("--critic-out", type=str, default="critic_trained.pt")
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--train-only", action="store_true")
    p.add_argument("--tushare-token", type=str, default=os.environ.get("TUSHARE_TOKEN"),
                   help="tushare token；默认读环境变量 TUSHARE_TOKEN。不设则用东方财富")
    args = p.parse_args()

    # 数据源：有 token 用 tushare（稳定），否则回退东方财富
    fetch_fn = None
    if args.tushare_token:
        fetch_fn = make_tushare_fetch(args.tushare_token)
        print("数据源: tushare")
    else:
        print("数据源: 东方财富（未提供 tushare token）")

    # ---- 阶段1：构造样本 ----
    if not args.train_only:
        from model import Kronos, KronosTokenizer
        print("加载 Kronos 模型...")
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        predictor = CriticKronosPredictor(
            model, tokenizer,
            critic=FundamentalCritic(in_dim=6),   # 构样本阶段不用其打分，占位
            reranker=CriticReranker(),
            corr_provider=CorrelatedDataProvider(
                lookback=args.lookback, fetch_fn=fetch_fn,
                request_delay=0 if fetch_fn else 1.5),  # tushare 无需 IP 节流
            device=args.device,
        )
        samples = build_samples(args, predictor, fetch_fn=fetch_fn)
        if not samples:
            print("未构造出任何样本，终止。")
            return
        save_samples(samples, args.samples_cache)
        print(f"已缓存 {len(samples)} 个样本 -> {args.samples_cache}")
        if args.build_only:
            return
    else:
        samples = load_samples(args.samples_cache)
        print(f"从缓存加载 {len(samples)} 个样本")

    # ---- 阶段2：训练 ----
    critic = FundamentalCritic(in_dim=6)
    print(f"开始训练 Critic（{args.epochs} epochs, {len(samples)} 样本）...")
    history = train_critic(critic, samples, epochs=args.epochs, lr=args.lr,
                           device=args.device, verbose=True)
    save_critic(critic, args.critic_out)
    print(f"训练完成，损失 {history[0]:.4f} -> {history[-1]:.4f}，权重 -> {args.critic_out}")


if __name__ == "__main__":
    main()
