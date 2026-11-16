"""Critic 重排序消融对比实验（设计文档 §8 E1/E3）。

在同一批候选路径上对比三种策略的预测精度：
  - simple_mean : baseline，等价原 Kronos 简单平均
  - critic_weighted : Critic softmax 加权
  - critic_top1 : Critic 选最高分单条

用法：
    # 合成数据冒烟（仅验证流程）
    python examples/run_critic_comparison.py --synthetic --n-samples 5
    # 真实数据 + 训练好的 Critic（产出论文核心结果表）
    python examples/run_critic_comparison.py --real --codes-file data/references/codes.txt \
        --critic-ckpt critic_trained.pt --start 20240101 --end 20241231

数据泄漏红线：测试时段（--start/--end，默认 2024 全年）必须与训练时段
（train_critic_pipeline 用的 2022~2023）完全不重叠，才是真正的样本外评估。
真实模式会接入 corr_provider，Critic 才能用到关联标的特征。
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append("../")
sys.path.append(".")

from critic import (
    CriticKronosPredictor, FundamentalCritic, RandomCritic, CriticReranker,
    CorrelatedDataProvider, make_tushare_fetch,
    strategy_simple_mean, strategy_median, make_critic_strategy,
    make_prob_top1_strategy,
    evaluate_strategies, aggregate_results, format_comparison,
)
from examples.train_critic_pipeline import (
    _load_dotenv, fetch_target_history, make_walk_forward_windows, _load_codes,
)


def make_synth_kline(n_rows, seed=0):
    """合成 K 线（几何随机游走 + 合理 OHLC）。"""
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


def build_synth_windows(n_samples, lookback, pred_len):
    """合成评估窗口：每个样本一段历史 + 紧随的真实未来（code=None）。"""
    windows = []
    for i in range(n_samples):
        df = make_synth_kline(lookback + pred_len, seed=100 + i)
        x_df = df.iloc[:lookback][["open", "high", "low", "close", "volume", "amount"]]
        x_ts = df.iloc[:lookback]["timestamps"]
        y_ts = df.iloc[lookback:lookback + pred_len]["timestamps"]
        y_true = df.iloc[lookback:lookback + pred_len][
            ["open", "high", "low", "close", "volume", "amount"]].values.astype(np.float32)
        windows.append((None, x_df, x_ts, y_ts, y_true))
    return windows


def build_real_windows(args, fetch_fn):
    """真实评估窗口：从 tushare 抓测试时段数据并 walk-forward 切窗。

    每个窗口附带 target_code，供 corr_provider 抓关联标的。测试时段
    （args.start/end）应与训练时段不重叠，保证样本外评估。
    """
    codes = _load_codes(args)
    windows = []
    for ci, code in enumerate(codes):
        print(f"[{ci + 1}/{len(codes)}] 抓取测试段 {code} ...")
        df = fetch_target_history(code, args.start, args.end, fetch_fn=fetch_fn)
        if df is None:
            print(f"  跳过 {code}（抓取失败）")
            continue
        cols = ["open", "high", "low", "close", "volume", "amount"]
        for w in make_walk_forward_windows(df, args.lookback, args.pred_len, args.n_windows):
            windows.append((code, w["x_df"], w["x_ts"], w["y_ts"], w["y_true"]))
    return windows


def run(args):
    from model import Kronos, KronosTokenizer

    # ---- 数据源与关联特征提供器（仅真实模式需要）----
    fetch_fn = None
    corr_provider = None
    if args.real:
        _load_dotenv()
        token = args.tushare_token or os.environ.get("TUSHARE_TOKEN")
        if token:
            fetch_fn = make_tushare_fetch(token)
            print("数据源: tushare")
        else:
            print("数据源: 东方财富（未提供 tushare token）")
        corr_provider = CorrelatedDataProvider(
            lookback=args.lookback, fetch_fn=fetch_fn,
            request_delay=0 if fetch_fn else 1.5)

    print("加载 Kronos 模型...")
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small")

    critic = FundamentalCritic(in_dim=6)
    if args.critic_ckpt:
        import torch
        critic.load_state_dict(torch.load(args.critic_ckpt, map_location=args.device))
        print(f"已加载 Critic 权重: {args.critic_ckpt}")
    elif args.random_critic:
        critic = RandomCritic(seed=0)
        print("使用 RandomCritic（未训练，仅验证流程）")

    predictor = CriticKronosPredictor(
        model, tokenizer, critic=critic,
        reranker=CriticReranker(mode="weighted", temperature=args.temperature),
        corr_provider=corr_provider,
        device=args.device,
    )

    # 定义对比策略（在同一批候选上评估）
    strategies = {
        "simple_mean": strategy_simple_mean,
        "median": strategy_median,
        "critic_weighted": make_critic_strategy(
            critic, CriticReranker(mode="weighted", temperature=args.temperature)),
        "critic_top1": make_critic_strategy(critic, CriticReranker(mode="top1")),
    }

    if args.real:
        windows = build_real_windows(args, fetch_fn)
    else:
        windows = build_synth_windows(args.n_samples, args.lookback, args.pred_len)
    if not windows:
        print("未构造出任何评估窗口，终止。")
        return

    per_sample = []
    for i, (code, x_df, x_ts, y_ts, y_true) in enumerate(windows):
        # 关键：生成一次候选，供所有策略共享 -> 公平对比
        paths, hist_norm, corr_feat, (x_mean, x_std), log_probs = predictor.generate_paths(
            x_df, x_ts, y_ts, args.pred_len, sample_count=args.sample_count,
            verbose=False, target_code=code, return_log_probs=True)
        # prob_top1 需按窗口的 log_probs 构建（生成器最自信路径）
        window_strategies = dict(strategies)
        window_strategies["prob_top1"] = make_prob_top1_strategy(log_probs)
        # y_true 归一化到候选同一空间
        y_true_norm = (y_true - x_mean) / (x_std + 1e-5)
        res = evaluate_strategies(paths, y_true_norm, window_strategies,
                                  hist=hist_norm, corr_feat=corr_feat)
        per_sample.append(res)
        print(f"  窗口 {i + 1}/{len(windows)} 完成"
              + (f"（{code}）" if code else ""))

    agg = aggregate_results(per_sample)
    print("\n" + "=" * 62)
    print(f"消融对比结果（{len(windows)} 窗口, N={args.sample_count}）")
    print("=" * 62)
    print(format_comparison(agg, baseline="simple_mean"))
    if args.real:
        print(f"\n测试时段: {args.start}~{args.end}（样本外，与训练段不重叠）")
    else:
        print("\n注: 合成数据 + 未训练 Critic，结果仅验证流程；真实结论需真实数据训练后得出。")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true", help="使用合成数据（默认）")
    p.add_argument("--real", action="store_true", help="用真实数据（tushare）跑样本外评估")
    p.add_argument("--random-critic", action="store_true", help="用随机Critic验证流程")
    p.add_argument("--critic-ckpt", type=str, default=None, help="训练好的Critic权重路径")
    p.add_argument("--codes", type=str, default="300750,002594")
    p.add_argument("--codes-file", type=str, default=None)
    p.add_argument("--tushare-token", type=str, default=None,
                   help="tushare token；默认读环境变量 TUSHARE_TOKEN")
    p.add_argument("--start", type=str, default="20240101", help="测试时段起（默认2024，样本外）")
    p.add_argument("--end", type=str, default="20241231", help="测试时段止")
    p.add_argument("--n-samples", type=int, default=5, help="合成模式的样本数")
    p.add_argument("--n-windows", type=int, default=5, help="真实模式每只股票的切窗数")
    p.add_argument("--sample-count", type=int, default=16)
    p.add_argument("--lookback", type=int, default=90)
    p.add_argument("--pred-len", type=int, default=10)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cpu")
    run(p.parse_args())
