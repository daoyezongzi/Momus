"""全域 Critic 训练（离线 + GPU 批处理高效版）。

目的：验证"critic 池内有效、池外失效"是训练域覆盖问题——把全部 86 只
2022-2023 数据都纳入训练，再看 2024 回测是否在池外也变有效。

两个关键优化（相对 train_critic_pipeline.py）：
1. 离线：目标股历史 + 关联股都从本地 CSV 读，零联网、零 sleep。
2. GPU 批处理：把一只股票的全部窗口 stack 成一个大 batch（K×sample_count）
   一次喂给 auto_regressive_inference，而非每窗口单独 batch=16，填满 GPU。
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append(".")

from critic import DualHeadCritic, FundamentalCritic, CriticSample, compute_distances, train_critic
from critic.train_critic import (
    compute_coarse_trading_distances,
    compute_directional_trading_distances,
    compute_hybrid_trading_distances,
    compute_return_weighted_distances,
)
from critic.config import get_correlated_symbols, MAX_CORRELATED
from critic.train_critic import save_critic
from model.kronos import auto_regressive_inference, calc_time_stamps

COLS = ["open", "high", "low", "close", "volume", "amount"]


def load_local(code, data_dir):
    f = os.path.join(data_dir, f"{code}.csv")
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    return df


def make_windows(df, lookback, pred_len, n_windows):
    """按时间等距切 n_windows 个 (hist, future) 窗口。"""
    total = len(df)
    if total < lookback + pred_len:
        return []
    last_start = total - lookback - pred_len
    starts = ([last_start] if n_windows == 1
              else np.linspace(0, last_start, n_windows).astype(int))
    wins = []
    for s in np.unique(starts):
        hist = df.iloc[s:s + lookback]
        fut = df.iloc[s + lookback:s + lookback + pred_len]
        wins.append((hist, fut))
    return wins


def corr_feat_local(code, T, lookback, data_dir):
    """从本地 CSV 读关联股收益率特征 (n_corr, lookback, 6)，截止 T。None=无关联。"""
    syms = get_correlated_symbols(code)
    if not syms:
        return None
    feats = []
    for sym in syms[:MAX_CORRELATED]:
        df = load_local(sym, data_dir)
        if df is None:
            feats.append(np.zeros((lookback, 6), dtype=np.float32))
            continue
        df = df[df["timestamps"] <= T]
        vals = df[COLS].values.astype(np.float32)
        rets = np.zeros((len(vals), 6), dtype=np.float32)
        if len(vals) > 1:
            rets[1:] = vals[1:] / (vals[:-1] + 1e-8) - 1.0
        if len(rets) >= lookback:
            rets = rets[-lookback:]
        else:
            rets = np.vstack([np.zeros((lookback - len(rets), 6), np.float32), rets])
        feats.append(rets)
    return np.stack(feats, axis=0)


def build_samples_batched(model, tokenizer, code, windows, pred_len, dev,
                          sample_count, max_context, clip, data_dir, batch_windows,
                          reward_mode="distance", include_granularity_distances=False):
    """把一只股票的多个窗口拼成大 batch 一次推理，填满 GPU。

    auto_regressive_inference 支持前导 batch 维 K：内部扩成 K×sample_count，
    末尾 reshape 回 (K, sample_count, seq, feat)。相比逐窗口 batch=16，
    K 个窗口一起算，GPU 利用率显著提升。
    """
    samples = []
    for base in range(0, len(windows), batch_windows):
        chunk = windows[base:base + batch_windows]
        x_norms, x_stamps, y_stamps, metas = [], [], [], []
        for hist, fut in chunk:
            x = hist[COLS].values.astype(np.float32)
            mean, std = x.mean(0), x.std(0)
            x_norm = np.clip((x - mean) / (std + 1e-5), -clip, clip)
            x_norms.append(x_norm)
            x_stamps.append(calc_time_stamps(hist["timestamps"]).values.astype(np.float32))
            y_stamps.append(calc_time_stamps(fut["timestamps"]).values.astype(np.float32))
            metas.append((hist, fut, mean, std, x_norm))
        xb = torch.from_numpy(np.stack(x_norms)).to(dev)
        xsb = torch.from_numpy(np.stack(x_stamps)).to(dev)
        ysb = torch.from_numpy(np.stack(y_stamps)).to(dev)
        preds = auto_regressive_inference(
            tokenizer, model, xb, xsb, ysb, max_context, pred_len, clip,
            1.0, 0, 0.9, sample_count, False, return_individual=True,
        )  # (K, sample_count, seq, feat)
        for k, (hist, fut, mean, std, x_norm) in enumerate(metas):
            paths = preds[k][:, -pred_len:, :]
            y_true = fut[COLS].values.astype(np.float32)
            y_norm = (y_true - mean) / (std + 1e-5)
            if reward_mode == "distance":
                distances = compute_distances(paths, y_norm)
            elif reward_mode == "direction":
                distances = compute_directional_trading_distances(paths, x_norm, y_norm, long_only=True)
            elif reward_mode == "return_weighted":
                distances = compute_return_weighted_distances(paths, x_norm, y_norm, long_only=True)
            elif reward_mode == "hybrid":
                distances = compute_hybrid_trading_distances(paths, x_norm, y_norm)
            else:
                raise ValueError(f"unknown reward_mode: {reward_mode}")
            fine_distances = None
            coarse_distances = None
            if include_granularity_distances:
                fine_distances = compute_distances(paths, y_norm)
                coarse_distances = compute_coarse_trading_distances(paths, x_norm, y_norm)
            T = hist["timestamps"].max()
            cf = corr_feat_local(code, T, len(hist), data_dir)
            samples.append(CriticSample(paths=paths, hist=x_norm, corr_feat=cf,
                                        distances=distances, fine_distances=fine_distances,
                                        coarse_distances=coarse_distances, code=code))
    return samples


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/market/train")
    p.add_argument("--codes-file", default=None)
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--pred-len", type=int, default=10)
    p.add_argument("--n-windows", type=int, default=20)
    p.add_argument("--sample-count", type=int, default=16)
    p.add_argument("--batch-windows", type=int, default=16,
                   help="每次推理拼多少窗口(K)。GPU batch=K×sample_count。")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--critic-arch", choices=["fundamental", "dual_head"], default="fundamental")
    p.add_argument("--final-combiner", choices=["explicit", "residual", "calibrated"], default="explicit")
    p.add_argument("--setwise", action="store_true",
                   help="启用 Deep-Sets 跨候选混合(setwise 打分)。")
    p.add_argument("--no-path-corr-interaction", action="store_true",
                   help="关闭 path⊙corr 乘性交互。用于训练 small_mlp baseline"
                        "(单头、无双颗粒度、无 setwise、无 corr 交互)。默认关闭=保持原行为。")
    p.add_argument("--fine-loss-weight", type=float, default=0.0)
    p.add_argument("--coarse-loss-weight", type=float, default=0.0)
    p.add_argument(
        "--reward-mode",
        choices=["distance", "direction", "return_weighted", "hybrid"],
        default="distance",
        help="训练标签模式：原始路径 MSE、方向交易 reward、收益幅度加权 reward、或双颗粒度 hybrid reward。",
    )
    p.add_argument("--val-ratio", type=float, default=0.0,
                   help="训练样本验证集比例；0 表示不做验证监控。")
    p.add_argument("--val-seed", type=int, default=0, help="验证集划分随机种子。")
    p.add_argument("--history-out", default=None, help="训练/验证 loss 历史输出 CSV。")
    p.add_argument("--restore-best", action="store_true",
                   help="训练结束后恢复验证集 loss 最低 epoch 的权重。")
    p.add_argument("--best-critic-out", default=None,
                   help="额外保存 best-val 权重；未设置时只保存 --critic-out。")
    p.add_argument("--best-metric", choices=["val_loss"], default="val_loss",
                   help="选择 best checkpoint 的指标；当前仅支持 val_loss。")
    p.add_argument("--save-epochs", default=None,
                   help="逗号分隔的 epoch 列表；训练到这些 epoch 时保存 checkpoint。")
    p.add_argument("--checkpoint-dir", default=None,
                   help="--save-epochs 的 checkpoint 输出目录。")
    p.add_argument("--device", default="cuda")
    p.add_argument("--critic-out", default="critic_full86.pt")
    p.add_argument("--samples-cache", default="critic_samples_full86.pkl")
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--train-only", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.codes_file:
        with open(args.codes_file) as f:
            codes = [ln.strip() for ln in f if ln.strip()]
    else:
        codes = [os.path.basename(x)[:-4]
                 for x in glob.glob(os.path.join(args.data_dir, "*.csv"))]

    from critic.train_critic import save_samples, load_samples

    if not args.train_only:
        from model import Kronos, KronosTokenizer
        print(f"加载 Kronos... device={args.device}")
        tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
        model = model.to(args.device).eval()
        tokenizer = tokenizer.to(args.device).eval()
        max_context, clip = 512, 5

        all_samples = []
        for ci, code in enumerate(codes):
            df = load_local(code, args.data_dir)
            if df is None:
                print(f"[{ci+1}/{len(codes)}] {code} 本地缺失，跳过")
                continue
            wins = make_windows(df, args.lookback, args.pred_len, args.n_windows)
            s = build_samples_batched(
                model, tokenizer, code, wins, args.pred_len, args.device,
                args.sample_count, max_context, clip, args.data_dir, args.batch_windows,
                reward_mode=args.reward_mode,
                include_granularity_distances=args.critic_arch == "dual_head")
            all_samples.extend(s)
            print(f"[{ci+1}/{len(codes)}] {code}: {len(s)} 样本 (累计 {len(all_samples)})",
                  flush=True)
        save_samples(all_samples, args.samples_cache)
        print(f"已缓存 {len(all_samples)} 样本 -> {args.samples_cache}")
        if args.build_only:
            return
    else:
        all_samples = load_samples(args.samples_cache)
        print(f"加载 {len(all_samples)} 样本")

    path_corr_interaction = not args.no_path_corr_interaction
    critic = (DualHeadCritic(in_dim=6, final_combiner=args.final_combiner, setwise=args.setwise,
                             path_corr_interaction=path_corr_interaction)
              if args.critic_arch == "dual_head"
              else FundamentalCritic(in_dim=6, setwise=args.setwise,
                                     path_corr_interaction=path_corr_interaction))
    train_samples = all_samples
    val_samples = None
    if args.val_ratio > 0:
        if not 0 < args.val_ratio < 1:
            raise ValueError("--val-ratio must be between 0 and 1")
        rng = np.random.default_rng(args.val_seed)
        indices = rng.permutation(len(all_samples))
        val_size = max(1, int(round(len(all_samples) * args.val_ratio)))
        val_idx = set(indices[:val_size])
        train_samples = [sample for i, sample in enumerate(all_samples) if i not in val_idx]
        val_samples = [sample for i, sample in enumerate(all_samples) if i in val_idx]
        print(f"验证集划分：train={len(train_samples)}, val={len(val_samples)}, val_ratio={args.val_ratio:.2f}")

    if args.best_critic_out and val_samples is None:
        raise ValueError("--best-critic-out requires --val-ratio > 0")
    save_epochs = None
    if args.save_epochs:
        if not args.checkpoint_dir:
            raise ValueError("--checkpoint-dir requires --save-epochs")
        save_epochs = {int(x) for x in args.save_epochs.split(",") if x.strip()}

    print(f"训练 Critic：{args.epochs} epochs, {len(train_samples)} 训练样本, device={args.device}")
    hist = train_critic(critic, train_samples, epochs=args.epochs, lr=args.lr,
                        device=args.device, verbose=True, val_samples=val_samples,
                        restore_best=args.restore_best,
                        best_path=args.best_critic_out,
                        save_epochs=save_epochs,
                        checkpoint_dir=args.checkpoint_dir,
                        fine_loss_weight=args.fine_loss_weight,
                        coarse_loss_weight=args.coarse_loss_weight)
    save_critic(critic, args.critic_out)
    if args.best_critic_out:
        print(f"best-val 权重 -> {args.best_critic_out}")
    if args.history_out:
        if isinstance(hist, dict):
            hist_df = pd.DataFrame({
                "epoch": np.arange(1, len(hist["train_loss"]) + 1),
                "train_loss": hist["train_loss"],
            })
            if "val_loss" in hist:
                hist_df["val_loss"] = hist["val_loss"]
            if "best_epoch" in hist:
                hist_df["is_best"] = hist_df["epoch"] == hist["best_epoch"]
                hist_df["best_epoch"] = hist["best_epoch"]
                hist_df["best_val_loss"] = hist["best_val_loss"]
                if "restored_val_loss" in hist:
                    hist_df["restored_val_loss"] = hist["restored_val_loss"]
        else:
            hist_df = pd.DataFrame({
                "epoch": np.arange(1, len(hist) + 1),
                "train_loss": hist,
            })
        hist_df.to_csv(args.history_out, index=False)
        print(f"训练历史 -> {args.history_out}")
    if isinstance(hist, dict):
        if "val_loss" in hist:
            print(f"完成：loss {hist['train_loss'][0]:.4f} -> {hist['train_loss'][-1]:.4f}, "
                  f"val_loss {hist['val_loss'][0]:.4f} -> {hist['val_loss'][-1]:.4f}, 权重 -> {args.critic_out}")
        else:
            print(f"完成：loss {hist['train_loss'][0]:.4f} -> {hist['train_loss'][-1]:.4f}, 权重 -> {args.critic_out}")
    else:
        print(f"完成：loss {hist[0]:.4f} -> {hist[-1]:.4f}, 权重 -> {args.critic_out}")


if __name__ == "__main__":
    main()
