"""Data scaling robustness experiments.

Stage A re-runs the historical pool-size scaling check with training-consistent
history normalization. Stage B utilities are added incrementally by the plan.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sys

sys.path.insert(0, ".")

from critic.model import DualHeadCritic, FundamentalCritic
from critic.reranker import CriticReranker
from critic.schemas import CriticSample
from critic.train_critic import (
    compute_coarse_trading_distances,
    compute_distances,
    compute_hybrid_trading_distances,
    load_critic,
    pairwise_ranking_loss,
    save_critic,
    train_critic,
)


def normalize_history(hist, clip=5.0):
    hist = np.asarray(hist, dtype=np.float32)
    mean = hist.mean(axis=0, keepdims=True)
    std = hist.std(axis=0, keepdims=True) + 1e-5
    return np.clip((hist - mean) / std, -clip, clip).astype(np.float32)


def normalize_future_from_history(hist, y_true):
    hist = np.asarray(hist, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.float32)
    mean = hist.mean(axis=0, keepdims=True)
    std = hist.std(axis=0, keepdims=True) + 1e-5
    return ((y_true - mean) / std).astype(np.float32)


def pool_slices(full_count, new_pool_count, windows_per_stock=15):
    all_count = full_count + new_pool_count
    return [
        {"label": "G1_31", "stock_count": 31, "indices": list(range(full_count))},
        {"label": "G2_61", "stock_count": 61, "indices": list(range(full_count + 30 * windows_per_stock))},
        {"label": "G3_91", "stock_count": 91, "indices": list(range(full_count + 60 * windows_per_stock))},
        {"label": "G4_117", "stock_count": 117, "indices": list(range(all_count))},
    ]


def load_cache(path):
    with open(path, "rb") as handle:
        cache = pickle.load(handle)
    required = {"windows", "candidates", "metadata"}
    if set(cache) != required:
        raise ValueError(f"{path} keys={sorted(cache)} expected={sorted(required)}")
    return cache


def cache_to_critic_samples(cache, pool_label):
    samples = []
    for window, candidates in zip(cache["windows"], cache["candidates"]):
        code, x_df, _, _, y_true = window
        hist_raw = x_df[["open", "high", "low", "close", "volume", "amount"]].values.astype(np.float32)
        hist_norm = normalize_history(hist_raw)
        y_true_norm = normalize_future_from_history(hist_raw, y_true)
        paths = np.asarray(candidates, dtype=np.float32)
        distances = compute_hybrid_trading_distances(paths, hist_norm, y_true_norm)
        fine_distances = compute_distances(paths, y_true_norm)
        coarse_distances = compute_coarse_trading_distances(paths, hist_norm, y_true_norm)
        samples.append(CriticSample(
            paths=paths,
            hist=hist_norm,
            corr_feat=None,
            distances=distances,
            code=str(code),
            y_true=y_true_norm,
            fine_distances=fine_distances,
            coarse_distances=coarse_distances,
        ))
    return samples


def samples_from_cache(cache):
    rows = []
    for window, candidates in zip(cache["windows"], cache["candidates"]):
        code, x_df, _, _, y_true = window
        hist_raw = x_df[["open", "high", "low", "close", "volume", "amount"]].values.astype(np.float32)
        hist_norm = normalize_history(hist_raw)
        y_true_norm = normalize_future_from_history(hist_raw, y_true)
        paths = np.asarray(candidates, dtype=np.float32)
        distances = np.mean((paths[:, :, :4] - y_true_norm[None, :, :4]) ** 2, axis=(1, 2)).astype(np.float32)
        rows.append({
            "code": str(code),
            "paths": paths,
            "hist": hist_norm,
            "y_true": y_true_norm,
            "distances": distances,
        })
    return rows


def build_stage_b_pools(full_samples, new_samples):
    return {
        "G1_31": list(full_samples),
        "G4_117": list(full_samples) + list(new_samples),
    }


def stage_b_checkpoint_jobs(pools, seeds=(0, 1, 2)):
    models = ["dualhead_calibrated", "small_mlp_no_interaction"]
    return [
        {"pool": pool, "model": model, "seed": int(seed)}
        for pool in pools
        for model in models
        for seed in seeds
    ]


def build_stage_b_model(model_name):
    if model_name == "dualhead_calibrated":
        return DualHeadCritic(
            in_dim=6,
            path_corr_interaction=False,
            final_combiner="calibrated",
            final_fusion=True,
        )
    if model_name == "small_mlp_no_interaction":
        return FundamentalCritic(in_dim=6, setwise=False, path_corr_interaction=False)
    raise ValueError(f"unknown Stage B model: {model_name}")


def split_train_val(samples, seed, val_fraction=0.1):
    if not samples:
        return [], []
    indices = np.arange(len(samples))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_count = max(1, int(round(len(samples) * val_fraction))) if len(samples) > 1 else 0
    val_indices = set(int(i) for i in indices[:val_count])
    train = [sample for idx, sample in enumerate(samples) if idx not in val_indices]
    val = [sample for idx, sample in enumerate(samples) if idx in val_indices]
    return train, val


def write_stage_b_checkpoint_sidecars(job_dir, history, metadata):
    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (job_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def load_stage_b_samples(output_dir, pool):
    cache_path = Path(output_dir) / f"cache_{pool.lower()}.pkl"
    with open(cache_path, "rb") as handle:
        return pickle.load(handle)


def train_stage_b(args):
    output_dir = Path(args.output_dir)
    checkpoint_root = output_dir / "stage_b_checkpoints"
    seeds = [int(seed) for seed in args.stage_b_seeds.split(",") if seed.strip()]
    jobs = stage_b_checkpoint_jobs(["G1_31", "G4_117"], seeds=seeds)
    if args.stage_b_pool:
        jobs = [job for job in jobs if job["pool"] == args.stage_b_pool]
    if args.stage_b_model:
        jobs = [job for job in jobs if job["model"] == args.stage_b_model]
    rows = []
    for job in jobs:
        pool = job["pool"]
        model_name = job["model"]
        seed = job["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        samples = load_stage_b_samples(output_dir, pool)
        train_samples, val_samples = split_train_val(samples, seed=args.val_seed + seed, val_fraction=args.val_fraction)
        model = build_stage_b_model(model_name)
        job_dir = checkpoint_root / pool / model_name / f"seed_{seed}"
        best_path = job_dir / "best.pt"
        last_path = job_dir / "last.pt"
        if best_path.exists() and last_path.exists() and (job_dir / "metadata.json").exists() and not args.force:
            print(f"skip existing pool={pool} model={model_name} seed={seed}", flush=True)
            continue
        history = train_critic(
            model,
            train_samples,
            epochs=args.stage_b_epochs,
            lr=args.lr,
            device=args.device,
            verbose=True,
            val_samples=val_samples,
            restore_best=False,
            best_path=str(best_path),
            fine_loss_weight=args.fine_loss_weight if model_name == "dualhead_calibrated" else 0.0,
            coarse_loss_weight=args.coarse_loss_weight if model_name == "dualhead_calibrated" else 0.0,
        )
        save_critic(model, str(last_path))
        metadata = {
            "pool": pool,
            "model": model_name,
            "seed": seed,
            "val_seed": args.val_seed + seed,
            "epochs": args.stage_b_epochs,
            "lr": args.lr,
            "device": args.device,
            "train_sample_count": len(train_samples),
            "val_sample_count": len(val_samples),
            "total_sample_count": len(samples),
            "source_cache": str(output_dir / f"cache_{pool.lower()}.pkl"),
            "best_path": str(best_path),
            "last_path": str(last_path),
            "selected_for_eval": "best.pt",
            "selection_basis": "validation_loss_only_no_new_pool_II",
            "new_pool_II_used_for_training": False,
            "architecture": model_name,
            "fine_loss_weight": args.fine_loss_weight if model_name == "dualhead_calibrated" else 0.0,
            "coarse_loss_weight": args.coarse_loss_weight if model_name == "dualhead_calibrated" else 0.0,
        }
        write_stage_b_checkpoint_sidecars(job_dir, history, metadata)
        rows.append({
            "pool": pool,
            "model": model_name,
            "seed": seed,
            "train_sample_count": len(train_samples),
            "val_sample_count": len(val_samples),
            "best_epoch": history.get("best_epoch", "") if isinstance(history, dict) else "",
            "best_val_loss": history.get("best_val_loss", "") if isinstance(history, dict) else "",
            "best_path": str(best_path),
            "last_path": str(last_path),
        })
    if rows:
        audit_path = output_dir / f"stage_b_checkpoint_audit_{args.stage_b_pool or 'all'}_{args.stage_b_model or 'all'}.csv"
        write_csv(audit_path, rows)
    print(json.dumps({"stage_b_checkpoint_audit": rows}, ensure_ascii=False, indent=2))


def write_stage_b_cache_audit(output_dir, rows):
    write_csv(output_dir / "stage_b_cache_audit.csv", rows)


def train_stage_a_critic(samples, device, epochs, lr, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    critic = FundamentalCritic(in_dim=6, setwise=False, path_corr_interaction=False).to(device)
    optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
    order = np.arange(len(samples))
    final_loss = 0.0
    for _ in range(epochs):
        np.random.shuffle(order)
        total = 0.0
        for idx in order:
            sample = samples[int(idx)]
            paths = torch.from_numpy(np.ascontiguousarray(sample["paths"])).to(device)
            hist = torch.from_numpy(np.ascontiguousarray(sample["hist"])).to(device)
            hist = hist.unsqueeze(0).expand(paths.shape[0], -1, -1)
            distances = torch.from_numpy(np.ascontiguousarray(sample["distances"])).to(device)
            scores = critic(paths, hist)
            loss = pairwise_ranking_loss(scores, distances)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        final_loss = total / max(len(samples), 1)
    return critic, final_loss


def evaluate_stage_a(critic, holdout_samples, device, temperature=1.0):
    reranker = CriticReranker(mode="weighted", temperature=temperature)
    simple = []
    critic_weighted = []
    oracle = []
    for sample in holdout_samples:
        candidates = sample["paths"]
        y_true = sample["y_true"]
        simple_path = candidates.mean(axis=0)
        simple.append(float(np.mean((simple_path[:, :4] - y_true[:, :4]) ** 2)))
        scores = critic.score(candidates, sample["hist"])
        weighted_path = reranker.combine(candidates, scores)
        critic_weighted.append(float(np.mean((weighted_path[:, :4] - y_true[:, :4]) ** 2)))
        dists = np.mean((candidates[:, :, :4] - y_true[None, :, :4]) ** 2, axis=(1, 2))
        oracle.append(float(np.min(dists)))
    simple_mean = float(np.mean(simple))
    critic_mean = float(np.mean(critic_weighted))
    oracle_mean = float(np.mean(oracle))
    return {
        "simple_mean_ohlc_mse": simple_mean,
        "critic_weighted_ohlc_mse": critic_mean,
        "oracle_ohlc_mse": oracle_mean,
        "improvement_pct": (simple_mean - critic_mean) / simple_mean * 100.0,
    }


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_prediction_metrics(pred, y_true):
    pred = np.asarray(pred, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.float32)
    return {
        "ohlc_mse": float(np.mean((pred[:, :4] - y_true[:, :4]) ** 2)),
        "ohlc_mae": float(np.mean(np.abs(pred[:, :4] - y_true[:, :4]))),
        "six_feature_mse": float(np.mean((pred - y_true) ** 2)),
    }


def direction_accuracy(hist, pred, y_true):
    last_close = float(hist[-1, 3])
    pred_dir = np.sign(float(pred[-1, 3]) - last_close)
    true_dir = np.sign(float(y_true[-1, 3]) - last_close)
    return float(pred_dir == true_dir)


def new_pool_ii_eval_caches(cache_root):
    root = Path(cache_root) / "output/candidates"
    return {
        "2024": root / "new_pool_II_2024_N16.pkl",
        "2025": root / "new_pool_II_2025_N16.pkl",
        "2026H1": root / "new_pool_II_2026H1_N16.pkl",
    }


def load_trained_stage_b_model(output_dir, pool, model_name, seed, device):
    model = build_stage_b_model(model_name)
    checkpoint_path = Path(output_dir) / "stage_b_checkpoints" / pool / model_name / f"seed_{seed}" / "best.pt"
    load_critic(model, str(checkpoint_path), device=device)
    model.to(device)
    model.eval()
    return model, checkpoint_path


def append_strategy_metrics(rows, window_id, period, pool, model_name, seed, strategy, hist, pred, y_true):
    metrics = evaluate_prediction_metrics(pred, y_true)
    rows.append({
        "window_id": window_id,
        "period": period,
        "pool": pool,
        "model": model_name,
        "seed": seed,
        "strategy": strategy,
        **metrics,
        "direction_accuracy": direction_accuracy(hist, pred, y_true),
    })


def evaluate_stage_b(args):
    output_dir = Path(args.output_dir)
    seeds = [int(seed) for seed in args.stage_b_seeds.split(",") if seed.strip()]
    window_rows = []
    summary_rows = []
    rerankers = {
        "weighted": CriticReranker(mode="weighted", temperature=args.temperature),
        "top1": CriticReranker(mode="top1", temperature=args.temperature),
    }
    for period, cache_path in new_pool_ii_eval_caches(args.cache_root).items():
        eval_samples = cache_to_critic_samples(load_cache(cache_path), pool_label=f"new_pool_II_{period}")
        baseline_period_rows = []
        for window_id, sample in enumerate(eval_samples):
            candidates = sample.paths
            hist = sample.hist
            y_true = sample.y_true
            simple_pred = candidates.mean(axis=0)
            median_pred = np.median(candidates, axis=0)
            oracle_idx = int(np.argmin(np.mean((candidates[:, :, :4] - y_true[None, :, :4]) ** 2, axis=(1, 2))))
            oracle_pred = candidates[oracle_idx]
            for strategy, pred in [
                ("simple_mean", simple_pred),
                ("median", median_pred),
                ("oracle_ohlc", oracle_pred),
            ]:
                append_strategy_metrics(
                    baseline_period_rows,
                    window_id,
                    period,
                    "baseline",
                    "baseline",
                    -1,
                    strategy,
                    hist,
                    pred,
                    y_true,
                )
        for pool in ["G1_31", "G4_117"]:
            for model_name in ["dualhead_calibrated", "small_mlp_no_interaction"]:
                for seed in seeds:
                    model, checkpoint_path = load_trained_stage_b_model(output_dir, pool, model_name, seed, args.device)
                    learned_rows = []
                    for window_id, sample in enumerate(eval_samples):
                        scores = model.score(sample.paths, sample.hist, sample.corr_feat)
                        for strategy, reranker in rerankers.items():
                            pred = reranker.combine(sample.paths, scores)
                            append_strategy_metrics(
                                learned_rows,
                                window_id,
                                period,
                                pool,
                                model_name,
                                seed,
                                strategy,
                                sample.hist,
                                pred,
                                sample.y_true,
                            )
                    run_rows = list(baseline_period_rows) + learned_rows
                    window_rows.extend(run_rows)
                    by_strategy = {}
                    for row in run_rows:
                        by_strategy.setdefault(row["strategy"], []).append(row)
                    simple_mse = float(np.mean([row["ohlc_mse"] for row in by_strategy["simple_mean"]]))
                    for strategy, rows_for_strategy in sorted(by_strategy.items()):
                        ohlc_mse = float(np.mean([row["ohlc_mse"] for row in rows_for_strategy]))
                        summary_rows.append({
                            "period": period,
                            "pool": pool,
                            "model": model_name,
                            "seed": seed,
                            "strategy": strategy,
                            "checkpoint_path": str(checkpoint_path) if strategy in rerankers else "baseline",
                            "ohlc_mse": ohlc_mse,
                            "ohlc_mae": float(np.mean([row["ohlc_mae"] for row in rows_for_strategy])),
                            "six_feature_mse": float(np.mean([row["six_feature_mse"] for row in rows_for_strategy])),
                            "direction_accuracy": float(np.mean([row["direction_accuracy"] for row in rows_for_strategy])),
                            "improvement_pct_vs_simple_mean": (simple_mse - ohlc_mse) / simple_mse * 100.0,
                        })
                    print(f"evaluated period={period} pool={pool} model={model_name} seed={seed}", flush=True)
    write_csv(output_dir / "stage_b_window_metrics.csv", window_rows)
    write_csv(output_dir / "stage_b_per_seed.csv", summary_rows)
    print(json.dumps({"stage_b_per_seed_rows": len(summary_rows), "stage_b_window_rows": len(window_rows)}, ensure_ascii=False, indent=2))


def bootstrap_mean_ci(values, seed=0, n_boot=2000, alpha=0.05):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = rng.choice(values, size=values.size, replace=True)
        means[i] = float(np.mean(sample))
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def aggregate_stage_b_summary(rows):
    grouped = {}
    for row in rows:
        key = (row["period"], row["pool"], row["model"], row["strategy"])
        grouped.setdefault(key, []).append(row)
    summary = []
    for (period, pool, model, strategy), items in sorted(grouped.items()):
        mse = np.asarray([float(row["ohlc_mse"]) for row in items], dtype=np.float64)
        improvements = np.asarray([float(row.get("improvement_pct_vs_simple_mean", 0.0)) for row in items], dtype=np.float64)
        win_count = int(np.sum(improvements > 0))
        summary.append({
            "period": period,
            "pool": pool,
            "model": model,
            "strategy": strategy,
            "mean_ohlc_mse": float(np.mean(mse)),
            "std_ohlc_mse": float(np.std(mse, ddof=0)),
            "min_ohlc_mse": float(np.min(mse)),
            "max_ohlc_mse": float(np.max(mse)),
            "mean_improvement_pct_vs_simple_mean": float(np.mean(improvements)),
            "win_count": win_count,
            "positive_seed_ratio": float(win_count / len(items)),
            "seed_count": len(items),
        })
    return summary


def compute_stage_b_paired_deltas(rows):
    keyed = {}
    for row in rows:
        if row["pool"] not in {"G1_31", "G4_117"}:
            continue
        key = (row["period"], row["model"], row["strategy"], int(row["seed"]))
        keyed.setdefault(key, {})[row["pool"]] = row
    deltas = []
    for (period, model, strategy, seed), pair in sorted(keyed.items()):
        if "G1_31" not in pair or "G4_117" not in pair:
            continue
        g1 = float(pair["G1_31"]["ohlc_mse"])
        g4 = float(pair["G4_117"]["ohlc_mse"])
        deltas.append({
            "period": period,
            "model": model,
            "strategy": strategy,
            "seed": seed,
            "g1_ohlc_mse": g1,
            "g4_ohlc_mse": g4,
            "g4_minus_g1_ohlc_mse": g4 - g1,
        })
    return deltas


def run_stage_a(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_cache = load_cache(Path(args.cache_root) / "output/candidates/full86_2024_N16.pkl")
    new_cache = load_cache(Path(args.cache_root) / "output/candidates/new_pool_2024_N16.pkl")
    holdout_cache = load_cache(Path(args.cache_root) / "output/candidates/holdout_2025_N16.pkl")
    full_samples = samples_from_cache(full_cache)
    new_samples = samples_from_cache(new_cache)
    holdout_samples = samples_from_cache(holdout_cache)
    all_samples = full_samples + new_samples
    rows = []
    metadata = {
        "seed": args.seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "device": args.device,
        "full_samples": len(full_samples),
        "new_pool_samples": len(new_samples),
        "holdout_windows": len(holdout_samples),
        "history_input": "hist_norm_clip_5",
        "new_pool_II_used_for_training": False,
    }
    for group in pool_slices(len(full_samples), len(new_samples)):
        pool_samples = [all_samples[i] for i in group["indices"]]
        critic, final_loss = train_stage_a_critic(pool_samples, args.device, args.epochs, args.lr, args.seed)
        metrics = evaluate_stage_a(critic, holdout_samples, args.device, args.temperature)
        rows.append({
            "group": group["label"],
            "stock_count": group["stock_count"],
            "train_samples": len(pool_samples),
            "train_loss": final_loss,
            **metrics,
        })
    write_csv(output_dir / "stage_a_scaling_summary.csv", rows)
    (output_dir / "stage_a_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": rows, "metadata": metadata}, ensure_ascii=False, indent=2))




def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_stage_b(args):
    output_dir = Path(args.output_dir)
    rows = read_csv_rows(output_dir / "stage_b_per_seed.csv")
    summary_rows = aggregate_stage_b_summary(rows)
    delta_rows = compute_stage_b_paired_deltas(rows)
    ci_rows = []
    grouped = {}
    for row in delta_rows:
        key = (row["period"], row["model"], row["strategy"])
        grouped.setdefault(key, []).append(float(row["g4_minus_g1_ohlc_mse"]))
    for (period, model, strategy), values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=np.float64)
        lo, hi = bootstrap_mean_ci(arr, seed=20260724, n_boot=args.bootstrap_samples)
        ci_rows.append({
            "period": period,
            "model": model,
            "strategy": strategy,
            "mean_g4_minus_g1_ohlc_mse": float(np.mean(arr)),
            "ci95_low": lo,
            "ci95_high": hi,
            "g4_better_seed_count": int(np.sum(arr < 0)),
            "seed_count": int(arr.size),
        })
    write_csv(output_dir / "stage_b_summary.csv", summary_rows)
    write_csv(output_dir / "stage_b_paired_deltas.csv", delta_rows)
    write_csv(output_dir / "stage_b_paired_delta_ci.csv", ci_rows)
    print(json.dumps({
        "stage_b_summary_rows": len(summary_rows),
        "stage_b_paired_delta_rows": len(delta_rows),
        "stage_b_paired_delta_ci_rows": len(ci_rows),
    }, ensure_ascii=False, indent=2))


def prepare_stage_b(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_cache = load_cache(Path(args.cache_root) / "output/candidates/full86_2024_N16.pkl")
    new_cache = load_cache(Path(args.cache_root) / "output/candidates/new_pool_2024_N16.pkl")
    full_samples = cache_to_critic_samples(full_cache, pool_label="G1_31")
    new_samples = cache_to_critic_samples(new_cache, pool_label="new_pool_86")
    pools = build_stage_b_pools(full_samples, new_samples)
    audit_rows = []
    for pool, samples in pools.items():
        cache_path = output_dir / f"cache_{pool.lower()}.pkl"
        with open(cache_path, "wb") as handle:
            pickle.dump(samples, handle)
        codes = sorted({sample.code for sample in samples})
        audit_rows.append({
            "pool": pool,
            "cache_path": str(cache_path),
            "sample_count": len(samples),
            "stock_count": len(codes),
            "first_code": codes[0] if codes else "",
            "last_code": codes[-1] if codes else "",
            "schema_fields": "paths,hist,corr_feat,distances,fine_distances,coarse_distances,y_true,code",
            "uses_new_pool_II_for_training": False,
        })
    write_stage_b_cache_audit(output_dir, audit_rows)
    print(json.dumps({"stage_b_cache_audit": audit_rows}, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["a", "prepare-b", "train-b", "eval-b", "summarize-b"], default="a")
    parser.add_argument("--cache-root", default="D:/Github_Storage/Monos")
    parser.add_argument("--output-dir", default="output/results/data_scaling_robustness")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--stage-b-seeds", default="0,1,2")
    parser.add_argument("--stage-b-epochs", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--val-seed", type=int, default=1000)
    parser.add_argument("--fine-loss-weight", type=float, default=0.5)
    parser.add_argument("--coarse-loss-weight", type=float, default=0.5)
    parser.add_argument("--stage-b-pool", choices=["G1_31", "G4_117"], default=None)
    parser.add_argument("--stage-b-model", choices=["dualhead_calibrated", "small_mlp_no_interaction"], default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.stage == "a":
        run_stage_a(args)
    elif args.stage == "prepare-b":
        prepare_stage_b(args)
    elif args.stage == "train-b":
        train_stage_b(args)
    elif args.stage == "eval-b":
        evaluate_stage_b(args)
    elif args.stage == "summarize-b":
        summarize_stage_b(args)


if __name__ == "__main__":
    main()
