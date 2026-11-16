"""语义分解归因消融：8 臂 x 5 seed 训练与评估驱动。

预注册协议：docs/development/protocols/2026-07-28_semantic_decomposition_ablation_preregistration.md

臂定义（全部 path_corr_interaction=False、setwise=False、final_combiner=calibrated）：
  A0 single_head            : 无辅助头，基线
  A1 single_head_wide       : 单头加宽至与 B0 参数量匹配，排除"只是容量更大"
  B0 dual_head              : fine+coarse 真目标，完整设计
  C1 dual_head_dup_final    : 辅助目标均 = final 目标，排除"只是多任务正则"
  C2 dual_head_shuffled     : 辅助目标组内打乱，排除"只是梯度噪声"
  C3 dual_head_swapped      : fine<->coarse 互换，检验语义配对
  D1 dual_head_fine_only    : 仅 fine 监督（coarse head 存在但无 loss）
  D2 dual_head_coarse_only  : 仅 coarse 监督

C1/C2/C3 通过切换训练 cache 实现（见 build_ablation_label_caches.py），
D1/D2 通过置零对应 loss weight 实现，因此三者参数量与 B0 完全相同。
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")

from critic.model import DualHeadCritic, FundamentalCritic
from critic.train_critic import train_critic

DEFAULT_SOURCE_CACHE = "output/results/regime_generalization/samples_multiregime_train_dual.pkl"
ABLATION_DIR = "output/results/semantic_decomposition_ablation"

# arm -> (kind, cache_key, fine_weight_on, coarse_weight_on)
ARMS = {
    "A0_single_head":           ("single", "source", False, False),
    "A1_single_head_wide":      ("single_wide", "source", False, False),
    "B0_dual_head":             ("dual", "source", True, True),
    "C1_dual_head_dup_final":   ("dual", "C1_duplicate_final", True, True),
    "C2_dual_head_shuffled":    ("dual", "C2_shuffled", True, True),
    "C3_dual_head_swapped":     ("dual", "C3_swapped", True, True),
    "D1_dual_head_fine_only":   ("dual", "source", True, False),
    "D2_dual_head_coarse_only": ("dual", "source", False, True),
}


def param_count(model):
    return sum(p.numel() for p in model.parameters())


def build_dual():
    return DualHeadCritic(
        in_dim=6,
        path_corr_interaction=False,
        final_combiner="calibrated",
        final_fusion=True,
        setwise=False,
    )


def solve_wide_head_dim(target_params, lo=64, hi=512):
    """穷举求使 single-head 总参数量最接近 target 的 head_hidden_dim。"""
    best, best_gap = lo, None
    for h in range(lo, hi + 1):
        gap = abs(param_count(FundamentalCritic(
            in_dim=6, path_corr_interaction=False, head_hidden_dim=h)) - target_params)
        if best_gap is None or gap < best_gap:
            best, best_gap = h, gap
    return best


def build_arm_model(arm):
    kind = ARMS[arm][0]
    if kind == "single":
        return FundamentalCritic(in_dim=6, setwise=False, path_corr_interaction=False)
    if kind == "single_wide":
        width = solve_wide_head_dim(param_count(build_dual()))
        return FundamentalCritic(
            in_dim=6, setwise=False, path_corr_interaction=False, head_hidden_dim=width)
    if kind == "dual":
        return build_dual()
    raise ValueError(f"unknown arm kind: {kind}")


def cache_path_for(arm, source_cache, cache_dir):
    key = ARMS[arm][1]
    if key == "source":
        return Path(source_cache)
    return Path(cache_dir) / f"cache_{key}.pkl"


def load_samples(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def split_train_val(samples, seed, val_fraction=0.2):
    indices = np.arange(len(samples))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_count = max(1, int(round(len(samples) * val_fraction)))
    val_indices = set(int(i) for i in indices[:val_count])
    train = [s for i, s in enumerate(samples) if i not in val_indices]
    val = [s for i, s in enumerate(samples) if i in val_indices]
    return train, val


def train_one(arm, seed, args):
    _, cache_key, fine_on, coarse_on = ARMS[arm]
    job_dir = Path(args.output_dir) / "checkpoints" / arm / f"seed_{seed}"
    best_path = job_dir / "best.pt"
    last_path = job_dir / "last.pt"
    meta_path = job_dir / "metadata.json"
    if best_path.exists() and last_path.exists() and meta_path.exists() and not args.force:
        print(f"skip existing arm={arm} seed={seed}", flush=True)
        return None

    # 真 seed 控制：权重初始化、batch shuffle、val split 共用同一 seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cache = cache_path_for(arm, args.source_cache, args.cache_dir)
    samples = load_samples(cache)
    train_samples, val_samples = split_train_val(samples, seed=seed, val_fraction=args.val_fraction)

    model = build_arm_model(arm)
    fine_w = args.fine_loss_weight if fine_on else 0.0
    coarse_w = args.coarse_loss_weight if coarse_on else 0.0

    job_dir.mkdir(parents=True, exist_ok=True)
    history = train_critic(
        model,
        train_samples,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        verbose=args.verbose,
        val_samples=val_samples,
        restore_best=False,
        best_path=str(best_path),
        fine_loss_weight=fine_w,
        coarse_loss_weight=coarse_w,
    )
    torch.save(model.state_dict(), last_path)

    metadata = {
        "arm": arm,
        "seed": seed,
        "cache_key": cache_key,
        "cache_path": str(cache),
        "param_count": param_count(model),
        "head_hidden_dim": getattr(model, "head_hidden_dim", None),
        "fine_loss_weight": fine_w,
        "coarse_loss_weight": coarse_w,
        "epochs": args.epochs,
        "lr": args.lr,
        "val_fraction": args.val_fraction,
        "train_sample_count": len(train_samples),
        "val_sample_count": len(val_samples),
        "path_corr_interaction": False,
        "setwise": False,
        "final_combiner": "calibrated" if ARMS[arm][0] == "dual" else None,
        "selected_for_eval": "best.pt",
        "selection_basis": "validation_loss_only_no_new_pool_II",
        "best_epoch": history.get("best_epoch"),
        "best_val_loss": history.get("best_val_loss"),
        "protocol": "docs/development/protocols/2026-07-28_semantic_decomposition_ablation_preregistration.md",
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (job_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"done arm={arm} seed={seed} params={metadata['param_count']} "
          f"best_epoch={metadata['best_epoch']} best_val={metadata['best_val_loss']}", flush=True)
    return metadata


def write_param_counts(output_dir):
    """实测每臂参数量，含剔除 corr_encoder 的有效参数量（本轮 corr_feat 全为 None）。"""
    import csv
    rows = []
    b0 = param_count(build_dual())
    for arm in ARMS:
        model = build_arm_model(arm)
        total = param_count(model)
        effective = sum(p.numel() for k, p in model.named_parameters()
                        if not k.startswith(("corr_encoder", "corr_missing")))
        rows.append({
            "arm": arm,
            "total_params": total,
            "effective_params_excl_corr": effective,
            "head_hidden_dim": getattr(model, "head_hidden_dim", None),
            "delta_vs_B0": total - b0,
            "delta_vs_B0_pct": round((total - b0) / b0 * 100, 4),
        })
    path = Path(output_dir) / "param_counts.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['arm']:26} total={row['total_params']:6d} "
              f"eff={row['effective_params_excl_corr']:6d} "
              f"head_h={row['head_hidden_dim']} dB0={row['delta_vs_B0']:+d}")
    print(f"param counts -> {path}")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["params", "train"], default="train")
    parser.add_argument("--source-cache", default=DEFAULT_SOURCE_CACHE)
    parser.add_argument("--output-dir", default=ABLATION_DIR)
    parser.add_argument("--cache-dir", default=ABLATION_DIR,
                        help="置换 cache（C1/C2/C3）所在目录，与 output-dir 解耦")
    parser.add_argument("--arm", action="append", choices=sorted(ARMS), default=None)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--fine-loss-weight", type=float, default=0.5)
    parser.add_argument("--coarse-loss-weight", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", dest="verbose", action="store_false")
    args = parser.parse_args()

    if args.stage == "params":
        write_param_counts(args.output_dir)
        return

    arms = args.arm or list(ARMS)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    write_param_counts(args.output_dir)
    for arm in arms:
        for seed in seeds:
            train_one(arm, seed, args)


if __name__ == "__main__":
    main()
