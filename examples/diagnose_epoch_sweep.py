"""Batch diagnostics for epoch checkpoint sweep."""

import argparse
import os
import re
import sys

import pandas as pd
import torch

sys.path.append(".")

from critic import DualHeadCritic, FundamentalCritic
from critic.diagnostics import evaluate_candidate_reranking, score_samples_for_diagnostics
from critic.train_critic import load_samples


_EPOCH_RE = re.compile(r"critic_e(\d+)\.pt$")


def infer_critic_arch(state_dict):
    """Identify the critic architecture represented by a checkpoint."""
    has_final = any(key.startswith("final_head.") for key in state_dict)
    has_fine = any(key.startswith("fine_head.") for key in state_dict)
    has_coarse = any(key.startswith("coarse_head.") for key in state_dict)
    dual_heads = (has_final, has_fine, has_coarse)
    if all(dual_heads):
        return "dual_head"
    if any(dual_heads):
        raise ValueError("dual_head checkpoint must include final_head, fine_head, and coarse_head")
    return "fundamental"


def build_critic_from_state_dict(state_dict):
    """Build a critic matching the checkpoint architecture and feature width."""
    architecture = infer_critic_arch(state_dict)
    kwargs = {}
    if architecture == "dual_head":
        fine_weight = state_dict.get("fine_head.0.weight")
        final_weight = state_dict.get("final_head.0.weight")
        if fine_weight is not None:
            feature_width = fine_weight.shape[1]
            if feature_width == 96:
                kwargs["path_corr_interaction"] = False
            elif feature_width == 128:
                kwargs["path_corr_interaction"] = True
        if fine_weight is not None and final_weight is not None:
            feature_width = fine_weight.shape[1]
            final_width = final_weight.shape[1]
            has_base = any(key.startswith("base_head.") for key in state_dict)
            has_residual_weights = {
                "residual_fine_weight",
                "residual_coarse_weight",
            }.issubset(state_dict)
            if has_residual_weights:
                kwargs["final_fusion"] = True
                kwargs["final_combiner"] = "residual"
            elif has_base and final_width == 3:
                kwargs["final_fusion"] = True
                kwargs["final_combiner"] = "calibrated"
            else:
                kwargs["final_fusion"] = final_width == feature_width + 2
                kwargs["final_combiner"] = "explicit"
        return DualHeadCritic(in_dim=6, **kwargs)

    head_weight = state_dict.get("head.0.weight")
    if head_weight is not None:
        head_in = head_weight.shape[1]
        if head_in == 96:
            kwargs["path_corr_interaction"] = False
        elif head_in == 128:
            kwargs["path_corr_interaction"] = True
    return FundamentalCritic(in_dim=6, **kwargs)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True, help="包含 critic_e{epoch}.pt 的目录。")
    p.add_argument("--samples-cache", required=True, help="CriticSample pickle 缓存路径。")
    p.add_argument("--out", required=True, help="epoch diagnostics summary CSV 输出路径。")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-corr", action="store_true")
    return p.parse_args(argv)


def find_epoch_checkpoints(checkpoint_dir):
    rows = []
    for name in os.listdir(checkpoint_dir):
        match = _EPOCH_RE.match(name)
        if match:
            rows.append((int(match.group(1)), os.path.join(checkpoint_dir, name)))
    if not rows:
        raise ValueError(f"No epoch checkpoints found in {checkpoint_dir}")
    return sorted(rows)


def main(argv=None):
    args = parse_args(argv)
    samples = load_samples(args.samples_cache)
    rows = []
    for epoch, path in find_epoch_checkpoints(args.checkpoint_dir):
        state_dict = torch.load(path, map_location=args.device)
        critic = build_critic_from_state_dict(state_dict).to(args.device)
        critic.load_state_dict(state_dict)
        critic.eval()
        score_rows, distance_rows = score_samples_for_diagnostics(
            critic,
            samples,
            use_corr=not args.no_corr,
        )
        metrics = evaluate_candidate_reranking(
            score_rows,
            distance_rows,
            top_k=args.top_k,
            temperature=args.temperature,
        )
        metrics["epoch"] = epoch
        metrics["checkpoint"] = path
        rows.append(metrics)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(rows).sort_values("epoch")
    columns = ["epoch", "checkpoint"] + [c for c in df.columns if c not in {"epoch", "checkpoint"}]
    df[columns].to_csv(args.out, index=False)
    print(df[columns].to_string(index=False))


if __name__ == "__main__":
    main()
