"""Run oracle-gap diagnostics for a trained Critic on cached samples."""

import argparse
import os
import sys

import pandas as pd
import torch

sys.path.append(".")

from critic import DualHeadCritic, FundamentalCritic
from critic.diagnostics import evaluate_candidate_reranking, score_samples_for_diagnostics
from critic.train_critic import load_samples


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
    if any(key.startswith("set_mixer.") for key in state_dict):
        kwargs["setwise"] = True
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
            # 从 state_dict 反推训练时用的 combiner 模式:
            #   1) 有 residual_fine_weight 参数 => residual
            #   2) 有 base_head.* 且 final_head 输入维度=3 => calibrated
            #   3) 否则根据 final_head 输入维度推断 explicit 或旧版 parallel
            has_base = any(key.startswith("base_head.") for key in state_dict)
            has_residual_weights = {
                "residual_fine_weight",
                "residual_coarse_weight",
            }.issubset(state_dict)
            if has_residual_weights:
                kwargs["final_fusion"] = True
                kwargs["final_combiner"] = "residual"
            elif has_base and final_width == 3:
                # calibrated: final_head 的输入被严格限制为 [base, fine, coarse] 三个标量
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
    p.add_argument("--critic", required=True, help="训练好的 critic checkpoint 路径。")
    p.add_argument("--samples-cache", required=True, help="CriticSample pickle 缓存路径。")
    p.add_argument("--out", required=True, help="诊断 summary CSV 输出路径。")
    p.add_argument("--top-k", type=int, default=3, help="oracle candidate 是否落入 score top-k。")
    p.add_argument("--temperature", type=float, default=1.0, help="softmax 权重熵温度。")
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-corr", action="store_true", help="诊断时忽略关联特征。")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    samples = load_samples(args.samples_cache)
    state_dict = torch.load(args.critic, map_location=args.device)
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
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(args.out, index=False)
    print(pd.Series(metrics).to_string())


if __name__ == "__main__":
    main()
