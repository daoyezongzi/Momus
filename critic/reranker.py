"""CriticReranker：把 N 条候选路径 + N 个分数融合成最终预测。

两种模式（设计文档 §6）：
- weighted：w = softmax(scores / τ)，final = Σ w_i · path_i。τ→∞ 退化为简单平均。
- top1：直接取 argmax(scores) 那条路径。

融合在归一化空间进行（与原 np.mean 一致），之后由 predictor 反归一化。
"""
import numpy as np


class CriticReranker:
    def __init__(self, mode="weighted", temperature=1.0):
        if mode not in ("weighted", "top1"):
            raise ValueError(f"未知的重排序模式: {mode}")
        self.mode = mode
        self.temperature = temperature

    def combine(self, paths: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """融合候选路径。

        Args:
            paths: 形状 (N, seq, feat) 的候选路径（单个样本）。
            scores: 形状 (N,) 的 Critic 分数，越高越好。

        Returns:
            形状 (seq, feat) 的最终路径。
        """
        if paths.ndim != 3:
            raise ValueError(f"paths 期望 (N, seq, feat)，得到 {paths.shape}")
        if scores.shape[0] != paths.shape[0]:
            raise ValueError(f"scores 数量 {scores.shape[0]} 与路径数 {paths.shape[0]} 不匹配")

        if self.mode == "top1":
            return paths[int(np.argmax(scores))]

        # weighted：数值稳定的 softmax
        logits = scores / self.temperature
        logits = logits - np.max(logits)
        w = np.exp(logits)
        w = w / np.sum(w)
        # (N,1,1) 广播到 (N, seq, feat) 后按 N 求和
        return np.sum(w[:, None, None] * paths, axis=0)
