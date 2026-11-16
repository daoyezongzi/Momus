"""CriticKronosPredictor：包装 KronosPredictor，用 Critic 重排序替代简单平均。

设计要点（设计文档 §7）：
- 子类化 KronosPredictor，use_critic=False 时逐字节退回原 predict（消融开关）。
- use_critic=True 时：生成 N 条单独路径 -> Critic 打分 -> reranker 融合 -> 反归一化。
- 归一化/反归一化逻辑与 model/kronos.py:519-559 保持一致。
"""
import numpy as np
import pandas as pd
import torch

from model.kronos import KronosPredictor, auto_regressive_inference, calc_time_stamps


class CriticKronosPredictor(KronosPredictor):
    def __init__(self, model, tokenizer, critic, reranker, corr_provider=None,
                 device=None, max_context=512, clip=5):
        super().__init__(model, tokenizer, device=device, max_context=max_context, clip=clip)
        self.critic = critic
        self.reranker = reranker
        self.corr_provider = corr_provider
        self.use_critic = True  # 消融开关

    def generate_paths(self, df, x_timestamp, y_timestamp, pred_len, T=1.0,
                       top_k=0, top_p=0.9, sample_count=16, verbose=False,
                       target_code=None, return_log_probs=False):
        """生成 N 条归一化空间候选路径 + 归一化参数 + 关联特征。

        抽出此方法是为了让对比实验"生成一次候选、套用多种重排序策略"，
        保证加 Critic vs 不加在同一批候选上公平对比（消除采样随机性干扰）。

        Args:
            return_log_probs: True 时额外返回每条候选路径的生成对数似然
                （Kronos 自回归 token logprob 累加，temperature-scaled 分布上），
                供 prob_top1 baseline 使用。默认 False，保持 4 元组向后兼容。

        Returns:
            paths: (N, pred_len, feat) 归一化空间候选路径。
            hist_norm: (lookback, feat) 目标股归一化历史。
            corr_feat: (n_corr, lookback, feat) 关联特征或 None。
            norm: (x_mean, x_std) 反归一化参数。
            log_probs: (N,) 每条候选路径对数似然。仅 return_log_probs=True 时追加。
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        if not all(col in df.columns for col in self.price_cols):
            raise ValueError(f"Price columns {self.price_cols} not found in DataFrame.")

        df = df.copy()
        if self.vol_col not in df.columns:
            df[self.vol_col] = 0.0
            df[self.amt_vol] = 0.0
        if self.amt_vol not in df.columns and self.vol_col in df.columns:
            df[self.amt_vol] = df[self.vol_col] * df[self.price_cols].mean(axis=1)
        if df[self.price_cols + [self.vol_col, self.amt_vol]].isnull().values.any():
            raise ValueError("Input DataFrame contains NaN values in price or volume columns.")

        # ---- 归一化（与 kronos.py:537-551 一致）----
        x_stamp = calc_time_stamps(x_timestamp).values.astype(np.float32)
        y_stamp = calc_time_stamps(y_timestamp).values.astype(np.float32)
        x = df[self.price_cols + [self.vol_col, self.amt_vol]].values.astype(np.float32)
        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        x_norm = np.clip((x - x_mean) / (x_std + 1e-5), -self.clip, self.clip)

        # ---- 生成 N 条单独路径（决策-A：return_individual=True）----
        x_tensor = torch.from_numpy(x_norm[None].astype(np.float32)).to(self.device)
        x_stamp_tensor = torch.from_numpy(x_stamp[None].astype(np.float32)).to(self.device)
        y_stamp_tensor = torch.from_numpy(y_stamp[None].astype(np.float32)).to(self.device)

        inference_out = auto_regressive_inference(
            self.tokenizer, self.model, x_tensor, x_stamp_tensor, y_stamp_tensor,
            self.max_context, pred_len, self.clip, T, top_k, top_p, sample_count,
            verbose, return_individual=True, return_log_probs=return_log_probs,
        )
        if return_log_probs:
            preds, log_probs = inference_out
            log_probs = np.asarray(log_probs)[0]  # (B=1, N) -> (N,)
        else:
            preds = inference_out
        paths = preds[0][:, -pred_len:, :]  # (N, pred_len, feat)

        corr_feat = None
        if self.corr_provider is not None and target_code is not None:
            corr_feat = self.corr_provider.get(target_code, x_timestamp)

        if return_log_probs:
            return paths, x_norm, corr_feat, (x_mean, x_std), log_probs
        return paths, x_norm, corr_feat, (x_mean, x_std)

    def predict(self, df, x_timestamp, y_timestamp, pred_len, T=1.0, top_k=0,
                top_p=0.9, sample_count=16, verbose=True, target_code=None):
        # 消融：关闭 Critic 时完全退回原行为
        if not self.use_critic:
            return super().predict(df, x_timestamp, y_timestamp, pred_len, T=T,
                                   top_k=top_k, top_p=top_p, sample_count=sample_count,
                                   verbose=verbose)

        paths, hist_norm, corr_feat, (x_mean, x_std) = self.generate_paths(
            df, x_timestamp, y_timestamp, pred_len, T, top_k, top_p,
            sample_count, verbose, target_code,
        )

        # ---- Critic 打分 ----
        scores = self.critic.score(paths, hist=hist_norm, corr_feat=corr_feat)

        # ---- 重排序融合（归一化空间）----
        final_norm = self.reranker.combine(paths, scores)  # (pred_len, feat)

        # ---- 反归一化（与 kronos.py:556 一致）----
        final = final_norm * (x_std + 1e-5) + x_mean
        pred_df = pd.DataFrame(final, columns=self.price_cols + [self.vol_col, self.amt_vol],
                               index=y_timestamp)
        return pred_df
