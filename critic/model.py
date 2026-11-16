"""Critic 打分模型。

本文件包含两个类：
- RandomCritic：假 Critic（随机打分），用于步骤 2 端到端管道验证。
- FundamentalCritic：真 Critic 网络骨架（设计文档 §5.2），步骤 1 填充训练逻辑。

打分接口统一为 score(paths, hist, corr_feat) -> scores，对任意 path 长度都可打分
（设计文档 §10：为方案 B 复用预留）。
"""
import numpy as np
import torch
import torch.nn as nn
from typing import NamedTuple

from .config import CRITIC_HIDDEN_DIM, CRITIC_EMBED_DIM


class CriticScoreOutput(NamedTuple):
    final: torch.Tensor
    fine: torch.Tensor
    coarse: torch.Tensor


class RandomCritic:
    """假 Critic：对每条路径给随机分数。仅用于验证管道连通性。"""

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def score(self, paths: np.ndarray, hist=None, corr_feat=None) -> np.ndarray:
        """返回形状 (N,) 的随机分数。

        Args:
            paths: (N, seq, feat) 候选路径。
            hist: 目标股历史（本假实现忽略）。
            corr_feat: 关联特征（本假实现忽略）。
        """
        n = paths.shape[0]
        return self.rng.standard_normal(n).astype(np.float32)


class _SeqEncoder(nn.Module):
    """序列编码器：GRU + 末隐状态。用于路径和历史（孪生共享）。"""

    def __init__(self, in_dim, hidden_dim, embed_dim):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x):
        # x: (B, seq, in_dim)
        _, h = self.gru(x)          # h: (1, B, hidden)
        return self.proj(h.squeeze(0))  # (B, embed)


class _CorrEncoder(nn.Module):
    """关联标的编码器（设计文档 §5.2）：

    对每个关联标的用 GRU 编码成向量，再用注意力池化跨标的聚合成单一 h_corr。
    注意力池化让 Critic 自动学习"哪些关联标的此刻更重要"（如上游刚异动的标的）。
    """

    def __init__(self, in_dim, hidden_dim, embed_dim):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, embed_dim)
        # 注意力打分：每个标的向量 -> 标量权重
        self.attn = nn.Linear(embed_dim, 1)

    def forward(self, corr):
        # corr: (B, n_corr, seq, in_dim)
        b, n, seq, d = corr.shape
        flat = corr.reshape(b * n, seq, d)
        _, h = self.gru(flat)                    # (1, B*n, hidden)
        emb = self.proj(h.squeeze(0))            # (B*n, embed)
        emb = emb.reshape(b, n, -1)              # (B, n, embed)
        # 跨标的注意力池化
        w = torch.softmax(self.attn(emb), dim=1)  # (B, n, 1)
        return torch.sum(w * emb, dim=1)          # (B, embed)


class _SetMixer(nn.Module):
    """Deep-Sets 跨候选混合：让每条候选看到全组(mean/max)上下文。

    置换等变：单条候选输出只依赖 自身特征 + 全组聚合(mean/max),重排候选→输出同序重排。
    末层零初始化 => 训练起点等价 pointwise,再学习是否利用 set-context(干净消融)。
    残差形式保证维度不变,下游打分头无需改动。
    """

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(dim * 3, dim), nn.ReLU(), nn.Linear(dim, dim))
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, feat):
        # feat: (N, dim)，N=候选数(=batch)
        mean = feat.mean(0, keepdim=True).expand_as(feat)
        mx = feat.max(0, keepdim=True).values.expand_as(feat)
        ctx = torch.cat([feat, mean, mx], dim=-1)
        return feat + self.proj(ctx)


class FundamentalCritic(nn.Module):
    """真 Critic 网络骨架（设计文档 §5.2）。

    结构：路径 encoder 与历史 encoder 孪生共享权重，学习"路径相对历史的
    延续是否合理"，而非记忆绝对价位。关联特征 encoder 步骤 1 接入。

    步骤 2 只需保证 forward 可跑通形状；训练逻辑在 train_critic.py（步骤 1）。
    """

    def __init__(self, in_dim=6, hidden_dim=CRITIC_HIDDEN_DIM, embed_dim=CRITIC_EMBED_DIM,
                 path_corr_interaction=True, resonance_dim=0, setwise=False,
                 head_hidden_dim=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.resonance_dim = resonance_dim
        self.setwise = setwise
        # head_hidden_dim 与 encoder 的 hidden_dim 解耦，用于参数量匹配对照：
        # 加宽打分头而不改动 encoder，使 single-head 模型能与 DualHead 参数量对齐。
        self.head_hidden_dim = hidden_dim if head_hidden_dim is None else head_hidden_dim
        # 是否加 path-corr 候选级交互（修复 corr 对候选为常量、被 softmax 抵消的缺陷）
        self.path_corr_interaction = path_corr_interaction
        # 路径与历史共享同一 encoder（孪生）
        self.seq_encoder = _SeqEncoder(in_dim, hidden_dim, embed_dim)
        # 关联标的编码器（GRU + 跨标的注意力池化）
        self.corr_encoder = _CorrEncoder(in_dim, hidden_dim, embed_dim)
        # 无关联特征时用可学习的占位向量，保持打分头输入维度稳定
        self.corr_missing = nn.Parameter(torch.zeros(embed_dim))
        if resonance_dim > 0:
            self.resonance_encoder = nn.Sequential(
                nn.Linear(resonance_dim, embed_dim),
                nn.Tanh(),
            )
            self.resonance_missing = nn.Parameter(torch.zeros(embed_dim))
        else:
            self.resonance_encoder = None
            self.register_parameter("resonance_missing", None)
        # 打分头输入：[h_path, h_hist, h_corr] (+ 交互项 h_path⊙h_corr)
        # resonance_dim>0 时额外加入 [h_res, h_path⊙h_res]，让共振随候选变化参与排序。
        head_parts = 4 if path_corr_interaction else 3
        if resonance_dim > 0:
            head_parts += 2
        head_in = embed_dim * head_parts
        # setwise: 在拼好的特征上做跨候选混合(残差,零初始化=起点等价 pointwise)
        self.set_mixer = None
        if setwise:
            rng_state = torch.random.get_rng_state()
            self.set_mixer = _SetMixer(head_in)
            torch.random.set_rng_state(rng_state)
        self.head = nn.Sequential(
            nn.Linear(head_in, self.head_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.head_hidden_dim, 1),
        )

    def encode_features(self, path, hist, corr=None, resonance=None):
        """Encode candidate, historical, correlation, and resonance features."""
        h_path = self.seq_encoder(path)
        h_hist = self.seq_encoder(hist)
        if corr is None:
            h_corr = self.corr_missing.unsqueeze(0).expand(h_path.size(0), -1)
        else:
            h_corr = self.corr_encoder(corr)
        parts = [h_path, h_hist, h_corr]
        if self.path_corr_interaction:
            # 逐候选乘性交互：每条候选路径与关联特征的匹配度（随候选变化）
            parts.append(h_path * h_corr)
        if self.resonance_dim > 0:
            if resonance is None:
                h_res = self.resonance_missing.unsqueeze(0).expand(h_path.size(0), -1)
            else:
                h_res = self.resonance_encoder(resonance)
            parts.extend([h_res, h_path * h_res])
        feat = torch.cat(parts, dim=-1)
        if self.set_mixer is not None:
            feat = self.set_mixer(feat)
        return feat

    def forward(self, path, hist, corr=None, resonance=None):
        """
        Args:
            path: (B, pred_len, in_dim) 候选路径（收益率序列）。
            hist: (B, lookback, in_dim) 目标股历史（收益率序列）。
            corr: (B, n_corr, seq, in_dim) 关联标的特征；None 时用占位向量。
            resonance: (B, resonance_dim) 共振特征；None 时用占位向量。
        Returns:
            (B,) 分数。
        """
        feat = self.encode_features(path, hist, corr, resonance)
        return self.head(feat).squeeze(-1)

    def _prepare_score_inputs(self, paths, hist, corr_feat=None, resonance_feat=None):
        """Convert score inputs to tensors and broadcast shared features."""
        device = next(self.parameters()).device
        paths_arr = np.asarray(paths, dtype=np.float32)
        hist_arr = np.asarray(hist, dtype=np.float32)
        n = paths_arr.shape[0]
        path_t = torch.from_numpy(np.ascontiguousarray(paths_arr)).to(device)
        hist_t = torch.from_numpy(np.ascontiguousarray(hist_arr)).to(device)
        hist_t = hist_t.unsqueeze(0).expand(n, -1, -1)

        corr_t = None
        if corr_feat is not None:
            corr_arr = np.asarray(corr_feat, dtype=np.float32)
            corr_t = torch.from_numpy(np.ascontiguousarray(corr_arr)).to(device)
            corr_t = corr_t.unsqueeze(0).expand(n, -1, -1, -1)

        resonance_t = None
        if resonance_feat is not None:
            resonance_arr = np.asarray(resonance_feat, dtype=np.float32)
            if resonance_arr.ndim == 1:
                resonance_arr = np.broadcast_to(resonance_arr[None, :], (n, resonance_arr.shape[0]))
            resonance_t = torch.from_numpy(np.ascontiguousarray(resonance_arr)).to(device)
        return path_t, hist_t, corr_t, resonance_t

    @torch.no_grad()
    def score(self, paths: np.ndarray, hist: np.ndarray, corr_feat=None,
              resonance_feat=None, corr_mask=None) -> np.ndarray:
        """推理期打分接口，与 RandomCritic 对齐。

        Args:
            paths: (N, seq, feat) 候选路径。
            hist: (lookback, feat) 目标股历史，会广播到 N 条。
            corr_feat: (n_corr, lookback, feat) 关联特征；None 时退化为仅用 path+hist。
            resonance_feat: (resonance_dim,) 或 (N, resonance_dim) 共振特征。
        Returns:
            (N,) 分数。
        """
        self.eval()
        path_t, hist_t, corr_t, resonance_t = self._prepare_score_inputs(
            paths, hist, corr_feat, resonance_feat
        )
        return self.forward(path_t, hist_t, corr_t, resonance_t).cpu().numpy()


class DualHeadCritic(FundamentalCritic):
    """Critic with fine/coarse auxiliary heads and a final score combiner.

    Combiner 设计将 final score 的输入显式约束为三个标量（calibrated 模式）
    或 residual 加权的形式，使 fine/coarse auxiliary heads 成为 final score
    的显式决策变量，而非仅在训练时提供梯度信号、推理时闲置的中间层。
    """

    def __init__(self, in_dim=6, hidden_dim=CRITIC_HIDDEN_DIM, embed_dim=CRITIC_EMBED_DIM,
                 path_corr_interaction=True, resonance_dim=0, final_fusion=True,
                 final_combiner="explicit", setwise=False, head_hidden_dim=None):
        super().__init__(in_dim, hidden_dim, embed_dim, path_corr_interaction, resonance_dim,
                         setwise=setwise, head_hidden_dim=head_hidden_dim)
        if final_combiner not in {"explicit", "residual", "calibrated"}:
            raise ValueError("final_combiner must be 'explicit', 'residual', or 'calibrated'")
        head_in = self.head[0].in_features
        hidden_dim = self.head_hidden_dim
        self.final_fusion = final_fusion
        self.final_combiner = final_combiner
        self.fine_head = nn.Sequential(nn.Linear(head_in, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.coarse_head = nn.Sequential(nn.Linear(head_in, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        if not final_fusion or final_combiner == "explicit":
            # explicit: final_head 看到全量 shared feature + fine/coarse 标量。
            # 问题是 fine/coarse 在高维特征中占比极小，auxiliary heads 容易退化为闲置训练信号。
            final_in = head_in + 2 if final_fusion else head_in
            self.final_head = nn.Sequential(nn.Linear(final_in, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        elif final_combiner == "residual":
            # residual: final = base(feat) + α·fine + β·coarse。
            # α,β 可学习（零初始化），在 shared feature 基础上做残差修正。
            # 仍依赖全量 feat，但 fine/coarse 通过标量权重直接贡献到 final。
            self.base_head = nn.Sequential(nn.Linear(head_in, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
            self.final_head = self.base_head
            self.residual_fine_weight = nn.Parameter(torch.zeros(()))
            self.residual_coarse_weight = nn.Parameter(torch.zeros(()))
        else:
            # calibrated: final_head 的输入严格限制为 [base, fine, coarse] 三个标量。
            # fine/coarse 不再是"特征海洋中的两个维度"，而是与 base 平权的决策变量。
            # 这迫使 auxiliary heads 承载明确语义：fine 捕捉细节结构，coarse 捕捉整体趋势，
            # final_score 是三者的小型 MLP 融合。
            self.base_head = nn.Sequential(nn.Linear(head_in, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
            self.final_head = nn.Sequential(nn.Linear(3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        del self.head

    def forward(self, path, hist, corr=None, resonance=None):
        feat = self.encode_features(path, hist, corr, resonance)
        fine = self.fine_head(feat)     # auxiliary head: 细颗粒度评分
        coarse = self.coarse_head(feat) # auxiliary head: 粗颗粒度评分
        if not self.final_fusion:
            final = self.final_head(feat)
        elif self.final_combiner == "explicit":
            # fine/coarse 被 concat 进 ~200 维特征中，占比 ~1%，auxiliary 信号易被淹没
            final_inputs = torch.cat([feat, fine, coarse], dim=-1)
            final = self.final_head(final_inputs)
        elif self.final_combiner == "residual":
            # fine/coarse 通过可学习标量权重直接修正 base 分，参与度透明可控
            base = self.base_head(feat)
            final = base + self.residual_fine_weight * fine + self.residual_coarse_weight * coarse
        else:
            # calibrated: final 的输入空间就是 {base, fine, coarse} 三个标量。
            # 三者平权进入一个小型 MLP，fine/coarse 必须"自己说话"才能影响最终排序。
            base = self.base_head(feat)
            final_inputs = torch.cat([base, fine, coarse], dim=-1)
            final = self.final_head(final_inputs)
        return CriticScoreOutput(
            final=final.squeeze(-1),
            fine=fine.squeeze(-1),
            coarse=coarse.squeeze(-1),
        )

    @torch.no_grad()
    def score(self, paths: np.ndarray, hist: np.ndarray, corr_feat=None,
              resonance_feat=None, corr_mask=None, head="final") -> np.ndarray:
        if head not in ("final", "fine", "coarse"):
            raise ValueError("head must be 'final', 'fine', or 'coarse'")
        self.eval()
        path_t, hist_t, corr_t, resonance_t = self._prepare_score_inputs(
            paths, hist, corr_feat, resonance_feat
        )
        out = self.forward(path_t, hist_t, corr_t, resonance_t)
        return getattr(out, head).cpu().numpy()
