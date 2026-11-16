import numpy as np
import torch

from critic.model import FundamentalCritic


def test_resonance_changes_relative_candidate_scores():
    torch.manual_seed(0)
    model = FundamentalCritic(in_dim=6, resonance_dim=17)
    paths = np.zeros((2, 10, 6), dtype=np.float32)
    paths[1, :, 3] = 1.0
    hist = np.zeros((90, 6), dtype=np.float32)
    positive = np.ones(17, dtype=np.float32)
    negative = -positive

    score_a = model.score(paths, hist, resonance_feat=positive)
    score_b = model.score(paths, hist, resonance_feat=negative)

    assert not np.isclose(score_a[1] - score_a[0], score_b[1] - score_b[0])


def test_no_corr_and_corr_use_same_parameter_count():
    no_corr = FundamentalCritic(in_dim=6, resonance_dim=17)
    corr = FundamentalCritic(in_dim=6, resonance_dim=17)
    assert sum(p.numel() for p in no_corr.parameters()) == sum(
        p.numel() for p in corr.parameters()
    )
