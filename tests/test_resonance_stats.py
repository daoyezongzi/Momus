import numpy as np

from critic.resonance_analysis import bootstrap_bin_difference


def test_bootstrap_bin_difference_reports_high_minus_low_effect():
    low = np.array([3.0, 4.0, 5.0], dtype=float)
    high = np.array([1.0, 2.0, 3.0], dtype=float)
    result = bootstrap_bin_difference(low, high, n_boot=200, seed=0)
    assert result["effect_high_minus_low"] < 0
    assert result["p_value"] <= 0.1
    assert result["ci_low"] <= result["effect_high_minus_low"] <= result["ci_high"]
