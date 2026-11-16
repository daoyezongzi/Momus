import pytest

from examples.run_resonance_experiment import STRATEGIES, build_parser, main


def test_all_preregistered_strategies_exist():
    assert STRATEGIES == (
        "simple_mean",
        "critic_no_corr",
        "critic_price_only",
        "critic_volume_only",
        "critic_price_volume",
        "critic_random_mapping",
        "critic_past_shifted",
    )


def test_dev_parser_does_not_allow_2025_by_default():
    args = build_parser().parse_args(["--phase", "dev", "--synthetic"])
    assert args.allow_2025_labels is False


def test_dev_phase_rejects_allow_2025_labels():
    with pytest.raises(RuntimeError, match="development phase"):
        main(["--phase", "dev", "--synthetic", "--allow-2025-labels"])


def test_small_trial_synthetic_runs_without_2025_or_network(tmp_path):
    result = main(["--phase", "small-trial", "--synthetic", "--output-dir", str(tmp_path)])
    assert result["phase"] == "small-trial"
    assert result["synthetic"] is True
    assert result["strategies"] == list(STRATEGIES)
    assert result["allow_2025_labels"] is False
