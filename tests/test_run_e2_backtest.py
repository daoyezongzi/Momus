import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import examples.run_e2_backtest as runner
from examples.run_e2_backtest import (
    _critic_kwargs_from_state_dict,
    aggregate_pnls,
    evaluate_window,
    parse_temperatures,
    render_report,
    render_temperature_report,
)


def test_script_can_run_help_when_executed_by_path():
    result = subprocess.run(
        [sys.executable, "examples/run_e2_backtest.py", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--critic-ckpt" in result.stdout


def test_script_adds_repo_root_to_python_path_when_executed_by_path():
    repo_root = Path.cwd().resolve().as_posix()
    script = (
        "import runpy, sys\n"
        "sys.argv=['examples/run_e2_backtest.py','--help']\n"
        "try:\n"
        "    runpy.run_path('examples/run_e2_backtest.py', run_name='__main__')\n"
        "except SystemExit:\n"
        "    pass\n"
        f"assert {repo_root!r} in [p.replace('\\\\', '/') for p in sys.path]\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_module_import_does_not_load_model_package():
    script = (
        "import sys; import examples.run_e2_backtest; "
        "assert 'model' not in sys.modules and 'model.kronos' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_main_entrypoint_is_after_runtime_helpers():
    source = open("examples/run_e2_backtest.py", encoding="utf-8").read()

    assert source.rfind('if __name__ == "__main__":') > source.find("def evaluate_window")
    assert source.rfind('if __name__ == "__main__":') > source.find("def aggregate_pnls")


def test_critic_kwargs_detects_legacy_non_interaction_checkpoint():
    state_dict = {"head.0.weight": np.zeros((64, 96), dtype=np.float32)}

    assert _critic_kwargs_from_state_dict(state_dict) == {"path_corr_interaction": False}


def test_critic_kwargs_detects_interaction_checkpoint():
    state_dict = {"head.0.weight": np.zeros((64, 128), dtype=np.float32)}

    assert _critic_kwargs_from_state_dict(state_dict) == {"path_corr_interaction": True}


def test_run_requires_checkpoint_before_loading_runtime_dependencies(monkeypatch):
    def fail_if_called():
        raise AssertionError("runtime dependencies should not load without a checkpoint")

    args = SimpleNamespace(critic_ckpt=None, seed=0)
    monkeypatch.setattr(runner, "_load_runtime_dependencies", fail_if_called)

    with pytest.raises(ValueError, match="E2 requires --critic-ckpt"):
        runner.run(args)


def test_parse_temperatures_accepts_comma_separated_values():
    assert parse_temperatures("0.25,0.5,1.0,2,5") == [0.25, 0.5, 1.0, 2.0, 5.0]


def test_render_temperature_report_focuses_on_weighted_sensitivity():
    per_window_by_temperature = {
        0.5: [
            {"net_long_only": {"simple_mean": -0.01, "critic_weighted": 0.02, "critic_top1": 0.03}},
            {"net_long_only": {"simple_mean": 0.00, "critic_weighted": 0.01, "critic_top1": 0.04}},
        ],
        1.0: [
            {"net_long_only": {"simple_mean": -0.01, "critic_weighted": 0.03, "critic_top1": 0.03}},
            {"net_long_only": {"simple_mean": 0.00, "critic_weighted": 0.02, "critic_top1": 0.04}},
        ],
    }

    text = render_temperature_report(per_window_by_temperature, pred_len=10, sample_count=16, seed=0)

    assert "E2 温度 τ 敏感性" in text
    assert "| 0.5 | net_long_only | 2 | 1.5000%" in text
    assert "| 1 | net_long_only | 2 | 2.5000%" in text
    assert "simple_mean 固定" in text


def test_render_report_records_reproducibility_and_caveats():
    per_window = [{"gross_long_short": {"simple_mean": 0.01, "critic_weighted": 0.02, "critic_top1": 0.0}}]
    results = aggregate_pnls(per_window, pred_len=10)
    text = render_report(results, window_count=1, pred_len=10, sample_count=16, seed=0)
    assert "seed=0" in text
    assert "sample_count=16" in text
    assert "窗口可能重叠" in text
    assert "理论参考" in text


def test_run_uses_temperature_output_path(monkeypatch):
    calls = {"loaded": False}

    class DummyTorch:
        @staticmethod
        def load(path, map_location=None):
            return {"head.0.weight": np.zeros((64, 96), dtype=np.float32)}

    class DummyCritic:
        def __init__(self, in_dim=6, **kwargs):
            self.kwargs = kwargs

        def load_state_dict(self, state_dict):
            self.state_dict = state_dict

        def eval(self):
            return None

        def score(self, paths, hist=None, corr_feat=None):
            return np.arange(paths.shape[0], dtype=float)

    class DummyReranker:
        def __init__(self, mode="weighted", temperature=1.0):
            self.mode = mode
            self.temperature = temperature

        def combine(self, paths, scores):
            return paths[-1] if self.mode == "top1" else np.mean(paths, axis=0)

    class DummyPredictor:
        def __init__(self, *args, **kwargs):
            pass

        def generate_paths(self, x_df, x_ts, y_ts, pred_len, sample_count=16, verbose=False,
                            target_code=None, return_log_probs=False):
            paths = np.zeros((2, pred_len, 6), dtype=float)
            paths[:, -1, 3] = 1.0
            norm = (np.array([0, 0, 0, 100, 0, 0], dtype=float), np.ones(6))
            if return_log_probs:
                return paths, np.zeros((1, 6)), None, norm, np.array([-1.0, -2.0])
            return paths, np.zeros((1, 6)), None, norm

    def strategy_simple_mean(paths, hist=None, corr_feat=None):
        return np.mean(paths, axis=0)

    def strategy_median(paths, hist=None, corr_feat=None):
        return np.median(paths, axis=0)

    def make_critic_strategy(critic, reranker):
        def _strategy(paths, hist=None, corr_feat=None):
            return reranker.combine(paths, critic.score(paths, hist=hist, corr_feat=corr_feat))
        return _strategy

    def make_prob_top1_strategy(log_probs):
        log_probs = np.asarray(log_probs)
        def _strategy(paths, hist=None, corr_feat=None):
            return paths[int(np.argmax(log_probs))]
        return _strategy

    def build_synth_windows(n_samples, lookback, pred_len):
        x_df = pd.DataFrame({"close": [100.0]})
        y_true = np.zeros((pred_len, 6), dtype=float)
        y_true[0, 0] = 100.0
        y_true[-1, 3] = 101.0
        return [(None, x_df, None, None, y_true)]

    def fake_load_runtime_dependencies():
        calls["loaded"] = True
        return {
            "CorrelatedDataProvider": object,
            "CriticKronosPredictor": DummyPredictor,
            "CriticReranker": DummyReranker,
            "FundamentalCritic": DummyCritic,
            "Kronos": SimpleNamespace(from_pretrained=lambda name: object()),
            "KronosTokenizer": SimpleNamespace(from_pretrained=lambda name: object()),
            "_load_dotenv": lambda: None,
            "build_real_windows": lambda args, fetch_fn: [],
            "build_synth_windows": build_synth_windows,
            "make_critic_strategy": make_critic_strategy,
            "make_prob_top1_strategy": make_prob_top1_strategy,
            "make_tushare_fetch": lambda token: None,
            "strategy_median": strategy_median,
            "strategy_simple_mean": strategy_simple_mean,
            "torch": DummyTorch,
        }

    output_dir = Path(".pytest_local_tmp") / "run_e2_temperature"
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)

    monkeypatch.setattr(runner, "_load_runtime_dependencies", fake_load_runtime_dependencies)
    args = SimpleNamespace(
        critic_ckpt="dummy.pt",
        seed=0,
        real=False,
        tushare_token=None,
        lookback=90,
        pred_len=10,
        sample_count=2,
        temperature=1.0,
        temperatures="0.5,1.0",
        n_samples=1,
        output=str(output_dir / "report.txt"),
        temperature_output=str(output_dir / "tau.txt"),
        figures_dir=str(output_dir / "figures"),
        device="cpu",
    )

    try:
        runner.run(args)

        assert calls["loaded"] is True
        assert (output_dir / "report.txt").exists()
        assert "E2 温度 τ 敏感性" in (output_dir / "tau.txt").read_text(encoding="utf-8")
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


    x_df = pd.DataFrame({"close": [100.0]})
    y_true = np.zeros((2, 6), dtype=float)
    y_true[0, 0] = 105.0
    y_true[-1, 3] = 110.0
    x_mean = np.array([0, 0, 0, 100, 0, 0], dtype=float)
    x_std = np.ones(6, dtype=float)
    up = np.zeros((2, 6))
    up[-1, 3] = 1.0
    down = np.zeros((2, 6))
    down[-1, 3] = -1.0
    flat = np.zeros((2, 6))
    finals = {"simple_mean": up, "critic_weighted": down, "critic_top1": flat}

    result = evaluate_window(x_df, y_true, finals, x_mean, x_std, net_cost=0.00075)

    assert np.isclose(result["gross_long_short"]["simple_mean"], 0.10)
    assert np.isclose(result["gross_long_short"]["critic_weighted"], -0.10)
    assert result["gross_long_only"]["critic_weighted"] == 0.0
    assert result["net_long_only"]["critic_top1"] == 0.0
    assert np.isclose(result["net_long_only"]["simple_mean"], (110 - 105) / 105 - 0.00075)


def test_aggregate_pnls_computes_ir_against_same_regime_baseline():
    per_window = [
        {"gross_long_short": {"simple_mean": 0.01, "critic_weighted": 0.02}},
        {"gross_long_short": {"simple_mean": -0.01, "critic_weighted": 0.00}},
    ]

    results = aggregate_pnls(per_window, pred_len=10)

    assert results["gross_long_short"]["simple_mean"]["metrics"]["information_ratio"] == 0.0
    assert results["gross_long_short"]["critic_weighted"]["metrics"]["count"] == 2
    assert results["gross_long_short"]["critic_weighted"]["invalid_count"] == 0


def test_aggregate_pnls_counts_invalid_values_per_strategy_with_aligned_baseline():
    per_window = [
        {"gross_long_short": {"simple_mean": 0.01, "critic_weighted": 0.02}},
        {"gross_long_short": {"simple_mean": 0.03, "critic_weighted": np.nan}},
    ]

    results = aggregate_pnls(per_window, pred_len=10)

    assert results["gross_long_short"]["simple_mean"]["metrics"]["count"] == 2
    assert results["gross_long_short"]["critic_weighted"]["metrics"]["count"] == 1
    assert results["gross_long_short"]["critic_weighted"]["invalid_count"] == 1
