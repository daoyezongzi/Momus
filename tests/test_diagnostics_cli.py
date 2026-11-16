"""CLI tests for critic diagnostics."""

from pathlib import Path
from uuid import uuid4

import pytest

from examples.diagnose_critic import parse_args
from examples.diagnose_epoch_sweep import find_epoch_checkpoints, parse_args as parse_sweep_args


def test_diagnose_critic_parser_accepts_required_paths_and_options():
    args = parse_args([
        "--critic", "critic.pt",
        "--samples-cache", "samples.pkl",
        "--out", "summary.csv",
        "--top-k", "4",
        "--temperature", "0.7",
        "--no-corr",
    ])

    assert args.critic == "critic.pt"
    assert args.samples_cache == "samples.pkl"
    assert args.out == "summary.csv"
    assert args.top_k == 4
    assert args.temperature == 0.7
    assert args.no_corr is True


def test_diagnose_epoch_sweep_parser_accepts_checkpoint_dir():
    args = parse_sweep_args([
        "--checkpoint-dir", "output/results/epoch_sweep",
        "--samples-cache", "samples.pkl",
        "--out", "epoch_diagnostics_summary.csv",
        "--top-k", "3",
        "--device", "cuda",
    ])

    assert args.checkpoint_dir == "output/results/epoch_sweep"
    assert args.samples_cache == "samples.pkl"
    assert args.out == "epoch_diagnostics_summary.csv"
    assert args.top_k == 3
    assert args.device == "cuda"


def test_checkpoint_arch_detection_identifies_dual_head():
    from examples.diagnose_critic import infer_critic_arch

    dual_state = {"final_head.0.weight": object(), "fine_head.0.weight": object(), "coarse_head.0.weight": object()}
    fundamental_state = {"head.0.weight": object()}

    assert infer_critic_arch(dual_state) == "dual_head"
    assert infer_critic_arch(fundamental_state) == "fundamental"


def test_checkpoint_arch_detection_rejects_partial_dual_head():
    from examples.diagnose_critic import infer_critic_arch

    with pytest.raises(ValueError, match="final_head, fine_head, and coarse_head"):
        infer_critic_arch({"fine_head.0.weight": object()})


def test_build_critic_from_state_dict_uses_fusion_for_new_dual_head_checkpoint():
    from critic import DualHeadCritic
    from examples.diagnose_critic import build_critic_from_state_dict

    critic = DualHeadCritic(in_dim=6)
    rebuilt = build_critic_from_state_dict(critic.state_dict())

    assert isinstance(rebuilt, DualHeadCritic)
    assert rebuilt.final_fusion is True
    assert rebuilt.final_combiner == "explicit"
    assert rebuilt.final_head[0].in_features == critic.final_head[0].in_features


def test_build_critic_from_state_dict_identifies_residual_combiner_checkpoint():
    from critic import DualHeadCritic
    from examples.diagnose_critic import build_critic_from_state_dict

    critic = DualHeadCritic(in_dim=6, final_combiner="residual")
    rebuilt = build_critic_from_state_dict(critic.state_dict())
    rebuilt.load_state_dict(critic.state_dict())

    assert isinstance(rebuilt, DualHeadCritic)
    assert rebuilt.final_fusion is True
    assert rebuilt.final_combiner == "residual"


def test_build_critic_from_state_dict_identifies_calibrated_combiner_checkpoint():
    from critic import DualHeadCritic
    from examples.diagnose_critic import build_critic_from_state_dict

    critic = DualHeadCritic(in_dim=6, final_combiner="calibrated")
    rebuilt = build_critic_from_state_dict(critic.state_dict())
    rebuilt.load_state_dict(critic.state_dict())

    assert isinstance(rebuilt, DualHeadCritic)
    assert rebuilt.final_fusion is True
    assert rebuilt.final_combiner == "calibrated"
    assert rebuilt.final_head[0].in_features == 3


def test_build_critic_from_state_dict_preserves_old_parallel_dual_head_checkpoint():
    from critic import DualHeadCritic
    from examples.diagnose_critic import build_critic_from_state_dict

    old_parallel = DualHeadCritic(in_dim=6, final_fusion=False)
    rebuilt = build_critic_from_state_dict(old_parallel.state_dict())
    rebuilt.load_state_dict(old_parallel.state_dict())

    assert isinstance(rebuilt, DualHeadCritic)
    assert rebuilt.final_fusion is False
    assert rebuilt.final_head[0].in_features == old_parallel.final_head[0].in_features


def test_find_epoch_checkpoints_requires_at_least_one_checkpoint():
    checkpoint_dir = Path("output/results") / f"test_empty_epoch_checkpoints_{uuid4().hex}"
    checkpoint_dir.mkdir(parents=True)
    notes_path = checkpoint_dir / "notes.txt"
    notes_path.write_text("not a checkpoint")

    try:
        with pytest.raises(ValueError, match="No epoch checkpoints"):
            find_epoch_checkpoints(checkpoint_dir)
    finally:
        notes_path.unlink(missing_ok=True)
        checkpoint_dir.rmdir()
