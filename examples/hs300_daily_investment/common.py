"""Shared identities and persistence for the separate daily investment run."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/hs300_daily_investment_20260908"
DATA = OUT / "data"
COLS = ["open", "high", "low", "close", "volume", "amount"]
CHECKPOINTS = [ROOT / f"output/results/regime_generalization/multiregime_train/seed_{i}/fusion_best.pt" for i in range(5)]
PERIODS = {"2024": ("2024-01-01", "2024-12-31"), "2025": ("2025-01-01", "2025-12-31"), "2026H1": ("2026-01-01", "2026-06-30")}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def identity(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def protect():
    """Read prior audit's protected path list; freeze actual current identities."""
    prior = ROOT / "output/results/hs300_evidence_audit_20260908/protected_inputs_before.json"
    paths = json.loads(prior.read_text(encoding="utf-8"))
    now = {p: digest(p) for p in paths}
    target = OUT / "protected_inputs.json"
    if target.exists():
        before = json.loads(target.read_text(encoding="utf-8"))
        changes = [p for p, h in before.items() if now.get(p) != h]
        write_json(OUT / "protected_check.json", {"files": len(before), "passed": not changes, "changed": changes})
        if changes:
            raise RuntimeError(f"Protected inputs changed: {changes}")
    else:
        write_json(target, now)
    return len(now)


if __name__ == "__main__":
    print({"protected_files": protect()})
