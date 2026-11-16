import hashlib
import json
from pathlib import Path


def _canonical_bytes(config: dict) -> bytes:
    return json.dumps(config, sort_keys=True, separators=(",", ":")).encode()


def freeze_config(config: dict, path) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(config)
    digest = hashlib.sha256(payload).hexdigest()
    target.write_bytes(payload)
    Path(str(target) + ".sha256").write_text(digest, encoding="utf-8")
    return digest


def validate_phase(phase, allow_2025_labels, frozen_config=None, expected_hash=None):
    if phase in {"small-trial", "dev", "freeze"} and allow_2025_labels:
        raise RuntimeError("development phase cannot access 2025 labels")
    if phase != "sealed-2025":
        return None
    if not allow_2025_labels or frozen_config is None or expected_hash is None:
        raise RuntimeError("sealed-2025 requires explicit label permission and frozen config")
    actual = hashlib.sha256(Path(frozen_config).read_bytes()).hexdigest()
    if actual != expected_hash:
        raise RuntimeError("frozen config hash mismatch")
    return actual


def reserve_sealed_output(output_dir, config_hash: str) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    marker = path / "SEALED_RUN_RESERVED.json"
    try:
        with marker.open("x", encoding="utf-8") as handle:
            json.dump({"config_hash": config_hash}, handle, ensure_ascii=False, indent=2)
    except FileExistsError as exc:
        raise RuntimeError("sealed result already exists") from exc
