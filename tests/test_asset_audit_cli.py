import json
import subprocess
import sys
from pathlib import Path


def test_audit_cli_can_filter_to_mapped_codes_only(tmp_path):
    codes = tmp_path / "codes.txt"
    codes.write_text("300750\n002594\n000002\n", encoding="utf-8")
    output = subprocess.check_output(
        [
            sys.executable,
            "examples/audit_resonance_assets.py",
            "--codes-file",
            str(codes),
            "--mapped-only",
        ],
        text=True,
    )
    payload = json.loads(output)
    assert payload["n_codes"] == 2
    assert payload["n_mapped_targets"] == 2
    assert payload["excluded_unmapped"] == ["000002"]
