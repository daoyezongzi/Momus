import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_market_coverage_cli_reports_date_ranges_without_prices(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pd.DataFrame(
        {
            "timestamps": ["2022-01-04", "2022-01-05", "2024-12-31"],
            "open": [1, 2, 3],
            "high": [1, 2, 3],
            "low": [1, 2, 3],
            "close": [1, 2, 3],
            "volume": [100, 110, 120],
            "amount": [1000, 1100, 1200],
        }
    ).to_csv(data_dir / "300750.csv", index=False)
    codes = tmp_path / "codes.txt"
    codes.write_text("300750\n002594\n", encoding="utf-8")

    output = subprocess.check_output(
        [
            sys.executable,
            "examples/audit_market_coverage.py",
            "--codes-file",
            str(codes),
            "--data-dir",
            str(data_dir),
        ],
        text=True,
    )
    payload = json.loads(output)
    assert payload["n_codes"] == 2
    assert payload["missing_files"] == ["002594"]
    assert payload["coverage"]["300750"] == {
        "start": "2022-01-04",
        "end": "2024-12-31",
        "rows": 3,
        "has_2025_or_later": False,
    }
