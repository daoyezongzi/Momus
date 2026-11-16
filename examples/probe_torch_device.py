import json
from dataclasses import asdict
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from critic.device import select_existing_device


if __name__ == "__main__":
    print(json.dumps(asdict(select_existing_device()), ensure_ascii=False, indent=2))
