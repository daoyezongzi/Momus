"""Durable supervisor: resume candidates, run paired portfolios, verify inputs."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

import psutil

from common import ROOT, OUT, digest, protect, write_json

GEN_PYTHON = "C:/Users/Soyo/anaconda3/envs/torch5060/python.exe"
EVAL_PYTHON = str(ROOT / "output/runtime/hs300_investment/.venv/Scripts/python.exe")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pid_path = OUT / "workflow_pid.json"
    if pid_path.exists():
        previous = json.loads(pid_path.read_text(encoding="utf-8"))
        if psutil.pid_exists(previous["pid"]):
            process = psutil.Process(previous["pid"])
            if "workflow.py" in " ".join(process.cmdline()) and previous["pid"] != os.getpid():
                raise RuntimeError("Another daily investment workflow is active")
    write_json(pid_path, {"pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(), "python": sys.executable})
    stage, started = "preflight", time.time()
    try:
        protected = protect()
        smoke = json.loads((OUT / "smoke_results/evaluation_status.json").read_text(encoding="utf-8"))
        if smoke["status"] != "SMOKE_PASSED" or smoke["portfolios"] != 6:
            raise RuntimeError("Full-run gate requires all six real-data pilot portfolios")
        sources = {str(p.relative_to(ROOT)): digest(p) for p in (ROOT / "examples/hs300_daily_investment").glob("*.py")}
        write_json(OUT / "workflow_code_manifest.json", sources)
        stages = [("daily_candidates", GEN_PYTHON, "generate.py"), ("paired_portfolios", EVAL_PYTHON, "evaluate.py")]
        for stage, python, script in stages:
            write_json(OUT / "workflow_status.json", {"status": "RUNNING", "stage": stage, "pid": os.getpid(), "elapsed_seconds": time.time()-started, "manuscript_modified": False})
            for relative, sha in sources.items():
                if digest(ROOT / relative) != sha:
                    raise RuntimeError(f"Workflow source changed: {relative}")
            command = [python, "-u", str(ROOT / "examples/hs300_daily_investment" / script)]
            with (OUT / f"{stage}.log").open("a", encoding="utf-8") as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                write_json(OUT / "workflow_child.json", {"stage": stage, "pid": child.pid, "command": command})
                returncode = child.wait()
            if returncode:
                raise RuntimeError(f"{stage} exited {returncode}; inspect {stage}.log")
        stage = "verification"
        protected = protect()
        result = json.loads((OUT / "results/evaluation_status.json").read_text(encoding="utf-8"))
        if result["status"] != "COMPLETE" or result["portfolios"] != 18:
            raise RuntimeError("Incomplete final portfolio evaluation")
        write_json(OUT / "workflow_status.json", {"status": "COMPLETE", "stage": "verified", "pid": os.getpid(), "elapsed_seconds": time.time()-started, "protected_files": protected, "portfolios": 18, "manuscript_modified": False, "research_assumptions": json.loads((OUT / "data_ready.json").read_text(encoding="utf-8"))["limits"]})
    except BaseException as exc:
        write_json(OUT / "workflow_status.json", {"status": "FAILED", "stage": stage, "pid": os.getpid(), "elapsed_seconds": time.time()-started, "error": f"{type(exc).__name__}: {exc}", "manuscript_modified": False})
        raise


if __name__ == "__main__":
    main()
