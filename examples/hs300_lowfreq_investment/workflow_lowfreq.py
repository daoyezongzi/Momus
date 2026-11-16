"""Watch the explicitly launched generator, then evaluate and seal outputs."""
import json
import os
import subprocess
import time
from datetime import datetime, timezone
import psutil
from common_lowfreq import ROOT, OUT, digest, identity, write_json, protect


def generation_running():
    matches = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            if any(str(arg).replace("\\", "/").endswith("hs300_lowfreq_investment/generate_lowfreq.py")
                   for arg in process.info["cmdline"] or []):
                matches.append(process.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return matches


def main():
    started = time.time()
    stage = "waiting_for_generation"
    target = OUT / "workflow_status.json"
    if target.exists():
        previous = json.loads(target.read_text(encoding="utf-8"))
        pid = previous.get("pid")
        if pid and psutil.pid_exists(pid) and pid != os.getpid():
            if any("workflow_lowfreq.py" in x for x in psutil.Process(pid).cmdline()):
                raise ValueError("Another low-frequency supervisor is active")
    try:
        sources = {str(p):digest(p) for p in (ROOT / "examples/hs300_lowfreq_investment").glob("*.py")}
        write_json(OUT / "workflow_code_manifest.json", sources)
        while True:
            state_path = OUT / "generation_status.json"
            state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
            if state.get("status") == "COMPLETE" and state.get("completed") == 37:
                break
            active = generation_running()
            if not active:
                raise RuntimeError("Generator exited before 37 frozen dates were complete")
            write_json(target, dict(status="RUNNING", stage=stage, pid=os.getpid(), generator_pids=active,
                completed=state.get("completed",0), requested=37, elapsed_seconds=time.time()-started))
            time.sleep(10)
        stage = "evaluation_preflight"
        for path, sha in sources.items():
            if digest(path) != sha:
                raise ValueError(f"Source changed while generation ran: {path}")
        smoke = json.loads((OUT / "smoke_results/evaluation_status.json").read_text(encoding="utf-8"))
        if smoke["status"] != "SMOKE_PASSED" or smoke["portfolios"] != 6:
            raise ValueError("Six low-frequency real-data smoke portfolios required")
        stage = "paired_portfolios"
        write_json(target, dict(status="RUNNING", stage=stage, pid=os.getpid(), completed=37))
        command = [str(ROOT / "output/runtime/hs300_investment/.venv/Scripts/python.exe"), "-u",
                   str(ROOT / "examples/hs300_lowfreq_investment/evaluate_lowfreq.py")]
        with (OUT / "evaluation.log").open("a", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            write_json(OUT / "evaluation_process.json", dict(pid=child.pid, command=command))
            code = child.wait()
        if code:
            raise RuntimeError(f"Evaluation failed with exit {code}")
        status = json.loads((OUT / "results/evaluation_status.json").read_text(encoding="utf-8"))
        if status["status"] != "COMPLETE" or status["portfolios"] != 12 or status["forecast_dates"] != 37:
            raise ValueError("Incomplete final results")
        stage = "seal"
        protected = protect("workflow_complete")
        for path, sha in sources.items():
            if digest(path) != sha:
                raise ValueError(f"Evaluation source changed: {path}")
        files = {str(p.relative_to(OUT)):digest(p) for p in (OUT / "results").rglob("*") if p.is_file()}
        write_json(OUT / "results_manifest.json", dict(files=files, protocol_sha256=digest(OUT / "protocol.json"),
            generation_manifest_sha256=digest(OUT / "generation_manifest.json")))
        write_json(target, dict(status="COMPLETE", stage="verified", pid=os.getpid(), completed=37,
            portfolios=12, protected_files=protected, elapsed_seconds=time.time()-started,
            completed_utc=datetime.now(timezone.utc).isoformat(), manuscript_modified=False))
    except BaseException as exc:
        write_json(target, dict(status="FAILED", stage=stage, pid=os.getpid(),
            error=f"{type(exc).__name__}: {exc}", elapsed_seconds=time.time()-started))
        raise


if __name__ == "__main__":
    main()
