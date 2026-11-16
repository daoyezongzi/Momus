"""Generate only the frozen 37 full-universe dates; preserve daily caches."""
import json
import os
import time
import numpy as np
import pandas as pd
import torch
from common_lowfreq import ROOT, BASE, OUT, CHECKPOINTS, digest, identity, write_json, freeze, protect, calendar
import generate as daily


def main():
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    protocol = freeze()
    protect("before_generation")
    sources = [__file__, str(ROOT / "examples/hs300_lowfreq_investment/common_lowfreq.py"),
               daily.__file__, str(ROOT / "model/kronos.py"), str(ROOT / "critic/model.py")]
    manifest = dict(protocol=identity(protocol), parameters=daily.PARAMETERS,
        source_sha256={p: digest(p) for p in sources}, checkpoints=[digest(p) for p in CHECKPOINTS],
        foundation_models=daily.foundation_identities(), torch_version=torch.__version__,
        cuda_version=torch.version.cuda, device=torch.cuda.get_device_name(0))
    target = OUT / "generation_manifest.json"
    if target.exists() and json.loads(target.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Generation identity changed; refusing mixed caches")
    write_json(target, manifest)
    manifest_id = identity(manifest)
    cal = calendar()
    markets = daily.load_markets(sorted(set(sum(protocol["universes"].values(), []))))
    folder = OUT / "candidates"
    folder.mkdir(exist_ok=True)
    models = None
    started = time.time()
    for i, row in enumerate(protocol["schedule"]):
        day, origin = pd.Timestamp(row["execution"]), pd.Timestamp(row["origin"])
        target = folder / f"{day:%Y%m%d}.npz"
        if target.exists() and target.with_suffix(".json").exists():
            meta = json.loads(target.with_suffix(".json").read_text(encoding="utf-8"))
            if meta["manifest"] != manifest_id or meta["sha256"] != digest(target):
                raise ValueError(f"Invalid existing shard: {target}")
            continue
        j = cal.get_loc(day)
        future = cal[j:j+10]
        batch = daily.inputs(markets, protocol["universes"][row["execution"]], origin, future)
        if batch["excluded"] or len(batch["codes"]) != 300:
            raise ValueError("A scheduled date does not cover all 300 members")
        if models is None:
            models = daily.load_models()
        t0 = time.time()
        paths, scores, signals = daily.infer(*models, batch, origin)
        with target.with_suffix(".tmp").open("wb") as handle:
            np.savez_compressed(handle, paths=paths, scores=scores, signals=signals,
                hist=batch["hist"], mean=batch["mean"], scale=batch["scale"], last_close=batch["last_close"],
                codes=batch["codes"], history_end=batch["history_end"], origin=row["origin"],
                execution=row["execution"], future=future.strftime("%Y-%m-%d").to_numpy(dtype=str))
        os.replace(target.with_suffix(".tmp"), target)
        write_json(target.with_suffix(".json"), dict(manifest=manifest_id, sha256=digest(target),
            origin=row["origin"], execution=row["execution"], windows=300, seconds=time.time()-t0))
        write_json(OUT / "generation_status.json", dict(status="RUNNING", completed=i+1,
            requested=37, last_execution=row["execution"], elapsed_seconds=time.time()-started, pid=os.getpid()))
        print(f"generated {i+1}/37 {row['execution']} stocks=300 seconds={time.time()-t0:.2f}", flush=True)
    protect("after_generation")
    write_json(OUT / "generation_status.json", dict(status="COMPLETE", completed=37,
        requested=37, elapsed_seconds=time.time()-started, models_trained=False))


if __name__ == "__main__":
    main()
