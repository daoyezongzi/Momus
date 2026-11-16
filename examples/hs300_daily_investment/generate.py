"""Resume daily paired N16 forecasts using frozen models and isolated shards."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import ROOT, OUT, DATA, COLS, CHECKPOINTS, identity, digest, write_json

sys.path.insert(0, str(ROOT))
from model import Kronos, KronosTokenizer
from model.kronos import auto_regressive_inference, calc_time_stamps
from critic.model import DualHeadCritic

BATCH = 16
PARAMETERS = dict(lookback=90, horizon=10, candidates=16, temperature=1., top_p=.9, top_k=0, clip=5., base_seed=100, batch_windows=BATCH, history="last 90 observed positive-volume bars at or before origin; all future timestamps from market calendar", weights="softmax(score), tau=1, no correlated-stock inputs")


def load_markets(codes):
    return {code: pd.read_csv(DATA / "market" / f"{code}.csv", parse_dates=["date"]).set_index("date").sort_index() for code in codes}


def inputs(markets, codes, origin, future):
    histories, stamps, means, scales, lasts, accepted, excluded, history_ends = [], [], [], [], [], [], [], []
    for code in sorted(codes):
        frame = markets[code]
        history = frame.loc[:origin]
        history = history[history.volume > 0].tail(90)
        if len(history) < 90:
            excluded.append({"instrument": code, "reason": "fewer_than_90_past_observations", "observations": len(history)})
            continue
        raw = history[COLS].to_numpy(np.float32)
        if not np.isfinite(raw).all():
            raise ValueError(f"nonfinite history: {code}/{origin}")
        mean, scale = raw.mean(0), raw.std(0) + 1e-5
        histories.append(np.clip((raw - mean) / scale, -5, 5))
        stamps.append(calc_time_stamps(pd.Series(history.index)).to_numpy(np.float32))
        means.append(mean); scales.append(scale); lasts.append(float(history.close.iloc[-1])); accepted.append(code)
        history_ends.append(str(history.index[-1].date()))
    if not accepted:
        raise ValueError("empty forecast batch")
    future_stamps = calc_time_stamps(pd.Series(future)).to_numpy(np.float32)
    return dict(hist=np.stack(histories), x_stamp=np.stack(stamps), y_stamp=np.repeat(future_stamps[None], len(accepted), axis=0), mean=np.stack(means), scale=np.stack(scales), last_close=np.array(lasts), codes=np.array(accepted), excluded=excluded, history_end=np.array(history_ends))


def load_models():
    # Restrict to already installed snapshots; no remote model update.
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base", local_files_only=True).to("cuda").eval()
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small", local_files_only=True).to("cuda").eval()
    critics = []
    for path in CHECKPOINTS:
        critic = DualHeadCritic(final_combiner="calibrated")
        critic.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        critics.append(critic.eval())
    return tokenizer, model, critics


def foundation_identities():
    hub = Path.home() / ".cache/huggingface/hub"
    files = {}
    for name in ("Kronos-small", "Kronos-Tokenizer-base"):
        root = hub / ("models--NeoQuasar--" + name)
        revision = (root / "refs/main").read_text().strip()
        for path in sorted((root / "snapshots" / revision).iterdir()):
            if path.is_file():
                files[str(path)] = digest(path)
    return files


@torch.no_grad()
def score(critic, paths, hist):
    b, n, h, f = paths.shape
    hp = critic.seq_encoder(torch.from_numpy(paths.reshape(b*n, h, f)))
    hx = critic.seq_encoder(torch.from_numpy(hist)).repeat_interleave(n, dim=0)
    hc = critic.corr_missing[None].expand(b*n, -1)
    feat = torch.cat([hp, hx, hc, hp*hc], dim=-1)
    return critic.final_head(torch.cat([critic.base_head(feat), critic.fine_head(feat), critic.coarse_head(feat)], dim=-1)).reshape(b,n).numpy()


def infer(tokenizer, model, critics, batch, origin):
    paths, scores = [], []
    for start in range(0, len(batch["codes"]), BATCH):
        stop = min(start+BATCH, len(batch["codes"]))
        # Fixed batch membership/order is part of manifest. Resuming a day uses
        # the identical RNG stream independent of previous days or interruptions.
        key = {"seed": 100, "origin": str(origin.date()), "codes": batch["codes"][start:stop].tolist()}
        seed = int(identity(key)[:8], 16)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        pred = auto_regressive_inference(tokenizer, model, torch.from_numpy(batch["hist"][start:stop]).cuda(), torch.from_numpy(batch["x_stamp"][start:stop]).cuda(), torch.from_numpy(batch["y_stamp"][start:stop]).cuda(), 512, 10, 5., 1., 0, .9, 16, False, return_individual=True)[:, :, -10:, :].astype(np.float32)
        paths.append(pred)
        s = np.stack([score(critic, pred, batch["hist"][start:stop]) for critic in critics])
        if start == 0:
            for i, critic in enumerate(critics):
                np.testing.assert_allclose(s[i, 0], critic.score(pred[0], batch["hist"][0]), rtol=2e-5, atol=5e-5)
        scores.append(s)
    paths = np.concatenate(paths)
    scores = np.concatenate(scores, axis=1)
    w = np.exp(scores.astype(float) - scores.max(axis=2, keepdims=True)); w /= w.sum(axis=2, keepdims=True)
    predictions = np.concatenate([paths.mean(axis=1)[None], np.einsum("sbn,bnhf->sbhf", w, paths)])
    signals = (predictions[..., 3].mean(axis=2) * batch["scale"][:, 3] + batch["mean"][:, 3]) / batch["last_close"] - 1
    if not np.isfinite(paths).all() or not np.isfinite(signals).all():
        raise ValueError("nonfinite candidate or signal")
    return paths, scores, signals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2026-06-30")
    parser.add_argument("--max-days", type=int)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    calendar = pd.DatetimeIndex(pd.read_csv(DATA / "market/SH000300.csv", parse_dates=["date"]).date)
    if args.benchmark:
        markets = load_markets(["SZ002837"])
        day = pd.Timestamp("2024-07-01"); ix = calendar.get_loc(day)
        batch = inputs(markets, ["SZ002837"], calendar[ix-1], calendar[ix:ix+10])
        # Repeat the same observable history to measure a full production batch.
        for key in ("hist", "x_stamp", "y_stamp", "mean", "scale", "last_close", "codes", "history_end"):
            batch[key] = np.repeat(batch[key], BATCH, axis=0)
        models = load_models()
        for i in range(2):
            start = time.time(); _, _, s = infer(*models, batch, calendar[ix-1]); torch.cuda.synchronize()
            print({"benchmark_batch": i, "windows": BATCH, "seconds": time.time()-start, "finite_signals": bool(np.isfinite(s).all())}, flush=True)
        return
    gate = json.loads((OUT / "data_ready.json").read_text(encoding="utf-8"))
    if not gate["generation_ready"]:
        raise RuntimeError("Data audit gate has not passed")
    for relative, sha in gate["input_sha256"].items():
        if digest(OUT / relative) != sha:
            raise ValueError(f"Frozen input changed: {relative}")
    members = pd.read_csv(OUT / "membership.csv", parse_dates=["start_date", "end_date"])
    markets = load_markets(members.instrument.unique())
    manifest = {"parameters": PARAMETERS, "data_identity": identity(gate), "checkpoints": [digest(p) for p in CHECKPOINTS], "foundation_models": foundation_identities(), "generator_source": digest(ROOT / "model/kronos.py"), "critic_source": digest(ROOT / "critic/model.py"), "generation_code": digest(__file__), "torch_version": torch.__version__, "cuda_version": torch.version.cuda, "device": torch.cuda.get_device_name(0)}
    manifest_id = identity(manifest)
    mpath = OUT / "generation_manifest.json"
    if mpath.exists() and json.loads(mpath.read_text(encoding="utf-8")) != manifest:
        raise RuntimeError("Generation manifest changed; use a separate experiment directory")
    write_json(mpath, manifest)
    days = calendar[(calendar >= args.start) & (calendar <= args.end)]
    if args.max_days:
        days = days[:args.max_days]
    folder = OUT / "candidates"; folder.mkdir(exist_ok=True)
    models = None
    started = time.time(); newly_done = 0
    for i, day in enumerate(days):
        ix = calendar.get_loc(day); origin = calendar[ix-1]; future = calendar[ix:ix+10]
        if len(future) != 10:
            raise ValueError("Market calendar does not cover forecast horizon")
        target = folder / f"{day:%Y%m%d}.npz"
        meta_path = target.with_suffix(".json")
        if target.exists() and meta_path.exists():
            saved = json.loads(meta_path.read_text(encoding="utf-8"))
            if saved["manifest"] != manifest_id or digest(target) != saved["sha256"]:
                raise ValueError(f"Invalid saved candidate shard {target}")
            continue
        codes = members.loc[(members.start_date <= day) & (members.end_date >= day), "instrument"].tolist()
        if len(codes) != 300 or len(set(codes)) != 300:
            raise ValueError(f"Invalid membership on {day}")
        batch = inputs(markets, codes, origin, future)
        if models is None:
            models = load_models()
        t0 = time.time(); paths, scores, signals = infer(*models, batch, origin)
        with target.with_suffix(".tmp").open("wb") as handle:
            np.savez_compressed(handle, paths=paths, scores=scores, signals=signals, hist=batch["hist"], mean=batch["mean"], scale=batch["scale"], last_close=batch["last_close"], codes=batch["codes"], history_end=batch["history_end"], origin=str(origin.date()), execution=str(day.date()), future=future.strftime("%Y-%m-%d").to_numpy(dtype=str))
        os.replace(target.with_suffix(".tmp"), target)
        write_json(meta_path, {"manifest": manifest_id, "sha256": digest(target), "origin": str(origin.date()), "execution": str(day.date()), "windows": len(paths), "excluded": batch["excluded"], "seconds": time.time()-t0})
        newly_done += 1
        elapsed = time.time()-started
        write_json(OUT / "generation_status.json", {"status": "RUNNING", "requested_days": len(days), "last_execution": str(day.date()), "newly_completed_days": newly_done, "completed_shards": len(list(folder.glob("*.npz"))), "elapsed_seconds": elapsed, "mean_seconds_per_new_day": elapsed/newly_done, "models_trained": False})
        print(f"generated {i+1}/{len(days)} {day:%Y-%m-%d} windows={len(paths)} seconds={time.time()-t0:.2f}", flush=True)
    write_json(OUT / "generation_status.json", {"status": "REQUESTED_RANGE_COMPLETE", "requested_start": args.start, "requested_end": args.end, "requested_days": len(days), "partial_max_days": args.max_days, "completed_shards": len(list(folder.glob('*.npz'))), "elapsed_seconds": time.time()-started, "models_trained": False})


if __name__ == "__main__":
    main()
