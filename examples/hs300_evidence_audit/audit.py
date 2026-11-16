"""Read-only-input HS300 audit; all derived files go to a separate directory.

Run with the existing torch5060 Python, from the repository root.
No training, candidate generation, original-output overwrite or manuscript edit.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pickle
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OLD = ROOT / "output/results/hs300_main_20260907"
OUT = ROOT / "output/results/hs300_evidence_audit_20260908"
PERIODS = ("2024", "2025", "2026H1")
METRICS = ("ohlc_mse", "ohlc_mae", "price_ic", "price_rankic")
METHODS = ("simple_mean", "median", "momus_weighted", "momus_top1")
TRAIN = ROOT / "output/results/regime_generalization/samples_multiregime_train_dual.pkl"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def save(name, rows):
    pd.DataFrame(rows).to_csv(OUT / name, index=False, float_format="%.12g")


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def protected_paths():
    paths = list((ROOT / "paper").rglob("*.md"))
    paths += [Path("C:/Users/Soyo/Desktop/123/03.md"), TRAIN]
    paths += list((ROOT / "model").rglob("*.py")) + list((ROOT / "critic").rglob("*.py"))
    paths += list(OLD.glob("*"))
    paths += list((ROOT / "data/references").glob("hs300_*.txt"))
    paths += [ROOT / "docs/development/protocols/2026-09-07_hs300_main_preregistration.md"]
    paths += [ROOT / f"examples/{x}.py" for x in ("evaluate_hs300_main", "generate_hs300_main_cache", "run_hs300_investment")]
    for period in PERIODS:
        paths += [ROOT / f"output/candidates/hs300_{period}_N16.pkl"]
        paths += list((ROOT / f"data/market_{period}_hs300").glob("*.csv"))
    paths += list((ROOT / "output/results/regime_generalization/multiregime_train").rglob("*.pt"))
    return sorted(set(p for p in paths if p.is_file()))


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "protected_inputs_before.json"
    if target.exists():
        return verify_protection()
    dump("protected_inputs_before.json", {str(p): {"sha256": digest(p), "bytes": p.stat().st_size} for p in protected_paths()})
    status = subprocess.check_output(["git", "status", "--porcelain=v1", "-uall"], cwd=ROOT, text=True, encoding="utf-8")
    dump("repository_before.json", {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "status": status})


def verify_protection():
    before = json.loads((OUT / "protected_inputs_before.json").read_text(encoding="utf-8"))
    changed = [p for p, v in before.items() if not Path(p).exists() or digest(p) != v["sha256"]]
    result = {"checked_files": len(before), "changed": changed, "passed": not changed}
    dump("protected_inputs_check.json", result)
    assert not changed, changed
    return result


def metrics(pred, truth):
    """Independent array implementation: last two dimensions horizon, OHLC."""
    p, t = np.asarray(pred, float)[..., :4], np.asarray(truth, float)[..., :4]
    e = p - t
    out = {"ohlc_mse": np.mean(e*e, axis=(-2, -1)), "ohlc_mae": np.mean(abs(e), axis=(-2, -1))}
    for name, rank in (("price_ic", False), ("price_rankic", True)):
        if rank:
            # pandas' average ties are independent of the original hand-written rank function.
            pp = pd.DataFrame(p.reshape(-1, 10, 4).transpose(1, 0, 2).reshape(10, -1)).rank().to_numpy().reshape(10, -1, 4).transpose(1, 0, 2).reshape(p.shape)
            tt = pd.DataFrame(t.reshape(-1, 10, 4).transpose(1, 0, 2).reshape(10, -1)).rank().to_numpy().reshape(10, -1, 4).transpose(1, 0, 2).reshape(t.shape)
        else:
            pp, tt = p, t
        pc, tc = pp-pp.mean(axis=-2, keepdims=True), tt-tt.mean(axis=-2, keepdims=True)
        denom = np.sqrt((pc*pc).sum(axis=-2)*(tc*tc).sum(axis=-2))
        corr = np.divide((pc*tc).sum(axis=-2), denom, out=np.full(denom.shape, np.nan), where=denom>0)
        out[name] = np.nanmean(corr, axis=-1)
    return out


def inputs_and_forecasts():
    import torch
    from critic.model import DualHeadCritic
    torch.set_num_threads(2)
    ev = module("hs300_original_forecast", ROOT / "examples/evaluate_hs300_main.py")
    engine = ev.load_numpy_engine()
    critics, torch_critics = {}, {}
    for seed in range(5):
        p = ROOT / f"output/results/regime_generalization/multiregime_train/seed_{seed}/fusion_best.pt"
        critics[seed] = engine.load_dual_calibrated(p)
        tc = DualHeadCritic(final_combiner="calibrated")
        tc.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
        torch_critics[seed] = tc.eval()
    training = pickle.load(TRAIN.open("rb"))
    train_codes = set(str(s.code).zfill(6) for s in training)
    dump("training_audit.json", {"windows": len(training), "codes": sorted(train_codes), "hist_shapes": sorted(set(str(s.hist.shape) for s in training)), "missing_meta": sum(s.meta is None for s in training), "corr_available": sum(s.corr_feat is not None for s in training), "date_status": "cache metadata lacks timestamps; training period is documented, not independently timestamp-reconstructed"})
    f = pd.read_csv(OLD / "forecast_window_metrics.csv", dtype={"code": str, "period": str})
    assert not f.duplicated(["period", "seed", "window_index", "method"]).any()
    assert np.isfinite(f[list(METRICS)]).all().all()
    checks, coverage, quality, jumps, parity = [], [], [], [], []
    window_info, membership = [], []
    frozen = json.loads((OLD / "protocol.json").read_text())
    for period in PERIODS:
        print("audit inputs and forecasts", period, flush=True)
        cp = ROOT / f"output/candidates/hs300_{period}_N16.pkl"
        assert digest(cp) == frozen["caches"][period]["sha256"]
        cache = pickle.load(cp.open("rb"))
        windows, candidates = cache["windows"], cache["candidates"]
        codes = set(str(w[0]).zfill(6) for w in windows)
        frames = {}
        for code in sorted(codes):
            p = ROOT / f"data/market_{period}_hs300/{code}.csv"
            d = pd.read_csv(p, parse_dates=["timestamps"]).set_index("timestamps")
            frames[code] = d
            prices = d[["open", "high", "low", "close"]]
            bad = (d.high < prices.max(axis=1)-1e-8) | (d.low > prices.min(axis=1)+1e-8)
            quality.append({"period":period,"code":code,"rows":len(d),"duplicate_dates":int(d.index.duplicated().sum()),"finite":bool(np.isfinite(d.to_numpy(float)).all()),"positive_prices":bool((prices>0).all().all()),"ohlc_violations":int(bad.sum()),"nonpositive_volume":int((d.volume<=0).sum()),"sorted":bool(d.index.is_monotonic_increasing)})
            gap = d.open / d.close.shift(1)-1
            for date in gap.index[abs(gap)>0.2]:
                jumps.append({"period":period,"code":code,"date":date,"overnight_return":float(gap.loc[date]),"status":"flag only; corporate action versus market move unresolved"})
            membership.append({"period":period,"code":code,"in_training_codes":code in train_codes,"point_in_time_membership":"unconfirmed"})
        hist, truths, mus, scales, infos = [], [], [], [], []
        for ix, (code, x, xt, yt, y) in enumerate(windows):
            code = str(code).zfill(6)
            xt, yt = pd.DatetimeIndex(xt), pd.DatetimeIndex(yt)
            assert len(xt)==90 and len(yt)==10 and xt.max()<yt.min()
            raw = x.to_numpy(np.float32)
            assert np.array_equal(raw, frames[code].loc[xt].to_numpy(np.float32))
            assert np.array_equal(np.asarray(y,np.float32), frames[code].loc[yt].to_numpy(np.float32))
            mean, scale = raw.mean(axis=0), raw.std(axis=0)+1e-5
            hist.append(np.clip((raw-mean)/scale,-5,5))
            truths.append((np.asarray(y,np.float32)-mean)/scale)
            mus.append(mean); scales.append(scale)
            infos.append({"period":period,"window_index":ix,"code":code,"origin":str(xt[-1].date()),"entry":str(yt[0].date()),"horizon_end":str(yt[-1].date()),"history_start":str(xt[0].date()),"in_training_codes":code in train_codes})
        window_info += infos
        h, truth, paths = np.asarray(hist), np.asarray(truths), np.asarray(candidates)
        assert paths.shape == (len(windows),16,10,6) and np.isfinite(paths).all()
        pp = f[f.period==period]
        inf = pd.DataFrame(infos)
        freq = inf.groupby("origin").size()
        overlap = 0
        for _, g in inf.groupby("code"):
            g = g.sort_values("origin")
            overlap += int((g.entry.iloc[1:].to_numpy()<=g.horizon_end.iloc[:-1].to_numpy()).sum())
        coverage.append({"period":period,"stocks":len(codes),"windows":len(windows),"training_overlap_stocks":len(codes&train_codes),"unseen_stocks":len(codes-train_codes),"origin_dates":len(freq),"origins_with_at_least_300_signals":int((freq>=300).sum()),"min_cross_section":int(freq.min()),"max_cross_section":int(freq.max()),"first_origin":inf.origin.min(),"first_entry":inf.entry.min(),"last_horizon":inf.horizon_end.max(),"overlap_pairs":overlap,"windows_before_or_crossing_2024_07_01":int((inf.entry<"2024-07-01").sum()) if period=="2024" else 0})
        for method,pred in (("simple_mean",paths.mean(axis=1)),("median",np.median(paths,axis=1))):
            independent = metrics(pred,truth)
            for seed in range(5):
                prior = pp[(pp.method==method)&(pp.seed==seed)].sort_values("window_index")
                assert len(prior)==len(windows)
                for name in METRICS:
                    err = float(np.max(abs(independent[name]-prior[name].to_numpy())))
                    checks.append({"period":period,"method":method,"seed":seed,"metric":name,"max_abs_error":err,"scope":"all baseline windows"})
                    assert err<2e-7, (period,method,name,err)
        # Fixed evenly-spaced indices, independent of values; five seeds, 12 windows/period.
        sample_ids = np.unique(np.linspace(0,len(windows)-1,12,dtype=int))
        for seed in range(5):
            for ix in sample_ids:
                ns = critics[seed].score(paths[ix],h[ix])
                ts = torch_critics[seed].score(paths[ix],h[ix])
                score_err = float(np.max(abs(ns-ts)))
                assert score_err < 5e-5, score_err
                w = np.exp(ts.astype(float)-ts.max()); w/=w.sum()
                preds = {"momus_weighted":np.einsum("n,nhf->hf",w,paths[ix]),"momus_top1":paths[ix,int(np.argmax(ts))]}
                assert np.argmax(ns)==np.argmax(ts)
                row = {"period":period,"seed":seed,"window_index":int(ix),"score_error":score_err,"top1_same":True}
                for method,pred in preds.items():
                    im = metrics(pred,truth[ix])
                    old = pp[(pp.seed==seed)&(pp.method==method)&(pp.window_index==ix)].iloc[0]
                    for name in METRICS:
                        err=float(abs(im[name]-old[name]))
                        row[f"{method}_{name}_error"]=err
                        assert err<5e-5,(period,seed,ix,method,name,err)
                raw_pred=preds["momus_weighted"]*scales[ix]+mus[ix]
                raw_truth=truth[ix]*scales[ix]+mus[ix]
                a,b=metrics(preds["momus_weighted"],truth[ix]),metrics(raw_pred,raw_truth)
                row["affine_ic_error"]=float(abs(a["price_ic"]-b["price_ic"]))
                parity.append(row)
        # Reconcile original per-seed summaries from every stored window.
        oldsum=pd.read_csv(OLD/"forecast_per_seed.csv")
        for (seed,method),g in pp.groupby(["seed","method"]):
            prior=oldsum[(oldsum.period==period)&(oldsum.seed==seed)&(oldsum.method==method)].iloc[0]
            for name in METRICS:
                err=float(abs(g[name].mean()-prior[name]))
                assert err<2e-8,(period,seed,method,name,err)
        del cache, candidates, windows, paths
    save("data_quality.csv",quality); save("price_jump_flags.csv",jumps)
    save("coverage.csv",coverage); save("membership.csv",membership); save("windows.csv",window_info)
    save("forecast_baseline_recheck.csv",checks); save("inference_parity.csv",parity)
    dump("forecast_audit_checks.json",{"all_original_window_summaries_reconciled":True,"full_baseline_recomputation":True,"momus_full_reinference":False,"momus_spot_checks":len(parity),"max_numpy_torch_score_difference":max(r["score_error"] for r in parity),"cache_truth_matches_market_csv":True,"all_shapes_finite":True})


def bootstrap_effects(a,b,stocks,times,reps=5000,kind="stock",seed=20260908):
    """Ratio of resampled sums, preserving the window-weighted estimand.

    Crossed/pigeonhole bootstrap uses independent stock and contiguous time-block
    multiplicities. It is a sensitivity analysis, not an iid-window bootstrap.
    """
    rng=np.random.default_rng(seed)
    ns,nt=int(stocks.max()+1),int(times.max()+1)
    count=np.zeros((ns,nt)); sa=np.zeros((ns,nt,a.shape[1])); sb=np.zeros_like(sa)
    np.add.at(count,(stocks,times),1); np.add.at(sa,(stocks,times),a); np.add.at(sb,(stocks,times),b)
    chunks=[]
    for start in range(0,reps,100):
        n=min(100,reps-start)
        ws=rng.multinomial(ns,np.ones(ns)/ns,size=n) if kind in ("stock","crossed") else np.ones((n,ns))
        wt=rng.multinomial(nt,np.ones(nt)/nt,size=n) if kind in ("time","crossed") else np.ones((n,nt))
        denom=np.einsum("bs,st,bt->b",ws,count,wt,optimize=True)
        asum=np.einsum("bs,stm,bt->bm",ws,sa,wt,optimize=True)
        bsum=np.einsum("bs,stm,bt->bm",ws,sb,wt,optimize=True)
        delta=(asum-bsum)/denom[:,None]
        relative=(bsum[:,0]-asum[:,0])/bsum[:,0]*100
        chunks.append(np.column_stack([delta,relative]))
    samples=np.concatenate(chunks)
    return np.quantile(samples,[.025,.975,.05/6,1-.05/6],axis=0)


def statistics():
    f=pd.read_csv(OLD/"forecast_window_metrics.csv",dtype={"code":str,"period":str})
    windows=pd.read_csv(OUT/"windows.csv",dtype={"code":str})
    out, slices, time_effects=[],[],[]
    comparisons=(("momus_weighted","simple_mean"),("momus_top1","simple_mean"),("momus_weighted","median"),("momus_top1","momus_weighted"))
    for period in PERIODS:
        print("paired bootstrap",period,flush=True)
        pp=f[f.period==period]
        info=windows[windows.period==period].sort_values("window_index")
        stock=pd.factorize(info.code,sort=True)[0]
        calendar=pd.DatetimeIndex(sorted(set().union(*(set(pd.read_csv(p,usecols=["timestamps"]).timestamps) for p in (ROOT/f"data/market_{period}_hs300").glob("*.csv")))))
        pos=calendar.get_indexer(pd.to_datetime(info.origin)); pos-=pos.min()
        avg={m:pp[pp.method==m].groupby("window_index")[list(METRICS)].mean().sort_index().to_numpy() for m in METHODS}
        for m,base in comparisons:
            a,b=avg[m],avg[base]
            for name,mask in (("all",np.ones(len(info),bool)),("training_seen",info.in_training_codes.to_numpy(bool)),("training_unseen",~info.in_training_codes.to_numpy(bool)),("post_2024_06",(info.origin>="2024-07-01").to_numpy())):
                if not mask.any():continue
                d=a[mask].mean(0)-b[mask].mean(0)
                row={"period":period,"method":m,"baseline":base,"slice":name,"windows":int(mask.sum()),"stocks":info.loc[mask,"code"].nunique(),"mse_improvement_pct":float((1-a[mask,0].mean()/b[mask,0].mean())*100)}
                row.update({f"delta_{met}":v for met,v in zip(METRICS,d)})
                slices.append(row)
            stockmeans=pd.DataFrame(a-b,columns=METRICS).assign(code=info.code.to_numpy()).groupby("code").mean().mean()
            for kind,block in (("stock",10),("time",10),("crossed",10),("time",20),("crossed",20)):
                times=pd.factorize(pos//block,sort=True)[0]
                q=bootstrap_effects(a,b,stock,times,kind=kind)
                # One/two time blocks cannot estimate market-time uncertainty.
                # Suppress nominal intervals instead of publishing degenerate precision.
                time_insufficient=kind!="stock" and times.max()+1<5
                if time_insufficient:q[:]=np.nan
                for j,met in enumerate((*METRICS,"mse_improvement_pct")):
                    effect=float((a-b).mean(0)[j]) if j<4 else float((1-a[:,0].mean()/b[:,0].mean())*100)
                    out.append({"period":period,"method":m,"baseline":base,"metric":met,"resampling":kind,"block_days":block if kind!="stock" else 0,"stocks":int(stock.max()+1),"time_blocks":int(times.max()+1),"replicates":5000,"effect":effect,"ci95_low":q[0,j],"ci95_high":q[1,j],"three_period_bonferroni_low":q[2,j],"three_period_bonferroni_high":q[3,j],"stock_equal_delta":float(stockmeans[met]) if j<4 else np.nan,"interpretation":"insufficient_time_blocks_interval_withheld" if time_insufficient else "conditional_on_observed_dates" if kind=="stock" else "few_blocks_exploratory" if times.max()+1<10 else "block_sensitivity"})
            for block_id in sorted(set(pos//10)):
                mask=pos//10==block_id
                time_effects.append({"period":period,"method":m,"baseline":base,"block":int(block_id),"windows":int(mask.sum()),"first_origin":info.loc[mask,"origin"].min(),"last_origin":info.loc[mask,"origin"].max(),"delta_mse":float((a[mask,0]-b[mask,0]).mean()),"mse_improvement_pct":float((1-a[mask,0].mean()/b[mask,0].mean())*100)})
    save("paired_intervals.csv",out);save("forecast_subgroups.csv",slices);save("forecast_time_blocks.csv",time_effects)
    dump("statistics_specification.json",{"post_result_supplementary":True,"rng_seed":20260908,"replicates":5000,"periods":PERIODS,"primary":"weighted-minus-mean MSE by period","seed_handling":"average paired metrics over five seeds before market resampling","baseline":"deterministic repeated baseline not independent runs","intervals":"percentile 95%; 98.333% interval for primary three-period family, not all secondary comparisons","time_blocks":"non-overlapping 10/20 trading dates anchored at first origin, clustered by origin; history and horizon can cross block boundaries","crossed":"independent stock and time-block multiplicities; sparse-cell pigeonhole sensitivity","limitation":"conditional on fixed candidates/checkpoints and retrospective chosen universe; finite blocks do not establish future-market generalization"})


def corrected_benchmark(frame):
    """Original Tencent CSV named the API's open field close and vice versa."""
    fixed=frame.copy()
    fixed[["open","close"]]=frame[["close","open"]].to_numpy()
    return fixed


def investment_metrics(equities, benchmark_closes, initial_nav=None, benchmark_initial=None):
    e=np.asarray(equities,float); b=np.asarray(benchmark_closes,float)
    if initial_nav is not None:
        e=np.r_[initial_nav,e]; b=np.r_[benchmark_initial,b]
    r=e[1:]/e[:-1]-1; rb=b[1:]/b[:-1]-1; active=r-rb
    n=len(r); sd=active.std(ddof=1) if n>1 else np.nan
    return {"aer":float(252*active.mean()),"ir":float(np.sqrt(252)*active.mean()/sd) if sd>0 else np.nan,"annual_return":float((e[-1]/e[0])**(252/n)-1),"total_return_net":float(e[-1]/e[0]-1),"benchmark_total":float(b[-1]/b[0]-1),"mdd":float(np.min(e/np.maximum.accumulate(e)-1)),"valid_days":n},r,rb


def investment():
    raw=pd.read_csv(OLD/"benchmark_sh000300.csv",parse_dates=["date"]).set_index("date")
    fixed=corrected_benchmark(raw)
    fixed.to_csv(OUT/"benchmark_sh000300_corrected.csv",float_format="%.12g")
    daily=pd.read_csv(OLD/"investment_daily.csv",parse_dates=["date"])
    trades=pd.read_csv(OLD/"investment_trades.csv",dtype={"code":str,"period":str},parse_dates=["date","signal_date"])
    sig=pd.read_csv(OLD/"investment_signals.csv",dtype={"code":str,"period":str},parse_dates=["origin","entry","horizon_end"])
    prior=pd.read_csv(OLD/"investment_per_seed.csv")
    rows,checks,event_rows,corrected_daily=[],[],[],[]
    for period in PERIODS:
        print("investment independent ledger replay",period,flush=True)
        codes=sorted(sig[sig.period==period].code.unique())
        market={code:pd.read_csv(ROOT/f"data/market_{period}_hs300/{code}.csv",parse_dates=["timestamps"]).set_index("timestamps") for code in codes}
        calendar=pd.DatetimeIndex(sorted(set().union(*(set(x.index) for x in market.values()))))
        # Daily ffill marks recreate the documented stale-close valuation convention.
        close=pd.DataFrame({c:x.close for c,x in market.items()}).reindex(calendar).ffill()
        for (seed,method),g in daily[daily.period==period].groupby(["seed","method"]):
            g=g.sort_values("date")
            tr=trades[(trades.period==period)&(trades.seed==seed)&(trades.method==method)]
            snapshots=sig[(sig.period==period)&(sig.seed==seed)&(sig.method==method)]
            cash=1e8; shares=defaultdict(float); fees=0.; max_nav=0.; max_cash=0.; max_fee=0.; max_price=0.; max_notional=0.; min_cash=0.; min_shares=0.; missing_sells=0; initial_cash=1e8
            grouped={d:t for d,t in tr.groupby("date",sort=False)}
            acquired={}
            hold_violation=0
            for drow in g.itertuples():
                date=drow.date
                for t in grouped.get(date,pd.DataFrame()).itertuples():
                    assert t.signal_date < date
                    price=float(market[t.code].loc[date,"open"])
                    max_price=max(max_price,abs(price-t.price))
                    max_notional=max(max_notional,abs(t.quantity*price-t.notional))
                    expected_fee=max(t.notional*(.001 if t.side=="buy" else .0015),5.)
                    max_fee=max(max_fee,abs(expected_fee-t.fee))
                    if t.side=="buy":
                        cash-=t.notional+t.fee
                        if shares[t.code]<=1e-7:acquired[t.code]=date
                        shares[t.code]+=t.quantity
                    else:
                        if t.code in acquired:
                            age=calendar.get_loc(date)-calendar.get_loc(acquired[t.code])
                            hold_violation+=int(age<5)
                        current_codes=set(snapshots[snapshots.origin==t.signal_date].code)
                        missing_sells+=int(t.code not in current_codes)
                        cash+=t.notional-t.fee;shares[t.code]-=t.quantity
                    fees+=t.fee;min_cash=min(min_cash,cash);min_shares=min(min_shares,shares[t.code])
                nav=cash+sum(q*float(close.at[date,c]) for c,q in shares.items() if abs(q)>1e-7)
                max_nav=max(max_nav,abs(nav-drow.equity));max_cash=max(max_cash,abs(cash-drow.cash))
            # Input CSVs are rounded to 10 significant digits; allow absolute accounting tolerance.
            assert max_nav<2 and max_cash<2 and max_price<1e-7 and max_fee<.001 and max_notional<.02, (period,seed,method,max_nav,max_cash,max_price,max_fee,max_notional)
            checks.append({"period":period,"seed":int(seed),"method":method,"daily_rows":len(g),"max_nav_error_currency":max_nav,"max_cash_error_currency":max_cash,"max_fee_error":max_fee,"max_price_error":max_price,"max_notional_error":max_notional,"minimum_cash_roundoff":min_cash,"minimum_shares_roundoff":min_shares,"holding_violations":hold_violation,"sell_rows_without_current_signal":missing_sells,"initial_day_nav":float(g.equity.iloc[0]),"initial_day_fee":float(g.fees_paid.iloc[0])})
            dates=pd.DatetimeIndex(g.date)
            b0=raw.loc[dates,"close"].to_numpy();b1=fixed.loc[dates,"close"].to_numpy()
            original,_,_=investment_metrics(g.equity,b0)
            old=prior[(prior.period==period)&(prior.seed==seed)&(prior.method==method)].iloc[0]
            for met in ("aer","ir","annual_return","total_return_net","mdd"):
                assert abs(original[met]-old[met])<2e-7,(period,seed,method,met,original[met],old[met])
            origin=calendar[calendar.get_loc(dates[0])-1]
            benchmark_start=float(fixed.loc[origin,"close"])
            stages=(("original_reproduced",b0,None,None),("benchmark_column_fix_only",b1,None,None),("benchmark_and_initial_nav_fix",b1,initial_cash,benchmark_start))
            for stage,bs,nav0,bench0 in stages:
                result,r,rb=investment_metrics(g.equity,bs,nav0,bench0)
                rows.append({"period":period,"seed":int(seed),"method":method,"stage":stage,"first_trade":str(dates[0].date()),"last_valuation":str(dates[-1].date()),"initial_nav_date":str(origin.date()) if nav0 else str(dates[0].date()),**result})
                if nav0:
                    for i,day in enumerate(dates):
                        corrected_daily.append({"period":period,"seed":int(seed),"method":method,"date":day,"equity":float(g.equity.iloc[i]),"initial_nav":1e8,"strategy_return":r[i],"benchmark_return":rb[i],"active_return":r[i]-rb[i]})
            # This is fee add-back on the same fixed realized trades, not a no-cost strategy.
            for origin_date,sg in snapshots.groupby("origin"):
                event_rows.append({"period":period,"seed":int(seed),"method":method,"origin":origin_date,"signal_count":len(sg),"min_horizon":sg.horizon_end.min(),"max_horizon":sg.horizon_end.max()})
    save("investment_correction_per_seed.csv",rows);save("investment_ledger_checks.csv",checks)
    save("investment_signal_coverage.csv",event_rows);save("investment_corrected_daily.csv",corrected_daily)
    df=pd.DataFrame(rows)
    summary=df.groupby(["period","method","stage"])[["aer","ir","annual_return","total_return_net","benchmark_total","mdd","valid_days"]].agg(["mean","std"])
    summary.columns=["_".join(c) for c in summary.columns]
    summary.to_csv(OUT/"investment_correction_summary.csv",float_format="%.12g")
    dump("investment_scope.json",{"corrections":["Tencent open/close field labels","initial cash NAV before first entry; include first entry-day return and fee"],"kept_fixed":["all recorded trades","all model signals","all holdings and prices","fees and benchmark dates"],"strategy_redesign":False,"initial_benchmark":"previous close; benchmark close-to-close and strategy starts in cash then enters next open","gross_warning":"old gross_equity=equity+cumulative_fees is fixed-trade fee add-back, not separately simulated no-cost counterfactual","unresolved":["unadjusted stock prices without corporate action accounting","sparse asynchronous cross sections and treating unscored holdings as outside top-k","fractional shares; no limit-up/down/slippage constraints","point-in-time membership unknown"],"main_paper_investment_ready":False})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["freeze","forecast","statistics","investment","verify"])
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.stage=="freeze":freeze()
    elif args.stage=="forecast":inputs_and_forecasts()
    elif args.stage=="statistics":statistics()
    elif args.stage=="investment":investment()
    else:print(verify_protection())


if __name__=="__main__":main()
