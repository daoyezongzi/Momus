"""Post-result diagnostics; these exclusions never replace the frozen main table."""
import json

import numpy as np
import pandas as pd

from audit import ROOT,OLD,OUT,PERIODS,METRICS,digest,dump,save


def main():
    prior=json.loads((OLD/'protocol.json').read_text())
    identities=[]
    for seed,obj in prior['checkpoints'].items():
        p=ROOT/obj['path'];actual=digest(p)
        identities.append(dict(kind='checkpoint',name=seed,expected=obj['sha256'],actual=actual,match=actual==obj['sha256']))
    protocol=ROOT/prior['frozen_protocol_path']
    identities.append(dict(kind='protocol',name=protocol.name,expected=prior['frozen_protocol_sha256'],actual=digest(protocol),match=digest(protocol)==prior['frozen_protocol_sha256']))
    for p in sorted((ROOT/'data/references').glob('hs300_*.txt')):
        codes=p.read_text().split()
        identities.append(dict(kind='membership',name=p.name,count=len(codes),unique=len(set(codes)),actual=digest(p),match=True))
    assert all(x['match'] for x in identities)
    save('identity_checks.csv',identities)

    oldbench=pd.read_csv(OLD/'benchmark_sh000300.csv')
    raw=json.loads((OUT/'sources/benchmark_raw.json').read_text())['data']['sh000300']['day']
    parsed=pd.DataFrame(raw,columns=['date','open','close','high','low','volume'])
    merged=oldbench.merge(parsed,on='date',suffixes=('_old','_raw'))
    benchmark_check=dict(matched_rows=len(merged),original_rows=len(oldbench),old_close_equals_raw_open=bool(np.allclose(merged.close_old,merged.open_raw.astype(float),atol=1e-8)),old_open_equals_raw_close=bool(np.allclose(merged.open_old,merged.close_raw.astype(float),atol=1e-8)))
    assert benchmark_check['matched_rows']==benchmark_check['original_rows'] and benchmark_check['old_close_equals_raw_open'] and benchmark_check['old_open_equals_raw_close']
    dump('benchmark_field_evidence.json',benchmark_check)

    flags=pd.read_csv(OUT/'price_jump_flags.csv',dtype={'period':str,'code':str})
    win=pd.read_csv(OUT/'windows.csv',dtype={'period':str,'code':str})
    f=pd.read_csv(OLD/'forecast_window_metrics.csv',dtype={'period':str,'code':str})
    windows=[]
    for r in win.itertuples():
        events=flags[(flags.period==r.period)&(flags.code==r.code)]
        in_future=((events.date>=r.entry)&(events.date<=r.horizon_end)).any()
        in_input=((events.date>=r.history_start)&(events.date<=r.horizon_end)).any()
        windows.append(dict(period=r.period,window_index=r.window_index,future_large_gap=bool(in_future),history_or_future_large_gap=bool(in_input)))
    labels=pd.DataFrame(windows)
    save('forecast_gap_window_flags.csv',labels)
    results=[]
    for period in PERIODS:
        pp=f[f.period==period]
        info=labels[labels.period==period].sort_values('window_index')
        base=pp[pp.method=='simple_mean'].groupby('window_index')[list(METRICS)].mean().sort_index()
        for method in ('momus_weighted','momus_top1'):
            a=pp[pp.method==method].groupby('window_index')[list(METRICS)].mean().sort_index()
            for label,mask in [('all',np.ones(len(info),bool)),('exclude_future_abs_overnight_gap_gt_20pct',~info.future_large_gap.to_numpy()),('exclude_history_or_future_gap_gt_20pct',~info.history_or_future_large_gap.to_numpy())]:
                am,bm=a.loc[mask].mean(),base.loc[mask].mean()
                results.append(dict(period=period,method=method,slice=label,windows=int(mask.sum()),excluded=int((~mask).sum()),mse_improvement_pct=(1-am.ohlc_mse/bm.ohlc_mse)*100,delta_ic=am.price_ic-bm.price_ic,delta_rankic=am.price_rankic-bm.price_rankic,interpretation='post-hoc diagnostic; deletes genuine market jumps too; not a corporate-action correction'))
    save('large_gap_sensitivity.csv',results)

    trades=pd.read_csv(OLD/'investment_trades.csv',dtype={'period':str,'code':str})
    exposure=[]
    for r in flags.itertuples():
        t=trades[(trades.period==r.period)&(trades.code==r.code)&(trades.date<r.date)].copy()
        if t.empty:continue
        t['signed_qty']=t.quantity.where(t.side=='buy',-t.quantity)
        for (seed,method),qty in t.groupby(['seed','method']).signed_qty.sum().items():
            if qty<.01:continue
            exposure.append(dict(period=r.period,date=r.date,code=r.code,seed=int(seed),method=method,prior_close_quantity=qty,overnight_return=r.overnight_return,status='held across large raw-price gap; source confirmation required'))
    save('portfolio_large_gap_exposure.csv',exposure)
    # Bounded, source-confirmed omission example. Do not pretend correcting one
    # event reconstructs the full corporate-action-adjusted backtest.
    case=pd.DataFrame(exposure)
    case=case[(case.period=='2024')&(case.code=='002837')&(case.date=='2024-05-27')].copy()
    assert len(case)==10
    case['missing_bonus_shares']=case.prior_close_quantity*.3
    case['missing_gross_dividend_cash']=case.prior_close_quantity*.2
    case['missing_entitlement_value_at_close']=case.missing_bonus_shares*24.84+case.missing_gross_dividend_cash
    case['scope']='single-event gross entitlement diagnostic; not full-period repaired strategy'
    save('confirmed_corporate_action_omission.csv',case)
    event=pd.read_csv(OUT/'investment_signal_coverage.csv')
    event=event[(event.seed==0)&(event.method=='simple_mean')]
    summary=event.groupby('period').agg(events=('signal_count','size'),min_signals=('signal_count','min'),median_signals=('signal_count','median'),max_signals=('signal_count','max'),events_below_50=('signal_count',lambda x:int((x<50).sum())),events_at_least_300=('signal_count',lambda x:int((x>=300).sum())))
    summary.to_csv(OUT/'sparse_signal_summary.csv')
    print('diagnostics complete',flush=True)


if __name__=='__main__':main()
