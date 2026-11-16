"""Full independent PyTorch replay of fixed HS300 predictions and scaling.

Uses the original model's modules. Historical encodings are computed once per
window, rather than repeated for every candidate; direct forward is checked.
"""
import json
import pickle

import numpy as np
import pandas as pd
import torch

from audit import ROOT,OLD,OUT,PERIODS,METRICS,METHODS,metrics,dump,save
from critic.model import DualHeadCritic


@torch.no_grad()
def scores_batched(model,paths,hist):
    batch,n,h,c=paths.shape
    p=torch.from_numpy(paths.reshape(batch*n,h,c))
    x=torch.from_numpy(hist)
    hp=model.seq_encoder(p)
    hx=model.seq_encoder(x).repeat_interleave(n,dim=0)
    hc=model.corr_missing[None,:].expand(batch*n,-1)
    feat=torch.cat([hp,hx,hc,hp*hc],dim=-1)
    scores=model.final_head(torch.cat([model.base_head(feat),model.fine_head(feat),model.coarse_head(feat)],dim=-1))
    return scores.reshape(batch,n).numpy()


def main():
    torch.set_num_threads(2)
    original=pd.read_csv(OLD/'forecast_window_metrics.csv',dtype={'period':str,'code':str})
    original_signals=pd.read_csv(OLD/'investment_signals.csv',dtype={'period':str,'code':str})
    original_scaling=pd.read_csv(OLD/'scaling_per_seed.csv',dtype={'period':str})
    check_rows,scaling_rows,ranks=[],[],[]
    for period in PERIODS:
        c=pickle.load((ROOT/f'output/candidates/hs300_{period}_N16.pkl').open('rb'))
        raw=np.stack([w[1].to_numpy(np.float32) for w in c['windows']])
        truth_raw=np.stack([w[4] for w in c['windows']]).astype(np.float32)
        mean=raw.mean(1,keepdims=True);scale=raw.std(1,keepdims=True)+1e-5
        hist=np.clip((raw-mean)/scale,-5,5)
        truth=(truth_raw-mean)/scale
        paths=np.stack(c['candidates'])
        expected_candidate_mse=np.mean((paths[:,:,:,:4].astype(float)-truth[:,None,:,:4])**2,axis=(2,3))
        for seed in range(5):
            model=DualHeadCritic(final_combiner='calibrated')
            model.load_state_dict(torch.load(ROOT/f'output/results/regime_generalization/multiregime_train/seed_{seed}/fusion_best.pt',map_location='cpu',weights_only=True));model.eval()
            score=np.concatenate([scores_batched(model,paths[i:i+128],hist[i:i+128]) for i in range(0,len(paths),128)])
            # Direct original forward, at fixed first/middle/last windows.
            max_direct=0.
            for ix in (0,len(paths)//2,len(paths)-1):
                direct=model.score(paths[ix],hist[ix])
                max_direct=max(max_direct,float(np.max(abs(score[ix]-direct))))
            assert max_direct<5e-5
            for n in (1,4,8,16):
                sub=paths[:,:n]
                s=score[:,:n].astype(float);w=np.exp(s-s.max(1,keepdims=True));w/=w.sum(1,keepdims=True)
                preds={'simple_mean':sub.mean(1),'momus_weighted':np.einsum('bn,bnhf->bhf',w,sub)}
                if n==16:preds.update(median=np.median(sub,axis=1),momus_top1=sub[np.arange(len(sub)),np.argmax(s,axis=1)])
                if n==1:np.testing.assert_array_equal(preds['simple_mean'],preds['momus_weighted'])
                for method,pred in preds.items():
                    result=metrics(pred,truth)
                    if method in ('simple_mean','momus_weighted'):
                        oldscale=original_scaling[(original_scaling.period==period)&(original_scaling.seed==seed)&(original_scaling.candidate_count==n)&(original_scaling.method==method)].iloc[0]
                        for met in METRICS:
                            err=float(abs(result[met].mean()-oldscale[met]));assert err<5e-5,(period,seed,n,method,met,err)
                            scaling_rows.append(dict(period=period,seed=seed,n=n,method=method,metric=met,recomputed=float(result[met].mean()),original=float(oldscale[met]),abs_error=err))
                    if n!=16:continue
                    old=original[(original.period==period)&(original.seed==seed)&(original.method==method)].sort_values('window_index')
                    for met in METRICS:
                        err=float(np.max(abs(result[met]-old[met].to_numpy())))
                        assert err<2e-4,(period,seed,method,met,err)
                        check_rows.append(dict(period=period,seed=seed,method=method,metric=met,windows=len(paths),max_abs_error=err,mean_abs_error=float(np.mean(abs(result[met]-old[met].to_numpy()))),direct_forward_score_error=max_direct))
                    if method in ('simple_mean','momus_weighted'):
                        signals=(pred.astype(float)*scale.astype(float)+mean.astype(float))[:,:,3].mean(1)/raw[:,-1,3].astype(float)-1
                        oldsig=original_signals[(original_signals.period==period)&(original_signals.seed==seed)&(original_signals.method==method)].sort_values('window_index')
                        # Stored last_close was pandas float64; use same actual price denominator.
                        actual_close=np.array([float(x[1]['close'].iloc[-1]) for x in c['windows']])
                        signals=(pred.astype(float)*scale.astype(float)+mean.astype(float))[:,:,3].mean(1)/actual_close-1
                        err=float(np.max(abs(signals-oldsig.signal.to_numpy())))
                        assert err<2e-5,(period,seed,method,'signals',err)
                        check_rows.append(dict(period=period,seed=seed,method=method,metric='investment_signal',windows=len(paths),max_abs_error=err,direct_forward_score_error=max_direct))
            # Within-candidate score vs negative oracle error; descriptive mechanism,
            # computed on the same fixed inputs, not a capacity/semantic ablation.
            sr=pd.DataFrame(score).rank(axis=1).to_numpy()
            er=pd.DataFrame(-expected_candidate_mse).rank(axis=1).to_numpy()
            sr-=sr.mean(1,keepdims=True);er-=er.mean(1,keepdims=True)
            corr=(sr*er).sum(1)/np.sqrt((sr*sr).sum(1)*(er*er).sum(1))
            selected=np.argmax(score,axis=1);oracle=np.argmin(expected_candidate_mse,axis=1)
            ranks.append(dict(period=period,seed=seed,windows=len(paths),candidate_rank_spearman=float(np.mean(corr)),top1_oracle_hit=float(np.mean(selected==oracle)),selected_candidate_mse=float(np.mean(expected_candidate_mse[np.arange(len(paths)),selected])),random_candidate_expected_mse=float(expected_candidate_mse.mean())))
            print('full PyTorch replay',period,seed,'passed',flush=True)
        del c,paths
    save('full_forecast_replay_checks.csv',check_rows);save('full_scaling_replay.csv',scaling_rows);save('candidate_ranking_per_seed.csv',ranks)
    dump('full_replay_status.json',{'passed':True,'periods':PERIODS,'seeds':5,'all_methods':METHODS,'all_candidate_counts':[1,4,8,16],'all_investment_signals_replayed':True,'future_labels_only_used_after_scoring':True,'no_corr_inference':True,'new_candidate_generation':False,'model_training':False,'forecast_window_metric_checks':sum(r['windows'] for r in check_rows if r['metric'] in METRICS),'max_metric_abs_error':max(r['max_abs_error'] for r in check_rows if r['metric'] in METRICS)})


if __name__=='__main__':main()
