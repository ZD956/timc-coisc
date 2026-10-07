"""One fixed selection-only evaluation protocol for every model and seed."""
from __future__ import annotations
import time
import numpy as np
import torch
from scipy.stats import spearmanr
from paper_models import rollout
from paper_data import POINTS
from structural_training_common import _signed, fitted_segment_changes

HORIZONS = (10,30,60,90,120)


@torch.no_grad()
def predict(model, history, future, device, batch=256):
    model.eval()
    values = []
    for start in range(0,len(history),batch):
        h=torch.as_tensor(np.ascontiguousarray(history[start:start+batch]),device=device,dtype=torch.float32)
        u=torch.as_tensor(np.ascontiguousarray(future[start:start+batch]),device=device,dtype=torch.float32)
        values.append(rollout(model,h,u).cpu().numpy())
    result=np.concatenate(values)
    if not np.isfinite(result).all():
        raise FloatingPointError('Non-finite recursive predictions; no clipping permitted')
    return result


def prediction_metrics(prediction,target,origin,source_ids,threshold):
    assert prediction.shape==target.shape and target.shape[1:]==(12,7)
    error=prediction.astype(np.float64)-target
    rows=[]; regimes=[]
    for horizon in HORIZONS:
        k=horizon//10-1
        mae=np.abs(error[:,k]).mean(0)
        rmse=np.sqrt(np.square(error[:,k]).mean(0))
        for i,point in enumerate(POINTS):
            rows.append(dict(horizon_s=horizon,point=point,MAE_C=float(mae[i]),RMSE_C=float(rmse[i]),n=len(error)))
        rows.append(dict(horizon_s=horizon,point='all7',MAE_C=float(mae.mean()),RMSE_C=float(rmse.mean()),pooled_RMSE_C=float(np.sqrt(np.square(error[:,k]).mean())),n=len(error)))
        true_sign=_signed(target[:,k]-origin,threshold)
        pred_sign=_signed(prediction[:,k]-origin,threshold)
        for sign,name in ((1,'heating'),(-1,'cooling'),(0,'near-steady')):
            masks=true_sign==sign
            for i,point in enumerate(POINTS):
                mask=masks[:,i]; n=int(mask.sum())
                regimes.append(dict(horizon_s=horizon,point=point,regime=name,n=n,MAE_C=float(np.abs(error[:,k,i])[mask].mean()) if n else None,direction_agreement=float((pred_sign[:,i][mask]==sign).mean()) if n else None))
            n=int(masks.sum())
            regimes.append(dict(horizon_s=horizon,point='all7',regime=name,n=n,MAE_C=float(np.abs(error[:,k])[masks].mean()) if n else None,direction_agreement=float((pred_sign[masks]==sign).mean()) if n else None))
    # Selection criterion: equal source, output, and prescribed-horizon weight.
    idx=np.asarray(HORIZONS)//10-1
    macro=float(np.mean([np.abs(error[source_ids==s][:,idx]).mean() for s in np.unique(source_ids)]))
    legacy=dict(all7_MAE_C=float(np.abs(error).mean()),point_MAE_C=np.abs(error).mean((0,1)).tolist(),
                fitted_segment_MAE_C=float(np.abs(fitted_segment_changes(prediction)-fitted_segment_changes(target)).mean()))
    return rows,regimes,macro,legacy


@torch.no_grad()
def inference_time(model,history,future,device):
    h=torch.as_tensor(history[:1],device=device);u=torch.as_tensor(future[:1,0],device=device)
    model.eval()
    for _ in range(10): model.forward_one_step(h,u)
    if device.type=='cuda': torch.cuda.synchronize()
    start=time.perf_counter()
    for _ in range(100): model.forward_one_step(h,u)
    if device.type=='cuda': torch.cuda.synchronize()
    return (time.perf_counter()-start)*10  # ms per single-sample one-step call


def onset(curve,band):
    signs=_signed(curve,band)
    for k in range(len(curve)-2):
        if signs[k]!=0 and signs[k]==signs[k+1]==signs[k+2]:
            return (k+1)*10
    return None


def excitation_metrics(prediction,events,threshold):
    detail=[]
    for pred,event in zip(prediction,events):
        truth=event['target'];pre=event['pre_temperature']
        x=np.arange(-5,1,dtype=float)*10
        slope=((pre-pre.mean(0))* (x-x.mean())[:,None]).sum(0)/np.square(x-x.mean()).sum()
        baseline=event['origin']+np.arange(1,len(truth)+1)[:,None]*10*slope
        for j,point in enumerate(POINTS):
            measured=truth[:,j]-baseline[:,j];estimated=pred[:,j]-baseline[:,j]
            true_lag=onset(measured,threshold[j]);pred_lag=onset(estimated,threshold[j])
            amp=float(measured[-3:].mean());pred_amp=float(estimated[-3:].mean())
            detail.append(dict(event_id=event['event_id'],split=event['split'],source_id=event['source_id'],direction=event['direction'],magnitude=event['magnitude'],point=point,
                               delta_UH_W=event['delta_UH_W'],trajectory_MAE_C=float(np.abs(pred[:,j]-truth[:,j]).mean()),
                               true_onset_s=true_lag,pred_onset_s=pred_lag,lag_abs_error_s=abs(pred_lag-true_lag) if true_lag is not None and pred_lag is not None else None,
                               missed_onset=int(true_lag is not None and pred_lag is None),false_onset=int(true_lag is None and pred_lag is not None),
                               true_amplitude_C=amp,pred_amplitude_C=pred_amp,direction_agreement=int(_signed(amp,threshold[j])==_signed(pred_amp,threshold[j]))))
    aggregate=[]
    for split in ('fit','selection'):
        for direction in ('increase','decrease'):
            for magnitude in ('all','small','medium','large'):
                for point in (*POINTS,'all7'):
                    rr=[r for r in detail if r['split']==split and r['direction']==direction and (magnitude=='all' or r['magnitude']==magnitude) and (point=='all7' or r['point']==point)]
                    lag=[r['lag_abs_error_s'] for r in rr if r['lag_abs_error_s'] is not None]
                    rho=None
                    if point!='all7' and len(rr)>=5 and np.std([r['true_amplitude_C'] for r in rr])>0 and np.std([r['pred_amplitude_C'] for r in rr])>0:
                        rho=float(spearmanr([r['true_amplitude_C'] for r in rr],[r['pred_amplitude_C'] for r in rr]).statistic)
                    aggregate.append(dict(split=split,direction=direction,magnitude=magnitude,point=point,n_events=len({r['event_id'] for r in rr}),n_point_events=len(rr),lag_detected_pairs=len(lag),
                                          onset_lag_MAE_s=float(np.mean(lag)) if lag else None,post_action_MAE_C=float(np.mean([r['trajectory_MAE_C'] for r in rr])) if rr else None,
                                          direction_agreement=float(np.mean([r['direction_agreement'] for r in rr])) if rr else None,
                                          missed_onsets=sum(r['missed_onset'] for r in rr),false_onsets=sum(r['false_onset'] for r in rr),spearman_rho=rho,
                                          status='measured_excitation_response' if rr else 'no_eligible_events'))
    return aggregate,detail


def free_run_metrics(prediction,target,horizon):
    error=prediction.astype(np.float64)-target
    mae=np.abs(error).mean((0,1));rmse=np.sqrt(np.square(error).mean((0,1)))
    rows=[dict(horizon_s=horizon,point=p,n_windows=len(target),trajectory_MAE_C=float(mae[i]),trajectory_RMSE_C=float(rmse[i]),endpoint_MAE_C=float(np.abs(error[:,-1,i]).mean())) for i,p in enumerate(POINTS)]
    rows.append(dict(horizon_s=horizon,point='all7',n_windows=len(target),trajectory_MAE_C=float(mae.mean()),trajectory_RMSE_C=float(rmse.mean()),endpoint_MAE_C=float(np.abs(error[:,-1]).mean())))
    return rows
