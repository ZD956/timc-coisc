"""Optional in-memory diagnostics; imports existing helpers, never their loaders."""
from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
from frozen_evaluate import POINTS,HORIZONS,finite

THRESHOLD=np.array([.025106949981231728,.02568001146149004,.023586132975930818,.024577970153300977,.04628328596325579,.020546947946013867,.05129145146380387])
EVENT_CONFIG=dict(min_uh_change_W=5.,pre_dwell_frames=3,dwell_frames=6,hold_tolerance_W=1.,max_aux_change_W=1.,response_frames=12)

def _helpers():
    import paper_data
    import paper_eval
    return paper_data,paper_eval

def regime_rows(prediction,target,origin):
    p=finite(prediction);y=finite(target);origin=finite(origin)
    if p.shape!=y.shape or p.shape[1:]!=(12,7) or origin.shape!=(len(p),7): raise ValueError('regime shape mismatch')
    rows=[]
    for h in HORIZONS:
        k=h//10-1; truth=y[:,k]-origin;pred=p[:,k]-origin
        ts=(truth>THRESHOLD).astype(int)-(truth < -THRESHOLD).astype(int)
        ps=(pred>THRESHOLD).astype(int)-(pred < -THRESHOLD).astype(int)
        for o,point in enumerate(POINTS):
            for sign,name in ((1,'heating'),(-1,'cooling'),(0,'near-steady')):
                mask=ts[:,o]==sign;n=int(mask.sum())
                rows.append(dict(horizon_s=h,output=point,regime=name,n_windows=n,
                                 MAE_C=float(np.abs(p[:,k,o]-y[:,k,o])[mask].mean()) if n else None,
                                 direction_agreement=float((ps[mask,o]==sign).mean()) if n else None))
    return rows

def event_rows(trajectories,role,predict):
    data,evaluation=_helpers();events=data.extract_events(trajectories,role,EVENT_CONFIG)
    detail=[]
    if events:
        history=np.stack([e['history'] for e in events]);future=np.stack([e['future'] for e in events])
        pred=finite(predict(history,future))
        if pred.shape!=(len(events),12,7):raise ValueError('event prediction shape mismatch')
        # Existing details support actual role strings; ignore legacy aggregation.
        _,detail=evaluation.excitation_metrics(pred,events,THRESHOLD)
    summary=[]
    for direction in ('increase','decrease'):
        for magnitude in ('all','small','medium','large'):
            for point in POINTS+('all7',):
                rr=[r for r in detail if r['direction']==direction and (magnitude=='all' or r['magnitude']==magnitude) and (point=='all7' or r['point']==point)]
                sources=sorted({r['source_id'] for r in rr})
                def source_mean(key):
                    means=[]
                    for s in sources:
                        vals=[r[key] for r in rr if r['source_id']==s and r[key] is not None]
                        if vals:means.append(float(np.mean(vals)))
                    return float(np.mean(means)) if means else None
                rho=None
                if point!='all7' and len(rr)>=5 and np.std([r['true_amplitude_C'] for r in rr])>0 and np.std([r['pred_amplitude_C'] for r in rr])>0:
                    from scipy.stats import spearmanr
                    rho=float(spearmanr([r['true_amplitude_C'] for r in rr],[r['pred_amplitude_C'] for r in rr]).statistic)
                summary.append(dict(split=role,direction=direction,magnitude=magnitude,point=point,n_sources=len(sources),n_events=len({r['event_id'] for r in rr}),
                    aggregation='equal_event_containing_source',post_action_MAE_C=source_mean('trajectory_MAE_C'),direction_agreement=source_mean('direction_agreement'),
                    onset_lag_MAE_s=source_mean('lag_abs_error_s'),both_detected=sum(r['lag_abs_error_s'] is not None for r in rr),
                    missed_onsets=sum(r['missed_onset'] for r in rr),false_onsets=sum(r['false_onset'] for r in rr),spearman_rho=rho,
                    status='measured_excitation_response' if rr else 'no_eligible_events'))
    return dict(details=detail,summary=summary)

def free_run_rows(trajectories,source_ids,predict):
    data,_=_helpers();rows=[]
    for horizon in (300,600):
        records=data.free_run_windows(trajectories,horizon//10)
        for source in source_ids:
            rr=[r for r in records if r['source_id']==source]
            if not rr:
                rows.extend(dict(source_id=source,horizon_s=horizon,output=p,n_rollouts=0,status='no_eligible_continuous_segment',trajectory_MAE_C=None,trajectory_RMSE_C=None,trajectory_bias_C=None,endpoint_MAE_C=None,endpoint_RMSE_C=None,endpoint_bias_C=None) for p in POINTS+('all7',))
                continue
            h=np.stack([r['history'] for r in rr]);u=np.stack([r['future'] for r in rr]);y=finite(np.stack([r['target'] for r in rr]))
            p=finite(predict(h,u))
            if p.shape!=y.shape:raise ValueError('free-run paired shape mismatch')
            error=p-y
            metrics=dict(trajectory_MAE_C=np.abs(error).mean((0,1)),trajectory_RMSE_C=np.sqrt((error**2).mean((0,1))),trajectory_bias_C=error.mean((0,1)),
                         endpoint_MAE_C=np.abs(error[:,-1]).mean(0),endpoint_RMSE_C=np.sqrt((error[:,-1]**2).mean(0)),endpoint_bias_C=error[:,-1].mean(0))
            for o,point in enumerate(POINTS+('all7',)):
                rows.append(dict(source_id=source,horizon_s=horizon,output=point,n_rollouts=len(rr),n_segments=len({r['segment_id'] for r in rr}),n_frames=len(rr)*(horizon//10),status='OK',
                                 **{k:float(v[o]) if o<7 else float(v.mean()) for k,v in metrics.items()}))
        for point in POINTS+('all7',):
            eligible=[r for r in rows if r['source_id']!='ALL' and r['horizon_s']==horizon and r['output']==point and r['status']=='OK']
            fields=('trajectory_MAE_C','trajectory_RMSE_C','trajectory_bias_C','endpoint_MAE_C','endpoint_RMSE_C','endpoint_bias_C')
            rows.append(dict(source_id='ALL',aggregation='equal_eligible_source',horizon_s=horizon,output=point,n_sources=len(eligible),n_rollouts=sum(r['n_rollouts'] for r in eligible),
                             status='OK' if eligible else 'no_eligible_continuous_segment',**{k:float(np.mean([r[k] for r in eligible])) if eligible else None for k in fields}))
    return rows

def diagnostics_for_pairs(pairs,membership,predictors=None):
    """Always compute regimes; optional continuous/event predictions via callback.

    predictors[seed][variant](history,future) uses already supplied models;
    there is no checkpoint loader. Absent callback produces explicit NA rows.
    """
    from frozen_evaluate import SEEDS,validate_pairs
    from types import SimpleNamespace
    order=validate_pairs(pairs,membership);roles={r['source_id']:r['role'] for r in membership}
    output=dict(regime=[],free_run=[],event_details=[],event_summary=[],failure_source=[])
    for seed in SEEDS:
        trajectories={role:[] for role in ('frozen_external','frozen_final')}
        if predictors is not None:
            data,_=_helpers()
            for source in order:
                r=pairs[seed][source]['R0'];t=data.reconstruct(SimpleNamespace(**{k:r[k] for k in ('history','future','target','origin')}),np.array([source]*len(r['target'])))
                for segment in t:
                    segment['segment_id']=len(trajectories[roles[source]])
                    trajectories[roles[source]].append(segment)
        for variant in ('R0','COISC'):
            for source in order:
                r=pairs[seed][source][variant]
                output['regime'] += [dict(seed=seed,variant=variant,source_id=source,cohort=roles[source],**x) for x in regime_rows(r['prediction'],r['target'],r['origin'])]
            for role,tt in trajectories.items():
                ids=[s for s in order if roles[s]==role]
                if predictors is None:
                    output['free_run'] += [dict(seed=seed,variant=variant,cohort=role,source_id=s,horizon_s=h,output=p,status='not_requested',trajectory_MAE_C=None) for s in ids for h in (300,600) for p in POINTS]
                    output['event_summary'].append(dict(seed=seed,variant=variant,split=role,status='not_requested',n_events=None))
                    continue
                callback=predictors[seed][variant]
                output['free_run'] += [dict(seed=seed,variant=variant,cohort=role,**x) for x in free_run_rows(tt,ids,callback)]
                events=event_rows(tt,role,callback)
                output['event_details'] += [dict(seed=seed,variant=variant,**x) for x in events['details']]
                output['event_summary'] += [dict(seed=seed,variant=variant,**x) for x in events['summary']]
    # Source-equal regime aggregates with explicit eligible-source denominators.
    for seed in SEEDS:
        for variant in ('R0','COISC'):
            for role in ('all29','frozen_external','frozen_final'):
                for h in HORIZONS:
                    for point in POINTS:
                        for regime in ('heating','cooling','near-steady'):
                            rr=[r for r in output['regime'] if r['source_id']!='ALL' and r['seed']==seed and r['variant']==variant and (role=='all29' or r['cohort']==role) and r['horizon_s']==h and r['output']==point and r['regime']==regime and r['n_windows']>0]
                            output['regime'].append(dict(seed=seed,variant=variant,cohort=role,source_id='ALL',horizon_s=h,output=point,regime=regime,n_sources=len(rr),n_windows=sum(r['n_windows'] for r in rr),
                                MAE_C=float(np.mean([r['MAE_C'] for r in rr])) if rr else None,direction_agreement=float(np.mean([r['direction_agreement'] for r in rr])) if rr else None))
    token='S068'
    for seed in SEEDS:
        matches=[s for s in order if token in s]
        if not matches:output['failure_source'].append(dict(seed=seed,output='T5',status='not_present'))
        for source in matches:
            for h in HORIZONS:
                k=h//10-1;a=pairs[seed][source]['R0'];b=pairs[seed][source]['COISC']
                e0=a['prediction'][:,k,4]-a['target'][:,k,4];e1=b['prediction'][:,k,4]-b['target'][:,k,4]
                output['failure_source'].append(dict(seed=seed,source_id=source,horizon_s=h,output='T5',status='OK',R0_MAE_C=float(np.abs(e0).mean()),COISC_MAE_C=float(np.abs(e1).mean()),delta_MAE_C=float((np.abs(e1)-np.abs(e0)).mean()),R0_bias_C=float(e0.mean()),COISC_bias_C=float(e1.mean())))
    return output

def model_predictor(model,device):
    """Reuse the native model/rollout, no checkpoint I/O or architecture changes."""
    _,evaluation=_helpers()
    model.eval();model.requires_grad_(False)
    return lambda history,future:evaluation.predict(model,history,future,device,batch=256)

def paired_inference(models,history,future,device):
    import torch
    _helpers()
    import screen
    before={variant:{k:v.detach().cpu().clone() for k,v in model.state_dict().items()} for variant,model in models.items()}
    if set(models)!={'R0','COISC'}:raise ValueError('expected paired models')
    if set(before['R0'])!=set(before['COISC']):raise ValueError('state key mismatch')
    if any(not torch.equal(v,before['COISC'][k]) for k,v in before['R0'].items() if not k.startswith('head.')):raise ValueError('non-head mismatch')
    result={}
    for variant,model in models.items():
        predict=model_predictor(model,device);s=[]
        with torch.no_grad():
            prediction=predict(history,future)
            for start in range(0,len(history),256):
                h=torch.as_tensor(history[start:start+256],device=device,dtype=torch.float32);u=torch.as_tensor(future[start:start+256],device=device,dtype=torch.float32)
                s.append(screen.sensitivity(model,h,u).cpu().numpy())
        result[variant]=dict(prediction=prediction,sensitivity=finite(np.concatenate(s)))
        if any(not torch.equal(v.detach().cpu(),before[variant][k]) for k,v in model.state_dict().items()):raise ValueError('inference mutated tensor')
    return result

def inference_timing(models,history,future,device):
    import torch
    _helpers()
    from paper_models import rollout
    import screen
    if set(models)!={'R0','COISC'}:raise ValueError('paired timing models required')
    results=[]
    for kind in ('one_step','rollout_120s','batch256_rollout','sensitivity_120s'):
        for variant in ('R0','COISC'):
            model=models[variant].eval();model.requires_grad_(False)
            n=min(256,len(history)) if kind=='batch256_rollout' else 1
            h=torch.as_tensor(history[:n],device=device,dtype=torch.float32);u=torch.as_tensor(future[:n],device=device,dtype=torch.float32)
            call=(lambda:model.forward_one_step(h,u[:,0])) if kind=='one_step' else (lambda:screen.sensitivity(model,h,u)) if kind=='sensitivity_120s' else (lambda:rollout(model,h,u))
            with torch.no_grad():
                for _ in range(10):call()
                if device.type=='cuda':torch.cuda.synchronize()
                start=time.perf_counter()
                for _ in range(100):call()
                if device.type=='cuda':torch.cuda.synchronize()
                elapsed=time.perf_counter()-start
            results.append(dict(variant=variant,kind=kind,batch=n,warmup_calls=10,timed_calls=100,ms_per_call=elapsed*10,windows_per_second=n*100/elapsed,device=str(device),dtype='float32',transfers_included=False))
    return results
