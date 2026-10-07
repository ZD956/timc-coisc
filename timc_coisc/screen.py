"""Isolated, real-field-only single-seed DBASR development screening.

Imports existing B4 architecture, recursive rollout, losses and evaluation.
Never invokes the historical mixed-data loader or accesses frozen sources.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
import numpy as np
import torch

from _paths import PKG as HERE, PROTOCOLS, CONFIGS, OUTPUT_ROOT, require_private, public_ids
from paper_data import V12B, POINTS, sha, save_json, write_csv, reconstruct, extract_events
from paper_models import build_model, rollout
from paper_eval import predict, prediction_metrics, excitation_metrics
from model_structural_ablation import MIN_LAG_SECONDS
from structural_training_common import (ArraySource, EvaluationSet, compute_model_stats,
    weighted_huber, adjacent_delta_huber, endpoint_temperature_huber)

SEED = 20260922
EPS_PRE = .001  # degrees C / unit full-scale UH fraction; not output-z-scored
EPS_SIGN = .0001
SUBSET = 8
PILOT_STEPS = 32
DATA_SHA = '6c56bf5904c40a8567b9bb3c1202efcd2c42d33365801ccdd9a515a0b74914ca'
VERIFY_HASHES = os.environ.get('TIMC_VERIFY_HASHES') == '1'


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def perturb(future):
    plus, minus = future.clone(), future.clone()
    plus[:,0,1] = (future[:,0,1]+25.).clamp(0.,500.)
    minus[:,0,1] = (future[:,0,1]-25.).clamp(0.,500.)
    denominator = (plus[:,0,1]-minus[:,0,1])/500.
    if torch.any(denominator <= 0):
        raise ValueError('Invalid clipped UH perturbation denominator')
    return plus, minus, denominator


def lag_masks(lags=MIN_LAG_SECONDS):
    t = np.arange(1,13)[:,None]*10
    tau = np.asarray(lags)[None,:]
    return t < np.maximum(0,tau-10), t > tau+10


def sensitivity(model, history, future):
    plus, minus, denominator = perturb(future)
    # Native B4 feedback is retained, including perturbed current action history.
    return (rollout(model,history,plus)-rollout(model,history,minus))/denominator[:,None,None]


def regularizer(values):
    pre, post = [torch.as_tensor(v,device=values.device) for v in lag_masks()]
    return (torch.relu(values[:,pre].abs()-EPS_PRE).square().mean(),
            torch.relu(-values[:,post]).square().mean())


def excitation_score(history, future):
    uh = np.concatenate([history[:,:,7],future[:,:,1]],axis=1)
    return np.max(np.abs(np.diff(uh,axis=1)),axis=1)


def excitation_probabilities(base, score):
    # Fixed 5 W change scale, fixed capped score boost; no selection calibration.
    enhanced = base * (1.+np.minimum(score/25.,4.))
    enhanced /= enhanced.sum()
    return .5*base + .5*enhanced


def sensitivity_metrics(values):
    pre,post=lag_masks(); rows=[]
    for i,point in enumerate((*POINTS,'all7')):
        a,b=pre.copy(),post.copy()
        if point!='all7':
            a[:,np.arange(7)!=i]=False;b[:,np.arange(7)!=i]=False
        x,y=values[:,a],values[:,b]
        rows.append(dict(point=point,PLE=float(np.abs(x).mean()) if x.size else None,
                         WSR=float((y < -EPS_SIGN).mean()) if y.size else None,
                         PSM=float(np.median(np.abs(y))) if y.size else None,
                         pre_cells=int(x.size),post_cells=int(y.size)))
    return rows


def calibrated_lambda(records):
    p=np.asarray([r['prediction'] for r in records]);q=np.asarray([r['regularizer'] for r in records])
    if not np.isfinite(p).all() or not np.isfinite(q).all() or q.mean()<=1e-12:
        raise ValueError('No usable initial sensitivity scale; do not invent a lambda')
    positive=q>1e-12
    return float(min(.10*p.mean()/q.mean(), .20*np.min(p[positive]/q[positive])))


def load_real():
    old=json.loads((PROTOCOLS/'PAPER_PROTOCOL_DRAFT.json').read_text(encoding='utf-8'))
    fit=set(old['data_source_split']['fit_sources']);sel=set(old['data_source_split']['selection_sources'])
    assert len(fit)==91 and len(sel)==18 and not fit&sel
    path=require_private(V12B/'data'/'field_windows_development_109.npz','screen.load_real')
    if VERIFY_HASHES: assert sha(path)==DATA_SHA, 'Real development cache hash changed'
    with np.load(path,allow_pickle=False) as a:
        train_ids=public_ids(a['train_source_id']);sel_ids=public_ids(a['validation_source_id'])
        assert set(train_ids)==fit and set(sel_ids)==sel
        train=ArraySource('field',*[a['train_'+k].copy() for k in ('history','future','target','origin')],source_labels=train_ids)
        ev=EvaluationSet('field',*[a['validation_'+k].copy() for k in ('history','future','target','origin')],source_labels=sel_ids)
    stats=compute_model_stats({'field':train})
    previous=json.loads((CONFIGS/'normalization.json').read_text(encoding='utf-8'))
    for k in stats: np.testing.assert_allclose(stats[k],previous[k],rtol=0,atol=0)
    window_hashes=[]
    for obj in (train,ev):
        window_hashes.append({hashlib.sha256(h.tobytes()+u.tobytes()+y.tobytes()).digest()
                             for h,u,y in zip(obj.history,obj.future,obj.target)})
        for a in (obj.history,obj.future,obj.target,obj.origin): assert np.isfinite(a).all()
        assert obj.future[:,:,1].min()>=0 and obj.future[:,:,1].max()<=500
    assert not window_hashes[0]&window_hashes[1], 'Cross-split duplicate windows'
    base=np.zeros(len(train.history),np.float64)
    for g,p in zip(train._groups,train._group_probabilities): base[g]=p/len(g)
    scores=excitation_score(train.history,train.future)
    probabilities={False:base,True:excitation_probabilities(base,scores)}
    return train,ev,train_ids,sel_ids,stats,probabilities,scores,old


def sample(train,rng,size,probability,device):
    ids=rng.choice(len(train.history),size=size,replace=True,p=probability)
    return [torch.as_tensor(a[ids],device=device) for a in (train.history,train.future,train.target)]


def factual(model,h,u,y,stage):
    one=weighted_huber(model.forward_one_step(h,u[:,0]),y[:,0])
    loss=one
    if stage['rollout']:
        pred=rollout(model,h,u)
        loss=loss+stage['rollout']*weighted_huber(pred,y)
        # Existing B4 recursive loss components, not new DBASR losses.
        loss=loss+stage['adjacent']*adjacent_delta_huber(pred,y)
        loss=loss+stage['endpoint']*endpoint_temperature_huber(pred,y)
    return loss


def calibrate(train,stats,probs,stages,device,folder):
    start=time.perf_counter();groups=[];all_records=[]
    # Both sampling settings get their own short factual pilot from the same seed.
    # No selection calls; these pilot models are discarded, not checkpoints.
    for excitation in (False,True):
        seed_all(SEED);rng=np.random.default_rng(SEED)
        model=build_model('B4',stats).to(device).train()
        opt=torch.optim.Adam(model.parameters(),lr=stages[0]['lr'])
        for _ in range(PILOT_STEPS):
            h,u,y=sample(train,rng,stages[0]['batch'],probs[excitation],device)
            loss=factual(model,h,u,y,stages[0]);opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        records=[]
        for k in range(8):
            h,u,y=sample(train,rng,stages[0]['batch'],probs[excitation],device)
            prediction=factual(model,h,u,y,stages[0])
            pre,sign=regularizer(sensitivity(model,h[:SUBSET],u[:SUBSET]));reg=pre+sign
            grads=torch.autograd.grad(reg,tuple(model.parameters()),allow_unused=True)
            gradnorm=float(torch.sqrt(sum(g.square().sum() for g in grads if g is not None)))
            if not gradnorm>0: raise ValueError('DBASR has no effective gradient')
            records.append(dict(excitation=excitation,batch=k,prediction=float(prediction.detach()),
                                regularizer=float(reg.detach()),pre=float(pre.detach()),sign=float(sign.detach()),gradient_norm=gradnorm))
        groups.append(calibrated_lambda(records));all_records.extend(records)
    lam=min(groups)
    for r in all_records: r['initial_contribution_ratio']=lam*r['regularizer']/r['prediction']
    result=dict(lambda_s=lam,policy='shared conservative minimum of two fit-only pilot scales; mean <=10%, every probe <=20%',
                pilot_steps_per_sampler=PILOT_STEPS,probe_batches_per_sampler=8,
                records=all_records,time_s=time.perf_counter()-start)
    save_json(folder/'calibration.json',result)
    print('CALIBRATED',lam,'seconds',result['time_s'],flush=True)
    return result


def configuration(train,ev,fit_ids,sel_ids,stats,probs,scores,old):
    stages=[{k:s[k] for k in ('steps','batch','lr','rollout','adjacent','endpoint')} for s in old['training']['stages']]
    pre,post=lag_masks()
    return dict(status='PRECOMMITTED_DEVELOPMENT_SCREEN_NO_FROZEN',seed=SEED,
        variants=dict(R0={'excitation':False,'dbasr':False},R1={'excitation':True,'dbasr':False},
                      R2={'excitation':False,'dbasr':True},R3={'excitation':True,'dbasr':True}),
        data=dict(only_opened_dataset='field_windows_development_109.npz',sha256=DATA_SHA,
                  fit_sources=sorted(set(fit_ids)),selection_sources=sorted(set(sel_ids)),
                  fit_windows=len(train.history),selection_windows=len(ev.history),duplicates_across_splits=0,
                  synthetic_data=False,frozen_access=False),
        R0_retrained_reason='Historical B4 used paired/structured synthetic supervision and cooling sampling; not corresponding real-only checkpoint',
        architecture='unchanged paper_models.VanillaLSTM, 90887 parameters',normalization=stats,
        sampling=dict(base='existing sqrt(source count) source probability / source count',
                      score='max abs adjacent UH change in concatenated 20 history + 12 future controls (including boundary)',
                      probability='.5*p_base + .5*normalize(p_base*(1+min(score_raw/25,4)))',
                      fitted_on='fit field controls only',cooling=False,
                      ordinary_probability_excitation_ge_5W=float(probs[False][scores>=25].sum()),
                      enhanced_probability_excitation_ge_5W=float(probs[True][scores>=25].sum())),
        dbasr=dict(lag_seconds=list(MIN_LAG_SECONDS),band_half_width_s=10,
                   pre_mask=pre.tolist(),post_mask=post.tolist(),epsilon_pre=EPS_PRE,epsilon_sign=EPS_SIGN,
                   sensitivity_units='degrees C per unit UH full-scale fraction; no output normalization',
                   UH_raw_range=[0,500],UH_W_range=[0,100],delta_raw=25,delta_W=5,
                   perturbation='ONLY future[:,0,1]; every later action and other heater unchanged; actual clipped fraction denominator',
                   batch_subset='first 8 windows of factual batch; identical for R2/R3',
                   activation='after first 32 factual warmup steps; then each batch',
                   lambda_calibration='two discarded fit-only 32-step factual pilots; 8 loss probes each; common lambda=min(0.10*mean(prediction)/mean(reg),0.20*min(batch ratios)) across samplers; fixed thereafter',
                   losses='mean relu(abs(S_pre)-.001)^2 + mean relu(-S_post)^2; transition unconstrained'),
        training=dict(stages=stages,optimizer='Adam reset each stage',precision='FP32 deterministic TF32 disabled',
                      gradient_clip=1.,factual='unchanged B4 one-step weighted Huber plus original recursive weighted Huber/adjacent/endpoint terms',
                      checkpoint='minimum selection source-macro MAE over 7 outputs and 10/30/60/90/120s; evaluate every 100 steps; stage1 restores best stage0; no sensitivity/event criterion'),
        evaluation=dict(horizons_s=[10,30,60,90,120],prediction_aggregation='all-window MAE; RMSE=mean of 7 point RMSE, matching existing paper_eval',
                        sensitivity_aggregation='all selection windows and eligible output/horizon cells pooled; PSM median; empty point masks NA',
                        events='existing automatic extractor, selection only, after checkpoint fixed, sanity only',free_run=False),
        screening=dict(MAE_120s_max_C=.169,PLE_relative_reduction_min=.30,WSR_relative_reduction_min=.30,
                       PSM_ratio_min=.20,PSM_absolute_min=5*EPS_SIGN,
                       zero_reference='if R0 PLE or WSR is zero, required relative reduction not demonstrable: fail',
                       no_rescue=True),
        dependency_hashes={p.name:sha(p) for p in [Path(__file__),HERE/'paper_models.py',HERE/'paper_eval.py',HERE/'paper_data.py',HERE/'structural_training_common.py',HERE/'model_structural_ablation.py']})


def score_model(model,ev,ids,indices,threshold,device):
    p=predict(model,ev.history[indices],ev.future[indices],device)
    return prediction_metrics(p,ev.target[indices],ev.origin[indices],ids[indices],threshold)[2]


def train_variant(name,flags,train,ev,ids,stats,probs,cfg,lam,threshold,device,folder,smoke):
    run=folder/'results'/name
    if run.exists(): raise FileExistsError(f'Preserve existing attempt: {run}')
    run.mkdir(parents=True)
    seed_all(SEED);rng=np.random.default_rng(SEED)
    model=build_model('B4',stats).to(device)
    save_json(run/'config.json',dict(variant=name,**flags,seed=SEED,lambda_s=lam if flags['dbasr'] else 0.,protocol_sha=sha(folder/'PROTOCOL.json')))
    stages=[dict(s) for s in cfg['training']['stages']]
    indices=np.arange(len(ids))
    if smoke:
        for s in stages: s['steps']=2;s['batch']=8
        indices=np.concatenate([np.flatnonzero(ids==i)[:2] for i in np.unique(ids)])
    best=float('inf');state=None;warmstate=None;total=0;selected=None;records=[]
    train_start=time.perf_counter();compute_s=0.;eval_s=0.
    with (run/'training.jsonl').open('w',encoding='utf-8') as log:
        for si,stage in enumerate(stages):
            if si==1: model.load_state_dict(warmstate)
            opt=torch.optim.Adam(model.parameters(),lr=stage['lr']);warmbest=float('inf')
            for step in range(1,stage['steps']+1):
                torch.cuda.synchronize();t=time.perf_counter()
                model.train();h,u,y=sample(train,rng,stage['batch'],probs[flags['excitation']],device)
                pred=factual(model,h,u,y,stage);loss=pred
                pre=sign=torch.zeros((),device=device)
                if flags['dbasr'] and (smoke or total>=PILOT_STEPS):
                    pre,sign=regularizer(sensitivity(model,h[:SUBSET],u[:SUBSET]))
                    loss=loss+lam*(pre+sign)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                opt.zero_grad(set_to_none=True);loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();total+=1
                torch.cuda.synchronize();compute_s+=time.perf_counter()-t
                log.write(json.dumps(dict(stage=si,step=step,global_step=total,prediction_loss=float(pred.detach()),
                           pre=float(pre.detach()),sign=float(sign.detach()),lambda_s=lam if flags['dbasr'] else 0.,
                           loss=float(loss.detach()),gradient_norm=float(grad)))+'\n')
                if step%25==0: log.flush()
                if step%100==0 or step==stage['steps']:
                    t=time.perf_counter();value=score_model(model,ev,ids,indices,threshold,device);eval_s+=time.perf_counter()-t
                    records.append(dict(stage=si,step=step,score=value))
                    if value<best: best=value;state=cpu_state(model);selected=dict(stage=si,step=step)
                    if si==0 and value<warmbest: warmbest=value;warmstate=cpu_state(model)
                    elapsed=time.perf_counter()-train_start
                    print(name,'step',total,'/',sum(s['steps'] for s in stages),'score',round(value,6),
                          'elapsed_s',round(elapsed,1),'ETA_this_run_s',round(elapsed/total*(sum(s['steps'] for s in stages)-total),1),flush=True)
                    save_json(run/'progress.json',dict(step=total,elapsed_s=elapsed,latest_score=value,records=records))
    elapsed=time.perf_counter()-train_start
    model.load_state_dict(state)
    torch.save(dict(state_dict=state,stats=stats,seed=SEED,variant=name,selected=selected),run/'selected.pt')
    result=dict(variant=name,seed=SEED,parameter_count=sum(p.numel() for p in model.parameters()),
                training_wall_s=elapsed,optimizer_compute_s=compute_s,checkpoint_evaluation_s=eval_s,
                selected=selected,selection_score=best,lambda_s=lam if flags['dbasr'] else 0.,checkpoint_sha256=sha(run/'selected.pt'))
    save_json(run/'trained.json',result)
    return model,result,run,indices


@torch.no_grad()
def evaluate(model,ev,ids,indices,events,threshold,device,run,result):
    model.eval();p=predict(model,ev.history[indices],ev.future[indices],device)
    rows,_,_,_=prediction_metrics(p,ev.target[indices],ev.origin[indices],ids[indices],threshold)
    write_csv(run/'prediction_metrics.csv',rows)
    sensitivities=[]
    for start in range(0,len(indices),256):
        ii=indices[start:start+256]
        h=torch.as_tensor(ev.history[ii],device=device);u=torch.as_tensor(ev.future[ii],device=device)
        sensitivities.append(sensitivity(model,h,u).cpu().numpy())
    a=np.concatenate(sensitivities);assert np.isfinite(a).all()
    sr=sensitivity_metrics(a);write_csv(run/'sensitivity_metrics.csv',sr)
    np.savez_compressed(run/'selection_predictions_and_sensitivity.npz',indices=indices,prediction=p,sensitivity=a)
    for r in rows:
        if r['point']=='all7':
            result[f'MAE_{r["horizon_s"]}s']=r['MAE_C'];result[f'RMSE_{r["horizon_s"]}s']=r['RMSE_C']
    result.update({k:v for k,v in sr[-1].items() if k in ('PLE','WSR','PSM')})
    if events:
        ep=predict(model,np.stack([e['history'] for e in events]),np.stack([e['future'] for e in events]),device)
        aggregate,details=excitation_metrics(ep,events,threshold)
        write_csv(run/'event_sanity.csv',details)
        write_csv(run/'event_sanity_summary.csv',[r for r in aggregate if r['split']=='selection' and r['magnitude']=='all' and r['point']=='all7'])
        result['event_sanity_count']=len(events)
    save_json(run/'complete.json',result)
    return result


def aggregate(results,cfg,folder):
    rows=[];r0=results[0]
    metrics=[f'{m}_{h}s' for h in (10,30,60,90,120) for m in ('MAE','RMSE')]+['PLE','WSR','PSM']
    for r in results:
        for m in metrics:
            rows.append(dict(variant=r['variant'],metric=m,value=r[m],R0_value=r0[m],
                             difference=r[m]-r0[m],relative_change_percent=100*(r[m]/r0[m]-1) if r0[m] else None))
    write_csv(folder/'tables'/'summary.csv',[{k:v for k,v in r.items() if k!='selected'} for r in results])
    write_csv(folder/'tables'/'relative_to_R0.csv',rows)
    r3=results[3];rules=cfg['screening']
    checks=dict(MAE_120s=r3['MAE_120s']<=rules['MAE_120s_max_C'],
                PLE=bool(r0['PLE']>0 and r3['PLE']<=.7*r0['PLE']),
                WSR=bool(r0['WSR']>0 and r3['WSR']<=.7*r0['WSR']),
                PSM=bool(r3['PSM']>=.2*r0['PSM'] and r3['PSM']>=rules['PSM_absolute_min']))
    save_json(folder/'SCREENING_DECISION.json',dict(checks=checks,passes_all=all(checks.values()),rules=rules,
                                                  action='eligible for next-stage decision' if all(checks.values()) else 'FAIL: stop; no tuning or rescue'))
    print('SCREENING',checks,'PASS',all(checks.values()),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['smoke','development'],required=True)
    args=parser.parse_args();smoke=args.mode=='smoke'
    torch.set_num_threads(2);device=torch.device('cuda')
    train,ev,fit_ids,sel_ids,stats,probs,scores,old=load_real()
    cfg=configuration(train,ev,fit_ids,sel_ids,stats,probs,scores,old)
    folder=OUTPUT_ROOT/'screen'/args.mode;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'PROTOCOL.json').exists(): raise FileExistsError('Protocol/run already exists; no silent overwrite')
    save_json(folder/'PROTOCOL.json',cfg)  # before calibration, training or selection results
    np.savez_compressed(folder/'fit_sampling_audit.npz',source_ids=fit_ids,score=scores,ordinary=probs[False],excitation=probs[True])
    if smoke:
        lam=.01
    else:
        smoke_pass=json.loads((folder.parent/'smoke'/'VERIFIED.json').read_text(encoding='utf-8'))
        if VERIFY_HASHES: assert smoke_pass['code_sha256']==sha(Path(__file__)), 'Code changed since smoke'
        calibration=calibrate(train,stats,probs,cfg['training']['stages'],device,folder);lam=calibration['lambda_s']
    threshold=np.asarray(old['regime_thresholds']['regime_deadband_C'])
    events=None
    results=[]
    for name,flags in cfg['variants'].items():
        model,result,run,indices=train_variant(name,flags,train,ev,sel_ids,stats,probs,cfg,lam,threshold,device,folder,smoke)
        # Event data are not consulted until a prediction-only checkpoint is fixed.
        if events is None:
            events=extract_events(reconstruct(ev,sel_ids),'selection',old['UH_event_extraction'])
            assert len(events)==2, 'Expected two unchanged selection sanity events'
        results.append(evaluate(model,ev,sel_ids,indices,events,threshold,device,run,result))
        print('COMPLETE',json.dumps(results[-1]),flush=True)
    aggregate(results,cfg,folder)
    assert all(r['parameter_count']==90887 for r in results)
    save_json(folder/'VERIFIED.json',dict(code_sha256=sha(Path(__file__)),variants=4,seed=SEED,
        fit_sources=91,selection_sources=18,frozen_access=False,synthetic_labels=False,mode=args.mode))


if __name__=='__main__':
    main()
