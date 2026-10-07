"""Formal paired R0 versus COISC validation, three frozen protocol seeds."""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch

import screen
from _paths import PROTOCOLS, OUTPUT_ROOT, load_checkpoint
from paper_eval import prediction_metrics, predict, excitation_metrics

PROTOCOL_PATH = PROTOCOLS / 'COISC_VALIDATION_PROTOCOL_DRAFT.json'
OUT = OUTPUT_ROOT / 'formal_coisc'
SEEDS = (20260922, 20260923, 20260924)
TARGET_TOKEN = 'S068'  # pseudonymized selection source of the T5 failure diagnostic
HORIZONS = (10, 30, 60, 90, 120)
POINTS = ('T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'Tp')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def protocol_integrity():
    raw = PROTOCOL_PATH.read_bytes()
    protocol = json.loads(raw.decode('utf-8'))
    assert protocol['status'] == 'FORMAL_DEVELOPMENT_VALIDATION_PROTOCOL_DRAFT_NO_RUN'
    assert tuple(protocol['comparisons']['seeds']) == SEEDS
    assert protocol['comparisons']['paired_by_seed'] is True
    assert protocol['training']['head_only_steps'] == 75
    assert protocol['training']['head_only_batch'] == 128
    assert protocol['training']['head_only_learning_rate'] == 5e-5
    assert protocol['training']['optimizer'] == 'Adam'
    assert protocol['training']['dbasr_subset_per_batch'] == 8
    assert protocol['training']['lambda_s'] == 939.1430335805455
    assert protocol['training']['epsilon_pre'] == .001
    assert protocol['training']['epsilon_sign'] == .0001
    assert protocol['training']['lag_seconds'] == [102.108,77.091,30.108,50.112,10.0535,180.0,39.104]
    assert protocol['training']['no_parameter_or_loss_changes'] is True
    assert protocol['training']['no_early_stopping'] is True
    assert protocol['training']['no_probe_checkpoint_selection'] is True
    assert protocol['data_protocol']['training_sources'] == '91 fit sources only'
    assert protocol['data_protocol']['selection_sources'] == '18 selection sources only'
    assert protocol['data_protocol']['synthetic_or_paired_data'] is False
    assert protocol['data_protocol']['frozen_external_or_final_data'] is False
    gates=protocol['primary_endpoints']['per_seed_pass']
    assert gates['Tp_120s_MAE'] == 'COISC <= paired R0 + 0.005 C'
    assert gates['Tp_PLE'] == 'COISC <= 90% of paired R0'
    assert gates['Tp_WSR'] == 'COISC <= 90% of paired R0'
    assert gates['Tp_PSM'] == 'COISC >= 20% of paired R0 and >= 0.0005'
    return protocol, screen.sha(PROTOCOL_PATH), raw


def tp_metrics(pred, target, sensitivity):
    rows=[]
    for h in HORIZONS:
        k=h//10-1; e=pred[:,k,6]-target[:,k,6]
        rows.append(dict(horizon_s=h,MAE_C=float(np.abs(e).mean()),RMSE_C=float(np.sqrt(np.square(e).mean())),bias_C=float(e.mean())))
    pre,post=screen.lag_masks(); s=sensitivity[:,:,6]
    sm=dict(PLE=float(np.abs(s[:,pre[:,6]]).mean()),WSR=float((s[:,post[:,6]] < -screen.EPS_SIGN).mean()),
            PSM=float(np.median(np.abs(s[:,post[:,6]]))),pre_cells=int(s[:,pre[:,6]].size),post_cells=int(s[:,post[:,6]].size))
    return rows,sm


def all_metrics(pred, target, origin, source_ids, threshold):
    rows, regimes, macro, legacy=prediction_metrics(pred,target,origin,source_ids,threshold)
    return rows, regimes, macro, legacy


def sensitivity_all(model, ev, device):
    values=[]
    for start in range(0,len(ev.history),256):
        h=torch.as_tensor(ev.history[start:start+256],device=device)
        u=torch.as_tensor(ev.future[start:start+256],device=device)
        values.append(screen.sensitivity(model,h,u).detach().cpu().numpy())
    out=np.concatenate(values).astype(np.float64)
    assert np.isfinite(out).all()
    return out


def delta_audit(pred0, pred1, target, source_ids, seed):
    rows=[]
    for source in np.unique(source_ids):
        mask=source_ids==source
        for h in HORIZONS:
            k=h//10-1
            for oi,point in enumerate(POINTS):
                e0=np.abs(pred0[mask,k,oi]-target[mask,k,oi]); e1=np.abs(pred1[mask,k,oi]-target[mask,k,oi]); d=e0-e1
                rows.append(dict(seed=seed,source_id=source,horizon_s=h,point=point,n=int(len(d)),
                                 R0_MAE_C=float(e0.mean()),COISC_MAE_C=float(e1.mean()),
                                 improvement_C=float(d.mean()),median_improvement_C=float(np.median(d)),
                                 fraction_windows_improved=float((d>0).mean()),fraction_windows_degraded=float((d<0).mean()),
                                 p05_improvement_C=float(np.quantile(d,.05)),p95_improvement_C=float(np.quantile(d,.95))))
    return rows


def sensitivity_rows(s0,s1,source_ids,seed):
    pre,post=screen.lag_masks(); rows=[]
    for model,s in (('R0',s0),('COISC',s1)):
        for source in np.unique(source_ids):
            mask=source_ids==source
            for oi,point in enumerate(POINTS):
                pm=pre[:,oi]; qm=post[:,oi]; x=s[mask,:,oi]
                rows.append(dict(seed=seed,model=model,source_id=source,point=point,n=int(mask.sum()),
                                 PLE=float(np.abs(x[:,pm]).mean()) if pm.any() else None,
                                 WSR=float((x[:,qm]<-screen.EPS_SIGN).mean()) if qm.any() else None,
                                 PSM=float(np.median(np.abs(x[:,qm]))) if qm.any() else None,
                                 pre_cells=int(mask.sum()*pm.sum()),post_cells=int(mask.sum()*qm.sum())))
    return rows


def target_diagnostic(pred0,pred1,target,origin,source_ids,seed):
    mask=np.array([TARGET_TOKEN in s for s in source_ids])
    assert mask.any()
    rows=[]
    for h in HORIZONS:
        k=h//10-1; y=target[mask,k,4]; r0=pred0[mask,k,4]; r1=pred1[mask,k,4]
        rows.append(dict(seed=seed,source_id=TARGET_TOKEN,horizon_s=h,n=int(mask.sum()),
                         R0_MAE_C=float(np.abs(r0-y).mean()),COISC_MAE_C=float(np.abs(r1-y).mean()),
                         COISC_minus_R0_MAE_C=float(np.abs(r1-y).mean()-np.abs(r0-y).mean()),
                         R0_bias_C=float((r0-y).mean()),COISC_bias_C=float((r1-y).mean()),
                         fraction_windows_COISC_improved=float((np.abs(r1-y)<np.abs(r0-y)).mean())))
    return rows


def event_check(model, events, threshold, device):
    if not events:
        return [], []
    pred=predict(model,np.stack([e['history'] for e in events]),np.stack([e['future'] for e in events]),device)
    agg, detail=excitation_metrics(pred,events,threshold)
    return agg, detail


def adapt_head(model, train, ev, stats, probs, old, lam, seed, device, seed_dir):
    screen.seed_all(seed)
    model.train()
    trainable=[]; before={name:param.detach().cpu().clone() for name,param in model.named_parameters()}
    for name,param in model.named_parameters():
        param.requires_grad_(name.startswith('head.'))
        if param.requires_grad: trainable.append(name)
    assert trainable and all(name.startswith('head.') for name in trainable)
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=5e-5)
    rng=np.random.default_rng(seed); stage=dict(old['training']['stages'][2]); logs=[]; start=time.perf_counter()
    with (seed_dir/'coisc_training.jsonl').open('w',encoding='utf-8') as log:
        for step in range(1,76):
            h,u,y=screen.sample(train,rng,128,probs[False],device)
            factual=screen.factual(model,h,u,y,stage)
            sens=screen.sensitivity(model,h[:screen.SUBSET],u[:screen.SUBSET]); pre,sign=screen.regularizer(sens)
            loss=factual+lam*(pre+sign)
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite COISC loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True);optimizer.step()
            row=dict(seed=seed,step=step,prediction_loss=float(factual.detach()),L_pre=float(pre.detach()),L_sign=float(sign.detach()),
                     dbasr=float((pre+sign).detach()),lambda_s=lam,total_loss=float(loss.detach()),grad_norm=float(grad));logs.append(row);log.write(json.dumps(row)+'\n')
    frozen=[]
    for name,param in model.named_parameters():
        if not name.startswith('head.'):
            frozen.append(torch.equal(param.detach().cpu(),before[name]))
        else:
            frozen.append(param.requires_grad)
    assert all(frozen)
    return time.perf_counter()-start, trainable, logs, before


def run_seed(seed, protocol, protocol_sha, train, ev, fit_ids, sel_ids, stats, probs, scores, old, device):
    seed_dir=OUT/f'seed_{seed}'
    if seed_dir.exists(): raise FileExistsError(f'Existing formal seed directory: {seed_dir}')
    seed_dir.mkdir(parents=True)
    screen.save_json(seed_dir/'PROTOCOL.json', protocol)
    screen.SEED=seed
    cfg=screen.configuration(train,ev,fit_ids,sel_ids,stats,probs,scores,old)
    flags={'excitation':False,'dbasr':False}
    threshold=np.asarray(old['regime_thresholds']['regime_deadband_C'])
    base_model, base_train_result, base_run, indices=screen.train_variant('R0',flags,train,ev,sel_ids,stats,probs,cfg,0.,threshold,device,seed_dir,False)
    events=screen.extract_events(screen.reconstruct(ev,sel_ids),'selection',old['UH_event_extraction'])
    assert len(events)==2
    base_eval_result=screen.evaluate(base_model,ev,sel_ids,indices,events,threshold,device,base_run,base_train_result)
    base_pred=predict(base_model,ev.history,ev.future,device)
    base_sens=sensitivity_all(base_model,ev,device)
    model=screen.build_model('B4',stats).to(device)
    checkpoint=load_checkpoint(base_run/'selected.pt',map_location=device); model.load_state_dict(checkpoint['state_dict'])
    with (seed_dir/'per_seed_config.json').open('w',encoding='utf-8') as f:
        json.dump(dict(seed=seed,protocol_sha256=protocol_sha,protocol=protocol,base_config=json.loads((base_run/'config.json').read_text(encoding='utf-8')),
                       base_checkpoint=str(base_run/'selected.pt'),base_checkpoint_sha256=screen.sha(base_run/'selected.pt'),
                       coisc=dict(steps=75,batch=128,lr=5e-5,lambda_s=protocol['training']['lambda_s'],epsilon_pre=.001,epsilon_sign=.0001,
                                  lag_seconds=protocol['training']['lag_seconds'],trainable_scope='head.* only')),f,ensure_ascii=False,indent=2)
    adapt_time,trainable,logs,before=adapt_head(model,train,ev,stats,probs,old,protocol['training']['lambda_s'],seed,device,seed_dir)
    coisc_model=model.eval(); coisc_pred=predict(coisc_model,ev.history,ev.future,device); coisc_sens=sensitivity_all(coisc_model,ev,device)
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in coisc_model.state_dict().items()},seed=seed,steps=75,lambda_s=protocol['training']['lambda_s']),seed_dir/'coisc_final.pt')
    # Tensor equality check uses the pre-adaptation R0 tensors for every non-head name.
    checkpoint_state=checkpoint['state_dict']; nonhead_equal=all(torch.equal(coisc_model.state_dict()[n].detach().cpu(),checkpoint_state[n].detach().cpu()) for n in checkpoint_state if not n.startswith('head.'))
    requires_grad_ok=all((p.requires_grad == n.startswith('head.')) for n,p in coisc_model.named_parameters())
    assert nonhead_equal and requires_grad_ok
    base_rows,base_reg,base_macro,base_legacy=all_metrics(base_pred,ev.target,ev.origin,sel_ids,threshold)
    coisc_rows,coisc_reg,coisc_macro,coisc_legacy=all_metrics(coisc_pred,ev.target,ev.origin,sel_ids,threshold)
    base_tp,base_sens_metrics=tp_metrics(base_pred,ev.target,base_sens); coisc_tp,coisc_sens_metrics=tp_metrics(coisc_pred,ev.target,coisc_sens)
    csv_rows=[]
    for r in base_rows: csv_rows.append(dict(seed=seed,variant='R0',**r))
    for r in coisc_rows: csv_rows.append(dict(seed=seed,variant='COISC',**r))
    screen.write_csv(seed_dir/'selection_prediction_metrics.csv',csv_rows)
    screen.write_csv(seed_dir/'sensitivity_metrics.csv',sensitivity_rows(base_sens,coisc_sens,sel_ids.astype(str),seed))
    delta=delta_audit(base_pred,coisc_pred,ev.target,sel_ids.astype(str),seed); screen.write_csv(seed_dir/'source_horizon_output_audit.csv',delta)
    t5=target_diagnostic(base_pred,coisc_pred,ev.target,ev.origin,sel_ids.astype(str),seed); screen.write_csv(seed_dir/'failure_source_T5_diagnostic.csv',t5)
    base_event_agg,base_event_detail=event_check(base_model,events,threshold,device); coisc_event_agg,coisc_event_detail=event_check(coisc_model,events,threshold,device)
    screen.write_csv(seed_dir/'event_sanity.csv',[dict(seed=seed,variant='R0',**r) for r in base_event_detail]+[dict(seed=seed,variant='COISC',**r) for r in coisc_event_detail])
    screen.write_csv(seed_dir/'event_sanity_summary.csv',[dict(seed=seed,variant='R0',**r) for r in base_event_agg if r['split']=='selection' and r['magnitude']=='all' and r['point']=='all7']+[dict(seed=seed,variant='COISC',**r) for r in coisc_event_agg if r['split']=='selection' and r['magnitude']=='all' and r['point']=='all7'])
    b120=next(x['MAE_C'] for x in base_tp if x['horizon_s']==120); c120=next(x['MAE_C'] for x in coisc_tp if x['horizon_s']==120)
    gates=dict(Tp_120s_MAE=dict(value=c120,threshold=b120+.005,pass_=c120<=b120+.005),PLE=dict(value=coisc_sens_metrics['PLE'],threshold=.9*base_sens_metrics['PLE'],pass_=coisc_sens_metrics['PLE']<=.9*base_sens_metrics['PLE']),WSR=dict(value=coisc_sens_metrics['WSR'],threshold=.9*base_sens_metrics['WSR'],pass_=coisc_sens_metrics['WSR']<=.9*base_sens_metrics['WSR']),PSM=dict(value=coisc_sens_metrics['PSM'],threshold=max(.2*base_sens_metrics['PSM'],.0005),pass_=coisc_sens_metrics['PSM']>=max(.2*base_sens_metrics['PSM'],.0005)))
    gates['overall_pass']=all(x['pass_'] for x in gates.values())
    model_summary=[]
    for variant,train_result,ctime,rows,sm in [('R0',base_train_result,base_train_result['training_wall_s'],base_tp,base_sens_metrics),('COISC',dict(training_wall_s=adapt_time),adapt_time,coisc_tp,coisc_sens_metrics)]:
        r120=next(x for x in rows if x['horizon_s']==120); model_summary.append(dict(seed=seed,variant=variant,parameter_count=sum(p.numel() for p in coisc_model.parameters()),training_time_s=ctime,
          MAE_10s=next(x['MAE_C'] for x in rows if x['horizon_s']==10),RMSE_10s=next(x['RMSE_C'] for x in rows if x['horizon_s']==10),MAE_30s=next(x['MAE_C'] for x in rows if x['horizon_s']==30),RMSE_30s=next(x['RMSE_C'] for x in rows if x['horizon_s']==30),MAE_60s=next(x['MAE_C'] for x in rows if x['horizon_s']==60),RMSE_60s=next(x['RMSE_C'] for x in rows if x['horizon_s']==60),MAE_90s=next(x['MAE_C'] for x in rows if x['horizon_s']==90),RMSE_90s=next(x['RMSE_C'] for x in rows if x['horizon_s']==90),MAE_120s=r120['MAE_C'],RMSE_120s=r120['RMSE_C'],PLE=sm['PLE'],WSR=sm['WSR'],PSM=sm['PSM']))
    screen.write_csv(seed_dir/'model_summary.csv',model_summary)
    result=dict(seed=seed,base_train_result=base_train_result,coisc_training_time_s=adapt_time,trainable_parameters=trainable,
                nonhead_tensor_equality=nonhead_equal,requires_grad_scope_ok=requires_grad_ok,selection_windows=len(ev.history),
                base_Tp=base_tp,coisc_Tp=coisc_tp,base_sensitivity=base_sens_metrics,coisc_sensitivity=coisc_sens_metrics,
                gates=gates,base_prediction_macro_MAE=base_macro,coisc_prediction_macro_MAE=coisc_macro,
                event_sanity_count=len(events),files=sorted(p.name for p in seed_dir.iterdir()))
    screen.save_json(seed_dir/'per_seed_base_and_coisc_metrics.json',result)
    return result,model_summary,delta,sensitivity_rows(base_sens,coisc_sens,sel_ids.astype(str),seed),t5


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--integrity-only',action='store_true');args=parser.parse_args()
    protocol,protocol_sha,protocol_raw=protocol_integrity()
    if args.integrity_only:
        print(json.dumps(dict(protocol_sha256=protocol_sha,seeds=SEEDS,lambda_s=protocol['training']['lambda_s'],steps=75,scope=protocol['comparisons']['COISC']),ensure_ascii=False,indent=2)); return
    if OUT.exists():
        integrity=json.loads((OUT/'PROTOCOL_INTEGRITY.json').read_text(encoding='utf-8'))
        if screen.VERIFY_HASHES: assert integrity['protocol_sha256']==protocol_sha, 'Protocol changed after formal run began'
    else:
        OUT.mkdir(parents=True)
        screen.save_json(OUT/'PROTOCOL_INTEGRITY.json',dict(protocol_sha256=protocol_sha,protocol_bytes=len(protocol_raw),status='PASSED_BEFORE_TRAINING',seeds=list(SEEDS),
                  exact_values=dict(steps=75,batch=128,lr=5e-5,lambda_s=939.1430335805455,epsilon_pre=.001,epsilon_sign=.0001,lag_seconds=protocol['training']['lag_seconds']),
                  frozen_access=False,synthetic_access=False))
    torch.set_num_threads(2);device=torch.device('cuda')
    train,ev,fit_ids,sel_ids,stats,probs,scores,old=screen.load_real()
    assert len(set(fit_ids))==91 and len(set(sel_ids))==18 and not set(fit_ids)&set(sel_ids)
    cfg=screen.configuration(train,ev,fit_ids,sel_ids,stats,probs,scores,old)
    all_results=[]; all_summary=[]; all_delta=[]; all_sensitivity=[]; all_t5=[]; all_prediction=[]; execution=[]
    for seed in SEEDS:
        start=time.perf_counter(); seed_dir=OUT/f'seed_{seed}'
        result_path=seed_dir/'per_seed_base_and_coisc_metrics.json'
        if result_path.exists():
            result=json.loads(result_path.read_text(encoding='utf-8'))
            summary=read_csv(seed_dir/'model_summary.csv')
            delta=read_csv(seed_dir/'source_horizon_output_audit.csv')
            sens=read_csv(seed_dir/'sensitivity_metrics.csv')
            t5=read_csv(seed_dir/'failure_source_T5_diagnostic.csv')
            print('RESUME SKIP COMPLETE SEED',seed,flush=True)
        else:
            result,summary,delta,sens,t5=run_seed(seed,protocol,protocol_sha,train,ev,fit_ids,sel_ids,stats,probs,scores,old,device)
        all_results.append(result);all_summary.extend(summary);all_delta.extend(delta);all_sensitivity.extend(sens);all_t5.extend(t5)
        all_prediction.extend(read_csv(OUT/f'seed_{seed}'/'selection_prediction_metrics.csv'))
        execution.append(dict(seed=seed,elapsed_s=time.perf_counter()-start,nonhead_tensor_equality=result['nonhead_tensor_equality'],requires_grad_scope_ok=result['requires_grad_scope_ok'],gates=result['gates']))
        screen.save_json(OUT/'progress.json',dict(completed_seeds=[r['seed'] for r in all_results],latest=result))
    screen.write_csv(OUT/'model_summary.csv',all_summary);screen.write_csv(OUT/'selection_prediction_metrics.csv',all_prediction)
    screen.write_csv(OUT/'sensitivity_metrics.csv',all_sensitivity);screen.write_csv(OUT/'source_horizon_output_audit.csv',all_delta);screen.write_csv(OUT/'failure_source_T5_diagnostic.csv',all_t5)
    decisions=[]
    for r in all_results:
        decisions.append(dict(seed=r['seed'],gates=r['gates'],PASS=bool(r['gates']['overall_pass']),nonhead_tensor_equality=r['nonhead_tensor_equality'],requires_grad_scope_ok=r['requires_grad_scope_ok']))
    overall=all(d['PASS'] for d in decisions)
    screen.save_json(OUT/'protocol_execution_audit.json',dict(status='completed_formal_coisc',protocol_sha256=protocol_sha,seeds=list(SEEDS),fit_sources=91,selection_sources=18,
      selection_windows=len(ev.history),frozen_access=False,synthetic_or_paired_label_access=False,early_stopping=False,checkpoint_selection='existing R0 source-macro rule only; COISC final state at step75',
      execution=execution,all_nonhead_tensor_equal=all(r['nonhead_tensor_equality'] for r in all_results),all_requires_grad_scopes_valid=all(r['requires_grad_scope_ok'] for r in all_results)))
    screen.save_json(OUT/'FORMAL_COISC_DECISION.json',dict(protocol_sha256=protocol_sha,per_seed=decisions,overall_PASS=overall,overall_decision='PASS' if overall else 'FAIL',
      stop_rule='Any seed failure is reported without rescue; no parameters or rules changed'))
    screen.save_json(OUT/'per_seed_config.json',dict(seeds=list(SEEDS),protocol_sha256=protocol_sha,configs=[json.loads((OUT/f'seed_{s}/per_seed_config.json').read_text(encoding='utf-8')) for s in SEEDS]))
    screen.save_json(OUT/'per_seed_base_and_coisc_metrics.json',dict(seeds=all_results,overall_PASS=overall))
    print(json.dumps(dict(overall_PASS=overall,per_seed=[dict(seed=d['seed'],PASS=d['PASS'],gates=d['gates']) for d in decisions]),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
