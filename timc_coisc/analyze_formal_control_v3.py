"""Locked hierarchical analysis of complete formal control traces."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

from _paths import PROTOCOLS
from paper_data import save_json, sha, write_csv
from run_formal_control_v3 import read, schedule, verify, SEEDS


def hierarchical_effect(by_source_family, draws):
    matrix=np.asarray(by_source_family,dtype=np.float64)
    indices=np.asarray(draws,dtype=np.int64)
    if matrix.shape!=(12,4) or indices.ndim!=2 or indices.shape[1]!=12 or np.any((indices<0)|(indices>=12)):
        raise ValueError('hierarchical effect input shape/draw mismatch')
    scenario_effect=matrix.mean(axis=1)
    sampled=np.mean(scenario_effect[indices],axis=1)
    lower,upper=np.percentile(sampled,[2.5,97.5])
    return float(np.mean(scenario_effect)),float(lower),float(upper)


def trace_metrics(data, previous_uh, initial_tp, runtime):
    y=np.asarray(data['plant_Tp'],dtype=np.float64)
    u=np.asarray(data['applied_UH'],dtype=np.float64)
    good=bool(y.shape==(60,) and np.isfinite(y).all() and np.isfinite(u).all() and np.all((0<=u)&(u<=100)))
    if not good:
        return dict(evaluable=False)
    error=y-37.
    du=np.diff(np.r_[previous_uh,u])
    nonzero=np.sign(du[np.abs(du)>1e-12])
    settling=next((10*(k+1) for k in range(60) if np.all(np.abs(error[k:])<=1.0)),None)
    overshoot=max(0.,float(np.max(y-37.))) if initial_tp<=37. else max(0.,float(np.max(37.-y)))
    return dict(evaluable=True,MAE_track=float(np.mean(np.abs(error))),RMSE_track=float(np.sqrt(np.mean(error**2))),
        ISE=float(10*np.sum(error**2)),max_abs_error=float(np.max(np.abs(error))),
        overshoot=overshoot,steady_state_error=float(np.mean(error[-6:])),settling_time_s=settling,
        mean_abs_delta_u=float(np.mean(np.abs(du))),total_variation_u=float(np.sum(np.abs(du))),
        saturation_fraction=float(np.mean((u<=0.)|(u>=100.))),
        command_reversal_count=int(np.sum(nonzero[1:]!=nonzero[:-1])) if len(nonzero)>1 else 0,
        mean_u=float(np.mean(u)),solver_success_rate=float(np.mean(data['solver_success'])),
        solver_failure_count=int(np.sum(~data['solver_success'])),
        iterations=float(np.mean(data['solver_iterations'])),objective_evaluations=float(np.mean(data['solver_evaluations'])),
        time_per_control_step_s=float(runtime/60.))


def source_strata(reach_csv):
    lookup={}
    with Path(reach_csv).open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            if float(row['UH_percent'])!=0.:
                continue
            zero=float(row['Tp_600s_UH_0'])
            high=float(row['Tp_600s_UH_100'])
            if zero<=37.<=high or high<=37.<=zero:
                label='BRACKETED_AT_600S'
            elif zero<37. and high<37.:
                label='BOTH_BELOW_37_AT_600S'
            else:
                label='BOTH_ABOVE_37_AT_600S'
            lookup[(row['plant_checkpoint'],row['scenario_id'])]=label
    assert len(lookup)==24*12
    return lookup


def main(execution_dir):
    execution=Path(execution_dir).resolve()
    manifest,protocol=verify(execution)
    plants,tasks=schedule()
    raw_dir=execution/'raw_runs'
    assert len(list(raw_dir.glob('*.npz')))==len(tasks)==2772
    scenario_lock=read(PROTOCOLS/'CONTROL_SCENARIO_LOCK_V2.json')
    scenarios=scenario_lock['scenarios']
    strata=source_strata(PROTOCOLS/'CONTROL_PLANT_REACHABILITY_AUDIT.csv')
    runs={}
    arrays={key:[] for key in ('plant_Tp','applied_UH','solver_success','solver_iterations','solver_evaluations','objective','internal_predicted_Tp','optimal_sequence','decision_history','plant_output')}
    metadata=[]
    run_rows=[]
    solver_rows=[]
    for index,task in enumerate(tasks):
        path=raw_dir/f'{index:04d}.npz'
        with np.load(path,allow_pickle=False) as z:
            data={key:z[key] for key in z.files if key!='metadata'}
            meta=json.loads(str(z['metadata']))
        plant=plants[task['plant_index']]
        scenario=scenarios[task['scenario_index']]
        assert meta['plant']==plant['plant_name'] and meta['scenario_id']==scenario['scenario_id']
        assert meta['controller_seed']==task['seed'] and meta['controller_variant']==task['variant']
        metric=trace_metrics(data,meta['previous_uh_percent'],meta['initial_Tp_C'],meta['runtime_s'])
        if not metric['evaluable']:
            raise ValueError(f'non-evaluable control trace {index}: {meta}')
        stratum=strata.get((meta['plant'],meta['scenario_id']),'SECONDARY_NOT_QUALIFIED')
        row={**meta,**metric,'reachability_stratum':stratum,'raw_trace_file':str(path)}
        run_rows.append(row)
        solver_rows.append({k:row[k] for k in ('plant','family','panel','scenario_id','controller_seed','controller_variant','solver_success_rate','solver_failure_count','iterations','objective_evaluations','time_per_control_step_s','saturation_fraction','mean_abs_delta_u','total_variation_u')})
        runs[(task['plant_index'],task['scenario_index'],task['seed'],task['variant'])]=(row,data)
        metadata.append(meta)
        for key in arrays:
            arrays[key].append(data[key])
    np.savez_compressed(execution/'CONTROL_RAW_TRACES.npz',metadata_json=np.array([json.dumps(m,ensure_ascii=False) for m in metadata]),
        **{key:np.stack(value) for key,value in arrays.items()})
    write_csv(execution/'CONTROL_CHECKPOINT_RESULTS.csv',run_rows)
    write_csv(execution/'CONTROL_SOLVER_DIAGNOSTICS.csv',solver_rows)

    paired=[]
    oracle=[]
    mechanistic=[]
    for pi,plant in enumerate(plants):
        for si,scenario in enumerate(scenarios):
            oracle_row,_=runs[(pi,si,None,'ORACLE')]
            for seed in SEEDS:
                r0,r0_data=runs[(pi,si,seed,'R0')]
                co,co_data=runs[(pi,si,seed,'COISC')]
                base=dict(plant=plant['plant_name'],family=plant.get('family'),
                    panel='primary' if 'family' in plant else plant['secondary_panel'],
                    scenario_id=scenario['scenario_id'],source_id=scenario['source_id'],seed=seed,
                    reachability_stratum=r0['reachability_stratum'])
                paired.append({**base,'MAE_R0':r0['MAE_track'],'MAE_COISC':co['MAE_track'],
                    'Delta_COISC_minus_R0':co['MAE_track']-r0['MAE_track'],
                    'MAE_oracle':oracle_row['MAE_track'],
                    'excess_R0':r0['MAE_track']-oracle_row['MAE_track'],
                    'excess_COISC':co['MAE_track']-oracle_row['MAE_track']})
                oracle.append({**base,'MAE_oracle':oracle_row['MAE_track'],
                    'excess_R0':r0['MAE_track']-oracle_row['MAE_track'],
                    'excess_COISC':co['MAE_track']-oracle_row['MAE_track'],
                    'oracle_solver_success_rate':oracle_row['solver_success_rate'],
                    'oracle_saturation_fraction':oracle_row['saturation_fraction']})
                for step in range(60):
                    previous_r0=float(scenario['previous_uh_percent']) if step==0 else float(r0_data['applied_UH'][step-1])
                    previous_co=float(scenario['previous_uh_percent']) if step==0 else float(co_data['applied_UH'][step-1])
                    before_r0=float(scenario['initial_Tp_C']) if step==0 else float(r0_data['plant_Tp'][step-1])
                    before_co=float(scenario['initial_Tp_C']) if step==0 else float(co_data['plant_Tp'][step-1])
                    pred_r0=float(r0_data['internal_predicted_Tp'][step,0])
                    pred_co=float(co_data['internal_predicted_Tp'][step,0])
                    mechanistic.append({**base,'step':step,
                        'first_action_difference_COISC_minus_R0':float(co_data['applied_UH'][step]-r0_data['applied_UH'][step]),
                        'action_direction_R0':int(np.sign(r0_data['applied_UH'][step]-previous_r0)),
                        'action_direction_COISC':int(np.sign(co_data['applied_UH'][step]-previous_co)),
                        'action_magnitude_R0':float(abs(r0_data['applied_UH'][step]-previous_r0)),
                        'action_magnitude_COISC':float(abs(co_data['applied_UH'][step]-previous_co)),
                        'predicted_response_direction_R0':int(np.sign(pred_r0-before_r0)) if np.isfinite(pred_r0) else None,
                        'predicted_response_direction_COISC':int(np.sign(pred_co-before_co)) if np.isfinite(pred_co) else None,
                        'plant_realized_response_direction_R0':int(np.sign(r0_data['plant_Tp'][step]-before_r0)),
                        'plant_realized_response_direction_COISC':int(np.sign(co_data['plant_Tp'][step]-before_co)),
                        'short_horizon_realized_improvement_COISC_minus_R0':float(np.mean(np.abs(co_data['plant_Tp'][step:min(step+3,60)]-37.))-np.mean(np.abs(r0_data['plant_Tp'][step:min(step+3,60)]-37.)))})
    write_csv(execution/'CONTROL_MECHANISTIC_DIAGNOSTICS.csv',mechanistic)
    write_csv(execution/'CONTROL_ORACLE_RESULTS.csv',oracle)
    write_csv(execution/'CONTROL_SECONDARY_PANEL_RESULTS.csv',[r for r in paired if r['panel']!='primary'])
    write_csv(execution/'CONTROL_REACHABILITY_STRATA_RESULTS.csv',[r for r in paired if r['panel']=='primary'])
    write_csv(execution/'CONTROL_SCENARIO_RESULTS.csv',[r for r in paired if r['panel']=='primary'])

    draw_rng=np.random.default_rng(20260927)
    draws=draw_rng.integers(0,12,size=(20000,12),dtype=np.int16)
    np.savez_compressed(execution/'CONTROL_BOOTSTRAP_DRAWS.npz',source_draws=draws,
        scenario_ids=np.array([s['scenario_id'] for s in scenarios]))
    family_ids=('F1','F2','F3','F4')
    family_rows=[]
    primary_rows=[]
    bootstrap_rows=[]
    seed_decisions=[]
    for seed in SEEDS:
        by_source_family=np.empty((12,4),dtype=np.float64)
        for si,scenario in enumerate(scenarios):
            for fi,family_id in enumerate(family_ids):
                values=[runs[(pi,si,seed,'COISC')][0]['MAE_track']-runs[(pi,si,seed,'R0')][0]['MAE_track']
                    for pi,plant in enumerate(plants) if plant.get('family')==family_id]
                assert len(values)==read(PROTOCOLS/'CONTROL_PLANT_FAMILY_LOCK.json')['primary_checkpoint_counts'][family_id]
                by_source_family[si,fi]=np.mean(values)
        for fi,family_id in enumerate(family_ids):
            family_rows.append(dict(seed=seed,family=family_id,checkpoint_count=sum(p.get('family')==family_id for p in plants),
                scenario_count=12,Delta_COISC_minus_R0=float(np.mean(by_source_family[:,fi]))))
        point,lower,upper=hierarchical_effect(by_source_family,draws)
        decision='SUPPORT' if upper<0. else 'NO_SUPERIORITY_SUPPORT'
        primary_rows.append(dict(seed=seed,primary_family_equal_Delta=point,ci_lower=float(lower),ci_upper=float(upper),decision=decision))
        bootstrap_rows.append(dict(seed=seed,replicates=20000,rng_seed=20260927,point=point,ci_lower=float(lower),ci_upper=float(upper),draws_file='CONTROL_BOOTSTRAP_DRAWS.npz'))
        seed_decisions.append(dict(seed=seed,primary_family_equal_Delta=point,ci_lower=float(lower),ci_upper=float(upper),decision=decision))
    write_csv(execution/'CONTROL_PRIMARY_RESULTS.csv',primary_rows)
    write_csv(execution/'CONTROL_FAMILY_RESULTS.csv',family_rows)
    write_csv(execution/'CONTROL_BOOTSTRAP_RESULTS.csv',bootstrap_rows)
    support=all(row['decision']=='SUPPORT' for row in seed_decisions)
    decision=dict(status='CONTROL_SUPPORT' if support else 'NO_CONTROL_SUPERIORITY',
        overall_support=support,per_seed=seed_decisions,
        delivered23_prediction_result='FAIL',
        control_evidence_scope='surrogate-plant/model-in-the-loop only',no_physical_control_claim=True,
        no_training=True,no_tuning=True,no_plant_deletion=True,no_scenario_deletion=True,no_rescue=True,
        primary_plant_checkpoints=24,secondary_plant_checkpoints=9,scenario_count=12,
        expected_runs=2772,complete_runs=len(run_rows),protocol_sha256=manifest['protocol_sha256'],
        control_outcome_accessed=True)
    save_json(execution/'CONTROL_FINAL_DECISION.json',decision)
    audit=read(execution/'CONTROL_EXECUTION_AUDIT.json')
    audit.update(status='COMPLETE',control_outcome_accessed=True,bootstrap_replicates=20000,
        completed_runs=len(run_rows),results_file_sha256={p.name:sha(p) for p in execution.iterdir()
            if p.is_file() and p.name not in ('CONTROL_EXECUTION_AUDIT.json','CONTROL_RESULTS_SUMMARY.md')})
    save_json(execution/'CONTROL_EXECUTION_AUDIT.json',audit)
    summaries={v:np.mean([r['MAE_track'] for r in run_rows if r['controller_variant']==v]) for v in ('R0','COISC','ORACLE')}
    solver={v:sum(r['solver_failure_count'] for r in run_rows if r['controller_variant']==v) for v in ('R0','COISC','ORACLE')}
    text=['# Formal control results','',f'Execution: `{manifest["execution_id"]}`',
        f'Overall: **{decision["status"]}**. Delivered23 prediction result remains FAIL.',
        'Scope: fixed learned surrogate plants only; no physical control claim.','',
        '## Primary paired family-equal effects','']
    text += [f'- Seed {r["seed"]}: Delta {r["primary_family_equal_Delta"]:.6f} C, 95% CI [{r["ci_lower"]:.6f}, {r["ci_upper"]:.6f}], {r["decision"]}' for r in primary_rows]
    text += ['',f'All-run mean MAE (descriptive, not the primary estimand): {summaries}',f'Solver failure counts: {solver}',
        'All 33 plant checkpoints and 12 scenarios retained. Oracle and reachability strata are secondary only.']
    (execution/'CONTROL_RESULTS_SUMMARY.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps({'status':decision['status'],'per_seed':seed_decisions,'runs':len(run_rows)}))


if __name__=='__main__':
    if len(sys.argv)!=2:
        raise SystemExit('usage: analyze_formal_control_v3.py EXECUTION_DIR')
    main(sys.argv[1])
