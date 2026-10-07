"""V3 control execution with audited global-normalization loader compatibility."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from _paths import PROTOCOLS, CONFIGS, VERIFY_HASHES, private_path, load_checkpoint, public_source_id
from paper_data import sha
from paper_models import build_model, RAW_POWER_FULL_SCALE
from control_mpc import LegacyNativeAdapter, MPCController, MPCConfig, apply_plant_step

SEEDS = (20260922, 20260923, 20260924)
VARIANTS = ('R0', 'COISC')
NORMALIZATION = CONFIGS/'normalization.json'
NORMALIZATION_SHA = '6e0a0a41664c962c7e0d36f3694bb42eddd3279a9ae82cc569d06a6fac572dc1'
STATS_FIELDS = ('temperature_mean','temperature_std','uh_mean','uh_std','auxiliary_mean','auxiliary_std')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)


def verify(execution):
    manifest = read(execution/'CONTROL_EXECUTION_MANIFEST.json')
    protocol_path = PROTOCOLS/'CONTROL_VALIDATION_PROTOCOL_V3.json'
    protocol = read(protocol_path)
    assert protocol['status']=='LOCKED_BEFORE_FORMAL_CONTROL_OUTCOME'
    assert not VERIFY_HASHES or sha(protocol_path)==manifest['protocol_sha256']
    for binding in manifest['bound_files']:
        assert not VERIFY_HASHES or sha(private_path(binding['path']))==binding['sha256'], binding['path']
    return manifest,protocol


def load_scenarios(protocol):
    lock=read(PROTOCOLS/'CONTROL_SCENARIO_LOCK_V2.json')
    cache=private_path(lock['source'])
    assert not VERIFY_HASHES or sha(cache)==lock['cache_sha256']
    histories=[]
    with np.load(cache,allow_pickle=False) as z:
        for item in lock['scenarios']:
            history=z[f'{item["split"]}_history'][item['cache_row_index']]
            assert hashlib.sha256(history.tobytes()).hexdigest()==item['history_sha256']
            assert public_source_id(z[f'{item["split"]}_source_id'][item['cache_row_index']])==item['source_id']
            np.testing.assert_array_equal(history[-1,[6,8,9,10,11]],item['background_raw_aux'])
            histories.append(history.copy())
    return lock['scenarios'],histories


def load_model(binding,model_id):
    path=private_path(binding['path'])
    assert not VERIFY_HASHES or sha(path)==binding['sha256']
    if model_id=='V10':
        script_path=private_path(binding['torchscript_path'])
        assert not VERIFY_HASHES or sha(script_path)==binding['torchscript_sha256']
        with script_path.open('rb') as stream:
            return LegacyNativeAdapter(torch.jit.load(stream,map_location='cpu'))
    payload=load_checkpoint(path)
    stats=resolve_normalization(payload)
    model=build_model(model_id,stats)
    model.load_state_dict(payload['state_dict'],strict=True)
    model.eval()
    for tensor in model.parameters():
        tensor.requires_grad_(False)
    return model


def resolve_normalization(payload, canonical_path=NORMALIZATION, expected_sha=NORMALIZATION_SHA):
    if VERIFY_HASHES and sha(Path(canonical_path)) != expected_sha:
        raise ValueError('canonical normalization SHA256 mismatch')
    canonical=read(canonical_path)
    checkpoint_stats=payload.get('stats')
    if checkpoint_stats is not None:
        if not isinstance(checkpoint_stats,dict) or set(checkpoint_stats)!=set(canonical):
            raise ValueError('checkpoint normalization stats fields mismatch')
        for field in STATS_FIELDS:
            actual=np.asarray(checkpoint_stats[field],dtype=np.float64)
            expected=np.asarray(canonical[field],dtype=np.float64)
            tolerance=1e-6 if field.startswith('temperature_') else 1e-12
            if actual.shape!=expected.shape or not np.all(np.isfinite(actual)) or not np.allclose(actual,expected,rtol=0,atol=tolerance):
                raise ValueError(f'checkpoint normalization stats conflict: {field}')
    state=payload.get('state_dict',{})
    for field,expected in (('temperature_mean',canonical['temperature_mean']),
                           ('temperature_std',canonical['temperature_std']),
                           ('power_full_scale',RAW_POWER_FULL_SCALE)):
        if field not in state:
            raise ValueError(f'missing checkpoint normalization buffer: {field}')
        actual=np.asarray(state[field].detach().cpu(),dtype=np.float64)
        target=np.asarray(expected,dtype=np.float64)
        if actual.shape!=target.shape or not np.all(np.isfinite(actual)) or not np.allclose(actual,target,rtol=0,atol=1e-6):
            raise ValueError(f'checkpoint normalization buffer conflict: {field}')
    return canonical


def schedule():
    panel=read(PROTOCOLS/'CONTROL_PLANT_FAMILY_LOCK.json')
    plants=panel['primary_checkpoints']+panel['secondary_checkpoints']
    tasks=[]
    for pi,plant in enumerate(plants):
        for si in range(12):
            for seed in SEEDS:
                for variant in VARIANTS:
                    tasks.append(dict(plant_index=pi,scenario_index=si,seed=seed,variant=variant))
            tasks.append(dict(plant_index=pi,scenario_index=si,seed=None,variant='ORACLE'))
    assert len(tasks)==33*12*7
    return plants,tasks


def run_one(task,plants,scenarios,histories,controllers,plant_model,output_path,execution_id):
    plant=plants[task['plant_index']]
    scenario=scenarios[task['scenario_index']]
    history=histories[task['scenario_index']]
    seed=task['seed']
    variant=task['variant']
    internal=plant_model if variant=='ORACLE' else controllers[(seed,variant)]
    controller=MPCController(internal,MPCConfig())
    reference=np.full(60+12-1,37.,dtype=np.float64)
    background=np.broadcast_to(np.asarray(scenario['background_raw_aux'],dtype=np.float64),(71,5)).copy()
    start=time.perf_counter()
    plant_history=torch.as_tensor(history,dtype=torch.float32).clone()
    previous=float(np.clip(float(plant_history[-1,7])/5.,0.,100.))
    trace=[]
    for step in range(60):
        before_history=plant_history.numpy().copy()
        step_started=time.perf_counter()
        decision=controller.decide(plant_history,reference[step:step+12],background[step:step+12],previous)
        plant_history,output=apply_plant_step(plant_model,plant_history,decision['command_percent'],background[step])
        trace.append(dict(step=step,reference_c=float(reference[step]),plant_tp=float(output[6]),
            plant_output=output.numpy().copy(),command_percent=decision['command_percent'],
            success=decision['success'],iterations=decision['iterations'],evaluations=decision['evaluations'],
            objective=decision['objective'],decision_history=before_history,
            internal_prediction=decision.get('predicted_trajectory'),optimal=decision.get('optimal_percent'),
            solver_message=decision['message'],time_per_control_step_s=time.perf_counter()-step_started))
        previous=decision['command_percent']
    runtime=time.perf_counter()-start
    assert len(trace)==60
    metadata=dict(execution_id=execution_id,plant=plant['plant_name'],family=plant.get('family'),
        panel='primary' if 'family' in plant else plant['secondary_panel'],scenario_id=scenario['scenario_id'],
        source_id=scenario['source_id'],controller_seed=seed,controller_variant=variant,
        plant_checkpoint_sha256=plant['sha256'],initial_Tp_C=scenario['initial_Tp_C'],
        previous_uh_percent=scenario['previous_uh_percent'],runtime_s=runtime)
    arrays=dict(metadata=np.array(json.dumps(metadata,ensure_ascii=False)),
        step=np.arange(60,dtype=np.int16),reference=np.array([r['reference_c'] for r in trace]),
        plant_Tp=np.array([r['plant_tp'] for r in trace]),
        plant_output=np.stack([r['plant_output'] for r in trace]),
        applied_UH=np.array([r['command_percent'] for r in trace]),
        solver_success=np.array([r['success'] for r in trace]),
        solver_iterations=np.array([r['iterations'] for r in trace]),
        solver_evaluations=np.array([r['evaluations'] for r in trace]),
        internal_predicted_Tp=np.stack([r['internal_prediction'][:,6] if r['internal_prediction'] is not None else np.full(12,np.nan) for r in trace]),
        optimal_sequence=np.stack([r['optimal'] if r['optimal'] is not None else np.full(12,np.nan) for r in trace]),
        decision_history=np.stack([r['decision_history'] if r['decision_history'] is not None else np.full((20,12),np.nan) for r in trace]),
        objective=np.array([r['objective'] if r['objective'] is not None else np.nan for r in trace]),
        solver_message=np.array([r['solver_message'] for r in trace]),
        time_per_control_step_s=np.array([r['time_per_control_step_s'] for r in trace]),
        background_raw=background[:60])
    temp=output_path.with_suffix('.tmp.npz')
    np.savez_compressed(temp,**arrays)
    temp.replace(output_path)
    return runtime


def execute(execution_dir):
    execution=Path(execution_dir).resolve()
    manifest,protocol=verify(execution)
    assert manifest['protocol_mutable'] is False
    torch.set_num_threads(2)
    scenarios,histories=load_scenarios(protocol)
    plants,tasks=schedule()
    trace_dir=execution/'raw_runs'
    trace_dir.mkdir(exist_ok=True)
    progress_path=execution/'CONTROL_EXECUTION_AUDIT.json'
    if progress_path.is_file():
        progress=read(progress_path)
        assert progress['protocol_sha256']==manifest['protocol_sha256']
        progress['resume_count']+=1
        progress.setdefault('resume_audit',[]).append(dict(at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),reason='exact same protocol technical resume'))
    else:
        progress=dict(status='RUNNING',execution_id=manifest['execution_id'],protocol_sha256=manifest['protocol_sha256'],
            total_runs=len(tasks),completed_runs=0,resume_count=0,resume_audit=[],
            control_outcome_accessed=False,no_training=True,no_tuning=True,no_plant_deletion=True,no_scenario_deletion=True,no_rescue=True)
    atomic_json(progress_path,progress)
    controllers={}
    for seed in SEEDS:
        for variant in VARIANTS:
            binding=protocol['controller_models']['checkpoint_bindings'][f'{seed}:{variant}']
            controllers[(seed,variant)]=load_model(binding,'B4')
    progress['controller_checkpoints_loaded']=6
    panel=read(PROTOCOLS/'CONTROL_PLANT_FAMILY_LOCK.json')
    try:
        for pi,plant in enumerate(plants):
            remaining=[(index,task) for index,task in enumerate(tasks) if task['plant_index']==pi and not (trace_dir/f'{index:04d}.npz').is_file()]
            if not remaining:
                continue
            binding={'path':plant['checkpoint_path'],'sha256':plant['sha256'],
                **({k:plant[k] for k in ('torchscript_path','torchscript_sha256')} if plant['model_id']=='V10' else {})}
            plant_model=load_model(binding,plant['model_id'])
            for index,task in remaining:
                file=trace_dir/f'{index:04d}.npz'
                run_one(task,plants,scenarios,histories,controllers,plant_model,file,manifest['execution_id'])
                progress['completed_runs']=sum((trace_dir/f'{i:04d}.npz').is_file() for i in range(len(tasks)))
                progress['control_outcome_accessed']=True
                progress['last_completed_task_index']=index
                atomic_json(progress_path,progress)
                if progress['completed_runs']%12==0:
                    print(f'completed {progress["completed_runs"]}/{len(tasks)} runs',flush=True)
            del plant_model
        assert progress['completed_runs']==len(tasks)
        progress['status']='RAW_RUNS_COMPLETE'
        atomic_json(progress_path,progress)
    except BaseException as exc:
        progress['status']='INCOMPLETE'
        progress['technical_failure']=dict(type=type(exc).__name__,message=str(exc),task_index=locals().get('index'))
        atomic_json(progress_path,progress)
        raise
    print(json.dumps({'status':progress['status'],'completed_runs':progress['completed_runs']}))


if __name__=='__main__':
    if len(sys.argv)!=2:
        raise SystemExit('usage: run_formal_control_v3a.py EXECUTION_DIR')
    execute(sys.argv[1])
