"""Post-hoc MIL runner: gates G4/G6, freeze, and author-authorized execution.

usage:
  run_posthoc_mil.py gates              G4 reuse reproduction + G6 timing smoke (outcome-blind)
  run_posthoc_mil.py freeze             bind hashes into POSTHOC_MIL_PROTOCOL.json (no runs)
  run_posthoc_mil.py execute [WORKERS]  requires EXECUTION_AUTHORIZATION.json from the author
  run_posthoc_mil.py worker EXEC_DIR INDEX COUNT
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from _paths import PKG as HERE, PROTOCOLS, OUTPUT_ROOT, VERIFY_HASHES, private_path  # noqa: E402
from de_mpc import DisturbanceEstimatingMPC, MHEConfig, run_closed_loop_trace  # noqa: E402
import run_formal_control_v3a as rf  # noqa: E402

PYTHON = sys.executable
CONTROL_DIR = PROTOCOLS
MIL_OUT = OUTPUT_ROOT/'posthoc_mil'
DRAFT = PROTOCOLS/'POSTHOC_MIL_PROTOCOL_DRAFT.json'
FROZEN = PROTOCOLS/'POSTHOC_MIL_PROTOCOL.json'
FROZEN_SHA = PROTOCOLS/'POSTHOC_MIL_PROTOCOL.sha256'
AUTHORIZATION = MIL_OUT/'EXECUTION_AUTHORIZATION.json'
TEST_REPORT = PROTOCOLS/'POSTHOC_MIL_TEST_REPORT.json'
GATE_REPORT = PROTOCOLS/'POSTHOC_MIL_GATE_REPORT.json'
ANALYSIS_CHECK = PROTOCOLS/'POSTHOC_MIL_ANALYSIS_REPRODUCTION_CHECK.json'
SELFTEST = PROTOCOLS/'POSTHOC_MIL_ANALYSIS_SELFTEST.json'
FORMAL_PROTOCOL = PROTOCOLS/'CONTROL_VALIDATION_PROTOCOL_V3.json'
FORMAL_EXECUTION = OUTPUT_ROOT/'control'/'control_v3a_20260926T153411234230Z_0b9ad32c21'
SEEDS = rf.SEEDS
PRIMARY_PLANTS = 24
NEW_VARIANTS = ('L0', 'R0-DE', 'COISC-DE', 'L0-DE')
RERUN_VARIANTS = ('R0', 'COISC')
CODE_FILES = ('de_mpc.py', 'run_posthoc_mil.py', 'analyze_posthoc_mil.py', 'test_de_mpc.py')
UPDATES = 60


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def active_protocol():
    return read(FROZEN) if FROZEN.is_file() else read(DRAFT)


def load_controller_models(variants=('R0', 'COISC', 'L0'), seeds=SEEDS):
    formal = read(FORMAL_PROTOCOL)['controller_models']['checkpoint_bindings']
    l0 = active_protocol()['controller_models']['L0_bindings']
    models = {}
    for seed in seeds:
        for variant in variants:
            binding = l0[str(seed)] if variant == 'L0' else formal[f'{seed}:{variant}']
            models[(seed, variant)] = rf.load_model(binding, 'B4')
    return models


def plant_binding(plant):
    binding = {'path': plant['checkpoint_path'], 'sha256': plant['sha256']}
    if plant['model_id'] == 'V10':
        binding.update({k: plant[k] for k in ('torchscript_path', 'torchscript_sha256')})
    return binding


def plants_and_formal_index():
    plants, formal_tasks = rf.schedule()
    lock = read(CONTROL_DIR/'CONTROL_PLANT_FAMILY_LOCK.json')
    assert [p['plant_name'] for p in plants[:PRIMARY_PLANTS]] == [p['plant_name'] for p in lock['primary_checkpoints']]
    assert all('family' in p for p in plants[:PRIMARY_PLANTS])
    index = {(t['plant_index'], t['scenario_index'], t['seed'], t['variant']): i for i, t in enumerate(formal_tasks)}
    return plants, index


def task_list(rerun_formal_nominal):
    variants = NEW_VARIANTS + (RERUN_VARIANTS if rerun_formal_nominal else ())
    tasks = [dict(plant_index=pi, scenario_index=si, seed=seed, variant=v)
             for pi in range(PRIMARY_PLANTS) for si in range(12) for seed in SEEDS for v in variants]
    assert len(tasks) == PRIMARY_PLANTS*12*3*len(variants)
    return tasks


def run_de_one(task, plants, scenarios, histories, models, plant_model, output_path, execution_id, mhe=MHEConfig()):
    plant = plants[task['plant_index']]
    scenario = scenarios[task['scenario_index']]
    history = histories[task['scenario_index']]
    seed = task['seed']
    base_variant = task['variant'][:-3]
    assert task['variant'].endswith('-DE') and base_variant in ('R0', 'COISC', 'L0')
    controller = DisturbanceEstimatingMPC(models[(seed, base_variant)], mhe_config=mhe)
    reference = np.full(UPDATES+12-1, 37., dtype=np.float64)
    background = np.broadcast_to(np.asarray(scenario['background_raw_aux'], dtype=np.float64), (71, 5)).copy()
    start = time.perf_counter()
    trace = run_closed_loop_trace(plant_model, controller, history, reference, background, UPDATES)
    runtime = time.perf_counter()-start
    assert len(trace) == UPDATES
    est = [r['estimator'] for r in trace]
    metadata = dict(execution_id=execution_id, plant=plant['plant_name'], family=plant.get('family'),
                    panel='primary', scenario_id=scenario['scenario_id'], source_id=scenario['source_id'],
                    controller_seed=seed, controller_variant=task['variant'],
                    plant_checkpoint_sha256=plant['sha256'], initial_Tp_C=scenario['initial_Tp_C'],
                    previous_uh_percent=scenario['previous_uh_percent'], runtime_s=runtime,
                    mhe_config=dict(window=mhe.window, rho=mhe.rho, lower=mhe.lower, upper=mhe.upper, xatol=mhe.xatol))
    if output_path is None:
        return runtime, trace
    arrays = dict(metadata=np.array(json.dumps(metadata, ensure_ascii=False)),
        step=np.arange(UPDATES, dtype=np.int16), reference=np.array([r['reference_c'] for r in trace]),
        plant_Tp=np.array([r['plant_tp'] for r in trace]),
        plant_output=np.stack([r['plant_output'] for r in trace]),
        applied_UH=np.array([r['command_percent'] for r in trace]),
        solver_success=np.array([r['success'] for r in trace]),
        solver_iterations=np.array([r['iterations'] for r in trace]),
        solver_evaluations=np.array([r['evaluations'] for r in trace]),
        internal_predicted_Tp=np.stack([r['internal_prediction'][:, 6] if r['internal_prediction'] is not None else np.full(12, np.nan) for r in trace]),
        optimal_sequence=np.stack([r['optimal'] if r['optimal'] is not None else np.full(12, np.nan) for r in trace]),
        decision_history=np.stack([r['decision_history'] for r in trace]),
        objective=np.array([r['objective'] if r['objective'] is not None else np.nan for r in trace]),
        solver_message=np.array([r['solver_message'] for r in trace]),
        time_per_control_step_s=np.array([r['time_per_control_step_s'] for r in trace]),
        background_raw=background[:UPDATES],
        d_hat=np.array([e['d_hat'] for e in est]),
        estimator_status=np.array([e['status'] for e in est]),
        estimator_cost=np.array([e['cost'] for e in est], dtype=np.float64),
        estimator_cost_at_zero=np.array([e['cost_at_zero'] for e in est], dtype=np.float64),
        estimator_evaluations=np.array([e['evaluations'] for e in est], dtype=np.int32),
        estimator_window_start=np.array([-1 if e['window_start'] is None else e['window_start'] for e in est], dtype=np.int32),
        estimator_time_s=np.array([e['estimator_time_s'] for e in est]))
    temp = output_path.with_suffix('.tmp.npz')
    np.savez_compressed(temp, **arrays)
    temp.replace(output_path)
    return runtime, trace


def run_task(task, plants, scenarios, histories, models, plant_model, output_path, execution_id):
    if task['variant'].endswith('-DE'):
        return run_de_one(task, plants, scenarios, histories, models, plant_model, output_path, execution_id)[0]
    return rf.run_one(task, plants, scenarios, histories, models, plant_model, output_path, execution_id)


def gates():
    torch.set_num_threads(2)
    report = dict(created_utc=utc(), outcome_blind=True,
                  note='G4 reruns 48 already-reported formal nominal runs; G6 uses the V10 plant outside the analysis population.')
    scenarios, histories = rf.load_scenarios(read(FORMAL_PROTOCOL))
    plants, formal_index = plants_and_formal_index()
    models = load_controller_models()
    gate_dir = MIL_OUT/'gate_runs'
    (gate_dir/'G4').mkdir(parents=True, exist_ok=True)

    g4_rows = []
    for pi in range(PRIMARY_PLANTS):
        plant_model = rf.load_model(plant_binding(plants[pi]), plants[pi]['model_id'])
        si = pi % 12
        for variant in RERUN_VARIANTS:
            task = dict(plant_index=pi, scenario_index=si, seed=20260923, variant=variant)
            out = gate_dir/'G4'/f'p{pi:02d}_s{si:02d}_{variant}.npz'
            runtime = rf.run_one(task, plants, scenarios, histories, models, plant_model, out, 'posthoc_mil_gate_G4')
            formal_file = FORMAL_EXECUTION/'raw_runs'/f'{formal_index[(pi, si, 20260923, variant)]:04d}.npz'
            with np.load(out, allow_pickle=False) as new, np.load(formal_file, allow_pickle=False) as old:
                equal_u = bool(np.array_equal(new['applied_UH'], old['applied_UH']))
                equal_y = bool(np.array_equal(new['plant_Tp'], old['plant_Tp']))
            g4_rows.append(dict(plant=plants[pi]['plant_name'], scenario_index=si, variant=variant,
                                formal_file=formal_file.name, applied_UH_bitwise=equal_u, plant_Tp_bitwise=equal_y,
                                runtime_s=runtime))
            print(f'G4 {pi:02d} {variant}: {equal_u and equal_y} ({runtime:.1f} s)', flush=True)
        del plant_model
    g4_pass = all(r['applied_UH_bitwise'] and r['plant_Tp_bitwise'] for r in g4_rows)
    report['G4_reuse_reproduction'] = dict(passed=g4_pass, runs=len(g4_rows), rows=g4_rows,
        consequence='reuse formal R0/COISC primary raw runs' if g4_pass else 'rerun 1728 R0/COISC primary nominal runs inside the execution')

    v10_index = next(i for i, p in enumerate(plants) if p['model_id'] == 'V10')
    v10 = rf.load_model(plant_binding(plants[v10_index]), 'V10')
    g6_rows = []
    for variant, si in (('R0-DE', 0), ('COISC-DE', 1)):
        task = dict(plant_index=v10_index, scenario_index=si, seed=20260922, variant=variant)
        runtime, trace = run_de_one(task, plants, scenarios, histories, models, v10, None, 'posthoc_mil_gate_G6')
        est_time = float(sum(r['estimator']['estimator_time_s'] for r in trace))
        g6_rows.append(dict(variant=variant, scenario_index=si, runtime_s=runtime, estimator_time_s=est_time))
        print(f'G6 {variant}: {runtime:.1f} s (estimator {est_time:.1f} s)', flush=True)
    de_mean = float(np.mean([r['runtime_s'] for r in g6_rows]))
    nominal_mean = float(np.mean([r['runtime_s'] for r in g4_rows]))
    new_runs = 864 + 2592 + (0 if g4_pass else 1728)
    serial_h = (864*nominal_mean + 2592*de_mean + (0 if g4_pass else 1728*nominal_mean))/3600
    report['G6_timing_smoke'] = dict(passed=True, results_discarded=True, rows=g6_rows,
        mean_de_runtime_s=de_mean, mean_nominal_runtime_s_from_G4=nominal_mean, new_runs=new_runs,
        estimated_serial_hours=serial_h, workers=6, threads_per_worker=2,
        estimated_wall_hours_6_workers=serial_h/6)
    report['all_passed'] = g4_pass
    GATE_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'G4': g4_pass, 'G6_mean_de_runtime_s': de_mean, 'est_wall_h_6w': serial_h/6}))


def bound_files():
    formal_manifest = read(FORMAL_EXECUTION/'CONTROL_EXECUTION_MANIFEST.json')
    files = [FORMAL_PROTOCOL, FORMAL_EXECUTION/'CONTROL_EXECUTION_MANIFEST.json',
             FORMAL_EXECUTION/'CONTROL_BOOTSTRAP_DRAWS.npz', FORMAL_EXECUTION/'CONTROL_PRIMARY_RESULTS.csv',
             HERE/'control_mpc.py', HERE/'run_formal_control_v3a.py',
             HERE/'analyze_formal_control_v3.py', HERE/'run_formal_control_v3.py',
             CONTROL_DIR/'CONTROL_SCENARIO_LOCK_V2.json', CONTROL_DIR/'CONTROL_PLANT_FAMILY_LOCK.json',
             CONTROL_DIR/'CONTROL_PLANT_REACHABILITY_AUDIT.csv', CONTROL_DIR/'CONTROL_REACHABILITY_STRATA_LOCK.csv',
             DRAFT, TEST_REPORT, GATE_REPORT, ANALYSIS_CHECK, SELFTEST]
    files += [private_path(b['path']) for b in formal_manifest['bound_files']]
    files += [HERE/name for name in CODE_FILES]
    unique = []
    for f in files:
        if str(Path(f).resolve()) not in {str(Path(u).resolve()) for u in unique}:
            unique.append(Path(f))
    return [dict(path=str(f), sha256=sha(f)) for f in unique]


def freeze():
    assert not FROZEN.is_file(), 'protocol already frozen'
    tests = read(TEST_REPORT)
    gate = read(GATE_REPORT)
    check = read(ANALYSIS_CHECK)
    assert tests['all_passed'], 'G1/G2/G3/G5 not all passed: execution cannot be authorized'
    assert gate['G6_timing_smoke']['passed']
    assert check['passed'], 'analysis pipeline does not reproduce the formal primary result'
    assert read(SELFTEST)['selftest_passed'], 'analysis selftest failed'
    protocol = read(DRAFT)
    protocol.update(
        version='FROZEN-1 (2026-09-29)', status='FROZEN_BEFORE_POSTHOC_MIL_OUTCOME', execution_authorized=False,
        execution_authorization_rule='execute requires EXECUTION_AUTHORIZATION.json written on the author\'s explicit instruction, naming this protocol SHA-256',
        author_decisions_2026_09_29={
            'population': 'APPROVED: primary panel only (24 x 12 x 3)',
            'de_parameters': 'APPROVED: W = 12, rho = 0.01, d bounds +-50 %, xatol 0.05',
            'reuse_formal_R0_COISC': 'APPROVED: reuse, subject to G4',
            'seed23_attribution_rule': 'APPROVED: if A1 holds only on some seeds, report partial attribution only'},
        g4_outcome=dict(passed=gate['G4_reuse_reproduction']['passed'],
                        rerun_formal_nominal=not gate['G4_reuse_reproduction']['passed']),
        gate_summary=dict(G1=tests['G1']['passed'], G2=tests['G2']['passed'], G3=tests['G3']['passed'],
                          G5=tests['G5']['passed'], G4=gate['G4_reuse_reproduction']['passed'],
                          G6_estimated_wall_hours_6_workers=gate['G6_timing_smoke']['estimated_wall_hours_6_workers'],
                          analysis_reproduces_formal_primary=check['passed'], analysis_selftest=True),
        frozen_utc=utc(), bound_files=bound_files())
    estimator = protocol['arm_B_disturbance_estimating_mpc']['estimator']
    estimator['solver_draft_1'] = estimator.pop('solver')
    estimator['solver'] = ('global grid d in {-50, -49, ..., 50} (101 batched rollouts), then '
                           "scipy.optimize.minimize_scalar(method='bounded', options={'xatol': 0.05}) on "
                           '[g-1, g+1] clipped to the bounds, where g is the grid minimiser; the returned d is the '
                           'lowest-cost point among {grid minimiser, refined point, d = 0} (ties broken by smaller |d|)')
    estimator['parameter_status'] += ' The grid step (1 %) is likewise fixed a priori.'
    protocol['pre_freeze_amendments'] = [dict(
        id='AM1', gate='G2', outcome_blind=True,
        observed=('With the DRAFT-1 solver (bounded Brent on [-50, 50] only), G2 FAILED: on the matched plant d_hat drifted '
                  'to about -2.7 % although J(0) = 0 exactly. Cause: the clip of effective UH at 0 % makes J(d) '
                  'non-unimodal (plateaus for d below minus the applied UH), and Brent converged to a non-global point.'),
        change='Global 1 % grid before local refinement, plus d = 0 as a candidate. W, rho, bounds and xatol unchanged.',
        after_change='G1, G2, G3, G5 all PASS (POSTHOC_MIL_TEST_REPORT.json).',
        no_population_outcome_seen=True)]
    protocol['code_to_be_written_and_bound_at_freeze']['sha256'] = {n: sha(HERE/n) for n in CODE_FILES}
    FROZEN.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding='utf-8')
    digest = sha(FROZEN)
    FROZEN_SHA.write_text(f'{digest}  {FROZEN.name}\n', encoding='utf-8')
    print(json.dumps({'frozen': str(FROZEN), 'sha256': digest}))


def verify_frozen():
    protocol = read(FROZEN)
    digest = sha(FROZEN)
    assert not VERIFY_HASHES or FROZEN_SHA.read_text(encoding='utf-8').split()[0] == digest
    assert protocol['status'] == 'FROZEN_BEFORE_POSTHOC_MIL_OUTCOME'
    for b in protocol['bound_files']:
        assert not VERIFY_HASHES or sha(private_path(b['path'])) == b['sha256'], f'G0 hash mismatch: {b["path"]}'
    return protocol, digest


def execute(workers=6):
    protocol, digest = verify_frozen()
    auth = read(AUTHORIZATION)
    assert auth.get('authorized_by_author') is True and auth.get('protocol_sha256') == digest, 'no valid author authorization'
    existing = sorted(MIL_OUT.glob('execution_*'))
    execution = existing[-1] if existing else MIL_OUT/f'execution_{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}_{digest[:10]}'
    execution.mkdir(parents=True, exist_ok=True)
    (execution/'raw_runs').mkdir(exist_ok=True)
    audit_path = execution/'POSTHOC_MIL_EXECUTION_AUDIT.json'
    tasks = task_list(protocol['g4_outcome']['rerun_formal_nominal'])
    if audit_path.is_file():
        audit = read(audit_path)
        assert audit['protocol_sha256'] == digest
        audit['resume_log'].append(dict(at_utc=utc(), reason='same-protocol technical resume'))
    else:
        audit = dict(status='RUNNING', protocol_sha256=digest, execution_id=execution.name, total_runs=len(tasks),
                     authorization=auth, started_utc=utc(), resume_log=[], workers=workers, threads_per_worker=2,
                     no_tuning=True, no_plant_deletion=True, no_scenario_deletion=True)
    (execution/'TASKS.json').write_text(json.dumps(tasks), encoding='utf-8')
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    procs = []
    for w in range(workers):
        log = (execution/f'worker_{w}.log').open('a', encoding='utf-8')
        procs.append(subprocess.Popen([PYTHON, str(Path(__file__).resolve()), 'worker', str(execution), str(w), str(workers)],
                                      stdout=log, stderr=subprocess.STDOUT, env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}))
    codes = [p.wait() for p in procs]
    done = sum((execution/'raw_runs'/f'{i:04d}.npz').is_file() for i in range(len(tasks)))
    audit.update(worker_exit_codes=codes, completed_runs=done, finished_utc=utc(),
                 status='RAW_RUNS_COMPLETE' if done == len(tasks) and not any(codes) else 'INCOMPLETE',
                 worker_audits=[read(p) for p in sorted(execution.glob('worker_*_audit.json'))])
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'status': audit['status'], 'completed': done, 'total': len(tasks)}))


def worker(execution_dir, index, count):
    execution = Path(execution_dir)
    protocol, digest = verify_frozen()
    assert read(execution/'POSTHOC_MIL_EXECUTION_AUDIT.json')['protocol_sha256'] == digest
    torch.set_num_threads(2)
    tasks = json.loads((execution/'TASKS.json').read_text(encoding='utf-8'))
    assert tasks == task_list(protocol['g4_outcome']['rerun_formal_nominal'])
    scenarios, histories = rf.load_scenarios(read(FORMAL_PROTOCOL))
    plants, _ = plants_and_formal_index()
    models = load_controller_models()
    audit_path = execution/f'worker_{index}_audit.json'
    audit = dict(worker=index, count=count, completed=[], failures=[], started_utc=utc())
    for pi in range(index, PRIMARY_PLANTS, count):
        remaining = [(i, t) for i, t in enumerate(tasks) if t['plant_index'] == pi
                     and not (execution/'raw_runs'/f'{i:04d}.npz').is_file()]
        if not remaining:
            continue
        plant_model = rf.load_model(plant_binding(plants[pi]), plants[pi]['model_id'])
        for i, task in remaining:
            runtime = run_task(task, plants, scenarios, histories, models, plant_model,
                               execution/'raw_runs'/f'{i:04d}.npz', execution.name)
            audit['completed'].append(dict(task_index=i, runtime_s=runtime))
            audit_path.write_text(json.dumps(audit), encoding='utf-8')
            print(f'{utc()} task {i} plant {pi} {task["variant"]} {runtime:.1f}s', flush=True)
        del plant_model
    audit['finished_utc'] = utc()
    audit_path.write_text(json.dumps(audit), encoding='utf-8')


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else ''
    if command == 'gates':
        gates()
    elif command == 'freeze':
        freeze()
    elif command == 'execute':
        execute(int(sys.argv[2]) if len(sys.argv) > 2 else 6)
    elif command == 'worker':
        worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        raise SystemExit(__doc__)
