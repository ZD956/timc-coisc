"""MIL runner for the post-hoc onset-aligned study (POSTHOC_ONSET_ALIGNED_PROTOCOL).

usage:
  onset_mil.py gates                 R5 runner reproduction + guard unit test (existing models and already-reported runs only)
  onset_mil.py execute [WORKERS]     frozen protocol + author's EXECUTION_AUTHORIZATION.json required
  onset_mil.py worker EXEC_DIR INDEX COUNT
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

from _paths import PKG as HERE, PROTOCOLS, OUTPUT_ROOT, VERIFY_HASHES  # noqa: E402
import run_formal_control_v3a as rf  # noqa: E402
from control_mpc import MPCConfig, MPCController, apply_plant_step, quantize_command  # noqa: E402

PYTHON = sys.executable
CONTROL_DIR = PROTOCOLS
OUT = OUTPUT_ROOT / 'posthoc_onset_aligned'
PROTO = PROTOCOLS / 'posthoc_onset_aligned'
FORMAL_PROTOCOL = PROTOCOLS / 'CONTROL_VALIDATION_PROTOCOL_V3.json'
FORMAL_EXECUTION = OUTPUT_ROOT / 'control' / 'control_v3a_20260926T153411234230Z_0b9ad32c21'
MIL_PROTOCOL = PROTOCOLS / 'POSTHOC_MIL_PROTOCOL.json'
L0_EXECUTION = OUTPUT_ROOT / 'posthoc_mil' / 'execution_20261001T050813Z_a3bd1ebd55'
PROTOCOL = PROTO / 'POSTHOC_ONSET_ALIGNED_PROTOCOL.json'
PROTOCOL_SHA_FILE = PROTO / 'POSTHOC_ONSET_ALIGNED_PROTOCOL.sha256'
AUTHORIZATION = OUT / 'EXECUTION_AUTHORIZATION.json'
GATE_REPORT = PROTO / 'GATES_MIL_REPORT.json'
E1_SEEDS = (20260922, 20260923, 20260924)
E2_SEEDS = tuple(range(20261001, 20261011))
PRIMARY_PLANTS = 24
UPDATES = 60
SCENARIO_LABELS = tuple(f'S{i + 1:02d}' for i in range(12))
GUARD_ID_PLANTS = (0, 12)
FIRST_ACTION_MODELS = {**{s: ('R0', 'COISC', 'L0', 'COISC_onset') for s in E1_SEEDS},
                       **{s: ('R0', 'COISC', 'COISC_onset', 'L0') for s in E2_SEEDS}}


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


def frozen_protocol():
    digest = sha(PROTOCOL)
    assert not VERIFY_HASHES or PROTOCOL_SHA_FILE.read_text(encoding='utf-8').split()[0] == digest, 'protocol changed after freeze'
    protocol = read(PROTOCOL)
    assert protocol['status'].startswith('FROZEN'), 'protocol not frozen'
    for name, h in protocol['code_sha256'].items():
        assert not VERIFY_HASHES or sha(HERE / name) == h, f'code changed after freeze: {name}'
    auth = read(AUTHORIZATION)
    assert auth.get('authorized_by_author') is True and auth.get('protocol_sha256') == digest, 'no valid author authorization'
    return protocol, digest


def plant_binding(plant):
    binding = {'path': plant['checkpoint_path'], 'sha256': plant['sha256']}
    if plant['model_id'] == 'V10':
        binding.update({k: plant[k] for k in ('torchscript_path', 'torchscript_sha256')})
    return binding


def plants_and_formal_index():
    plants, formal_tasks = rf.schedule()
    lock = read(CONTROL_DIR / 'CONTROL_PLANT_FAMILY_LOCK.json')
    assert [p['plant_name'] for p in plants[:PRIMARY_PLANTS]] == [p['plant_name'] for p in lock['primary_checkpoints']]
    index = {(t['plant_index'], t['scenario_index'], t['seed'], t['variant']): i for i, t in enumerate(formal_tasks)}
    return plants[:PRIMARY_PLANTS], index


def l0_index():
    tasks = read(L0_EXECUTION / 'TASKS.json')
    return {(t['plant_index'], t['scenario_index'], t['seed']): i for i, t in enumerate(tasks) if t['variant'] == 'L0'}


def existing_bindings():
    formal = read(FORMAL_PROTOCOL)['controller_models']['checkpoint_bindings']
    l0 = read(MIL_PROTOCOL)['controller_models']['L0_bindings']
    out = {}
    for seed in E1_SEEDS:
        out[(seed, 'R0')] = formal[f'{seed}:R0']
        out[(seed, 'COISC')] = formal[f'{seed}:COISC']
        out[(seed, 'L0')] = l0[str(seed)]
    return out


def all_bindings(digest):
    out = existing_bindings()
    for mode in ('e1', 'e2'):
        record = read(OUT / mode / f'{mode.upper()}_CHECKPOINT_BINDINGS.json')
        assert record['protocol_sha256'] == digest, f'{mode} checkpoints not produced under this protocol'
        for b in record['bindings']:
            key = (int(b['seed']), b['variant'])
            if key in out:
                assert out[key]['sha256'] == b['sha256'], f'binding conflict {key}'
            else:
                out[key] = dict(path=b['path'], sha256=b['sha256'])
    return out


def load_models(bindings, keys):
    return {key: rf.load_model(bindings[key], 'B4') for key in keys}


def runner_previous(history):
    return float(np.clip(float(torch.as_tensor(history, dtype=torch.float32)[-1, 7]) / 5., 0., 100.))


def want_sign(initial_tp):
    return 1 if float(initial_tp) < 37. else -1


def move_sign(command, previous):
    return int(np.sign(round(float(command) - float(previous), 9)))


def guard_decision(command, previous, initial_tp):
    """Hold the previous command if the first move is strictly opposite to the side of 37 C."""
    if move_sign(command, previous) == -want_sign(initial_tp):
        return quantize_command(previous), True
    return command, False


def first_decision(model, scenario, history):
    controller = MPCController(model, MPCConfig())
    background = np.broadcast_to(np.asarray(scenario['background_raw_aux'], dtype=np.float64), (71, 5)).copy()
    reference = np.full(UPDATES + 12 - 1, 37., dtype=np.float64)
    plant_history = torch.as_tensor(history, dtype=torch.float32).clone()
    decision = controller.decide(plant_history, reference[:12], background[:12], runner_previous(history))
    return decision['command_percent'], bool(decision['success'])


def first_action_class(u0, scenario):
    s = move_sign(u0, scenario['previous_uh_percent'])
    want = want_sign(scenario['initial_Tp_C'])
    return 'correct' if s == want else ('wrong' if s == -want else 'zero')


def run_one(task, plants, scenarios, histories, models, bindings, plant_model, output_path, execution_id):
    """rf.run_one with an optional first-step direction guard; metadata carries no source or scenario identifiers."""
    plant = plants[task['plant_index']]
    scenario = scenarios[task['scenario_index']]
    history = histories[task['scenario_index']]
    seed = task['seed']
    variant = task['variant']
    guard = variant in ('R0_GUARD', 'R0_GUARD_ID')
    model_key = (seed, 'R0' if guard else variant)
    controller = MPCController(models[model_key], MPCConfig())
    reference = np.full(60 + 12 - 1, 37., dtype=np.float64)
    background = np.broadcast_to(np.asarray(scenario['background_raw_aux'], dtype=np.float64), (71, 5)).copy()
    start = time.perf_counter()
    plant_history = torch.as_tensor(history, dtype=torch.float32).clone()
    previous = float(np.clip(float(plant_history[-1, 7]) / 5., 0., 100.))
    trace = []
    guard_fired = False
    for step in range(60):
        before_history = plant_history.numpy().copy()
        step_started = time.perf_counter()
        decision = controller.decide(plant_history, reference[step:step + 12], background[step:step + 12], previous)
        command = decision['command_percent']
        mpc_command = command
        if guard and step == 0:
            command, guard_fired = guard_decision(command, previous, scenario['initial_Tp_C'])
        plant_history, output = apply_plant_step(plant_model, plant_history, command, background[step])
        trace.append(dict(plant_tp=float(output[6]), plant_output=output.numpy().copy(), command_percent=command,
                          mpc_command_percent=mpc_command, success=decision['success'], iterations=decision['iterations'],
                          evaluations=decision['evaluations'], objective=decision['objective'], decision_history=before_history,
                          internal_prediction=decision.get('predicted_trajectory'), optimal=decision.get('optimal_percent'),
                          solver_message=decision['message'], time_per_control_step_s=time.perf_counter() - step_started))
        previous = command
    runtime = time.perf_counter() - start
    metadata = dict(execution_id=execution_id, plant_index=task['plant_index'], family=plant['family'], panel='primary',
                    scenario_index=task['scenario_index'], scenario_label=SCENARIO_LABELS[task['scenario_index']],
                    controller_seed=seed, controller_variant=variant, controller_checkpoint_sha256=bindings[model_key]['sha256'],
                    plant_checkpoint_sha256=plant['sha256'], initial_Tp_C=scenario['initial_Tp_C'],
                    previous_uh_percent=scenario['previous_uh_percent'], runtime_s=runtime, guard=guard, guard_fired=guard_fired)
    arrays = dict(metadata=np.array(json.dumps(metadata)), step=np.arange(60, dtype=np.int16),
                  reference=reference[:60].copy(), plant_Tp=np.array([r['plant_tp'] for r in trace]),
                  plant_output=np.stack([r['plant_output'] for r in trace]),
                  applied_UH=np.array([r['command_percent'] for r in trace]),
                  mpc_UH=np.array([r['mpc_command_percent'] for r in trace]),
                  solver_success=np.array([r['success'] for r in trace]),
                  solver_iterations=np.array([r['iterations'] for r in trace]),
                  solver_evaluations=np.array([r['evaluations'] for r in trace]),
                  internal_predicted_Tp=np.stack([r['internal_prediction'][:, 6] if r['internal_prediction'] is not None else np.full(12, np.nan) for r in trace]),
                  optimal_sequence=np.stack([r['optimal'] if r['optimal'] is not None else np.full(12, np.nan) for r in trace]),
                  objective=np.array([r['objective'] if r['objective'] is not None else np.nan for r in trace]),
                  solver_message=np.array([r['solver_message'] for r in trace]),
                  time_per_control_step_s=np.array([r['time_per_control_step_s'] for r in trace]),
                  background_raw=background[:60])
    temp = Path(output_path).with_suffix('.tmp.npz')
    np.savez_compressed(temp, **arrays)
    temp.replace(output_path)
    return runtime


def bitwise(new_path, old_path):
    with np.load(new_path, allow_pickle=False) as new, np.load(old_path, allow_pickle=False) as old:
        return bool(np.array_equal(new['applied_UH'], old['applied_UH'])), bool(np.array_equal(new['plant_Tp'], old['plant_Tp']))


def first_u(path):
    with np.load(path, allow_pickle=False) as z:
        return float(z['applied_UH'][0])


def guard_unit_test():
    cases = [  # (mpc command, previous, initial T_p, expected applied, expected fired)
        (40.0, 50.0, 35.0, 50.0, True), (60.0, 50.0, 35.0, 60.0, False), (50.0, 50.0, 35.0, 50.0, False),
        (60.0, 50.0, 39.0, 50.0, True), (40.0, 50.0, 39.0, 40.0, False), (50.0, 50.0, 39.0, 50.0, False),
        (0.0, 12.34, 36.9, 12.3, True), (100.0, 99.96, 38.0, 100.0, True), (50.0000000001, 50.0, 39.0, 50.0000000001, False)]
    rows = []
    for command, previous, tp, expected, fired in cases:
        got, got_fired = guard_decision(command, previous, tp)
        rows.append(dict(command=command, previous=previous, initial_Tp=tp, applied=got, fired=got_fired,
                         ok=bool(got == expected and got_fired == fired)))
    return dict(cases=rows, pass_=all(r['ok'] for r in rows))


def gates():
    torch.set_num_threads(2)
    report = dict(created_utc=utc(), outcome_blind=True,
                  note='R5 re-executes 12 already-reported formal R0 runs and recomputes already-reported first actions of existing R0/COISC/L0.')
    scenarios, histories = rf.load_scenarios(read(FORMAL_PROTOCOL))
    plants, formal_index = plants_and_formal_index()
    l0_idx = l0_index()
    bindings = existing_bindings()
    models = load_models(bindings, list(bindings))
    gate_dir = OUT / 'gates' / 'R5_mil'
    gate_dir.mkdir(parents=True, exist_ok=True)

    prev_rows = [dict(scenario=SCENARIO_LABELS[si], abs_diff=abs(runner_previous(histories[si]) - float(s['previous_uh_percent'])),
                      exact=runner_previous(histories[si]) == float(s['previous_uh_percent'])) for si, s in enumerate(scenarios)]
    report['R5_previous_consistency'] = dict(rows=prev_rows, max_abs_diff=max(r['abs_diff'] for r in prev_rows),
                                             pass_=all(r['exact'] for r in prev_rows))

    fa_rows, mismatches = [], 0
    for (seed, variant), model in models.items():
        for si, scenario in enumerate(scenarios):
            u0, ok = first_decision(model, scenario, histories[si])
            if variant == 'L0':
                formal_u = [first_u(L0_EXECUTION / 'raw_runs' / f'{l0_idx[(pi, si, seed)]:04d}.npz') for pi in range(PRIMARY_PLANTS)]
            else:
                formal_u = [first_u(FORMAL_EXECUTION / 'raw_runs' / f'{formal_index[(pi, si, seed, variant)]:04d}.npz') for pi in range(PRIMARY_PLANTS)]
            equal = all(u == u0 for u in formal_u)
            mismatches += not equal
            fa_rows.append(dict(seed=seed, variant=variant, scenario=SCENARIO_LABELS[si], solver_success=ok,
                                first_action=first_action_class(u0, scenario), equal_to_all_24_formal_first_u=equal))
    counts = {f'{s}:{v}': sum(r['first_action'] == 'wrong' for r in fa_rows if r['seed'] == s and r['variant'] == v) for s, v in models}
    report['R5_first_action_reproduction'] = dict(models=len(models), checks=len(fa_rows), mismatches=mismatches,
                                                  wrong_direction_counts=counts, pass_=mismatches == 0)
    print('R5 first action', mismatches == 0, counts, flush=True)

    rerun_rows = []
    seed = 20260923
    for k in range(12):
        pi, si = 2 * k, k
        task = dict(plant_index=pi, scenario_index=si, seed=seed, variant='R0')
        out = gate_dir / f'p{pi:02d}_{SCENARIO_LABELS[si]}_R0.npz'
        runtime = run_one(task, plants, scenarios, histories, models, bindings, rf.load_model(plant_binding(plants[pi]), plants[pi]['model_id']),
                          out, 'onset_gate_R5')
        eu, ey = bitwise(out, FORMAL_EXECUTION / 'raw_runs' / f'{formal_index[(pi, si, seed, "R0")]:04d}.npz')
        rerun_rows.append(dict(plant_index=pi, scenario=SCENARIO_LABELS[si], applied_UH_bitwise=eu, plant_Tp_bitwise=ey, runtime_s=runtime))
        print(f'R5 rerun p{pi:02d} {SCENARIO_LABELS[si]}: {eu and ey} ({runtime:.1f} s)', flush=True)
    report['R5_rerun'] = dict(rows=rerun_rows, pass_=all(r['applied_UH_bitwise'] and r['plant_Tp_bitwise'] for r in rerun_rows))

    wrong = [si for si in range(12) if next(r for r in fa_rows if r['seed'] == seed and r['variant'] == 'R0' and r['scenario'] == SCENARIO_LABELS[si])['first_action'] == 'wrong']
    si = min(s for s in range(12) if s not in wrong)
    id_rows = []
    for pi in GUARD_ID_PLANTS:
        task = dict(plant_index=pi, scenario_index=si, seed=seed, variant='R0_GUARD_ID')
        out = gate_dir / f'p{pi:02d}_{SCENARIO_LABELS[si]}_R0_GUARD_ID.npz'
        run_one(task, plants, scenarios, histories, models, bindings, rf.load_model(plant_binding(plants[pi]), plants[pi]['model_id']),
                out, 'onset_gate_R7_preview')
        eu, ey = bitwise(out, FORMAL_EXECUTION / 'raw_runs' / f'{formal_index[(pi, si, seed, "R0")]:04d}.npz')
        with np.load(out, allow_pickle=False) as z:
            fired = json.loads(str(z['metadata']))['guard_fired']
        id_rows.append(dict(plant_index=pi, scenario=SCENARIO_LABELS[si], guard_fired=fired, applied_UH_bitwise=eu, plant_Tp_bitwise=ey))
    report['R7_guard_identity_preview'] = dict(rows=id_rows, pass_=all(r['applied_UH_bitwise'] and r['plant_Tp_bitwise'] and not r['guard_fired'] for r in id_rows))
    report['guard_unit_test'] = guard_unit_test()

    nominal = float(np.mean([r['runtime_s'] for r in rerun_rows]))
    planned = 864 + 8640
    report['timing'] = dict(mean_runtime_s=nominal, planned_runs_excluding_guard=planned,
                            estimated_wall_hours_6_workers=planned * nominal / 3600 / 6)
    report['all_pass'] = all(report[k]['pass_'] for k in ('R5_previous_consistency', 'R5_first_action_reproduction', 'R5_rerun',
                                                          'R7_guard_identity_preview', 'guard_unit_test'))
    report['code_sha256'] = {'onset_mil.py': sha(Path(__file__))}
    GATE_REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: report[k]['pass_'] for k in report if isinstance(report[k], dict) and 'pass_' in report[k]}
                     | {'all_pass': report['all_pass'], 'est_wall_h_6w': report['timing']['estimated_wall_hours_6_workers']}))


def first_action_check(bindings, scenarios, histories):
    rows = []
    for seed, variants in FIRST_ACTION_MODELS.items():
        for variant in variants:
            model = rf.load_model(bindings[(seed, variant)], 'B4')
            for si, scenario in enumerate(scenarios):
                u0, ok = first_decision(model, scenario, histories[si])
                rows.append(dict(seed=seed, variant=variant, scenario_index=si, scenario=SCENARIO_LABELS[si],
                                 first_u=u0, solver_success=ok, first_action=first_action_class(u0, scenario)))
            print(f'{utc()} first-action check {seed} {variant}', flush=True)
    return rows


def task_list(fa_rows):
    tasks = []
    for pi in range(PRIMARY_PLANTS):
        for si in range(12):
            tasks += [dict(plant_index=pi, scenario_index=si, seed=s, variant='COISC_onset') for s in E1_SEEDS]
            tasks += [dict(plant_index=pi, scenario_index=si, seed=s, variant=v) for s in E2_SEEDS for v in ('R0', 'COISC_onset', 'L0')]
    for seed in E1_SEEDS + E2_SEEDS:
        wrong = [r['scenario_index'] for r in fa_rows if r['seed'] == seed and r['variant'] == 'R0' and r['first_action'] == 'wrong']
        tasks += [dict(plant_index=pi, scenario_index=si, seed=seed, variant='R0_GUARD') for si in wrong for pi in range(PRIMARY_PLANTS)]
        rest = [si for si in range(12) if si not in wrong]
        if wrong and rest:
            tasks += [dict(plant_index=pi, scenario_index=min(rest), seed=seed, variant='R0_GUARD_ID') for pi in GUARD_ID_PLANTS]
    return tasks


def model_keys(tasks):
    return sorted({(t['seed'], 'R0' if t['variant'].startswith('R0_GUARD') else t['variant']) for t in tasks})


def execute(workers=6):
    protocol, digest = frozen_protocol()
    auth = read(AUTHORIZATION)
    torch.set_num_threads(2)
    existing = sorted(OUT.glob('mil_execution_*'))
    execution = existing[-1] if existing else OUT / f'mil_execution_{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}_{digest[:10]}'
    execution.mkdir(parents=True, exist_ok=True)
    (execution / 'raw_runs').mkdir(exist_ok=True)
    audit_path = execution / 'MIL_EXECUTION_AUDIT.json'
    bindings = all_bindings(digest)
    scenarios, histories = rf.load_scenarios(read(FORMAL_PROTOCOL))
    fa_path = execution / 'FIRST_ACTION_CHECK.json'
    if not fa_path.is_file():
        fa_rows = first_action_check(bindings, scenarios, histories)
        fa_path.write_text(json.dumps(dict(protocol_sha256=digest, rows=fa_rows), indent=1), encoding='utf-8')
        import csv
        with (execution / 'FIRST_ACTION_CHECK.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(fa_rows[0]))
            writer.writeheader()
            writer.writerows(fa_rows)
    fa = read(fa_path)
    assert fa['protocol_sha256'] == digest
    tasks = task_list(fa['rows'])
    if audit_path.is_file():
        audit = read(audit_path)
        assert audit['protocol_sha256'] == digest
        audit['resume_log'].append(dict(at_utc=utc(), reason='same-protocol technical resume'))
    else:
        audit = dict(status='RUNNING', protocol_sha256=digest, execution_id=execution.name, total_runs=len(tasks),
                     authorization=auth, started_utc=utc(), resume_log=[], workers=workers, threads_per_worker=2,
                     no_tuning=True, no_plant_deletion=True, no_scenario_deletion=True,
                     guard_runs=sum(t['variant'] == 'R0_GUARD' for t in tasks),
                     guard_identity_runs=sum(t['variant'] == 'R0_GUARD_ID' for t in tasks))
    (execution / 'TASKS.json').write_text(json.dumps(tasks), encoding='utf-8')
    audit_path.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    procs = []
    for w in range(workers):
        log = (execution / f'worker_{w}.log').open('a', encoding='utf-8')
        procs.append(subprocess.Popen([PYTHON, str(Path(__file__).resolve()), 'worker', str(execution), str(w), str(workers)],
                                      stdout=log, stderr=subprocess.STDOUT, env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}))
    codes = [p.wait() for p in procs]
    done = sum((execution / 'raw_runs' / f'{i:05d}.npz').is_file() for i in range(len(tasks)))
    audit.update(worker_exit_codes=codes, completed_runs=done, finished_utc=utc(),
                 status='RAW_RUNS_COMPLETE' if done == len(tasks) and not any(codes) else 'INCOMPLETE',
                 worker_audits=[read(p) for p in sorted(execution.glob('worker_*_audit.json'))])
    audit_path.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps({'status': audit['status'], 'completed': done, 'total': len(tasks)}))


def worker(execution_dir, index, count):
    execution = Path(execution_dir)
    protocol, digest = frozen_protocol()
    assert read(execution / 'MIL_EXECUTION_AUDIT.json')['protocol_sha256'] == digest
    torch.set_num_threads(2)
    fa = read(execution / 'FIRST_ACTION_CHECK.json')
    tasks = read(execution / 'TASKS.json')
    assert fa['protocol_sha256'] == digest and tasks == task_list(fa['rows'])
    scenarios, histories = rf.load_scenarios(read(FORMAL_PROTOCOL))
    plants, _ = plants_and_formal_index()
    bindings = all_bindings(digest)
    models = load_models(bindings, model_keys(tasks))
    audit_path = execution / f'worker_{index}_audit.json'
    audit = dict(worker=index, count=count, completed=[], started_utc=utc())
    for pi in range(index, PRIMARY_PLANTS, count):
        remaining = [(i, t) for i, t in enumerate(tasks) if t['plant_index'] == pi and not (execution / 'raw_runs' / f'{i:05d}.npz').is_file()]
        if not remaining:
            continue
        plant_model = rf.load_model(plant_binding(plants[pi]), plants[pi]['model_id'])
        for i, task in remaining:
            runtime = run_one(task, plants, scenarios, histories, models, bindings, plant_model,
                              execution / 'raw_runs' / f'{i:05d}.npz', execution.name)
            audit['completed'].append(dict(task_index=i, runtime_s=runtime))
            audit_path.write_text(json.dumps(audit), encoding='utf-8')
            print(f'{utc()} task {i} plant {pi} {task["seed"]} {task["variant"]} {runtime:.1f}s', flush=True)
        del plant_model
    audit['finished_utc'] = utc()
    audit_path.write_text(json.dumps(audit), encoding='utf-8')


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else ''
    if command == 'gates':
        gates()
    elif command == 'execute':
        execute(int(sys.argv[2]) if len(sys.argv) > 2 else 6)
    elif command == 'worker':
        worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        raise SystemExit(__doc__)
