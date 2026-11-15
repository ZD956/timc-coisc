"""Closed-loop MIL runs of R0/COISC/L0 for the 40 screening seeds (onset_mil.run_one, unchanged).

  cl_mil.py gate                 G1: re-run 12 stored E2 runs of seed 20261001 (R0/L0) and compare bitwise
  cl_mil.py smoke                one task per variant on the smoke checkpoints (seed 1), outside the study directory
  cl_mil.py execute [WORKERS]    frozen protocol and author authorization required
  cl_mil.py worker DIR I N
"""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

import cl_common as cc
import onset_mil as om  # noqa: E402
import torch  # noqa: E402

E2_EXECUTION = cc.ONSET / 'mil_execution_20261003T104638Z_23b8f69525'


def e2_bindings(seed):
    """Seed-20261001 E2 checkpoints of the onset-aligned study."""
    rec = cc.read(cc.ONSET / 'e2' / f'seed_{seed}' / 'SEED_COMPLETE.json')
    out = {}
    for b in rec['bindings']:
        name = 'results/R0/selected.pt' if b['variant'] == 'R0' else f'{b["variant"]}_final.pt'
        out[(seed, b['variant'])] = dict(path=str(cc.ONSET / 'e2' / f'seed_{seed}' / name), sha256=b['sha256'])
    return out


def context():
    scenarios, histories = om.rf.load_scenarios(om.read(om.FORMAL_PROTOCOL))
    plants, _ = om.plants_and_formal_index()
    return scenarios, histories, plants


def gate():
    torch.set_num_threads(2)
    scenarios, histories, plants = context()
    stored = om.read(E2_EXECUTION / 'TASKS.json')
    bindings = e2_bindings(cc.REPRO_SEED)
    picks = []
    for k in range(12):
        variant = 'R0' if k % 2 == 0 else 'L0'
        pi, si = (2 * k + 1) % 24, k
        idx = next(i for i, t in enumerate(stored) if t == dict(plant_index=pi, scenario_index=si, seed=cc.REPRO_SEED, variant=variant))
        picks.append((idx, stored[idx]))
    models = om.load_models(bindings, sorted({(t['seed'], t['variant']) for _, t in picks}))
    out_dir = cc.GATES / 'G1_rerun'
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, task in picks:
        pi = task['plant_index']
        out = out_dir / f'{idx:05d}.npz'
        runtime = om.run_one(task, plants, scenarios, histories, models, bindings,
                             om.rf.load_model(om.plant_binding(plants[pi]), plants[pi]['model_id']), out, 'closedloop_gate_G1')
        eu, ey = om.bitwise(out, E2_EXECUTION / 'raw_runs' / f'{idx:05d}.npz')
        rows.append(dict(task_index=idx, **task, applied_UH_bitwise=eu, plant_Tp_bitwise=ey, runtime_s=runtime))
        print('G1', rows[-1], flush=True)
    mean_rt = sum(r['runtime_s'] for r in rows) / len(rows)
    fa = cc.first_actions()
    n_tasks = len(cc.task_list(fa))
    report = dict(label='Closed-loop study gate G1: onset_mil.run_one reproduces 12 stored E2 runs bitwise',
                  rows=rows, pass_=all(r['applied_UH_bitwise'] and r['plant_Tp_bitwise'] for r in rows),
                  timing=dict(mean_runtime_s=mean_rt, planned_runs=n_tasks, est_wall_h_6_workers=n_tasks * mean_rt / 3600 / 6),
                  code_sha256=cc.code_hashes(('cl_common.py', 'cl_mil.py')), created_utc=cc.utc())
    cc.write(cc.PROTO / 'GATE_G1_MIL_REPORT.json', report)
    print('G1 PASS', report['pass_'], report['timing'], flush=True)


def launch(root, tasks, bindings, workers):
    root = Path(root)
    (root / 'raw_runs').mkdir(parents=True, exist_ok=True)
    cc.write(root / 'TASKS.json', tasks)
    cc.write(root / 'BINDINGS.json', [dict(seed=s, variant=v, **b) for (s, v), b in sorted(bindings.items())])
    procs = []
    for w in range(workers):
        log = (root / f'worker_{w}.log').open('a', encoding='utf-8')
        procs.append(subprocess.Popen([cc.PYTHON, str(Path(__file__).resolve()), 'worker', str(root), str(w), str(workers)],
                                      stdout=log, stderr=subprocess.STDOUT, env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}))
    codes = [p.wait() for p in procs]
    done = sum((root / 'raw_runs' / f'{i:05d}.npz').is_file() for i in range(len(tasks)))
    return codes, done


def worker(root, index, count):
    root = Path(root)
    torch.set_num_threads(2)
    if root.resolve() == cc.STUDY.resolve():
        protocol, digest = cc.frozen_protocol()
        assert cc.read(root / 'EXECUTION_AUDIT.json')['protocol_sha256'] == digest
    tasks = cc.read(root / 'TASKS.json')
    bindings = {(b['seed'], b['variant']): dict(path=b['path'], sha256=b['sha256']) for b in cc.read(root / 'BINDINGS.json')}
    scenarios, histories, plants = context()
    models = om.load_models(bindings, sorted({(t['seed'], t['variant']) for t in tasks}))
    audit_path = root / f'worker_{index}_audit.json'
    audit = cc.read(audit_path) if audit_path.is_file() else dict(worker=index, count=count, completed=[])
    audit.setdefault('starts_utc', []).append(cc.utc())
    plant_models = {}
    # tasks are interleaved across workers (panel B concentrates on 6 plants); plant models are cached per worker
    for i, task in enumerate(tasks):
        if i % count != index or (root / 'raw_runs' / f'{i:05d}.npz').is_file():
            continue
        pi = task['plant_index']
        if pi not in plant_models:
            plant_models[pi] = om.rf.load_model(om.plant_binding(plants[pi]), plants[pi]['model_id'])
        runtime = om.run_one(task, plants, scenarios, histories, models, bindings, plant_models[pi],
                             root / 'raw_runs' / f'{i:05d}.npz', root.name)
        audit['completed'].append(dict(task_index=i, runtime_s=runtime))
        cc.write(audit_path, audit)
        print(f'{cc.utc()} task {i} plant {pi} {task["seed"]} {task["variant"]} {runtime:.1f}s', flush=True)
    audit['finished_utc'] = cc.utc()
    cc.write(audit_path, audit)


def smoke():
    rec = cc.read(cc.SCREEN / 'smoke' / f'seed_{cc.SMOKE_SEED}' / 'SEED_COMPLETE.json')
    bindings = {(cc.SMOKE_SEED, b['variant']): dict(path=b['path'], sha256=b['sha256']) for b in rec['bindings'] if b['variant'] in cc.VARIANTS}
    tasks = [dict(plant_index=pi, scenario_index=0, seed=cc.SMOKE_SEED, variant=v) for pi, v in zip((0, 8, 20), cc.VARIANTS)]
    codes, done = launch(cc.SMOKE, tasks, bindings, 3)
    assert not any(codes) and done == len(tasks), (codes, done)
    cc.write(cc.SMOKE / 'SMOKE_MIL_OK.json', dict(tasks=len(tasks), code_sha256=cc.code_hashes(('cl_common.py', 'cl_mil.py')), at=cc.utc()))
    print('SMOKE MIL OK', flush=True)


def execute(workers):
    protocol, digest = cc.frozen_protocol()
    fa = cc.first_actions()
    tasks = cc.task_list(fa)
    assert len(tasks) == protocol['planned_runs'], 'task list differs from the frozen plan'
    bindings = cc.study_bindings()
    assert len(bindings) == len(cc.NEW_SEEDS) * len(cc.VARIANTS)
    audit_path = cc.STUDY / 'EXECUTION_AUDIT.json'
    cc.STUDY.mkdir(parents=True, exist_ok=True)
    if audit_path.is_file():
        audit = cc.read(audit_path)
        assert audit['protocol_sha256'] == digest
        assert cc.read(cc.STUDY / 'TASKS.json') == tasks
        audit['resume_log'].append(dict(at_utc=cc.utc(), reason='same-protocol technical resume'))
    else:
        audit = dict(status='RUNNING', protocol_sha256=digest, total_runs=len(tasks), authorization=cc.read(cc.AUTHORIZATION),
                     started_utc=cc.utc(), resume_log=[], workers=workers, threads_per_worker=2,
                     no_tuning=True, no_plant_deletion=True, no_scenario_deletion=True, no_seed_deletion=True)
    cc.write(audit_path, audit)
    codes, done = launch(cc.STUDY, tasks, bindings, workers)
    audit.update(worker_exit_codes=codes, completed_runs=done, finished_utc=cc.utc(),
                 status='RAW_RUNS_COMPLETE' if done == len(tasks) and not any(codes) else 'INCOMPLETE')
    cc.write(audit_path, audit)
    print(dict(status=audit['status'], completed=done, total=len(tasks)), flush=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'gate':
        gate()
    elif mode == 'smoke':
        smoke()
    elif mode == 'execute':
        execute(int(sys.argv[2]) if len(sys.argv) > 2 else 6)
    elif mode == 'worker':
        worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
