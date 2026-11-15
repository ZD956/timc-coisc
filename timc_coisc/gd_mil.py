"""Closed-loop MIL runs of R0_GUARD on the six R0-wrong pairs (onset_mil.run_one and guard_decision, unchanged).

  gd_mil.py gate                 G1: re-run 6 stored onset-study R0_GUARD runs (seed 20261002) and 4 stored
                                     closed-loop-study R0 runs on R0-wrong pairs; compare bitwise
  gd_mil.py smoke                one R0_GUARD task on the smoke checkpoint (seed 1), outside the study directory
  gd_mil.py execute [WORKERS]    frozen protocol and author authorization required
  gd_mil.py worker DIR I N
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

import gd_common as gc
import cl_common as cc  # noqa: E402
import cl_mil  # noqa: E402
import onset_mil as om  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

GUARD_REPRO_SEED = 20261002


def gate():
    torch.set_num_threads(2)
    scenarios, histories, plants = cl_mil.context()
    rows = []
    # (a) the guard path: stored R0_GUARD runs of the onset-aligned study
    stored = om.read(gc.ONSET_EXECUTION / 'TASKS.json')
    guard_idx = [i for i, t in enumerate(stored) if t['variant'] == gc.GUARD and t['seed'] == GUARD_REPRO_SEED]
    picks = [(guard_idx[k], stored[guard_idx[k]], gc.ONSET_EXECUTION) for k in (0, 5, 17, 18, 20, 23)]
    bindings_a = cl_mil.e2_bindings(GUARD_REPRO_SEED)
    # (b) the study checkpoints: stored R0 runs of the closed-loop study on R0-wrong pairs
    cl_tasks = cc.read(gc.CL_STUDY / 'TASKS.json')
    wrong = set(gc.wrong_pairs())
    cl_idx = [i for i, t in enumerate(cl_tasks) if t['variant'] == 'R0' and (t['seed'], t['scenario_index']) in wrong]
    picks += [(cl_idx[k], cl_tasks[cl_idx[k]], gc.CL_STUDY) for k in (0, len(cl_idx) // 3, 2 * len(cl_idx) // 3, len(cl_idx) - 1)]
    # (c) identity: on pairs whose R0 first move is not reversed the guard must not fire, so R0_GUARD == stored R0 run
    ok_idx = [i for i, t in enumerate(cl_tasks) if t['variant'] == 'R0' and (t['seed'], t['scenario_index']) not in wrong]
    picks += [(ok_idx[k], dict(cl_tasks[ok_idx[k]], variant=gc.GUARD), gc.CL_STUDY) for k in (7, len(ok_idx) // 2, len(ok_idx) - 5)]
    bindings_b = cc.study_bindings()
    out_dir = gc.GATES / 'G1_rerun'
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, task, src in picks:
        bindings = bindings_a if src == gc.ONSET_EXECUTION else bindings_b
        key = (task['seed'], 'R0')
        models = om.load_models(bindings, [key])
        pi = task['plant_index']
        out = out_dir / f'{src.parent.name[:5]}_{idx:05d}.npz'
        runtime = om.run_one(task, plants, scenarios, histories, models, bindings,
                             om.rf.load_model(om.plant_binding(plants[pi]), plants[pi]['model_id']), out, 'guard_gate_G1')
        eu, ey = om.bitwise(out, src / 'raw_runs' / f'{idx:05d}.npz')
        with np.load(out, allow_pickle=False) as z:
            fired = json.loads(str(z['metadata']))['guard_fired']
        rows.append(dict(source=src.name, task_index=idx, **task, guard_fired=fired, applied_UH_bitwise=eu, plant_Tp_bitwise=ey, runtime_s=runtime))
        print('G1', rows[-1], flush=True)
    guard_test = om.guard_unit_test()
    mean_rt = sum(r['runtime_s'] for r in rows) / len(rows)
    n_tasks = len(gc.task_list())
    identity = [r for r in rows if r['source'] == gc.CL_STUDY.name and r['variant'] == gc.GUARD]
    report = dict(label='Guard study gate G1: onset_mil.run_one reproduces 6 stored R0_GUARD runs and 4 stored closed-loop R0 runs '
                        'bitwise; R0_GUARD equals the stored R0 run on 3 pairs without a reversed R0 first move; guard unit test',
                  rows=rows, guard_unit_test=guard_test,
                  identity_pass=bool(len(identity) == 3 and not any(r['guard_fired'] for r in identity)),
                  pass_=all(r['applied_UH_bitwise'] and r['plant_Tp_bitwise'] for r in rows) and guard_test['pass_']
                  and len(identity) == 3 and not any(r['guard_fired'] for r in identity),
                  timing=dict(mean_runtime_s=mean_rt, planned_runs=n_tasks, est_wall_h_6_workers=n_tasks * mean_rt / 3600 / 6),
                  code_sha256=gc.code_hashes(('gd_common.py', 'gd_mil.py')), created_utc=gc.utc())
    gc.write(gc.PROTO / 'GATE_G1_MIL_REPORT.json', report)
    print('G1 PASS', report['pass_'], report['timing'], flush=True)


def launch(root, tasks, bindings, workers):
    root = Path(root)
    (root / 'raw_runs').mkdir(parents=True, exist_ok=True)
    gc.write(root / 'TASKS.json', tasks)
    gc.write(root / 'BINDINGS.json', [dict(seed=s, variant=v, **b) for (s, v), b in sorted(bindings.items())])
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
    if root.resolve() == gc.STUDY.resolve():
        protocol, digest = gc.frozen_protocol()
        assert gc.read(root / 'EXECUTION_AUDIT.json')['protocol_sha256'] == digest
    tasks = gc.read(root / 'TASKS.json')
    bindings = {(b['seed'], b['variant']): dict(path=b['path'], sha256=b['sha256']) for b in gc.read(root / 'BINDINGS.json')}
    scenarios, histories, plants = cl_mil.context()
    models = om.load_models(bindings, sorted({(t['seed'], 'R0') for t in tasks}))
    audit_path = root / f'worker_{index}_audit.json'
    audit = gc.read(audit_path) if audit_path.is_file() else dict(worker=index, count=count, completed=[])
    audit.setdefault('starts_utc', []).append(gc.utc())
    plant_models = {}
    for i, task in enumerate(tasks):
        if i % count != index or (root / 'raw_runs' / f'{i:05d}.npz').is_file():
            continue
        pi = task['plant_index']
        if pi not in plant_models:
            plant_models[pi] = om.rf.load_model(om.plant_binding(plants[pi]), plants[pi]['model_id'])
        runtime = om.run_one(task, plants, scenarios, histories, models, bindings, plant_models[pi],
                             root / 'raw_runs' / f'{i:05d}.npz', root.name)
        audit['completed'].append(dict(task_index=i, runtime_s=runtime))
        gc.write(audit_path, audit)
        print(f'{gc.utc()} task {i} plant {pi} {task["seed"]} {task["variant"]} {runtime:.1f}s', flush=True)
    audit['finished_utc'] = gc.utc()
    gc.write(audit_path, audit)


def r0_bindings():
    """R0 checkpoints of the seeds with an R0-wrong pair (the guard arm reuses them unchanged)."""
    allb = cc.study_bindings()
    return {(s, 'R0'): allb[(s, 'R0')] for s in sorted({s for s, _ in gc.wrong_pairs()})}


def smoke():
    rec = cc.read(cc.SCREEN / 'smoke' / f'seed_{cc.SMOKE_SEED}' / 'SEED_COMPLETE.json')
    bindings = {(cc.SMOKE_SEED, b['variant']): dict(path=b['path'], sha256=b['sha256']) for b in rec['bindings'] if b['variant'] == 'R0'}
    tasks = [dict(plant_index=pi, scenario_index=si, seed=cc.SMOKE_SEED, variant=gc.GUARD) for pi, si in ((0, 3), (20, 9))]
    codes, done = launch(gc.SMOKE, tasks, bindings, 2)
    assert not any(codes) and done == len(tasks), (codes, done)
    gc.write(gc.SMOKE / 'SMOKE_MIL_OK.json', dict(tasks=len(tasks), code_sha256=gc.code_hashes(('gd_common.py', 'gd_mil.py')), at=gc.utc()))
    print('SMOKE MIL OK', flush=True)


def execute(workers):
    protocol, digest = gc.frozen_protocol()
    tasks = gc.task_list()
    assert len(tasks) == protocol['planned_runs'], 'task list differs from the frozen plan'
    bindings = r0_bindings()
    audit_path = gc.STUDY / 'EXECUTION_AUDIT.json'
    gc.STUDY.mkdir(parents=True, exist_ok=True)
    if audit_path.is_file():
        audit = gc.read(audit_path)
        assert audit['protocol_sha256'] == digest
        assert gc.read(gc.STUDY / 'TASKS.json') == tasks
        audit['resume_log'].append(dict(at_utc=gc.utc(), reason='same-protocol technical resume'))
    else:
        audit = dict(status='RUNNING', protocol_sha256=digest, total_runs=len(tasks), authorization=gc.read(gc.AUTHORIZATION),
                     started_utc=gc.utc(), resume_log=[], workers=workers, threads_per_worker=2,
                     no_tuning=True, no_plant_deletion=True, no_pair_deletion=True)
    gc.write(audit_path, audit)
    codes, done = launch(gc.STUDY, tasks, bindings, workers)
    audit.update(worker_exit_codes=codes, completed_runs=done, finished_utc=gc.utc(),
                 status='RAW_RUNS_COMPLETE' if done == len(tasks) and not any(codes) else 'INCOMPLETE')
    gc.write(audit_path, audit)
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
