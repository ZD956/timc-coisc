"""MIL runs for the move-penalty baseline study (onset_mil.run_one, unchanged; lambda_du set per task through MPCConfig).

  du_mil.py gate                 G1: with lambda_du = 1.0, re-run 6 stored S4 runs bitwise (R0, COISC, L0) and 2 R0_DU runs that
                                     must equal the stored R0 runs; config unit test for lambda_du != 1
  du_mil.py smoke                two R0_DU runs (lambda 4) on seed 20261101, plants not in panel B, outside the study directory
  du_mil.py execute [WORKERS]    frozen protocol and author authorization required
  du_mil.py worker DIR I N
"""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

import du_common as dc
import cl_common as cc  # noqa: E402
import cl_mil  # noqa: E402
import onset_mil as om  # noqa: E402
import torch  # noqa: E402


def launch(root, tasks, bindings, workers):
    """Each task carries lambda_du; workers set it before every run."""
    root = Path(root)
    (root / 'raw_runs').mkdir(parents=True, exist_ok=True)
    dc.write(root / 'TASKS.json', tasks)
    dc.write(root / 'BINDINGS.json', [dict(seed=s, variant=v, **b) for (s, v), b in sorted(bindings.items())])
    procs = []
    for w in range(workers):
        log = (root / f'worker_{w}.log').open('a', encoding='utf-8')
        procs.append(subprocess.Popen([dc.cc.PYTHON, str(Path(__file__).resolve()), 'worker', str(root), str(w), str(workers)],
                                      stdout=log, stderr=subprocess.STDOUT, env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}))
    codes = [p.wait() for p in procs]
    done = sum((root / 'raw_runs' / f'{i:05d}.npz').is_file() for i in range(len(tasks)))
    return codes, done


def run_task(task, plants, scenarios, histories, models, bindings, plant_model, out, execution_id):
    dc.use_lambda_du(task['lambda_du'])
    plain = {k: task[k] for k in ('plant_index', 'scenario_index', 'seed', 'variant')}
    return om.run_one(plain, plants, scenarios, histories, models, bindings, plant_model, out, execution_id)


def worker(root, index, count):
    root = Path(root)
    torch.set_num_threads(2)
    if root.resolve() == dc.STUDY.resolve():
        protocol, digest = dc.frozen_protocol()
        assert dc.read(root / 'EXECUTION_AUDIT.json')['protocol_sha256'] == digest
    tasks = dc.read(root / 'TASKS.json')
    bindings = {(b['seed'], b['variant']): dict(path=b['path'], sha256=b['sha256']) for b in dc.read(root / 'BINDINGS.json')}
    scenarios, histories, plants = cl_mil.context()
    models = om.load_models(bindings, sorted({(t['seed'], t['variant']) for t in tasks}))
    audit_path = root / f'worker_{index}_audit.json'
    audit = dc.read(audit_path) if audit_path.is_file() else dict(worker=index, count=count, completed=[])
    audit.setdefault('starts_utc', []).append(dc.utc())
    plant_models = {}
    for i, task in enumerate(tasks):
        if i % count != index or (root / 'raw_runs' / f'{i:05d}.npz').is_file():
            continue
        pi = task['plant_index']
        if pi not in plant_models:
            plant_models[pi] = om.rf.load_model(om.plant_binding(plants[pi]), plants[pi]['model_id'])
        runtime = run_task(task, plants, scenarios, histories, models, bindings, plant_models[pi],
                           root / 'raw_runs' / f'{i:05d}.npz', root.name)
        audit['completed'].append(dict(task_index=i, runtime_s=runtime))
        dc.write(audit_path, audit)
        print(f'{dc.utc()} task {i} plant {pi} {task["seed"]} {task["variant"]} lambda_du {task["lambda_du"]} {runtime:.1f}s', flush=True)
    audit['finished_utc'] = dc.utc()
    dc.write(audit_path, audit)


def gate():
    torch.set_num_threads(2)
    scenarios, histories, plants = cl_mil.context()
    stored = cc.read(dc.CL_STUDY / 'TASKS.json')
    b_idx = [i for i, t in enumerate(stored) if t['plant_index'] in cc.PANEL_B_PLANTS]
    picks = [(b_idx[k], stored[b_idx[k]]) for k in (2, 700, 1501, 2402, 3303, len(b_idx) - 1)]
    picks += [(b_idx[k], dict(stored[b_idx[k]], variant=dc.ARM)) for k in (9, 4806) if stored[b_idx[k]]['variant'] == 'R0']
    assert len(picks) == 8, 'identity picks must be R0 tasks'
    allb = cc.study_bindings()
    bindings = {**allb, **dc.study_bindings()}
    out_dir = dc.GATES / 'G1_rerun'
    if out_dir.exists(): raise FileExistsError(f'preserve existing gate attempt: {out_dir}')
    out_dir.mkdir(parents=True)
    rows = []
    for idx, task in picks:
        pi = task['plant_index']
        models = om.load_models(bindings, [(task['seed'], task['variant'])])
        out = out_dir / f'{idx:05d}_{task["variant"]}.npz'
        runtime = run_task(dict(task, lambda_du=1.0), plants, scenarios, histories, models, bindings,
                           om.rf.load_model(om.plant_binding(plants[pi]), plants[pi]['model_id']), out, 'du_gate_G1')
        eu, ey = om.bitwise(out, dc.CL_STUDY / 'raw_runs' / f'{idx:05d}.npz')
        rows.append(dict(task_index=idx, **task, applied_UH_bitwise=eu, plant_Tp_bitwise=ey, runtime_s=runtime))
        print('G1', rows[-1], flush=True)
    dc.use_lambda_du(8.0)
    cfg = om.MPCConfig()
    default = dc.MPCConfig()
    unit = dict(lambda_du_set=cfg.lambda_du == 8.0,
                others_default=all(getattr(cfg, f) == getattr(default, f) for f in ('horizon', 'sigma_y_c', 'lambda_u', 'u_ref_percent', 'maxiter', 'ftol')))
    dc.use_lambda_du(1.0)
    unit['reset'] = om.MPCConfig().lambda_du == 1.0
    mean_rt = sum(r['runtime_s'] for r in rows) / len(rows)
    report = dict(label='Move-penalty study gate G1: run_task with lambda_du = 1.0 reproduces 6 stored S4 runs bitwise, R0_DU at 1.0 '
                        'equals stored R0 runs, and the MPCConfig replacement changes lambda_du only',
                  rows=rows, unit_test=unit,
                  pass_=all(r['applied_UH_bitwise'] and r['plant_Tp_bitwise'] for r in rows) and all(unit.values()),
                  timing=dict(mean_runtime_s_at_lambda_1=mean_rt, planned_runs=len(dc.task_list()),
                              est_wall_h_6_workers=len(dc.task_list()) * mean_rt / 3600 / 6),
                  code_sha256=dc.code_hashes(('du_common.py', 'du_mil.py')), created_utc=dc.utc())
    dc.write(dc.PROTO / 'GATE_G1_MIL_REPORT.json', report)
    print('G1 PASS', report['pass_'], unit, report['timing'], flush=True)


def smoke():
    root = dc.SMOKE / 'mil'
    if root.exists():
        import shutil; shutil.rmtree(root)
    bindings = {k: v for k, v in dc.study_bindings().items() if k[0] == 20261101}
    tasks = [dict(plant_index=pi, scenario_index=si, seed=20261101, variant=dc.ARM, lambda_du=4.0) for pi, si in ((1, 3), (12, 9))]
    codes, done = launch(root, tasks, bindings, 2)
    assert not any(codes) and done == len(tasks), (codes, done)
    dc.write(dc.SMOKE / 'SMOKE_MIL_OK.json', dict(tasks=len(tasks), code_sha256=dc.code_hashes(('du_common.py', 'du_mil.py')), at=dc.utc()))
    print('SMOKE MIL OK', flush=True)


def execute(workers):
    protocol, digest = dc.frozen_protocol()
    lam = protocol['lambda_du']
    tasks = [dict(t, lambda_du=lam) for t in dc.task_list()]
    assert len(tasks) == protocol['planned_runs'], 'task list differs from the frozen plan'
    bindings = dc.study_bindings()
    audit_path = dc.STUDY / 'EXECUTION_AUDIT.json'
    dc.STUDY.mkdir(parents=True, exist_ok=True)
    if audit_path.is_file():
        audit = dc.read(audit_path)
        assert audit['protocol_sha256'] == digest
        assert dc.read(dc.STUDY / 'TASKS.json') == tasks
        audit['resume_log'].append(dict(at_utc=dc.utc(), reason='same-protocol technical resume'))
    else:
        audit = dict(status='RUNNING', protocol_sha256=digest, total_runs=len(tasks), authorization=dc.read(dc.AUTHORIZATION),
                     started_utc=dc.utc(), resume_log=[], workers=workers, threads_per_worker=2, lambda_du=lam,
                     no_tuning=True, no_plant_deletion=True, no_scenario_deletion=True, no_seed_deletion=True)
    dc.write(audit_path, audit)
    codes, done = launch(dc.STUDY, tasks, bindings, workers)
    audit.update(worker_exit_codes=codes, completed_runs=done, finished_utc=dc.utc(),
                 status='RAW_RUNS_COMPLETE' if done == len(tasks) and not any(codes) else 'INCOMPLETE')
    dc.write(audit_path, audit)
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
