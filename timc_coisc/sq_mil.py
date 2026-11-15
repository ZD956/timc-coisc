"""Closed-loop MIL runs of COISC_seq on the stage-S4 bench (onset_mil.run_one, unchanged); R0/COISC/L0 runs are reused from S4.

  sq_mil.py gate                 G5: re-run 6 stored S4 runs (R0/COISC/L0, panels A and B) and compare bitwise
  sq_mil.py smoke                two COISC_seq tasks on the gate checkpoint (seed 20261101), outside the study directory
  sq_mil.py execute [WORKERS]    frozen protocol and author authorization required
  sq_mil.py worker DIR I N
"""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

import sq_common as sq
import cl_common as cc  # noqa: E402
import cl_mil  # noqa: E402
import onset_mil as om  # noqa: E402
import torch  # noqa: E402

GATE_CKPT = sq.GATES / 'G1_G2_train' / f'{sq.VARIANT}_final.pt'


def task_list():
    """S4 task list with every variant replaced by COISC_seq (panel B: 40 seeds x 12 scenarios x 6 plants; panel A: other 18 plants on R0-wrong pairs)."""
    out, seen = [], set()
    for t in cc.task_list(cc.first_actions()):
        key = (t['plant_index'], t['scenario_index'], t['seed'])
        if key not in seen:
            seen.add(key); out.append(dict(plant_index=t['plant_index'], scenario_index=t['scenario_index'], seed=t['seed'], variant=sq.VARIANT))
    return out


def gate():
    torch.set_num_threads(2)
    scenarios, histories, plants = cl_mil.context()
    stored = cc.read(sq.CL_STUDY / 'TASKS.json')
    wrong = set(cc.wrong_pairs(cc.first_actions()))
    panel_a = [i for i, t in enumerate(stored) if t['plant_index'] not in cc.PANEL_B_PLANTS]
    panel_b = [i for i, t in enumerate(stored) if t['plant_index'] in cc.PANEL_B_PLANTS and (t['seed'], t['scenario_index']) not in wrong]
    picks = [panel_b[k] for k in (1, len(panel_b) // 2, len(panel_b) - 3)] + [panel_a[k] for k in (0, len(panel_a) // 2 + 1, len(panel_a) - 1)]
    bindings = sq.screening_bindings()
    out_dir = sq.GATES / 'G5_rerun'
    if out_dir.exists(): raise FileExistsError(f'preserve existing gate attempt: {out_dir}')
    out_dir.mkdir(parents=True)
    rows = []
    for idx in picks:
        task = stored[idx]; pi = task['plant_index']
        models = om.load_models(bindings, [(task['seed'], task['variant'])])
        out = out_dir / f'{idx:05d}.npz'
        runtime = om.run_one(task, plants, scenarios, histories, models, bindings,
                             om.rf.load_model(om.plant_binding(plants[pi]), plants[pi]['model_id']), out, 'seq_gate_G5')
        eu, ey = om.bitwise(out, sq.CL_STUDY / 'raw_runs' / f'{idx:05d}.npz')
        rows.append(dict(task_index=idx, **task, applied_UH_bitwise=eu, plant_Tp_bitwise=ey, runtime_s=runtime))
        print('G5', rows[-1], flush=True)
    mean_rt = sum(r['runtime_s'] for r in rows) / len(rows)
    n = len(task_list())
    report = dict(label='Sequence study gate G5: onset_mil.run_one reproduces 6 stored stage-S4 runs bitwise (screening checkpoints)',
                  rows=rows, pass_=all(r['applied_UH_bitwise'] and r['plant_Tp_bitwise'] for r in rows),
                  variants_covered=sorted({r['variant'] for r in rows}),
                  timing=dict(mean_runtime_s=mean_rt, planned_runs=n, est_wall_h_6_workers=n * mean_rt / 3600 / 6),
                  code_sha256=sq.code_hashes(('sq_common.py', 'sq_mil.py')), created_utc=sq.utc())
    sq.write(sq.PROTO / 'GATE_G5_MIL_REPORT.json', report)
    print('G5 PASS', report['pass_'], report['timing'], flush=True)


def launch(root, tasks, bindings, workers):
    root = Path(root)
    (root / 'raw_runs').mkdir(parents=True, exist_ok=True)
    sq.write(root / 'TASKS.json', tasks)
    sq.write(root / 'BINDINGS.json', [dict(seed=s, variant=v, **b) for (s, v), b in sorted(bindings.items())])
    procs = []
    for w in range(workers):
        log = (root / f'worker_{w}.log').open('a', encoding='utf-8')
        procs.append(subprocess.Popen([sq.PYTHON, str(Path(__file__).resolve()), 'worker', str(root), str(w), str(workers)],
                                      stdout=log, stderr=subprocess.STDOUT, env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}))
    codes = [p.wait() for p in procs]
    done = sum((root / 'raw_runs' / f'{i:05d}.npz').is_file() for i in range(len(tasks)))
    return codes, done


def worker(root, index, count):
    root = Path(root)
    torch.set_num_threads(2)
    if root.resolve() == (sq.STUDY / 'mil').resolve():
        protocol, digest = sq.frozen_protocol()
        assert sq.read(root / 'EXECUTION_AUDIT.json')['protocol_sha256'] == digest
    tasks = sq.read(root / 'TASKS.json')
    bindings = {(b['seed'], b['variant']): dict(path=b['path'], sha256=b['sha256']) for b in sq.read(root / 'BINDINGS.json')}
    scenarios, histories, plants = cl_mil.context()
    models = om.load_models(bindings, sorted({(t['seed'], t['variant']) for t in tasks}))
    audit_path = root / f'worker_{index}_audit.json'
    audit = sq.read(audit_path) if audit_path.is_file() else dict(worker=index, count=count, completed=[])
    audit.setdefault('starts_utc', []).append(sq.utc())
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
        sq.write(audit_path, audit)
        print(f'{sq.utc()} task {i} plant {pi} {task["seed"]} {task["variant"]} {runtime:.1f}s', flush=True)
    audit['finished_utc'] = sq.utc()
    sq.write(audit_path, audit)


def smoke():
    root = sq.SMOKE / 'mil'
    if root.exists():
        import shutil; shutil.rmtree(root)
    bindings = {(20261101, sq.VARIANT): dict(path=str(GATE_CKPT), sha256=sq.sha(GATE_CKPT))}
    tasks = [dict(plant_index=pi, scenario_index=si, seed=20261101, variant=sq.VARIANT) for pi, si in ((1, 3), (12, 9))]  # plants 1, 12 never run for this seed in the study
    codes, done = launch(root, tasks, bindings, 2)
    assert not any(codes) and done == len(tasks), (codes, done)
    sq.write(sq.SMOKE / 'SMOKE_MIL_OK.json', dict(tasks=len(tasks), code_sha256=sq.code_hashes(('sq_common.py', 'sq_mil.py')), at=sq.utc()))
    print('SMOKE MIL OK', flush=True)


def execute(workers):
    protocol, digest = sq.frozen_protocol()
    tasks = task_list()
    assert len(tasks) == protocol['planned_runs'], 'task list differs from the frozen plan'
    bindings = sq.seq_bindings()
    assert len(bindings) == len(sq.NEW_SEEDS), 'COISC_seq training incomplete'
    root = sq.STUDY / 'mil'
    audit_path = root / 'EXECUTION_AUDIT.json'
    root.mkdir(parents=True, exist_ok=True)
    if audit_path.is_file():
        audit = sq.read(audit_path)
        assert audit['protocol_sha256'] == digest
        assert sq.read(root / 'TASKS.json') == tasks
        audit['resume_log'].append(dict(at_utc=sq.utc(), reason='same-protocol technical resume'))
    else:
        audit = dict(status='RUNNING', protocol_sha256=digest, total_runs=len(tasks), authorization=sq.read(sq.AUTHORIZATION),
                     started_utc=sq.utc(), resume_log=[], workers=workers, threads_per_worker=2,
                     no_tuning=True, no_plant_deletion=True, no_scenario_deletion=True, no_seed_deletion=True)
    sq.write(audit_path, audit)
    codes, done = launch(root, tasks, bindings, workers)
    audit.update(worker_exit_codes=codes, completed_runs=done, finished_utc=sq.utc(),
                 status='RAW_RUNS_COMPLETE' if done == len(tasks) and not any(codes) else 'INCOMPLETE')
    sq.write(audit_path, audit)
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
