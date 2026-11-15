"""Plant-free MPC first-decision check on the 12 locked scenarios (onset_mil.first_decision / first_action_class, unchanged).

  scr_first_action.py gate              G3: existing seed-20261001 models reproduce FIRST_ACTION_CHECK.csv first_u exactly
  scr_first_action.py smoke             code path on the smoke checkpoints (R0 only)
  scr_first_action.py run [WORKERS]     all study checkpoints; frozen protocol and author authorization required
  scr_first_action.py worker ROOT I N
"""
from __future__ import annotations
import csv
import os
import subprocess
import sys
from pathlib import Path

import scr_common as sc
import onset_mil as om  # noqa: E402
import torch  # noqa: E402


def check_model(path, sha256, scenarios, histories):
    model = om.rf.load_model(dict(path=str(path), sha256=sha256), 'B4')
    rows = []
    for si, scenario in enumerate(scenarios):
        u0, ok = om.first_decision(model, scenario, histories[si])
        rows.append(dict(scenario_index=si, scenario=om.SCENARIO_LABELS[si], first_u=u0, solver_success=ok,
                         first_action=om.first_action_class(u0, scenario)))
    return rows


def scenarios():
    return om.rf.load_scenarios(om.read(om.FORMAL_PROTOCOL))


def gate():
    torch.set_num_threads(2)
    sc_list, histories = scenarios()
    with (om.OUT / 'mil_execution_20261003T104638Z_23b8f69525' / 'FIRST_ACTION_CHECK.csv').open(encoding='utf-8') as f:
        stored = {(r['variant'], int(r['scenario_index'])): r for r in csv.DictReader(f) if int(r['seed']) == sc.REPRO_SEED}
    ref = {b['variant']: b for b in sc.read(om.OUT / 'e2' / f'seed_{sc.REPRO_SEED}' / 'SEED_COMPLETE.json')['bindings']}
    rows = []
    for v in sc.VARIANTS:
        path = om.OUT / 'e2' / f'seed_{sc.REPRO_SEED}' / ('results/R0/selected.pt' if v == 'R0' else f'{v}_final.pt')
        got = check_model(path, ref[v]['sha256'], sc_list, histories)
        eq = [repr(float(g['first_u'])) == repr(float(stored[(v, g['scenario_index'])]['first_u'])) and
              g['first_action'] == stored[(v, g['scenario_index'])]['first_action'] for g in got]
        rows.append(dict(variant=v, scenarios=len(got), exact=sum(eq), pass_=all(eq)))
        print('G3', rows[-1], flush=True)
    previous = [om.runner_previous(histories[i]) == float(s['previous_uh_percent']) for i, s in enumerate(sc_list)]
    report = dict(label='Screening study gate G3: plant-free first decisions of seed-20261001 models reproduce the stored check',
                  rows=rows, previous_consistency=all(previous), pass_=all(r['pass_'] for r in rows) and all(previous),
                  code_sha256=sc.code_hashes(('scr_common.py', 'scr_first_action.py')))
    sc.write(sc.PROTO / 'GATE_G3_FIRST_ACTION_REPORT.json', report)
    print('G3 PASS', report['pass_'], flush=True)


def jobs(root):
    return [(r['seed'], b) for r in sc.seed_records(root) for b in r['bindings']]


def worker(root, index, count):
    torch.set_num_threads(2)
    root = Path(root); out = root / 'first_action'; out.mkdir(parents=True, exist_ok=True)
    sc_list, histories = scenarios()
    for k, (seed, b) in enumerate(jobs(root)):
        if k % count != index: continue
        target = out / f'{seed}_{b["variant"]}.json'
        if target.is_file(): continue
        rows = check_model(b['path'], b['sha256'], sc_list, histories)
        tmp = target.with_suffix('.tmp')
        sc.write(tmp, dict(seed=seed, variant=b['variant'], checkpoint_sha256=b['sha256'], rows=rows)); tmp.replace(target)
        print(sc.utc(), 'FIRST ACTION', seed, b['variant'], flush=True)


def launch(root, workers):
    procs = []
    for w in range(workers):
        log = (root / f'first_action_worker_{w}.log').open('a', encoding='utf-8')
        procs.append(subprocess.Popen([sc.PYTHON, str(Path(__file__).resolve()), 'worker', str(root), str(w), str(workers)],
                                      stdout=log, stderr=subprocess.STDOUT, env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}))
    codes = [p.wait() for p in procs]
    assert not any(codes), f'worker failure {codes}'
    rows = []
    for seed, b in jobs(root):
        rec = sc.read(root / 'first_action' / f'{seed}_{b["variant"]}.json')
        assert rec['checkpoint_sha256'] == b['sha256']
        rows += [dict(seed=seed, variant=b['variant'], **r) for r in rec['rows']]
    return rows


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'gate':
        gate()
    elif mode == 'smoke':
        rows = launch(sc.SMOKE, 1); assert len(rows) == 12 * len(sc.VARIANTS)
        print('SMOKE FIRST ACTION OK', flush=True)
    elif mode == 'run':
        protocol, digest = sc.frozen_protocol()
        rows = launch(sc.STUDY, int(sys.argv[2]) if len(sys.argv) > 2 else 6)
        assert len(rows) == 12 * len(sc.VARIANTS) * len(protocol['new_seeds']), 'first-action check incomplete'
        sc.write(sc.STUDY / 'FIRST_ACTION_CHECK.json', dict(protocol_sha256=digest, rows=rows))
        print('FIRST ACTION COMPLETE', len(rows), flush=True)
    elif mode == 'worker':
        worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
