"""Secondary model-level readouts of the sequence-sensitivity study (descriptive): holdout estimates and plant-free first decisions.

  sq_eval.py gate                 G3: holdout estimates of screening R0 (3 seeds) reproduce HOLDOUT_ESTIMATES.json exactly
                                  G4: first decisions of screening COISC (seed 20261109) reproduce FIRST_ACTION_CHECK.json exactly
  sq_eval.py smoke                both code paths on the screening COISC of seed 20261101 (no study checkpoint involved)
  sq_eval.py holdout              COISC_seq, COISC, L0 of all seeds; own process (screen.seed_all disables TF32)
  sq_eval.py first_action [W]     COISC_seq of all seeds on the 12 locked scenarios
  sq_eval.py fa_worker ROOT I N

Both commands of the study modes need the frozen protocol and the author's authorization.
"""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

import sq_common as sq
import onset_holdout as oh  # noqa: E402
import onset_mil as om  # noqa: E402
import scr_holdout as sh  # noqa: E402
import scr_first_action as sf  # noqa: E402
from onset_common import torch  # noqa: E402

HOLDOUT_VARIANTS = (sq.VARIANT, 'COISC', 'L0')
G3_SEEDS = (20261101, 20261120, 20261140)
G4_SEED = 20261109


def holdout_jobs(root):
    """[(seed, variant, binding)] for COISC_seq of the study root plus screening COISC and L0 of the same seeds."""
    scr = sq.screening_bindings(); seq = sq.seq_bindings(root)
    jobs = []
    for seed in sorted({s for s, _ in seq}):
        jobs += [(seed, v, seq[(seed, v)] if v == sq.VARIANT else scr[(seed, v)]) for v in HOLDOUT_VARIANTS]
    return jobs


def holdout(root, digest):
    data, counts, stats, device = sh.setup()
    out = Path(root) / 'HOLDOUT_ESTIMATES.json'
    done = sq.read(out)['rows'] if out.is_file() else []
    have = {(r['seed'], r['variant']) for r in done}
    for seed, v, b in holdout_jobs(root):
        if (seed, v) in have: continue
        done.append(dict(seed=seed, variant=v, **sh.estimate(b['path'], b['sha256'], data, counts, stats, device)))
        sq.write(out, dict(protocol_sha256=digest, rows=done))
        print(sq.utc(), 'HOLDOUT', seed, v, flush=True)
    return done


def fa_jobs(root):
    return [(seed, sq.VARIANT, b) for (seed, _), b in sorted(sq.seq_bindings(root).items())]


def fa_worker(root, index, count):
    torch.set_num_threads(2)
    root = Path(root); out = root / 'first_action'; out.mkdir(parents=True, exist_ok=True)
    scenarios, histories = sf.scenarios()
    for k, (seed, v, b) in enumerate(fa_jobs(root)):
        if k % count != index: continue
        target = out / f'{seed}_{v}.json'
        if target.is_file(): continue
        rows = sf.check_model(b['path'], b['sha256'], scenarios, histories)
        tmp = target.with_suffix('.tmp')
        sq.write(tmp, dict(seed=seed, variant=v, checkpoint_sha256=b['sha256'], rows=rows)); tmp.replace(target)
        print(sq.utc(), 'FIRST ACTION', seed, v, flush=True)


def fa_launch(root, workers):
    procs = []
    for w in range(workers):
        log = (Path(root) / f'first_action_worker_{w}.log').open('a', encoding='utf-8')
        procs.append(subprocess.Popen([sq.PYTHON, str(Path(__file__).resolve()), 'fa_worker', str(root), str(w), str(workers)],
                                      stdout=log, stderr=subprocess.STDOUT, env={**os.environ, 'PYTHONIOENCODING': 'utf-8'}))
    codes = [p.wait() for p in procs]
    assert not any(codes), f'worker failure {codes}'
    rows = []
    for seed, v, b in fa_jobs(root):
        rec = sq.read(Path(root) / 'first_action' / f'{seed}_{v}.json')
        assert rec['checkpoint_sha256'] == b['sha256']
        rows += [dict(seed=seed, variant=v, **r) for r in rec['rows']]
    return rows


def gate():
    scr = sq.screening_bindings()
    # G3: holdout path
    data, counts, stats, device = sh.setup()
    stored = {(r['seed'], r['variant']): r for r in sq.read(sq.SCREEN_STUDY / 'HOLDOUT_ESTIMATES.json')['rows']}
    g3 = []
    for seed in G3_SEEDS:
        est = sh.estimate(scr[(seed, 'R0')]['path'], scr[(seed, 'R0')]['sha256'], data, counts, stats, device)
        diff = max(abs(est[k] - stored[(seed, 'R0')][k]) for k in sh.KEYS)
        g3.append(dict(seed=seed, max_abs_diff=diff, pass_=diff == 0))
        print('G3', g3[-1], flush=True)
    # G4: first-decision path (spawned so that it does not share the holdout process)
    out = sq.GATES / 'G4_first_action'
    if out.exists(): raise FileExistsError(f'preserve existing gate attempt: {out}')
    out.mkdir(parents=True)
    code = subprocess.call([sq.PYTHON, str(Path(__file__).resolve()), 'g4_child', str(out)],
                           env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    g4 = sq.read(out / 'G4.json'); g4['child_exit'] = code
    report = dict(label='Sequence study gates G3 (holdout path) and G4 (first-decision path) on stored screening outputs',
                  G3=dict(rows=g3, pass_=all(r['pass_'] for r in g3)), G4=g4,
                  pass_=all(r['pass_'] for r in g3) and g4['pass_'] and code == 0,
                  code_sha256=sq.code_hashes(('sq_common.py', 'sq_eval.py')), created_utc=sq.utc())
    sq.write(sq.PROTO / 'GATE_G3_G4_EVAL_REPORT.json', report)
    print('G3/G4 PASS', report['pass_'], flush=True)


def g4_child(out):
    torch.set_num_threads(2)
    b = sq.screening_bindings()[(G4_SEED, 'COISC')]
    scenarios, histories = sf.scenarios()
    got = sf.check_model(b['path'], b['sha256'], scenarios, histories)
    stored = {(r['seed'], r['variant'], r['scenario_index']): r for r in sq.read(sq.SCREEN_STUDY / 'FIRST_ACTION_CHECK.json')['rows']}
    eq = [repr(float(g['first_u'])) == repr(float(stored[(G4_SEED, 'COISC', g['scenario_index'])]['first_u'])) and
          g['first_action'] == stored[(G4_SEED, 'COISC', g['scenario_index'])]['first_action'] for g in got]
    sq.write(Path(out) / 'G4.json', dict(seed=G4_SEED, variant='COISC', scenarios=len(got), exact=sum(eq), pass_=all(eq)))
    print('G4', sum(eq), len(got), flush=True)


def smoke():
    """Smoke root holds only the screening COISC binding of seed 20261101 relabelled as the study variant."""
    root = sq.SMOKE / 'eval'
    if root.exists():
        import shutil; shutil.rmtree(root)
    folder = root / 'seed_20261101'; folder.mkdir(parents=True)
    b = sq.screening_bindings()[(20261101, 'COISC')]
    sq.write(folder / 'SEED_COMPLETE.json', dict(seed=20261101, variant=sq.VARIANT, path=b['path'], sha256=b['sha256']))
    rows = holdout(root, 'SMOKE'); assert len(rows) == len(HOLDOUT_VARIANTS)
    fa = fa_launch(root, 1); assert len(fa) == 12
    sq.write(sq.SMOKE / 'SMOKE_EVAL_OK.json', dict(code_sha256=sq.code_hashes(('sq_common.py', 'sq_eval.py')), at=sq.utc()))
    print('SMOKE EVAL OK', flush=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'gate':
        gate()
    elif mode == 'g4_child':
        g4_child(sys.argv[2])
    elif mode == 'smoke':
        smoke()
    elif mode == 'holdout':
        protocol, digest = sq.frozen_protocol()
        rows = holdout(sq.STUDY, digest)
        assert len(rows) == len(HOLDOUT_VARIANTS) * len(protocol['new_seeds']), 'holdout incomplete'
        print('HOLDOUT COMPLETE', len(rows), flush=True)
    elif mode == 'first_action':
        protocol, digest = sq.frozen_protocol()
        rows = fa_launch(sq.STUDY, int(sys.argv[2]) if len(sys.argv) > 2 else 6)
        assert len(rows) == 12 * len(protocol['new_seeds']), 'first-action check incomplete'
        sq.write(sq.STUDY / 'FIRST_ACTION_CHECK.json', dict(protocol_sha256=digest, rows=rows))
        print('FIRST ACTION COMPLETE', len(rows), flush=True)
    elif mode == 'fa_worker':
        fa_worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
