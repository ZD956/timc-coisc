"""Training side of the sequence-sensitivity study: COISC_seq head adaptation of the 40 screening R0 checkpoints.

  sq_train.py gate    G1 calibrate_seq(STEPS=(0,)) reproduces lambda_prior and the stored calibration records bitwise
                      G2 adapt_seq(STEPS=(0,)) on screening R0 of seed 20261101 reproduces its COISC checkpoint bitwise
                      then calibrates lambda_seq for SEQ_STEPS (rule unchanged) and times one full adaptation
  sq_train.py smoke   code path on screening R0 of seed 20261101 with 2 steps; output deleted afterwards
  sq_train.py run     NEW_SEEDS; requires the frozen protocol and EXECUTION_AUTHORIZATION.json
"""
from __future__ import annotations
import argparse
import shutil

import sq_common as sq
import onset_common as oc  # noqa: E402  (path set by sq_common)
import onset_train as ot  # noqa: E402
from onset_common import screen, torch  # noqa: E402

SETUP = None
GATE_SEED = 20261101


def load_state(path):
    return sq.load_checkpoint(sq.private_path(path))['state_dict']


def adapt_one(r0_path, lam, seed, folder, plan_steps, n_steps, name=sq.VARIANT):
    device, train, ev, fit_ids, sel_ids, stats, probs, scores, old = SETUP
    r0_state = load_state(r0_path)
    model = ot.fresh(stats, r0_state, device)
    wall = sq.adapt_seq(model, train, probs, old, lam, oc.TAU_PRIOR, seed, device, folder / f'{name}_training.jsonl',
                        plan_steps, n_steps)
    assert ot.nonhead_equal(model, r0_state)
    path = folder / f'{name}_final.pt'
    digest = ot.save_ckpt(model, path, seed=seed, variant=name, w_pre=lam, w_sign=lam, lag_seconds=list(oc.TAU_PRIOR),
                          plan_steps=list(plan_steps), steps=n_steps)
    return model, path, digest, wall


def stages():
    old = SETUP[-1]
    return [{k: s[k] for k in ('steps', 'batch', 'lr', 'rollout', 'adjacent', 'endpoint')} for s in old['training']['stages']]


def gate():
    out = sq.GATES / 'G1_G2_train'
    if out.exists(): raise FileExistsError(f'preserve existing gate attempt: {out}')
    out.mkdir(parents=True)
    device, train, ev, fit_ids, sel_ids, stats, probs, scores, old = SETUP
    bind = sq.screening_bindings()
    report = dict(label='Sequence-sensitivity study gates G1-G2 (reduction to frozen COISC) and lambda_seq calibration')
    # G1: calibration with a single pulse at step 0 equals the formal calibration
    cal0 = sq.calibrate_seq(train, stats, probs, stages(), device, (0,))
    stored = oc.read_json(oc.SCREEN_DEV / 'calibration.json')['records']
    fields = ('prediction', 'regularizer', 'pre', 'sign', 'gradient_norm')
    diffs = [abs(a[f] - b[f]) for a, b in zip(cal0['records'], stored) for f in fields]
    order_ok = [(a['excitation'], a['batch']) for a in cal0['records']] == [(b['excitation'], b['batch']) for b in stored]
    report['G1_calibration'] = dict(lambda_reproduced=cal0['lambda_s'], lambda_prior=oc.LAMBDA_PRIOR, records_max_abs_diff=max(diffs),
                                    order_ok=order_ok, time_s=cal0['time_s'],
                                    pass_=bool(order_ok and max(diffs) == 0 and cal0['lambda_s'] == oc.LAMBDA_PRIOR))
    print('G1', report['G1_calibration'], flush=True)
    # G2: adaptation with STEPS=(0,) equals the screening COISC checkpoint of the same seed
    screen.SEED = GATE_SEED
    r0 = bind[(GATE_SEED, 'R0')]; ref = bind[(GATE_SEED, 'COISC')]
    assert sq.sha(sq.private_path(r0['path'])) == r0['sha256'] and sq.sha(sq.private_path(ref['path'])) == ref['sha256']
    model, path, digest, wall0 = adapt_one(r0['path'], oc.LAMBDA_PRIOR, GATE_SEED, out, (0,), oc.ADAPT_STEPS, name='COISC_step0')
    report['G2_adaptation'] = dict(seed=GATE_SEED, state_dict_bitwise_equal=oc.states_equal(load_state(path), load_state(ref['path'])),
                                   wall_s=wall0)
    report['G2_adaptation']['pass_'] = report['G2_adaptation']['state_dict_bitwise_equal']
    print('G2', report['G2_adaptation'], flush=True)
    # lambda_seq calibration (same pilots, probes and loss-ratio rule) and timing of one full COISC_seq adaptation
    cal = sq.calibrate_seq(train, stats, probs, stages(), device, sq.SEQ_STEPS)
    sq.write(sq.LAMBDA_FILE, dict(lambda_seq=cal['lambda_s'], per_sampler=cal['per_sampler'], steps=cal['steps'],
                                  records=cal['records'], time_s=cal['time_s'], code_sha256=sq.code_hashes(('sq_common.py', 'sq_train.py'))))
    _, _, _, wall = adapt_one(r0['path'], cal['lambda_s'], GATE_SEED, out, sq.SEQ_STEPS, oc.ADAPT_STEPS)
    report['lambda_seq'] = dict(value=cal['lambda_s'], per_sampler=cal['per_sampler'], calibration_time_s=cal['time_s'],
                                adaptation_wall_s=wall, adaptation_wall_ratio_vs_coisc=wall / wall0)
    print('LAMBDA_SEQ', report['lambda_seq'], flush=True)
    report['all_pass'] = report['G1_calibration']['pass_'] and report['G2_adaptation']['pass_']
    report['code_sha256'] = sq.code_hashes(('sq_common.py', 'sq_train.py'))
    sq.write(sq.PROTO / 'GATE_G1_G2_TRAIN_REPORT.json', report)
    print('ALL_PASS', report['all_pass'], flush=True)


def train_seed(seed, root, protocol_file, digest, lam, bind, n_steps=oc.ADAPT_STEPS):
    folder = root / f'seed_{seed}'
    if (folder / 'SEED_COMPLETE.json').exists():
        return sq.read(folder / 'SEED_COMPLETE.json')
    if folder.exists():
        raise FileExistsError(f'incomplete seed folder, inspect before rerun: {folder}')
    folder.mkdir(parents=True); screen.SEED = seed
    shutil.copyfile(protocol_file, folder / 'PROTOCOL.json')
    r0 = bind[(seed, 'R0')]
    assert sq.sha(sq.private_path(r0['path'])) == r0['sha256'], f'R0 checkpoint changed: {seed}'
    _, path, ckpt_sha, wall = adapt_one(r0['path'], lam, seed, folder, sq.SEQ_STEPS, n_steps)
    record = dict(protocol_sha256=digest, seed=seed, variant=sq.VARIANT, path=str(path), sha256=ckpt_sha, lambda_seq=lam,
                  r0_path=r0['path'], r0_sha256=r0['sha256'], train_s=wall)
    sq.write(folder / 'SEED_COMPLETE.json', record)
    print(sq.utc(), 'SEED DONE', seed, round(wall, 1), flush=True)
    return record


def smoke():
    root = sq.SMOKE / 'train'
    if root.exists(): shutil.rmtree(root)
    root.mkdir(parents=True)
    lam = sq.read(sq.LAMBDA_FILE)['lambda_seq']
    rec = train_seed(GATE_SEED, root, sq.DRAFT, 'SMOKE', lam, sq.screening_bindings(), n_steps=2)
    assert rec['variant'] == sq.VARIANT
    for f in root.rglob('*.pt'): f.unlink()
    sq.write(sq.SMOKE / 'SMOKE_TRAIN_OK.json', dict(code_sha256=sq.code_hashes(('sq_common.py', 'sq_train.py')), at=sq.utc()))
    print('SMOKE TRAIN OK', flush=True)


def run():
    protocol, digest = sq.frozen_protocol()
    sq.STUDY.mkdir(parents=True, exist_ok=True)
    bind = sq.screening_bindings()
    for seed in protocol['new_seeds']:
        train_seed(seed, sq.STUDY, sq.PROTOCOL, digest, protocol['lambda_seq'], bind)
    sq.write(sq.STUDY / 'TRAIN_COMPLETE.json', dict(protocol_sha256=digest, seeds=protocol['new_seeds'], at=sq.utc()))


def main():
    global SETUP
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['gate', 'smoke', 'run']); a = p.parse_args()
    SETUP = ot.setup()
    dict(gate=gate, smoke=smoke, run=run)[a.mode]()


if __name__ == '__main__':
    main()
