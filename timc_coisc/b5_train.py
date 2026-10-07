"""Training side of the B5 Jacobian-norm baseline.

  b5_train.py gates   reproduction gates R1-R2 (existing models only) and lambda_jac
  b5_train.py smoke   code-path test: throwaway adaptation seed 1, 2 steps
  b5_train.py run     JAC for seeds 20260922/23/24; requires the frozen protocol and EXECUTION_AUTHORIZATION.json
"""
from __future__ import annotations
import argparse
import shutil
import numpy as np
import b5_common as bc
from b5_common import oc, screen
import onset_train as ot  # noqa: E402

GATES_DIR = bc.OUT / 'gates' / 'train'
SMOKE_DIR = bc.OUT / 'smoke'
RUN_DIR = bc.OUT / 'run'


def setup():
    return ot.setup()


def gates():
    device, train, ev, fit_ids, sel_ids, stats, probs, scores, old = setup()
    if GATES_DIR.exists(): raise FileExistsError(f'preserve existing gate attempt: {GATES_DIR}')
    GATES_DIR.mkdir(parents=True)
    report = dict(label='POST-HOC B5 Jacobian baseline: pre-freeze reproduction gates on existing models only')
    stages = [{k: s[k] for k in ('steps', 'batch', 'lr', 'rollout', 'adjacent', 'endpoint')} for s in old['training']['stages']]
    cal = bc.calibrate(train, stats, probs, stages, device)
    stored = oc.read_json(ot.SCREEN_DIR_DEV / 'calibration.json')['records']
    fields = ('prediction', 'regularizer', 'pre', 'sign', 'gradient_norm')
    diffs = [abs(a[f] - b[f]) for a, b in zip(cal['prior']['records'], stored) for f in fields]
    order_ok = [(a['excitation'], a['batch']) for a in cal['prior']['records']] == [(b['excitation'], b['batch']) for b in stored]
    report['R1_calibration'] = dict(lambda_prior_reproduced=cal['prior']['lambda_s'], lambda_prior_formal=oc.LAMBDA_PRIOR,
                                    records_max_abs_diff=max(diffs), order_ok=order_ok, n_records=len(stored),
                                    pass_=bool(order_ok and len(diffs) == 5 * len(stored) and max(diffs) == 0 and cal['prior']['lambda_s'] == oc.LAMBDA_PRIOR),
                                    lambda_jac=cal['jacobian']['lambda_s'], lambda_jac_per_sampler=cal['jacobian']['per_sampler'],
                                    time_s=cal['time_s'])
    screen.save_json(GATES_DIR / 'calibration_b5.json', cal)
    print('R1', report['R1_calibration'], flush=True)
    rows = []
    for seed in bc.SEEDS:
        screen.SEED = seed
        r0, ckpt, path = oc.load_r0(seed, stats, device)
        assert oc.sha(path) == oc.read_json(oc.FORMAL / f'seed_{seed}/results/R0/trained.json')['checkpoint_sha256']
        r0_state = oc.cpu_state(r0)
        for name, terms, ref_path in (('COISC', bc.coisc_terms(oc.LAMBDA_PRIOR, oc.LAMBDA_PRIOR, oc.TAU_PRIOR), oc.FORMAL / f'seed_{seed}/coisc_final.pt'),
                                      ('L0', bc.coisc_terms(0., 0., oc.TAU_PRIOR), oc.ABLATION / f'seed_{seed}/L0_final.pt')):
            model = ot.fresh(stats, r0_state, device)
            bc.adapt(model, train, probs, old, terms, seed, device, GATES_DIR / f'R2_{name}_seed_{seed}.jsonl')
            ref = oc.load_checkpoint(ref_path)['state_dict']
            rows.append(dict(seed=seed, variant=name, reference=str(ref_path), reference_sha256=oc.sha(ref_path),
                             bitwise_equal=oc.states_equal(oc.cpu_state(model), ref), nonhead_equal=ot.nonhead_equal(model, r0_state)))
            print('R2', rows[-1], flush=True)
    report['R2_adaptation'] = dict(per_seed=rows, pass_=all(r['bitwise_equal'] and r['nonhead_equal'] for r in rows))
    report['all_pass'] = report['R1_calibration']['pass_'] and report['R2_adaptation']['pass_']
    report['code_sha256'] = {n: oc.sha(bc.HERE / n) for n in ('b5_common.py', 'b5_train.py')}
    screen.save_json(bc.PROTO / 'GATES_TRAIN_REPORT.json', report)
    print('ALL_PASS', report['all_pass'], flush=True)


def run_jac(lam, protocol_digest, root, adapt_seed=None, steps=oc.ADAPT_STEPS, seeds=bc.SEEDS):
    device, train, ev, fit_ids, sel_ids, stats, probs, scores, old = setup()
    threshold = np.asarray(old['regime_thresholds']['regime_deadband_C'])
    root.mkdir(parents=True, exist_ok=True); rows = []; bindings = []
    for seed in seeds:
        folder = root / f'seed_{seed}'
        if (folder / 'SEED_COMPLETE.json').exists():
            done = oc.read_json(folder / 'SEED_COMPLETE.json'); rows.extend(done['rows']); bindings.extend(done['bindings']); continue
        if folder.exists(): raise FileExistsError(f'incomplete seed folder, inspect before rerun: {folder}')
        folder.mkdir(parents=True); screen.SEED = seed
        r0, ckpt, path = oc.load_r0(seed, stats, device)
        assert oc.sha(path) == oc.read_json(oc.FORMAL / f'seed_{seed}/results/R0/trained.json')['checkpoint_sha256']
        r0_state = oc.cpu_state(r0); s = adapt_seed or seed
        model = ot.fresh(stats, r0_state, device)
        wall = bc.adapt(model, train, probs, old, bc.jacobian_terms(lam), s, device, folder / 'JAC_training.jsonl', steps)
        assert ot.nonhead_equal(model, r0_state)
        digest = ot.save_ckpt(model, folder / 'JAC_final.pt', seed=seed, adapt_seed=s, variant='JAC', w_jac=lam, steps=steps)
        bind = dict(seed=seed, variant='JAC', path=str(folder / 'JAC_final.pt'), sha256=digest)
        base = ot.dev_record(r0, ev, sel_ids, threshold, device); rec = ot.dev_record(model, ev, sel_ids, threshold, device)
        rec['gates_prior_masks'] = ot.dev_gates(rec, base, 'sens_prior')
        seed_rows = [dict(seed=seed, variant='R0', **base), dict(seed=seed, variant='JAC', train_s=wall, **rec)]
        screen.save_json(folder / 'SEED_COMPLETE.json', dict(protocol_sha256=protocol_digest, rows=seed_rows, bindings=[bind]))
        rows.extend(seed_rows); bindings.append(bind)
        print('SEED DONE', seed, flush=True)
    screen.save_json(root / 'B5_DEVELOPMENT_RESULTS.json', dict(protocol_sha256=protocol_digest,
                     label='POST-HOC exploratory; development gates reported, never used for stopping or selection', rows=rows))
    screen.save_json(root / 'B5_CHECKPOINT_BINDINGS.json', dict(protocol_sha256=protocol_digest, bindings=bindings))


def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['gates', 'smoke', 'run']); a = p.parse_args()
    if a.mode == 'gates':
        gates(); return
    if a.mode == 'smoke':
        if SMOKE_DIR.exists(): shutil.rmtree(SMOKE_DIR)
        lam = oc.read_json(bc.PROTO / 'GATES_TRAIN_REPORT.json')['R1_calibration']['lambda_jac']
        run_jac(lam, 'SMOKE', SMOKE_DIR, adapt_seed=1, steps=2, seeds=bc.SEEDS[:1])
        rows = oc.read_json(SMOKE_DIR / 'B5_DEVELOPMENT_RESULTS.json')['rows']
        assert [r['variant'] for r in rows] == ['R0', 'JAC']
        for f in SMOKE_DIR.rglob('*.pt'): f.unlink()
        print('SMOKE OK', flush=True); return
    protocol, digest = bc.frozen_protocol()
    run_jac(protocol['lambda_jac'], digest, RUN_DIR)


if __name__ == '__main__':
    main()
