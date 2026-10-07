"""Training side of the onset-aligned COISC study.

  onset_train.py gates   reproduction gates R1-R3 (existing models only) and lambda_onset
  onset_train.py smoke   code-path test with a throwaway seed, 2 steps (no study seed touched)
  onset_train.py e1|e2   requires the frozen protocol and EXECUTION_AUTHORIZATION.json
"""
from __future__ import annotations
import argparse
import shutil
import time
import numpy as np

import onset_common as oc
from onset_common import screen, torch
from paper_eval import predict, prediction_metrics  # noqa: E402
import formal_coisc  # noqa: E402

GATES_DIR = oc.OUT / 'gates' / 'train'
SMOKE_DIR = oc.OUT / 'smoke'


def setup():
    torch.set_num_threads(2); device = torch.device('cuda')
    train, ev, fit_ids, sel_ids, stats, probs, scores, old = screen.load_real()
    assert len(set(fit_ids)) == 91 and len(set(sel_ids)) == 18 and len(ev.history) == 11073
    return device, train, ev, fit_ids, sel_ids, stats, probs, scores, old


def sens_summary(s, lags):
    pre, post = oc.masks(lags); out = {}
    for o, label in enumerate(oc.LABELS):
        pm, qm = pre[:, o], post[:, o]; x = s[:, :, o]
        out[label] = dict(PLE=float(np.abs(x[:, pm]).mean()) if pm.any() else None,
                          WSR=float((x[:, qm] < -screen.EPS_SIGN).mean()) if qm.any() else None,
                          PSM=float(np.median(np.abs(x[:, qm]))) if qm.any() else None)
    return out


def dev_record(model, ev, sel_ids, threshold, device):
    pred = predict(model, ev.history, ev.future, device); sens = formal_coisc.sensitivity_all(model, ev, device)
    tp = {h: float(np.abs(pred[:, h // 10 - 1, 6] - ev.target[:, h // 10 - 1, 6]).mean()) for h in (10, 30, 60, 90, 120)}
    macro = prediction_metrics(pred, ev.target, ev.origin, sel_ids, threshold)[2]
    return dict(Tp_MAE=tp, macro7_MAE=macro, sens_onset=sens_summary(sens, oc.TAU_ONSET), sens_prior=sens_summary(sens, oc.TAU_PRIOR))


def dev_gates(rec, base, key):
    s, b = rec[key]['Tp'], base[key]['Tp']
    g = dict(Tp_120s_MAE=rec['Tp_MAE'][120] <= base['Tp_MAE'][120] + .005,
             PLE=None if s['PLE'] is None else s['PLE'] <= .9 * b['PLE'],
             WSR=s['WSR'] <= .9 * b['WSR'], PSM=s['PSM'] >= max(.2 * b['PSM'], .0005))
    g['all_defined_pass'] = all(v for v in g.values() if v is not None)
    return g


def fresh(stats, state, device):
    m = screen.build_model('B4', stats).to(device); m.load_state_dict(state); return m


def save_ckpt(model, path, **meta):
    torch.save(dict(state_dict=oc.cpu_state(model), **meta), path)
    return oc.sha(path)


def nonhead_equal(model, state):
    cur = model.state_dict()
    return all(torch.equal(cur[k].detach().cpu(), state[k].detach().cpu()) for k in state if not k.startswith('head.'))


def gates():
    device, train, ev, fit_ids, sel_ids, stats, probs, scores, old = setup()
    if GATES_DIR.exists(): raise FileExistsError(f'preserve existing gate attempt: {GATES_DIR}')
    GATES_DIR.mkdir(parents=True)
    report = dict(label='POST-HOC onset-aligned study: pre-freeze reproduction gates on existing models only')
    stages = [{k: s[k] for k in ('steps', 'batch', 'lr', 'rollout', 'adjacent', 'endpoint')} for s in old['training']['stages']]
    # R1: calibration
    cal = oc.calibrate_both(train, stats, probs, stages, device)
    stored = oc.read_json(SCREEN_DIR_DEV / 'calibration.json')['records']
    fields = ('prediction', 'regularizer', 'pre', 'sign', 'gradient_norm')
    diffs = [abs(a[f] - b[f]) for a, b in zip(cal['prior']['records'], stored) for f in fields]
    order_ok = [(a['excitation'], a['batch']) for a in cal['prior']['records']] == [(b['excitation'], b['batch']) for b in stored]
    report['R1_calibration'] = dict(lambda_prior_reproduced=cal['prior']['lambda_s'], lambda_prior_formal=oc.LAMBDA_PRIOR,
                                    exact=cal['prior']['lambda_s'] == oc.LAMBDA_PRIOR, records_max_abs_diff=max(diffs), order_ok=order_ok,
                                    pass_=bool(order_ok and max(diffs) == 0 and cal['prior']['lambda_s'] == oc.LAMBDA_PRIOR),
                                    lambda_onset=cal['onset']['lambda_s'], lambda_onset_per_sampler=cal['onset']['per_sampler'],
                                    time_s=cal['time_s'])
    screen.save_json(GATES_DIR / 'calibration_both.json', cal)
    print('R1', {k: v for k, v in report['R1_calibration'].items()}, flush=True)
    # R2: adaptation harness
    r2 = []
    for seed in oc.E1_SEEDS:
        screen.SEED = seed
        model, ckpt, path = oc.load_r0(seed, stats, device)
        assert oc.sha(path) == oc.read_json(oc.FORMAL / f'seed_{seed}/results/R0/trained.json')['checkpoint_sha256']
        oc.adapt(model, train, probs, old, oc.LAMBDA_PRIOR, oc.LAMBDA_PRIOR, oc.TAU_PRIOR, seed, device, GATES_DIR / f'R2_seed_{seed}.jsonl')
        ref = oc.load_checkpoint(oc.FORMAL / f'seed_{seed}/coisc_final.pt', map_location='cpu')['state_dict']
        r2.append(dict(seed=seed, bitwise_equal=oc.states_equal(oc.cpu_state(model), ref), nonhead_equal=nonhead_equal(model, ckpt['state_dict'])))
        print('R2', r2[-1], flush=True)
    report['R2_adaptation'] = dict(per_seed=r2, pass_=all(r['bitwise_equal'] for r in r2))
    # R3: trainer
    seed = oc.E1_SEEDS[0]; folder = GATES_DIR / 'R3_trainer'; folder.mkdir()
    shutil.copyfile(oc.DRAFT, folder / 'PROTOCOL.json')
    screen.SEED = seed
    cfg = screen.configuration(train, ev, fit_ids, sel_ids, stats, probs, scores, old)
    threshold = np.asarray(old['regime_thresholds']['regime_deadband_C'])
    t = time.perf_counter()
    _, result, run, _ = screen.train_variant('R0', {'excitation': False, 'dbasr': False}, train, ev, sel_ids, stats, probs, cfg, 0., threshold, device, folder, False)
    new = oc.load_checkpoint(run / 'selected.pt', map_location='cpu')
    ref = oc.load_checkpoint(oc.FORMAL / f'seed_{seed}/results/R0/selected.pt', map_location='cpu')
    report['R3_trainer'] = dict(seed=seed, state_dict_bitwise_equal=oc.states_equal(new['state_dict'], ref['state_dict']),
                                selected_equal=new['selected'] == ref['selected'], file_sha_equal=oc.sha(run / 'selected.pt') == oc.sha(oc.FORMAL / f'seed_{seed}/results/R0/selected.pt'),
                                wall_s=time.perf_counter() - t)
    report['R3_trainer']['pass_'] = bool(report['R3_trainer']['state_dict_bitwise_equal'] and report['R3_trainer']['selected_equal'])
    report['all_pass'] = all(report[k]['pass_'] for k in ('R1_calibration', 'R2_adaptation', 'R3_trainer'))
    report['code_sha256'] = {n: oc.sha(oc.HERE / n) for n in ('onset_common.py', 'onset_train.py')}
    screen.save_json(oc.PROTO / 'GATES_TRAIN_REPORT.json', report)
    print('ALL_PASS', report['all_pass'], 'R3', report['R3_trainer'], flush=True)


SCREEN_DIR_DEV = oc.SCREEN_DEV


def adapt_variants(r0_state, which, train, probs, old, lam_onset, seed, device, stats, folder, steps):
    spec = dict(COISC=(oc.LAMBDA_PRIOR, oc.LAMBDA_PRIOR, oc.TAU_PRIOR),
                COISC_onset=(lam_onset, lam_onset, oc.TAU_ONSET),
                COISC_onset_lamprior=(oc.LAMBDA_PRIOR, oc.LAMBDA_PRIOR, oc.TAU_ONSET),
                L0=(0., 0., oc.TAU_PRIOR))
    out = {}
    for name in which:
        w_pre, w_sign, lags = spec[name]
        model = fresh(stats, r0_state, device)
        wall = oc.adapt(model, train, probs, old, w_pre, w_sign, lags, seed, device, folder / f'{name}_training.jsonl', steps)
        assert nonhead_equal(model, r0_state)
        digest = save_ckpt(model, folder / f'{name}_final.pt', seed=seed, variant=name, w_pre=w_pre, w_sign=w_sign, lag_seconds=list(lags), steps=steps)
        out[name] = (model, dict(path=str(folder / f'{name}_final.pt'), sha256=digest, train_s=wall))
    return out


def run_study(mode, seeds, lam_onset, protocol_digest, root, smoke=False):
    device, train, ev, fit_ids, sel_ids, stats, probs, scores, old = setup()
    threshold = np.asarray(old['regime_thresholds']['regime_deadband_C'])
    steps = 2 if smoke else oc.ADAPT_STEPS
    root.mkdir(parents=True, exist_ok=True); rows = []; bindings = []
    for seed in seeds:
        folder = root / f'seed_{seed}'
        if (folder / 'SEED_COMPLETE.json').exists():
            done = oc.read_json(folder / 'SEED_COMPLETE.json'); rows.extend(done['rows']); bindings.extend(done['bindings']); continue
        if folder.exists(): raise FileExistsError(f'incomplete seed folder, inspect before rerun: {folder}')
        folder.mkdir(parents=True); screen.SEED = seed
        if mode == 'e1':
            r0, ckpt, path = oc.load_r0(seed, stats, device); r0_bind = dict(path=str(path), sha256=oc.sha(path)); which = ('COISC_onset', 'COISC_onset_lamprior')
        else:
            shutil.copyfile(oc.PROTOCOL if not smoke else oc.DRAFT, folder / 'PROTOCOL.json')
            cfg = screen.configuration(train, ev, fit_ids, sel_ids, stats, probs, scores, old)
            r0, result, run, _ = screen.train_variant('R0', {'excitation': False, 'dbasr': False}, train, ev, sel_ids, stats, probs, cfg, 0., threshold, device, folder, smoke)
            r0.eval(); r0_bind = dict(path=str(run / 'selected.pt'), sha256=result['checkpoint_sha256']); which = ('COISC', 'COISC_onset', 'L0')
        r0_state = oc.cpu_state(r0)
        seed_bind = [dict(seed=seed, variant='R0', **r0_bind)]
        models = adapt_variants(r0_state, which, train, probs, old, lam_onset, seed, device, stats, folder, steps)
        seed_bind += [dict(seed=seed, variant=n, **{k: v for k, v in b.items() if k != 'train_s'}) for n, (m, b) in models.items()]
        base = dev_record(r0, ev, sel_ids, threshold, device)
        seed_rows = [dict(seed=seed, variant='R0', **base)]
        for name, (model, b) in models.items():
            rec = dev_record(model, ev, sel_ids, threshold, device)
            rec['gates_onset_masks'] = dev_gates(rec, base, 'sens_onset'); rec['gates_prior_masks'] = dev_gates(rec, base, 'sens_prior')
            seed_rows.append(dict(seed=seed, variant=name, train_s=b['train_s'], **rec))
        screen.save_json(folder / 'SEED_COMPLETE.json', dict(protocol_sha256=protocol_digest, rows=seed_rows, bindings=seed_bind))
        rows.extend(seed_rows); bindings.extend(seed_bind)
        print('SEED DONE', mode, seed, flush=True)
    screen.save_json(root / f'{mode.upper()}_DEVELOPMENT_RESULTS.json', dict(protocol_sha256=protocol_digest, smoke=smoke,
                     label='POST-HOC exploratory; development gates reported, never used for stopping or selection', rows=rows))
    screen.save_json(root / f'{mode.upper()}_CHECKPOINT_BINDINGS.json', dict(protocol_sha256=protocol_digest, bindings=bindings))


def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['gates', 'smoke', 'e1', 'e2']); a = p.parse_args()
    if a.mode == 'gates':
        gates(); return
    if a.mode == 'smoke':
        if SMOKE_DIR.exists(): shutil.rmtree(SMOKE_DIR)
        lam = oc.read_json(oc.PROTO / 'GATES_TRAIN_REPORT.json')['R1_calibration']['lambda_onset']
        run_study('e2', (1,), lam, 'SMOKE', SMOKE_DIR, smoke=True)
        rows = oc.read_json(SMOKE_DIR / 'E2_DEVELOPMENT_RESULTS.json')['rows']
        assert [r['variant'] for r in rows] == ['R0', 'COISC', 'COISC_onset', 'L0']
        for f in SMOKE_DIR.rglob('*.pt'): f.unlink()
        print('SMOKE OK', flush=True); return
    protocol, digest = oc.frozen_protocol()
    lam = protocol['lambda_onset']
    if a.mode == 'e1':
        run_study('e1', oc.E1_SEEDS, lam, digest, oc.OUT / 'e1')
    else:
        run_study('e2', oc.E2_SEEDS, lam, digest, oc.OUT / 'e2')


if __name__ == '__main__':
    main()
