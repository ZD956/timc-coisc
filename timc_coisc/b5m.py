"""B5M: Jacobian-norm baseline at a weight matched to the COISC initial loss share (POSTHOC_B5M_PROTOCOL).

  b5m.py gates           R1 probe reproduction + lambda_match, R2 JAC reproduction
  b5m.py gates_holdout   R3 holdout reproduction, R4 rule self-test (fresh process: training disables TF32)
  b5m.py smoke           code-path test: throwaway adaptation seed 1, 2 steps
  b5m.py freeze          writes the frozen protocol after all gates pass
  b5m.py train|holdout   separate processes; require the frozen protocol and EXECUTION_AUTHORIZATION.json
"""
from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
import numpy as np

from _paths import PKG, PROTOCOLS, OUTPUT_ROOT, VERIFY_HASHES  # noqa: E402
import b5_common as bc  # noqa: E402
from b5_common import oc, screen, torch  # noqa: E402
import b5_holdout as bh  # noqa: E402
import onset_train as ot  # noqa: E402
import onset_holdout as oh  # noqa: E402

OUT = OUTPUT_ROOT / 'posthoc_jacobian_b5m'
PROTO = PROTOCOLS / 'posthoc_jacobian_b5m'
B5 = bc.OUT
DRAFT = PROTO / 'POSTHOC_B5M_PROTOCOL_DRAFT.json'
PROTOCOL = PROTO / 'POSTHOC_B5M_PROTOCOL.json'
PROTOCOL_SHA_FILE = PROTO / 'POSTHOC_B5M_PROTOCOL.sha256'
AUTHORIZATION = OUT / 'EXECUTION_AUTHORIZATION.json'
PROBES = 8


def probe_terms(r0, train, probs, seed, device):
    """Regularizer values on R0 for the leading SUBSET windows of the first PROBES adaptation batches of this seed."""
    screen.seed_all(seed); r0.train(); rng = np.random.default_rng(seed); out = []
    for k in range(PROBES):
        h, u, y = screen.sample(train, rng, 128, probs[False], device)
        values = screen.sensitivity(r0, h[:screen.SUBSET], u[:screen.SUBSET])
        pre, sign = oc.regularizer(values, oc.TAU_PRIOR)
        out.append(dict(seed=seed, batch=k, L_pre=float(pre.detach()), L_sign=float(sign.detach()), L_jac=float(bc.jacobian_term(values).detach())))
    r0.eval()
    return out


def lambda_match(records):
    coisc = sum(r['L_pre'] + r['L_sign'] for r in records); jac = sum(r['L_jac'] for r in records)
    return oc.LAMBDA_PRIOR * coisc / jac


def first_log(path):
    with Path(path).open(encoding='utf-8') as f:
        return json.loads(f.readline())


def gates():
    device, train, ev, fit_ids, sel_ids, stats, probs, scores, old = ot.setup()
    report = dict(label='POST-HOC B5M: pre-freeze gates on existing models and already-reported results only')
    records, r1 = [], []
    for seed in bc.SEEDS:
        screen.SEED = seed
        r0, ckpt, path = oc.load_r0(seed, stats, device)
        rec = probe_terms(r0, train, probs, seed, device); records += rec
        c = first_log(oc.FORMAL / f'seed_{seed}/coisc_training.jsonl'); j = first_log(B5 / f'run/seed_{seed}/JAC_training.jsonl')
        r1.append(dict(seed=seed, L_pre_diff=abs(rec[0]['L_pre'] - c['L_pre']), L_sign_diff=abs(rec[0]['L_sign'] - c['L_sign']),
                       L_jac_diff=abs(rec[0]['L_jac'] - j['L_jac'])))
    lam = lambda_match(records)
    report['R1_probe'] = dict(per_seed_batch1_vs_logs=r1, records=records, lambda_match=lam,
                              pass_=all(max(r['L_pre_diff'], r['L_sign_diff'], r['L_jac_diff']) == 0 for r in r1))
    print('R1', r1, 'lambda_match', lam, flush=True)
    bind = oc.read_json(B5 / 'run' / 'B5_CHECKPOINT_BINDINGS.json')['bindings']; lam_jac = oc.read_json(bc.PROTOCOL)['lambda_jac']
    gdir = OUT / 'gates'; gdir.mkdir(parents=True, exist_ok=True); r2 = []
    for b in bind:
        screen.SEED = b['seed']
        r0, ckpt, path = oc.load_r0(b['seed'], stats, device); r0_state = oc.cpu_state(r0)
        model = ot.fresh(stats, r0_state, device)
        bc.adapt(model, train, probs, old, bc.jacobian_terms(lam_jac), b['seed'], device, gdir / f'R2_JAC_seed_{b["seed"]}.jsonl')
        assert oc.sha(oc.private_path(b['path'])) == b['sha256']
        ref = oc.load_checkpoint(oc.private_path(b['path']))['state_dict']
        r2.append(dict(seed=b['seed'], bitwise_equal=oc.states_equal(oc.cpu_state(model), ref)))
        print('R2', r2[-1], flush=True)
    report['R2_adaptation'] = dict(per_seed=r2, pass_=all(r['bitwise_equal'] for r in r2))
    report['all_pass'] = report['R1_probe']['pass_'] and report['R2_adaptation']['pass_']
    report['code_sha256'] = {'b5m.py': oc.sha(PKG / 'b5m.py')}
    screen.save_json(PROTO / 'GATES_TRAIN_REPORT.json', report)
    print('ALL_PASS', report['all_pass'], flush=True)


def gates_holdout():
    """Separate process: seed_all (training) disables TF32, and all stored holdout values were computed with default backends."""
    assert torch.backends.cudnn.allow_tf32, 'holdout must run in a fresh process with default backend flags'
    torch.set_num_threads(2); device = torch.device('cuda')
    report = dict(label='POST-HOC B5M: holdout gates on existing models and already-reported results only')
    bind = oc.read_json(B5 / 'run' / 'B5_CHECKPOINT_BINDINGS.json')['bindings']
    raw, stored_s, counts, order, rec, data = oh.load_holdout()
    hstats = oc.read_json(oh.NORMALIZATION)
    per, summary = bh.estimates(oh.existing_bindings() + bind, data, counts, device, hstats)
    ref_sum = {(int(r['seed']), r['variant']): r for r in bh.read_csv(B5 / 'B5_HOLDOUT_SUMMARY.csv')}
    sum_diff = max(abs(r[k] - float(ref_sum[(r['seed'], r['variant'])][k])) for r in summary for k in r if k not in ('seed', 'variant'))
    new = bh.contrasts(per, ('R0', 'COISC', 'L0'))
    ref_con = {(int(r['seed']), r['variant'], r['reference'], r['metric']): r for r in bh.read_csv(B5 / 'B5_HOLDOUT_CONTRASTS.csv')}
    mism = [(r['seed'], r['variant'], r['reference'], r['metric'], k) for r in new for k in r
            if k not in ('seed', 'variant', 'reference', 'metric') and not bh.close(r[k], ref_con[(r['seed'], r['variant'], r['reference'], r['metric'])][k])]
    report['R3_holdout'] = dict(models=len(summary), summary_max_abs_diff=sum_diff, contrast_rows=len(new), reference_rows=len(ref_con),
                                mismatches=mism[:20], pass_=bool(sum_diff <= bh.TOL and not mism and len(new) == len(ref_con)))
    print('R3', {k: v for k, v in report['R3_holdout'].items() if k != 'mismatches'}, flush=True)
    verdicts = {v: bh.rule(new, v)['verdict'] for v in ('COISC', 'L0', 'JAC')}
    known = dict(COISC='B5_reproduces_COISC_profile', L0='B5_mixed', JAC='B5_does_not_reproduce_profile')
    report['R4_rule'] = dict(verdicts=verdicts, known=known, pass_=verdicts == known)
    print('R4', verdicts, flush=True)
    report['all_pass'] = report['R3_holdout']['pass_'] and report['R4_rule']['pass_']
    report['code_sha256'] = {'b5m.py': oc.sha(PKG / 'b5m.py')}
    screen.save_json(PROTO / 'GATES_HOLDOUT_REPORT.json', report)
    print('ALL_PASS', report['all_pass'], flush=True)


def train_jacm(lam, digest, root, adapt_seed=None, steps=oc.ADAPT_STEPS, seeds=bc.SEEDS):
    device, train, ev, fit_ids, sel_ids, stats, probs, scores, old = ot.setup()
    threshold = np.asarray(old['regime_thresholds']['regime_deadband_C'])
    root.mkdir(parents=True, exist_ok=True); rows, bindings = [], []
    for seed in seeds:
        folder = root / f'seed_{seed}'
        if (folder / 'SEED_COMPLETE.json').exists():
            done = oc.read_json(folder / 'SEED_COMPLETE.json'); rows += done['rows']; bindings += done['bindings']; continue
        if folder.exists(): raise FileExistsError(f'incomplete seed folder, inspect before rerun: {folder}')
        folder.mkdir(parents=True); screen.SEED = seed
        r0, ckpt, path = oc.load_r0(seed, stats, device)
        assert oc.sha(path) == oc.read_json(oc.FORMAL / f'seed_{seed}/results/R0/trained.json')['checkpoint_sha256']
        r0_state = oc.cpu_state(r0); s = adapt_seed or seed
        model = ot.fresh(stats, r0_state, device)
        wall = bc.adapt(model, train, probs, old, bc.jacobian_terms(lam), s, device, folder / 'JAC_M_training.jsonl', steps)
        assert ot.nonhead_equal(model, r0_state)
        sha = ot.save_ckpt(model, folder / 'JAC_M_final.pt', seed=seed, adapt_seed=s, variant='JAC_M', w_jac=lam, steps=steps)
        bind = dict(seed=seed, variant='JAC_M', path=str(folder / 'JAC_M_final.pt'), sha256=sha)
        base = ot.dev_record(r0, ev, sel_ids, threshold, device); rec = ot.dev_record(model, ev, sel_ids, threshold, device)
        rec['gates_prior_masks'] = ot.dev_gates(rec, base, 'sens_prior')
        seed_rows = [dict(seed=seed, variant='R0', **base), dict(seed=seed, variant='JAC_M', train_s=wall, **rec)]
        screen.save_json(folder / 'SEED_COMPLETE.json', dict(protocol_sha256=digest, rows=seed_rows, bindings=[bind]))
        rows += seed_rows; bindings.append(bind); print('SEED DONE', seed, flush=True)
    screen.save_json(root / 'B5M_DEVELOPMENT_RESULTS.json', dict(protocol_sha256=digest,
                     label='POST-HOC exploratory; development gates reported, never used for stopping or selection', rows=rows))
    screen.save_json(root / 'B5M_CHECKPOINT_BINDINGS.json', dict(protocol_sha256=digest, bindings=bindings))
    return bindings


def holdout(digest):
    assert torch.backends.cudnn.allow_tf32, 'holdout must run in a fresh process with default backend flags'
    b = oc.read_json(OUT / 'run' / 'B5M_CHECKPOINT_BINDINGS.json'); assert b['protocol_sha256'] == digest
    new_bindings = b['bindings']
    raw, stored_s, counts, order, rec, data = oh.load_holdout()
    stats = oc.read_json(oh.NORMALIZATION); torch.set_num_threads(2); device = torch.device('cuda')
    bindings = oh.existing_bindings() + oc.read_json(B5 / 'run' / 'B5_CHECKPOINT_BINDINGS.json')['bindings'] + new_bindings
    store = OUT / 'holdout'; store.mkdir(parents=True, exist_ok=True)
    per, summary = bh.estimates(bindings, data, counts, device, stats, store)
    rows = bh.contrasts(per, ('R0', 'COISC', 'L0', 'JAC'))
    oh.write_csv(OUT / 'B5M_HOLDOUT_SUMMARY.csv', summary); oh.write_csv(OUT / 'B5M_HOLDOUT_CONTRASTS.csv', rows)
    jm, j = bh.rule(rows, 'JAC_M'), bh.rule(rows, 'JAC')
    repro = 'B5_reproduces_COISC_profile'
    combined = ('profile_obtainable_without_lag_structure' if repro in (jm['verdict'], j['verdict'])
                else 'neither_weight_reproduces_profile' if all(v['verdict'] == 'B5_does_not_reproduce_profile' for v in (jm, j))
                else 'partial')
    screen.save_json(OUT / 'B5M_RULE_RESULT.json', dict(protocol_sha256=digest, JAC_M=jm, JAC=j, combined=combined,
                     label='POST-HOC exploratory; holdout reused, not independent confirmation; cannot change any locked verdict'))
    print('VERDICT', jm['verdict'], 'COMBINED', combined, flush=True)


def frozen_protocol():
    digest = oc.sha(PROTOCOL)
    assert not VERIFY_HASHES or PROTOCOL_SHA_FILE.read_text(encoding='utf-8').split()[0] == digest, 'protocol changed after freeze'
    protocol = oc.read_json(PROTOCOL)
    assert protocol['status'].startswith('FROZEN'), 'protocol not frozen'
    for b in protocol['code_sha256']:
        assert not VERIFY_HASHES or oc.sha(oc.private_path(b['path'])) == b['sha256'], f'code changed after freeze: {b["path"]}'
    auth = oc.read_json(AUTHORIZATION)
    assert auth.get('authorized_by_author') is True and auth.get('protocol_sha256') == digest, 'no valid author authorization'
    return protocol, digest


CODE = (PKG / 'b5m.py', PKG / 'b5_common.py', PKG / 'b5_holdout.py', PKG / 'onset_common.py', PKG / 'onset_train.py', PKG / 'onset_holdout.py', PKG / 'screen.py')
BOUND = (bc.PROTOCOL, B5 / 'run' / 'B5_CHECKPOINT_BINDINGS.json', B5 / 'B5_HOLDOUT_SUMMARY.csv',
         B5 / 'B5_HOLDOUT_CONTRASTS.csv', B5 / 'B5_RULE_RESULT.json')


def freeze():
    import time
    assert not PROTOCOL.is_file(), 'protocol already frozen'
    reports = {n: oc.read_json(PROTO / n) for n in ('GATES_TRAIN_REPORT.json', 'GATES_HOLDOUT_REPORT.json')}
    assert all(r['all_pass'] for r in reports.values()), 'a gate did not pass'
    assert all(oc.sha(PKG / 'b5m.py') == r['code_sha256']['b5m.py'] for r in reports.values()), 'b5m.py changed after the gates; rerun them'
    report = reports['GATES_TRAIN_REPORT.json'] | reports['GATES_HOLDOUT_REPORT.json']
    smoke = OUT / 'smoke' / 'B5M_DEVELOPMENT_RESULTS.json'
    assert smoke.is_file() and smoke.stat().st_mtime > (PKG / 'b5m.py').stat().st_mtime, 'rerun the smoke test with the current code'
    protocol = oc.read_json(DRAFT)
    assert protocol['lambda_match'] == report['R1_probe']['lambda_match'], 'draft lambda_match differs from the R1 gate value'
    protocol.update(version='FROZEN-1', status='FROZEN_BEFORE_B5M_OUTCOME', execution_authorized=False,
                    execution_authorization_rule="b5m.py train and b5m.py holdout (separate processes, in that order) require EXECUTION_AUTHORIZATION.json written on the author's explicit instruction, naming this file's SHA-256",
                    gate_summary={k: report[k]['pass_'] for k in ('R1_probe', 'R2_adaptation', 'R3_holdout', 'R4_rule')} | dict(smoke=True),
                    gate_reports_sha256={n: oc.sha(PROTO / n) for n in reports},
                    code_sha256=[dict(path=str(p), sha256=oc.sha(p)) for p in CODE],
                    bound_files=[dict(path=str(p), sha256=oc.sha(p)) for p in BOUND],
                    frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    PROTOCOL.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding='utf-8')
    digest = oc.sha(PROTOCOL); PROTOCOL_SHA_FILE.write_text(f'{digest}  {PROTOCOL.name}\n', encoding='utf-8')
    print(json.dumps({'frozen': PROTOCOL.name, 'sha256': digest}))


def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['gates', 'gates_holdout', 'smoke', 'freeze', 'train', 'holdout']); a = p.parse_args()
    if a.mode == 'gates':
        gates(); return
    if a.mode == 'gates_holdout':
        gates_holdout(); return
    if a.mode == 'freeze':
        freeze(); return
    if a.mode == 'smoke':
        d = OUT / 'smoke'
        if d.exists(): shutil.rmtree(d)
        lam = oc.read_json(PROTO / 'GATES_TRAIN_REPORT.json')['R1_probe']['lambda_match']
        train_jacm(lam, 'SMOKE', d, adapt_seed=1, steps=2, seeds=bc.SEEDS[:1])
        for f in d.rglob('*.pt'): f.unlink()
        print('SMOKE OK', flush=True); return
    protocol, digest = frozen_protocol()
    if a.mode == 'train':
        train_jacm(protocol['lambda_match'], digest, OUT / 'run')
    else:
        holdout(digest)


if __name__ == '__main__':
    main()
