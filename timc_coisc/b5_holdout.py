"""Holdout (delivered-23) inference and the declared B5 rule.

  b5_holdout.py gate   R3: existing R0/COISC/L0 reproduce the onset-study holdout summaries and contrasts;
                       R4: the declared rule returns the known answers on already-reported models
  b5_holdout.py run    adds JAC; requires the frozen protocol and author authorization
"""
from __future__ import annotations
import argparse
import csv
import numpy as np
import b5_common as bc
from b5_common import oc, screen, torch
import onset_holdout as oh  # noqa: E402

METRICS = oh.CONTRAST_METRICS
TOL = 1e-9


def contrasts(per, refs):
    out = []
    for (seed, v), (e, bt) in per.items():
        for ref in refs:
            if v == ref or (seed, ref) not in per: continue
            e0, b0 = per[(seed, ref)]
            for m in METRICS:
                lo, hi = oh.percentile_ci(bt[m] - b0[m])
                row = dict(seed=seed, variant=v, reference=ref, metric=m, value=e[m], ref_value=e0[m], diff=e[m] - e0[m],
                           ratio=e[m] / e0[m] if e0[m] else None, CI_low=lo, CI_high=hi)
                if m.startswith('Tp_MAE') and ref == 'R0': row['NI_0.005'] = bool(hi <= .005)
                if ('_WSR_' in m or '_PLE_' in m) and ref == 'R0':
                    row['CI_high_minus_0.9R0'] = float(oh.percentile_ci(bt[m] - .9 * b0[m])[1])
                if '_PSM_' in m and ref == 'R0':
                    row['CI_low_minus_0.2R0'] = float(oh.percentile_ci(bt[m] - .2 * b0[m])[0])
                out.append(row)
    return out


def rule(rows, variant):
    """Declared B5 rule on holdout contrasts (prior masks, T_p); point estimates per seed."""
    get = {(int(r['seed']), r['reference'], r['metric']): r for r in rows if r['variant'] == variant}
    per_seed = []
    for seed in bc.SEEDS:
        d1 = get[(seed, 'R0', 'Tp_PLE_PRIOR')]['ratio'] <= .9
        d2 = get[(seed, 'R0', 'Tp_WSR_PRIOR')]['ratio'] <= .9
        d3 = get[(seed, 'R0', 'Tp_PSM_PRIOR')]['ratio'] >= .2
        acc = get[(seed, 'COISC', 'Tp_MAE_120')]['diff'] <= 0 if (seed, 'COISC', 'Tp_MAE_120') in get else True
        per_seed.append(dict(seed=seed, D1_PLE=bool(d1), D2_WSR=bool(d2), D3_PSM=bool(d3), A_not_worse_than_COISC=bool(acc)))
    n = {k: sum(r[k] for r in per_seed) for k in ('D1_PLE', 'D2_WSR', 'D3_PSM', 'A_not_worse_than_COISC')}
    profile = all(r['D1_PLE'] and r['D2_WSR'] and r['D3_PSM'] for r in per_seed)
    if profile and n['A_not_worse_than_COISC'] >= 2: verdict = 'B5_reproduces_COISC_profile'
    elif profile: verdict = 'B5_reproduces_profile_at_higher_accuracy_cost'
    elif n['D2_WSR'] <= 1 or n['D3_PSM'] <= 1: verdict = 'B5_does_not_reproduce_profile'
    else: verdict = 'B5_mixed'
    return dict(variant=variant, per_seed=per_seed, counts=n, verdict=verdict)


def estimates(bindings, data, counts, device, stats, store=None):
    per, summary = {}, []
    for b in bindings:
        assert oc.sha(oc.private_path(b['path'])) == b['sha256'], f'hash mismatch {b["seed"]} {b["variant"]}'
        model = oh.load_model(oc.private_path(b['path']), stats, device); arrays = {}

        def keep(i, pred, sens):
            arrays[f'src{i:02d}_pred_Tp'] = pred[:, :, oh.TP].astype(np.float32); arrays[f'src{i:02d}_S_Tp_T6'] = sens[:, :, [oh.TP, oh.T6]].astype(np.float32)
        est, boot = oh.model_estimates(model, data, counts, device, keep if store else None)
        if store: np.savez_compressed(store / f'seed_{b["seed"]}_{b["variant"]}.npz', **arrays)
        per[(b['seed'], b['variant'])] = (est, boot); summary.append(dict(seed=b['seed'], variant=b['variant'], **est))
        print('HOLDOUT', b['seed'], b['variant'], flush=True)
    return per, summary


def read_csv(path):
    with path.open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def close(a, b):
    if b in ('', None): return a is None or a == ''
    if isinstance(a, bool): return str(a) == b
    return abs(float(a) - float(b)) <= TOL


def gate():
    raw, stored_s, counts, order, rec, data = oh.load_holdout()
    stats = oc.read_json(oh.NORMALIZATION); torch.set_num_threads(2); device = torch.device('cuda')
    per, summary = estimates(oh.existing_bindings(), data, counts, device, stats)
    ref_sum = {(int(r['seed']), r['variant']): r for r in read_csv(oc.OUT / 'HOLDOUT_SUMMARY.csv')}
    sum_diff = max(abs(r[k] - float(ref_sum[(r['seed'], r['variant'])][k])) for r in summary for k in r if k not in ('seed', 'variant'))
    new = contrasts(per, ('R0', 'COISC'))
    ref_con = {(int(r['seed']), r['variant'], r['reference'], r['metric']): r for r in read_csv(oc.OUT / 'HOLDOUT_CONTRASTS.csv')}
    mism = [(r['seed'], r['variant'], r['reference'], r['metric'], k) for r in new
            for k in r if k not in ('seed', 'variant', 'reference', 'metric')
            and not close(r[k], ref_con[(r['seed'], r['variant'], r['reference'], r['metric'])][k])]
    r3 = dict(models=len(summary), summary_max_abs_diff=sum_diff, contrast_rows=len(new), contrast_mismatches=mism[:20],
              pass_=bool(sum_diff <= TOL and not mism and len(new) == 12 * len(METRICS)))
    print('R3', {k: v for k, v in r3.items() if k != 'contrast_mismatches'}, flush=True)
    coisc, l0 = rule(new, 'COISC'), rule(new, 'L0')
    good, cases = ((.5, .5, 1.), -1), []
    cases.append(([good] * 3, 'B5_reproduces_COISC_profile'))
    cases.append(([((.5, .5, 1.), 1)] * 3, 'B5_reproduces_profile_at_higher_accuracy_cost'))
    cases.append(([((.5, .95, 1.), -1)] * 2 + [good], 'B5_does_not_reproduce_profile'))
    cases.append(([((.5, .5, .1), -1)] * 2 + [good], 'B5_does_not_reproduce_profile'))
    cases.append(([((.95, .5, 1.), -1)] * 3, 'B5_mixed'))
    synth = []
    for seed_vals, expect in cases:
        rows = []
        for seed, ((ple, wsr, psm), da) in zip(bc.SEEDS, seed_vals):
            rows += [dict(seed=seed, variant='X', reference='R0', metric=m, ratio=v) for m, v in (('Tp_PLE_PRIOR', ple), ('Tp_WSR_PRIOR', wsr), ('Tp_PSM_PRIOR', psm))]
            rows.append(dict(seed=seed, variant='X', reference='COISC', metric='Tp_MAE_120', diff=da))
        synth.append(rule(rows, 'X')['verdict'] == expect)
    r4 = dict(COISC=coisc, L0=l0, known_answers=dict(COISC='B5_reproduces_COISC_profile', L0='B5_mixed'), synthetic_branches=synth,
              pass_=bool(coisc['verdict'] == 'B5_reproduces_COISC_profile' and l0['verdict'] == 'B5_mixed' and all(synth)))
    print('R4', r4['pass_'], coisc['verdict'], l0['verdict'], synth, flush=True)
    report = dict(label='POST-HOC B5 Jacobian baseline: holdout gates on existing models only', R3_holdout_reproduction=r3, R4_rule_selftest=r4,
                  all_pass=r3['pass_'] and r4['pass_'], code_sha256={n: oc.sha(bc.HERE / n) for n in ('b5_common.py', 'b5_holdout.py')})
    screen.save_json(bc.PROTO / 'GATES_HOLDOUT_REPORT.json', report)
    print('ALL_PASS', report['all_pass'], flush=True)


def run():
    protocol, digest = bc.frozen_protocol()
    raw, stored_s, counts, order, rec, data = oh.load_holdout()
    stats = oc.read_json(oh.NORMALIZATION); torch.set_num_threads(2); device = torch.device('cuda')
    b = oc.read_json(bc.OUT / 'run' / 'B5_CHECKPOINT_BINDINGS.json'); assert b['protocol_sha256'] == digest
    bindings = oh.existing_bindings() + b['bindings']
    store = bc.OUT / 'holdout'; store.mkdir(parents=True, exist_ok=True)
    per, summary = estimates(bindings, data, counts, device, stats, store)
    rows = contrasts(per, ('R0', 'COISC', 'L0'))
    oh.write_csv(bc.OUT / 'B5_HOLDOUT_SUMMARY.csv', summary); oh.write_csv(bc.OUT / 'B5_HOLDOUT_CONTRASTS.csv', rows)
    screen.save_json(bc.OUT / 'B5_RULE_RESULT.json', dict(protocol_sha256=digest, result=rule(rows, 'JAC'),
                     label='POST-HOC exploratory; holdout reused, not independent confirmation; cannot change any locked verdict'))


def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['gate', 'run']); a = p.parse_args()
    gate() if a.mode == 'gate' else run()


if __name__ == '__main__':
    main()
