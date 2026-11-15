"""Pre-registered analysis of the screening study. Rules are read from the protocol; nothing here is tuned on study data.

  scr_analyze.py gate      G4: analysis applied to the 13 exploratory seeds reproduces the numbers that motivated the study
  scr_analyze.py analyze   study seeds; frozen protocol and author authorization required
"""
from __future__ import annotations
import csv
import sys

import numpy as np
from scipy.stats import spearmanr

import scr_common as sc

ONSET_MIL = sc.ONSET / 'mil_execution_20261003T104638Z_23b8f69525' / 'FIRST_ACTION_CHECK.csv'


def auc(score, y):
    """Mann-Whitney AUC, ties count one half; higher score predicts y=1."""
    s = np.asarray(score, float); y = np.asarray(y, bool)
    p, n = s[y], s[~y]
    return float(((p[:, None] > n[None, :]).sum() + .5 * (p[:, None] == n[None, :]).sum()) / (len(p) * len(n)))


def boot_auc(scores, y, draws, rng_seed):
    """Stratified seed-level bootstrap; returns {name: (lo, hi)} and paired difference draws against each name."""
    y = np.asarray(y, bool); pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    rng = np.random.default_rng(rng_seed); out = {k: [] for k in scores}
    for _ in range(draws):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for k, s in scores.items():
            out[k].append(auc(np.asarray(s)[idx], y[idx]))
    return {k: np.asarray(v) for k, v in out.items()}


def wilson(k, n, z=1.959963984540054):
    if n == 0: return (None, None)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def analyze(table, fa, rules):
    """table: {seed: dict of R0 metrics}; fa: {(seed, variant, scenario_index): class}."""
    seeds = sorted(table)
    wrong = {s: sum(fa[(s, 'R0', i)] == 'wrong' for i in range(12)) for s in seeds}
    y = np.array([wrong[s] > 0 for s in seeds])
    res = dict(n_seeds=len(seeds), n_positive=int(y.sum()), n_negative=int((~y).sum()),
               r0_wrong_pairs=int(sum(wrong.values())), wrong_per_seed={str(s): wrong[s] for s in seeds})
    scores = {name: [table[s][name] for s in seeds] for name in rules['scores']}
    res['auc'] = {k: auc(v, y) for k, v in scores.items()} if 0 < y.sum() < len(y) else None
    if res['auc'] is None or min(y.sum(), (~y).sum()) < rules['min_class_size']:
        res['primary_verdict'] = 'INCONCLUSIVE_LOW_PREVALENCE'
    else:
        b = boot_auc(scores, y, rules['bootstrap_draws'], rules['bootstrap_rng_seed'])
        prim, comp = rules['primary_score'], rules['primary_comparator']
        res['auc_ci95'] = {k: np.percentile(v, [2.5, 97.5]).tolist() for k, v in b.items()}
        res['delta_auc_vs_primary'] = {k: dict(point=res['auc'][prim] - res['auc'][k],
                                               ci95=np.percentile(b[prim] - v, [2.5, 97.5]).tolist())
                                       for k, v in b.items() if k != prim}
        h1 = (res['auc'][prim] >= rules['auc_min'] and res['auc_ci95'][prim][0] > .5 and res['auc'][prim] > res['auc'][comp])
        res['H1'] = dict(auc=res['auc'][prim], ci_low=res['auc_ci95'][prim][0], comparator_auc=res['auc'][comp], pass_=bool(h1))
        res['H1b_delta_ci_excludes_0'] = bool(res['delta_auc_vs_primary'][comp]['ci95'][0] > 0)
        res['primary_verdict'] = 'SUPPORTED' if h1 else 'NOT_SUPPORTED'
    t = rules['threshold']; flag = np.array([table[s][rules['primary_score']] > t for s in seeds])
    tp, fn = int((flag & y).sum()), int((~flag & y).sum()); tn, fp = int((~flag & ~y).sum()), int((flag & ~y).sum())
    res['threshold_rule'] = dict(threshold=t, TP=tp, FN=fn, TN=tn, FP=fp,
                                 sensitivity=tp / (tp + fn) if tp + fn else None, sensitivity_ci95=wilson(tp, tp + fn),
                                 specificity=tn / (tn + fp) if tn + fp else None, specificity_ci95=wilson(tn, tn + fp))
    rho = spearmanr([table[s][rules['primary_score']] for s in seeds], [wrong[s] for s in seeds])
    res['spearman_primary_vs_wrong_count'] = dict(rho=float(rho.statistic), p=float(rho.pvalue))
    corr = {}
    for v in ('COISC', 'COISC_onset', 'L0'):
        if not all((s, v, 0) in fa for s in seeds): continue
        bad = [(s, i) for s in seeds for i in range(12) if fa[(s, 'R0', i)] == 'wrong']
        good = [(s, i) for s in seeds for i in range(12) if fa[(s, 'R0', i)] == 'correct']
        fixed = sum(fa[(s, v, i)] == 'correct' for s, i in bad); broke = sum(fa[(s, v, i)] == 'wrong' for s, i in good)
        corr[v] = dict(r0_wrong=len(bad), corrected=fixed, corrected_rate=fixed / len(bad) if bad else None,
                       corrected_ci95=wilson(fixed, len(bad)), r0_correct=len(good), made_wrong=broke,
                       made_wrong_rate=broke / len(good) if good else None, made_wrong_ci95=wilson(broke, len(good)),
                       total_wrong=sum(fa[(s, v, i)] == 'wrong' for s in seeds for i in range(12)))
        policy = sum(fa[(s, v if table[s][rules['primary_score']] > t else 'R0', i)] == 'wrong' for s in seeds for i in range(12))
        corr[v]['screen_then_adapt_total_wrong'] = policy
    res['correction'] = corr
    res['r0_total_wrong'] = res['r0_wrong_pairs']
    return res


def exploratory_inputs():
    with (sc.ONSET / 'HOLDOUT_SUMMARY.csv').open(encoding='utf-8-sig') as f:
        hold = {int(r['seed']): r for r in csv.DictReader(f) if r['variant'] == 'R0'}
    dev = {}
    for m in ('e1', 'e2'):
        for r in sc.read(sc.ONSET / m / f'{m.upper()}_DEVELOPMENT_RESULTS.json')['rows']:
            if r['variant'] == 'R0':
                dev[r['seed']] = dict(dev_Tp_WSR_onset=r['sens_onset']['Tp']['WSR'], dev_Tp_MAE_120=r['Tp_MAE']['120'],
                                      dev_macro7_MAE=r['macro7_MAE'])
    table = {s: dict({k: float(hold[s][k]) for k in ('Tp_WSR_ONSET', 'Tp_WSR_PRIOR', 'Tp_MAE_120', 'all7_MAE_5h')}, **dev[s])
             for s in sc.EXPLORATORY_SEEDS}
    with ONSET_MIL.open(encoding='utf-8') as f:
        fa = {(int(r['seed']), r['variant'], int(r['scenario_index'])): r['first_action'] for r in csv.DictReader(f)}
    return table, fa


def gate():
    rules = sc.read(sc.DRAFT)['analysis_rules']
    table, fa = exploratory_inputs()
    res = analyze(table, fa, rules)
    expect = dict(Tp_WSR_ONSET=1.0, Tp_WSR_PRIOR=1.0, Tp_MAE_120=0.7667, all7_MAE_5h=0.9333, dev_Tp_WSR_onset=0.8333, dev_Tp_MAE_120=0.6)
    ok = all(abs(res['auc'][k] - v) < 5e-5 for k, v in expect.items()) and res['n_positive'] == 3 and res['r0_wrong_pairs'] == 7
    tr = res['threshold_rule']; ok = ok and tr['FN'] == 0 and tr['FP'] == 0
    report = dict(label='Screening study gate G4: analysis code reproduces the exploratory 13-seed numbers (not evidence)',
                  exploratory_result=res, expected_auc=expect, pass_=bool(ok),
                  code_sha256=sc.code_hashes(('scr_common.py', 'scr_analyze.py')))
    sc.write(sc.PROTO / 'GATE_G4_ANALYSIS_REPORT.json', report)
    print('G4 PASS', ok, {k: round(v, 4) for k, v in res['auc'].items()}, flush=True)


def study_inputs(digest):
    hold = {r['seed']: r for r in sc.read(sc.STUDY / 'HOLDOUT_ESTIMATES.json')['rows']}
    recs = {r['seed']: r for r in sc.seed_records(sc.STUDY)}
    fa_rec = sc.read(sc.STUDY / 'FIRST_ACTION_CHECK.json'); assert fa_rec['protocol_sha256'] == digest
    for r in recs.values(): assert r['protocol_sha256'] == digest
    table = {s: dict({k: hold[s][k] for k in ('Tp_WSR_ONSET', 'Tp_WSR_PRIOR', 'Tp_MAE_120', 'all7_MAE_5h')},
                     dev_Tp_WSR_onset=recs[s]['r0_dev']['Tp_WSR_onset'], dev_Tp_MAE_120=recs[s]['r0_dev']['Tp_MAE_120'],
                     dev_macro7_MAE=recs[s]['r0_dev']['macro7_MAE']) for s in recs}
    fa = {(r['seed'], r['variant'], r['scenario_index']): r['first_action'] for r in fa_rec['rows']}
    failures = sum(not r['solver_success'] for r in fa_rec['rows'])
    return table, fa, failures


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'gate':
        gate(); return
    if mode != 'analyze': raise SystemExit(__doc__)
    protocol, digest = sc.frozen_protocol()
    table, fa, failures = study_inputs(digest)
    assert sorted(table) == list(protocol['new_seeds'])
    res = analyze(table, fa, protocol['analysis_rules'])
    res['solver_failures_first_step'] = failures
    pooled_table, pooled_fa = exploratory_inputs()
    pooled_table.update(table); pooled_fa.update(fa)
    res['descriptive_pooled_53_seeds'] = analyze(pooled_table, pooled_fa, protocol['analysis_rules'])
    sc.write(sc.STUDY / 'SCREENING_RESULTS.json', dict(protocol_sha256=digest, at=sc.utc(), per_seed=table, result=res))
    print('PRIMARY', res['primary_verdict'], res.get('H1'), flush=True)


if __name__ == '__main__':
    main()
