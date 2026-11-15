"""Post-hoc trajectory analyses of the stage-S4 closed-loop runs quoted in Section 5.5 of the Technical Gazette article (descriptive).

  python timc_coisc/posthoc_tg.py           recompute every quoted number from results/posthoc_tg/POSTHOC_RUN_STATS.csv
  python timc_coisc/posthoc_tg.py export    rebuild that table from the raw S4 runs ($TIMC_OUTPUT_ROOT/closedloop/study/raw_runs)

None of these analyses was prespecified. A reversal is an update whose preceding T_p is more than BAND from 37 C and whose
applied command moves away from 37 C. 'plant' judges T_p by the plant output before the update (step 0: initial T_p);
'internal' judges it by the controller's own one-step prediction made at the previous update (updates 2-60). Rates are pooled
within a plant family and averaged over the four families per seed, unless stated otherwise. Bootstrap streams and draw counts
are those of the original exploration scripts, so the printed numbers match them digit for digit.
"""
from __future__ import annotations
import csv
import json
import sys
from collections import defaultdict

import numpy as np

from _paths import OUTPUT_ROOT, RESULTS

RAW = OUTPUT_ROOT / 'closedloop' / 'study' / 'raw_runs'
TABLE = RESULTS / 'posthoc_tg' / 'POSTHOC_RUN_STATS.csv'
METRICS = RESULTS / 'closedloop' / 'CLOSEDLOOP_RUN_METRICS.csv'
PANEL_B = {0, 5, 11, 8, 20, 23}
VARIANTS = ('R0', 'L0', 'COISC')
LEGACY = dict(applied_UH='applied_H2', plant_Tp='plant_C8', internal_predicted_Tp='internal_predicted_C8',
              previous_uh_percent='previous_h2_percent', initial_Tp_C='initial_C8_C')


class _Run(dict):
    """Run arrays or metadata under the released key names; runs written before the public renaming use the legacy names."""
    def __missing__(self, key):
        return dict.__getitem__(self, LEGACY[key])


def run_stats(meta, z):
    u = z['applied_UH'].astype(np.float64)
    tp = z['plant_Tp'].astype(np.float64)
    est = z['internal_predicted_Tp'][:, 0].astype(np.float64)
    prev = np.concatenate([[float(meta['previous_uh_percent'])], u[:-1]])
    du = u - prev
    move = np.sign(np.round(du, 9))
    err = 37. - np.concatenate([[float(meta['initial_Tp_C'])], tp[:-1]])
    out = {}
    for band in (1.0, 2.0):
        far = np.abs(err) > band
        out[f'far_{band:g}'] = int(far.sum())
        out[f'rev_{band:g}'] = int((far & (move == -np.sign(err))).sum())
    far = np.abs(err) > 2.
    out['far_at_correcting_bound'] = int((far & (((err > 0) & (prev >= 100 - 1e-9)) | ((err < 0) & (prev <= 1e-9)))).sum())
    e_int = 37. - est[:-1]
    far_int = np.abs(e_int) > 2.  # NaN (solver failure) compares False
    out['ifar'] = int(far_int.sum())
    out['irev'] = int((far_int & (move[1:] == -np.sign(e_int))).sum())
    out['ifar_sign_disagree'] = int((far_int & (np.sign(e_int) != np.sign(37. - tp[:-1]))).sum())
    out['internal_nan'] = int(np.isnan(est).sum())
    hi, lo = u >= 100., u <= 0.
    out['updates'] = int(len(u))
    out['sat'] = int((hi | lo).sum())
    corr, opp = (hi & (err > 0)) | (lo & (err < 0)), (hi & (err < 0)) | (lo & (err > 0))
    out['sat_correcting'] = int(corr.sum())
    out['sat_opposite'] = int(opp.sum())
    out['sat_correcting_far'] = int((corr & far).sum())
    out['sat_near'] = int(((hi | lo) & ~far).sum())
    interior = (u > 0) & (u < 100) & (prev > 0) & (prev < 100)
    nz = np.round(du, 9) != 0
    sg = np.sign(np.round(du, 9))[nz]
    out['tv'] = repr(float(np.abs(du).sum()))
    out['tv_interior'] = repr(float(np.abs(du[interior]).sum()))
    out['interior_updates'] = int(interior.sum())
    out['moves'] = int(nz.sum())
    out['direction_changes'] = int((sg[1:] != sg[:-1]).sum()) if len(sg) > 1 else 0
    out['bound_to_bound'] = int((((prev <= 0) & (u >= 100)) | ((prev >= 100) & (u <= 0))).sum())
    out['mean_u'] = repr(float(u.mean()))
    return out


def export():
    runs = sorted(RAW.glob('*.npz'))
    if len(runs) != 8964:
        raise FileNotFoundError(f'posthoc_tg.py export needs the 8964 private S4 runs (found {len(runs)} in {RAW}; set TIMC_OUTPUT_ROOT).')
    rows = []
    for f in runs:
        with np.load(f, allow_pickle=False) as z:
            meta = _Run(json.loads(str(z['metadata'])))
            data = _Run({k: z[k] for k in z.files if k != 'metadata'})
        rows.append(dict(variant=meta['controller_variant'], seed=meta['controller_seed'], plant_index=meta['plant_index'],
                         family=meta['family'], scenario=meta['scenario_label'], **run_stats(meta, data)))
    TABLE.parent.mkdir(parents=True, exist_ok=True)
    with TABLE.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\r\n')
        w.writeheader()
        w.writerows(rows)
    print('wrote', TABLE, len(rows), 'runs')


def load():
    with TABLE.open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            if k not in ('variant', 'family', 'scenario'):
                r[k] = float(v) if k in ('tv', 'tv_interior', 'mean_u') else int(v)
    return rows


def family_equal(acc, seeds, fams, v, num=0, den=1):
    return np.array([np.mean([acc[(s, v, f)][num] / max(acc[(s, v, f)][den], 1) for f in fams]) for s in seeds])


def report(rows):
    seeds = sorted({r['seed'] for r in rows})
    fams = sorted({r['family'] for r in rows})
    print('# reversal fraction, plant-judged, all 24 plants (seed means, family-equal)')
    rng = np.random.default_rng(0)
    rate2 = None
    for band in ('1', '2'):
        acc = defaultdict(lambda: [0, 0])
        for r in rows:
            a = acc[(r['seed'], r['variant'], r['family'])]
            a[0] += r[f'rev_{band}']; a[1] += r[f'far_{band}']
        rate = {v: family_equal(acc, seeds, fams, v) for v in VARIANTS}
        for a, b in (('COISC', 'R0'), ('L0', 'R0'), ('COISC', 'L0')):
            d = rate[a] - rate[b]
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(20000)]
            print(f'band {band}.0 {a}-{b}: mean {d.mean():.4f} CI [{np.percentile(bs, 2.5):.4f}; {np.percentile(bs, 97.5):.4f}] '
                  f'lower in {int((d < 0).sum())}/{len(d)} seeds; ratio {rate[a].mean() / rate[b].mean():.2f}')
        print(' means', {v: round(float(rate[v].mean()), 4) for v in rate})
        rate2 = rate if band == '2' else rate2

    print('# by family: reversal, share of far updates held at the correcting bound, correctly directed moves')
    fam = defaultdict(lambda: np.zeros(3))
    for r in rows:
        fam[(r['variant'], r['family'])] += [r['rev_2'], r['far_2'], r['far_at_correcting_bound']]
    for f in fams:
        print(f, '  '.join(f"{v}: rev {fam[(v, f)][0] / fam[(v, f)][1]:.3f} atbound {fam[(v, f)][2] / fam[(v, f)][1]:.2f}" for v in VARIANTS))

    print('# Spearman correlation across seeds: COISC-R0 reversal fraction vs COISC-R0 tracking MAE')
    from scipy.stats import spearmanr
    mae = defaultdict(list)
    with METRICS.open(encoding='utf-8') as f:
        for r in csv.DictReader(f):
            mae[(int(r['seed']), r['variant'], r['family'])].append(float(r['MAE_track']))
    dm = np.array([np.mean([np.mean(mae[(s, 'COISC', f)]) for f in fams]) - np.mean([np.mean(mae[(s, 'R0', f)]) for f in fams])
                   for s in seeds])
    print('spearman', round(float(spearmanr(rate2['COISC'] - rate2['R0'], dm).statistic), 4))

    print("# reversal fraction judged by the controller's own one-step T_p estimate (updates 2-60)")
    acc = defaultdict(lambda: np.zeros(4))
    for r in rows:
        acc[(r['seed'], r['variant'], r['family'])] += [r['irev'], r['ifar'], r['ifar'] - r['ifar_sign_disagree'], r['internal_nan']]
    rate = {v: family_equal(acc, seeds, fams, v) for v in sorted(VARIANTS)}
    for v in sorted(VARIANTS):
        sel = [a for k, a in acc.items() if k[1] == v]
        print(v, 'n seeds', len(rate[v]), 'mean %.4f' % rate[v].mean(),
              'sign agreement with plant %.3f' % (sum(a[2] for a in sel) / sum(a[1] for a in sel)), 'nan', int(sum(a[3] for a in sel)))
    rng = np.random.default_rng(0)
    for a, b in (('COISC', 'R0'), ('L0', 'R0'), ('COISC', 'L0')):
        d = rate[a] - rate[b]
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(20000)]
        print(f'{a}-{b}: {d.mean():.4f} CI [{np.percentile(bs, 2.5):.4f}; {np.percentile(bs, 97.5):.4f}] lower {int((d < 0).sum())}/{len(d)} '
              f'ratio {rate[a].mean() / rate[b].mean():.2f}')
    far = sum(r['ifar'] for r in rows)
    print('far updates (internal)', far, 'sign disagreements with the plant', sum(r['ifar_sign_disagree'] for r in rows))

    b_rows = [r for r in rows if r['plant_index'] in PANEL_B]
    print('# panel B: saturated updates by bound and distance from the setpoint (fractions of all updates)')
    sat = defaultdict(lambda: np.zeros(6))
    for r in b_rows:
        sat[r['variant']] += [r['updates'], r['sat'], r['sat_correcting'], r['sat_opposite'], r['sat_correcting_far'], r['sat_near']]
    for v in VARIANTS:
        n, s, c, o, cf, sn = sat[v]
        print(v, f'sat {s / n:.3f} correcting {c / n:.3f} opposite {o / n:.3f} correcting&far {cf / n:.3f} sat&near(<=2C) {sn / n:.3f}')

    print('# panel B: smoothness of u_h away from the bounds (pooled over runs)')
    sm = {v: defaultdict(list) for v in VARIANTS}
    for r in b_rows:
        a = sm[r['variant']]
        for k in ('tv', 'tv_interior', 'interior_updates', 'moves', 'direction_changes', 'bound_to_bound', 'mean_u', 'seed'):
            a[k].append(r[k])
    for v in VARIANTS:
        a = {k: np.asarray(x, float) for k, x in sm[v].items()}
        print(v, f"TV {a['tv'].mean():.1f}  TV_interior {a['tv_interior'].mean():.1f}  interior_updates {a['interior_updates'].mean():.1f}"
              f"  TV_per_interior_update {a['tv_interior'].sum() / a['interior_updates'].sum():.2f}  moves {a['moves'].mean():.1f}"
              f"  direction_changes {a['direction_changes'].mean():.2f}  bound_to_bound {a['bound_to_bound'].mean():.3f}  mean_u {a['mean_u'].mean():.1f}")
    for k in ('direction_changes', 'tv_interior', 'moves'):
        per = {v: defaultdict(list) for v in ('R0', 'COISC')}
        for v in per:
            for s, x in zip(sm[v]['seed'], sm[v][k]):
                per[v][s].append(x)
        d = np.array([np.mean(per['COISC'][s]) - np.mean(per['R0'][s]) for s in sorted(per['R0'])])
        rng = np.random.default_rng(1)
        bs = rng.choice(d, (20000, len(d))).mean(1)
        print(k, f'COISC-R0 {d.mean():.3f} [{np.percentile(bs, 2.5):.3f}; {np.percentile(bs, 97.5):.3f}] lower on {int((d < 0).sum())}/{len(d)}')


def main():
    if sys.argv[1:] == ['export']:
        export()
    elif not sys.argv[1:]:
        report(load())
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
