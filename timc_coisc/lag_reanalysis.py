"""Post-hoc alternative-lag reanalysis of existing sensitivity arrays (no training). See POSTHOC_ABLATION_PROTOCOL.json item3."""
from __future__ import annotations
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np

import screen  # noqa: E402
import torch  # noqa: E402
import formal_coisc  # noqa: E402
from frozen_evaluate import SourceMedian, draw_means, percentile_ci  # noqa: E402

from _paths import PROTOCOLS, OUTPUT_ROOT, load_checkpoint  # noqa: E402

VERIFY_HASHES = os.environ.get('TIMC_VERIFY_HASHES') == '1'
OUT = OUTPUT_ROOT / 'posthoc_ablation'
PROTOCOL_SHA = '15d873cc468b0cab777a3ab0ec05b1a4919049480951a5870b4423081705cab9'
FORMAL = formal_coisc.OUT
HOLD = OUTPUT_ROOT / 'delivered23' / 'delivered23_20260926T111610620117Z_0415da4e90'
SEEDS = ('20260922', '20260923', '20260924')
POINTS = ('T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'Tp')
T = np.arange(1, 13) * 10
EPS = screen.EPS_SIGN
ABLATIONS = ('L0', 'SIGN_B', 'SIGN_A', 'PRE_B')


def masks(spec):
    if isinstance(spec, str):  # protocol ALL_STEPS entry: lag-free, post = all 12 steps
        return np.zeros(12, bool), np.ones(12, bool)
    return T < max(0., spec['tau'] - spec['g']), T > spec['tau'] + spec['g']


def steps(m):
    return ' '.join(str(t) for t in T[m]) or '-'


def per_source(cells_by_source, pm, qm):
    """cells_by_source: list of (n,12) arrays for one output."""
    ple = np.array([np.abs(x[:, pm]).mean() for x in cells_by_source]) if pm.any() else None
    wsr = np.array([(x[:, qm] < -EPS).mean() for x in cells_by_source]) if qm.any() else None
    psm = SourceMedian([np.abs(x[:, qm]) for x in cells_by_source]) if qm.any() else None
    return ple, wsr, psm


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def development(scen):
    ids = np.load(OUT / 'selection_source_ids.npy'); sources = np.unique(ids)
    _, ev, _, sel_ids, stats, *_ = screen.load_real()
    assert (np.asarray(sel_ids).astype(str) == ids).all()
    device = torch.device('cuda'); torch.set_num_threads(2)
    arrays = {}
    for seed in SEEDS:
        ckpt = load_checkpoint(FORMAL / f'seed_{seed}/results/R0/selected.pt', map_location=device)
        m = screen.build_model('B4', stats).to(device); m.load_state_dict(ckpt['state_dict']); m.eval()
        arrays[(seed, 'R0')] = formal_coisc.sensitivity_all(m, ev, device)
        arrays[(seed, 'R0_npz_float32')] = np.load(FORMAL / f'seed_{seed}/results/R0/selection_predictions_and_sensitivity.npz')['sensitivity'].astype(np.float64)
        arrays[(seed, 'COISC')] = np.load(FORMAL / f'seed_{seed}/selection_coisc_predictions_and_sensitivity.npz')['sensitivity']
        for a in ABLATIONS:
            arrays[(seed, a)] = np.load(OUT / f'seed_{seed}/{a}_selection_prediction_sensitivity.npz')['sensitivity']
    rows = []; profiles = []
    for (seed, variant), s in arrays.items():
        for point, specs in scen.items():
            o = POINTS.index(point)
            for name, spec in specs.items():
                pm, qm = masks(spec); x = s[:, :, o]
                by = [x[ids == z] for z in sources]
                ple, wsr, psm = per_source(by, pm, qm)
                rows.append(dict(split='development', seed=seed, variant=variant, output=point, scenario=name,
                                 pre_steps_s=steps(pm), post_steps_s=steps(qm),
                                 PLE_pooled=float(np.abs(x[:, pm]).mean()) if pm.any() else None,
                                 WSR_pooled=float((x[:, qm] < -EPS).mean()) if qm.any() else None,
                                 PSM_pooled=float(np.median(np.abs(x[:, qm]))) if qm.any() else None,
                                 PLE_src=float(ple.mean()) if ple is not None else None,
                                 WSR_src=float(wsr.mean()) if wsr is not None else None,
                                 PSM_src=float(psm.many(np.ones((1, len(sources)), int))[0]) if psm else None))
        if variant == 'R0_npz_float32':
            continue
        for o, point in enumerate(POINTS):
            for k in range(12):
                v = s[:, k, o]
                profiles.append(dict(split='development', aggregation='window_pooled', seed=seed, variant=variant, output=point, t_s=int(T[k]),
                                     frac_negative=float((v < -EPS).mean()), median_S=float(np.median(v)), median_absS=float(np.median(np.abs(v)))))
    return rows, profiles


def holdout(scen):
    raw = np.load(HOLD / 'window_level_raw_results.npz', allow_pickle=True)
    sens = np.load(HOLD / 'window_level_sensitivity.npz', allow_pickle=True)
    draws = np.load(HOLD / 'bootstrap_draws.npz', allow_pickle=True)
    order = [str(x) for x in draws['source_order']]; counts = draws['counts']
    assert counts.shape == (20000, 23) and len(order) == 23
    table = {}
    for i in range(138):
        seed, source, cohort, variant = [str(x) for x in raw[f'record_{i}_identity']]
        s = sens[f'record_{i}_S']; assert s.shape[0] == len(raw[f'record_{i}_window_ids'])
        table[(seed, variant, source)] = s
    assert len(table) == 138 and all((s, v, z) in table for s in SEEDS for v in ('R0', 'COISC') for z in order)
    rows = []; profiles = []
    one = np.ones((1, 23), int)
    for seed in SEEDS:
        for point, specs in scen.items():
            o = POINTS.index(point)
            for name, spec in specs.items():
                pm, qm = masks(spec); est = {}; boot = {}
                for variant in ('R0', 'COISC'):
                    by = [table[(seed, variant, z)][:, :, o] for z in order]
                    ple, wsr, psm = per_source(by, pm, qm)
                    est[variant] = dict(PLE=float(ple.mean()) if ple is not None else None,
                                        WSR=float(wsr.mean()) if wsr is not None else None,
                                        PSM=float(psm.many(one)[0]) if psm else None)
                    boot[variant] = dict(PLE=draw_means(ple, counts) if ple is not None else None,
                                         WSR=draw_means(wsr, counts) if wsr is not None else None,
                                         PSM=psm.many(counts) if psm else None)
                r = dict(split='holdout23', seed=seed, output=point, scenario=name, pre_steps_s=steps(pm), post_steps_s=steps(qm))
                for k, factor in (('PLE', .9), ('WSR', .9), ('PSM', .2)):
                    r[f'R0_{k}'] = est['R0'][k]; r[f'COISC_{k}'] = est['COISC'][k]
                    if est['R0'][k] is None:
                        continue
                    r[f'ratio_{k}'] = est['COISC'][k] / est['R0'][k] if est['R0'][k] else None
                    lo, hi = percentile_ci(boot['COISC'][k] - factor * boot['R0'][k])
                    r[f'CI_COISC_minus_{factor}R0_{k}'] = f'[{lo:.6g}, {hi:.6g}]'
                    r[f'readout_{k}'] = bool(hi <= 0) if k != 'PSM' else bool(lo >= 0 and est['COISC'][k] >= .0005)
                rows.append(r)
        for variant in ('R0', 'COISC'):
            for o, point in enumerate(POINTS):
                for k in range(12):
                    by = [table[(seed, variant, z)][:, k, o] for z in order]
                    profiles.append(dict(split='holdout23', aggregation='equal_source', seed=seed, variant=variant, output=point, t_s=int(T[k]),
                                         frac_negative=float(np.mean([(v < -EPS).mean() for v in by])),
                                         median_S=float(SourceMedian(by).many(one)[0]),
                                         median_absS=float(SourceMedian([np.abs(v) for v in by]).many(one)[0])))
    return rows, profiles


def main():
    assert not VERIFY_HASHES or hashlib.sha256((PROTOCOLS / 'POSTHOC_ABLATION_PROTOCOL.json').read_bytes()).hexdigest() == PROTOCOL_SHA
    protocol = json.loads((PROTOCOLS / 'POSTHOC_ABLATION_PROTOCOL.json').read_text(encoding='utf-8'))
    scen = protocol['item3_alternative_lag_reanalysis']['scenarios']
    dev_rows, dev_prof = development(scen)
    hold_rows, hold_prof = holdout(scen)
    write_csv(OUT / 'lag_scenarios_development.csv', dev_rows)
    write_csv(OUT / 'lag_scenarios_holdout23.csv', hold_rows)
    write_csv(OUT / 'sensitivity_step_profiles.csv', dev_prof + hold_prof)
    # Reproduction check of the PRIOR scenario against the formal records.
    check = []
    for seed in SEEDS:
        formal = json.loads((FORMAL / f'seed_{seed}/per_seed_base_and_coisc_metrics.json').read_text(encoding='utf-8'))
        for variant, key in (('R0', 'base_sensitivity'), ('R0_npz_float32', 'base_sensitivity'), ('COISC', 'coisc_sensitivity')):
            r = next(x for x in dev_rows if x['seed'] == seed and x['variant'] == variant and x['output'] == 'Tp' and x['scenario'] == 'PRIOR')
            check.append(dict(split='development', seed=seed, variant=variant,
                              max_abs_diff=max(abs(r[f'{k}_pooled'] - formal[key][k]) for k in ('PLE', 'WSR', 'PSM'))))
    with (HOLD / 'delivered23_sensitivity_metrics.csv').open(encoding='utf-8-sig') as f:
        ref = [x for x in csv.DictReader(f) if x['cohort'] == 'all23' and x['aggregation'] == 'equal_source' and x['output'] == 'Tp']
    for x in ref:
        r = next(y for y in hold_rows if y['seed'] == x['seed'] and y['output'] == 'Tp' and y['scenario'] == 'PRIOR')
        check.append(dict(split='holdout23', seed=x['seed'], variant=x['variant'],
                          max_abs_diff=max(abs(r[f"{x['variant']}_{k}"] - float(x[k])) for k in ('PLE', 'WSR', 'PSM'))))
    screen.save_json(OUT / 'lag_reanalysis_reproduction_check.json', dict(protocol_sha256=PROTOCOL_SHA, tolerance=1e-9, rows=check,
                     all_within_tolerance=all(c['max_abs_diff'] <= 1e-9 for c in check)))
    print(json.dumps(check, indent=0))


if __name__ == '__main__':
    main()
