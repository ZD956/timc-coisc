"""Recompute the decisions of the preregistered studies S3-S7 from the released summary results.

  python timc_coisc/reproduce_tg_results.py

Reruns the frozen analysis functions (scr_analyze.analyze, cl_analyze.compute, gd_analyze.compute, sq_analyze.compute,
du_analyze.compute) on the released per-seed, first-move and per-run tables, and compares every number with the released
SCREENING_RESULTS.json, CLOSEDLOOP_RESULTS.json, GUARD_RESULTS.json, SEQ_RESULTS.json and DU_RESULTS.json. For S7 it also
reapplies the lambda_du calibration rule to the released calibration runs. Needs no private data.
"""
from __future__ import annotations
import csv
import json
import math

from _paths import PROTOCOLS, RESULTS
import scr_analyze
import cl_analyze
import cl_common as cc


def read(name):
    return json.loads((RESULTS / name).read_text(encoding='utf-8'))


def same(a, b, path='', tol=1e-12):
    """Recursive comparison; returns a list of differing paths."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f'{path}/{k}: missing' for k in set(a) ^ set(b)]
        return out + [d for k in set(a) & set(b) for d in same(a[k], b[k], f'{path}/{k}', tol)]
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f'{path}: length {len(a)} != {len(b)}']
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in same(x, y, f'{path}[{i}]', tol)]
    if isinstance(a, float) or isinstance(b, float):
        ok = a is not None and b is not None and (math.isclose(float(a), float(b), rel_tol=0, abs_tol=tol)
                                                  or (math.isnan(float(a)) and math.isnan(float(b))))
        return [] if ok else [f'{path}: {a} != {b}']
    return [] if a == b else [f'{path}: {a!r} != {b!r}']


def screening():
    stored = read('screening/SCREENING_RESULTS.json')
    protocol = json.loads((cc.SCREEN_PROTO / 'SCREENING_PROTOCOL.json').read_text(encoding='utf-8'))
    fa = {(r['seed'], r['variant'], r['scenario_index']): r['first_action'] for r in read('screening/FIRST_ACTION_CHECK.json')['rows']}
    table = {int(s): v for s, v in stored['per_seed'].items()}
    hold = {r['seed']: r for r in read('screening/HOLDOUT_ESTIMATES.json')['rows']}
    recs = {r['seed']: r for r in read('screening/SEED_RECORDS.json')['seeds']}
    rebuilt = {s: dict({k: hold[s][k] for k in ('Tp_WSR_ONSET', 'Tp_WSR_PRIOR', 'Tp_MAE_120', 'all7_MAE_5h')},
                       dev_Tp_WSR_onset=recs[s]['r0_dev']['Tp_WSR_onset'], dev_Tp_MAE_120=recs[s]['r0_dev']['Tp_MAE_120'],
                       dev_macro7_MAE=recs[s]['r0_dev']['macro7_MAE']) for s in recs}
    diffs = same(rebuilt, table, '/per_seed')
    res = scr_analyze.analyze(rebuilt, fa, protocol['analysis_rules'])
    expected = {k: v for k, v in stored['result'].items() if k not in ('solver_failures_first_step', 'descriptive_pooled_53_seeds')}
    diffs += same(json.loads(json.dumps(res, default=float)), expected, '/result')
    return stored['result']['primary_verdict'], diffs


def closedloop():
    stored = read('closedloop/CLOSEDLOOP_RESULTS.json')
    fa = {(r['seed'], r['variant'], r['scenario_index']): r for r in read('screening/FIRST_ACTION_CHECK.json')['rows']}
    labels = [f'S{i + 1:02d}' for i in range(12)]
    rows, families = {}, {}
    with (RESULTS / 'closedloop' / 'CLOSEDLOOP_RUN_METRICS.csv').open(encoding='utf-8') as f:
        for r in csv.DictReader(f):
            pi, si, s = int(r['plant_index']), labels.index(r['scenario']), int(r['seed'])
            families[pi] = r['family']
            rows[(r['variant'], pi, si, s)] = {k: float(r[k]) for k in ('MAE_track', 'total_variation_u', 'saturation_fraction',
                                                                         'solver_success_rate')}
    mae = cl_analyze._Mae({k: r['MAE_track'] for k, r in rows.items()})
    mae['_rows'] = rows
    res = cl_analyze.compute(mae, fa, cc.NEW_SEEDS, [families[i] for i in range(24)], [])
    expected = {k: v for k, v in stored.items() if k not in ('protocol_sha256', 'at', 'ni_margin_C')}
    return stored['primary_verdict'], same(res, expected, '', tol=1e-9)


INT_FIELDS = {'far', 'rev', 'rev_after0', 'irev', 'ifar', 'direction_changes'}


def run_cells(name):
    """{(variant, plant, scenario, seed): cell} from a released per-run table, with the field types of the frozen load_cells."""
    labels = [f'S{i + 1:02d}' for i in range(12)]
    cells, families = {}, {}
    with (RESULTS / name).open(encoding='utf-8') as f:
        for r in csv.DictReader(f):
            pi = int(r['plant_index'])
            families[pi] = r['family']
            cell = {}
            for k, v in r.items():
                if k in ('variant', 'seed', 'plant_index', 'family', 'scenario'):
                    continue
                cell[k] = (v == 'True') if k == 'guard_fired' else int(v) if k in INT_FIELDS else float(v)
            cells[(r['variant'], pi, labels.index(r['scenario']), int(r['seed']))] = cell
    return cells, families


def first_actions():
    return {(r['seed'], r['variant'], r['scenario_index']): r for r in read('screening/FIRST_ACTION_CHECK.json')['rows']}


def guard():
    import gd_analyze
    stored = read('guard/GUARD_RESULTS.json')
    cells, families = run_cells('guard/GUARD_RUN_CELLS.csv')
    res = gd_analyze.compute(cells, cc.wrong_pairs(first_actions()), [families[i] for i in range(24)], [])
    expected = {k: v for k, v in stored.items() if k not in ('protocol_sha256', 'at', 'far_band_C')}
    return stored['primary_verdict'], same(json.loads(json.dumps(res, default=float)), expected, '', tol=1e-12)


def sequence():
    import sq_analyze
    stored = read('sequence/SEQ_RESULTS.json')
    cells, families = run_cells('sequence/SEQ_RUN_CELLS.csv')
    res = sq_analyze.compute(cells, list(cc.NEW_SEEDS), cc.wrong_pairs(first_actions()), [families[i] for i in range(24)], [])
    expected = {k: v for k, v in stored.items() if k not in ('protocol_sha256', 'at', 'far_band_C', 'ni_margin_C')}
    return stored['primary_verdict'], same(json.loads(json.dumps(res, default=float)), expected, '', tol=1e-12)


def move_penalty():
    import du_analyze
    stored = read('move_penalty/DU_RESULTS.json')
    cells, families = run_cells('move_penalty/DU_RUN_CELLS.csv')
    fam = [families.get(i) for i in range(24)]
    res = du_analyze.compute(cells, list(cc.NEW_SEEDS), cc.wrong_pairs(first_actions()), fam, [])
    expected = {k: v for k, v in stored.items() if k not in ('protocol_sha256', 'at', 'lambda_du')}
    diffs = same(json.loads(json.dumps(res, default=float)), expected, '', tol=1e-12)
    # calibration rule: smallest grid value whose mean R0 command TV is at most that of COISC at lambda_du = 1
    protocol = json.loads((PROTOCOLS / 'move_penalty' / 'DU_PROTOCOL.json').read_text(encoding='utf-8'))
    tv = {}
    with (RESULTS / 'move_penalty' / 'calibration' / 'CALIBRATION_RUN_TV.csv').open(encoding='utf-8') as f:
        for r in csv.DictReader(f):
            tv.setdefault((r['variant'], float(r['lambda_du'])), []).append(float(r['total_variation_u']))
    mean = {k: sum(v) / len(v) for k, v in tv.items()}
    target = mean[('COISC', 1.0)]
    grid = sorted(g for v, g in mean if v != 'COISC')
    chosen = next(g for g in grid if mean[('R0_DU', g)] <= target)
    cal = protocol['calibration_result']
    diffs += same(dict(coisc_tv_target=target, r0_tv_by_lambda={str(g): mean[('R0_DU', g)] for g in grid}, lambda_du=chosen),
                  dict(coisc_tv_target=cal['coisc_tv_target'], r0_tv_by_lambda=cal['r0_tv_by_lambda'], lambda_du=protocol['lambda_du']),
                  '/calibration', tol=1e-9)
    return stored['primary_verdict'], diffs


def main():
    ok = True
    for name, fn in (('screening (S3)', screening), ('closed loop (S4)', closedloop), ('first-move rule (S5)', guard),
                     ('sequence variant (S6)', sequence), ('move-penalty baseline (S7)', move_penalty)):
        verdict, diffs = fn()
        ok &= not diffs
        print(f'{name}: verdict {verdict}; {"reproduced" if not diffs else f"{len(diffs)} differences"}')
        for d in diffs[:20]:
            print('   ', d)
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
