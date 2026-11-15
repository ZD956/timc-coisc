"""Pre-registered analysis of the move-penalty baseline study.

  du_analyze.py gate       G2: reproduces the stored S4 H2 COISC - R0 point from raw runs + synthetic selftest
  du_analyze.py smoke      metric path on the smoke directory
  du_analyze.py analyze    frozen protocol and author authorization required; writes study/DU_RESULTS.json
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

import du_common as dc
import cl_common as cc  # noqa: E402
import cl_analyze as ca  # noqa: E402
import onset_mil as om  # noqa: E402
import onset_analyze_am1 as oa  # noqa: E402

ARMS = dc.ALL_VARIANTS
FAR_BAND_C = 2.0


def smooth_stats(meta, data):
    u = np.asarray(data['applied_UH'], dtype=np.float64)
    prev = np.concatenate([[float(meta['previous_uh_percent'])], u[:-1]])
    du = u - prev
    interior = (u > 0) & (u < 100) & (prev > 0) & (prev < 100)
    nz = np.round(du, 9) != 0
    sg = np.sign(np.round(du, 9))[nz]
    tp = np.asarray(data['plant_Tp'], dtype=np.float64)
    err = 37. - np.concatenate([[float(meta['initial_Tp_C'])], tp[:-1]])
    far = np.abs(err) > FAR_BAND_C
    return dict(tv_interior=float(np.abs(du[interior]).sum()), direction_changes=int((sg[1:] != sg[:-1]).sum()) if len(sg) > 1 else 0,
                mean_u=float(u.mean()), rev=int((far & (np.sign(np.round(du, 9)) == -np.sign(err))).sum()), far=int(far.sum()))


def load_cells(root, tasks, plants, keep=lambda t: True):
    cells, missing = {}, []
    for i, t in enumerate(tasks):
        if not keep(t):
            continue
        meta, data = oa.load_run(Path(root) / 'raw_runs' / f'{i:05d}.npz')
        assert (meta['plant_index'], meta['scenario_index'], meta['controller_seed'], meta['controller_variant']) == \
            (t['plant_index'], t['scenario_index'], t['seed'], t['variant']), f'task/metadata mismatch at {i} in {root}'
        row = oa.metric_row(meta, data, t['variant'], t['plant_index'], t['scenario_index'], t['seed'], plants[t['plant_index']]['family'])
        if row is None:
            missing.append((str(root), i)); continue
        cells[(t['variant'], t['plant_index'], t['scenario_index'], t['seed'])] = dict(
            MAE_track=row['MAE_track'], total_variation_u=row['total_variation_u'], saturation_fraction=row['saturation_fraction'],
            **smooth_stats(meta, data))
    return cells, missing


def complete_mae(cells):
    ok = {}
    for (v, pi, si, s), c in cells.items():
        if all((a, pi, si, s) in cells for a in ARMS):
            ok[(v, pi, si, s)] = c['MAE_track']
    return ok


def seed_units(cells, a, b, seeds, exclude, families):
    saved = cc.VARIANTS
    try:
        cc.VARIANTS = ARMS
        return ca.seed_units(complete_mae(cells), a, b, seeds, exclude, families)
    finally:
        cc.VARIANTS = saved


def bootstrap(units):
    return ca.bootstrap(units, draws=dc.BOOTSTRAP_DRAWS, seed=dc.BOOTSTRAP_SEED)


def seed_metric(cells, arm, seeds, key):
    out = []
    for s in seeds:
        vals = [c[key] for (v, pi, si, ss), c in cells.items() if v == arm and ss == s and pi in cc.PANEL_B_PLANTS
                and all((x, pi, si, s) in cells for x in ARMS)]
        out.append(float(np.mean(vals)) if vals else None)
    return out


def compute(cells, seeds, pairs, families, missing):
    excl = set(pairs)
    out = dict(seeds=len(seeds), non_evaluable_runs=len(missing))
    out['tracking_unaffected'] = {f'{a} - {b}': bootstrap(seed_units(cells, a, b, seeds, excl, families))
                                  for a, b in (('COISC', dc.ARM), (dc.ARM, 'R0'), ('COISC', 'R0'), ('L0', dc.ARM))}
    h = out['tracking_unaffected'][f'COISC - {dc.ARM}']
    out['H_supported'] = bool(h['units'] == len(seeds) and h['ci_upper'] < 0)
    out['primary_verdict'] = 'SUPPORTED' if out['H_supported'] else 'NOT_SUPPORTED'
    out['tracking_all_pairs'] = {f'{a} - {b}': bootstrap(seed_units(cells, a, b, seeds, set(), families))
                                 for a, b in (('COISC', dc.ARM), (dc.ARM, 'R0'))}
    tv = {a: seed_metric(cells, a, seeds, 'total_variation_u') for a in ARMS}
    d = [x - y for x, y in zip(tv[dc.ARM], tv['COISC']) if x is not None and y is not None]
    out['matching_check'] = dict(tv_mean={a: float(np.mean([x for x in v if x is not None])) for a, v in tv.items()},
                                 tv_R0_DU_minus_COISC=bootstrap(d))
    out['matching_check']['matched_or_smoother'] = not (out['matching_check']['tv_R0_DU_minus_COISC']['ci_lower'] > 0)
    out['descriptive_means'] = {a: {k: float(np.mean([c[k] for (v, pi, *_), c in cells.items() if v == a and pi in cc.PANEL_B_PLANTS]))
                                    for k in ('MAE_track', 'total_variation_u', 'tv_interior', 'direction_changes', 'saturation_fraction', 'mean_u')}
                                for a in ARMS}
    out['reversal_fraction_pooled'] = {a: float(sum(c['rev'] for (v, *_), c in cells.items() if v == a) /
                                                max(1, sum(c['far'] for (v, *_), c in cells.items() if v == a))) for a in ARMS}
    return out


def study_cells(root, plants):
    cl_tasks = cc.read(dc.CL_STUDY / 'TASKS.json')
    cells, missing = load_cells(dc.CL_STUDY, cl_tasks, plants, lambda t: t['plant_index'] in cc.PANEL_B_PLANTS)
    d_cells, d_missing = load_cells(root, dc.read(Path(root) / 'TASKS.json'), plants)
    cells.update(d_cells)
    return cells, missing + d_missing


def selftest():
    families = ['F1'] * 17 + ['F2'] * 3 + ['F3'] * 3 + ['F4']
    seeds = (1, 2, 3, 4); pairs = [(1, 0)]
    base = dict(R0=1.80, L0=1.80, COISC=1.77, R0_DU=1.85)
    cells = {}
    for s in seeds:
        for si in range(12):
            for pi in cc.PANEL_B_PLANTS:
                for a, m in base.items():
                    cells[(a, pi, si, s)] = dict(MAE_track=m + 0.01 * pi + 0.001 * s, total_variation_u=180. if a in ('COISC', 'R0_DU') else 270.,
                                                 saturation_fraction=0., tv_interior=0., direction_changes=0, mean_u=0., rev=1, far=10)
    r = compute(cells, seeds, pairs, families, [])
    checks = dict(point=abs(r['tracking_unaffected']['COISC - R0_DU']['point'] - (-0.08)) < 1e-9, verdict=r['primary_verdict'] == 'SUPPORTED',
                  cost=abs(r['tracking_unaffected']['R0_DU - R0']['point'] - 0.05) < 1e-9, matched=r['matching_check']['matched_or_smoother'] is True)
    for k in list(cells):
        if k[0] == 'COISC':
            cells[k]['MAE_track'] += 0.2
    r2 = compute(cells, seeds, pairs, families, [])
    checks.update(fails=r2['primary_verdict'] == 'NOT_SUPPORTED')
    return dict(checks=checks, pass_=all(checks.values()))


def gate():
    plants, _ = om.plants_and_formal_index()
    families = [p['family'] for p in plants]
    cells, missing = load_cells(dc.CL_STUDY, cc.read(dc.CL_STUDY / 'TASKS.json'), plants, lambda t: t['plant_index'] in cc.PANEL_B_PLANTS)
    for k in [k for k in cells if k[0] == 'R0']:
        cells[(dc.ARM,) + k[1:]] = cells[k]
    pairs = cc.wrong_pairs(cc.first_actions())
    got = bootstrap(seed_units(cells, 'COISC', 'R0', list(cc.NEW_SEEDS), set(pairs), families))
    stored = cc.read(dc.CL_STUDY / 'CLOSEDLOOP_RESULTS.json')['panel_B_unaffected']['COISC - R0']
    st = selftest()
    report = dict(label='Move-penalty study gate G2: S4 H2 COISC - R0 point reproduced from raw runs; synthetic selftest',
                  missing=len(missing), point=got['point'], stored=stored['point'], exact=repr(got['point']) == repr(stored['point']),
                  selftest=st, pass_=bool(not missing and repr(got['point']) == repr(stored['point']) and st['pass_']),
                  code_sha256=dc.code_hashes(('du_common.py', 'du_analyze.py')), created_utc=dc.utc())
    dc.write(dc.PROTO / 'GATE_G2_ANALYSIS_REPORT.json', report)
    print('G2', {k: report[k] for k in ('missing', 'exact', 'pass_')}, st, flush=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'gate':
        gate()
    elif mode == 'smoke':
        plants, _ = om.plants_and_formal_index()
        root = dc.SMOKE / 'mil'
        tasks = dc.read(root / 'TASKS.json')
        cells, missing = load_cells(root, tasks, plants)
        assert len(cells) + len(missing) == len(tasks)
        dc.write(dc.SMOKE / 'SMOKE_ANALYSIS_OK.json', dict(rows=len(cells), missing=len(missing), selftest=selftest()['pass_'],
                                                         code_sha256=dc.code_hashes(('du_common.py', 'du_analyze.py')), at=dc.utc()))
        print('SMOKE ANALYSIS OK', len(cells), len(missing), flush=True)
    elif mode == 'analyze':
        protocol, digest = dc.frozen_protocol()
        audit = dc.read(dc.STUDY / 'EXECUTION_AUDIT.json')
        assert audit['status'] == 'RAW_RUNS_COMPLETE' and audit['protocol_sha256'] == digest
        plants, _ = om.plants_and_formal_index()
        cells, missing = study_cells(dc.STUDY, plants)
        res = compute(cells, list(cc.NEW_SEEDS), cc.wrong_pairs(cc.first_actions()), [p['family'] for p in plants], missing)
        res.update(protocol_sha256=digest, lambda_du=protocol['lambda_du'], at=dc.utc())
        dc.write(dc.STUDY / 'DU_RESULTS.json', res)
        print('VERDICT', res['primary_verdict'], 'matched', res['matching_check']['matched_or_smoother'], flush=True)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
