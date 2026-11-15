"""Pre-registered analysis of the first-move rule baseline study.

  gd_analyze.py gate           G2: reproduces the stored closed-loop panel-A per-pair contrasts from raw runs + synthetic selftest
  gd_analyze.py smoke          runs the metric code path on the smoke directory (no inference)
  gd_analyze.py analyze        frozen protocol and author authorization required; writes study/GUARD_RESULTS.json

Tracking unit: MAE_track (onset_analyze_am1.metric_row, unchanged), family-equal over the 24 plants, one unit per R0-wrong pair.
Reversal unit: among updates at which the plant T_p is more than FAR_BAND_C from 37 C, the fraction at which the applied command
moves away from 37 C; counts pooled within a plant family, then family-equal, one unit per pair.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

import gd_common as gc
import cl_common as cc  # noqa: E402
import cl_analyze as ca  # noqa: E402
import onset_mil as om  # noqa: E402
import onset_analyze_am1 as oa  # noqa: E402

CONTRASTS = (('COISC', gc.GUARD), (gc.GUARD, 'R0'), ('L0', gc.GUARD), ('COISC', 'R0'))


def reversal_counts(meta, data, band=gc.FAR_BAND_C):
    """(reversed far updates, far updates, reversed far updates after step 0) for one run."""
    u = np.asarray(data['applied_UH'], dtype=np.float64)
    tp = np.asarray(data['plant_Tp'], dtype=np.float64)
    prev = np.concatenate([[float(meta['previous_uh_percent'])], u[:-1]])
    move = np.sign(np.round(u - prev, 9))
    err = 37. - np.concatenate([[float(meta['initial_Tp_C'])], tp[:-1]])
    far = np.abs(err) > band
    rev = far & (move == -np.sign(err))
    return int(rev.sum()), int(far.sum()), int(rev[1:].sum())


def load_cells(root, tasks, plants, keep):
    """{(variant, plant, scenario, seed): dict(MAE_track, rev, far, rev_after0, guard_fired)} for tasks accepted by keep()."""
    cells, missing = {}, []
    for i, t in enumerate(tasks):
        if not keep(t):
            continue
        meta, data = oa.load_run(Path(root) / 'raw_runs' / f'{i:05d}.npz')
        assert (meta['plant_index'], meta['scenario_index'], meta['controller_seed'], meta['controller_variant']) == \
            (t['plant_index'], t['scenario_index'], t['seed'], t['variant']), f'task/metadata mismatch at {i} in {root}'
        row = oa.metric_row(meta, data, t['variant'], t['plant_index'], t['scenario_index'], t['seed'], plants[t['plant_index']]['family'])
        if row is None:
            missing.append((str(root), i))
            continue
        rev, far, rev1 = reversal_counts(meta, data)
        cells[(t['variant'], t['plant_index'], t['scenario_index'], t['seed'])] = dict(
            MAE_track=row['MAE_track'], rev=rev, far=far, rev_after0=rev1, guard_fired=bool(meta.get('guard_fired', False)),
            total_variation_u=row['total_variation_u'])
    return cells, missing


def pair_value(cells, variant, seed, si, families, kind):
    """Family-equal value of one variant on one pair over the 24 plants; None if a family has no complete cell."""
    per_family = []
    for f in cc.FAMILIES:
        pis = [pi for pi in cc.ALL_PLANTS if families[pi] == f and all((v, pi, si, seed) in cells for v in gc.ALL_VARIANTS)]
        if not pis:
            continue
        if kind == 'mae':
            per_family.append(float(np.mean([cells[(variant, pi, si, seed)]['MAE_track'] for pi in pis])))
        else:
            num = sum(cells[(variant, pi, si, seed)][kind] for pi in pis)
            den = sum(cells[(variant, pi, si, seed)]['far'] for pi in pis)
            per_family.append(num / den if den else 0.)
    return float(np.mean(per_family)) if per_family else None


def bootstrap(units):
    return ca.bootstrap(units, draws=gc.BOOTSTRAP_DRAWS, seed=gc.BOOTSTRAP_SEED)


def compute(cells, pairs, families, missing):
    out = dict(pairs=[dict(seed=s, scenario=om.SCENARIO_LABELS[si]) for s, si in pairs], non_evaluable_runs=len(missing))
    for kind, label in (('mae', 'tracking_MAE'), ('rev', 'reversal_fraction'), ('rev_after0', 'reversal_fraction_after_first_update')):
        block = {}
        for a, b in CONTRASTS:
            units = []
            for s, si in pairs:
                va, vb = pair_value(cells, a, s, si, families, kind), pair_value(cells, b, s, si, families, kind)
                if va is not None and vb is not None:
                    units.append(va - vb)
            block[f'{a} - {b}'] = dict(bootstrap(units), per_pair=units, pairs_lower=int(sum(u < 0 for u in units)))
        block['means'] = {v: [pair_value(cells, v, s, si, families, kind) for s, si in pairs] for v in gc.ALL_VARIANTS}
        out[label] = block
    g1 = out['tracking_MAE'][f'COISC - {gc.GUARD}']
    g2 = out['reversal_fraction'][f'COISC - {gc.GUARD}']
    out['G_H1_supported'] = bool(g1['units'] == len(pairs) and g1['units'] > 0 and g1['ci_upper'] < 0)
    out['G_H2_supported'] = bool(g2['units'] == len(pairs) and g2['units'] > 0 and g2['ci_upper'] < 0)
    out['primary_verdict'] = ('SUPPORTED' if out['G_H1_supported'] and out['G_H2_supported'] else
                              'PARTIAL_H1_ONLY' if out['G_H1_supported'] else 'PARTIAL_H2_ONLY' if out['G_H2_supported'] else 'NOT_SUPPORTED')
    guard_cells = [c for (v, *_), c in cells.items() if v == gc.GUARD]
    out['guard_fired'] = dict(runs=len(guard_cells), fired=int(sum(c['guard_fired'] for c in guard_cells)))
    out['total_variation_u_mean'] = {v: float(np.mean([c['total_variation_u'] for (vv, *_), c in cells.items() if vv == v]))
                                     for v in gc.ALL_VARIANTS}
    return out


def study_cells(guard_root, plants):
    pairs = gc.wrong_pairs()
    wrong = set(pairs)
    cl_tasks = cc.read(gc.CL_STUDY / 'TASKS.json')
    cells, missing = load_cells(gc.CL_STUDY, cl_tasks, plants, lambda t: (t['seed'], t['scenario_index']) in wrong)
    g_cells, g_missing = load_cells(guard_root, cc.read(Path(guard_root) / 'TASKS.json'), plants, lambda t: True)
    cells.update(g_cells)
    return cells, missing + g_missing, pairs


def selftest():
    """Synthetic check: known contrasts are recovered and the verdict logic is correct."""
    families = ['F1'] * 17 + ['F2'] * 3 + ['F3'] * 3 + ['F4']
    pairs = [(1, 0), (2, 3), (3, 5)]
    base = dict(R0=(2.0, 6), L0=(1.8, 5), COISC=(1.0, 1), R0_GUARD=(1.9, 5))  # (MAE, reversed updates of 20 far updates)
    cells = {}
    for s, si in pairs:
        for pi in range(24):
            for v, (m, r) in base.items():
                cells[(v, pi, si, s)] = dict(MAE_track=m + 0.01 * pi, rev=r, far=20, rev_after0=max(r - 1, 0), guard_fired=v == 'R0_GUARD',
                                             total_variation_u=0.)
    r = compute(cells, pairs, families, [])
    checks = dict(h1_point=abs(r['tracking_MAE']['COISC - R0_GUARD']['point'] - (-0.9)) < 1e-9,
                  h2_point=abs(r['reversal_fraction']['COISC - R0_GUARD']['point'] - (-0.2)) < 1e-9,
                  guard_r0=abs(r['tracking_MAE']['R0_GUARD - R0']['point'] - (-0.1)) < 1e-9,
                  verdict=r['primary_verdict'] == 'SUPPORTED', fired=r['guard_fired'] == dict(runs=72, fired=72))
    for s, si in pairs:  # COISC tracking worse than the guard on every pair -> H1 must fail
        for pi in range(24):
            cells[('COISC', pi, si, s)]['MAE_track'] += 1.5
    r2 = compute(cells, pairs, families, [])
    checks.update(h1_fails=r2['G_H1_supported'] is False, verdict_h2_only=r2['primary_verdict'] == 'PARTIAL_H2_ONLY')
    return dict(checks=checks, pass_=all(checks.values()))


def gate():
    """Reproduce the stored panel-A per-pair COISC-R0, L0-R0 tracking contrasts of the closed-loop study from its raw runs."""
    plants, _ = om.plants_and_formal_index()
    families = [p['family'] for p in plants]
    pairs = gc.wrong_pairs()
    wrong = set(pairs)
    cells, missing = load_cells(gc.CL_STUDY, cc.read(gc.CL_STUDY / 'TASKS.json'), plants, lambda t: (t['seed'], t['scenario_index']) in wrong)
    stored = cc.read(gc.CL_RESULTS)['panel_A']
    rows = []
    for a, b in (('COISC', 'R0'), ('L0', 'R0'), ('COISC', 'L0')):
        got = []
        for s, si in pairs:
            per_family = []
            for f in cc.FAMILIES:
                vals = [cells[(a, pi, si, s)]['MAE_track'] - cells[(b, pi, si, s)]['MAE_track'] for pi in cc.ALL_PLANTS if families[pi] == f]
                per_family.append(float(np.mean(vals)))
            got.append(float(np.mean(per_family)))
        rows.append(dict(contrast=f'{a} - {b}', exact=[repr(x) == repr(y) for x, y in zip(got, stored[f'{a} - {b}']['per_pair'])]))
    st = selftest()
    report = dict(label='Guard study gate G2: panel-A per-pair contrasts reproduced bitwise from raw runs; synthetic analysis selftest',
                  runs_loaded=len(cells), missing=len(missing), rows=rows, selftest=st,
                  pass_=len(cells) == 3 * 24 * len(pairs) and not missing and all(all(r['exact']) for r in rows) and st['pass_'],
                  code_sha256=gc.code_hashes(('gd_common.py', 'gd_analyze.py')), created_utc=gc.utc())
    gc.write(gc.PROTO / 'GATE_G2_ANALYSIS_REPORT.json', report)
    print('G2', dict(runs=len(cells), rows=rows, selftest=st['pass_'], pass_=report['pass_']), flush=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'gate':
        gate()
    elif mode == 'smoke':
        plants, _ = om.plants_and_formal_index()
        tasks = cc.read(gc.SMOKE / 'TASKS.json')
        cells, missing = load_cells(gc.SMOKE, tasks, plants, lambda t: True)
        assert len(cells) + len(missing) == len(tasks)
        gc.write(gc.SMOKE / 'SMOKE_ANALYSIS_OK.json', dict(rows=len(cells), missing=len(missing), selftest=selftest()['pass_'],
                                                         code_sha256=gc.code_hashes(('gd_common.py', 'gd_analyze.py')), at=gc.utc()))
        print('SMOKE ANALYSIS OK', len(cells), len(missing), flush=True)
    elif mode == 'analyze':
        protocol, digest = gc.frozen_protocol()
        audit = gc.read(gc.STUDY / 'EXECUTION_AUDIT.json')
        assert audit['status'] == 'RAW_RUNS_COMPLETE' and audit['protocol_sha256'] == digest
        plants, _ = om.plants_and_formal_index()
        cells, missing, pairs = study_cells(gc.STUDY, plants)
        res = compute(cells, pairs, [p['family'] for p in plants], missing)
        res.update(protocol_sha256=digest, at=gc.utc(), far_band_C=gc.FAR_BAND_C)
        gc.write(gc.STUDY / 'GUARD_RESULTS.json', res)
        print('VERDICT', res['primary_verdict'], 'H1', res['G_H1_supported'], 'H2', res['G_H2_supported'], flush=True)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
