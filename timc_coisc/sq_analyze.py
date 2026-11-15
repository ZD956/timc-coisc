"""Pre-registered analysis of the sequence-sensitivity study.

  sq_analyze.py gate       G6: reproduces the stored S4 panel-B H2 contrast and the Fig. 3(d) seed-level reversal fractions from raw
                               runs (with COISC relabelled as the study arm) + synthetic selftest
  sq_analyze.py smoke      metric code path on the smoke MIL directory
  sq_analyze.py analyze    frozen protocol and author authorization required; writes study/SEQ_RESULTS.json

Reversal unit (H1): on panel B (40 seeds x 12 scenarios x 6 plants), among updates at which the plant T_p before the update is more
than FAR_BAND_C from 37 C, the fraction at which the applied command moves away from 37 C (gd_analyze.reversal_counts, unchanged);
counts pooled within a plant family over the seed's 12 scenarios, then family-equal; one unit per seed.
Tracking unit (H2): cl_analyze.seed_units, unchanged (panel B, R0-wrong pairs excluded, family-equal, one unit per seed).
A cell enters a contrast only if all four arms are evaluable there; dropped cells are counted.
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

import numpy as np

import sq_common as sq
import cl_common as cc  # noqa: E402
import cl_analyze as ca  # noqa: E402
import gd_analyze as ga  # noqa: E402
import onset_mil as om  # noqa: E402
import onset_analyze_am1 as oa  # noqa: E402

ARMS = ('R0', 'COISC', 'L0', sq.VARIANT)
BOOTSTRAP_SEED = 20261113
BOOTSTRAP_DRAWS = 20000
FAR_BAND_C = 2.0
NI_MARGIN_C = 0.10
FIG3D_CSV = sq.RESULTS / 'figures_data' / 'derived_reversal_seed.csv'


def internal_counts(meta, data, band=FAR_BAND_C):
    """(reversed, far) judged by the controller's own one-step T_p estimate from the previous update (updates 2-60)."""
    u = np.asarray(data['applied_UH'], dtype=np.float64)
    est = np.asarray(data['internal_predicted_Tp'], dtype=np.float64)[:, 0]
    prev = np.concatenate([[float(meta['previous_uh_percent'])], u[:-1]])
    move = np.sign(np.round(u - prev, 9))
    err = 37. - est[:-1]
    far = np.abs(err) > band  # NaN (solver failure) compares False
    return int((far & (move[1:] == -np.sign(err))).sum()), int(far.sum())


def load_cells(root, tasks, plants, keep=lambda t: True, relabel=None):
    """{(arm, plant, scenario, seed): cell} with MAE_track, plant- and internal-judged reversal counts, total variation."""
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
        rev, far, _ = ga.reversal_counts(meta, data, FAR_BAND_C)
        irev, ifar = internal_counts(meta, data)
        arm = (relabel or {}).get(t['variant'], t['variant'])
        cells[(arm, t['plant_index'], t['scenario_index'], t['seed'])] = dict(
            MAE_track=row['MAE_track'], rev=rev, far=far, irev=irev, ifar=ifar,
            total_variation_u=row['total_variation_u'], saturation_fraction=row['saturation_fraction'])
    return cells, missing


def complete(cells, pi, si, s, arms=ARMS):
    return all((a, pi, si, s) in cells for a in arms)


def seed_reversal(cells, arm, seed, families, num='rev', den='far', plants=cc.PANEL_B_PLANTS, arms=ARMS):
    """Family-equal reversal fraction of one arm on one seed (panel B, all 12 scenarios); None if no family has a complete cell."""
    per_family = []
    for f in cc.FAMILIES:
        keys = [(arm, pi, si, seed) for pi in plants if families[pi] == f for si in range(12) if complete(cells, pi, si, seed, arms)]
        if keys:
            d = sum(cells[k][den] for k in keys)
            per_family.append(sum(cells[k][num] for k in keys) / d if d else 0.)
    return float(np.mean(per_family)) if per_family else None


def bootstrap(units):
    return ca.bootstrap(units, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED)


class Mae(dict):
    def __init__(self, cells):
        super().__init__({k: c['MAE_track'] for k, c in cells.items()})
        self.cells = cells


def mae_seed_units(cells, a, b, seeds, exclude, families):
    """cl_analyze.seed_units with the four-arm completeness rule."""
    ok = {k: v for k, v in Mae(cells).items() if complete(cells, k[1], k[2], k[3])}
    saved = cc.VARIANTS
    try:
        cc.VARIANTS = ARMS
        return ca.seed_units(ok, a, b, seeds, exclude, families)
    finally:
        cc.VARIANTS = saved


def pair_units(cells, a, b, pairs, families):
    saved = cc.VARIANTS
    try:
        cc.VARIANTS = ARMS
        units = [ca.cell_effect(Mae(cells), a, b, s, si, cc.ALL_PLANTS, families) for s, si in pairs]
    finally:
        cc.VARIANTS = saved
    return [u for u in units if u is not None]


def compute(cells, seeds, pairs, families, missing):
    out = dict(seeds=len(seeds), wrong_pairs=[dict(seed=s, scenario=om.SCENARIO_LABELS[si]) for s, si in pairs],
               non_evaluable_runs=len(missing))
    rev = {a: [seed_reversal(cells, a, s, families) for s in seeds] for a in ARMS}
    irev = {a: [seed_reversal(cells, a, s, families, 'irev', 'ifar') for s in seeds] for a in ARMS}
    out['reversal_seed_means'] = {a: float(np.mean([x for x in v if x is not None])) for a, v in rev.items()}
    out['reversal_internal_seed_means'] = {a: float(np.mean([x for x in v if x is not None])) for a, v in irev.items()}

    def contrast(table, a, b):
        units = [x - y for x, y in zip(table[a], table[b]) if x is not None and y is not None]
        return dict(bootstrap(units), seeds_lower=int(sum(u < 0 for u in units)))
    # H1 (primary): COISC_seq makes fewer reversed moves than COISC over whole trajectories (panel B, seed units)
    out['H1_reversal'] = {f'{a} - {b}': contrast(rev, a, b) for a, b in ((sq.VARIANT, 'COISC'), (sq.VARIANT, 'R0'), ('COISC', 'R0'), ('L0', 'R0'))}
    h1 = out['H1_reversal'][f'{sq.VARIANT} - COISC']
    out['H1_supported'] = bool(h1['units'] == len(seeds) and h1['ci_upper'] < 0)
    # H2 (primary): COISC_seq is non-inferior to R0 in tracking on the pairs without a reversed R0 first move (S4 H2 unit)
    excl = set(pairs)
    out['H2_tracking_unaffected'] = {f'{a} - {b}': bootstrap(mae_seed_units(cells, a, b, seeds, excl, families))
                                     for a, b in ((sq.VARIANT, 'R0'), (sq.VARIANT, 'COISC'), ('COISC', 'R0'))}
    h2 = out['H2_tracking_unaffected'][f'{sq.VARIANT} - R0']
    out['H2_supported'] = bool(h2['units'] == len(seeds) and h2['ci_upper'] < NI_MARGIN_C)
    out['primary_verdict'] = ('SUPPORTED' if out['H1_supported'] and out['H2_supported'] else
                              'PARTIAL_H1_ONLY' if out['H1_supported'] else 'PARTIAL_H2_ONLY' if out['H2_supported'] else 'NOT_SUPPORTED')
    # secondary, descriptive
    out['reversal_internal'] = {f'{a} - {b}': contrast(irev, a, b) for a, b in ((sq.VARIANT, 'COISC'), (sq.VARIANT, 'R0'))}
    out['tracking_all_pairs_panel_B'] = {f'{a} - {b}': bootstrap(mae_seed_units(cells, a, b, seeds, set(), families))
                                         for a, b in ((sq.VARIANT, 'R0'), (sq.VARIANT, 'COISC'))}
    out['panel_A_wrong_pairs'] = {f'{a} - {b}': dict(bootstrap(u := pair_units(cells, a, b, pairs, families)), per_pair=u)
                                  for a, b in ((sq.VARIANT, 'R0'), (sq.VARIANT, 'COISC'), ('COISC', 'R0'))}
    out['panel_B_means'] = {a: {k: float(np.mean([c[k] for (arm, pi, si, s), c in cells.items() if arm == a and pi in cc.PANEL_B_PLANTS]))
                                for k in ('MAE_track', 'total_variation_u', 'saturation_fraction')} for a in ARMS}
    return out


def study_cells(seq_root, plants):
    cl_tasks = cc.read(sq.CL_STUDY / 'TASKS.json')
    cells, missing = load_cells(sq.CL_STUDY, cl_tasks, plants)
    s_cells, s_missing = load_cells(seq_root, sq.read(Path(seq_root) / 'TASKS.json'), plants)
    cells.update(s_cells)
    return cells, missing + s_missing


def selftest():
    families = ['F1'] * 17 + ['F2'] * 3 + ['F3'] * 3 + ['F4']
    seeds = (1, 2, 3, 4)
    pairs = [(1, 0)]
    base = dict(R0=(2.0, 6), L0=(1.9, 5), COISC=(1.95, 2), COISC_seq=(1.97, 1))  # (MAE, reversed updates of 20 far updates)
    cells = {}
    for s in seeds:
        for si in range(12):
            for pi in range(24):
                if pi not in cc.PANEL_B_PLANTS and (s, si) not in pairs:
                    continue
                for a, (m, r) in base.items():
                    cells[(a, pi, si, s)] = dict(MAE_track=m + 0.01 * pi + 0.001 * s, rev=r + (s % 2), far=20, irev=r, ifar=20,
                                                 total_variation_u=0., saturation_fraction=0.)
    r = compute(cells, seeds, pairs, families, [])
    checks = dict(h1_point=abs(r['H1_reversal']['COISC_seq - COISC']['point'] - (-0.05)) < 1e-9,
                  h2_point=abs(r['H2_tracking_unaffected']['COISC_seq - R0']['point'] - (-0.03)) < 1e-9,
                  panel_a=abs(r['panel_A_wrong_pairs']['COISC_seq - R0']['point'] - (-0.03)) < 1e-9,
                  verdict=r['primary_verdict'] == 'SUPPORTED')
    for k in list(cells):  # COISC_seq reverses more than COISC everywhere -> H1 must fail
        if k[0] == sq.VARIANT:
            cells[k]['rev'] += 2
    r2 = compute(cells, seeds, pairs, families, [])
    checks.update(h1_fails=r2['H1_supported'] is False, verdict_h2_only=r2['primary_verdict'] == 'PARTIAL_H2_ONLY')
    for k in list(cells):  # and tracking worse than the margin -> H2 must fail too
        if k[0] == sq.VARIANT:
            cells[k]['MAE_track'] += 0.2
    r3 = compute(cells, seeds, pairs, families, [])
    checks.update(verdict_none=r3['primary_verdict'] == 'NOT_SUPPORTED')
    return dict(checks=checks, pass_=all(checks.values()))


def gate():
    """With COISC relabelled as the study arm, the code must reproduce the stored S4 H2 contrast and the Fig. 3(d) seed fractions."""
    plants, _ = om.plants_and_formal_index()
    families = [p['family'] for p in plants]
    tasks = cc.read(sq.CL_STUDY / 'TASKS.json')
    cells, missing = load_cells(sq.CL_STUDY, tasks, plants)
    for k in [k for k in cells if k[0] == 'COISC']:
        cells[(sq.VARIANT,) + k[1:]] = cells[k]
    pairs = cc.wrong_pairs(cc.first_actions())
    stored = cc.read(sq.CL_STUDY / 'CLOSEDLOOP_RESULTS.json')
    got_h2 = bootstrap(mae_seed_units(cells, 'COISC', 'R0', list(sq.NEW_SEEDS), set(pairs), families))
    ref_h2 = ca.bootstrap(ca.seed_units(Mae({k: v for k, v in cells.items() if k[0] != sq.VARIANT}), 'COISC', 'R0',
                                        list(sq.NEW_SEEDS), set(pairs), families))
    h2_exact = repr(got_h2['point']) == repr(stored['panel_B_unaffected']['COISC - R0']['point']) == repr(ref_h2['point'])
    pa = pair_units(cells, 'COISC', 'R0', pairs, families)
    pa_exact = [repr(x) == repr(y) for x, y in zip(pa, stored['panel_A']['COISC - R0']['per_pair'])]
    # Fig. 3(d) used all plants; recompute with plants=ALL on full cells for the exact check
    with FIG3D_CSV.open(encoding='utf-8') as f:
        fig = {(int(r['seed']), r['variant']): r for r in csv.DictReader(f)}
    fig_diff = 0.
    for s in sq.NEW_SEEDS:
        for a in ('R0', 'L0', 'COISC'):
            p = seed_reversal(cells, a, s, families, plants=cc.ALL_PLANTS, arms=('R0', 'COISC', 'L0'))
            i = seed_reversal(cells, a, s, families, 'irev', 'ifar', plants=cc.ALL_PLANTS, arms=('R0', 'COISC', 'L0'))
            fig_diff = max(fig_diff, abs(p - float(fig[(s, a)]['reversal_plant'])), abs(i - float(fig[(s, a)]['reversal_internal'])))
    st = selftest()
    report = dict(label='Sequence study gate G6: analysis code reproduces stored S4 H2 point, panel-A per-pair COISC-R0 and the '
                        'Fig. 3(d) seed-level reversal fractions from raw runs; synthetic selftest',
                  runs_loaded=len(tasks), missing=len(missing), h2_point_exact=h2_exact, panel_a_exact=pa_exact,
                  fig3d_max_abs_diff=fig_diff, selftest=st,
                  pass_=bool(not missing and h2_exact and all(pa_exact) and len(pa_exact) == 6 and fig_diff < 1e-12 and st['pass_']),
                  code_sha256=sq.code_hashes(('sq_common.py', 'sq_analyze.py')), created_utc=sq.utc())
    sq.write(sq.PROTO / 'GATE_G6_ANALYSIS_REPORT.json', report)
    print('G6', {k: report[k] for k in ('missing', 'h2_point_exact', 'panel_a_exact', 'fig3d_max_abs_diff', 'pass_')}, 'selftest', st, flush=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'gate':
        gate()
    elif mode == 'smoke':
        plants, _ = om.plants_and_formal_index()
        root = sq.SMOKE / 'mil'
        tasks = sq.read(root / 'TASKS.json')
        cells, missing = load_cells(root, tasks, plants)
        assert len(cells) + len(missing) == len(tasks)
        sq.write(sq.SMOKE / 'SMOKE_ANALYSIS_OK.json', dict(rows=len(cells), missing=len(missing), selftest=selftest()['pass_'],
                                                         code_sha256=sq.code_hashes(('sq_common.py', 'sq_analyze.py')), at=sq.utc()))
        print('SMOKE ANALYSIS OK', len(cells), len(missing), flush=True)
    elif mode == 'analyze':
        protocol, digest = sq.frozen_protocol()
        root = sq.STUDY / 'mil'
        audit = sq.read(root / 'EXECUTION_AUDIT.json')
        assert audit['status'] == 'RAW_RUNS_COMPLETE' and audit['protocol_sha256'] == digest
        plants, _ = om.plants_and_formal_index()
        cells, missing = study_cells(root, plants)
        res = compute(cells, list(sq.NEW_SEEDS), cc.wrong_pairs(cc.first_actions()), [p['family'] for p in plants], missing)
        res.update(protocol_sha256=digest, at=sq.utc(), far_band_C=FAR_BAND_C, ni_margin_C=NI_MARGIN_C)
        sq.write(sq.STUDY / 'SEQ_RESULTS.json', res)
        print('VERDICT', res['primary_verdict'], 'H1', res['H1_supported'], 'H2', res['H2_supported'], flush=True)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
