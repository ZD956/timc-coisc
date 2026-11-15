"""Pre-registered analysis of the closed-loop confirmation study.

  cl_analyze.py gate           G2: metric extraction reproduces stored E2 MAE_track (seed 20261001) + synthetic selftest
  cl_analyze.py smoke          runs the full analysis code path on the smoke directory (no inference)
  cl_analyze.py analyze        frozen protocol and author authorization required; writes study/CLOSEDLOOP_RESULTS.json

Unit of tracking error: MAE_track (onset_analyze_am1.metric_row / analyze_formal_control_v3.trace_metrics, unchanged).
A (seed, scenario, plant) cell enters a contrast only if all three variants are evaluable there; dropped cells are counted.
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

import numpy as np

import cl_common as cc
import onset_mil as om  # noqa: E402
import onset_analyze_am1 as oa  # noqa: E402

E2_EXECUTION = cc.ONSET / 'mil_execution_20261003T104638Z_23b8f69525'


def load_rows(root, tasks, plants):
    rows, missing = {}, []
    for i, t in enumerate(tasks):
        path = Path(root) / 'raw_runs' / f'{i:05d}.npz'
        meta, data = oa.load_run(path)
        assert (meta['plant_index'], meta['scenario_index'], meta['controller_seed'], meta['controller_variant']) == \
            (t['plant_index'], t['scenario_index'], t['seed'], t['variant']), f'task/metadata mismatch at {i}'
        row = oa.metric_row(meta, data, t['variant'], t['plant_index'], t['scenario_index'], t['seed'], plants[t['plant_index']]['family'])
        if row is None:
            missing.append(i)
        else:
            rows[(t['variant'], t['plant_index'], t['scenario_index'], t['seed'])] = row
    return rows, missing


def cell_effect(mae, a, b, seed, si, plant_set, families):
    """Family-equal mean of MAE[a] - MAE[b] over the plants of plant_set; None if no complete family remains."""
    per_family = []
    for f in cc.FAMILIES:
        vals = [mae[(a, pi, si, seed)] - mae[(b, pi, si, seed)] for pi in plant_set if families[pi] == f
                and all((v, pi, si, seed) in mae for v in cc.VARIANTS)]
        if vals:
            per_family.append(float(np.mean(vals)))
    return float(np.mean(per_family)) if per_family else None


def bootstrap(units, draws=cc.BOOTSTRAP_DRAWS, seed=cc.BOOTSTRAP_SEED):
    values = np.asarray(units, dtype=np.float64)
    if len(values) == 0:
        return dict(units=0, point=None, ci_lower=None, ci_upper=None)
    idx = np.random.default_rng(seed).integers(0, len(values), size=(draws, len(values)))
    lower, upper = np.percentile(values[idx].mean(axis=1), [2.5, 97.5])
    return dict(units=int(len(values)), point=float(values.mean()), ci_lower=float(lower), ci_upper=float(upper))


def seed_units(mae, a, b, seeds, exclude, families):
    """Panel B: per-seed mean over its scenarios (excluding R0-wrong pairs) of the family-equal effect on the 6 plants."""
    units = []
    for s in seeds:
        cells = [cell_effect(mae, a, b, s, si, cc.PANEL_B_PLANTS, families) for si in range(12) if (s, si) not in exclude]
        cells = [c for c in cells if c is not None]
        if cells:
            units.append(float(np.mean(cells)))
    return units


def compute(mae, fa, seeds, families, missing_runs):
    wrong = cc.wrong_pairs(fa, seeds)
    out = dict(wrong_pairs=[dict(seed=s, scenario=om.SCENARIO_LABELS[si]) for s, si in wrong], non_evaluable_runs=len(missing_runs))
    # H1 (primary, panel A): on R0-wrong pairs, COISC improves closed-loop tracking over R0 (all 24 plants, family-equal)
    pair_units = {f'{a} - {b}': [cell_effect(mae, a, b, s, si, cc.ALL_PLANTS, families) for s, si in wrong]
                  for a, b in (('COISC', 'R0'), ('L0', 'R0'), ('COISC', 'L0'))}
    pair_units = {k: [u for u in v if u is not None] for k, v in pair_units.items()}
    out['panel_A'] = {k: dict(bootstrap(v), per_pair=v) for k, v in pair_units.items()}
    h1 = out['panel_A']['COISC - R0']
    out['H1_supported'] = bool(h1['units'] == len(wrong) and h1['units'] > 0 and h1['ci_upper'] < 0)
    out['H1_pairs_improved'] = int(sum(u < 0 for u in pair_units['COISC - R0']))
    # H2 (primary, panel B): on all other seed-scenario pairs COISC is non-inferior to R0 at NI_MARGIN_C
    excl = set(wrong)
    out['panel_B_unaffected'] = {f'{a} - {b}': bootstrap(seed_units(mae, a, b, seeds, excl, families))
                                 for a, b in (('COISC', 'R0'), ('L0', 'R0'), ('COISC', 'L0'))}
    h2 = out['panel_B_unaffected']['COISC - R0']
    out['H2_supported'] = bool(h2['units'] == len(seeds) and h2['ci_upper'] < cc.NI_MARGIN_C)
    # secondary (descriptive): panel B over all pairs, including the R0-wrong ones
    out['panel_B_all_pairs'] = {f'{a} - {b}': bootstrap(seed_units(mae, a, b, seeds, set(), families))
                                for a, b in (('COISC', 'R0'), ('L0', 'R0'), ('COISC', 'L0'))}
    out['primary_verdict'] = ('SUPPORTED' if out['H1_supported'] and out['H2_supported'] else
                              'PARTIAL_H1_ONLY' if out['H1_supported'] else 'PARTIAL_H2_ONLY' if out['H2_supported'] else 'NOT_SUPPORTED')
    # secondary descriptive control-effort and solver summaries per variant (panel B, all pairs)
    sec = {}
    for v in cc.VARIANTS:
        rs = [r for (vv, pi, si, s), r in mae['_rows'].items() if vv == v and pi in cc.PANEL_B_PLANTS]
        sec[v] = {k: float(np.mean([r[k] for r in rs])) for k in ('MAE_track', 'total_variation_u', 'saturation_fraction', 'solver_success_rate')}
    out['panel_B_descriptive_means'] = sec
    return out


def analysis(root, fa, seeds, plants):
    tasks = cc.read(Path(root) / 'TASKS.json')
    rows, missing = load_rows(root, tasks, plants)
    mae = {k: r['MAE_track'] for k, r in rows.items()}
    mae['_rows'] = rows
    families = [p['family'] for p in plants]
    res = compute(_Mae(mae), fa, seeds, families, missing)
    with (Path(root) / 'CLOSEDLOOP_RUN_METRICS.csv').open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(rows.values()))))
        w.writeheader()
        w.writerows(rows.values())
    return res


class _Mae(dict):
    """MAE lookup that also carries the full metric rows under '_rows' without counting them as cells."""
    def __contains__(self, key):
        return key != '_rows' and dict.__contains__(self, key)


def selftest():
    """Synthetic check: known COISC-R0 effects are recovered and verdict logic is correct."""
    families = ['F1'] * 17 + ['F2'] * 3 + ['F3'] * 3 + ['F4']
    seeds = (1, 2, 3, 4)
    fa = {(s, v, si): dict(first_action='wrong' if (v == 'R0' and s in (1, 2) and si == 0) else 'correct')
          for s in seeds for v in cc.VARIANTS for si in range(12)}
    mae = _Mae()
    for s in seeds:
        for si in range(12):
            for pi in range(24):
                base = 1.0 + 0.01 * pi
                mae[('R0', pi, si, s)] = base + (2.0 if (s in (1, 2) and si == 0) else 0.)
                mae[('L0', pi, si, s)] = base + (1.0 if (s in (1, 2) and si == 0) else 0.)
                mae[('COISC', pi, si, s)] = base + 0.05
    mae['_rows'] = {}
    rows = {k: dict(MAE_track=v, total_variation_u=0., saturation_fraction=0., solver_success_rate=1.) for k, v in mae.items() if k != '_rows'}
    mae['_rows'] = rows
    r = compute(mae, fa, seeds, families, [])
    checks = dict(h1_point=abs(r['panel_A']['COISC - R0']['point'] - (-1.95)) < 1e-9,
                  h1_supported=r['H1_supported'] is True,
                  h2_point=abs(r['panel_B_unaffected']['COISC - R0']['point'] - 0.05) < 1e-9,
                  h2_supported=r['H2_supported'] is True, verdict=r['primary_verdict'] == 'SUPPORTED',
                  l0_pair=abs(r['panel_A']['L0 - R0']['point'] - (-1.0)) < 1e-9)
    for s in seeds:  # make COISC worse than the margin -> H2 must fail
        for si in range(12):
            for pi in range(24):
                mae[('COISC', pi, si, s)] += 0.2
    r2 = compute(mae, fa, seeds, families, [])
    checks.update(h2_fails_when_worse=r2['H2_supported'] is False, verdict_h1_only=r2['primary_verdict'] == 'PARTIAL_H1_ONLY')
    return dict(checks=checks, pass_=all(checks.values()))


def gate():
    plants, _ = om.plants_and_formal_index()
    tasks = cc.read(E2_EXECUTION / 'TASKS.json')
    with (E2_EXECUTION / 'ONSET_MIL_SECONDARY.csv').open(encoding='utf-8') as f:
        stored = {(r['variant'], int(r['plant_index']), r['scenario'], int(r['seed'])): float(r['MAE_track'])
                  for r in csv.DictReader(f) if int(r['seed']) == cc.REPRO_SEED and r['variant'] in ('R0', 'L0')}
    mism, n = 0, 0
    for i, t in enumerate(tasks):
        if t['seed'] != cc.REPRO_SEED or t['variant'] not in ('R0', 'L0'):
            continue
        meta, data = oa.load_run(E2_EXECUTION / 'raw_runs' / f'{i:05d}.npz')
        row = oa.metric_row(meta, data, t['variant'], t['plant_index'], t['scenario_index'], t['seed'], plants[t['plant_index']]['family'])
        key = (t['variant'], t['plant_index'], om.SCENARIO_LABELS[t['scenario_index']], t['seed'])
        n += 1
        if row is None:
            mism += key in stored
        else:
            mism += repr(row['MAE_track']) != repr(stored[key])
    st = selftest()
    report = dict(label='Closed-loop study gate G2: metric extraction reproduces stored E2 MAE_track; synthetic analysis selftest',
                  e2_runs_checked=n, mismatches=mism, selftest=st, pass_=mism == 0 and n == 576 and st['pass_'],
                  code_sha256=cc.code_hashes(('cl_common.py', 'cl_analyze.py')), created_utc=cc.utc())
    cc.write(cc.PROTO / 'GATE_G2_ANALYSIS_REPORT.json', report)
    print('G2', dict(checked=n, mismatches=mism, selftest=st['pass_'], pass_=report['pass_']), flush=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'gate':
        gate()
    elif mode == 'smoke':
        plants, _ = om.plants_and_formal_index()
        tasks = cc.read(cc.SMOKE / 'TASKS.json')
        rows, missing = load_rows(cc.SMOKE, tasks, plants)
        assert len(rows) + len(missing) == len(tasks)
        cc.write(cc.SMOKE / 'SMOKE_ANALYSIS_OK.json', dict(rows=len(rows), missing=len(missing), selftest=selftest()['pass_'],
                                                         code_sha256=cc.code_hashes(('cl_common.py', 'cl_analyze.py')), at=cc.utc()))
        print('SMOKE ANALYSIS OK', len(rows), len(missing), flush=True)
    elif mode == 'analyze':
        protocol, digest = cc.frozen_protocol()
        audit = cc.read(cc.STUDY / 'EXECUTION_AUDIT.json')
        assert audit['status'] == 'RAW_RUNS_COMPLETE' and audit['protocol_sha256'] == digest
        plants, _ = om.plants_and_formal_index()
        res = analysis(cc.STUDY, cc.first_actions(), cc.NEW_SEEDS, plants)
        res.update(protocol_sha256=digest, at=cc.utc(), ni_margin_C=cc.NI_MARGIN_C)
        cc.write(cc.STUDY / 'CLOSEDLOOP_RESULTS.json', res)
        print('VERDICT', res['primary_verdict'], 'H1', res['H1_supported'], 'H2', res['H2_supported'], flush=True)
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
