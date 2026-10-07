"""Post-hoc MIL contrasts with the formal hierarchical_effect and the shared 20000 draws.

usage:
  analyze_posthoc_mil.py check-formal   reproduce the formal COISC-R0 primary result (already reported, T1)
  analyze_posthoc_mil.py analyze EXEC_DIR
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

from run_posthoc_mil import (ANALYSIS_CHECK, CONTROL_DIR, FORMAL_EXECUTION, PRIMARY_PLANTS, SEEDS, SELFTEST,  # noqa: E402
                             plants_and_formal_index, read, task_list, utc, verify_frozen)
import analyze_formal_control_v3 as af  # noqa: E402
from paper_data import write_csv  # noqa: E402

FAMILIES = ('F1', 'F2', 'F3', 'F4')
CONTRASTS = (
    ('A1_primary', 'COISC - L0', lambda m, k: m['COISC', k] - m['L0', k]),
    ('A2', 'L0 - R0', lambda m, k: m['L0', k] - m['R0', k]),
    ('B1_primary', 'COISC-DE - R0-DE', lambda m, k: m['COISC-DE', k] - m['R0-DE', k]),
    ('B2', 'COISC-DE - L0-DE', lambda m, k: m['COISC-DE', k] - m['L0-DE', k]),
    ('B3_R0', 'R0-DE - R0', lambda m, k: m['R0-DE', k] - m['R0', k]),
    ('B3_COISC', 'COISC-DE - COISC', lambda m, k: m['COISC-DE', k] - m['COISC', k]),
    ('B3_L0', 'L0-DE - L0', lambda m, k: m['L0-DE', k] - m['L0', k]),
    ('B4_secondary_DiD', '(COISC-DE - R0-DE) - (COISC - R0)',
     lambda m, k: (m['COISC-DE', k] - m['R0-DE', k]) - (m['COISC', k] - m['R0', k])),
)


def shared_draws():
    draws = np.random.default_rng(20260927).integers(0, 12, size=(20000, 12), dtype=np.int16)
    with np.load(FORMAL_EXECUTION/'CONTROL_BOOTSTRAP_DRAWS.npz', allow_pickle=False) as z:
        if not np.array_equal(z['source_draws'], draws):
            raise ValueError('regenerated bootstrap draws differ from the formal draws')
    return draws


def family_of(plants):
    return [plants[pi]['family'] for pi in range(PRIMARY_PLANTS)]


def contrast(mae, fn, seed, families, draws):
    counts = read(CONTROL_DIR/'CONTROL_PLANT_FAMILY_LOCK.json')['primary_checkpoint_counts']
    matrix = np.empty((12, 4), dtype=np.float64)
    for si in range(12):
        for fi, family in enumerate(FAMILIES):
            values = [fn(mae, (pi, si, seed)) for pi in range(PRIMARY_PLANTS) if families[pi] == family]
            assert len(values) == counts[family]
            matrix[si, fi] = np.mean(values)
    point, lower, upper = af.hierarchical_effect(matrix, draws)
    return point, lower, upper


def load_run(path):
    with np.load(path, allow_pickle=False) as z:
        data = {key: z[key] for key in z.files if key != 'metadata'}
        meta = json.loads(str(z['metadata']))
    return meta, data


def formal_nominal(plants, formal_index, variants=('R0', 'COISC')):
    runs = {}
    for pi in range(PRIMARY_PLANTS):
        for si in range(12):
            for seed in SEEDS:
                for variant in variants:
                    meta, data = load_run(FORMAL_EXECUTION/'raw_runs'/f'{formal_index[(pi, si, seed, variant)]:04d}.npz')
                    assert meta['plant'] == plants[pi]['plant_name'] and meta['controller_variant'] == variant
                    runs[(variant, (pi, si, seed))] = (meta, data)
    return runs


def check_formal():
    plants, formal_index = plants_and_formal_index()
    draws = shared_draws()
    families = family_of(plants)
    runs = formal_nominal(plants, formal_index)
    mae = {key: af.trace_metrics(d, m['previous_uh_percent'], m['initial_Tp_C'], m['runtime_s'])['MAE_track']
           for key, (m, d) in runs.items()}
    with (FORMAL_EXECUTION/'CONTROL_PRIMARY_RESULTS.csv').open(encoding='utf-8-sig', newline='') as stream:
        reported = {int(r['seed']): r for r in csv.DictReader(stream)}
    rows = []
    for seed in SEEDS:
        point, lower, upper = contrast(mae, lambda m, k: m['COISC', k] - m['R0', k], seed, families, draws)
        ref = reported[seed]
        rows.append(dict(seed=seed, point=point, ci_lower=lower, ci_upper=upper,
                         reported=[float(ref['primary_family_equal_Delta']), float(ref['ci_lower']), float(ref['ci_upper'])],
                         exact=(point, lower, upper) == (float(ref['primary_family_equal_Delta']), float(ref['ci_lower']), float(ref['ci_upper']))))
    result = dict(created_utc=utc(), purpose='analysis pipeline reproduces the already-reported formal T1 COISC-R0 primary result',
                  draws_identical_to_formal=True, rows=rows, passed=all(r['exact'] for r in rows))
    ANALYSIS_CHECK.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'passed': result['passed'], 'rows': [(r['seed'], r['exact']) for r in rows]}))


def interpret(decisions):
    a1, a2 = decisions['A1_primary'], decisions['A2']
    s23 = 20260923
    if a1[s23]:
        text_a = 'The seed-23 gain is regularizer-specific relative to matched fine-tuning.'
    elif a2[s23]:
        text_a = 'The seed-23 gain is at least partly produced by head-only fine-tuning itself, not specifically by the regularizer.'
    else:
        text_a = 'The seed-23 gain cannot be attributed to either factor.'
    if all(a1.values()):
        overall_a = 'Regularizer-specific control benefit supported on 3/3 seeds.'
    elif any(a1.values()):
        overall_a = 'Partial attribution only: A1 supported on seeds ' + ', '.join(str(s) for s, v in a1.items() if v) + '; no overall regularizer control benefit.'
    else:
        overall_a = 'A1 supported on no seed; no regularizer control benefit.'
    if all(decisions['B1_primary'].values()):
        text_b = ('Under DE-MPC, COISC gave lower tracking error than R0 on every seed (post-hoc, conditional on the MIL '
                  'plant panel). This does not overturn the formal T1 verdict.')
    else:
        text_b = ('Adding disturbance estimation from peripheral measurements did not produce a consistent COISC benefit; '
                  'the non-equivalence finding is robust to this controller design choice.')
    return dict(A_seed23=text_a, A_overall=overall_a, B1=text_b)


def secondary_row(meta, data, key, variant):
    metric = af.trace_metrics(data, meta['previous_uh_percent'], meta['initial_Tp_C'], meta['runtime_s'])
    pi, si, seed = key
    row = dict(plant_index=pi, plant=meta['plant'], family=meta['family'], scenario_id=meta['scenario_id'],
               seed=seed, variant=variant, MAE_track=metric['MAE_track'],
               terminal_signed_offset_last30=float(np.mean(np.asarray(data['plant_Tp'][-30:], dtype=np.float64)-37.)),
               total_variation_u=metric['total_variation_u'], saturation_fraction=metric['saturation_fraction'],
               solver_success_rate=metric['solver_success_rate'], iterations=metric['iterations'],
               time_per_control_step_s=metric['time_per_control_step_s'])
    if 'd_hat' in data:
        status = [str(s) for s in data['estimator_status']]
        row.update(d_hat_final=float(data['d_hat'][-1]), d_hat_max_abs=float(np.max(np.abs(data['d_hat']))),
                   estimator_failures=sum(s.startswith(('ERROR', 'NONFINITE')) for s in status))
    return row, metric


def analyze(execution_dir):
    execution = Path(execution_dir).resolve()
    protocol, digest = verify_frozen()
    audit = read(execution/'POSTHOC_MIL_EXECUTION_AUDIT.json')
    assert audit['protocol_sha256'] == digest and audit['status'] == 'RAW_RUNS_COMPLETE'
    plants, formal_index = plants_and_formal_index()
    rerun = protocol['g4_outcome']['rerun_formal_nominal']
    tasks = task_list(rerun)
    runs = {} if rerun else formal_nominal(plants, formal_index)
    for i, task in enumerate(tasks):
        path = execution/'raw_runs'/f'{i:04d}.npz'
        if not path.is_file():
            continue
        meta, data = load_run(path)
        assert meta['plant'] == plants[task['plant_index']]['plant_name'] and meta['controller_variant'] == task['variant']
        runs[(task['variant'], (task['plant_index'], task['scenario_index'], task['seed']))] = (meta, data)
    return compute(runs, plants, execution, digest)


def selftest(out_dir):
    """Synthetic pipeline check: new variants are relabelled copies of formal runs, so A2 and B3_R0 must be exactly 0."""
    plants, formal_index = plants_and_formal_index()
    runs = formal_nominal(plants, formal_index)
    for (variant, key), (meta, data) in list(runs.items()):
        if variant == 'R0':
            fake = dict(data, d_hat=np.zeros(60), estimator_status=np.array(['OK']*60))
            runs[('L0', key)] = ({**meta, 'controller_variant': 'L0'}, data)
            runs[('R0-DE', key)] = ({**meta, 'controller_variant': 'R0-DE'}, fake)
            runs[('L0-DE', key)] = ({**meta, 'controller_variant': 'L0-DE'}, fake)
        elif variant == 'COISC':
            runs[('COISC-DE', key)] = ({**meta, 'controller_variant': 'COISC-DE'},
                                       dict(data, d_hat=np.zeros(60), estimator_status=np.array(['OK']*60)))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = compute(runs, plants, out, 'SELFTEST')
    with (out/'POSTHOC_MIL_PRIMARY_CONTRASTS.csv').open(encoding='utf-8-sig', newline='') as stream:
        table = list(csv.DictReader(stream))
    zero = all(float(r['point']) == 0. for r in table if r['contrast'] in ('A2', 'B3_R0', 'B3_COISC', 'B3_L0', 'B4_secondary_DiD'))
    b1_equals_formal = [float(r['point']) for r in table if r['contrast'] == 'B1_primary'] == \
        [r['point'] for r in read(ANALYSIS_CHECK)['rows']]
    ok = zero and b1_equals_formal and len(table) == len(CONTRASTS)*3 and summary['runs_evaluated'] == 864*6
    result = {'created_utc': utc(), 'selftest_passed': ok, 'zero_contrasts_exact': zero,
              'B1_equals_formal_COISC_minus_R0': b1_equals_formal, 'contrast_rows': len(table)}
    SELFTEST.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))
    return ok


def compute(runs, plants, execution, digest):
    families = family_of(plants)
    draws = shared_draws()
    mae, rows = {}, []
    for (variant, key), (meta, data) in sorted(runs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        row, metric = secondary_row(meta, data, key, variant)
        if not metric['evaluable']:
            continue
        mae[(variant, key)] = metric['MAE_track']
        rows.append({**row, 'source_execution_id': meta['execution_id']})
    strata = af.source_strata(CONTROL_DIR/'CONTROL_PLANT_REACHABILITY_AUDIT.csv')
    primary_rows, strata_rows, decisions = [], [], {}
    for name, label, fn in CONTRASTS:
        decisions[name] = {}
        for seed in SEEDS:
            try:
                point, lower, upper = contrast(mae, fn, seed, families, draws)
                decision = 'SUPPORTED' if upper < 0. else 'NOT_SUPPORTED'
            except KeyError:
                point = lower = upper = float('nan')
                decision = 'NOT_EVALUABLE'
            decisions[name][seed] = decision == 'SUPPORTED'
            primary_rows.append(dict(contrast=name, definition=label, seed=seed, point=point, ci_lower=lower,
                                     ci_upper=upper, decision=decision,
                                     role='descriptive only' if name.startswith('B4') else 'support rule: CI upper < 0'))
            if decision != 'NOT_EVALUABLE':
                by_stratum = {}
                for pi in range(PRIMARY_PLANTS):
                    for si in range(12):
                        by_stratum.setdefault(strata[(plants[pi]['plant_name'], read_scenario_id(si))], []).append(fn(mae, (pi, si, seed)))
                for stratum, values in sorted(by_stratum.items()):
                    strata_rows.append(dict(contrast=name, seed=seed, stratum=stratum, pairs=len(values),
                                            mean_unweighted=float(np.mean(values))))
    write_csv(execution/'POSTHOC_MIL_PRIMARY_CONTRASTS.csv', primary_rows)
    write_csv(execution/'POSTHOC_MIL_SECONDARY.csv', rows)
    write_csv(execution/'POSTHOC_MIL_REACHABILITY_STRATA.csv', strata_rows)
    b3_worse = {}
    for name in ('B3_R0', 'B3_COISC', 'B3_L0'):
        b3_worse[name] = sum(r['point'] > 0 for r in primary_rows if r['contrast'] == name)
    summary = dict(created_utc=utc(), protocol_sha256=digest, evidence_tier='T3 post-hoc; does not revise T1',
                   per_contrast={n: {str(s): v for s, v in d.items()} for n, d in decisions.items()},
                   overall_3_of_3={n: all(d.values()) for n, d in decisions.items() if not n.startswith('B4')},
                   B3_point_estimate_positive_seed_count=b3_worse,
                   B3_worsened={n: c >= 2 for n, c in b3_worse.items()},
                   interpretation=interpret(decisions),
                   runs_evaluated=len(mae), estimator_failures_total=int(sum(r.get('estimator_failures', 0) for r in rows)))
    (execution/'POSTHOC_MIL_RESULTS_SUMMARY.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    if digest != 'SELFTEST':
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


_SCENARIO_IDS = None


def read_scenario_id(si):
    global _SCENARIO_IDS
    if _SCENARIO_IDS is None:
        _SCENARIO_IDS = [s['scenario_id'] for s in read(CONTROL_DIR/'CONTROL_SCENARIO_LOCK_V2.json')['scenarios']]
    return _SCENARIO_IDS[si]


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'check-formal':
        check_formal()
    elif len(sys.argv) == 3 and sys.argv[1] == 'selftest':
        raise SystemExit(0 if selftest(sys.argv[2]) else 1)
    elif len(sys.argv) == 3 and sys.argv[1] == 'analyze':
        analyze(sys.argv[2])
    else:
        raise SystemExit(__doc__)
