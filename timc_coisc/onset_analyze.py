"""MIL analysis for the post-hoc onset-aligned study; formal hierarchical_effect and the shared 20000 draws.

usage:
  onset_analyze.py check-formal        R6: reproduce already-reported COISC-R0 (formal) and A1/A2 (post-hoc MIL) contrasts exactly
  onset_analyze.py selftest            relabelled already-reported runs through the full pipeline; known answers must come out
  onset_analyze.py analyze EXEC_DIR    frozen protocol + authorization; writes ONSET_MIL_* outputs into EXEC_DIR
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

import onset_mil as om  # noqa: E402
import analyze_formal_control_v3 as af  # noqa: E402

FAMILIES = ('F1', 'F2', 'F3', 'F4')
UNIT_BOOTSTRAP_SEED = 20261004
UNIT_BOOTSTRAP_DRAWS = 20000
ANALYSIS_GATE_REPORT = om.PROTO / 'GATES_ANALYSIS_REPORT.json'
SELFTEST_DIR = om.OUT / 'gates' / 'R6_selftest'


def write_csv(path, rows):
    fields = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def shared_draws():
    draws = np.random.default_rng(20260927).integers(0, 12, size=(20000, 12), dtype=np.int16)
    with np.load(om.FORMAL_EXECUTION / 'CONTROL_BOOTSTRAP_DRAWS.npz', allow_pickle=False) as z:
        if not np.array_equal(z['source_draws'], draws):
            raise ValueError('regenerated bootstrap draws differ from the formal draws')
    return draws


class Bench:
    def __init__(self, plants):
        self.families = [p['family'] for p in plants]
        self.counts = om.read(om.CONTROL_DIR / 'CONTROL_PLANT_FAMILY_LOCK.json')['primary_checkpoint_counts']
        self.draws = shared_draws()

    def matrix(self, mae, a, b, seed, b_override=None, a_override=None):
        """12x4 family-equal matrix of MAE[a] - MAE[b]; overrides map scenario index -> replacement variant."""
        out = np.empty((12, 4), dtype=np.float64)
        for si in range(12):
            ref = (b_override or {}).get(si, b)
            new = (a_override or {}).get(si, a)
            for fi, family in enumerate(FAMILIES):
                values = [mae[(new, pi, si, seed)] - mae[(ref, pi, si, seed)] for pi in range(om.PRIMARY_PLANTS) if self.families[pi] == family]
                assert len(values) == self.counts[family]
                out[si, fi] = np.mean(values)
        return out

    def effect(self, matrix):
        return af.hierarchical_effect(matrix, self.draws)


def unit_bootstrap(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return dict(units=0, point=None, ci_lower=None, ci_upper=None)
    idx = np.random.default_rng(UNIT_BOOTSTRAP_SEED).integers(0, len(values), size=(UNIT_BOOTSTRAP_DRAWS, len(values)))
    lower, upper = np.percentile(values[idx].mean(axis=1), [2.5, 97.5])
    return dict(units=len(values), point=float(values.mean()), ci_lower=float(lower), ci_upper=float(upper))


def load_run(path):
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files if k != 'metadata'}
        meta = json.loads(str(z['metadata']))
    return meta, data


def metric_row(meta, data, variant, pi, si, seed, family):
    m = af.trace_metrics(data, meta['previous_uh_percent'], meta['initial_Tp_C'], meta['runtime_s'])
    if not m['evaluable']:
        return None
    return dict(variant=variant, seed=seed, plant_index=pi, family=family, scenario=om.SCENARIO_LABELS[si],
                MAE_track=m['MAE_track'], terminal_signed_offset_last30=float(np.mean(np.asarray(data['plant_Tp'][-30:], dtype=np.float64) - 37.)),
                total_variation_u=m['total_variation_u'], saturation_fraction=m['saturation_fraction'],
                solver_success_rate=m['solver_success_rate'], iterations=m['iterations'],
                first_u=float(data['applied_UH'][0]))


def existing_runs(plants, variants=('R0', 'COISC', 'L0')):
    _, formal_index = om.plants_and_formal_index()
    l0_idx = om.l0_index()
    rows = {}
    for pi in range(om.PRIMARY_PLANTS):
        for si in range(12):
            for seed in om.E1_SEEDS:
                for v in variants:
                    path = (om.L0_EXECUTION / 'raw_runs' / f'{l0_idx[(pi, si, seed)]:04d}.npz' if v == 'L0'
                            else om.FORMAL_EXECUTION / 'raw_runs' / f'{formal_index[(pi, si, seed, v)]:04d}.npz')
                    meta, data = load_run(path)
                    assert meta['plant'] == plants[pi]['plant_name'] and meta['controller_variant'] == v
                    row = metric_row(meta, data, v, pi, si, seed, plants[pi]['family'])
                    if row is not None:
                        rows[(v, pi, si, seed)] = row
    return rows


def check_formal():
    plants, _ = om.plants_and_formal_index()
    bench = Bench(plants)
    rows = existing_runs(plants)
    mae = {k: r['MAE_track'] for k, r in rows.items()}
    with (om.FORMAL_EXECUTION / 'CONTROL_PRIMARY_RESULTS.csv').open(encoding='utf-8-sig', newline='') as stream:
        formal = {int(r['seed']): r for r in csv.DictReader(stream)}
    with (om.L0_EXECUTION / 'POSTHOC_MIL_PRIMARY_CONTRASTS.csv').open(encoding='utf-8-sig', newline='') as stream:
        mil = {(r['contrast'], int(r['seed'])): r for r in csv.DictReader(stream)}
    out = []
    for seed in om.E1_SEEDS:
        for name, a, b, ref in (('COISC-R0 (formal T1)', 'COISC', 'R0', [formal[seed][k] for k in ('primary_family_equal_Delta', 'ci_lower', 'ci_upper')]),
                                ('A1 COISC-L0 (post-hoc MIL)', 'COISC', 'L0', [mil[('A1_primary', seed)][k] for k in ('point', 'ci_lower', 'ci_upper')]),
                                ('A2 L0-R0 (post-hoc MIL)', 'L0', 'R0', [mil[('A2', seed)][k] for k in ('point', 'ci_lower', 'ci_upper')])):
            got = bench.effect(bench.matrix(mae, a, b, seed))
            reported = tuple(float(x) for x in ref)
            out.append(dict(seed=seed, contrast=name, recomputed=got, reported=reported, exact=got == reported))
    result = dict(created_utc=om.utc(), rows=out, draws_identical_to_formal=True, pass_=all(r['exact'] for r in out))
    report = om.read(ANALYSIS_GATE_REPORT) if ANALYSIS_GATE_REPORT.is_file() else {}
    report.update(R6_check_formal=result, code_sha256={'onset_analyze.py': om.sha(Path(__file__))})
    report['all_pass'] = all(report[k]['pass_'] for k in ('R6_check_formal', 'R6_selftest') if k in report)
    ANALYSIS_GATE_REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'R6': result['pass_'], 'rows': [(r['seed'], r['contrast'], r['exact']) for r in out]}))
    return result['pass_']


def classes_from_rows(rows):
    """First-action class per (seed, variant, scenario) from the first applied command of any run."""
    scenarios, _ = om.rf.load_scenarios(om.read(om.FORMAL_PROTOCOL))
    out = {}
    for (v, pi, si, seed), r in rows.items():
        c = om.first_action_class(r['first_u'], scenarios[si])
        assert out.setdefault((seed, v, si), c) == c
    return out


def subset_units(bench, mae, a, b, seed, scenarios_subset, b_override=None):
    if not scenarios_subset:
        return []
    m = bench.matrix(mae, a, b, seed, b_override)
    return [float(m[si].mean()) for si in scenarios_subset]


def effect_row(label, seed, point_lower_upper, **extra):
    p, lo, hi = point_lower_upper
    return dict(contrast=label, seed=seed, point=p, ci_lower=lo, ci_upper=hi, ci_upper_below_0=bool(hi < 0), **extra)


def compute(rows, fa, out_dir, digest, e2_seeds=om.E2_SEEDS):
    plants, _ = om.plants_and_formal_index()
    bench = Bench(plants)
    mae = {k: r['MAE_track'] for k, r in rows.items()}
    cls = lambda seed, v, si: fa[(seed, v, si)]  # noqa: E731
    wrong = {s: [si for si in range(12) if cls(s, 'R0', si) == 'wrong'] for s in om.E1_SEEDS + tuple(e2_seeds)}
    right = {s: [si for si in range(12) if cls(s, 'R0', si) == 'correct'] for s in wrong}
    contrasts, split = [], []

    for s in om.E1_SEEDS:
        for a, b in (('COISC_onset', 'R0'), ('COISC_onset', 'COISC'), ('COISC_onset', 'L0'), ('COISC', 'R0'), ('L0', 'R0')):
            contrasts.append(effect_row(f'E1 {a} - {b}', s, bench.effect(bench.matrix(mae, a, b, s)), study='E1'))
        if wrong[s]:
            contrasts.append(effect_row('E1 R0_GUARD - R0 (R0-wrong scenarios only, others identical)', s,
                                        bench.effect(bench.matrix(mae, 'R0', 'R0', s, a_override={si: 'R0_GUARD' for si in wrong[s]})), study='E1'))
    for s in e2_seeds:
        for a, b in (('COISC_onset', 'R0'), ('L0', 'R0'), ('COISC_onset', 'L0')):
            contrasts.append(effect_row(f'E2 {a} - {b}', s, bench.effect(bench.matrix(mae, a, b, s)), study='E2'))

    for s in wrong:
        study = 'E1' if s in om.E1_SEEDS else 'E2'
        models = ('COISC_onset', 'COISC', 'L0')
        for subset_name, subset in (('R0_wrong', wrong[s]), ('R0_correct', right[s])):
            row = dict(study=study, seed=s, subset=subset_name, scenarios=len(subset))
            for v in models:
                row[f'{v}_correct'] = sum(cls(s, v, si) == 'correct' for si in subset)
                row[f'{v}_wrong_or_zero'] = sum(cls(s, v, si) != 'correct' for si in subset)
            for a in ('COISC_onset', 'L0') + (('COISC',) if study == 'E1' else ()):
                u = subset_units(bench, mae, a, 'R0', s, subset)
                row[f'dMAE_{a}_minus_R0'] = float(np.mean(u)) if u else None
            split.append(row)

    triggered = [s for s in e2_seeds if wrong[s]]
    assessable = len(triggered) >= 2
    pooled = lambda a, b, override=None: bench.effect(np.mean([bench.matrix(mae, a, b, s, override(s) if override else None) for s in triggered], axis=0))  # noqa: E731
    q = dict(Q1_incidence=dict(triggered_seeds=triggered, n_triggered=len(triggered), n_fresh=len(e2_seeds),
                               wrong_scenarios_per_seed={str(s): len(wrong[s]) for s in e2_seeds}))
    e2_wrong = [r for r in split if r['study'] == 'E2' and r['subset'] == 'R0_wrong' and r['scenarios']]
    e2_right = [r for r in split if r['study'] == 'E2' and r['subset'] == 'R0_correct' and r['scenarios']]
    if assessable:
        q['Q2_repair'] = dict(
            R0_wrong_pairs=sum(r['scenarios'] for r in e2_wrong),
            corrected={v: sum(r[f'{v}_correct'] for r in e2_wrong) for v in ('COISC_onset', 'COISC', 'L0')},
            dMAE_unit_bootstrap={f'{a} - R0': unit_bootstrap([u for s in triggered for u in subset_units(bench, mae, a, 'R0', s, wrong[s])])
                                 for a in ('COISC_onset', 'L0')})
        p, lo, hi = pooled('COISC_onset', 'R0')
        pl, lol, hil = pooled('COISC_onset', 'L0')
        more = sum(r['COISC_onset_correct'] > r['L0_correct'] for r in e2_wrong)
        q['Q4_policy'] = dict(policy_minus_always_R0=dict(point=p, ci_lower=lo, ci_upper=hi),
                              policy_minus_always_L0_same_trigger=dict(point=pl, ci_lower=lol, ci_upper=hil),
                              triggered_seeds_where_COISC_onset_corrects_more_than_L0=more)
        q['E2_conditional_benefit'] = bool(hi < 0 and more >= 2)
    else:
        q['Q2_repair'] = q['Q4_policy'] = 'NOT_ASSESSABLE (fewer than 2 triggered fresh seeds)'
        q['E2_conditional_benefit'] = 'NOT_ASSESSABLE'
    q['Q3_harm'] = dict(R0_correct_pairs=sum(r['scenarios'] for r in e2_right),
                        made_wrong_or_zero={v: sum(r[f'{v}_wrong_or_zero'] for r in e2_right) for v in ('COISC_onset', 'COISC', 'L0')},
                        dMAE_unit_bootstrap={f'{a} - R0': unit_bootstrap([u for s in e2_seeds for u in subset_units(bench, mae, a, 'R0', s, right[s])])
                                             for a in ('COISC_onset', 'L0')})
    if triggered:
        guard_units = lambda a: unit_bootstrap([u for s in triggered for u in subset_units(bench, mae, a, 'R0_GUARD', s, wrong[s])])  # noqa: E731
        q['Q5_guard'] = dict(
            R0_wrong_pairs=sum(len(wrong[s]) for s in triggered),
            COISC_onset_minus_R0_GUARD=guard_units('COISC_onset'), L0_minus_R0_GUARD=guard_units('L0'),
            R0_minus_R0_GUARD=guard_units('R0'),
            policy_minus_guard_always=dict(zip(('point', 'ci_lower', 'ci_upper'),
                                               pooled('COISC_onset', 'R0', lambda s: {si: 'R0_GUARD' for si in wrong[s]}))))
    else:
        q['Q5_guard'] = 'NOT_ASSESSABLE (no triggered fresh seed)'
    q['E1_seed_R0_GUARD'] = {str(s): dict(R0_wrong_scenarios=len(wrong[s]),
                                          **{f'{a} - R0_GUARD': unit_bootstrap(subset_units(bench, mae, a, 'R0_GUARD', s, wrong[s]))
                                             for a in ('COISC_onset', 'COISC', 'L0', 'R0')})
                             for s in om.E1_SEEDS if wrong[s]}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'ONSET_MIL_CONTRASTS.csv', contrasts)
    write_csv(out / 'ONSET_MIL_FIRST_ACTION_SPLIT.csv', split)
    write_csv(out / 'ONSET_MIL_SECONDARY.csv', [rows[k] for k in sorted(rows, key=lambda k: (k[3], k[1], k[2], k[0]))])
    summary = dict(created_utc=om.utc(), protocol_sha256=digest,
                   evidence_tier='POST-HOC exploratory (T3); descriptive only; does not revise any locked verdict',
                   runs_evaluated=len(mae), questions=q,
                   E1_MIL_COISC_onset_minus_COISC={str(r['seed']): r['point'] for r in contrasts if r['contrast'] == 'E1 COISC_onset - COISC'},
                   unit_bootstrap=dict(seed=UNIT_BOOTSTRAP_SEED, draws=UNIT_BOOTSTRAP_DRAWS,
                                       unit='(seed, scenario) family-equal mean over the 24 primary plants'))
    (out / 'ONSET_MIL_SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return summary, contrasts


def e1_verdict(holdout_csv, mil_point_vs_coisc):
    """Pre-declared E1 rule; WSR/PSM/accuracy point estimates from HOLDOUT_CONTRASTS.csv."""
    with Path(holdout_csv).open(encoding='utf-8-sig', newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if r['variant'] == 'COISC_onset' and int(r['seed']) in om.E1_SEEDS]
    get = lambda s, ref, m: next(r for r in rows if int(r['seed']) == s and r['reference'] == ref and r['metric'] == m)  # noqa: E731
    h1 = {s: float(get(s, 'R0', 'Tp_WSR_ONSET')['ratio']) <= 0.9 for s in om.E1_SEEDS}
    h2 = {s: float(get(s, 'R0', 'Tp_PSM_ONSET')['ratio']) >= 0.2 for s in om.E1_SEEDS}
    h5 = {s: float(get(s, 'COISC', 'Tp_MAE_120')['diff']) <= 0. for s in om.E1_SEEDS}
    mil = {s: mil_point_vs_coisc[str(s)] < 0. for s in om.E1_SEEDS}
    windows_met = all(h1.values()) and all(h2.values())
    if windows_met and sum(h5.values()) >= 2:
        verdict = 'E1_window_explanation_supported'
    elif sum(not v for v in h1.values()) >= 2 or (windows_met and sum(h5.values()) < 2 and sum(mil.values()) < 2):
        verdict = 'E1_window_explanation_not_supported'
    else:
        verdict = 'E1_mixed (neither declared rule met)'
    return dict(H1_WSR_le_0p9_R0=h1, H2_PSM_ge_0p2_R0=h2, H5_Tp120_not_worse_than_COISC=h5,
                MIL_COISC_onset_better_than_COISC=mil, verdict=verdict)


def selftest():
    """E1: COISC_onset := COISC, R0_GUARD := R0. E2: every fresh seed := relabelled seed-20260923 runs (COISC as COISC_onset)."""
    plants, _ = om.plants_and_formal_index()
    base = existing_runs(plants)
    rows = dict(base)
    for (v, pi, si, seed), r in base.items():
        if v == 'COISC':
            rows[('COISC_onset', pi, si, seed)] = {**r, 'variant': 'COISC_onset'}
        if v == 'R0':
            rows[('R0_GUARD', pi, si, seed)] = {**r, 'variant': 'R0_GUARD'}
    donor = 20260923
    e2 = om.E2_SEEDS
    for (v, pi, si, seed), r in base.items():
        if seed == donor:
            for s in e2:
                rows[('COISC_onset' if v == 'COISC' else v, pi, si, s)] = {**r, 'seed': s}
                if v == 'COISC':
                    rows[('COISC', pi, si, s)] = {**r, 'seed': s}
                if v == 'R0':
                    rows[('R0_GUARD', pi, si, s)] = {**r, 'seed': s, 'variant': 'R0_GUARD'}
    fa = classes_from_rows(rows)
    summary, contrasts = compute(rows, fa, SELFTEST_DIR, 'SELFTEST')
    check = {r['seed']: r for r in om.read(ANALYSIS_GATE_REPORT)['R6_check_formal']['rows'] if r['contrast'].startswith('COISC-R0')}
    pt = {(r['contrast'], r['seed']): r['point'] for r in contrasts}
    tests = dict(
        e1_onset_minus_coisc_zero=all(pt[('E1 COISC_onset - COISC', s)] == 0. for s in om.E1_SEEDS),
        e1_onset_minus_r0_equals_formal=all(pt[('E1 COISC_onset - R0', s)] == check[s]['reported'][0] for s in om.E1_SEEDS),
        e2_equals_donor=all(pt[('E2 COISC_onset - R0', s)] == check[donor]['reported'][0] for s in e2),
        q1_all_triggered=summary['questions']['Q1_incidence']['n_triggered'] == (10 if any(fa[(donor, 'R0', si)] == 'wrong' for si in range(12)) else 0),
        q5_guard_zero=summary['questions']['Q5_guard']['R0_minus_R0_GUARD']['point'] == 0.,
        q4_policy_equals_donor=abs(summary['questions']['Q4_policy']['policy_minus_always_R0']['point'] - check[donor]['reported'][0]) < 1e-12,
        q5_policy_equals_q4=summary['questions']['Q5_guard']['policy_minus_guard_always']['point'] == summary['questions']['Q4_policy']['policy_minus_always_R0']['point'],
        no_identifier_columns=all('source' not in k and 'scenario_id' not in k for r in rows.values() for k in r))
    result = dict(created_utc=om.utc(), tests=tests, pass_=all(tests.values()))
    report = om.read(ANALYSIS_GATE_REPORT)
    report.update(R6_selftest=result, code_sha256={'onset_analyze.py': om.sha(Path(__file__))})
    report['all_pass'] = all(report[k]['pass_'] for k in ('R6_check_formal', 'R6_selftest'))
    ANALYSIS_GATE_REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(result))
    return result['pass_']


def analyze(execution_dir):
    execution = Path(execution_dir).resolve()
    protocol, digest = om.frozen_protocol()
    audit = om.read(execution / 'MIL_EXECUTION_AUDIT.json')
    assert audit['protocol_sha256'] == digest and audit['status'] == 'RAW_RUNS_COMPLETE'
    plants, _ = om.plants_and_formal_index()
    rows = existing_runs(plants)
    fa_file = om.read(execution / 'FIRST_ACTION_CHECK.json')
    tasks = om.read(execution / 'TASKS.json')
    assert fa_file['protocol_sha256'] == digest and tasks == om.task_list(fa_file['rows'])
    fa = {(r['seed'], r['variant'], r['scenario_index']): r['first_action'] for r in fa_file['rows']}
    identity, fired = [], []
    for i, t in enumerate(tasks):
        meta, data = load_run(execution / 'raw_runs' / f'{i:05d}.npz')
        assert meta['controller_variant'] == t['variant'] and meta['plant_index'] == t['plant_index'] and meta['controller_seed'] == t['seed']
        if t['variant'] == 'R0_GUARD':
            fired.append(meta['guard_fired'])
        if t['variant'] == 'R0_GUARD_ID':
            identity.append(dict(i=i, task=t, data=data, fired=meta['guard_fired']))
            continue
        row = metric_row(meta, data, t['variant'], t['plant_index'], t['scenario_index'], t['seed'], meta['family'])
        if row is not None:
            rows[(t['variant'], t['plant_index'], t['scenario_index'], t['seed'])] = row
    index = {(t['variant'], t['plant_index'], t['scenario_index'], t['seed']): i for i, t in enumerate(tasks)}
    _, formal_index = om.plants_and_formal_index()
    r7 = []
    for item in identity:
        t = item['task']
        ref = (om.FORMAL_EXECUTION / 'raw_runs' / f'{formal_index[(t["plant_index"], t["scenario_index"], t["seed"], "R0")]:04d}.npz'
               if t['seed'] in om.E1_SEEDS else execution / 'raw_runs' / f'{index[("R0", t["plant_index"], t["scenario_index"], t["seed"])]:05d}.npz')
        _, ref_data = load_run(ref)
        r7.append(bool(not item['fired'] and np.array_equal(item['data']['applied_UH'], ref_data['applied_UH'])
                       and np.array_equal(item['data']['plant_Tp'], ref_data['plant_Tp'])))
    # the plant-free first decision must equal every run's first applied command for the same checkpoint
    fa_u = {(r['seed'], r['variant'], r['scenario_index']): r['first_u'] for r in fa_file['rows']}
    first_ok = all(fa_u[(k[3], k[0], k[2])] == r['first_u'] for k, r in rows.items() if k[0] != 'R0_GUARD' and (k[3], k[0], k[2]) in fa_u)
    gates = dict(R7_guard_identity=dict(runs=len(r7), pass_=all(r7)), guard_fired_on_all_triggered_runs=all(fired),
                 first_action_check_consistent_with_runs=first_ok)
    summary, contrasts = compute(rows, fa, execution, digest)
    summary['execution_gates'] = gates
    holdout = om.OUT / 'HOLDOUT_CONTRASTS.csv'
    summary['E1_rule'] = e1_verdict(holdout, summary['E1_MIL_COISC_onset_minus_COISC']) if holdout.is_file() else 'holdout run missing'
    (execution / 'ONSET_MIL_SUMMARY.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else ''
    if command == 'check-formal':
        raise SystemExit(0 if check_formal() else 1)
    elif command == 'selftest':
        raise SystemExit(0 if selftest() else 1)
    elif command == 'analyze' and len(sys.argv) == 3:
        analyze(sys.argv[2])
    else:
        raise SystemExit(__doc__)
