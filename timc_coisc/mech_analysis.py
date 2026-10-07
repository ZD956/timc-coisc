"""POSTHOC_MECHANISM_STEADY_MAP_V1: inference-only reanalysis (M1 decomposition, M2 static map, M3 association)."""
from __future__ import annotations

import csv
import hashlib
import json
import time

import numpy as np
import torch
from scipy.stats import rankdata

from _paths import PKG, PROTOCOLS, OUTPUT_ROOT, VERIFY_HASHES  # noqa: E402
import run_formal_control_v3a as rf  # noqa: E402
from analyze_formal_control_v3 import source_strata  # noqa: E402
from paper_data import sha, write_csv  # noqa: E402

OUT = OUTPUT_ROOT/'posthoc_mechanism'
MECH_PROTO = PROTOCOLS/'posthoc_mechanism'
CONTROL = PROTOCOLS
FORMAL = OUTPUT_ROOT/'control'/'control_v3a_20260926T153411234230Z_0b9ad32c21'
POSTHOC = PROTOCOLS
POSTHOC_EXEC = OUTPUT_ROOT/'posthoc_mil'/'execution_20261001T050813Z_a3bd1ebd55'
SEEDS = rf.SEEDS
SEED_LABEL = {20260922: 'Seed1', 20260923: 'Seed2', 20260924: 'Seed3'}
CTRL = ('R0', 'COISC', 'L0')
GRID = np.arange(0., 101., 10.)
LOG = []


def log(message):
    line = f'[{time.strftime("%H:%M:%S")}] {message}'
    print(line, flush=True)
    LOG.append(line)


def check_freeze():
    if not VERIFY_HASHES:
        return
    lines = dict(l.split('  ', 1)[::-1] for l in (MECH_PROTO/'POSTHOC_MECHANISM_PROTOCOL.sha256').read_text(encoding='utf-8').split('\n') if l.strip())
    for path in (MECH_PROTO/'POSTHOC_MECHANISM_PROTOCOL.json', PKG/'mech_analysis.py'):
        assert sha(path) == lines[path.name], f'freeze hash mismatch: {path.name}'
    log('freeze hashes verified')


# ---------------- data loading ----------------
def load_traces():
    plants, tasks = rf.schedule()
    scenarios = rf.read(CONTROL/'CONTROL_SCENARIO_LOCK_V2.json')['scenarios']
    runs = {}
    for index, task in enumerate(tasks):
        if task['plant_index'] >= 24:
            continue
        with np.load(FORMAL/'raw_runs'/f'{index:04d}.npz', allow_pickle=False) as z:
            meta = json.loads(str(z['metadata']))
            data = {k: z[k] for k in ('plant_Tp', 'applied_UH', 'internal_predicted_Tp', 'optimal_sequence')}
        assert meta['plant'] == plants[task['plant_index']]['plant_name']
        assert meta['scenario_id'] == scenarios[task['scenario_index']]['scenario_id']
        assert meta['controller_seed'] == task['seed'] and meta['controller_variant'] == task['variant']
        runs[(task['plant_index'], task['scenario_index'], task['seed'], task['variant'])] = (meta, data)
    posthoc_tasks = json.loads((POSTHOC_EXEC/'TASKS.json').read_text(encoding='utf-8'))
    for index, task in enumerate(posthoc_tasks):
        if task['variant'] != 'L0':
            continue
        with np.load(POSTHOC_EXEC/'raw_runs'/f'{index:04d}.npz', allow_pickle=False) as z:
            meta = json.loads(str(z['metadata']))
            data = {k: z[k] for k in ('plant_Tp', 'applied_UH', 'internal_predicted_Tp')}
        assert meta['plant'] == plants[task['plant_index']]['plant_name']
        assert meta['scenario_id'] == scenarios[task['scenario_index']]['scenario_id']
        assert meta['controller_seed'] == task['seed'] and meta['controller_variant'] == 'L0'
        runs[(task['plant_index'], task['scenario_index'], task['seed'], 'L0')] = (meta, data)
    assert len(runs) == 24*12*(3*3+1), len(runs)
    log(f'loaded {len(runs)} primary traces (R0/COISC/L0 x 3 seeds + ORACLE)')
    return plants[:24], scenarios, runs


def run_metrics(meta, data):
    y = np.asarray(data['plant_Tp'], dtype=np.float64)
    u = np.asarray(data['applied_UH'], dtype=np.float64)
    pred = np.asarray(data['internal_predicted_Tp'], dtype=np.float64)
    assert y.shape == (60,) and u.shape == (60,) and pred.shape == (60, 12)
    e = y - 37.
    late = slice(54, 60)
    return dict(
        MAE_all=float(np.mean(np.abs(e))),
        MAE_transient=float(np.mean(np.abs(e[:30]))),
        MAE_late=float(np.mean(np.abs(e[30:]))),
        ess_signed=float(np.mean(e[late])),
        ess_abs=float(abs(np.mean(e[late]))),
        u_late=float(np.mean(u[late])),
        late_saturation=float(np.mean((u[late] <= 0.) | (u[late] >= 100.))),
        first_action_sign=int(np.sign(round(u[0]-float(meta['previous_uh_percent']), 9))),
        one_step_bias_late=float(np.nanmean(pred[late, 0]-y[late])),
        perceived_error_late=float(np.nanmean(pred[late, 11]-37.)),
        prediction_nan_late=int(np.isnan(pred[late]).any(axis=1).sum()))


# ---------------- gates ----------------
def gate_reproduction(plants, scenarios, runs, metrics):
    expected = {}
    with (FORMAL/'CONTROL_CHECKPOINT_RESULTS.csv').open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            seed = int(r['controller_seed']) if r['controller_seed'] not in ('', 'None') else None
            expected[(r['plant'], r['scenario_id'], seed, r['controller_variant'])] = float(r['MAE_track'])
    with (POSTHOC_EXEC/'POSTHOC_MIL_SECONDARY.csv').open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            if r['variant'] == 'L0':
                expected[(r['plant'], r['scenario_id'], int(r['seed']), 'L0')] = float(r['MAE_track'])
    worst = {'formal': 0., 'L0': 0.}
    for key, m in metrics.items():
        pi, si, seed, variant = key
        ref = expected[(plants[pi]['plant_name'], scenarios[si]['scenario_id'], seed, variant)]
        group = 'L0' if variant == 'L0' else 'formal'
        worst[group] = max(worst[group], abs(m['MAE_all']-ref))
    oracle_bias = max(abs(m['one_step_bias_late']) for k, m in metrics.items() if k[3] == 'ORACLE')
    zero_gap_bias, unexplained = 0., 0
    for key, (meta, data) in runs.items():
        if key[3] != 'ORACLE':
            continue
        bias = np.abs(data['internal_predicted_Tp'][:, 0]-data['plant_Tp'])
        gap = np.abs(data['optimal_sequence'][:, 0]-data['applied_UH'])
        finite = np.isfinite(bias)
        zero = finite & (gap < 1e-9)
        if zero.any():
            zero_gap_bias = max(zero_gap_bias, float(bias[zero].max()))
        unexplained += int((finite & (bias > 1e-6) & (gap < 1e-9)).sum())
    nan_late = sum(m['prediction_nan_late'] for m in metrics.values())
    log(f'gate MAE reproduction: max|diff| formal={worst["formal"]:.3e} L0={worst["L0"]:.3e}')
    log(f'gate (d, AM1) ORACLE one-step bias: zero-quantization-gap max|.|={zero_gap_bias:.3e}; '
        f'unexplained steps={unexplained}; late-mean max|.| incl. quantization={oracle_bias:.3e}; late NaN predictions={nan_late}')
    ok = worst['formal'] <= 1e-9 and worst['L0'] <= 1e-9 and zero_gap_bias <= 1e-6 and unexplained == 0
    return ok, dict(max_abs_MAE_diff_formal=worst['formal'], max_abs_MAE_diff_L0=worst['L0'],
                    oracle_zero_gap_bias_max_abs=zero_gap_bias, oracle_unexplained_bias_steps=unexplained,
                    oracle_late_bias_incl_quantization_max_abs=oracle_bias, late_nan_predictions=nan_late)


# ---------------- M2 open-loop static map ----------------
def open_loop(model, histories, backgrounds):
    out = np.empty((len(GRID), len(histories), 2))
    with torch.no_grad():
        for li, level in enumerate(GRID):
            history = torch.from_numpy(histories.copy()).float()
            future = torch.empty((len(history), 6), dtype=torch.float32)
            future[:, [0, 2, 3, 4, 5]] = torch.from_numpy(backgrounds.copy()).float()
            future[:, 1] = float(level)*5.0
            for step in range(60):
                output = model.forward_one_step(history, future)
                if step == 11:
                    out[li, :, 0] = output[:, 6].numpy()
                if step == 59:
                    out[li, :, 1] = output[:, 6].numpy()
                next_frame = torch.cat((output[:, :6], future), dim=1)
                history = torch.cat((history[:, 1:], next_frame[:, None]), dim=1)
    assert np.isfinite(out).all()
    return out  # [level, scenario, (T120, T600)]


def u_star(curve):
    """First upward crossing of 37 C on the 11-level grid (see protocol u_star_edge_cases)."""
    for i in range(len(GRID)-1):
        a, b = curve[i], curve[i+1]
        if a < 37. <= b:
            return float(GRID[i]+(37.-a)/(b-a)*(GRID[i+1]-GRID[i])), 'upward'
    if curve.min() >= 37.:
        return 0., 'all_above'
    if curve.max() < 37.:
        return 100., 'all_below'
    for i in range(len(GRID)-1):
        a, b = curve[i], curve[i+1]
        if (a-37.)*(b-37.) <= 0. and a != b:
            return float(GRID[i]+(37.-a)/(b-a)*(GRID[i+1]-GRID[i])), 'fallback_any_crossing'
    raise AssertionError('unreachable u_star case')


def static_maps(plants, scenarios):
    lock = rf.read(CONTROL/'CONTROL_SCENARIO_LOCK_V2.json')
    _, histories = rf.load_scenarios(lock)
    histories = np.stack(histories)
    backgrounds = np.array([s['background_raw_aux'] for s in scenarios], dtype=np.float32)
    v3 = rf.read(CONTROL/'CONTROL_VALIDATION_PROTOCOL_V3.json')
    l0 = rf.read(POSTHOC/'POSTHOC_MIL_PROTOCOL.json')['controller_models']['L0_bindings']
    maps = {}
    for seed in SEEDS:
        for variant in CTRL:
            binding = l0[str(seed)] if variant == 'L0' else v3['controller_models']['checkpoint_bindings'][f'{seed}:{variant}']
            maps[(seed, variant)] = open_loop(rf.load_model(binding, 'B4'), histories, backgrounds)
    log('controller static maps computed (9 checkpoints, sha256-checked)')
    for pi, plant in enumerate(plants):
        model = rf.load_model({'path': plant['checkpoint_path'], 'sha256': plant['sha256']}, plant['model_id'])
        maps[('plant', pi)] = open_loop(model, histories, backgrounds)
    log('plant static maps computed (24 checkpoints, sha256-checked)')
    return maps


def gate_plant_endpoints(plants, scenarios, maps):
    want = {}
    with (CONTROL/'CONTROL_PLANT_REACHABILITY_AUDIT.csv').open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            want[(r['plant_checkpoint'], r['scenario_id'])] = [float(r[f'Tp_600s_UH_{k}']) for k in (0, 50, 100)]
    worst = 0.
    for pi, plant in enumerate(plants):
        for si, s in enumerate(scenarios):
            got = maps[('plant', pi)][[0, 5, 10], si, 1]
            worst = max(worst, float(np.max(np.abs(got-np.array(want[(plant['plant_name'], s['scenario_id'])])))))
    log(f'gate plant endpoint reproduction: max|diff|={worst:.3e} C (tolerance 1e-4)')
    return worst <= 1e-4, worst


# ---------------- bootstrap helpers ----------------
def boot_mean(values, scen, counts):
    values = np.asarray(values, dtype=np.float64)
    sums = np.bincount(scen, weights=values, minlength=12)
    ns = np.bincount(scen, minlength=12).astype(np.float64)
    den = counts@ns
    keep = den > 0
    reps = (counts[keep]@sums)/den[keep]
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi), int((~keep).sum())


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def boot_spearman(x, y, scen, draws):
    x, y, scen = map(np.asarray, (x, y, scen))
    members = [np.flatnonzero(scen == s) for s in range(12)]
    point = spearman(x, y)
    reps, dropped = [], 0
    for draw in draws:
        idx = np.concatenate([members[s] for s in draw])
        r = spearman(x[idx], y[idx]) if len(idx) > 2 else np.nan
        if np.isnan(r):
            dropped += 1
        else:
            reps.append(r)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return point, float(lo), float(hi), dropped


# ---------------- main ----------------
def main():
    began = time.perf_counter()
    check_freeze()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    plants, scenarios, runs = load_traces()
    metrics = {key: run_metrics(*value) for key, value in runs.items()}
    ok, gate_info = gate_reproduction(plants, scenarios, runs, metrics)
    if not ok:
        log('STOP: trace reproduction gate failed')
        (OUT/'MECH_RUN_LOG.txt').write_text('\n'.join(LOG)+'\n', encoding='utf-8')
        raise SystemExit(1)
    maps = static_maps(plants, scenarios)
    ok, endpoint_diff = gate_plant_endpoints(plants, scenarios, maps)
    gate_info['plant_endpoint_max_abs_diff_C'] = endpoint_diff
    if not ok:
        log('STOP: plant endpoint reproduction gate failed')
        (OUT/'MECH_RUN_LOG.txt').write_text('\n'.join(LOG)+'\n', encoding='utf-8')
        raise SystemExit(1)

    strata = source_strata(CONTROL/'CONTROL_PLANT_REACHABILITY_AUDIT.csv')
    label = {si: f'S{si+1:02d}' for si in range(12)}
    pair_stratum = {(pi, si): strata[(p['plant_name'], s['scenario_id'])] for pi, p in enumerate(plants) for si, s in enumerate(scenarios)}
    pair_side = {(pi, si): ('start_below' if float(s['initial_Tp_C']) < 37. else 'start_above') for pi in range(24) for si, s in enumerate(scenarios)}
    counts = {}
    for v in pair_stratum.values():
        counts[v] = counts.get(v, 0)+1
    log(f'strata counts {counts}')
    assert counts == {'BRACKETED_AT_600S': 150, 'BOTH_ABOVE_37_AT_600S': 121, 'BOTH_BELOW_37_AT_600S': 17}

    # ---- M2 per-model quantities ----
    ustar, urule = {}, {}
    for key, grid in maps.items():
        for si in range(12):
            ustar[(key, si)], urule[(key, si)] = u_star(grid[:, si, 1])
    nonmono = {key: int(sum(np.any(np.diff(grid[:, si, 1]) < -0.01) for si in range(12))) for key, grid in maps.items()}
    fallback = {key: int(sum(urule[(key, si)] == 'fallback_any_crossing' for si in range(12))) for key in maps}

    def plant_T600(pi, si, u):
        return float(np.interp(u, GRID, maps[('plant', pi)][:, si, 1]))

    run_rows, pair_values = [], {}
    for (pi, si, seed, variant), m in metrics.items():
        ctrl_key = ('plant', pi) if variant == 'ORACLE' else (seed, variant)
        us = ustar[(ctrl_key, si)]
        pso = plant_T600(pi, si, us)-37.
        ctrl_curve = maps[ctrl_key][:, si, 1]
        mo = float(np.interp(m['u_late'], GRID, ctrl_curve))-plant_T600(pi, si, m['u_late'])
        extra = dict(u_star=us, u_star_rule=urule[(ctrl_key, si)], predicted_static_offset=pso, map_offset_at_u_late=mo)
        pair_values[(pi, si, seed, variant)] = {**m, **extra}
        run_rows.append(dict(plant=plants[pi]['plant_name'], family=plants[pi]['family'], scenario=label[si],
            seed='NA' if seed is None else SEED_LABEL[seed], variant=variant,
            stratum=pair_stratum[(pi, si)], initial_side=pair_side[(pi, si)], **m, **extra))
    run_rows.sort(key=lambda r: (r['variant'], r['seed'], r['plant'], r['scenario']))
    write_csv(OUT/'MECH_RUN_METRICS.csv', run_rows)

    grid_rows = []
    for key, grid in maps.items():
        model = plants[key[1]]['plant_name'] if key[0] == 'plant' else f'{key[1]}_{SEED_LABEL[key[0]]}'
        for si in range(12):
            for li, u in enumerate(GRID):
                grid_rows.append(dict(model=model, scenario=label[si], u_h_percent=float(u),
                                      T_p_120s=float(grid[li, si, 0]), T_p_600s=float(grid[li, si, 1])))
    write_csv(OUT/'MECH_STATIC_MAP_GRID.csv', grid_rows)

    draws = np.load(FORMAL/'CONTROL_BOOTSTRAP_DRAWS.npz')['source_draws'].astype(np.int64)
    assert draws.shape == (20000, 12)
    draw_counts = np.stack([np.bincount(d, minlength=12) for d in draws]).astype(np.float64)

    subsets = {
        'ALL': lambda k: True,
        'BRACKETED': lambda k: pair_stratum[k] == 'BRACKETED_AT_600S',
        'BRACKETED_start_below': lambda k: pair_stratum[k] == 'BRACKETED_AT_600S' and pair_side[k] == 'start_below',
        'BRACKETED_start_above': lambda k: pair_stratum[k] == 'BRACKETED_AT_600S' and pair_side[k] == 'start_above',
        'BOTH_ABOVE': lambda k: pair_stratum[k] == 'BOTH_ABOVE_37_AT_600S',
        'BOTH_BELOW': lambda k: pair_stratum[k] == 'BOTH_BELOW_37_AT_600S'}
    m1_metrics = ('MAE_all', 'MAE_transient', 'MAE_late', 'ess_signed', 'ess_abs', 'u_late', 'late_saturation',
                  'first_action_sign', 'one_step_bias_late', 'perceived_error_late', 'predicted_static_offset')

    # ---- M1 summary ----
    m1_rows = []
    lookup = {}
    for sub, pred in subsets.items():
        pairs = [(pi, si) for pi in range(24) for si in range(12) if pred((pi, si))]
        scen = np.array([si for _, si in pairs])
        for seed in SEEDS:
            def val(variant, metric):
                s = None if variant == 'ORACLE' else seed
                return np.array([pair_values[(pi, si, s, variant)][metric] for pi, si in pairs])
            comparisons = [(v, None) for v in CTRL+('ORACLE',)] + [('COISC', 'R0'), ('COISC', 'L0'), ('L0', 'R0'),
                           ('R0', 'ORACLE'), ('COISC', 'ORACLE'), ('L0', 'ORACLE')]
            for a, b in comparisons:
                for metric in m1_metrics:
                    values = val(a, metric) if b is None else val(a, metric)-val(b, metric)
                    point, lo, hi, dropped = boot_mean(values, scen, draw_counts)
                    name = a if b is None else f'{a}-{b}'
                    lookup[(sub, seed, name, metric)] = (point, lo, hi)
                    m1_rows.append(dict(subset=sub, n_pairs=len(pairs), n_scenarios=len(set(scen.tolist())),
                        seed=SEED_LABEL[seed], quantity=name, metric=metric, point=point, ci_lower=lo, ci_upper=hi,
                        empty_replicates_dropped=dropped))
    write_csv(OUT/'MECH_M1_SUMMARY.csv', m1_rows)
    log(f'M1 written: {len(m1_rows)} rows')

    # ---- M2 gain summary ----
    def gains(key):
        g = maps[key]
        return g[10, :, 0]-g[0, :, 0], g[10, :, 1]-g[0, :, 1]
    m2_rows, gain_med = [], {}
    q = lambda a: (float(np.median(a)), float(np.percentile(a, 25)), float(np.percentile(a, 75)))
    for seed in SEEDS:
        for variant in CTRL:
            g120, g600 = gains((seed, variant))
            for name, arr in (('G120', g120), ('G600', g600)):
                med, q1, q3 = q(arr)
                m2_rows.append(dict(seed=SEED_LABEL[seed], model=variant, quantity=name, median=med, q25=q1, q75=q3,
                    nonmonotone_scenarios=nonmono[(seed, variant)], u_star_fallback_scenarios=fallback[(seed, variant)]))
        for a, b in (('COISC', 'R0'), ('L0', 'R0'), ('COISC', 'L0')):
            for gi, name in ((0, 'G120'), (1, 'G600')):
                ratio = gains((seed, a))[gi]/gains((seed, b))[gi]
                med, q1, q3 = q(ratio)
                gain_med[(seed, f'{a}/{b}', name)] = med
                m2_rows.append(dict(seed=SEED_LABEL[seed], model=f'{a}/{b}', quantity=f'{name}_ratio', median=med, q25=q1, q75=q3))
    for family in ('F1', 'F2', 'F3', 'F4'):
        idx = [pi for pi, p in enumerate(plants) if p['family'] == family]
        for gi, name in ((0, 'G120'), (1, 'G600')):
            arr = np.concatenate([gains(('plant', pi))[gi] for pi in idx])
            med, q1, q3 = q(arr)
            m2_rows.append(dict(seed='NA', model=f'plant_family_{family}', quantity=name, median=med, q25=q1, q75=q3,
                nonmonotone_scenarios=sum(nonmono[('plant', pi)] for pi in idx),
                u_star_fallback_scenarios=sum(fallback[('plant', pi)] for pi in idx)))
    arr120 = np.concatenate([gains(('plant', pi))[0] for pi in range(24)])
    arr600 = np.concatenate([gains(('plant', pi))[1] for pi in range(24)])
    for name, arr in (('G120', arr120), ('G600', arr600)):
        med, q1, q3 = q(arr)
        m2_rows.append(dict(seed='NA', model='plant_all_24', quantity=name, median=med, q25=q1, q75=q3))
    write_csv(OUT/'MECH_M2_GAIN_SUMMARY.csv', m2_rows)
    log(f'M2 written: {len(m2_rows)} rows')

    # ---- M3 association ----
    m3_rows, assoc = [], {}
    for sub in ('BRACKETED', 'ALL', 'BRACKETED_start_below'):
        pairs = [(pi, si) for pi in range(24) for si in range(12) if subsets[sub]((pi, si))]
        scen = [si for _, si in pairs]
        for seed in SEEDS:
            for variant in CTRL:
                y = [pair_values[(pi, si, seed, variant)]['ess_signed'] for pi, si in pairs]
                for xname, kind in (('predicted_static_offset', 'primary'), ('map_offset_at_u_late', 'secondary')):
                    x = [pair_values[(pi, si, seed, variant)][xname] for pi, si in pairs]
                    point, lo, hi, dropped = boot_spearman(x, y, scen, draws)
                    assoc[(sub, seed, variant, xname)] = (point, lo, hi)
                    m3_rows.append(dict(kind=kind, subset=sub, n_pairs=len(pairs), seed=SEED_LABEL[seed], comparison=variant,
                        x=xname, y='ess_signed', spearman_rho=point, ci_lower=lo, ci_upper=hi, replicates_dropped=dropped))
            for a, b in (('COISC', 'R0'), ('COISC', 'L0')):
                x = [pair_values[(pi, si, seed, a)]['predicted_static_offset']-pair_values[(pi, si, seed, b)]['predicted_static_offset'] for pi, si in pairs]
                y = [pair_values[(pi, si, seed, a)]['ess_signed']-pair_values[(pi, si, seed, b)]['ess_signed'] for pi, si in pairs]
                point, lo, hi, dropped = boot_spearman(x, y, scen, draws)
                assoc[(sub, seed, f'{a}-{b}', 'contrast')] = (point, lo, hi)
                m3_rows.append(dict(kind='contrast', subset=sub, n_pairs=len(pairs), seed=SEED_LABEL[seed], comparison=f'{a}-{b}',
                    x='delta_predicted_static_offset', y='delta_ess_signed', spearman_rho=point, ci_lower=lo, ci_upper=hi,
                    replicates_dropped=dropped))
        log(f'M3 subset {sub} done')
    write_csv(OUT/'MECH_M3_ASSOCIATION.csv', m3_rows)

    # ---- interpretation rules (as declared in the frozen protocol) ----
    hm = [assoc[('BRACKETED', s, v, 'predicted_static_offset')] for s in SEEDS for v in ('R0', 'COISC')]
    if all(r >= 0.5 and lo > 0 for r, lo, _ in hm):
        h_map = 'H_map_consistent'
    elif any(r > 0 and lo > 0 for r, lo, _ in hm):
        h_map = 'H_map_partial'
    else:
        h_map = 'H_map_not_supported'
    sb = 'BRACKETED_start_below'
    trade = [lookup[(sb, s, 'COISC-R0', 'MAE_transient')][0] < 0 and lookup[(sb, s, 'COISC-R0', 'ess_abs')][0] > 0 for s in SEEDS]
    g120 = [gain_med[(s, 'COISC/R0', 'G120')] < 1 for s in SEEDS]
    g600 = [gain_med[(s, 'COISC/R0', 'G600')] < 1 for s in SEEDS]
    reg = {}
    for metric in ('MAE_transient', 'ess_abs'):
        reg[metric] = [np.sign(lookup[(sb, s, 'COISC-L0', metric)][0]) == np.sign(lookup[(sb, s, 'COISC-R0', metric)][0]) for s in SEEDS]
    for name in ('G120', 'G600'):
        reg[name] = [np.sign(gain_med[(s, 'COISC/L0', name)]-1) == np.sign(gain_med[(s, 'COISC/R0', name)]-1) for s in SEEDS]
    seed2 = {SEED_LABEL[s]: dict(
        R0_predicted_static_offset_bracketed=lookup[('BRACKETED', s, 'R0', 'predicted_static_offset')],
        R0_predicted_static_offset_start_below=lookup[(sb, s, 'R0', 'predicted_static_offset')],
        COISC_predicted_static_offset_start_below=lookup[(sb, s, 'COISC', 'predicted_static_offset')],
        R0_ess_signed_start_below=lookup[(sb, s, 'R0', 'ess_signed')],
        COISC_ess_signed_start_below=lookup[(sb, s, 'COISC', 'ess_signed')],
        G600_R0_median=float(np.median(gains((s, 'R0'))[1])), G600_COISC_median=float(np.median(gains((s, 'COISC'))[1])))
        for s in SEEDS}
    summary = dict(protocol='POSTHOC_MECHANISM_STEADY_MAP_V1', evidence_tier='post hoc, descriptive',
        gates=gate_info, strata_counts=counts,
        H_map=dict(verdict=h_map, bracketed_rho={f'{SEED_LABEL[s]}:{v}': assoc[('BRACKETED', s, v, 'predicted_static_offset')] for s in SEEDS for v in CTRL}),
        H_tradeoff=dict(verdict='H_tradeoff_consistent' if sum(trade) >= 2 else 'H_tradeoff_not_consistent',
            per_seed={SEED_LABEL[s]: dict(dMAE_transient=lookup[(sb, s, 'COISC-R0', 'MAE_transient')],
                d_ess_abs=lookup[(sb, s, 'COISC-R0', 'ess_abs')], meets=trade[i]) for i, s in enumerate(SEEDS)}),
        H_gain=dict(G120_verdict='H_gain_consistent' if all(g120) else 'H_gain_not_consistent',
            G600_verdict='H_gain_consistent' if all(g600) else 'H_gain_not_consistent',
            median_ratios={f'{SEED_LABEL[s]}:{r}:{n}': gain_med[(s, r, n)] for s in SEEDS for r in ('COISC/R0', 'L0/R0', 'COISC/L0') for n in ('G120', 'G600')}),
        H_reg=dict(verdict='H_reg_consistent' if all(all(v) for v in reg.values()) else 'H_reg_not_consistent',
            sign_agreement_per_metric={k: [bool(x) for x in v] for k, v in reg.items()}),
        contrast_association_bracketed={f'{SEED_LABEL[s]}:{c}': assoc[('BRACKETED', s, c, 'contrast')] for s in SEEDS for c in ('COISC-R0', 'COISC-L0')},
        seed_comparison=seed2,
        u_star_fallback_total=int(sum(fallback.values())), nonmonotone_curves_total=int(sum(nonmono.values())),
        runtime_s=time.perf_counter()-began)
    (OUT/'MECH_RESULTS_SUMMARY.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    log(f'verdicts: {h_map}; tradeoff {trade}; G120 {g120}; G600 {g600}; reg {summary["H_reg"]["verdict"]}')
    log(f'done in {time.perf_counter()-began:.1f} s')
    (OUT/'MECH_RUN_LOG.txt').write_text('\n'.join(LOG)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
