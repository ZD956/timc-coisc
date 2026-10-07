"""Descriptive analysis of one immutable delivered23 execution; never runs a model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch

sys.dont_write_bytecode = True
from frozen_evaluate import weighted_psm
from _paths import PROTOCOLS, DATA_ROOT, OUTPUT_ROOT, VERIFY_HASHES, load_checkpoint, require_private, public_source_id

HERE = OUTPUT_ROOT / 'delivered23_posthoc'

POINTS = ('T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'Tp')
HORIZONS = (10, 30, 60, 90, 120)
SEEDS = (20260922, 20260923, 20260924)
HEAD_KEYS = ('head.0.weight', 'head.0.bias', 'head.2.weight', 'head.2.bias')
LABEL = 'DESCRIPTIVE / EXPLORATORY / POST-HOC'


def sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            hasher.update(block)
    return hasher.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def relative(a, b):
    return None if a == 0 else 100 * (b / a - 1)


def ci_text(row):
    value = row['absolute_ci']
    return json.loads(value) if isinstance(value, str) and value else None


def write_csv(name, rows):
    require(rows, f'empty artifact {name}')
    HERE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(HERE / name, index=False, encoding='utf-8-sig')


def cohort_ids(membership):
    all_ids = [row['source_id'] for row in membership]
    return {'all23': all_ids,
            'frozen_external': [row['source_id'] for row in membership if row['role'] == 'frozen_external'],
            'frozen_final': [row['source_id'] for row in membership if row['role'] == 'frozen_final']}


def paired_bootstrap(boot, cohort, seed, output, horizon, endpoint):
    rows = boot[(boot.seed == seed) & (boot.cohort == cohort) & (boot.output == output)
                & (boot.horizon_s.fillna(-1).astype(int) == horizon) & (boot.endpoint == endpoint)
                & (boot.status == 'OK') & boot.absolute_change.notna()]
    require(len(rows) == 1, f'missing unique bootstrap {cohort}/{seed}/{output}/{horizon}/{endpoint}')
    return rows.iloc[0]


def paired_sensitivity_bootstrap(boot, seed, endpoint):
    rows = boot[(boot.seed == seed) & (boot.cohort == 'all23') & (boot.output == 'Tp')
                & (boot.endpoint == endpoint) & (boot.status == 'OK') & boot.absolute_change.notna()]
    require(len(rows) == 1, f'missing sensitivity bootstrap {seed}/{endpoint}')
    return rows.iloc[0]


def association(x, y):
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(a) < 3 or not np.isfinite(a).all() or not np.isfinite(b).all() or np.std(a) == 0 or np.std(b) == 0:
        return dict(pearson=None, spearman=None)
    return dict(pearson=float(np.corrcoef(a, b)[0, 1]), spearman=float(spearmanr(a, b).statistic))


def run():
    protocol_path = PROTOCOLS / 'DELIVERED23_POSTHOC_ANALYSIS_PROTOCOL.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    require(protocol['confirmatory_decision'] == 'FAIL' and protocol['confirmatory_result_immutable']
            and protocol['posthoc_analysis_can_not_rescue_primary'], 'post-hoc protocol does not freeze original failure')
    source = OUTPUT_ROOT / 'delivered23' / protocol['source_execution_id']
    for name, expected in protocol['input_sha256'].items():
        require(not VERIFY_HASHES or sha256(source / name) == expected, f'IMMUTABLE_INPUT_HASH_MISMATCH: {name}')
    decision = json.loads((source / 'DELIVERED23_FINAL_DECISION.json').read_text(encoding='utf-8'))
    execution = json.loads((source / 'DELIVERED23_EVALUATION_PROTOCOL.json').read_text(encoding='utf-8'))
    require(decision['overall_decision'] == 'FAIL' and source.name == protocol['source_execution_id'], 'original decision mismatch')
    require(len(execution['exact_membership']) == 23 and decision['complete_source_count'] == 23, 'population mismatch')
    manifest = require_private(DATA_ROOT / 'data' / '01_locked_source_manifest.csv', 'source metadata')
    require(not VERIFY_HASHES or sha256(manifest) == protocol['source_manifest_sha256'], 'locked manifest hash mismatch')
    membership = execution['exact_membership']
    cohorts = cohort_ids(membership)
    require(len(cohorts['frozen_external']) == 17 and len(cohorts['frozen_final']) == 6, 'cohort size mismatch')
    source_ids = set(cohorts['all23'])
    meta = pd.read_csv(manifest, encoding='utf-8-sig')
    meta = meta.assign(source_id=meta.source_id.map(public_source_id)).set_index('source_id').loc[cohorts['all23']]
    audit = pd.read_csv(source / 'delivered23_source_horizon_output_audit.csv', encoding='utf-8-sig')
    boot = pd.read_csv(source / 'delivered23_cluster_bootstrap.csv', encoding='utf-8-sig')
    sens = pd.read_csv(source / 'delivered23_sensitivity_metrics.csv', encoding='utf-8-sig')
    pred = pd.read_csv(source / 'delivered23_prediction_metrics.csv', encoding='utf-8-sig')
    collateral = pd.read_csv(source / 'delivered23_collateral_effect_audit.csv', encoding='utf-8-sig')
    per_seed = json.loads((source / 'per_seed_delivered23_metrics.json').read_text(encoding='utf-8'))
    require({int(row['seed']) for row in per_seed} == set(SEEDS), 'per-seed result mismatch')
    require(set(audit.source_id) == source_ids and set(audit.seed) == set(SEEDS), 'source audit coverage mismatch')
    require(set(boot.draw_sha256.dropna()) == {decision['per_seed'][0]['draw_sha256']}, 'bootstrap draw mismatch')
    require(len(collateral) > 0 and len(pred) > 0, 'missing sealed secondary results')

    with np.load(source / 'bootstrap_draws.npz', allow_pickle=False) as draws:
        order = [str(x) for x in draws['source_order']]
        counts = draws['counts'].astype(np.float64)
        require(counts.shape == (20000, 23) and set(order) == source_ids, 'sealed draw table invalid')
        require(np.all(counts[:, :17].sum(1) == 17) and np.all(counts[:, 17:].sum(1) == 6), 'stratum resampling mismatch')
    source_audit = audit.set_index(['seed', 'source_id', 'horizon_s', 'output'])
    source_sens = sens[(sens.source_id != 'ALL') & (sens.aggregation == 'source')]
    if source_sens.empty:
        source_sens = sens[sens.source_id != 'ALL']
    require(set(source_sens.source_id) == source_ids, 'source sensitivity coverage mismatch')
    sens_index = source_sens.set_index(['seed', 'source_id', 'variant', 'output'])

    # Primary values are quoted from the immutable decision and bootstrap rows.
    primary = []
    for record in per_seed:
        seed = int(record['seed'])
        require(record['decision'] == 'FAIL', 'seed confirmatory decision changed')
        for endpoint in ('MAE', 'PLE', 'WSR', 'PSM'):
            v = record['primary_values'][endpoint]
            row = paired_bootstrap(boot, 'all23', seed, 'Tp', 120, 'MAE') if endpoint == 'MAE' else paired_sensitivity_bootstrap(boot, seed, endpoint)
            require(np.isclose(float(row.R0), v['R0']) and np.isclose(float(row.COISC), v['COISC']), 'primary point estimate disagreement')
            primary.append(dict(seed=seed, endpoint=endpoint, R0=v['R0'], COISC=v['COISC'],
                                absolute_delta=v['COISC'] - v['R0'], relative_delta_percent=relative(v['R0'], v['COISC']),
                                descriptive_absolute_delta_CI=ci_text(row), original_gate=record['gates']['Tp_120s_MAE' if endpoint == 'MAE' else endpoint]['decision']))

    cohort_rows, horizon_rows = [], []
    for seed in SEEDS:
        for cohort in cohorts:
            for h in HORIZONS:
                row = paired_bootstrap(boot, cohort, seed, 'Tp', h, 'MAE')
                item = dict(analysis_class=LABEL, tag='POST-HOC STRATIFIED ANALYSIS', seed=seed,
                            cohort=cohort, n_sources=len(cohorts[cohort]), horizon_s=h,
                            R0_MAE_C=float(row.R0), COISC_MAE_C=float(row.COISC),
                            delta_MAE_C=float(row.absolute_change), relative_delta_percent=float(row.relative_change_percent),
                            paired_source_cluster_CI=ci_text(row), original_draw_sha256=row.draw_sha256)
                cohort_rows.append(item)
                horizon_rows.append(dict(item, delta_vs_10s_C=None))
    for item in horizon_rows:
        first = next(row for row in cohort_rows if row['seed'] == item['seed'] and row['cohort'] == item['cohort'] and row['horizon_s'] == 10)
        item['delta_vs_10s_C'] = item['delta_MAE_C'] - first['delta_MAE_C']

    output_rows = []
    for seed in SEEDS:
        for output in POINTS:
            for h in HORIZONS:
                row = paired_bootstrap(boot, 'all23', seed, output, h, 'MAE')
                output_rows.append(dict(analysis_class=LABEL, seed=seed, output=output, horizon_s=h,
                                        n_sources=23, R0_MAE_C=float(row.R0), COISC_MAE_C=float(row.COISC),
                                        delta_MAE_C=float(row.absolute_change), relative_delta_percent=float(row.relative_change_percent),
                                        paired_source_cluster_CI=ci_text(row), CI_origin='original sealed bootstrap'))
            series = audit[(audit.seed == seed) & (audit.output == output)]
            by_source = series.groupby('source_id')[['R0_MAE_C', 'COISC_MAE_C']].mean().loc[order]
            require(len(by_source) == 23 and series.groupby('source_id').size().eq(5).all(), 'five-horizon output macro incomplete')
            a, b = by_source.R0_MAE_C.to_numpy(), by_source.COISC_MAE_C.to_numpy()
            delta_draws = (counts @ (b - a)) / 23
            ci = np.percentile(delta_draws, [2.5, 97.5]).tolist()
            output_rows.append(dict(analysis_class=LABEL, seed=seed, output=output, horizon_s='macro_5_horizons',
                                    n_sources=23, R0_MAE_C=float(a.mean()), COISC_MAE_C=float(b.mean()),
                                    delta_MAE_C=float((b-a).mean()), relative_delta_percent=relative(float(a.mean()), float(b.mean())),
                                    paired_source_cluster_CI=ci, CI_origin='descriptive macro from original shared draws'))

    source_rows = []
    for seed in SEEDS:
        for source_id in cohorts['all23']:
            tp = source_audit.loc[(seed, source_id, 120, 'Tp')]
            others = [source_audit.loc[(seed, source_id, 120, o)] for o in POINTS]
            a = float(np.mean([float(x.R0_MAE_C) for x in others]))
            b = float(np.mean([float(x.COISC_MAE_C) for x in others]))
            s = {}
            for variant in ('R0', 'COISC'):
                x = sens_index.loc[(seed, source_id, variant, 'Tp')]
                s[variant] = {key: float(x[key]) for key in ('PLE', 'WSR', 'PSM')}
            item = dict(analysis_class=LABEL, seed=seed, source_id=source_id,
                        role=next(m['role'] for m in membership if m['source_id'] == source_id),
                        source_group=meta.loc[source_id, 'source_group'], domain=meta.loc[source_id, 'domain'],
                        scene=meta.loc[source_id, 'scene'], machine=meta.loc[source_id, 'machine'],
                        R0_Tp_120s_MAE_C=float(tp.R0_MAE_C), COISC_Tp_120s_MAE_C=float(tp.COISC_MAE_C),
                        Tp_120s_delta_MAE_C=float(tp.delta_MAE_C),
                        R0_all7_120s_macro_MAE_C=a, COISC_all7_120s_macro_MAE_C=b, all7_120s_macro_delta_MAE_C=b-a)
            for key in ('PLE', 'WSR', 'PSM'):
                item[f'R0_Tp_{key}'] = s['R0'][key]
                item[f'COISC_Tp_{key}'] = s['COISC'][key]
                item[f'Tp_{key}_delta'] = s['COISC'][key] - s['R0'][key]
            source_rows.append(item)
    source_rows.sort(key=lambda x: (x['seed'], -x['Tp_120s_delta_MAE_C']))
    source_frame = pd.DataFrame(source_rows)

    # S arrays are consumed only for a source-balanced PSM in descriptive subgroups.
    post_cells = {}
    with np.load(source / 'window_level_sensitivity.npz', allow_pickle=False) as snpz, np.load(source / 'window_level_raw_results.npz', allow_pickle=False) as rnpz:
        post = snpz['post_mask'][:, 6]
        identities = [key for key in rnpz.files if key.endswith('_identity')]
        require(len(identities) == 138, 'raw paired record count mismatch')
        raw_error = {}
        for i in range(138):
            seed_s, source_id, role, variant = map(str, rnpz[f'record_{i}_identity'])
            seed = int(seed_s)
            require(seed in SEEDS and source_id in source_ids and variant in ('R0', 'COISC'), 'unexpected raw identity')
            key = (seed, source_id, variant)
            require(key not in raw_error, 'duplicate raw identity')
            errors = rnpz[f'record_{i}_signed_error']
            absolute = rnpz[f'record_{i}_absolute_error']
            require(errors.ndim == 3 and errors.shape[1:] == (12, 7) and np.isfinite(errors).all(), 'invalid raw error shape/value')
            require(np.array_equal(np.abs(errors), absolute), 'absolute/signed error mismatch')
            raw_error[key] = (errors.mean(axis=0), np.square(errors).mean(axis=0), absolute.mean(axis=0))
            svalue = snpz[f'record_{i}_S']
            require(svalue.shape == errors.shape and np.isfinite(svalue).all(), 'invalid sensitivity record')
            post_cells[key] = np.abs(svalue[:, post, 6]).ravel()
            if variant == 'COISC':
                prior = i-1
                require(list(map(str, rnpz[f'record_{prior}_identity']))[:3] == [seed_s, source_id, role]
                        and str(rnpz[f'record_{prior}_identity'][3]) == 'R0', 'raw pair adjacency mismatch')
                require(np.array_equal(rnpz[f'record_{prior}_window_ids'], rnpz[f'record_{i}_window_ids']), 'paired windows mismatch')
        require(len(raw_error) == 138, 'raw coverage incomplete')

    groups = []
    for field in ('source_group', 'domain', 'scene', 'machine', 'role'):
        values = {str(v): [source_id for source_id in cohorts['all23'] if str(meta.loc[source_id, field]) == str(v)]
                  for v in meta[field].unique()} if field != 'role' else {
                      role: cohorts[role] for role in ('frozen_external', 'frozen_final')}
        groups += [(field, name, ids) for name, ids in values.items()]
    groups += [('loop_topology', 'open_loop_metadata', [s for s in cohorts['all23'] if meta.loc[s, 'domain'] == 'single_open']),
               ('loop_topology', 'unknown', [s for s in cohorts['all23'] if meta.loc[s, 'domain'] != 'single_open']),
               ('heater_scope', 'single_heater_metadata', [s for s in cohorts['all23'] if meta.loc[s, 'domain'] == 'single_open']),
               ('heater_scope', 'multi_heater_metadata', [s for s in cohorts['all23'] if meta.loc[s, 'domain'] == 'multiheater_old'])]
    group_rows = []
    for field, name, ids in groups:
        if not ids:
            continue
        for seed in SEEDS:
            frame = source_frame[(source_frame.seed == seed) & source_frame.source_id.isin(ids)]
            require(len(frame) == len(ids), 'group membership incomplete')
            psm = {}
            for variant in ('R0', 'COISC'):
                psm[variant] = weighted_psm([post_cells[(seed, sid, variant)] for sid in ids])
            item = dict(analysis_class=LABEL, seed=seed, grouping=field, group=name, n_sources=len(ids),
                        descriptive_only=True, small_group=len(ids) < 5,
                        R0_Tp_120s_MAE_C=float(frame.R0_Tp_120s_MAE_C.mean()),
                        COISC_Tp_120s_MAE_C=float(frame.COISC_Tp_120s_MAE_C.mean()),
                        Tp_120s_delta_MAE_C=float(frame.Tp_120s_delta_MAE_C.mean()),
                        R0_Tp_PSM=psm['R0'], COISC_Tp_PSM=psm['COISC'],
                        Tp_PSM_delta=None if psm['R0'] is None or psm['COISC'] is None else psm['COISC']-psm['R0'])
            for metric in ('PLE', 'WSR'):
                item[f'R0_Tp_{metric}'] = float(frame[f'R0_Tp_{metric}'].mean())
                item[f'COISC_Tp_{metric}'] = float(frame[f'COISC_Tp_{metric}'].mean())
                item[f'Tp_{metric}_delta'] = float(frame[f'Tp_{metric}_delta'].mean())
            group_rows.append(item)

    leave_rows, tradeoff_rows, correlation_summary = [], [], []
    for seed in SEEDS:
        frame = source_frame[source_frame.seed == seed].set_index('source_id').loc[cohorts['all23']]
        original = float(frame.Tp_120s_delta_MAE_C.mean())
        require(np.isclose(original, float(next(x for x in primary if x['seed'] == seed and x['endpoint'] == 'MAE')['absolute_delta'])), 'primary delta inconsistent with source audit')
        for sid in cohorts['all23']:
            remaining = frame.drop(index=sid)
            leave_rows.append(dict(analysis_class=LABEL, tag='NOT A CONFIRMATORY REANALYSIS', seed=seed,
                                   omitted_source_id=sid, n_remaining=22, original_all23_delta_MAE_C=original,
                                   descriptive_22_source_delta_MAE_C=float(remaining.Tp_120s_delta_MAE_C.mean()),
                                   shift_from_original_C=float(remaining.Tp_120s_delta_MAE_C.mean()-original),
                                   original_decision_unchanged='FAIL'))
        for y_col, y_name in (('Tp_120s_delta_MAE_C', 'Tp_120s_MAE_delta'),
                              ('all7_120s_macro_delta_MAE_C', 'all7_120s_macro_MAE_delta')):
            for x_col in ('Tp_PLE_delta', 'Tp_WSR_delta', 'Tp_PSM_delta'):
                corr = association(frame[x_col], frame[y_col])
                correlation_summary.append(dict(seed=seed, x=x_col, y=y_name, n_sources=23, **corr))
                for sid, row in frame.iterrows():
                    tradeoff_rows.append(dict(analysis_class=LABEL, seed=seed, source_id=sid,
                                              x_metric=x_col, x_delta=float(row[x_col]), y_metric=y_name,
                                              y_delta_C=float(row[y_col]), n_sources=23, **corr))

    rollout_rows = []
    for seed in SEEDS:
        for cohort, ids in cohorts.items():
            for output in ('Tp', 'T3', 'T4', 'T5'):
                oi = POINTS.index(output)
                for step in range(12):
                    delta = [raw_error[(seed, sid, 'COISC')][2][step, oi]
                             - raw_error[(seed, sid, 'R0')][2][step, oi] for sid in ids]
                    rollout_rows.append(dict(analysis_class=LABEL, seed=seed, cohort=cohort,
                                             output=output, horizon_s=(step+1)*10, n_sources=len(ids),
                                             equal_source_delta_MAE_C=float(np.mean(delta)),
                                             worsened_sources=int(np.count_nonzero(np.asarray(delta)>0)),
                                             improved_sources=int(np.count_nonzero(np.asarray(delta)<0))))

    bias_rows = []
    for seed in SEEDS:
        for cohort, ids in cohorts.items():
            for output in POINTS:
                oi = POINTS.index(output)
                for h in HORIZONS:
                    j = h//10-1
                    vals = {}
                    for variant in ('R0', 'COISC'):
                        mu = float(np.mean([raw_error[(seed,sid,variant)][0][j,oi] for sid in ids]))
                        e2 = float(np.mean([raw_error[(seed,sid,variant)][1][j,oi] for sid in ids]))
                        mae = float(np.mean([raw_error[(seed,sid,variant)][2][j,oi] for sid in ids]))
                        vals[variant] = dict(bias=mu, spread=float(np.sqrt(max(e2-mu*mu,0))), mae=mae,
                                             rmse=float(np.sqrt(e2)))
                    bias_rows.append(dict(analysis_class=LABEL, seed=seed, cohort=cohort, output=output,
                                          horizon_s=h, n_sources=len(ids), R0_bias_C=vals['R0']['bias'],
                                          COISC_bias_C=vals['COISC']['bias'], bias_delta_C=vals['COISC']['bias']-vals['R0']['bias'],
                                          R0_error_spread_C=vals['R0']['spread'], COISC_error_spread_C=vals['COISC']['spread'],
                                          spread_delta_C=vals['COISC']['spread']-vals['R0']['spread'],
                                          R0_MAE_C=vals['R0']['mae'], COISC_MAE_C=vals['COISC']['mae'],
                                          delta_MAE_C=vals['COISC']['mae']-vals['R0']['mae']))

    head_rows, head_associations = [], []
    checkpoint_map = {(int(x['seed']), x['variant']): x for x in execution['checkpoints']}
    for seed in SEEDS:
        states = {}
        for variant in ('R0', 'COISC'):
            item = checkpoint_map[(seed, variant)]
            require(not VERIFY_HASHES or sha256(item['path']) == item['expected_sha256'], f'checkpoint hash changed {seed}/{variant}')
            payload = load_checkpoint(item['path'])
            require(int(payload['seed']) == seed, 'checkpoint seed mismatch')
            states[variant] = payload['state_dict']
        require(set(states['R0']) == set(states['COISC']), 'checkpoint tensor keys changed')
        require(all(torch.equal(t, states['COISC'][key]) for key,t in states['R0'].items() if not key.startswith('head.')),
                'non-head tensor changed')
        for key in HEAD_KEYS:
            a = states['R0'][key].detach().cpu().double().numpy().ravel()
            b = states['COISC'][key].detach().cpu().double().numpy().ravel()
            d = b-a
            den = np.linalg.norm(a)*np.linalg.norm(b)
            head_rows.append(dict(analysis_class=LABEL, seed=seed, parameter=key, output='ALL',
                                  count=len(a), delta_L1=float(np.abs(d).sum()), delta_L2=float(np.linalg.norm(d)),
                                  delta_max_abs=float(np.max(np.abs(d))),
                                  cosine_similarity=None if den == 0 else float(np.dot(a,b)/den), bias_delta=None,
                                  output_macro_delta_MAE_C=None))
        a = states['R0']['head.2.weight'].detach().cpu().double().numpy()
        b = states['COISC']['head.2.weight'].detach().cpu().double().numpy()
        ab = states['R0']['head.2.bias'].detach().cpu().double().numpy()
        bb = states['COISC']['head.2.bias'].detach().cpu().double().numpy()
        norms, errors = [], []
        for oi, output in enumerate(POINTS):
            d = b[oi]-a[oi]
            den = np.linalg.norm(a[oi])*np.linalg.norm(b[oi])
            macro = next(r['delta_MAE_C'] for r in output_rows if r['seed'] == seed and r['output'] == output and r['horizon_s'] == 'macro_5_horizons')
            norms.append(float(np.linalg.norm(d)))
            errors.append(float(macro))
            head_rows.append(dict(analysis_class=LABEL, seed=seed, parameter='head.2.output_row', output=output,
                                  count=len(a[oi]), delta_L1=float(np.abs(d).sum()), delta_L2=float(np.linalg.norm(d)),
                                  delta_max_abs=float(np.max(np.abs(d))),
                                  cosine_similarity=None if den == 0 else float(np.dot(a[oi],b[oi])/den),
                                  bias_delta=float(bb[oi]-ab[oi]), output_macro_delta_MAE_C=float(macro)))
        head_associations.append(dict(seed=seed, x='head.2 row delta L2', y='five-horizon output MAE delta',
                                      n_outputs=7, **association(norms, errors)))

    write_csv('POSTHOC_COHORT_COMPARISON.csv', cohort_rows)
    write_csv('POSTHOC_Tp_HORIZON_TRADEOFF.csv', horizon_rows)
    write_csv('POSTHOC_OUTPUT_INTERFERENCE.csv', output_rows)
    write_csv('POSTHOC_HEAD_PARAMETER_DRIFT.csv', head_rows)
    write_csv('POSTHOC_SENSITIVITY_PREDICTION_TRADEOFF.csv', tradeoff_rows)
    write_csv('POSTHOC_SOURCE_FAILURE_TABLE.csv', source_rows)
    write_csv('POSTHOC_LEAVE_ONE_SOURCE_OUT.csv', leave_rows)
    write_csv('POSTHOC_ROLLOUT_ACCUMULATION.csv', rollout_rows)
    write_csv('POSTHOC_BIAS_DECOMPOSITION.csv', bias_rows)
    write_csv('POSTHOC_SOURCE_GROUP_DECOMPOSITION.csv', group_rows)

    seed_summary = []
    for seed in SEEDS:
        frame = source_frame[source_frame.seed == seed]
        loo = [x['descriptive_22_source_delta_MAE_C'] for x in leave_rows if x['seed'] == seed]
        ext = [x for x in cohort_rows if x['seed'] == seed and x['cohort'] == 'frozen_external' and x['horizon_s'] == 120][0]
        fin = [x for x in cohort_rows if x['seed'] == seed and x['cohort'] == 'frozen_final' and x['horizon_s'] == 120][0]
        seed_summary.append(dict(seed=seed, confirmatory_decision='FAIL',
                                 worsened_sources=int((frame.Tp_120s_delta_MAE_C>0).sum()),
                                 improved_sources=int((frame.Tp_120s_delta_MAE_C<0).sum()),
                                 external17_Tp_120_delta_C=ext['delta_MAE_C'], final6_Tp_120_delta_C=fin['delta_MAE_C'],
                                 leave_one_out_delta_min_C=float(min(loo)), leave_one_out_delta_max_C=float(max(loo)),
                                 top5_worsened=frame.nlargest(5,'Tp_120s_delta_MAE_C')[['source_id','Tp_120s_delta_MAE_C']].to_dict('records'),
                                 top5_improved=frame.nsmallest(5,'Tp_120s_delta_MAE_C')[['source_id','Tp_120s_delta_MAE_C']].to_dict('records')))
    seed_stats = {}
    for endpoint in ('MAE','PLE','WSR','PSM'):
        records = [x for x in primary if x['endpoint'] == endpoint]
        seed_stats[endpoint] = {key: dict(mean=float(np.mean([r[key] for r in records])), median=float(np.median([r[key] for r in records])))
                                for key in ('R0', 'COISC', 'absolute_delta', 'relative_delta_percent')}
    summary = dict(analysis_class=LABEL, confirmatory_decision='FAIL', confirmatory_result_immutable=True,
                   posthoc_analysis_can_not_rescue_primary=True, source_execution_id=source.name,
                   protocol_sha256=sha256(protocol_path), source_protocol_sha256=protocol['source_protocol_sha256'],
                   n_sources=23, external=17, final=6, n_seeds=3,
                   primary_tradeoff=primary, three_seed_descriptive_mean_median=seed_stats,
                   per_seed_source_distribution=seed_summary, source_level_correlations=correlation_summary,
                   head_row_drift_output_error_associations=head_associations,
                   preprocessing_provenance_status=execution['preprocessing_provenance_status'],
                   notes=['Three seeds are not independent statistical samples; mean and median are descriptive only.',
                          'All subgroup and leave-one-out results are post-hoc and do not alter the original all23 FAIL.',
                          'Historical preprocessing provenance is partially unresolved; four legacy metadata mismatches remain informational.'])
    summary['answers_A_to_I'] = {
        'A': 'Heterogeneous, not universal and not attributable to one source alone: 12-16 of 23 sources worsen per seed; all 23 leave-one-source-out point deltas remain positive. Several high-power fixed sources contribute strongly, partly offset by improving sources.',
        'B': 'Yes, Tp degradation generally accumulates over the 10-120 s recursive rollout. Seed 20260923 already worsens at 10 s; seed 20260924 changes from improvement at 10 s to degradation by 120 s.',
        'C': 'Yes, descriptively: external17 Tp 120 s delta is positive in all three seeds, whereas final6 delta is negative in all three. Cohort, source_group, domain and heater scope are confounded here; this is not an alternate primary result.',
        'D': 'T3 and T4 have consistent positive five-horizon MAE deltas in all three seeds; Tp 120 s worsens in all three. T1 also worsens modestly. T2 and T6 are less uniform.',
        'E': 'T5 five-horizon macro MAE improves slightly in all three seeds, but its 120 s MAE worsens in two seeds and improves in one; improvement is not uniform across horizons.',
        'F': 'No stable PLE/WSR-vs-prediction association is established: source-level Pearson/Spearman signs and sizes vary. PSM change is positively correlated with Tp degradation in these samples, but all correlations are exploratory (23 clustered sources and dependent model seeds), not a confirmatory statistical finding.',
        'G': 'No reproducible association between head.2 output-row L2 drift and output MAE degradation: the seven-output Pearson correlations change sign across seeds. Non-head tensors are equal, while the shared head.0 and all output rows change; drift magnitude is not causal proof.',
        'H': 'The most plausible working explanation is that sensitivity-constrained adaptation of a shared head shifts factual predictions (including bias and error spread), with recursive rollout amplifying some shifts. Source-domain heterogeneity and the inconsistent drift correlations prevent a causal mechanism claim.',
        'I': 'A separately authorized control-level validation could test whether improved local UH sensitivity/ordering yields better action decisions despite degraded 120 s Tp prediction. Keep the existing checkpoints and original FAIL fixed; this is only a hypothesis, not an experiment launched here.'
    }
    (HERE / 'DELIVERED23_POSTHOC_SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    make_markdown(summary, cohort_rows, output_rows, rollout_rows, bias_rows, group_rows)
    print(json.dumps(dict(status='COMPLETE', decision='FAIL', artifacts=12, protocol_sha256=summary['protocol_sha256'])), flush=True)


def make_markdown(summary, cohort_rows, output_rows, rollout_rows, bias_rows, group_rows):
    lines = ['# Delivered23 failure analysis', '',
             '**DESCRIPTIVE / EXPLORATORY / POST-HOC. The original all23 confirmatory decision is FAIL and cannot be rescued.**', '',
             'Population: fixed 17 external + 6 final sources; three fixed seeds. Source-balanced point estimates; no model execution or training.', '',
             '## Primary trade-off', '',
             '| Seed | Tp 120s MAE R0 | COISC | Delta | 95% CI | PLE delta | WSR delta | PSM delta |',
             '|---|---:|---:|---:|---|---:|---:|---:|']
    for seed in SEEDS:
        v = {x['endpoint']: x for x in summary['primary_tradeoff'] if x['seed'] == seed}
        x = v['MAE']
        lines.append(f"| {seed} | {x['R0']:.4f} | {x['COISC']:.4f} | {x['absolute_delta']:+.4f} | [{x['descriptive_absolute_delta_CI'][0]:+.4f}, {x['descriptive_absolute_delta_CI'][1]:+.4f}] | {v['PLE']['absolute_delta']:+.4f} | {v['WSR']['absolute_delta']:+.4f} | {v['PSM']['absolute_delta']:+.4f} |")
    lines += ['', 'All three Tp non-inferiority gates fail against the unchanged +0.005 C margin. All three PLE, WSR and PSM gates pass. Three-seed mean/median in the JSON are descriptive, not independent-sample statistics.', '',
              '## Cohort and source heterogeneity', '']
    for x in summary['per_seed_source_distribution']:
        lines.append(f"- Seed {x['seed']}: Tp 120s worsens in {x['worsened_sources']}/23 sources; external17 delta {x['external17_Tp_120_delta_C']:+.4f} C, final6 delta {x['final6_Tp_120_delta_C']:+.4f} C; leave-one-out descriptive delta range [{x['leave_one_out_delta_min_C']:+.4f}, {x['leave_one_out_delta_max_C']:+.4f}] C.")
    lines += ['', 'The external/final comparison and source groups are **POST-HOC STRATIFIED ANALYSIS**, not alternate primary populations. Complete 23-source rows, top-five worsened/improved examples, and small-group flags are in the CSVs.', '',
              '## Horizon and collateral pattern', '']
    for seed in SEEDS:
        vals = [x for x in cohort_rows if x['seed'] == seed and x['cohort'] == 'all23']
        lines.append(f"- Seed {seed} Tp MAE delta at 10/30/60/90/120s: " + ', '.join(f"{x['delta_MAE_C']:+.4f}" for x in vals) + ' C.')
        outputs = [x for x in output_rows if x['seed'] == seed and x['horizon_s'] == 'macro_5_horizons']
        lines.append('  Five-horizon output delta: ' + ', '.join(f"{x['output']} {x['delta_MAE_C']:+.4f}" for x in outputs) + ' C.')
    lines += ['', 'The 12-step source-balanced trajectories, bias and residual-spread decomposition are supplied in their full CSVs. A head-row drift/error correlation is an association across seven outputs only, not causal proof.', '',
              '## Descriptive associations', '']
    for x in summary['head_row_drift_output_error_associations']:
        lines.append(f"- Seed {x['seed']} head.2 row L2 drift vs five-horizon output MAE delta: Pearson {x['pearson']:+.3f}, Spearman {x['spearman']:+.3f} (n=7 outputs).")
    lines += ['', 'Source-level PLE/WSR/PSM change versus Tp and seven-output prediction deltas are reported with Pearson/Spearman in `DELIVERED23_POSTHOC_SUMMARY.json` and every source point in `POSTHOC_SENSITIVITY_PREDICTION_TRADEOFF.csv`.', '',
              '## Answers A-I (exploratory)', '']
    for key, answer in summary['answers_A_to_I'].items():
        lines.append(f'- **{key}.** {answer}')
    lines += ['',
              '## Mechanistic interpretation and testable control hypothesis', '',
              'Head-only adaptation changes shared `head.0` features and every `head.2` output row while preserving the encoder. The observed sensitivity improvements coexist with source-dependent Tp prediction error and collateral changes. This is consistent with a prediction-versus-local-sensitivity trade-off, but neither head drift nor cross-source correlation proves mechanism or causality. A later, separately authorized control-level study could test whether better local sensitivity ordering improves decisions despite worse open-loop prediction; it must keep the present model and failed confirmatory result unchanged.', '',
              '## Limitations', '',
              '- Original 29-source estimand was not tested; these observations concern only the pre-fixed delivered23 population.',
              '- Three seeds are not independent statistical units. Every subgroup, correlation, and leave-one-out result is exploratory.',
              '- Historical preprocessing provenance remains partially unresolved (87/90 available development sources reproduced bitwise; 3 mismatches; 19 raw sources absent).',
              '- Machine and loop topology are only reported where locked metadata supports them; no filename-based source exclusion or manual selection was performed.', '']
    (HERE / 'DELIVERED23_FAILURE_ANALYSIS.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    run()
