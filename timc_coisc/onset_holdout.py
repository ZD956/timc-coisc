"""Holdout (delivered-23) inference for the onset-aligned study.

  onset_holdout.py gate   R4: existing R0/COISC/L0 reproduce stored holdout records / summaries
  onset_holdout.py run    all E1/E2 models; requires the frozen protocol and author authorization
"""
from __future__ import annotations
import argparse
import csv
import json
import numpy as np

import onset_common as oc
from onset_common import screen, torch
from paper_models import VanillaLSTM  # noqa: E402
from paper_eval import predict  # noqa: E402
from frozen_evaluate import SourceMedian, draw_means, percentile_ci  # noqa: E402

HOLD = oc.DELIVERED23 / 'delivered23_20260926T111610620117Z_0415da4e90'
NORMALIZATION = oc.CONFIGS / 'normalization.json'
HORIZONS = (10, 30, 60, 90, 120)
T = np.arange(1, 13) * 10
TP, T6 = 6, 5
EPS = screen.EPS_SIGN


def lag_mask(tau, g=10):
    return T < max(0., tau - g), T > tau + g


MASKS = {
    'Tp': {'ONSET': lag_mask(oc.TAU_ONSET[TP]), 'PRIOR': lag_mask(oc.TAU_PRIOR[TP]), 'T30': lag_mask(30.), 'T10': lag_mask(10.),
           'T20_G0': lag_mask(20., 0), 'T10_G0': lag_mask(10., 0), 'ALL_STEPS': (np.zeros(12, bool), np.ones(12, bool))},
    'T6': {'ONSET': lag_mask(oc.TAU_ONSET[T6]), 'PRIOR': lag_mask(oc.TAU_PRIOR[T6]), 'T60': lag_mask(60.), 'T80': lag_mask(80.),
           'T40_G0': lag_mask(40., 0), 'ALL_STEPS': (np.zeros(12, bool), np.ones(12, bool))},
}
OUT_INDEX = {'Tp': TP, 'T6': T6}


def load_holdout():
    raw = np.load(HOLD / 'window_level_raw_results.npz', allow_pickle=True)
    stored_s = np.load(HOLD / 'window_level_sensitivity.npz', allow_pickle=True)
    draws = np.load(HOLD / 'bootstrap_draws.npz', allow_pickle=True)
    order = [str(x) for x in draws['source_order']]; counts = draws['counts']
    assert counts.shape == (20000, 23) and len(order) == 23
    rec = {}
    for i in range(138):
        seed, source, _, variant = [str(x) for x in raw[f'record_{i}_identity']]
        rec[(seed, variant, source)] = i
    first = str(oc.E1_SEEDS[0])
    data = [dict(h=raw[f'record_{rec[(first, "R0", z)]}_history'], u=raw[f'record_{rec[(first, "R0", z)]}_future'],
                 y=raw[f'record_{rec[(first, "R0", z)]}_target'].astype(np.float64)) for z in order]
    return raw, stored_s, counts, order, rec, data


def load_model(path, stats, device):
    m = VanillaLSTM(stats); m.load_state_dict(oc.load_checkpoint(path)['state_dict'], strict=True)
    return m.eval().requires_grad_(False).to(device)


def infer(model, h, u, device):
    pred = predict(model, h, u, device, batch=256); s = []
    with torch.no_grad():
        for i in range(0, len(h), 256):
            s.append(screen.sensitivity(model, torch.as_tensor(h[i:i + 256], device=device, dtype=torch.float32),
                                        torch.as_tensor(u[i:i + 256], device=device, dtype=torch.float32)).cpu().numpy())
    return pred, np.concatenate(s)


def summarize(pred, target, sens):
    e = np.abs(pred.astype(np.float64) - target)[:, np.array(HORIZONS) // 10 - 1]
    r = {f'Tp_MAE_{h}': e[:, i, TP].mean() for i, h in enumerate(HORIZONS)}
    r['all7_MAE_120'] = e[:, 4].mean(0).mean(); r['all7_MAE_5h'] = e.mean(0).mean()
    cells = {}
    for out, specs in MASKS.items():
        x = sens[:, :, OUT_INDEX[out]]
        for name, (pm, qm) in specs.items():
            if pm.any(): r[f'{out}_PLE_{name}'] = np.abs(x[:, pm]).mean()
            if qm.any(): r[f'{out}_WSR_{name}'] = (x[:, qm] < -EPS).mean(); cells[f'{out}_PSM_{name}'] = np.abs(x[:, qm]).ravel()
    for k in range(6):  # early lobe, 10-60 s
        r[f'Tp_fracneg_{(k + 1) * 10}s'] = (sens[:, k, TP] < -EPS).mean()
        cells[f'Tp_medS_{(k + 1) * 10}s'] = sens[:, k, TP].astype(np.float64)
    return r, cells


def model_estimates(model, data, counts, device, keep=None):
    comps, cells = [], []
    for i, d in enumerate(data):
        pred, sens = infer(model, d['h'], d['u'], device)
        if keep is not None: keep(i, pred, sens)
        a, c = summarize(pred, d['y'], sens); comps.append(a); cells.append(c)
    est, boot = {}, {}
    one = np.ones((1, 23), int)
    for k in comps[0]:
        vals = np.array([c[k] for c in comps]); est[k] = float(vals.mean()); boot[k] = draw_means(vals, counts)
    for k in cells[0]:
        calc = SourceMedian([c[k] for c in cells]); est[k] = float(calc.many(one)[0]); boot[k] = calc.many(counts)
    return est, boot


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def existing_bindings():
    formal = oc.read_json(HOLD / 'DELIVERED23_EVALUATION_PROTOCOL.json')
    abl = oc.read_json(oc.PROTOCOLS / 'POSTHOC_HOLDOUT_ABLATION_PROTOCOL.json')['models']['ablation_checkpoint_sha256']
    out = []
    for c in formal['checkpoints']:
        if c['variant'] in ('R0', 'COISC'):
            out.append(dict(seed=int(c['seed']), variant=c['variant'], path=c['path'], sha256=c['expected_sha256']))
    for seed in oc.E1_SEEDS:
        out.append(dict(seed=seed, variant='L0', path=str(oc.ABLATION / f'seed_{seed}/L0_final.pt'), sha256=abl[str(seed)]['L0']))
    return out


def gate():
    raw, stored_s, counts, order, rec, data = load_holdout()
    stats = oc.read_json(NORMALIZATION); torch.set_num_threads(2); device = torch.device('cuda')
    with (oc.ABLATION / 'holdout_ablation_summary.csv').open(encoding='utf-8-sig') as f:
        abl_rows = {(r['seed'], r['variant']): r for r in csv.DictReader(f)}
    rows = []
    for b in existing_bindings():
        assert oc.sha(oc.private_path(b['path'])) == b['sha256'], f'hash mismatch {b["seed"]} {b["variant"]}'
        seed = str(b['seed']); model = load_model(oc.private_path(b['path']), stats, device); diffs = []

        def keep(i, pred, sens, seed=seed, v=b['variant']):
            if v in ('R0', 'COISC'):
                j = rec[(seed, v, order[i])]
                diffs.append(max(float(np.abs(pred - raw[f'record_{j}_prediction']).max()), float(np.abs(sens - stored_s[f'record_{j}_S']).max())))
        est, _ = model_estimates(model, data, counts, device, keep)
        ref = abl_rows[(seed, b['variant'])]
        summary_diff = max(abs(est[a] - float(ref[c])) for a, c in (('Tp_MAE_120', 'Tp_MAE_120'), ('all7_MAE_5h', 'all7_MAE_5h'),
                           ('Tp_PLE_PRIOR', 'PLE_PRIOR'), ('Tp_WSR_PRIOR', 'WSR_PRIOR'), ('Tp_PSM_PRIOR', 'PSM_PRIOR'), ('Tp_WSR_ALL_STEPS', 'WSR_ALL_STEPS')))
        rows.append(dict(seed=b['seed'], variant=b['variant'], window_max_abs_diff=max(diffs) if diffs else None, summary_max_abs_diff=summary_diff,
                         pass_=bool((not diffs or max(diffs) <= 1e-5) and summary_diff <= 1e-9)))
        print('R4', rows[-1], flush=True)
    report = dict(label='POST-HOC onset-aligned study: gate R4 (existing models only)', rows=rows, pass_=all(r['pass_'] for r in rows),
                  code_sha256={n: oc.sha(oc.HERE / n) for n in ('onset_common.py', 'onset_holdout.py')})
    screen.save_json(oc.PROTO / 'GATES_HOLDOUT_REPORT.json', report)
    print('R4 PASS', report['pass_'])


CONTRAST_METRICS = ('Tp_MAE_10', 'Tp_MAE_30', 'Tp_MAE_60', 'Tp_MAE_90', 'Tp_MAE_120', 'all7_MAE_5h',
                    'Tp_WSR_ONSET', 'Tp_PSM_ONSET', 'Tp_WSR_PRIOR', 'Tp_PLE_PRIOR', 'Tp_PSM_PRIOR', 'Tp_WSR_ALL_STEPS', 'Tp_PSM_ALL_STEPS',
                    'T6_PLE_ONSET', 'T6_WSR_ONSET', 'T6_PSM_ONSET', 'T6_PLE_PRIOR')


def run():
    protocol, digest = oc.frozen_protocol()
    raw, stored_s, counts, order, rec, data = load_holdout()
    stats = oc.read_json(NORMALIZATION); torch.set_num_threads(2); device = torch.device('cuda')
    bindings = existing_bindings()
    for mode in ('e1', 'e2'):
        bfile = oc.OUT / mode / f'{mode.upper()}_CHECKPOINT_BINDINGS.json'
        if bfile.exists():
            b = oc.read_json(bfile); assert b['protocol_sha256'] == digest
            bindings += [x for x in b['bindings'] if not (mode == 'e1' and x['variant'] == 'R0')]
    store = oc.OUT / 'holdout'; store.mkdir(parents=True, exist_ok=True)
    per = {}; summary = []
    for b in bindings:
        assert oc.sha(oc.private_path(b['path'])) == b['sha256'], f'hash mismatch {b["seed"]} {b["variant"]}'
        model = load_model(oc.private_path(b['path']), stats, device); arrays = {}

        def keep(i, pred, sens):
            arrays[f'src{i:02d}_pred_Tp'] = pred[:, :, TP].astype(np.float32); arrays[f'src{i:02d}_S_Tp_T6'] = sens[:, :, [TP, T6]].astype(np.float32)
        est, boot = model_estimates(model, data, counts, device, keep)
        np.savez_compressed(store / f'seed_{b["seed"]}_{b["variant"]}.npz', **arrays)
        per[(b['seed'], b['variant'])] = (est, boot)
        summary.append(dict(seed=b['seed'], variant=b['variant'], **est))
        print('HOLDOUT', b['seed'], b['variant'], flush=True)
    contrasts = []
    for (seed, v), (e, bt) in per.items():
        for ref in ('R0', 'COISC'):
            if v == ref or (seed, ref) not in per: continue
            e0, b0 = per[(seed, ref)]
            for m in CONTRAST_METRICS:
                lo, hi = percentile_ci(bt[m] - b0[m])
                row = dict(seed=seed, variant=v, reference=ref, metric=m, value=e[m], ref_value=e0[m], diff=e[m] - e0[m],
                           ratio=e[m] / e0[m] if e0[m] else None, CI_low=lo, CI_high=hi)
                if m.startswith('Tp_MAE') and ref == 'R0': row['NI_0.005'] = bool(hi <= .005)
                if ('_WSR_' in m or '_PLE_' in m) and ref == 'R0':
                    row['CI_high_minus_0.9R0'] = float(percentile_ci(bt[m] - .9 * b0[m])[1])
                if '_PSM_' in m and ref == 'R0':
                    row['CI_low_minus_0.2R0'] = float(percentile_ci(bt[m] - .2 * b0[m])[0])
                contrasts.append(row)
    write_csv(oc.OUT / 'HOLDOUT_SUMMARY.csv', summary); write_csv(oc.OUT / 'HOLDOUT_CONTRASTS.csv', contrasts)
    screen.save_json(oc.OUT / 'HOLDOUT_RUN_AUDIT.json', dict(protocol_sha256=digest, models=len(bindings),
                     label='POST-HOC exploratory; holdout reused, not independent confirmation'))


def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['gate', 'run']); a = p.parse_args()
    gate() if a.mode == 'gate' else run()


if __name__ == '__main__':
    main()
