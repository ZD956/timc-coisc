"""Post-hoc item 2a: ablation variants on the delivered23 holdout (inference only). See POSTHOC_HOLDOUT_ABLATION_PROTOCOL.json."""
from __future__ import annotations
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
import numpy as np

import screen  # noqa: E402
import torch  # noqa: E402
from paper_models import VanillaLSTM  # noqa: E402
from paper_eval import predict  # noqa: E402
from frozen_evaluate import SourceMedian, draw_means, percentile_ci  # noqa: E402

from _paths import PROTOCOLS, CONFIGS, OUTPUT_ROOT, load_checkpoint  # noqa: E402

VERIFY_HASHES = os.environ.get('TIMC_VERIFY_HASHES') == '1'
OUT = OUTPUT_ROOT / 'posthoc_ablation'
PROTOCOL = PROTOCOLS / 'POSTHOC_HOLDOUT_ABLATION_PROTOCOL.json'
PROTOCOL_SHA = 'd44b63e3aa8c00d4b424a5b08f2262f516e356e7f0be8e1c97440ab33c002ad4'
HOLD = OUTPUT_ROOT / 'delivered23' / 'delivered23_20260926T111610620117Z_0415da4e90'
NORMALIZATION = CONFIGS / 'normalization.json'
SEEDS = ('20260922', '20260923', '20260924')
ABL = ('L0', 'SIGN_B', 'SIGN_A', 'PRE_B')
VARIANTS = ('R0', 'COISC') + ABL
HORIZONS = (10, 30, 60, 90, 120)
T = np.arange(1, 13) * 10
EPS = screen.EPS_SIGN
Tp, T6 = 6, 5
MASKS = {'PRIOR': (T < 39.104 - 10, T > 39.104 + 10), 'T20_G0': (T < 20, T > 20), 'ALL_STEPS': (np.zeros(12, bool), np.ones(12, bool))}
T6_PRE = T < 170.


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_model(path, stats, device):
    m = VanillaLSTM(stats); m.load_state_dict(load_checkpoint(path)['state_dict'], strict=True)
    return m.eval().requires_grad_(False).to(device)


def infer(model, h, u, device):
    pred = predict(model, h, u, device, batch=256); s = []
    with torch.no_grad():
        for i in range(0, len(h), 256):
            s.append(screen.sensitivity(model, torch.as_tensor(h[i:i + 256], device=device, dtype=torch.float32),
                                        torch.as_tensor(u[i:i + 256], device=device, dtype=torch.float32)).cpu().numpy())
    return pred, np.concatenate(s)


def summarize(pred, target, sens):
    """Per-source scalars (equal-source components) and PSM cells."""
    e = np.abs(pred.astype(np.float64) - target)[:, np.array(HORIZONS) // 10 - 1]
    r = {f'Tp_MAE_{h}': e[:, i, Tp].mean() for i, h in enumerate(HORIZONS)}
    r['all7_MAE_120'] = e[:, 4].mean(0).mean(); r['all7_MAE_5h'] = e.mean(0).mean()
    x = sens[:, :, Tp]; cells = {}
    for name, (pm, qm) in MASKS.items():
        if pm.any(): r[f'PLE_{name}'] = np.abs(x[:, pm]).mean()
        r[f'WSR_{name}'] = (x[:, qm] < -EPS).mean(); cells[f'PSM_{name}'] = np.abs(x[:, qm]).ravel()
    r['T6_PLE_PRIOR'] = np.abs(sens[:, T6_PRE, T6]).mean()
    return r, cells


def fmt(ci):
    return f'[{ci[0]:.6g}, {ci[1]:.6g}]'


def main():
    assert not VERIFY_HASHES or sha(PROTOCOL) == PROTOCOL_SHA
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    formal = json.loads((HOLD / 'DELIVERED23_EVALUATION_PROTOCOL.json').read_text(encoding='utf-8'))
    stats = json.loads(NORMALIZATION.read_text(encoding='utf-8'))
    raw = np.load(HOLD / 'window_level_raw_results.npz', allow_pickle=True)
    stored_s = np.load(HOLD / 'window_level_sensitivity.npz', allow_pickle=True)
    draws = np.load(HOLD / 'bootstrap_draws.npz', allow_pickle=True)
    order = [str(x) for x in draws['source_order']]; counts = draws['counts']
    rec = {}
    for i in range(138):
        seed, source, _, variant = [str(x) for x in raw[f'record_{i}_identity']]
        rec[(seed, variant, source)] = i
    data = {z: dict(h=raw[f'record_{rec[(SEEDS[0], "R0", z)]}_history'], u=raw[f'record_{rec[(SEEDS[0], "R0", z)]}_future'],
                    y=raw[f'record_{rec[(SEEDS[0], "R0", z)]}_target'].astype(np.float64)) for z in order}
    torch.set_num_threads(2); device = torch.device('cuda')
    repro = []; per = {}; rows = []; contrasts = []
    for seed in SEEDS:
        paths = {c['variant']: c['path'] for c in formal['checkpoints'] if str(c['seed']) == seed}
        for c in formal['checkpoints']:
            if str(c['seed']) == seed: assert not VERIFY_HASHES or sha(c['path']) == c['expected_sha256']
        for a in ABL:
            paths[a] = OUT / f'seed_{seed}/{a}_final.pt'
            assert not VERIFY_HASHES or sha(paths[a]) == protocol['models']['ablation_checkpoint_sha256'][seed][a]
        for v in VARIANTS:
            model = load_model(paths[v], stats, device); comps = []; cells = []
            for z in order:
                d = data[z]; pred, sens = infer(model, d['h'], d['u'], device)
                if v in ('R0', 'COISC'):
                    i = rec[(seed, v, z)]
                    repro.append(dict(seed=seed, variant=v, source=z,
                                      pred_diff=float(np.abs(pred - raw[f'record_{i}_prediction']).max()),
                                      sens_diff=float(np.abs(sens - stored_s[f'record_{i}_S']).max())))
                a, c = summarize(pred, d['y'], sens); comps.append(a); cells.append(c)
            est, boot = {}, {}
            for k in comps[0]:
                vals = np.array([c[k] for c in comps]); est[k] = float(vals.mean()); boot[k] = draw_means(vals, counts)
            for k in cells[0]:
                calc = SourceMedian([c[k] for c in cells])
                est[k] = float(calc.many(np.ones((1, 23), int))[0]); boot[k] = calc.many(counts)
            per[(seed, v)] = (est, boot)
            print(seed, v, 'Tp_120', round(est['Tp_MAE_120'], 4), 'PLE', round(est['PLE_PRIOR'], 4), 'WSR', round(est['WSR_PRIOR'], 4),
                  'PSM', round(est['PSM_PRIOR'], 4), flush=True)
        worst = max(max(r['pred_diff'], r['sens_diff']) for r in repro if r['seed'] == seed)
        if worst > 1e-5:
            raise SystemExit(f'HOLDOUT_PATH_NOT_REPRODUCED seed {seed} max diff {worst}')
        e0, b0 = per[(seed, 'R0')]
        for v in VARIANTS:
            e, b = per[(seed, v)]
            row = dict(seed=seed, variant=v, **{k: e[k] for k in e})
            if v != 'R0':
                row['dTp_MAE_120_vs_R0'] = e['Tp_MAE_120'] - e0['Tp_MAE_120']
                ci = percentile_ci(b['Tp_MAE_120'] - b0['Tp_MAE_120']); row['dTp_MAE_120_CI'] = fmt(ci); row['readout_MAE_NI'] = ci[1] <= .005
                row['dAll7_5h_vs_R0'] = e['all7_MAE_5h'] - e0['all7_MAE_5h']
                for m in ('PLE_PRIOR', 'WSR_PRIOR'):
                    ci = percentile_ci(b[m] - .9 * b0[m]); row[f'ratio_{m}'] = e[m] / e0[m]
                    row[f'readout_{m}'] = bool(e[m] <= .9 * e0[m] and ci[1] <= 0)
                ci1 = percentile_ci(b['PSM_PRIOR'] - .2 * b0['PSM_PRIOR']); ci2 = percentile_ci(b['PSM_PRIOR'] - .0005)
                row['ratio_PSM_PRIOR'] = e['PSM_PRIOR'] / e0['PSM_PRIOR']
                row['readout_PSM_PRIOR'] = bool(e['PSM_PRIOR'] >= max(.2 * e0['PSM_PRIOR'], .0005) and ci1[0] >= 0 and ci2[0] >= 0)
                row['ratio_WSR_ALL_STEPS'] = e['WSR_ALL_STEPS'] / e0['WSR_ALL_STEPS']
                row['ratio_PSM_ALL_STEPS'] = e['PSM_ALL_STEPS'] / e0['PSM_ALL_STEPS']
                row['ratio_T6_PLE_PRIOR'] = e['T6_PLE_PRIOR'] / e0['T6_PLE_PRIOR']
                row['readout_all4'] = all(row[k] for k in ('readout_MAE_NI', 'readout_PLE_PRIOR', 'readout_WSR_PRIOR', 'readout_PSM_PRIOR'))
            rows.append(row)
        e1, b1 = per[(seed, 'COISC')]
        for v in ('L0', 'PRE_B', 'SIGN_A'):
            e, b = per[(seed, v)]
            for m in ('Tp_MAE_120', 'all7_MAE_5h', 'PLE_PRIOR', 'WSR_PRIOR', 'PSM_PRIOR', 'WSR_ALL_STEPS', 'PSM_ALL_STEPS', 'WSR_T20_G0', 'T6_PLE_PRIOR'):
                ci = percentile_ci(b1[m] - b[m])
                contrasts.append(dict(seed=seed, contrast=f'COISC-{v}', metric=m, COISC=e1[m], other=e[m], diff=e1[m] - e[m],
                                      CI_low=ci[0], CI_high=ci[1], excludes_zero=bool(ci[0] > 0 or ci[1] < 0)))
    for name, data_rows in (('holdout_ablation_summary.csv', rows), ('holdout_ablation_contrasts.csv', contrasts), ('holdout_ablation_reproduction.csv', repro)):
        keys = list(dict.fromkeys(k for r in data_rows for k in r))
        with (OUT / name).open('w', encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(data_rows)
    print('max reproduction diff', max(max(r['pred_diff'], r['sens_diff']) for r in repro))


if __name__ == '__main__':
    main()
