"""Post-hoc matched head-only ablation (development only). See POSTHOC_ABLATION_PROTOCOL.json."""
from __future__ import annotations
import hashlib
import json
import os
import sys
import time
from pathlib import Path
import numpy as np

import screen  # noqa: E402  (sets CUBLAS_WORKSPACE_CONFIG before torch use)
import torch  # noqa: E402
from paper_eval import predict, prediction_metrics  # noqa: E402
import formal_coisc  # noqa: E402
from frozen_evaluate import SourceMedian  # noqa: E402

from _paths import PROTOCOLS, OUTPUT_ROOT, load_checkpoint  # noqa: E402

VERIFY_HASHES = os.environ.get('TIMC_VERIFY_HASHES') == '1'
OUT = OUTPUT_ROOT / 'posthoc_ablation'
PROTOCOL = PROTOCOLS / 'POSTHOC_ABLATION_PROTOCOL.json'
PROTOCOL_SHA = '15d873cc468b0cab777a3ab0ec05b1a4919049480951a5870b4423081705cab9'
FORMAL = formal_coisc.OUT
SEEDS = (20260922, 20260923, 20260924)
POINTS = ('T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'Tp')


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def adapt(model, train, probs, old, w_pre, w_sign, seed, device, log_path):
    screen.seed_all(seed)
    model.train()
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith('head.'))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=5e-5)
    rng = np.random.default_rng(seed); stage = dict(old['training']['stages'][2]); start = time.perf_counter()
    with log_path.open('w', encoding='utf-8') as log:
        for step in range(1, 76):
            h, u, y = screen.sample(train, rng, 128, probs[False], device)
            factual = screen.factual(model, h, u, y, stage)
            sens = screen.sensitivity(model, h[:screen.SUBSET], u[:screen.SUBSET]); pre, sign = screen.regularizer(sens)
            loss = factual + w_pre * pre + w_sign * sign
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True); optimizer.step()
            log.write(json.dumps(dict(step=step, prediction_loss=float(factual.detach()), L_pre=float(pre.detach()),
                                      L_sign=float(sign.detach()), total_loss=float(loss.detach()), grad_norm=float(grad))) + '\n')
    return time.perf_counter() - start


def sens_metrics(s, ids, lags=screen.MIN_LAG_SECONDS):
    """Window-pooled (formal convention) and source-equal PLE/WSR/PSM for each output."""
    pre, post = screen.lag_masks(lags); sources = np.unique(ids); out = {}
    for o, point in enumerate(POINTS):
        pm, qm = pre[:, o], post[:, o]; x = s[:, :, o]; r = {}
        r['PLE'] = float(np.abs(x[:, pm]).mean()) if pm.any() else None
        r['WSR'] = float((x[:, qm] < -screen.EPS_SIGN).mean()) if qm.any() else None
        r['PSM'] = float(np.median(np.abs(x[:, qm]))) if qm.any() else None
        r['PLE_src'] = float(np.mean([np.abs(x[ids == z][:, pm]).mean() for z in sources])) if pm.any() else None
        r['WSR_src'] = float(np.mean([(x[ids == z][:, qm] < -screen.EPS_SIGN).mean() for z in sources])) if qm.any() else None
        if qm.any():
            calc = SourceMedian([np.abs(x[ids == z][:, qm]) for z in sources])
            r['PSM_src'] = float(calc.many(np.ones((1, len(sources)), dtype=int))[0])
        else:
            r['PSM_src'] = None
        out[point] = r
    return out


def tp_mae(pred, target):
    return {h: float(np.abs(pred[:, h // 10 - 1, 6] - target[:, h // 10 - 1, 6]).mean()) for h in (10, 30, 60, 90, 120)}


def gates(tp, sm, base_tp, base_sm):
    b = base_sm['Tp']; s = sm['Tp']
    g = dict(Tp_120s_MAE=tp[120] <= base_tp[120] + .005, PLE=s['PLE'] <= .9 * b['PLE'],
             WSR=s['WSR'] <= .9 * b['WSR'], PSM=s['PSM'] >= max(.2 * b['PSM'], .0005))
    g['all4'] = all(g.values())
    return g


def main():
    assert not VERIFY_HASHES or file_sha(PROTOCOL) == PROTOCOL_SHA, 'protocol changed'
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    variants = protocol['item1_matched_head_only_ablation']['variants']
    for s in SEEDS:
        assert not VERIFY_HASHES or file_sha(FORMAL / f'seed_{s}/results/R0/selected.pt') == protocol['inputs_sha256']['R0_selected_pt'][str(s)]
    torch.set_num_threads(2); device = torch.device('cuda')
    train, ev, fit_ids, sel_ids, stats, probs, scores, old = screen.load_real()
    assert len(set(fit_ids)) == 91 and len(set(sel_ids)) == 18 and len(ev.history) == 11073
    ids = np.asarray(sel_ids).astype(str)
    threshold = np.asarray(old['regime_thresholds']['regime_deadband_C'])
    np.save(OUT / 'selection_source_ids.npy', ids)
    rows = []; repro = []
    for seed in SEEDS:
        screen.SEED = seed
        seed_dir = OUT / f'seed_{seed}'; seed_dir.mkdir(exist_ok=True)
        ckpt = load_checkpoint(FORMAL / f'seed_{seed}/results/R0/selected.pt', map_location=device)
        base = screen.build_model('B4', stats).to(device); base.load_state_dict(ckpt['state_dict']); base.eval()
        base_pred = predict(base, ev.history, ev.future, device); base_sens = formal_coisc.sensitivity_all(base, ev, device)
        base_tp = tp_mae(base_pred, ev.target); base_sm = sens_metrics(base_sens, ids)
        base_macro = prediction_metrics(base_pred, ev.target, ev.origin, sel_ids, threshold)[2]
        rows.append(dict(seed=seed, variant='R0', w_pre=None, w_sign=None, train_s=None, macro_MAE=base_macro, Tp_MAE=base_tp, sens=base_sm, gates=None))
        for name, v in variants.items():
            model = screen.build_model('B4', stats).to(device); model.load_state_dict(ckpt['state_dict'])
            wall = adapt(model, train, probs, old, v['w_pre'], v['w_sign'], seed, device, seed_dir / f'{name}_training.jsonl')
            nonhead = all(torch.equal(model.state_dict()[k].detach().cpu(), ckpt['state_dict'][k].detach().cpu())
                          for k in ckpt['state_dict'] if not k.startswith('head.'))
            assert nonhead
            model.eval(); pred = predict(model, ev.history, ev.future, device); sens = formal_coisc.sensitivity_all(model, ev, device)
            np.savez_compressed(seed_dir / f'{name}_selection_prediction_sensitivity.npz', prediction=pred, sensitivity=sens)
            torch.save(dict(state_dict={k: t.detach().cpu() for k, t in model.state_dict().items()}, seed=seed, variant=name,
                            w_pre=v['w_pre'], w_sign=v['w_sign'], protocol_sha256=PROTOCOL_SHA), seed_dir / f'{name}_final.pt')
            tp = tp_mae(pred, ev.target); sm = sens_metrics(sens, ids)
            macro = prediction_metrics(pred, ev.target, ev.origin, sel_ids, threshold)[2]
            rows.append(dict(seed=seed, variant=name, w_pre=v['w_pre'], w_sign=v['w_sign'], train_s=wall, macro_MAE=macro,
                             Tp_MAE=tp, sens=sm, gates=gates(tp, sm, base_tp, base_sm)))
            if name == 'COISC_REPRO':
                ref = load_checkpoint(FORMAL / f'seed_{seed}/coisc_final.pt')['state_dict']
                cur = {k: t.detach().cpu() for k, t in model.state_dict().items()}
                maxdiff = max(float((cur[k].double() - ref[k].double()).abs().max()) for k in ref)
                formal = json.loads((FORMAL / f'seed_{seed}/per_seed_base_and_coisc_metrics.json').read_text(encoding='utf-8'))
                ftp = next(x['MAE_C'] for x in formal['coisc_Tp'] if x['horizon_s'] == 120)
                fbtp = next(x['MAE_C'] for x in formal['base_Tp'] if x['horizon_s'] == 120)
                diffs = dict(Tp_120=abs(tp[120] - ftp), R0_Tp_120=abs(base_tp[120] - fbtp),
                             **{k: abs(sm['Tp'][k] - formal['coisc_sensitivity'][k]) for k in ('PLE', 'WSR', 'PSM')},
                             **{'R0_' + k: abs(base_sm['Tp'][k] - formal['base_sensitivity'][k]) for k in ('PLE', 'WSR', 'PSM')})
                repro.append(dict(seed=seed, bitwise_equal=all(torch.equal(cur[k], ref[k]) for k in ref), max_abs_param_diff=maxdiff,
                                  metric_abs_diffs=diffs, reproduced=maxdiff == 0 or max(diffs.values()) <= 1e-6))
            print(seed, name, f'{wall:.1f}s', 'Tp_120', round(tp[120], 5), 'PLE', round(sm['Tp']['PLE'], 5),
                  'WSR', round(sm['Tp']['WSR'], 5), 'PSM', round(sm['Tp']['PSM'], 5), flush=True)
    status = 'HARNESS_REPRODUCED' if all(r['reproduced'] for r in repro) else 'HARNESS_NOT_REPRODUCED'
    screen.save_json(OUT / 'ablation_results.json', dict(protocol_sha256=PROTOCOL_SHA, label='POST-HOC mechanism analysis; development only',
                                                         harness_status=status, harness_check=repro, rows=rows))
    flat = []
    for r in rows:
        f = dict(seed=r['seed'], variant=r['variant'], w_pre=r['w_pre'], w_sign=r['w_sign'], train_s=r['train_s'], macro7_MAE=r['macro_MAE'],
                 **{f'Tp_MAE_{h}s': r['Tp_MAE'][h] for h in (10, 30, 60, 90, 120)},
                 **{f'Tp_{k}': r['sens']['Tp'][k] for k in ('PLE', 'WSR', 'PSM', 'PLE_src', 'WSR_src', 'PSM_src')})
        if r['gates']: f.update({f'readout_{k}': v for k, v in r['gates'].items()})
        flat.append(f)
    screen.write_csv(OUT / 'ablation_summary.csv', flat)
    allout = [dict(seed=r['seed'], variant=r['variant'], output=p, **r['sens'][p]) for r in rows for p in POINTS]
    screen.write_csv(OUT / 'ablation_sensitivity_all_outputs.csv', allout)
    print(status, json.dumps(repro, default=str))


if __name__ == '__main__':
    main()
