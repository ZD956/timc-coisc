"""Holdout (delivered-23) inference for the screening study. Must run in its own process (screen.seed_all disables TF32).

  scr_holdout.py gate   G2: existing E2 seed-20261001 models reproduce HOLDOUT_SUMMARY.csv rows exactly
  scr_holdout.py smoke  code path on the smoke checkpoints
  scr_holdout.py run    all study checkpoints; requires the frozen protocol and author authorization
"""
from __future__ import annotations
import argparse
import csv

import scr_common as sc
import onset_common as oc  # noqa: E402
import onset_holdout as oh  # noqa: E402
from onset_common import torch  # noqa: E402

KEYS = ('Tp_MAE_10', 'Tp_MAE_30', 'Tp_MAE_60', 'Tp_MAE_90', 'Tp_MAE_120', 'all7_MAE_120', 'all7_MAE_5h',
        'Tp_WSR_ONSET', 'Tp_WSR_PRIOR', 'Tp_PLE_PRIOR', 'Tp_PSM_ONSET', 'Tp_WSR_ALL_STEPS', 'T6_WSR_ONSET')


def setup():
    raw, stored_s, counts, order, rec, data = oh.load_holdout()
    stats = oc.read_json(oh.NORMALIZATION); torch.set_num_threads(2)
    return data, counts, stats, torch.device('cuda')


def estimate(path, sha256, data, counts, stats, device):
    path = oc.private_path(path)
    assert sc.sha(path) == sha256, f'hash mismatch {path}'
    est, _ = oh.model_estimates(oh.load_model(path, stats, device), data, counts, device)
    return {k: est[k] for k in KEYS}


def gate():
    data, counts, stats, device = setup()
    with (oc.OUT / 'HOLDOUT_SUMMARY.csv').open(encoding='utf-8-sig') as f:
        stored = {(int(r['seed']), r['variant']): r for r in csv.DictReader(f)}
    ref = {b['variant']: b for b in sc.read(oc.OUT / 'e2' / f'seed_{sc.REPRO_SEED}' / 'SEED_COMPLETE.json')['bindings']}
    rows = []
    for v in sc.VARIANTS:
        path = oc.OUT / 'e2' / f'seed_{sc.REPRO_SEED}' / ('results/R0/selected.pt' if v == 'R0' else f'{v}_final.pt')
        est = estimate(path, ref[v]['sha256'], data, counts, stats, device)
        diff = max(abs(est[k] - float(stored[(sc.REPRO_SEED, v)][k])) for k in KEYS)
        rows.append(dict(variant=v, max_abs_diff=diff, pass_=diff <= 1e-12))
        print('G2', rows[-1], flush=True)
    report = dict(label='Screening study gate G2: holdout estimates of existing seed-20261001 models reproduce stored summary',
                  rows=rows, pass_=all(r['pass_'] for r in rows), code_sha256=sc.code_hashes(('scr_common.py', 'scr_holdout.py')))
    sc.write(sc.PROTO / 'GATE_G2_HOLDOUT_REPORT.json', report)
    print('G2 PASS', report['pass_'], flush=True)


def run_root(root, digest):
    data, counts, stats, device = setup()
    out = root / 'HOLDOUT_ESTIMATES.json'
    done = sc.read(out)['rows'] if out.is_file() else []
    have = {(r['seed'], r['variant']) for r in done}
    for record in sc.seed_records(root):
        for b in record['bindings']:
            if b['variant'] != 'R0' or (b['seed'], b['variant']) in have: continue
            done.append(dict(seed=b['seed'], variant=b['variant'], **estimate(b['path'], b['sha256'], data, counts, stats, device)))
            sc.write(out, dict(protocol_sha256=digest, rows=done))
            print(sc.utc(), 'HOLDOUT', b['seed'], b['variant'], flush=True)
    return done


def main():
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['gate', 'smoke', 'run']); a = p.parse_args()
    if a.mode == 'gate':
        gate()
    elif a.mode == 'smoke':
        rows = run_root(sc.SMOKE, 'SMOKE'); assert len(rows) == 1; print('SMOKE HOLDOUT OK', flush=True)
    else:
        protocol, digest = sc.frozen_protocol()
        rows = run_root(sc.STUDY, digest)
        assert len(rows) == len(protocol['new_seeds']), 'holdout incomplete'
        print('HOLDOUT COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()
