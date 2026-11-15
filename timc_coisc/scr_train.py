"""Training side of the screening study: fresh R0 seeds with the unchanged recipe, plus COISC / COISC_onset / L0 head adaptation.

  scr_train.py gate    G1: retrain seed 20261001 into gates/ and require bitwise equality with the onset-aligned E2 checkpoints
  scr_train.py smoke   code path with throwaway seed 1, 2 steps per stage; checkpoints deleted afterwards
  scr_train.py run     NEW_SEEDS; requires the frozen protocol and EXECUTION_AUTHORIZATION.json
"""
from __future__ import annotations
import argparse
import shutil

import scr_common as sc
import onset_common as oc  # noqa: E402  (path set by scr_common)
import onset_train as ot  # noqa: E402
from onset_common import screen, torch  # noqa: E402
import numpy as np


def train_seed(seed, root, protocol_file, digest, smoke=False):
    """onset_train.run_study e2 branch for one seed; development record kept for R0 only (selection-set secondary)."""
    folder = root / f'seed_{seed}'
    if (folder / 'SEED_COMPLETE.json').exists():
        return sc.read(folder / 'SEED_COMPLETE.json')
    if folder.exists():
        raise FileExistsError(f'incomplete seed folder, inspect before rerun: {folder}')
    device, train, ev, fit_ids, sel_ids, stats, probs, scores, old = SETUP
    threshold = np.asarray(old['regime_thresholds']['regime_deadband_C'])
    folder.mkdir(parents=True); screen.SEED = seed
    shutil.copyfile(protocol_file, folder / 'PROTOCOL.json')
    cfg = screen.configuration(train, ev, fit_ids, sel_ids, stats, probs, scores, old)
    r0, result, run, _ = screen.train_variant('R0', {'excitation': False, 'dbasr': False}, train, ev, sel_ids, stats, probs,
                                              cfg, 0., threshold, device, folder, smoke)
    r0.eval()
    r0_state = oc.cpu_state(r0)
    models = ot.adapt_variants(r0_state, sc.ADAPTED, train, probs, old, sc.LAMBDA_ONSET, seed, device, stats, folder,
                               2 if smoke else oc.ADAPT_STEPS)
    bindings = [dict(seed=seed, variant='R0', path=str(run / 'selected.pt'), sha256=result['checkpoint_sha256'])]
    bindings += [dict(seed=seed, variant=n, path=b['path'], sha256=b['sha256']) for n, (m, b) in models.items()]
    dev = ot.dev_record(r0, ev, sel_ids, threshold, device)
    record = dict(protocol_sha256=digest, seed=seed, bindings=bindings, selected=result['selected'],
                  r0_training_wall_s=result['training_wall_s'],
                  r0_dev=dict(Tp_MAE_120=dev['Tp_MAE'][120], macro7_MAE=dev['macro7_MAE'],
                              Tp_WSR_onset=dev['sens_onset']['Tp']['WSR'], Tp_WSR_prior=dev['sens_prior']['Tp']['WSR'],
                              Tp_PLE_prior=dev['sens_prior']['Tp']['PLE']))
    sc.write(folder / 'SEED_COMPLETE.json', record)
    print(sc.utc(), 'SEED DONE', seed, flush=True)
    return record


def gate():
    out = sc.GATES / 'G1_train'
    if out.exists(): raise FileExistsError(f'preserve existing gate attempt: {out}')
    out.mkdir(parents=True)
    rec = train_seed(sc.REPRO_SEED, out, sc.DRAFT, 'GATE')
    ref = sc.read(sc.ONSET / 'e2' / f'seed_{sc.REPRO_SEED}' / 'SEED_COMPLETE.json')
    ref_bind = {b['variant']: b for b in ref['bindings']}
    rows = []
    for b in rec['bindings']:
        new = oc.load_checkpoint(b['path'])['state_dict']
        old_path = sc.ONSET / 'e2' / f'seed_{sc.REPRO_SEED}' / (('results/R0/selected.pt') if b['variant'] == 'R0' else f'{b["variant"]}_final.pt')
        assert sc.sha(old_path) == ref_bind[b['variant']]['sha256']
        old = oc.load_checkpoint(old_path)['state_dict']
        rows.append(dict(variant=b['variant'], state_dict_bitwise_equal=oc.states_equal(new, old),
                         file_sha_equal=b['sha256'] == ref_bind[b['variant']]['sha256']))
        print('G1', rows[-1], flush=True)
    ref_r0 = next(r for r in ref['rows'] if r['variant'] == 'R0')
    dev_diff = max(abs(rec['r0_dev']['Tp_MAE_120'] - ref_r0['Tp_MAE']['120']),
                   abs(rec['r0_dev']['macro7_MAE'] - ref_r0['macro7_MAE']),
                   abs(rec['r0_dev']['Tp_WSR_onset'] - ref_r0['sens_onset']['Tp']['WSR']),
                   abs(rec['r0_dev']['Tp_WSR_prior'] - ref_r0['sens_prior']['Tp']['WSR']))
    report = dict(label='Screening study gate G1: trainer and adaptation reproduce onset-aligned E2 seed 20261001 bitwise',
                  rows=rows, dev_record_max_abs_diff=dev_diff, r0_training_wall_s=rec['r0_training_wall_s'],
                  pass_=all(r['state_dict_bitwise_equal'] for r in rows) and dev_diff == 0,
                  code_sha256=sc.code_hashes(('scr_common.py', 'scr_train.py')))
    sc.write(sc.PROTO / 'GATE_G1_TRAIN_REPORT.json', report)
    print('G1 PASS', report['pass_'], flush=True)


def smoke():
    if sc.SMOKE.exists(): shutil.rmtree(sc.SMOKE)
    sc.SMOKE.mkdir(parents=True)
    rec = train_seed(sc.SMOKE_SEED, sc.SMOKE, sc.DRAFT, 'SMOKE', smoke=True)
    assert [b['variant'] for b in rec['bindings']] == list(sc.VARIANTS)
    sc.write(sc.SMOKE / 'SMOKE_TRAIN_OK.json', dict(code_sha256=sc.code_hashes(('scr_common.py', 'scr_train.py')), at=sc.utc()))
    print('SMOKE TRAIN OK', flush=True)


def run():
    protocol, digest = sc.frozen_protocol()
    sc.STUDY.mkdir(parents=True, exist_ok=True)
    for seed in protocol['new_seeds']:
        train_seed(seed, sc.STUDY, sc.PROTOCOL, digest)
    sc.write(sc.STUDY / 'TRAIN_COMPLETE.json', dict(protocol_sha256=digest, seeds=protocol['new_seeds'], at=sc.utc()))


SETUP = None


def main():
    global SETUP
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=['gate', 'smoke', 'run']); a = p.parse_args()
    SETUP = ot.setup()
    dict(gate=gate, smoke=smoke, run=run)[a.mode]()


if __name__ == '__main__':
    main()
