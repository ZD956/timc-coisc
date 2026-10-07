"""Freeze POSTHOC_ONSET_ALIGNED_PROTOCOL: requires all reproduction gates passed with the current code; runs nothing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

from _paths import PKG as HERE, PROTOCOLS, OUTPUT_ROOT

OUT = OUTPUT_ROOT / 'posthoc_onset_aligned'
PROTO = PROTOCOLS / 'posthoc_onset_aligned'
DRAFT = PROTO / 'POSTHOC_ONSET_ALIGNED_PROTOCOL_DRAFT.json'
FROZEN = PROTO / 'POSTHOC_ONSET_ALIGNED_PROTOCOL.json'
FROZEN_SHA = PROTO / 'POSTHOC_ONSET_ALIGNED_PROTOCOL.sha256'
CODE = ('onset_common.py', 'onset_train.py', 'onset_holdout.py', 'onset_mil.py', 'onset_analyze.py', 'onset_freeze.py')
FORMAL_EXECUTION = OUTPUT_ROOT / 'control' / 'control_v3a_20260926T153411234230Z_0b9ad32c21'
MIL_EXECUTION = OUTPUT_ROOT / 'posthoc_mil' / 'execution_20261001T050813Z_a3bd1ebd55'
BOUND = [HERE / 'screen.py', HERE / 'control_mpc.py', HERE / 'run_formal_control_v3a.py',
         HERE / 'analyze_formal_control_v3.py', PROTOCOLS / 'CONTROL_VALIDATION_PROTOCOL_V3.json',
         PROTOCOLS / 'CONTROL_SCENARIO_LOCK_V2.json', PROTOCOLS / 'CONTROL_PLANT_FAMILY_LOCK.json',
         FORMAL_EXECUTION / 'CONTROL_BOOTSTRAP_DRAWS.npz', FORMAL_EXECUTION / 'CONTROL_PRIMARY_RESULTS.csv',
         PROTOCOLS / 'POSTHOC_MIL_PROTOCOL.json', MIL_EXECUTION / 'TASKS.json',
         MIL_EXECUTION / 'POSTHOC_MIL_PRIMARY_CONTRASTS.csv']
BOUND += [OUTPUT_ROOT / 'formal_coisc' / f'seed_{s}' / name for s in (20260922, 20260923, 20260924)
          for name in ('results/R0/selected.pt', 'coisc_final.pt')]
REPORTS = ('GATES_TRAIN_REPORT.json', 'GATES_HOLDOUT_REPORT.json', 'GATES_MIL_REPORT.json', 'GATES_ANALYSIS_REPORT.json')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    assert not FROZEN.is_file(), 'protocol already frozen'
    reports = {name: read(PROTO / name) for name in REPORTS}
    train, holdout, mil, analysis = (reports[n] for n in REPORTS)
    assert train['all_pass'] and holdout['pass_'] and mil['all_pass'] and analysis['all_pass'], 'a reproduction gate did not pass'
    for name, report in reports.items():
        for code, digest in report['code_sha256'].items():
            assert sha(HERE / code) == digest, f'{code} changed after {name} was produced; rerun that gate'
    smoke = OUT / 'smoke' / 'E2_DEVELOPMENT_RESULTS.json'
    assert smoke.is_file() and smoke.stat().st_mtime > (HERE / 'onset_train.py').stat().st_mtime, 'rerun the smoke test with the current code'
    protocol = read(DRAFT)
    lam = train['R1_calibration']['lambda_onset']
    assert protocol['lambda_onset'] == lam, 'draft lambda_onset differs from the R1 gate value'
    protocol.update(
        version='FROZEN-1 (2026-10-03)', status='FROZEN_BEFORE_ONSET_ALIGNED_OUTCOME', execution_authorized=False,
        execution_authorization_rule=("onset_train e1/e2, onset_holdout run, onset_mil execute and onset_analyze analyze require "
                                      "EXECUTION_AUTHORIZATION.json written on the author's explicit instruction, naming this file's SHA-256"),
        gate_summary=dict(R1=train['R1_calibration']['pass_'], R2=train['R2_adaptation']['pass_'], R3=train['R3_trainer']['pass_'],
                          R4=holdout['pass_'], R5=all(mil[k]['pass_'] for k in ('R5_previous_consistency', 'R5_first_action_reproduction', 'R5_rerun')),
                          R6=analysis['R6_check_formal']['pass_'] and analysis['R6_selftest']['pass_'],
                          R7_preview=mil['R7_guard_identity_preview']['pass_'], guard_unit_test=mil['guard_unit_test']['pass_'],
                          estimated_MIL_wall_hours_6_workers_excluding_guard=mil['timing']['estimated_wall_hours_6_workers']),
        gate_reports_sha256={n: sha(PROTO / n) for n in REPORTS},
        code_sha256={n: sha(HERE / n) for n in CODE},
        bound_files=[dict(path=str(p), sha256=sha(p)) for p in BOUND],
        frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    FROZEN.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding='utf-8')
    digest = sha(FROZEN)
    FROZEN_SHA.write_text(f'{digest}  {FROZEN.name}\n', encoding='utf-8')
    print(json.dumps({'frozen': FROZEN.name, 'sha256': digest}))


if __name__ == '__main__':
    main()
