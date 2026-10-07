"""Freeze POSTHOC_B5_JACOBIAN_PROTOCOL: requires all gates passed with the current code; runs nothing."""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path

from _paths import PKG as HERE, PROTOCOLS, OUTPUT_ROOT

OUT = OUTPUT_ROOT / 'posthoc_jacobian_b5'
PROTO = PROTOCOLS / 'posthoc_jacobian_b5'
ONSET_OUT = OUTPUT_ROOT / 'posthoc_onset_aligned'
DRAFT = PROTO / 'POSTHOC_B5_JACOBIAN_PROTOCOL_DRAFT.json'
FROZEN = PROTO / 'POSTHOC_B5_JACOBIAN_PROTOCOL.json'
FROZEN_SHA = PROTO / 'POSTHOC_B5_JACOBIAN_PROTOCOL.sha256'
CODE = ('b5_common.py', 'b5_train.py', 'b5_holdout.py', 'b5_freeze.py')
SHARED = (HERE / 'onset_common.py', HERE / 'onset_train.py', HERE / 'onset_holdout.py', HERE / 'screen.py')
REPORTS = ('GATES_TRAIN_REPORT.json', 'GATES_HOLDOUT_REPORT.json')
BOUND = [OUTPUT_ROOT / 'formal_coisc' / f'seed_{s}' / n for s in (20260922, 20260923, 20260924) for n in ('results/R0/selected.pt', 'coisc_final.pt')]
BOUND += [OUTPUT_ROOT / 'posthoc_ablation' / f'seed_{s}' / 'L0_final.pt' for s in (20260922, 20260923, 20260924)]
BOUND += [OUTPUT_ROOT / 'screen' / 'development' / 'calibration.json', ONSET_OUT / 'HOLDOUT_SUMMARY.csv', ONSET_OUT / 'HOLDOUT_CONTRASTS.csv']


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
    reports = {n: read(PROTO / n) for n in REPORTS}
    assert all(r['all_pass'] for r in reports.values()), 'a gate did not pass'
    for name, report in reports.items():
        for code, digest in report['code_sha256'].items():
            assert sha(HERE / code) == digest, f'{code} changed after {name} was produced; rerun that gate'
    smoke = OUT / 'smoke' / 'B5_DEVELOPMENT_RESULTS.json'
    assert smoke.is_file() and smoke.stat().st_mtime > max((HERE / n).stat().st_mtime for n in ('b5_common.py', 'b5_train.py')), \
        'rerun the smoke test with the current code'
    protocol = read(DRAFT)
    lam = reports['GATES_TRAIN_REPORT.json']['R1_calibration']['lambda_jac']
    assert protocol['lambda_jac'] == lam, 'draft lambda_jac differs from the R1 gate value'
    protocol.update(
        version='FROZEN-1', status='FROZEN_BEFORE_B5_OUTCOME', execution_authorized=False,
        execution_authorization_rule=("b5_train run and b5_holdout run require EXECUTION_AUTHORIZATION.json written on the "
                                      "author's explicit instruction, naming this file's SHA-256"),
        gate_summary=dict(R1=reports['GATES_TRAIN_REPORT.json']['R1_calibration']['pass_'],
                          R2=reports['GATES_TRAIN_REPORT.json']['R2_adaptation']['pass_'],
                          R3=reports['GATES_HOLDOUT_REPORT.json']['R3_holdout_reproduction']['pass_'],
                          R4=reports['GATES_HOLDOUT_REPORT.json']['R4_rule_selftest']['pass_'], smoke=True),
        gate_reports_sha256={n: sha(PROTO / n) for n in REPORTS},
        code_sha256={n: sha(HERE / n) for n in CODE},
        shared_code_sha256=[dict(path=str(p), sha256=sha(p)) for p in SHARED],
        bound_files=[dict(path=str(p), sha256=sha(p)) for p in BOUND],
        frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    FROZEN.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding='utf-8')
    digest = sha(FROZEN)
    FROZEN_SHA.write_text(f'{digest}  {FROZEN.name}\n', encoding='utf-8')
    print(json.dumps({'frozen': FROZEN.name, 'sha256': digest}))


if __name__ == '__main__':
    main()
