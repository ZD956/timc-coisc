"""Freeze SCREENING_PROTOCOL: requires gates G1-G4 and smoke passed with the current code; runs nothing."""
from __future__ import annotations
import json

import scr_common as sc

REPORTS = ('GATE_G1_TRAIN_REPORT.json', 'GATE_G2_HOLDOUT_REPORT.json', 'GATE_G3_FIRST_ACTION_REPORT.json', 'GATE_G4_ANALYSIS_REPORT.json')
BOUND_CODE = ('onset_common.py', 'onset_train.py', 'onset_holdout.py', 'onset_mil.py', 'screen.py', 'control_mpc.py',
              'run_formal_control_v3a.py')
BOUND_ONSET = ('HOLDOUT_SUMMARY.csv', 'mil_execution_20261003T104638Z_23b8f69525/FIRST_ACTION_CHECK.csv',
               'e1/E1_DEVELOPMENT_RESULTS.json', 'e2/E2_DEVELOPMENT_RESULTS.json')


def main():
    assert not sc.PROTOCOL.is_file(), 'protocol already frozen'
    import onset_common as oc
    import onset_mil as om
    reports = {n: sc.read(sc.PROTO / n) for n in REPORTS}
    for name, r in reports.items():
        assert r['pass_'], f'{name} did not pass'
        for code, digest in r['code_sha256'].items():
            assert sc.sha(sc.HERE / code) == digest, f'{code} changed after {name}; rerun that gate'
    smoke_marker = sc.SMOKE / 'SMOKE_TRAIN_OK.json'
    assert smoke_marker.is_file() and (sc.SMOKE / 'HOLDOUT_ESTIMATES.json').is_file() and (sc.SMOKE / 'first_action').is_dir(), 'run all smoke steps'
    for code, digest in sc.read(smoke_marker)['code_sha256'].items():
        assert sc.sha(sc.HERE / code) == digest, f'{code} changed after smoke'
    bound = [sc.HERE / n for n in BOUND_CODE] + [oc.PROTOCOL] + [sc.ONSET / n for n in BOUND_ONSET]
    bound += [om.FORMAL_PROTOCOL, om.CONTROL_DIR / 'CONTROL_SCENARIO_LOCK_V2.json']
    protocol = sc.read(sc.DRAFT)
    assert list(protocol['new_seeds']) == list(sc.NEW_SEEDS)
    protocol.update(version='FROZEN-1', status='FROZEN_BEFORE_ANY_STUDY_SEED_TRAINED', execution_authorized=False,
                    execution_authorization_rule='scr_train run, scr_holdout run, scr_first_action run and scr_analyze analyze require '
                                                 "EXECUTION_AUTHORIZATION.json written on the author's explicit instruction, naming this file's SHA-256",
                    gate_summary={n: reports[n]['pass_'] for n in REPORTS}, gate_reports_sha256={n: sc.sha(sc.PROTO / n) for n in REPORTS},
                    code_sha256=sc.code_hashes(), bound_files=[dict(path=str(p), sha256=sc.sha(p)) for p in bound], frozen_utc=sc.utc())
    assert not sc.STUDY.exists(), 'study directory exists before freeze'
    sc.PROTOCOL.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding='utf-8')
    digest = sc.sha(sc.PROTOCOL)
    sc.PROTOCOL_SHA_FILE.write_text(f'{digest}  {sc.PROTOCOL.name}\n', encoding='utf-8')
    print(json.dumps({'frozen': sc.PROTOCOL.name, 'sha256': digest}))


if __name__ == '__main__':
    main()
