"""Freeze CLOSEDLOOP_PROTOCOL: requires gates G1-G2 and smoke passed with the current code; runs nothing."""
from __future__ import annotations
import json

import cl_common as cc

REPORTS = ('GATE_G1_MIL_REPORT.json', 'GATE_G2_ANALYSIS_REPORT.json')
BOUND_CODE = ('onset_mil.py', 'onset_analyze_am1.py', 'onset_common.py', 'control_mpc.py', 'run_formal_control_v3a.py',
              'analyze_formal_control_v3.py')
BOUND_ONSET = ('mil_execution_20261003T104638Z_23b8f69525/TASKS.json',
               'mil_execution_20261003T104638Z_23b8f69525/ONSET_MIL_SECONDARY.csv')
BOUND_SCREEN = ('study/FIRST_ACTION_CHECK.json', 'study/SCREENING_RESULTS.json')


def main():
    assert not cc.PROTOCOL.is_file(), 'protocol already frozen'
    assert not cc.STUDY.exists(), 'study directory exists before freeze'
    import onset_mil as om
    reports = {n: cc.read(cc.PROTO / n) for n in REPORTS}
    for name, r in reports.items():
        assert r['pass_'], f'{name} did not pass'
        for code, digest in r['code_sha256'].items():
            assert cc.sha(cc.HERE / code) == digest, f'{code} changed after {name}; rerun that gate'
    for marker in ('SMOKE_MIL_OK.json', 'SMOKE_ANALYSIS_OK.json'):
        for code, digest in cc.read(cc.SMOKE / marker)['code_sha256'].items():
            assert cc.sha(cc.HERE / code) == digest, f'{code} changed after smoke'
    bindings = cc.study_bindings()
    assert len(bindings) == len(cc.NEW_SEEDS) * len(cc.VARIANTS)
    bound = [cc.HERE / n for n in BOUND_CODE] + [cc.ONSET / n for n in BOUND_ONSET]
    bound += [cc.SCREEN_PROTO / 'SCREENING_PROTOCOL.json'] + [cc.SCREEN / n for n in BOUND_SCREEN]
    bound += [om.FORMAL_PROTOCOL, om.CONTROL_DIR / 'CONTROL_SCENARIO_LOCK_V2.json', om.CONTROL_DIR / 'CONTROL_PLANT_FAMILY_LOCK.json']
    bound += [b['path'] for _, b in sorted(bindings.items())]
    tasks = cc.task_list(cc.first_actions())
    protocol = cc.read(cc.DRAFT)
    protocol.update(version='FROZEN-1', status='FROZEN_BEFORE_ANY_STUDY_RUN', execution_authorized=False,
                    execution_authorization_rule="cl_mil.py execute and cl_analyze.py analyze require EXECUTION_AUTHORIZATION.json "
                                                 "written on the author's explicit instruction, naming this file's SHA-256",
                    planned_runs=len(tasks), gate_summary={n: reports[n]['pass_'] for n in REPORTS},
                    gate_reports_sha256={n: cc.sha(cc.PROTO / n) for n in REPORTS}, code_sha256=cc.code_hashes(),
                    bound_files=[dict(path=str(p), sha256=cc.sha(p)) for p in bound], frozen_utc=cc.utc())
    cc.PROTOCOL.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding='utf-8')
    digest = cc.sha(cc.PROTOCOL)
    cc.PROTOCOL_SHA_FILE.write_text(f'{digest}  {cc.PROTOCOL.name}\n', encoding='utf-8')
    print(json.dumps({'frozen': cc.PROTOCOL.name, 'sha256': digest, 'planned_runs': len(tasks)}))


if __name__ == '__main__':
    main()
