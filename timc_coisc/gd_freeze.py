"""Freeze GUARD_PROTOCOL: requires gates G1-G2 and smoke passed with the current code; runs nothing."""
from __future__ import annotations
import json

import gd_common as gc
import cl_common as cc  # noqa: E402

REPORTS = ('GATE_G1_MIL_REPORT.json', 'GATE_G2_ANALYSIS_REPORT.json')
BOUND_CODE = ('cl_common.py', 'cl_mil.py', 'cl_analyze.py', 'onset_mil.py', 'onset_analyze_am1.py', 'onset_common.py',
              'control_mpc.py', 'run_formal_control_v3a.py', 'analyze_formal_control_v3.py')
BOUND_CL = ('study/TASKS.json', 'study/CLOSEDLOOP_RESULTS.json', 'study/CLOSEDLOOP_RUN_METRICS.csv', 'study/EXECUTION_AUDIT.json')


def main():
    assert not gc.PROTOCOL.is_file(), 'protocol already frozen'
    assert not gc.STUDY.exists(), 'study directory exists before freeze'
    import onset_mil as om
    reports = {n: gc.read(gc.PROTO / n) for n in REPORTS}
    for name, r in reports.items():
        assert r['pass_'], f'{name} did not pass'
        for code, digest in r['code_sha256'].items():
            assert gc.sha(gc.HERE / code) == digest, f'{code} changed after {name}; rerun that gate'
    for marker in ('SMOKE_MIL_OK.json', 'SMOKE_ANALYSIS_OK.json'):
        for code, digest in gc.read(gc.SMOKE / marker)['code_sha256'].items():
            assert gc.sha(gc.HERE / code) == digest, f'{code} changed after smoke'
    tasks = gc.task_list()
    cl_tasks = cc.read(gc.CL_STUDY / 'TASKS.json')
    wrong = set(gc.wrong_pairs())
    reused = [i for i, t in enumerate(cl_tasks) if (t['seed'], t['scenario_index']) in wrong]
    assert len(reused) == 3 * len(tasks)
    bound = [gc.HERE / n for n in BOUND_CODE] + [cc.PROTO / 'CLOSEDLOOP_PROTOCOL.json'] + [gc.CL / n for n in BOUND_CL]
    bound += [cc.SCREEN_STUDY / 'FIRST_ACTION_CHECK.json']
    bound += [om.FORMAL_PROTOCOL, om.CONTROL_DIR / 'CONTROL_SCENARIO_LOCK_V2.json', om.CONTROL_DIR / 'CONTROL_PLANT_FAMILY_LOCK.json']
    allb = cc.study_bindings()
    bound += [gc.private_path(allb[(s, 'R0')]['path']) for s in sorted({s for s, _ in wrong})]
    bound += [gc.CL_STUDY / 'raw_runs' / f'{i:05d}.npz' for i in reused]
    protocol = gc.read(gc.DRAFT)
    protocol.update(version='FROZEN-1', status='FROZEN_BEFORE_ANY_STUDY_RUN', execution_authorized=False,
                    execution_authorization_rule="gd_mil.py execute and gd_analyze.py analyze require EXECUTION_AUTHORIZATION.json "
                                                 "written on the author's explicit instruction, naming this file's SHA-256",
                    planned_runs=len(tasks), reused_closedloop_runs=len(reused), gate_summary={n: reports[n]['pass_'] for n in REPORTS},
                    gate_reports_sha256={n: gc.sha(gc.PROTO / n) for n in REPORTS}, code_sha256=gc.code_hashes(),
                    bound_files=[dict(path=str(p), sha256=gc.sha(p)) for p in bound], frozen_utc=gc.utc())
    gc.PROTOCOL.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding='utf-8')
    digest = gc.sha(gc.PROTOCOL)
    gc.PROTOCOL_SHA_FILE.write_text(f'{digest}  {gc.PROTOCOL.name}\n', encoding='utf-8')
    print(json.dumps({'frozen': gc.PROTOCOL.name, 'sha256': digest, 'planned_runs': len(tasks), 'bound_files': len(bound)}))


if __name__ == '__main__':
    main()
