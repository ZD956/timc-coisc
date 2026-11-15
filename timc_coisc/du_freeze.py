"""Freeze DU_PROTOCOL: requires gates G1-G2, smoke tests and the lambda_du calibration done with the current code; runs nothing."""
from __future__ import annotations
import json

import du_common as dc
import cl_common as cc  # noqa: E402
import onset_mil as om  # noqa: E402

REPORTS = ('GATE_G1_MIL_REPORT.json', 'GATE_G2_ANALYSIS_REPORT.json')
SMOKES = ('SMOKE_MIL_OK.json', 'SMOKE_ANALYSIS_OK.json')
BOUND_CODE = ('cl_common.py', 'cl_mil.py', 'cl_analyze.py', 'scr_common.py', 'onset_common.py', 'onset_mil.py', 'onset_analyze_am1.py', 'control_mpc.py')
BOUND_CL = ('study/TASKS.json', 'study/CLOSEDLOOP_RESULTS.json', 'study/CLOSEDLOOP_RUN_METRICS.csv', 'study/EXECUTION_AUDIT.json')
BOUND_SCREEN = ('study/FIRST_ACTION_CHECK.json',)


def main():
    assert not dc.PROTOCOL.is_file(), 'protocol already frozen'
    assert not dc.STUDY.exists(), 'study directory exists before freeze'
    reports = {n: dc.read(dc.PROTO / n) for n in REPORTS}
    for name, r in reports.items():
        assert r['pass_'], f'{name} did not pass'
        for code, digest in r['code_sha256'].items():
            assert dc.sha(dc.HERE / code) == digest, f'{code} changed after {name}; rerun that gate'
    for marker in SMOKES:
        for code, digest in dc.read(dc.SMOKE / marker)['code_sha256'].items():
            assert dc.sha(dc.HERE / code) == digest, f'{code} changed after smoke {marker}'
    lam = dc.read(dc.LAMBDA_FILE)
    for code, digest in lam['code_sha256'].items():
        assert dc.sha(dc.HERE / code) == digest, f'{code} changed after calibration'
    assert lam['lambda_du'] is not None and not any(lam['non_evaluable'].values())
    grid = dc.CALIBRATION / 'grid'
    cal_tasks = dc.read(grid / 'TASKS.json')
    assert all((grid / 'raw_runs' / f'{i:05d}.npz').is_file() for i in range(len(cal_tasks)))
    tasks = dc.task_list()
    cl_tasks = cc.read(dc.CL_STUDY / 'TASKS.json')
    bind = dc.study_bindings()
    assert len(bind) == len(cc.NEW_SEEDS)
    bound = [dc.LAMBDA_FILE, grid / 'TASKS.json', grid / 'BINDINGS.json']
    bound += [grid / 'raw_runs' / f'{i:05d}.npz' for i in range(len(cal_tasks))]
    bound += [dc.HERE / n for n in BOUND_CODE] + [cc.PROTO / 'CLOSEDLOOP_PROTOCOL.json', cc.SCREEN_PROTO / 'SCREENING_PROTOCOL.json']
    bound += [dc.CL / n for n in BOUND_CL] + [cc.SCREEN / n for n in BOUND_SCREEN]
    bound += [om.FORMAL_PROTOCOL, om.CONTROL_DIR / 'CONTROL_SCENARIO_LOCK_V2.json',
              om.CONTROL_DIR / 'CONTROL_PLANT_FAMILY_LOCK.json']
    bound += [dc.private_path(b['path']) for _, b in sorted(bind.items())]
    bound += [dc.CL_STUDY / 'raw_runs' / f'{i:05d}.npz' for i in range(len(cl_tasks))]
    protocol = dc.read(dc.DRAFT)
    protocol.update(version='FROZEN-1', status='FROZEN_BEFORE_ANY_STUDY_RUN', execution_authorized=False,
                    execution_authorization_rule="du_mil.py execute and du_analyze.py analyze require EXECUTION_AUTHORIZATION.json "
                                                 "written on the author's explicit instruction, naming this file's SHA-256",
                    lambda_du=lam['lambda_du'],
                    calibration_result=dict(coisc_tv_target=lam['coisc_tv_target'], r0_tv_by_lambda=lam['r0_tv_by_lambda'],
                                            runs=len(cal_tasks), non_evaluable=lam['non_evaluable']),
                    planned_runs=len(tasks), reused_closedloop_runs=len(cl_tasks),
                    gate_summary={n: reports[n]['pass_'] for n in REPORTS},
                    gate_reports_sha256={n: dc.sha(dc.PROTO / n) for n in REPORTS}, code_sha256=dc.code_hashes(),
                    bound_files=[dict(path=str(p), sha256=dc.sha(p)) for p in bound], frozen_utc=dc.utc())
    dc.PROTOCOL.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding='utf-8')
    digest = dc.sha(dc.PROTOCOL)
    dc.PROTOCOL_SHA_FILE.write_text(f'{digest}  {dc.PROTOCOL.name}\n', encoding='utf-8')
    print(json.dumps({'frozen': dc.PROTOCOL.name, 'sha256': digest, 'planned_runs': len(tasks), 'bound_files': len(bound),
                      'lambda_du': lam['lambda_du']}))


if __name__ == '__main__':
    main()
