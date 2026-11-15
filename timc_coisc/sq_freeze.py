"""Freeze SEQ_PROTOCOL: requires gates G1-G6 and all smoke tests passed with the current code; runs nothing."""
from __future__ import annotations
import json

import sq_common as sq
import cl_common as cc  # noqa: E402
import sq_mil  # noqa: E402

REPORTS = ('GATE_G1_G2_TRAIN_REPORT.json', 'GATE_G3_G4_EVAL_REPORT.json', 'GATE_G5_MIL_REPORT.json', 'GATE_G6_ANALYSIS_REPORT.json')
PASS_KEY = {'GATE_G1_G2_TRAIN_REPORT.json': 'all_pass'}
SMOKES = ('SMOKE_TRAIN_OK.json', 'SMOKE_EVAL_OK.json', 'SMOKE_MIL_OK.json', 'SMOKE_ANALYSIS_OK.json')
BOUND_CODE = ('gd_analyze.py', 'gd_common.py', 'cl_common.py', 'cl_mil.py', 'cl_analyze.py', 'scr_common.py', 'scr_holdout.py', 'scr_first_action.py',
              'onset_common.py', 'onset_train.py', 'onset_holdout.py', 'onset_mil.py', 'onset_analyze_am1.py',
              'control_mpc.py', 'run_formal_control_v3a.py', 'analyze_formal_control_v3.py')
BOUND_CL = ('study/TASKS.json', 'study/CLOSEDLOOP_RESULTS.json', 'study/CLOSEDLOOP_RUN_METRICS.csv', 'study/EXECUTION_AUDIT.json')
BOUND_SCREEN = ('study/HOLDOUT_ESTIMATES.json', 'study/FIRST_ACTION_CHECK.json')


def main():
    assert not sq.PROTOCOL.is_file(), 'protocol already frozen'
    assert not sq.STUDY.exists(), 'study directory exists before freeze'
    import onset_mil as om
    reports = {n: sq.read(sq.PROTO / n) for n in REPORTS}
    for name, r in reports.items():
        assert r[PASS_KEY.get(name, 'pass_')], f'{name} did not pass'
        for code, digest in r['code_sha256'].items():
            assert sq.sha(sq.HERE / code) == digest, f'{code} changed after {name}; rerun that gate'
    for marker in SMOKES:
        for code, digest in sq.read(sq.SMOKE / marker)['code_sha256'].items():
            assert sq.sha(sq.HERE / code) == digest, f'{code} changed after smoke {marker}'
    lam = sq.read(sq.LAMBDA_FILE)
    assert lam['lambda_seq'] == reports['GATE_G1_G2_TRAIN_REPORT.json']['lambda_seq']['value']
    tasks = sq_mil.task_list()
    cl_tasks = cc.read(sq.CL_STUDY / 'TASKS.json')
    assert len(cl_tasks) == 3 * len(tasks)
    bind = sq.screening_bindings()
    assert len(bind) == 3 * len(sq.NEW_SEEDS)
    bound = [sq.LAMBDA_FILE, sq.RESULTS / 'figures_data' / 'derived_reversal_seed.csv']
    bound += [sq.HERE / n for n in BOUND_CODE] + [sq.CL_PROTO / 'CLOSEDLOOP_PROTOCOL.json', sq.SCREEN_PROTO / 'SCREENING_PROTOCOL.json']
    bound += [sq.CL / n for n in BOUND_CL] + [sq.SCREEN / n for n in BOUND_SCREEN]
    bound += [om.FORMAL_PROTOCOL, om.CONTROL_DIR / 'CONTROL_SCENARIO_LOCK_V2.json', om.CONTROL_DIR / 'CONTROL_PLANT_FAMILY_LOCK.json']
    bound += [sq.private_path(b['path']) for _, b in sorted(bind.items())]
    bound += [sq.CL_STUDY / 'raw_runs' / f'{i:05d}.npz' for i in range(len(cl_tasks))]
    protocol = sq.read(sq.DRAFT)
    protocol.update(version='FROZEN-1', status='FROZEN_BEFORE_ANY_STUDY_RUN', execution_authorized=False,
                    execution_authorization_rule="sq_train.py run, sq_eval.py holdout/first_action, sq_mil.py execute and "
                                                 "sq_analyze.py analyze require EXECUTION_AUTHORIZATION.json written on the author's "
                                                 "explicit instruction, naming this file's SHA-256",
                    new_seeds=list(sq.NEW_SEEDS), plan_steps=list(sq.SEQ_STEPS), lambda_seq=lam['lambda_seq'],
                    planned_runs=len(tasks), reused_closedloop_runs=len(cl_tasks),
                    gate_summary={n: reports[n][PASS_KEY.get(n, 'pass_')] for n in REPORTS},
                    gate_reports_sha256={n: sq.sha(sq.PROTO / n) for n in REPORTS}, code_sha256=sq.code_hashes(),
                    bound_files=[dict(path=str(p), sha256=sq.sha(p)) for p in bound], frozen_utc=sq.utc())
    sq.PROTOCOL.write_text(json.dumps(protocol, indent=2, ensure_ascii=False), encoding='utf-8')
    digest = sq.sha(sq.PROTOCOL)
    sq.PROTOCOL_SHA_FILE.write_text(f'{digest}  {sq.PROTOCOL.name}\n', encoding='utf-8')
    print(json.dumps({'frozen': sq.PROTOCOL.name, 'sha256': digest, 'planned_runs': len(tasks), 'bound_files': len(bound),
                      'lambda_seq': lam['lambda_seq']}))


if __name__ == '__main__':
    main()
