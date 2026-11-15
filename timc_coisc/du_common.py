"""Shared pieces for the pre-registered move-penalty baseline study (DU_PROTOCOL).

Adds one arm, R0_DU: the R0 checkpoint of each screening seed inside the unchanged MPC with a larger move weight lambda_du
(MPCConfig.lambda_du; default 1.0). lambda_du is fixed before freeze on the development seeds 1-3 (formal R0/COISC checkpoints)
so that the R0 command total variation matches that of COISC on the same panel-B bench. The arm runs on panel B of the stage-S4
bench; R0/COISC/L0 runs are reused from $TIMC_OUTPUT_ROOT/closedloop/study (hash-bound). onset_mil.run_one is
unchanged: the worker replaces the module-level MPCConfig factory that run_one calls, and gate G1 shows that the replacement
with lambda_du = 1.0 reproduces stored S4 runs bitwise.
"""
from __future__ import annotations
import functools

from _paths import PKG as HERE, PROTOCOLS, OUTPUT_ROOT, VERIFY_HASHES, private_path  # noqa: E402
import cl_common as cc  # noqa: E402
import onset_mil as om  # noqa: E402
from control_mpc import MPCConfig  # noqa: E402

OUT = OUTPUT_ROOT / 'move_penalty'
PROTO = PROTOCOLS / 'move_penalty'
CL = OUTPUT_ROOT / 'closedloop'
CL_STUDY = CL / 'study'
DRAFT = PROTO / 'DU_PROTOCOL_DRAFT.json'
PROTOCOL = PROTO / 'DU_PROTOCOL.json'
PROTOCOL_SHA_FILE = PROTO / 'DU_PROTOCOL.sha256'
AUTHORIZATION = OUT / 'EXECUTION_AUTHORIZATION.json'
CALIBRATION = OUT / 'calibration'
LAMBDA_FILE = PROTO / 'LAMBDA_DU.json'
STUDY = OUT / 'study'
SMOKE = OUT / 'smoke'
GATES = OUT / 'gates'

ARM = 'R0_DU'
ALL_VARIANTS = ('R0', 'COISC', 'L0', ARM)
DEV_SEEDS = (20260922, 20260923, 20260924)
FORMAL = OUTPUT_ROOT / 'formal_coisc'
GRID = (1.0, 2.0, 4.0, 8.0, 16.0)
GRID_EXTENSION = (32.0, 64.0, 128.0, 256.0)
BOOTSTRAP_SEED = 20261114
BOOTSTRAP_DRAWS = 20000
CODE = ('du_common.py', 'du_calibrate.py', 'du_mil.py', 'du_analyze.py', 'du_freeze.py')
sha, read, write, utc = cc.sha, cc.read, cc.write, cc.utc


def code_hashes(names=CODE):
    return {n: sha(HERE / n) for n in names}


def use_lambda_du(value):
    """Make onset_mil.run_one build its controller with lambda_du = value (all other MPCConfig fields at their defaults)."""
    om.MPCConfig = functools.partial(MPCConfig, lambda_du=float(value))


def task_lambda(task, lam):
    return lam if task['variant'] == ARM else 1.0


def dev_bindings():
    """Formal R0 and COISC checkpoints of development seeds 1-3, with R0 also bound under the arm name."""
    out = {}
    for s in DEV_SEEDS:
        r0 = FORMAL / f'seed_{s}' / 'results' / 'R0' / 'selected.pt'
        co = FORMAL / f'seed_{s}' / 'coisc_final.pt'
        out[(s, 'R0')] = dict(path=str(r0), sha256=sha(r0))
        out[(s, ARM)] = dict(path=str(r0), sha256=sha(r0))
        out[(s, 'COISC')] = dict(path=str(co), sha256=sha(co))
    return out


def study_bindings():
    """R0 checkpoints of the 40 screening seeds bound under the arm name."""
    allb = cc.study_bindings()
    return {(s, ARM): allb[(s, 'R0')] for s in cc.NEW_SEEDS}


def task_list():
    """R0_DU on panel B of the S4 bench, in S4 order: 6 plants x 40 seeds x 12 scenarios = 2880 runs."""
    return [dict(plant_index=pi, scenario_index=si, seed=s, variant=ARM)
            for pi in cc.PANEL_B_PLANTS for s in cc.NEW_SEEDS for si in range(12)]


def frozen_protocol():
    """Return (protocol, sha) only if frozen, code and bound files unchanged, and authorized for that exact hash."""
    digest = sha(PROTOCOL)
    assert not VERIFY_HASHES or PROTOCOL_SHA_FILE.read_text(encoding='utf-8').split()[0] == digest, 'protocol changed after freeze'
    protocol = read(PROTOCOL)
    assert protocol['status'].startswith('FROZEN'), 'protocol not frozen'
    for name, h in protocol['code_sha256'].items():
        assert not VERIFY_HASHES or sha(HERE / name) == h, f'code changed after freeze: {name}'
    for b in protocol['bound_files']:
        assert not VERIFY_HASHES or sha(private_path(b['path'])) == b['sha256'], f'bound file changed after freeze: {b["path"]}'
    auth = read(AUTHORIZATION)
    assert auth.get('authorized_by_author') is True and auth.get('protocol_sha256') == digest, 'no valid author authorization'
    return protocol, digest
