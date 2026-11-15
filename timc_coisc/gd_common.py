"""Shared pieces for the pre-registered first-move rule baseline study (GUARD_PROTOCOL).

Adds one arm, R0_GUARD (R0 with the frozen first-step direction guard of onset_mil.run_one), on the six R0-wrong
seed-scenario pairs of the closed-loop confirmation study, on all 24 plants. R0/COISC/L0 runs are reused from that
study ($TIMC_OUTPUT_ROOT/closedloop/study, hash-bound); nothing is retrained.
"""
from __future__ import annotations
from _paths import PKG as HERE, PROTOCOLS, OUTPUT_ROOT, VERIFY_HASHES, private_path  # noqa: E402
import cl_common as cc  # noqa: E402

OUT = OUTPUT_ROOT / 'guard'
PROTO = PROTOCOLS / 'guard'
CL = OUTPUT_ROOT / 'closedloop'
DRAFT = PROTO / 'GUARD_PROTOCOL_DRAFT.json'
PROTOCOL = PROTO / 'GUARD_PROTOCOL.json'
PROTOCOL_SHA_FILE = PROTO / 'GUARD_PROTOCOL.sha256'
AUTHORIZATION = OUT / 'EXECUTION_AUTHORIZATION.json'
STUDY = OUT / 'study'
SMOKE = OUT / 'smoke'
GATES = OUT / 'gates'
CL_STUDY = CL / 'study'
CL_METRICS = CL_STUDY / 'CLOSEDLOOP_RUN_METRICS.csv'
CL_RESULTS = CL_STUDY / 'CLOSEDLOOP_RESULTS.json'
ONSET_EXECUTION = cc.ONSET / 'mil_execution_20261003T104638Z_23b8f69525'

GUARD = 'R0_GUARD'
ALL_VARIANTS = ('R0', 'COISC', 'L0', GUARD)
BOOTSTRAP_SEED = 20261112
BOOTSTRAP_DRAWS = 20000
FAR_BAND_C = 2.0
CODE = ('gd_common.py', 'gd_mil.py', 'gd_analyze.py', 'gd_freeze.py')
sha, read, write, utc = cc.sha, cc.read, cc.write, cc.utc


def code_hashes(names=CODE):
    return {n: sha(HERE / n) for n in names}


def wrong_pairs():
    return cc.wrong_pairs(cc.first_actions())


def task_list():
    """R0_GUARD on every R0-wrong pair x all 24 plants, in the order of the closed-loop study's panel A."""
    return [dict(plant_index=pi, scenario_index=si, seed=s, variant=GUARD) for s, si in wrong_pairs() for pi in cc.ALL_PLANTS]


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
