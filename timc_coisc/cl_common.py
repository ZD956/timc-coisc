"""Shared pieces for the pre-registered closed-loop confirmation study (CLOSEDLOOP_PROTOCOL).

Reuses the 40 screening-study checkpoints ($TIMC_OUTPUT_ROOT/screening/study) and the frozen onset-aligned MIL runner
(onset_mil.run_one) unchanged; both are hash-bound in the protocol.
"""
from __future__ import annotations
import hashlib
import json
import sys
import time
from pathlib import Path

from _paths import PKG as HERE, PROTOCOLS, OUTPUT_ROOT, VERIFY_HASHES, private_path  # noqa: E402

OUT = OUTPUT_ROOT / 'closedloop'
PROTO = PROTOCOLS / 'closedloop'
SCREEN = OUTPUT_ROOT / 'screening'
SCREEN_PROTO = PROTOCOLS / 'screening'
SCREEN_STUDY = SCREEN / 'study'
ONSET = OUTPUT_ROOT / 'posthoc_onset_aligned'   # outputs of the onset-aligned study (E2 checkpoints, MIL execution)

DRAFT = PROTO / 'CLOSEDLOOP_PROTOCOL_DRAFT.json'
PROTOCOL = PROTO / 'CLOSEDLOOP_PROTOCOL.json'
PROTOCOL_SHA_FILE = PROTO / 'CLOSEDLOOP_PROTOCOL.sha256'
AUTHORIZATION = OUT / 'EXECUTION_AUTHORIZATION.json'
STUDY = OUT / 'study'
SMOKE = OUT / 'smoke'
GATES = OUT / 'gates'

NEW_SEEDS = tuple(range(20261101, 20261141))
VARIANTS = ('R0', 'COISC', 'L0')
PANEL_B_PLANTS = (0, 5, 11, 8, 20, 23)  # F1: P, A1, A3; F2: A2; F3: B2; F4: B1 (fixed before any run, by model id only)
ALL_PLANTS = tuple(range(24))
FAMILIES = ('F1', 'F2', 'F3', 'F4')
NI_MARGIN_C = 0.10
BOOTSTRAP_SEED = 20261105
BOOTSTRAP_DRAWS = 20000
REPRO_SEED = 20261001
SMOKE_SEED = 1
CODE = ('cl_common.py', 'cl_mil.py', 'cl_analyze.py', 'cl_freeze.py')
PYTHON = sys.executable


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float), encoding='utf-8')


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def code_hashes(names=CODE):
    return {n: sha(HERE / n) for n in names}


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


def study_bindings(root=SCREEN_STUDY):
    """{(seed, variant): binding} for R0/COISC/L0 from the screening study SEED_COMPLETE records."""
    out = {}
    for p in sorted(Path(root).glob('seed_*/SEED_COMPLETE.json')):
        for b in read(p)['bindings']:
            if b['variant'] in VARIANTS:
                out[(int(b['seed']), b['variant'])] = dict(path=b['path'], sha256=b['sha256'])
    return out


def first_actions(path=SCREEN_STUDY / 'FIRST_ACTION_CHECK.json'):
    """{(seed, variant, scenario_index): row} of the stored plant-free first decisions."""
    return {(int(r['seed']), r['variant'], int(r['scenario_index'])): r for r in read(path)['rows']}


def wrong_pairs(fa, seeds=NEW_SEEDS):
    """(seed, scenario_index) pairs where the R0 first move is opposite to the side of 37 C."""
    return sorted((s, si) for (s, v, si), r in fa.items() if v == 'R0' and s in seeds and r['first_action'] == 'wrong')


def task_list(fa, seeds=NEW_SEEDS):
    """Panel B: all seeds x 12 scenarios x PANEL_B_PLANTS; panel A adds the other 18 plants on the R0-wrong pairs."""
    tasks = []
    for pi in PANEL_B_PLANTS:
        for s in seeds:
            for si in range(12):
                tasks += [dict(plant_index=pi, scenario_index=si, seed=s, variant=v) for v in VARIANTS]
    for s, si in wrong_pairs(fa, seeds):
        for pi in ALL_PLANTS:
            if pi not in PANEL_B_PLANTS:
                tasks += [dict(plant_index=pi, scenario_index=si, seed=s, variant=v) for v in VARIANTS]
    return tasks
