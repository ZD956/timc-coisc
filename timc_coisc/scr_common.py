"""Shared pieces for the pre-registered R0 screening study (SCREENING_PROTOCOL).

Reuses the frozen onset-aligned study code (onset_*.py) unchanged; its files are hash-bound in the protocol.
"""
from __future__ import annotations
import hashlib
import json
import sys
import time
from pathlib import Path

from _paths import PKG as HERE, PROTOCOLS, OUTPUT_ROOT, VERIFY_HASHES, private_path  # noqa: E402

OUT = OUTPUT_ROOT / 'screening'
PROTO = PROTOCOLS / 'screening'
ONSET = OUTPUT_ROOT / 'posthoc_onset_aligned'   # outputs of the onset-aligned study (E2 checkpoints, MIL execution)
DRAFT = PROTO / 'SCREENING_PROTOCOL_DRAFT.json'
PROTOCOL = PROTO / 'SCREENING_PROTOCOL.json'
PROTOCOL_SHA_FILE = PROTO / 'SCREENING_PROTOCOL.sha256'
AUTHORIZATION = OUT / 'EXECUTION_AUTHORIZATION.json'
STUDY = OUT / 'study'
SMOKE = OUT / 'smoke'
GATES = OUT / 'gates'

NEW_SEEDS = tuple(range(20261101, 20261141))
E1_SEEDS = (20260922, 20260923, 20260924)
E2_SEEDS = tuple(range(20261001, 20261011))
EXPLORATORY_SEEDS = E1_SEEDS + E2_SEEDS
REPRO_SEED = 20261001
VARIANTS = ('R0', 'COISC', 'COISC_onset', 'L0')
ADAPTED = VARIANTS[1:]
LAMBDA_ONSET = 7485.005103519812
SMOKE_SEED = 1
CODE = ('scr_common.py', 'scr_train.py', 'scr_holdout.py', 'scr_first_action.py', 'scr_analyze.py', 'scr_freeze.py')
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
    """Return (protocol, sha) only if frozen, code unchanged, and authorized by the author for that exact hash."""
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


def seed_records(root):
    """SEED_COMPLETE records of a study root, in seed order."""
    return [read(p) for p in sorted(Path(root).glob('seed_*/SEED_COMPLETE.json'))]
