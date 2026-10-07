"""Repository-relative locations; private data and outputs are configured by environment variables."""
import os
import re
from pathlib import Path

PKG = Path(__file__).resolve().parent
REPO = PKG.parent
PROTOCOLS = REPO / 'protocols'
CONFIGS = REPO / 'configs'
RESULTS = REPO / 'results'
DATA_ROOT = Path(os.environ.get('TIMC_DATA_ROOT', str(REPO / 'private_data')))
OUTPUT_ROOT = Path(os.environ.get('TIMC_OUTPUT_ROOT', str(REPO / 'outputs')))

LEGACY_TOKENS = {'c8': 'tp', 'h2': 'uh'}
_LEGACY = re.compile(r'(?<![A-Za-z0-9])(c8|h2)(?![0-9A-Za-z])')


def modern_key(key):
    return _LEGACY.sub(lambda m: LEGACY_TOKENS[m.group(1)], key)


def modernize_payload(payload):
    """Map checkpoints written before the public renaming onto the released key names."""
    if isinstance(payload, dict):
        if isinstance(payload.get('stats'), dict):
            payload['stats'] = {modern_key(k): v for k, v in payload['stats'].items()}
        if isinstance(payload.get('state_dict'), dict):
            payload['state_dict'] = {modern_key(k): v for k, v in payload['state_dict'].items()}
    return payload


def load_checkpoint(path, map_location='cpu', weights_only=False):
    import torch
    return modernize_payload(torch.load(path, map_location=map_location, weights_only=weights_only))


VERIFY_HASHES = os.environ.get('TIMC_VERIFY_HASHES') == '1'


def private_path(path):
    """Protocols record '<private>/x' under TIMC_DATA_ROOT, '<outputs>/x' under TIMC_OUTPUT_ROOT, and released files repo-relative."""
    text = str(path).replace('\\', '/')
    for prefix, root in (('<private>/', DATA_ROOT), ('<outputs>/', OUTPUT_ROOT)):
        if text.startswith(prefix):
            return root / text[len(prefix):]
    if text.startswith(('protocols/', 'timc_coisc/', 'configs/', 'results/')):
        return REPO / text
    return Path(path)


def require_private(path, what):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'{what} requires the private development data, which is not released '
                                f'(expected at {path}; set TIMC_DATA_ROOT).')
    return path


_PUBLIC = re.compile(r'^S\d{3}$')
_PSEUDONYMS = None


def _pseudonyms():
    """Private source names -> S001..S139, assigned in sorted order of the locked manifest."""
    global _PSEUDONYMS
    if _PSEUDONYMS is None:
        import csv
        table = DATA_ROOT / 'source_pseudonyms.csv'
        if table.exists():
            with table.open(encoding='utf-8-sig', newline='') as f:
                _PSEUDONYMS = {r['source_id']: r['public_id'] for r in csv.DictReader(f)}
        else:
            manifest = require_private(DATA_ROOT / 'data' / '01_locked_source_manifest.csv', 'source pseudonymisation')
            with manifest.open(encoding='utf-8-sig', newline='') as f:
                ids = sorted({r['source_id'] for r in csv.DictReader(f)})
            _PSEUDONYMS = {s: f'S{i + 1:03d}' for i, s in enumerate(ids)}
    return _PSEUDONYMS


def public_source_id(source):
    source = str(source)
    return source if _PUBLIC.match(source) else _pseudonyms()[source]


def public_ids(values):
    import numpy as np
    return np.array([public_source_id(v) for v in np.asarray(values).astype(str).tolist()])
