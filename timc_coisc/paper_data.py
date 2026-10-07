"""Development-only data access, adapters, and verified real-window stitching."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

from _paths import DATA_ROOT as V12B
sys.dont_write_bytecode = True
from model_structural_ablation import POINTS


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def reconstruct(source, ids):
    """Join only consecutive cached windows with exact 31-frame overlap.

    No interpolation, generated targets, or cross-source connections. The cache
    has no timestamps: these are verified relative-time 10 s trajectories.
    Tp is unavailable in the first 19 frames and stays NaN there (never input).
    """
    h, u, y, o = source.history, source.future, source.target, source.origin
    links = (ids[1:] == ids[:-1])
    links &= np.all(h[:-1, 1:] == h[1:, :-1], axis=(1, 2))
    links &= np.all(h[1:, -1, :6] == y[:-1, 0, :6], axis=1)
    links &= np.all(h[1:, -1, 6:] == u[:-1, 0], axis=1)
    links &= np.all(y[:-1, 1:] == y[1:, :-1], axis=(1, 2))
    links &= np.all(u[:-1, 1:] == u[1:, :-1], axis=(1, 2))
    links &= np.all(o[1:] == y[:-1, 0], axis=1)
    cuts = np.r_[0, np.flatnonzero(~links) + 1, len(h)]
    trajectories = []
    for segment, (start, stop) in enumerate(zip(cuts[:-1], cuts[1:])):
        temperature = np.concatenate([np.c_[h[start, :, :6], np.full(20, np.nan)], y[start:stop, 0], y[stop - 1, 1:]], axis=0).astype(np.float32)
        temperature[19] = o[start]
        power = np.concatenate([h[start, :, 6:], u[start:stop, 0], u[stop - 1, 1:]], axis=0)
        trajectories.append(dict(source_id=str(ids[start]), segment_id=segment, temperature=temperature, power=power, window_start=int(start), windows=int(stop-start)))
    return trajectories


def noise_thresholds(trajectories):
    """Robust second-difference noise estimate on steady-control measurements."""
    blocks = []
    for t in trajectories:
        temp, power = t['temperature'][19:], t['power'][19:] / 5
        stable = np.max(np.abs(np.diff(power, axis=0)), axis=1) <= 0.5
        if len(temp) >= 3:
            d2 = np.diff(temp, n=2, axis=0)
            blocks.append(d2[stable[:-1] & stable[1:]])
    values = np.concatenate(blocks)
    assert len(values) >= 100
    sigma = np.median(np.abs(values - np.median(values, axis=0)), axis=0) / (0.67448975 * np.sqrt(6))
    # Endpoint difference has sqrt(2) times single-observation noise.
    threshold = np.maximum(0.02, 3 * np.sqrt(2) * sigma)
    return dict(method='MAD(second differences)/(.67448975*sqrt(6)); stable all-heater changes <=0.5 W; endpoint deadband=max(.02 C,3*sqrt(2)*sigma)',
                fitted_on='91 fit sources only', sigma_C=sigma, regime_deadband_C=threshold,
                response_deadband_C=threshold, eligible_triplets=len(values))


def extract_events(trajectories, split, cfg):
    events = []
    for t in trajectories:
        y, p = t['temperature'], t['power'] / 5
        previous_end = -1
        for k in range(25, len(p) - cfg['response_frames'] + 1):
            delta = float(p[k, 1] - p[k-1, 1])
            if abs(delta) < cfg['min_uh_change_W'] or k < previous_end:
                continue
            pre = p[k-cfg['pre_dwell_frames']:k, 1]
            post = p[k:k+cfg['dwell_frames'], 1]
            if np.ptp(pre) > cfg['hold_tolerance_W'] or np.max(np.abs(post-p[k, 1])) > cfg['hold_tolerance_W']:
                continue
            aux = p[k-cfg['pre_dwell_frames']:k+cfg['response_frames']][:, [0,2,3,4,5]]
            if np.max(np.ptp(aux, axis=0)) > cfg['max_aux_change_W']:
                continue
            n = cfg['response_frames']
            if not np.isfinite(y[k-6:k+n]).all():
                continue
            magnitude = 'small' if abs(delta) < 20 else 'medium' if abs(delta) < 50 else 'large'
            events.append(dict(event_id=f'{split}:{t["segment_id"]}:{k}', split=split, source_id=t['source_id'], segment_id=t['segment_id'], frame_index=k,
                               relative_action_start_s=(k-1)*10, delta_UH_W=delta, direction='increase' if delta > 0 else 'decrease', magnitude=magnitude,
                               history=np.concatenate([y[k-20:k, :6], t['power'][k-20:k]], axis=1),
                               future=t['power'][k:k+n], target=y[k:k+n], origin=y[k-1], pre_temperature=y[k-6:k]))
            previous_end = k+n
    return events


def free_run_windows(trajectories, horizon):
    records = []
    for t in trajectories:
        for k in range(20, len(t['power']) - horizon + 1, horizon):
            records.append(dict(source_id=t['source_id'], segment_id=t['segment_id'], frame_index=k,
                                history=np.concatenate([t['temperature'][k-20:k,:6],t['power'][k-20:k]],axis=1),
                                future=t['power'][k:k+horizon],target=t['temperature'][k:k+horizon]))
    return records
