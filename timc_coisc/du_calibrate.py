"""Pre-freeze calibration of lambda_du on development seeds 1-3 (formal R0/COISC checkpoints; never part of any study population).

  du_calibrate.py run [WORKERS]    panel-B plants x 12 scenarios x seeds 1-3: COISC at lambda_du = 1 and R0 at every GRID value
  du_calibrate.py worker DIR I N
  du_calibrate.py decide           applies the rule and writes LAMBDA_DU.json

Rule (fixed before any calibration run): lambda_du is the smallest grid value at which the mean total variation of the R0 command
over the calibration runs is at most that of COISC at lambda_du = 1. If no GRID value qualifies, GRID_EXTENSION is run and searched
in the same way. Matching from above favours the baseline: R0_DU is at least as smooth as COISC on the development seeds.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

import du_common as dc
import cl_common as cc  # noqa: E402
import du_mil  # noqa: E402
import onset_mil as om  # noqa: E402
import onset_analyze_am1 as oa  # noqa: E402


def tasks_for(grid, with_coisc):
    out = []
    for pi in cc.PANEL_B_PLANTS:
        for s in dc.DEV_SEEDS:
            for si in range(12):
                if with_coisc:
                    out.append(dict(plant_index=pi, scenario_index=si, seed=s, variant='COISC', lambda_du=1.0))
                out += [dict(plant_index=pi, scenario_index=si, seed=s, variant=dc.ARM, lambda_du=g) for g in grid]
    return out


def run(workers, extension=False):
    root = dc.CALIBRATION / ('extension' if extension else 'grid')
    tasks = tasks_for(dc.GRID_EXTENSION if extension else dc.GRID, not extension)
    if (root / 'TASKS.json').is_file():
        assert dc.read(root / 'TASKS.json') == tasks, 'calibration task list changed'
    codes, done = du_mil.launch(root, tasks, dc.dev_bindings(), workers)
    print(dict(root=str(root), codes=codes, completed=done, total=len(tasks)), flush=True)
    assert not any(codes) and done == len(tasks)


def total_variation(root):
    plants, _ = om.plants_and_formal_index()
    tasks = dc.read(Path(root) / 'TASKS.json')
    tv = {}
    for i, t in enumerate(tasks):
        meta, data = oa.load_run(Path(root) / 'raw_runs' / f'{i:05d}.npz')
        assert (meta['plant_index'], meta['scenario_index'], meta['controller_seed'], meta['controller_variant']) == \
            (t['plant_index'], t['scenario_index'], t['seed'], t['variant'])
        row = oa.metric_row(meta, data, t['variant'], t['plant_index'], t['scenario_index'], t['seed'], plants[t['plant_index']]['family'])
        key = ('COISC', 1.0) if t['variant'] == 'COISC' else (dc.ARM, t['lambda_du'])
        tv.setdefault(key, []).append(np.nan if row is None else row['total_variation_u'])
    return {k: float(np.nanmean(v)) for k, v in tv.items()}, {k: int(np.isnan(v).sum()) for k, v in tv.items()}


def decide():
    grid_tv, grid_missing = total_variation(dc.CALIBRATION / 'grid')
    target = grid_tv[('COISC', 1.0)]
    table = {g: grid_tv[(dc.ARM, g)] for g in dc.GRID}
    missing = dict(grid_missing)
    chosen = next((g for g in dc.GRID if table[g] <= target), None)
    if chosen is None and (dc.CALIBRATION / 'extension' / 'TASKS.json').is_file():
        ext_tv, ext_missing = total_variation(dc.CALIBRATION / 'extension')
        table.update({g: ext_tv[(dc.ARM, g)] for g in dc.GRID_EXTENSION}); missing.update(ext_missing)
        chosen = next((g for g in dc.GRID_EXTENSION if table[g] <= target), None)
    record = dict(rule=__doc__.split('Rule')[1].strip(), dev_seeds=list(dc.DEV_SEEDS), plants=list(cc.PANEL_B_PLANTS),
                  coisc_tv_target=target, r0_tv_by_lambda={str(k): v for k, v in table.items()},
                  non_evaluable={f'{k[0]}@{k[1]}': v for k, v in missing.items()}, lambda_du=chosen,
                  code_sha256=dc.code_hashes(('du_common.py', 'du_mil.py', 'du_calibrate.py')), at=dc.utc())
    dc.write(dc.LAMBDA_FILE, record)
    print('LAMBDA_DU', chosen, 'target', round(target, 1), {k: round(v, 1) for k, v in table.items()}, flush=True)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'run':
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 6)
    elif mode == 'run_extension':
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 6, extension=True)
    elif mode == 'worker':
        du_mil.worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    elif mode == 'decide':
        decide()
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
