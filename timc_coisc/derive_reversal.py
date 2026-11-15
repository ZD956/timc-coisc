"""Derive seed-level reversal fractions of the stage-S4 closed-loop runs for Fig. 3(d) and Section 5.5 (post-hoc, descriptive).

  python timc_coisc/derive_reversal.py    reads $TIMC_OUTPUT_ROOT/closedloop/study/raw_runs, writes results/figures_data/derived_reversal_seed.csv

Reversal: at an update whose preceding T_p is more than BAND from 37 C, the applied command moves away from 37 C.
'plant' judges T_p by the plant output before the update (all 60 updates; step 0 uses the initial T_p);
'internal' judges it by the controller's own one-step prediction made at the previous update (updates 2-60).
Counts are pooled within a plant family, then averaged over the four families (family-equal), per seed and variant.
"""
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from _paths import OUTPUT_ROOT, RESULTS
from posthoc_tg import _Run

CL = OUTPUT_ROOT / "closedloop" / "study"
OUT = RESULTS / "figures_data" / "derived_reversal_seed.csv"
BAND = 2.0


def main():
    runs = sorted(glob.glob(str(CL / "raw_runs" / "*.npz")))
    if len(runs) != 8964:
        raise FileNotFoundError(f"derive_reversal.py needs the 8964 private S4 runs (found {len(runs)} in {CL / 'raw_runs'}; set TIMC_OUTPUT_ROOT).")
    acc = defaultdict(lambda: np.zeros(4))
    for f in runs:
        with np.load(f, allow_pickle=False) as z:
            m = _Run(json.loads(str(z["metadata"])))
            data = _Run({k: z[k] for k in z.files if k != "metadata"})
        u = data["applied_UH"].astype(np.float64)
        tp = data["plant_Tp"].astype(np.float64)
        est = data["internal_predicted_Tp"][:, 0].astype(np.float64)
        prev = np.concatenate([[float(m["previous_uh_percent"])], u[:-1]])
        move = np.sign(np.round(u - prev, 9))
        e_pl = 37. - np.concatenate([[float(m["initial_Tp_C"])], tp[:-1]])
        far = np.abs(e_pl) > BAND
        e_in = 37. - est[:-1]
        far_in = np.abs(e_in) > BAND  # NaN (solver failure) compares False
        acc[(m["controller_seed"], m["controller_variant"], m["family"])] += [
            (far & (move == -np.sign(e_pl))).sum(), far.sum(),
            (far_in & (move[1:] == -np.sign(e_in))).sum(), far_in.sum()]
    seeds = sorted({k[0] for k in acc})
    fams = sorted({k[2] for k in acc})
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["seed", "variant", "reversal_plant", "reversal_internal"])
        for s in seeds:
            for v in ("R0", "L0", "COISC"):
                a = [acc[(s, v, f)] for f in fams]
                w.writerow([s, v, float(np.mean([x[0] / x[1] for x in a])), float(np.mean([x[2] / x[3] for x in a]))])
    print("wrote", OUT, len(seeds), "seeds")


if __name__ == "__main__":
    main()
