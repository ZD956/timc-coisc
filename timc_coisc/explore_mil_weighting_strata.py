"""EXPLORATORY (descriptive, not in any frozen protocol): MIL primary contrast under alternative weightings and reachability strata.

Reuses the frozen 12-scenario cluster draws of the formal control execution. Output contains no source identifiers.
"""
import csv

import numpy as np
import pandas as pd

from _paths import OUTPUT_ROOT

HERE = OUTPUT_ROOT / "posthoc_mechanism"
EXEC = OUTPUT_ROOT / "control" / "control_v3a_20260926T153411234230Z_0b9ad32c21"

d = pd.read_csv(EXEC / "CONTROL_SCENARIO_RESULTS.csv", encoding="utf-8-sig")
d = d[d.panel == "primary"]
draws_npz = np.load(EXEC / "CONTROL_BOOTSTRAP_DRAWS.npz", allow_pickle=True)
draws, scen_ids = draws_npz["source_draws"], list(draws_npz["scenario_ids"])
sidx = {s: i for i, s in enumerate(scen_ids)}
plants = sorted(d.plant.unique())
fam = d.drop_duplicates("plant").set_index("plant").family.to_dict()
seeds = sorted(d.seed.unique())
SEED_NAME = {s: f"Seed{i + 1}" for i, s in enumerate(seeds)}


def estimate(sub, col, weighting):
    """Point and percentile CI of the scenario-cluster bootstrap for one seed and one column."""
    M = np.full((len(plants), len(scen_ids)), np.nan)
    for r in sub.itertuples():
        M[plants.index(r.plant), sidx[r.scenario_id]] = getattr(r, col)
    present = ~np.isnan(M)
    Mz = np.nan_to_num(M)
    counts = np.stack([np.bincount(row, minlength=len(scen_ids)) for row in draws])  # (B, 12)

    def agg(w):  # w: (B, 12) scenario multiplicities
        num, den = w @ Mz.T, w @ present.T  # (B, plants)
        with np.errstate(invalid="ignore", divide="ignore"):
            pm = num / den
        if weighting == "checkpoint_equal":
            return np.nanmean(pm, axis=1)
        fams = sorted(set(fam.values()))
        fm = np.stack([np.nanmean(pm[:, [fam[p] == f for p in plants]], axis=1) for f in fams], axis=1)
        return np.nanmean(fm, axis=1)

    point = agg(np.ones((1, len(scen_ids))))[0]
    boot = agg(counts)
    boot = boot[~np.isnan(boot)]
    return point, np.percentile(boot, 2.5), np.percentile(boot, 97.5), len(draws) - len(boot)


out = []
d["excess_R0"] = d.MAE_R0 - d.MAE_oracle
d["excess_COISC"] = d.MAE_COISC - d.MAE_oracle
d["oracle_minus_R0"] = d.MAE_oracle - d.MAE_R0
for seed in seeds:
    for stratum in ("ALL", "BRACKETED_AT_600S", "BOTH_ABOVE_37_AT_600S", "BOTH_BELOW_37_AT_600S"):
        sub = d[d.seed == seed] if stratum == "ALL" else d[(d.seed == seed) & (d.reachability_stratum == stratum)]
        for weighting in ("family_equal", "checkpoint_equal"):
            for col in ("MAE_R0", "MAE_COISC", "MAE_oracle", "Delta_COISC_minus_R0", "oracle_minus_R0"):
                p, lo, hi, dropped = estimate(sub, col, weighting)
                out.append(dict(seed=SEED_NAME[seed], stratum=stratum, n_pairs=len(sub), weighting=weighting,
                                quantity=col, point=p, ci_lower=lo, ci_upper=hi, empty_replicates_dropped=dropped))

with (HERE / "EXPLORATORY_MIL_WEIGHTING_STRATA.csv").open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
for r in out:
    if r["quantity"] in ("Delta_COISC_minus_R0", "oracle_minus_R0", "MAE_oracle"):
        print(r["seed"], r["stratum"][:12], r["n_pairs"], r["weighting"][:6], r["quantity"][:8],
              f"{r['point']:+.3f} [{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}] drop={r['empty_replicates_dropped']}")
