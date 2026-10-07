"""EXPLORATORY (not in frozen protocol; defined after M1-M3 were seen): MAE contrasts split by whether R0's first action has the wrong sign."""
import csv

from _paths import OUTPUT_ROOT

HERE = OUTPUT_ROOT/'posthoc_mechanism'
rows = list(csv.DictReader((HERE/'MECH_RUN_METRICS.csv').open(encoding='utf-8-sig')))
runs = {(r['plant'], r['scenario'], r['seed'], r['variant']): r for r in rows}
out = []
for seed in ('Seed1', 'Seed2', 'Seed3'):
    for subset in ('ALL', 'BRACKETED_start_below'):
        groups = {}
        for r in rows:
            if r['seed'] != seed or r['variant'] != 'R0':
                continue
            if subset != 'ALL' and not (r['stratum'] == 'BRACKETED_AT_600S' and r['initial_side'] == 'start_below'):
                continue
            k = (r['plant'], r['scenario'])
            want = 1 if r['initial_side'] == 'start_below' else -1
            sign = int(r['first_action_sign'])
            cls = 'R0_first_action_wrong_sign' if sign == -want else ('R0_first_action_correct' if sign == want else 'R0_first_action_zero')
            c, l = runs[k+(seed, 'COISC')], runs[k+(seed, 'L0')]
            groups.setdefault(cls, []).append((float(c['MAE_all'])-float(r['MAE_all']), float(l['MAE_all'])-float(r['MAE_all']),
                                               int(c['first_action_sign']) == want, int(l['first_action_sign']) == want))
        for cls, v in sorted(groups.items()):
            n = len(v)
            out.append(dict(seed=seed, subset=subset, R0_class=cls, n_pairs=n,
                            mean_dMAE_COISC_minus_R0=sum(x[0] for x in v)/n, mean_dMAE_L0_minus_R0=sum(x[1] for x in v)/n,
                            COISC_first_action_correct=sum(x[2] for x in v), L0_first_action_correct=sum(x[3] for x in v)))
with (HERE/'EXPLORATORY_FIRST_ACTION_SPLIT.csv').open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
for r in out:
    print(r)
