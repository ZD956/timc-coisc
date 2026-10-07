"""Outcome-blind gates G1, G2, G3, G5 for DE-MPC. No analysis-population plant is used."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

from de_mpc import (TpPerturbedPlant, DisturbanceEstimatingMPC, UHBiasedPlant, MHEConfig,  # noqa: E402
                    run_closed_loop_trace, shift_uh)
from control_mpc import MPCController, MPCConfig  # noqa: E402
import run_formal_control_v3a as rf  # noqa: E402
from run_posthoc_mil import FORMAL_PROTOCOL, TEST_REPORT, load_controller_models, read, utc  # noqa: E402

TEST_SEED = 20260922
TEST_SCENARIO = 0


def schedule(scenario):
    reference = np.full(71, 37., dtype=np.float64)
    background = np.broadcast_to(np.asarray(scenario['background_raw_aux'], dtype=np.float64), (71, 5)).copy()
    return reference, background


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    return bool(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True))


def test_shift_identity(histories):
    h = torch.as_tensor(histories[0], dtype=torch.float32)[None]
    f = torch.rand(1, 12, 6)*500
    h0, f0 = shift_uh(h, f, 0.0)
    h1, f1 = shift_uh(h, f, 10.0)
    ok = (torch.equal(h0, h) and torch.equal(f0, f)
          and torch.equal(h1[..., [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11]], h[..., [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11]])
          and torch.allclose(f1[..., 1], torch.clamp(f[..., 1]+50., 0., 500.))
          and float(shift_uh(h, torch.full((1, 12, 6), 490.), 10.0)[1][..., 1].max()) == 500.)
    return dict(passed=bool(ok))


def gate_g1(models, scenarios, histories):
    """d fixed at 0: DE decisions bitwise equal to the nominal controller (internal model as its own plant)."""
    rows = []
    for variant in ('R0', 'COISC', 'L0'):
        model = models[(TEST_SEED, variant)]
        for si in (0, 1, 2):
            reference, background = schedule(scenarios[si])
            nominal = run_closed_loop_trace(model, MPCController(model, MPCConfig()), histories[si], reference, background, 3)
            de = run_closed_loop_trace(model, DisturbanceEstimatingMPC(model, estimate=False), histories[si], reference, background, 3)
            for a, b in zip(nominal, de):
                equal = all(same(a[k], b[k]) for k in ('command_percent', 'optimal', 'objective', 'iterations',
                                                        'evaluations', 'internal_prediction', 'plant_output'))
                rows.append(dict(variant=variant, scenario_index=si, step=a['step'], bitwise_equal=equal,
                                 d_hat=b['estimator']['d_hat']))
    return dict(passed=all(r['bitwise_equal'] for r in rows), rows=rows)


def gate_g2(models, scenarios, histories):
    model = models[(TEST_SEED, 'R0')]
    reference, background = schedule(scenarios[TEST_SCENARIO])
    trace = run_closed_loop_trace(model, DisturbanceEstimatingMPC(model), histories[TEST_SCENARIO], reference, background, 12)
    d = [r['estimator']['d_hat'] for r in trace]
    status = [r['estimator']['status'] for r in trace]
    ok = all(abs(x) < 0.1 for x in d) and status[0] == 'INITIAL_ZERO' and all(s == 'OK' for s in status[1:])
    return dict(passed=bool(ok), criterion='|d_hat| < 0.1 % at every update', d_hat=d, status=status,
                applied_UH=[r['command_percent'] for r in trace])


def gate_g3(models, scenarios, histories):
    model = models[(TEST_SEED, 'R0')]
    reference, background = schedule(scenarios[TEST_SCENARIO])
    plant = UHBiasedPlant(model, 10.0)
    trace = run_closed_loop_trace(plant, DisturbanceEstimatingMPC(model), histories[TEST_SCENARIO], reference, background, 25)
    d = [r['estimator']['d_hat'] for r in trace]
    ok = abs(d[24]-10.0) <= 2.0
    return dict(passed=bool(ok), criterion='|d_hat - 10| <= 2 % at update 24 (estimate from updates 12..23)',
                d_hat=d, status=[r['estimator']['status'] for r in trace],
                cost=[r['estimator']['cost'] for r in trace], cost_at_zero=[r['estimator']['cost_at_zero'] for r in trace],
                applied_UH=[r['command_percent'] for r in trace])


def gate_g5(models, scenarios, histories):
    model = models[(TEST_SEED, 'R0')]
    reference, background = schedule(scenarios[TEST_SCENARIO])
    biased = UHBiasedPlant(model, 10.0)
    plants = dict(baseline=biased,
                  plus_1000=TpPerturbedPlant(biased, lambda tp: tp+1000.),
                  nan=TpPerturbedPlant(biased, float('nan')))
    traces = {name: run_closed_loop_trace(p, DisturbanceEstimatingMPC(model), histories[TEST_SCENARIO], reference, background, 6)
              for name, p in plants.items()}
    base = traces['baseline']
    rows = []
    for name in ('plus_1000', 'nan'):
        for a, b in zip(base, traces[name]):
            equal = (all(same(a[k], b[k]) for k in ('command_percent', 'optimal', 'objective', 'internal_prediction'))
                     and same(a['plant_output'][:6], b['plant_output'][:6])
                     and all(same(a['estimator'][k], b['estimator'][k]) for k in ('d_hat', 'cost', 'cost_at_zero', 'evaluations', 'status')))
            perturbed = not same(a['plant_tp'], b['plant_tp'])
            rows.append(dict(perturbation=name, step=a['step'], decisions_and_estimator_bitwise_equal=equal,
                             plant_Tp_actually_perturbed=perturbed))
    nonzero_estimate = any(abs(r['estimator']['d_hat']) > 0.1 for r in base)
    ok = all(r['decisions_and_estimator_bitwise_equal'] and r['plant_Tp_actually_perturbed'] for r in rows) and nonzero_estimate
    return dict(passed=bool(ok), rows=rows, baseline_d_hat=[r['estimator']['d_hat'] for r in base],
                estimator_active_nonzero=nonzero_estimate)


def main():
    torch.set_num_threads(2)
    started = time.perf_counter()
    scenarios, histories = rf.load_scenarios(read(FORMAL_PROTOCOL))
    models = load_controller_models(seeds=(TEST_SEED,))
    report = dict(created_utc=utc(), outcome_blind=True, controller_seed=TEST_SEED, scenario_index=TEST_SCENARIO,
                  mhe_config=MHEConfig().__dict__,
                  plants_used='controller-internal models and wrappers of them only; no analysis-population plant')
    report['shift_identity'] = test_shift_identity(histories)
    for name, fn in (('G1', gate_g1), ('G2', gate_g2), ('G3', gate_g3), ('G5', gate_g5)):
        t = time.perf_counter()
        report[name] = fn(models, scenarios, histories)
        report[name]['seconds'] = time.perf_counter()-t
        print(f'{name}: {"PASS" if report[name]["passed"] else "FAIL"} ({report[name]["seconds"]:.1f} s)', flush=True)
    report['all_passed'] = all(report[k]['passed'] for k in ('shift_identity', 'G1', 'G2', 'G3', 'G5'))
    report['seconds'] = time.perf_counter()-started
    TEST_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                      default=lambda o: None if isinstance(o, float) and math.isnan(o) else str(o)),
                           encoding='utf-8')
    print(json.dumps({k: report[k]['passed'] for k in ('shift_identity', 'G1', 'G2', 'G3', 'G5')}))
    print('G3 d_hat:', [round(x, 3) for x in report['G3']['d_hat']])
    return 0 if report['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
