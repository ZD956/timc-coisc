"""Disturbance-estimating MPC: the unchanged formal MPCController on an UH-shifted
model view, plus scalar moving-horizon estimation of d from T_1..T_6 only."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import minimize_scalar
import torch

from control_mpc import MPCController, MPCConfig, HISTORY_FRAMES, HORIZON, apply_plant_step  # noqa: E402

RAW_PER_PERCENT = 5.0
UH_RAW_MAX = 500.0
UH_HISTORY_COLUMN = 7
UH_FUTURE_COLUMN = 1
MEASURED_OUTPUTS = 6


@dataclass(frozen=True)
class MHEConfig:
    window: int = 12
    rho: float = 0.01
    lower: float = -50.0
    upper: float = 50.0
    xatol: float = 0.05
    grid_step: float = 1.0


def shift_uh_batch(history, future, d_values):
    """Batched shift_uh over a vector of d values; history (1,T,12), future (1,n,6)."""
    offsets = torch.as_tensor(RAW_PER_PERCENT*np.asarray(d_values, dtype=np.float64), dtype=torch.float32).view(-1, 1, 1)
    b = offsets.shape[0]
    hist = history.expand(b, -1, -1)
    fut = future.expand(b, -1, -1)
    c = UH_HISTORY_COLUMN
    f = UH_FUTURE_COLUMN
    h = torch.cat((hist[..., :c], torch.clamp(hist[..., c:c+1] + offsets, 0.0, UH_RAW_MAX), hist[..., c+1:]), dim=-1)
    u = torch.cat((fut[..., :f], torch.clamp(fut[..., f:f+1] + offsets, 0.0, UH_RAW_MAX), fut[..., f+1:]), dim=-1)
    return h, u


def shift_uh(history, future, d_percent):
    """Effective UH = clip(u + d, 0, 100) %, in raw units, on every history and future frame."""
    offset = RAW_PER_PERCENT * float(d_percent)
    c = UH_HISTORY_COLUMN
    f = UH_FUTURE_COLUMN
    h = torch.cat((history[..., :c], torch.clamp(history[..., c:c+1] + offset, 0.0, UH_RAW_MAX), history[..., c+1:]), dim=-1)
    u = torch.cat((future[..., :f], torch.clamp(future[..., f:f+1] + offset, 0.0, UH_RAW_MAX), future[..., f+1:]), dim=-1)
    return h, u


class DisturbedModelView:
    """Presents rollout_native_120s of the internal model evaluated at UH + d."""
    def __init__(self, model):
        if getattr(model, 'training', False):
            raise ValueError('internal model must be in eval mode')
        if hasattr(model, 'parameters') and any(p.requires_grad for p in model.parameters()):
            raise ValueError('internal model parameters must be frozen')
        self.model = model
        self.d_percent = 0.0

    def rollout_native_120s(self, history, future):
        h, u = shift_uh(history, future, self.d_percent)
        return self.model.rollout_native_120s(h, u)


def estimate_d(model, history_s, history_k, n, config=MHEConfig()):
    """Minimise mean squared T_1..T_6 residual over the last n updates, plus rho*(d/100)^2."""
    if not 1 <= n <= min(config.window, HISTORY_FRAMES):
        raise ValueError('estimation window out of range')
    hs = torch.as_tensor(history_s, dtype=torch.float32)
    hk = torch.as_tensor(history_k, dtype=torch.float32)
    if hs.shape != (HISTORY_FRAMES, 12) or hk.shape != (HISTORY_FRAMES, 12):
        raise ValueError('estimator history shape mismatch')
    measured = hk[-n:, :MEASURED_OUTPUTS].double().numpy()
    applied = hk[-n:, 6:].unsqueeze(0)
    base = hs.unsqueeze(0)
    evaluations = 0

    def cost(d):
        nonlocal evaluations
        evaluations += 1
        with torch.no_grad():
            h, u = shift_uh(base, applied, d)
            predicted = model.rollout_native_120s(h, u)[0, :, :MEASURED_OUTPUTS].double().numpy()
        value = float(np.mean((predicted - measured) ** 2) + config.rho * (d / 100.0) ** 2)
        if not np.isfinite(value):
            raise FloatingPointError('nonfinite estimator cost')
        return value

    cost_at_zero = cost(0.0)
    # J(d) is not unimodal (the clip at 0 % UH creates plateaus), so a global grid
    # brackets the minimum before the local bounded Brent refinement.
    grid = np.linspace(config.lower, config.upper, int(round((config.upper-config.lower)/config.grid_step))+1)
    with torch.no_grad():
        h, u = shift_uh_batch(base, applied, grid)
        predicted = model.rollout_native_120s(h, u)[:, :, :MEASURED_OUTPUTS].double().numpy()
    grid_cost = np.mean((predicted - measured[None]) ** 2, axis=(1, 2)) + config.rho * (grid / 100.0) ** 2
    evaluations += len(grid)
    if not np.all(np.isfinite(grid_cost)):
        raise FloatingPointError('nonfinite estimator grid cost')
    g = float(grid[int(np.argmin(grid_cost))])
    grid_best = (cost(g), g)
    result = minimize_scalar(cost, bounds=(max(config.lower, g-config.grid_step), min(config.upper, g+config.grid_step)),
                             method='bounded', options={'xatol': config.xatol})
    refined = (float(result.fun), float(result.x))
    best_cost, best_d = min(grid_best, refined, (cost_at_zero, 0.0), key=lambda c: (c[0], abs(c[1])))
    return dict(d=best_d, cost=best_cost, cost_at_zero=cost_at_zero, grid_best_d=g,
                evaluations=evaluations, success=bool(result.success))


class DisturbanceEstimatingMPC:
    """decide() receives only the plant history, whose temperature columns are output[:6]."""
    def __init__(self, model, mpc_config=MPCConfig(), mhe_config=MHEConfig(), estimate=True):
        self.model = model
        self.view = DisturbedModelView(model)
        self.controller = MPCController(self.view, mpc_config)
        self.config = mpc_config
        self.mhe = mhe_config
        self.estimate = estimate
        self.histories = []
        self.d_hat = 0.0

    def _update_estimate(self, history):
        k = len(self.histories)
        self.histories.append(np.asarray(history, dtype=np.float32).copy())
        info = dict(window_start=None, window_length=0, cost=np.nan, cost_at_zero=np.nan, evaluations=0)
        if not self.estimate:
            info['status'] = 'DISABLED'
        elif k == 0:
            self.d_hat = 0.0
            info['status'] = 'INITIAL_ZERO'
        else:
            s = max(0, k - self.mhe.window)
            info.update(window_start=s, window_length=k - s)
            try:
                est = estimate_d(self.model, self.histories[s], self.histories[k], k - s, self.mhe)
                info.update(cost=est['cost'], cost_at_zero=est['cost_at_zero'], evaluations=est['evaluations'])
                if np.isfinite(est['d']) and est['success']:
                    self.d_hat = est['d']
                    info['status'] = 'OK'
                else:
                    info['status'] = 'NONFINITE_OR_UNSUCCESSFUL_KEEP_PREVIOUS'
            except (FloatingPointError, RuntimeError, ValueError) as exc:
                info['status'] = 'ERROR_KEEP_PREVIOUS: ' + type(exc).__name__ + ': ' + str(exc)
        self.view.d_percent = self.d_hat
        info['d_hat'] = float(self.d_hat)
        return info

    def decide(self, history, reference_c, background_raw, previous_percent):
        started = time.perf_counter()
        info = self._update_estimate(history)
        info['estimator_time_s'] = time.perf_counter() - started
        decision = self.controller.decide(history, reference_c, background_raw, previous_percent)
        decision['estimator'] = info
        return decision


class UHBiasedPlant:
    """Test plant: receives UH + bias on every frame, i.e. exactly the DE disturbance model."""
    def __init__(self, model, bias_percent):
        self.model = model
        self.bias = float(bias_percent)

    def forward_one_step(self, history, future):
        h, u = shift_uh(history, future, self.bias)
        return self.model.forward_one_step(h, u)


class TpPerturbedPlant:
    """Test plant: output[6] replaced by an arbitrary value; output[:6] unchanged."""
    def __init__(self, model, value):
        self.model = model
        self.value = value

    def forward_one_step(self, history, future):
        out = self.model.forward_one_step(history, future).clone()
        out[:, 6] = self.value(out[:, 6]) if callable(self.value) else self.value
        return out


def run_closed_loop_trace(plant, controller, initial_history, reference, background, updates):
    """Same loop as run_formal_control_v3a.run_one, returning per-step records."""
    plant_history = torch.as_tensor(initial_history, dtype=torch.float32).clone()
    previous = float(np.clip(float(plant_history[-1, 7]) / 5., 0., 100.))
    trace = []
    for step in range(updates):
        before_history = plant_history.numpy().copy()
        step_started = time.perf_counter()
        decision = controller.decide(plant_history, reference[step:step+HORIZON], background[step:step+HORIZON], previous)
        plant_history, output = apply_plant_step(plant, plant_history, decision['command_percent'], background[step])
        trace.append(dict(step=step, reference_c=float(reference[step]), plant_tp=float(output[6]),
                          plant_output=output.numpy().copy(), command_percent=decision['command_percent'],
                          success=decision['success'], iterations=decision['iterations'],
                          evaluations=decision['evaluations'], objective=decision['objective'],
                          decision_history=before_history, internal_prediction=decision.get('predicted_trajectory'),
                          optimal=decision.get('optimal_percent'), solver_message=decision['message'],
                          estimator=decision.get('estimator'),
                          time_per_control_step_s=time.perf_counter() - step_started))
        previous = decision['command_percent']
    return trace
