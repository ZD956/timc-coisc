"""Deterministic u_h-to-T_p surrogate-plant MPC; no evaluation data reader."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
import torch

HISTORY_FRAMES = 20
HORIZON = 12
BACKGROUND_INDICES = (0, 2, 3, 4, 5)


@dataclass(frozen=True)
class MPCConfig:
    horizon: int = HORIZON
    sigma_y_c: float = 1.0
    lambda_du: float = 1.0
    lambda_u: float = 0.0
    u_ref_percent: float = 50.0
    maxiter: int = 100
    ftol: float = 1e-6


def quantize_command(value):
    """Nearest 0.1%, with half steps rounded upward on nonnegative commands."""
    return float(np.floor(np.clip(value, 0.0, 100.0) * 10.0 + 0.5) / 10.0)


def warm_start(previous_applied, last_optimal=None, horizon=HORIZON):
    if last_optimal is None:
        return np.full(horizon, float(previous_applied), dtype=np.float64)
    prior = np.asarray(last_optimal, dtype=np.float64)
    if prior.shape != (horizon,):
        raise ValueError('previous optimal sequence has wrong horizon')
    return np.r_[prior[1:], prior[-1]].clip(0.0, 100.0)


def future_power(commands_percent, background_raw):
    commands = torch.as_tensor(commands_percent)
    background = torch.as_tensor(background_raw, dtype=commands.dtype, device=commands.device)
    if commands.ndim != 1 or background.shape != (len(commands), 5):
        raise ValueError('future command/background shape mismatch')
    columns = []
    j = 0
    for heater in range(6):
        if heater == 1:
            columns.append(commands * 5.0)
        else:
            columns.append(background[:, j])
            j += 1
    return torch.stack(columns, dim=1)


def objective(predicted_tp, reference_c, commands_percent, previous_percent, config=MPCConfig()):
    y = torch.as_tensor(predicted_tp)
    u = torch.as_tensor(commands_percent, dtype=y.dtype, device=y.device)
    r = torch.as_tensor(reference_c, dtype=y.dtype, device=y.device)
    if y.shape != (config.horizon,) or u.shape != y.shape or r.shape != y.shape:
        raise ValueError('objective horizon mismatch')
    du = torch.cat((u[:1] - previous_percent, u[1:] - u[:-1])) / 100.0
    return ((y-r)/config.sigma_y_c).square().mean() + config.lambda_du*du.square().mean() + config.lambda_u*((u-config.u_ref_percent)/100.0).square().mean()


class MPCController:
    def __init__(self, model, config=MPCConfig()):
        self.model = model
        if hasattr(model, 'eval'):
            model.eval()
        if hasattr(model, 'parameters'):
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        self.config = config
        self.last_optimal = None

    def decide(self, history, reference_c, background_raw, previous_percent):
        cfg = self.config
        hist = torch.as_tensor(history, dtype=torch.float32)
        reference = np.asarray(reference_c, dtype=np.float64)
        background = np.asarray(background_raw, dtype=np.float64)
        if hist.shape != (HISTORY_FRAMES, 12) or reference.shape != (cfg.horizon,) or background.shape != (cfg.horizon, 5):
            raise ValueError('MPC input shape mismatch')
        if not (np.isfinite(hist.numpy()).all() and np.isfinite(reference).all() and np.isfinite(background).all()):
            raise ValueError('nonfinite MPC input')
        if not 0 <= previous_percent <= 100:
            raise ValueError('previous command out of range')
        x0 = warm_start(previous_percent, self.last_optimal, cfg.horizon)
        evaluations = 0

        def value_gradient(x):
            nonlocal evaluations
            evaluations += 1
            command = torch.tensor(x, dtype=torch.float32, requires_grad=True)
            future = future_power(command, background).unsqueeze(0)
            prediction = self.model.rollout_native_120s(hist.unsqueeze(0), future)[0, :, 6]
            loss = objective(prediction, reference, command, previous_percent, cfg)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite prediction/objective')
            derivative = torch.autograd.grad(loss, command)[0]
            if not torch.isfinite(derivative).all():
                raise FloatingPointError('nonfinite objective gradient')
            return float(loss.detach()), derivative.detach().numpy().astype(np.float64)

        try:
            solution = minimize(value_gradient, x0, jac=True, method='SLSQP',
                                bounds=[(0.0, 100.0)] * cfg.horizon,
                                options={'maxiter': cfg.maxiter, 'ftol': cfg.ftol, 'disp': False})
            valid = bool(solution.success and np.isfinite(solution.x).all() and np.isfinite(solution.fun))
        except (FloatingPointError, RuntimeError, ValueError) as exc:
            solution = None
            valid = False
            failure = type(exc).__name__ + ': ' + str(exc)
        if valid:
            optimal = np.clip(solution.x, 0.0, 100.0).astype(np.float64)
            self.last_optimal = optimal.copy()
            applied = quantize_command(optimal[0])
            with torch.no_grad():
                predicted = self.model.rollout_native_120s(hist.unsqueeze(0),
                            future_power(torch.tensor(optimal,dtype=torch.float32),background).unsqueeze(0))[0].detach().numpy()
            return {'command_percent': applied, 'optimal_percent': optimal, 'success': True,
                    'iterations': int(solution.nit), 'evaluations': evaluations,
                    'objective': float(solution.fun), 'message': str(solution.message),
                    'decision_history':hist.numpy().copy(),'reference_c':reference.copy(),
                    'predicted_trajectory':predicted.copy(),'background_raw':background.copy()}
        return {'command_percent': quantize_command(previous_percent), 'optimal_percent': None,
                'success': False, 'iterations': int(solution.nit) if solution is not None else 0,
                'evaluations': evaluations, 'objective': None,
                'message': str(solution.message) if solution is not None else failure}


class LegacyNativeAdapter:
    """Exact 10 s TorchScript interface used by historical V10 candidates."""
    def __init__(self, scripted_model):
        self.model = scripted_model.eval()

    def eval(self):
        self.model.eval()
        return self

    def forward_one_step(self, history, future):
        return self.model(history, future)

    def rollout_native_120s(self, history, future):
        state = history
        predictions = []
        for k in range(future.shape[1]):
            value = self.forward_one_step(state, future[:, k])
            predictions.append(value)
            state = torch.cat((state[:,1:], torch.cat((value[:,:6],future[:,k]),dim=1)[:,None]),dim=1)
        return torch.stack(predictions,dim=1)


def apply_plant_step(plant, history, command_percent, background_raw):
    """Only an applied action reaches the plant; recursive state uses its own output."""
    state = torch.as_tensor(history, dtype=torch.float32)
    future = future_power(torch.tensor([command_percent]), np.asarray(background_raw)[None, :])[0]
    with torch.no_grad():
        output = plant.forward_one_step(state.unsqueeze(0), future.unsqueeze(0))[0]
    next_frame = torch.cat((output[:6], future))
    return torch.cat((state[1:], next_frame[None]), dim=0), output.detach()


def run_closed_loop(plant, controller, initial_history, reference_c, background_raw, updates):
    """Generic simulation runner; caller must separately authorize real model pair runs."""
    if updates < 1:
        raise ValueError('updates must be positive')
    horizon = controller.config.horizon
    reference = np.asarray(reference_c,dtype=np.float64)
    background = np.asarray(background_raw,dtype=np.float64)
    if reference.shape != (updates+horizon-1,) or background.shape != (updates+horizon-1,5):
        raise ValueError('full reference/background schedule required')
    plant_history=torch.as_tensor(initial_history,dtype=torch.float32).clone()
    if plant_history.shape != (HISTORY_FRAMES,12):
        raise ValueError('initial history shape mismatch')
    previous=float(np.clip(float(plant_history[-1,7])/5.,0.,100.))
    trace=[]
    for k in range(updates):
        decision=controller.decide(plant_history,reference[k:k+horizon],background[k:k+horizon],previous)
        # The plant sees only the applied first action, never the controller's
        # speculative future state or recorded future temperature.
        plant_history,output=apply_plant_step(plant,plant_history,decision['command_percent'],background[k])
        trace.append(dict(step=k,reference_c=float(reference[k]),command_percent=decision['command_percent'],
                          plant_tp=float(output[6]),plant_output=output.numpy().copy(),
                          solver_success=decision['success'],solver_iterations=decision['iterations'],
                          solver_evaluations=decision['evaluations'],
                          internal_prediction=decision.get('predicted_trajectory'),
                          candidate_optimal=decision['optimal_percent']))
        previous=decision['command_percent']
    return trace
