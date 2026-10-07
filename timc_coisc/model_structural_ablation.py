"""Three-scale LSTM with native one-step deployment and G4-style state handling."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


POINTS = ("T1", "T2", "T3", "T4", "T5", "T6", "Tp")
MODEL_VARIANT = "stage0_cooling_continuation_v12b"
V11_WARMSTART_VARIANT = "v10_route_all_data_v11"
V10_BASELINE_VARIANT = "native_rollout_g4state_tpsafe_v10"
SCALE_STEPS = (4, 12, 20)
HISTORY_STEPS = 20
FUTURE_STEPS = 12
FRAME_SECONDS = 10
HISTORY_FEATURES = 12
HEATERS = 6
OUTPUTS = 7
UH_FUTURE_INDEX = 1
UH_HISTORY_INDEX = 7
AUX_FUTURE_INDICES = (0, 2, 3, 4, 5)
AUX_HISTORY_INDICES = (6, 8, 9, 10, 11)
RAW_POWER_FULL_SCALE = (50.0, 500.0, 50.0, 50.0, 50.0, 50.0)
SOURCE_POWER_FULL_SCALE = (500.0, 250.0)
MIN_LAG_SECONDS = (102.108, 77.091, 30.108, 50.112, 10.0535, 180.0, 39.104)
STATE_IDENTITY_FRAMES = 8


def inverse_sigmoid(probability: float) -> float:
    probability = min(max(float(probability), 1.0e-6), 1.0 - 1.0e-6)
    return math.log(probability / (1.0 - probability))


def build_lag_mask() -> torch.Tensor:
    """Select only historical powers old enough to affect each temperature point."""
    mask = torch.zeros(OUTPUTS, HISTORY_STEPS, dtype=torch.float32)
    for output_index, lag_seconds in enumerate(MIN_LAG_SECONDS):
        excluded_recent = int(math.ceil(lag_seconds / FRAME_SECONDS))
        eligible = max(0, HISTORY_STEPS - excluded_recent)
        if eligible:
            weights = torch.exp(torch.linspace(-2.0, 0.0, eligible))
            mask[output_index, :eligible] = weights / weights.sum()
    return mask


class FullHistoryStateEncoder(nn.Module):
    """Estimate a candidate-common state from the oldest eight history frames."""

    def __init__(self, input_features: int = 8, latent_features: int = 40) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_features, 40, batch_first=True)
        self.fusion = nn.Sequential(
            nn.Linear(40 + input_features * 3 + 6, 80),
            nn.GELU(),
            nn.LayerNorm(80),
            nn.Linear(80, latent_features),
            nn.Tanh(),
        )

    def forward(self, history_state: torch.Tensor) -> torch.Tensor:
        history_state = history_state.clamp(-8.0, 8.0)
        _, (hidden, _) = self.lstm(history_state)
        last = history_state[:, -1]
        recent = history_state[:, -4:].mean(1)
        full = history_state.mean(1)
        trend = history_state[:, -4:, :6].mean(1) - history_state[:, :4, :6].mean(1)
        return self.fusion(torch.cat([hidden[-1], last, recent, full, trend], dim=1))


class IndependentOutputHead(nn.Module):
    def __init__(self, context_features: int, maximum_delta_c: float) -> None:
        super().__init__()
        self.register_buffer("maximum_delta_c", torch.tensor(float(maximum_delta_c)))
        self.net = nn.Sequential(
            nn.Linear(context_features, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )
        nn.init.zeros_(self.net[4].weight)
        nn.init.zeros_(self.net[4].bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.maximum_delta_c * self.net(context)


class BoundedPositiveTwoSourceHead(nn.Module):
    """Return nonnegative UH and aggregate-auxiliary residual-heat gains."""

    def __init__(
        self,
        context_features: int,
        maximum_gains_c: torch.Tensor,
        initial_fraction: float,
    ) -> None:
        super().__init__()
        self.register_buffer("maximum_gains_c", maximum_gains_c.float())
        self.net = nn.Sequential(
            nn.Linear(context_features, 40),
            nn.GELU(),
            nn.Linear(40, 2),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.net[2].weight)
        nn.init.constant_(self.net[2].bias, inverse_sigmoid(initial_fraction))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.maximum_gains_c * self.net(context)


class BoundedCoolingHead(nn.Module):
    """Return a nonnegative loss term kept separate from residual heating."""

    def __init__(
        self, context_features: int, maximum_loss_c: float, initial_fraction: float
    ) -> None:
        super().__init__()
        self.register_buffer("maximum_loss_c", torch.tensor(float(maximum_loss_c)))
        self.net = nn.Sequential(
            nn.Linear(context_features, 40),
            nn.GELU(),
            nn.Linear(40, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.net[2].weight)
        nn.init.constant_(self.net[2].bias, inverse_sigmoid(initial_fraction))

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.maximum_loss_c * self.net(context)


class MultiHeaterStructuralLSTM(nn.Module):
    """Predict one 10-second endpoint and repeat that call exactly 12 times."""

    def __init__(
        self,
        stats: dict[str, Any],
        variant: str = MODEL_VARIANT,
        hidden_features: int = 48,
        device_features: int = 40,
    ) -> None:
        super().__init__()
        if variant not in (
            MODEL_VARIANT,
            V11_WARMSTART_VARIANT,
            V10_BASELINE_VARIANT,
        ):
            raise ValueError(f"unknown variant: {variant}")
        self.variant = variant
        self.register_buffer(
            "temperature_mean",
            torch.as_tensor(stats["temperature_mean"], dtype=torch.float32).reshape(6),
        )
        self.register_buffer(
            "temperature_std",
            torch.as_tensor(stats["temperature_std"], dtype=torch.float32)
            .reshape(6)
            .clamp_min(1.0e-4),
        )
        self.register_buffer(
            "power_full_scale", torch.tensor(RAW_POWER_FULL_SCALE, dtype=torch.float32)
        )
        self.register_buffer(
            "source_power_full_scale",
            torch.tensor(SOURCE_POWER_FULL_SCALE, dtype=torch.float32),
        )
        self.register_buffer("lag_mask", build_lag_mask())

        # Six external temperatures plus one aggregate auxiliary-power state.
        self.encoders = nn.ModuleList(
            [nn.LSTM(7, hidden_features, batch_first=True) for _ in SCALE_STEPS]
        )
        self.scale_score = nn.Linear(hidden_features, OUTPUTS, bias=False)
        self.scale_bias = nn.Parameter(torch.zeros(OUTPUTS, len(SCALE_STEPS)))
        nn.init.zeros_(self.scale_score.weight)

        # This state is computed once from the initial 20 real frames per rollout.
        self.initial_state_encoder = FullHistoryStateEncoder(8, device_features)
        self.tp_anchor_logits = nn.Linear(device_features, 6)
        self.tp_offset_head = nn.Sequential(
            nn.Linear(device_features, 40),
            nn.GELU(),
            nn.Linear(40, 1),
            nn.Tanh(),
        )
        nn.init.zeros_(self.tp_anchor_logits.weight)
        nn.init.zeros_(self.tp_anchor_logits.bias)
        nn.init.zeros_(self.tp_offset_head[2].weight)
        nn.init.zeros_(self.tp_offset_head[2].bias)
        self.register_buffer("maximum_tp_offset_c", torch.tensor(3.0))

        combined_features = hidden_features + device_features
        base_bounds = (0.45, 0.55, 0.40, 0.45, 2.40, 0.30, 0.80)
        self.output_heads = nn.ModuleList(
            [IndependentOutputHead(combined_features, bound) for bound in base_bounds]
        )
        residual_maximum = torch.tensor(
            [
                [0.20, 0.24],
                [0.28, 0.34],
                [0.18, 0.21],
                [0.23, 0.28],
                [0.58, 0.62],
                [0.14, 0.16],
                [0.44, 0.48],
            ],
            dtype=torch.float32,
        )
        cooling_maximum = (0.30, 0.34, 0.28, 0.30, 0.85, 0.22, 0.55)
        self.residual_heat_heads = nn.ModuleList(
            [
                BoundedPositiveTwoSourceHead(
                    combined_features, residual_maximum[index], 0.05
                )
                for index in range(OUTPUTS)
            ]
        )
        self.cooling_heads = nn.ModuleList(
            [
                BoundedCoolingHead(combined_features, cooling_maximum[index], 0.04)
                for index in range(OUTPUTS)
            ]
        )

    def normalized_components(
        self, history: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        temperatures_z = (history[..., :6] - self.temperature_mean) / self.temperature_std
        uh_fraction = (
            history[..., 7] / self.source_power_full_scale[0]
        ).clamp(0.0, 1.5)
        auxiliary = (
            history[:, :, 6]
            + history[:, :, 8]
            + history[:, :, 9]
            + history[:, :, 10]
            + history[:, :, 11]
        )
        auxiliary_fraction = (auxiliary / self.source_power_full_scale[1]).clamp(0.0, 1.5)
        return temperatures_z, uh_fraction, auxiliary_fraction

    def initial_device_state(self, history: torch.Tensor) -> torch.Tensor:
        temperatures_z, uh_fraction, auxiliary_fraction = self.normalized_components(history)
        state_history = torch.cat(
            [
                temperatures_z,
                uh_fraction[:, :, None],
                auxiliary_fraction[:, :, None],
            ],
            dim=2,
        )
        # During a 120-second native rollout, the oldest eight frames remain
        # from the real initial history.  Excluding newly fed-back frames keeps
        # the candidate-independent Tp baseline from becoming power-dependent.
        return self.initial_state_encoder(state_history[:, :8])

    def representations(
        self, temperatures_z: torch.Tensor, auxiliary_fraction: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        recurrent = torch.cat(
            [temperatures_z, auxiliary_fraction[:, :, None]], dim=2
        ).clamp(-8.0, 8.0)
        encoded_40, _ = self.encoders[0](recurrent[:, -4:])
        encoded_120, _ = self.encoders[1](recurrent[:, -12:])
        encoded_200, _ = self.encoders[2](recurrent[:, -20:])
        stacked = torch.stack(
            [encoded_40[:, -1], encoded_120[:, -1], encoded_200[:, -1]], dim=1
        )
        logits = self.scale_score(stacked).permute(0, 2, 1) + self.scale_bias[None]
        weights = torch.softmax(logits, dim=2)
        contexts = torch.einsum("bos,bsh->boh", weights, stacked)
        return contexts, weights

    def tp_anchor(
        self, history: torch.Tensor, initial_state: torch.Tensor
    ) -> torch.Tensor:
        anchor_weights = torch.softmax(self.tp_anchor_logits(initial_state), dim=1)
        spatial_anchor = (anchor_weights * history[:, -1, :6]).sum(1, keepdim=True)
        offset = self.maximum_tp_offset_c * self.tp_offset_head(initial_state)
        return spatial_anchor + offset

    def _forward_with_initial_state(
        self,
        history: torch.Tensor,
        future_step: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> torch.Tensor:
        # Every fitted point has a measured lag greater than one 10-second frame.
        # future_step only enters after execution through the rollout feedback below.
        temperatures_z, uh_fraction, auxiliary_fraction = self.normalized_components(history)
        contexts, _scale_weights = self.representations(
            temperatures_z, auxiliary_fraction
        )
        tp_anchor = self.tp_anchor(history, initial_state)
        anchors = torch.cat([history[:, -1, :6], tp_anchor], dim=1)
        combined = torch.cat(
            [contexts, initial_state[:, None].expand(-1, 7, -1)], dim=2
        )
        base_delta = torch.cat(
            [head(combined[:, index]) for index, head in enumerate(self.output_heads)],
            dim=1,
        )

        uh_dose = torch.einsum("bt,ot->bo", uh_fraction, self.lag_mask)
        auxiliary_dose = torch.einsum(
            "bt,ot->bo", auxiliary_fraction, self.lag_mask
        )
        residual_dose = torch.stack([uh_dose, auxiliary_dose], dim=2)
        residual_gains = torch.stack(
            [
                head(combined[:, index])
                for index, head in enumerate(self.residual_heat_heads)
            ],
            dim=1,
        )
        residual_heat = (residual_gains * residual_dose).sum(dim=2)
        cooling_loss = torch.cat(
            [head(combined[:, index]) for index, head in enumerate(self.cooling_heads)],
            dim=1,
        )
        prediction = anchors + base_delta + residual_heat - cooling_loss

        # Preserve the deployed two-input signature while enforcing the measured
        # one-frame minimum lag: current power only acts after feedback.
        prediction = prediction + future_step.sum(dim=1, keepdim=True) * 0.0
        return prediction

    def forward_one_step(
        self,
        history: torch.Tensor,
        future_step: torch.Tensor,
    ) -> torch.Tensor:
        if not torch.jit.is_tracing() and not torch.jit.is_scripting():
            if history.ndim != 3 or history.shape[1] != HISTORY_STEPS or history.shape[2] != HISTORY_FEATURES:
                raise ValueError(
                    f"history must have shape [B,{HISTORY_STEPS},{HISTORY_FEATURES}]"
                )
            if future_step.ndim != 2 or future_step.shape[1] != HEATERS:
                raise ValueError(f"future_step must have shape [B,{HEATERS}]")
        initial_state = self.initial_device_state(history)
        return self._forward_with_initial_state(history, future_step, initial_state)

    def forward(self, history: torch.Tensor, future_step: torch.Tensor) -> torch.Tensor:
        return self.forward_one_step(history, future_step)

    def rollout_native_120s(self, history: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_tracing():
            if future.ndim != 3 or tuple(future.shape[1:]) != (
                FUTURE_STEPS,
                HEATERS,
            ):
                raise ValueError(f"future must have shape [B,{FUTURE_STEPS},{HEATERS}]")
        state = history
        predictions = []
        for step in range(FUTURE_STEPS):
            # This is the deployment primitive used repeatedly: one model call,
            # one feedback update, then the next call.
            predicted = self.forward_one_step(state, future[:, step])
            predictions.append(predicted)
            feedback = torch.cat([predicted[:, :6], future[:, step]], dim=1)
            state = torch.cat([state[:, 1:], feedback[:, None]], dim=1)
        return torch.stack(predictions, dim=1)

    def rollout_120s(self, history: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        """Compatibility alias for training/evaluation code only.

        The implementation is the native one-step loop above.  It is not an
        exported deployment interface and is never traced into the final .pt.
        """
        return self.rollout_native_120s(history, future)


def checkpoint_contract(variant: str = MODEL_VARIANT) -> dict[str, Any]:
    if variant not in (
        MODEL_VARIANT,
        V11_WARMSTART_VARIANT,
        V10_BASELINE_VARIANT,
    ):
        raise ValueError(f"unknown variant: {variant}")
    return {
        "model_kind": (
            "MultiHeaterStageContinuationFixV12bLSTM"
            if variant == MODEL_VARIANT
            else (
                "MultiHeaterV10RouteAllDataLSTM"
                if variant == V11_WARMSTART_VARIANT
                else "MultiHeaterNativeOneStepG4PrincipledLSTM"
            )
        ),
        "variant": variant,
        "history_seconds": 200,
        "history_frames": 20,
        "frame_seconds": 10,
        "future_seconds": 120,
        "future_steps": 12,
        "history_input_shape": "[B,20,12]",
        "future_power_input_shape": "[B,6]",
        "single_step_output_shape": "[B,7]",
        "training_native_rollout_output_shape": "[B,12,7]",
        "deployment_interface": "native forward_one_step only; no 12-step wrapper",
        "torchscript_export": "scripted native one-step; portable between CPU and CUDA",
        "rollout_rule": "one external 10-second call, feedback, repeat exactly 12 times",
        "feedback": "predicted external six plus executed six powers",
        "Tp_input_or_feedback": False,
        "scale_attention": "independent per output over 40s, 120s and 200s LSTM encoders",
        "initial_Tp_state": "recomputed per call from oldest 8 current frames; candidate-independent during 120s",
        "power_protocol": "sum of five 2-second watt samples per 10-second frame",
        "candidate_control": "the controller changes UH; auxiliary five remain on supplied schedule",
        "internal_power_sources": "UH plus sum(aux1,aux2,aux3,aux4,aux5)",
        "independent_auxiliary_heater_gains": False,
        "thermal_balance": "signed base plus nonnegative residual heat minus independent nonnegative cooling loss",
        "future_power_application": "only after executed power is fed back and becomes lag-eligible",
        "candidate_direction_rule": "ordering only; candidate signs may differ and may cross zero",
        "training_curve_rule": (
            "warm-start from selected v11 seed21 stage0, then one-step temperature "
            "fit plus low-weight native 12-call rollout, adjacent-change fit, "
            "120-second endpoint fit, and candidate ordering; no second-difference "
            "or discrete whole-curve-shape objective"
        ),
        "training_data": (
            "109 development field sources plus formula-generated paired UH candidate "
            "families and 14 balanced structured trajectory scenarios"
        ),
        "training_sampling_curve": (
            "field cooling strata are selected only from true T120-T10 changes in "
            "the real future T5/Tp curves; remaining field samples stay source-balanced, "
            "structured samples stay scenario-balanced, and paired samples stay "
            "mother-state-balanced; no fixed power sign"
        ),
        "stage_checkpoint_guard": (
            "full field-selection population at every recovery checkpoint, relative "
            "to stage entrance, unchanged v10 and unchanged final v11 seed21; "
            "selection only"
        ),
        "direction_supervision": (
            "adjacent and net changes within the real future trajectory; the first "
            "predicted 10-second level is accuracy-only"
        ),
        "forbidden_training_rules": (
            "no fixed power-to-rise/cool labels and no prediction-minus-history-origin direction loss"
        ),
        "minimum_lag_seconds": list(MIN_LAG_SECONDS),
        "frozen_29_used": False,
    }
