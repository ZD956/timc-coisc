"""Formal all-data loading, v10-route losses, and selection audits."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from model_structural_ablation import (
    FUTURE_STEPS,
    HEATERS,
    HISTORY_FEATURES,
    HISTORY_STEPS,
    UH_FUTURE_INDEX,
    OUTPUTS,
    POINTS,
    RAW_POWER_FULL_SCALE,
    MultiHeaterStructuralLSTM,
)


POINT_WEIGHTS = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.8, 1.0, 2.0])
HORIZON_WEIGHTS = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.05, 1.10, 1.15, 1.20, 1.30, 1.40, 1.50, 1.65])
UH_LEVELS_W = (
    0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
    10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0,
)
FUTURE_CURVE_LABEL_DEADBAND_C = 0.06
SEGMENT_SLICES = ((0, 4), (3, 8), (7, 12))


def _curve_shape_name(signs: np.ndarray) -> str:
    """Name a complete future curve from its three fitted segment signs."""
    active = [int(value) for value in signs if value != 0]
    if not active:
        return "flat"
    compressed = [active[0]]
    for value in active[1:]:
        if value != compressed[-1]:
            compressed.append(value)
    if compressed == [1]:
        return "sustained_rise"
    if compressed == [-1]:
        return "sustained_cooling"
    if compressed == [1, -1]:
        return "rise_to_cool"
    if compressed == [-1, 1]:
        return "cool_to_rise"
    return "multi_turn"


def training_scenario_labels(
    history: np.ndarray,
    future: np.ndarray,
    target: np.ndarray,
    origin: np.ndarray,
) -> np.ndarray:
    """Group samples by future power regime and the real complete Tp curve."""
    del origin
    all_zero = np.all(np.abs(future) <= 1.0e-6, axis=(1, 2))
    full_scale = np.asarray(RAW_POWER_FULL_SCALE, dtype=np.float32)
    power_fraction = np.clip(future / full_scale[None, None], 0.0, 1.0)
    high_power = power_fraction.max(axis=(1, 2)) >= 0.70
    segments = fitted_segment_changes(target)[:, :, 6]
    signs = _signed(segments, FUTURE_CURVE_LABEL_DEADBAND_C)
    curve_shape = np.asarray([_curve_shape_name(row) for row in signs], dtype="<U20")
    regime = np.full(len(target), "powered_normal", dtype="<U20")
    regime[all_zero] = "allzero"
    regime[(~all_zero) & high_power] = "powered_high"
    return np.char.add(np.char.add(regime, "__Tp_"), curve_shape)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def require_shape(name: str, array: np.ndarray, tail: tuple[int, ...]) -> None:
    if array.ndim != len(tail) + 1 or tuple(array.shape[1:]) != tail:
        expected = ",".join(map(str, tail))
        raise ValueError(f"{name} must have shape [N,{expected}], got {array.shape}")


@dataclass
class ArraySource:
    name: str
    history: np.ndarray
    future: np.ndarray
    target: np.ndarray
    origin: np.ndarray
    source_labels: np.ndarray | None = None
    scenario_labels: np.ndarray | None = None
    normalization_indices: np.ndarray | None = None
    _groups: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _group_probabilities: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64), init=False, repr=False
    )

    def __post_init__(self) -> None:
        require_shape(f"{self.name}.history", self.history, (HISTORY_STEPS, HISTORY_FEATURES))
        require_shape(f"{self.name}.future", self.future, (FUTURE_STEPS, HEATERS))
        require_shape(f"{self.name}.target", self.target, (FUTURE_STEPS, OUTPUTS))
        require_shape(f"{self.name}.origin", self.origin, (OUTPUTS,))
        if not (len(self.history) == len(self.future) == len(self.target) == len(self.origin)):
            raise ValueError(f"length mismatch in source {self.name}")
        if self.source_labels is not None:
            if len(self.source_labels) != len(self.history):
                raise ValueError(f"source label length mismatch in {self.name}")
        if self.scenario_labels is None:
            self.scenario_labels = training_scenario_labels(
                self.history, self.future, self.target, self.origin
            )
        if len(self.scenario_labels) != len(self.history):
            raise ValueError(f"scenario label length mismatch in {self.name}")
        if self.normalization_indices is None:
            self.normalization_indices = np.arange(len(self.history), dtype=np.int64)
        if self.normalization_indices.ndim != 1 or not len(self.normalization_indices):
            raise ValueError(f"normalization indices are invalid in {self.name}")
        if int(self.normalization_indices.min()) < 0 or int(self.normalization_indices.max()) >= len(self.history):
            raise ValueError(f"normalization indices are out of range in {self.name}")
        # Training batches are balanced by source file when source identities are
        # available. Future curve labels remain available for diagnostics, but are
        # deliberately not used as sampling strata or hidden shape targets.
        if self.source_labels is None:
            self._groups = [np.arange(len(self.history), dtype=np.int64)]
        else:
            labels = self.source_labels.astype(str)
            self._groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
        # Square-root weighting prevents tiny files from dominating while still
        # giving each real source repeated exposure.
        group_sizes = np.asarray([len(group) for group in self._groups], dtype=np.float64)
        self._group_probabilities = np.sqrt(group_sizes)
        self._group_probabilities /= self._group_probabilities.sum()

    def batch(
        self,
        rng: np.random.Generator,
        size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._groups:
            selected_groups = rng.choice(
                len(self._groups), size=size, replace=True, p=self._group_probabilities
            )
            indices = np.asarray(
                [group[int(rng.integers(0, len(group)))] for group in (self._groups[index] for index in selected_groups)],
                dtype=np.int64,
            )
        else:
            indices = rng.integers(0, len(self.history), size=size)
        return (
            self.history[indices],
            self.future[indices],
            self.target[indices],
            self.origin[indices],
        )


@dataclass
class EvaluationSet:
    name: str
    history: np.ndarray
    future: np.ndarray
    target: np.ndarray
    origin: np.ndarray
    source_labels: np.ndarray | None = None
    scenario_labels: np.ndarray | None = None

    def __post_init__(self) -> None:
        require_shape(f"{self.name}.history", self.history, (HISTORY_STEPS, HISTORY_FEATURES))
        require_shape(f"{self.name}.future", self.future, (FUTURE_STEPS, HEATERS))
        require_shape(f"{self.name}.target", self.target, (FUTURE_STEPS, OUTPUTS))
        require_shape(f"{self.name}.origin", self.origin, (OUTPUTS,))
        if self.scenario_labels is None:
            self.scenario_labels = training_scenario_labels(
                self.history, self.future, self.target, self.origin
            )
        if len(self.scenario_labels) != len(self.history):
            raise ValueError(f"scenario label length mismatch in {self.name}")


def load_development_data(
    data_root: Path,
) -> tuple[dict[str, ArraySource], dict[str, EvaluationSet], dict[str, Any]]:
    field_data = load_npz(data_root / "field_windows_development_109.npz")
    paired = load_npz(data_root / "paired_UH_candidate_families_6000x22.npz")
    structured = load_npz(data_root / "structured_native10s_trajectories_2800.npz")
    summary_path = data_root / "00_FORMAL_DATA_SUMMARY.json"
    formal_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    def family_labels(values: np.ndarray) -> np.ndarray:
        return np.char.add("family_", values.astype(str))

    def first_per_family(values: np.ndarray) -> np.ndarray:
        return np.sort(np.unique(values, return_index=True)[1].astype(np.int64))

    scenario_names = structured["scenario_names"].astype(str)
    train_scenarios = scenario_names[structured["train_scenario_index"]]
    validation_scenarios = scenario_names[structured["validation_scenario_index"]]

    sources = {
        "field": ArraySource(
            "field_91_fit_sources",
            field_data["train_history"],
            field_data["train_future"],
            field_data["train_target"],
            field_data["train_origin"],
            field_data["train_source_file"],
        ),
        "paired": ArraySource(
            "paired_4800_mother_states_x22",
            paired["train_history"],
            paired["train_future"],
            paired["train_target"],
            paired["train_origin"],
            family_labels(paired["train_family_index"]),
            normalization_indices=first_per_family(paired["train_family_index"]),
        ),
        "structured": ArraySource(
            "structured_14_scenarios",
            structured["train_history"],
            structured["train_future"],
            structured["train_target"],
            structured["train_origin"],
            train_scenarios,
        ),
    }
    evaluation = {
        "field": EvaluationSet(
            "field_selection_18_sources",
            field_data["validation_history"],
            field_data["validation_future"],
            field_data["validation_target"],
            field_data["validation_origin"],
            field_data["validation_source_file"],
        ),
        "paired": EvaluationSet(
            "paired_validation_1200_mother_states_x22",
            paired["validation_history"],
            paired["validation_future"],
            paired["validation_target"],
            paired["validation_origin"],
            family_labels(paired["validation_family_index"]),
        ),
        "structured": EvaluationSet(
            "structured_validation_14_scenarios",
            structured["validation_history"],
            structured["validation_future"],
            structured["validation_target"],
            structured["validation_origin"],
            validation_scenarios,
        ),
    }
    metadata = {
        "training_windows": {name: int(len(source.history)) for name, source in sources.items()},
        "training_sampling_groups": {name: int(len(source._groups)) for name, source in sources.items()},
        "training_scenario_counts": {
            name: {
                str(label): int((source.scenario_labels == label).sum())
                for label in np.unique(source.scenario_labels)
            }
            for name, source in sources.items()
        },
        "selection_windows": {name: int(len(dataset.history)) for name, dataset in evaluation.items()},
        "field_gradient_source_count": int(len(np.unique(sources["field"].source_labels))),
        "field_selection_source_count": len(np.unique(evaluation["field"].source_labels)),
        "formal_data_status": formal_summary["status"],
        "formula_sha256": formal_summary["locked_inputs"]["formula_sha256"],
        "paired_candidate_levels_W": formal_summary["paired"]["candidate_levels_W"],
        "frozen_29_opened": False,
        "trend_only_used": False,
    }
    return sources, evaluation, metadata


def compute_model_stats(sources: dict[str, ArraySource]) -> dict[str, Any]:
    temperature_blocks = []
    uh_blocks = []
    auxiliary_blocks = []
    for source in sources.values():
        history = source.history[source.normalization_indices].astype(np.float64, copy=False)
        temperature_blocks.append(history[..., :6].reshape(-1, 6))
        uh_blocks.append(history[..., 7].reshape(-1))
        auxiliary_blocks.append(history[..., (6, 8, 9, 10, 11)].sum(axis=2).reshape(-1))
    temperatures = np.concatenate(temperature_blocks)
    uh = np.concatenate(uh_blocks)
    auxiliary = np.concatenate(auxiliary_blocks)
    return {
        "temperature_mean": temperatures.mean(axis=0).astype(np.float32),
        "temperature_std": temperatures.std(axis=0).clip(1.0e-4).astype(np.float32),
        "uh_mean": float(uh.mean()),
        "uh_std": float(max(uh.std(), 1.0e-4)),
        "auxiliary_mean": float(auxiliary.mean()),
        "auxiliary_std": float(max(auxiliary.std(), 1.0e-4)),
    }


def to_device(values: np.ndarray, device: torch.device) -> torch.Tensor:
    contiguous = np.ascontiguousarray(values)
    return torch.from_numpy(contiguous).to(device=device, dtype=torch.float32)


def weighted_huber(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    point_weights = POINT_WEIGHTS.to(prediction.device)
    loss = F.smooth_l1_loss(prediction, target, reduction="none", beta=0.15)
    if prediction.ndim == 2:
        element_weights = point_weights[None]
    elif prediction.ndim == 3:
        horizon = HORIZON_WEIGHTS[: prediction.shape[1]].to(prediction.device)
        element_weights = horizon[None, :, None] * point_weights[None, None]
    else:
        raise ValueError(f"unexpected prediction rank: {prediction.ndim}")
    denominator = element_weights.expand_as(loss).sum().clamp_min(1.0)
    return (loss * element_weights).sum() / denominator


def adjacent_delta_huber(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Fit all 11 changes inside the predicted future curve."""
    return weighted_huber(torch.diff(prediction, dim=1), torch.diff(target, dim=1))


def endpoint_temperature_huber(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Fit the absolute temperature at 120 s; this is not a direction label."""
    return weighted_huber(prediction[:, -1], target[:, -1])


def second_difference_huber(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Low-weight curvature fitting, following the useful generic part of G4."""
    return weighted_huber(
        torch.diff(prediction, n=2, dim=1),
        torch.diff(target, n=2, dim=1),
    )


def fitted_segment_changes_torch(trajectory: torch.Tensor) -> torch.Tensor:
    """Differentiable counterpart of fitted_segment_changes."""
    segments = []
    for start, stop in SEGMENT_SLICES:
        values = trajectory[:, start:stop]
        x = torch.arange(
            stop - start, dtype=trajectory.dtype, device=trajectory.device
        )
        centered = x - x.mean()
        slope = (
            (values - values.mean(dim=1, keepdim=True))
            * centered[None, :, None]
        ).sum(dim=1) / centered.square().sum()
        segments.append(slope * float(stop - start - 1))
    return torch.stack(segments, dim=1)


def three_segment_change_huber(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Fit T10-40, T40-80 and T80-120 changes without assigning signs."""
    return weighted_huber(
        fitted_segment_changes_torch(prediction),
        fitted_segment_changes_torch(target),
    )


def candidate_order_loss(
    model: MultiHeaterStructuralLSTM,
    history: torch.Tensor,
    future: torch.Tensor,
    heater_index: int,
    low_fraction: float,
    high_fraction: float,
) -> torch.Tensor:
    low = future.clone()
    high = future.clone()
    full_scale = model.power_full_scale[heater_index]
    low[:, :, heater_index] = full_scale * low_fraction
    high[:, :, heater_index] = full_scale * high_fraction
    low_prediction = model.rollout_native_120s(history, low)
    high_prediction = model.rollout_native_120s(history, high)
    # Equality is valid before a point's physical lag. Only inversions are penalized.
    return F.relu(-(high_prediction - low_prediction)).mean()


def uh_candidate_grid_order_loss(
    model: MultiHeaterStructuralLSTM,
    history: torch.Tensor,
    future: torch.Tensor,
) -> torch.Tensor:
    """Audit all 22 approved UH levels instead of sampling one adjacent pair."""
    predictions = []
    for level_w in UH_LEVELS_W:
        candidate = future.clone()
        candidate[:, :, UH_FUTURE_INDEX] = level_w * 5.0
        predictions.append(model.rollout_native_120s(history, candidate))
    stacked = torch.stack(predictions, dim=1)
    differences = stacked[:, 1:] - stacked[:, :-1]
    # Do not force all candidates to rise or to separate before the thermal lag.
    return F.relu(-differences).mean()


def _balanced_indices(dataset: EvaluationSet, maximum_windows: int) -> np.ndarray:
    count = min(len(dataset.history), maximum_windows)
    if count == len(dataset.history):
        return np.arange(len(dataset.history), dtype=np.int64)
    if dataset.source_labels is None:
        return np.linspace(0, len(dataset.history) - 1, count, dtype=np.int64)
    combined = np.char.add(
        np.char.add(dataset.source_labels.astype(str), "::"),
        dataset.scenario_labels.astype(str),
    )
    labels = np.unique(combined)
    per_source = max(1, maximum_windows // len(labels))
    selected: list[np.ndarray] = []
    for label in labels:
        available = np.flatnonzero(combined == label)
        source_count = min(len(available), per_source)
        positions = np.linspace(0, len(available) - 1, source_count, dtype=np.int64)
        selected.append(available[positions])
    indices = np.unique(np.concatenate(selected))
    if len(indices) > count:
        positions = np.linspace(0, len(indices) - 1, count, dtype=np.int64)
        indices = indices[positions]
    return indices


def fitted_segment_changes(trajectory: np.ndarray) -> np.ndarray:
    """Fit changes over 10-40 s, 40-80 s and 80-120 s."""
    segments = []
    for start, stop in SEGMENT_SLICES:
        values = trajectory[:, start:stop].astype(np.float64)
        x = np.arange(stop - start, dtype=np.float64)
        centered = x - x.mean()
        slope = np.einsum(
            "nsp,s->np", values - values.mean(axis=1, keepdims=True), centered
        ) / float(np.square(centered).sum())
        segments.append(slope * float(stop - start - 1))
    return np.stack(segments, axis=1)


def calibrate_curve_deadbands(field_source: ArraySource) -> dict[str, np.ndarray]:
    """Calibrate neutral bands from the 20 training sources, never selection data."""
    step = np.diff(field_source.target, axis=1)
    segment = fitted_segment_changes(field_source.target)
    net = field_source.target[:, -1] - field_source.target[:, 0]
    step_band = np.clip(np.quantile(np.abs(step), 0.25, axis=(0, 1)), 0.02, 0.05)
    segment_band = np.maximum(
        2.0 * step_band, np.quantile(np.abs(segment), 0.25, axis=(0, 1))
    )
    net_band = np.maximum(
        3.0 * step_band, np.quantile(np.abs(net), 0.25, axis=0)
    )
    return {
        "step": step_band.astype(np.float64),
        "segment": segment_band.astype(np.float64),
        "net": net_band.astype(np.float64),
    }


def curve_deadbands_json(deadbands: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: {point: float(value) for point, value in zip(POINTS, values)}
        for name, values in deadbands.items()
    }


def _signed(values: np.ndarray, threshold: np.ndarray | float) -> np.ndarray:
    threshold_array = np.asarray(threshold)
    return np.where(
        values > threshold_array,
        1,
        np.where(values < -threshold_array, -1, 0),
    ).astype(np.int8)


def _agreement(
    target_delta: np.ndarray,
    prediction_delta: np.ndarray,
    threshold: np.ndarray | float,
    prediction_epsilon: float = 1.0e-4,
) -> tuple[float, dict[str, float]]:
    target_sign = _signed(target_delta, threshold)
    prediction_sign = _signed(prediction_delta, prediction_epsilon)
    mask = target_sign != 0
    correct = target_sign == prediction_sign
    total = int(mask.sum())
    aggregate = float(correct[mask].sum() / max(total, 1))
    by_point: dict[str, float] = {}
    for index, point in enumerate(POINTS):
        point_mask = mask[..., index]
        by_point[point] = float(correct[..., index][point_mask].sum() / max(int(point_mask.sum()), 1))
    return aggregate, by_point


def _pattern_name(signs: np.ndarray) -> str:
    return _curve_shape_name(signs)


def _turning_agreement(
    target_segments: np.ndarray,
    predicted_segments: np.ndarray,
    threshold: np.ndarray,
) -> tuple[dict[str, float], dict[str, int]]:
    target_sign = _signed(target_segments, threshold)
    prediction_sign = _signed(predicted_segments, 0.002)
    by_point: dict[str, float] = {}
    counts: dict[str, int] = {}
    for index, point in enumerate(POINTS):
        true_patterns = np.asarray(
            [_pattern_name(row) for row in target_sign[:, :, index]]
        )
        predicted_patterns = np.asarray(
            [_pattern_name(row) for row in prediction_sign[:, :, index]]
        )
        mask = np.isin(true_patterns, ("rise_to_cool", "cool_to_rise", "multi_turn"))
        counts[point] = int(mask.sum())
        by_point[point] = float(
            (true_patterns[mask] == predicted_patterns[mask]).sum() / max(int(mask.sum()), 1)
        )
    return by_point, counts


def _whole_curve_shape_agreement(
    target_segments: np.ndarray,
    predicted_segments: np.ndarray,
    threshold: np.ndarray,
) -> tuple[float, dict[str, float], dict[str, dict[str, int]]]:
    target_sign = _signed(target_segments, threshold)
    prediction_sign = _signed(predicted_segments, 0.002)
    by_point: dict[str, float] = {}
    confusion: dict[str, dict[str, int]] = {}
    correct_total = 0
    total = target_segments.shape[0] * target_segments.shape[2]
    for index, point in enumerate(POINTS):
        target_names = np.asarray(
            [_curve_shape_name(row) for row in target_sign[:, :, index]]
        )
        predicted_names = np.asarray(
            [_curve_shape_name(row) for row in prediction_sign[:, :, index]]
        )
        correct = target_names == predicted_names
        correct_total += int(correct.sum())
        by_point[point] = float(correct.mean()) if len(correct) else 1.0
        confusion[point] = {
            name: int((target_names == name).sum())
            for name in (
                "sustained_rise",
                "sustained_cooling",
                "rise_to_cool",
                "cool_to_rise",
                "flat",
                "multi_turn",
            )
        }
    return float(correct_total / max(total, 1)), by_point, confusion


def _critical_scenario_metrics(
    future: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
    net_deadband: np.ndarray,
    segment_deadband: np.ndarray,
) -> dict[str, Any]:
    true_endpoint = target[:, -1] - target[:, 0]
    predicted_endpoint = prediction[:, -1] - prediction[:, 0]
    all_zero = np.all(np.abs(future) <= 1.0e-6, axis=(1, 2))
    scenarios = {
        "allzero_T5_rising": all_zero & (true_endpoint[:, 4] > net_deadband[4]),
        "allzero_T5_cooling": all_zero & (true_endpoint[:, 4] < -net_deadband[4]),
        "allzero_Tp_rising": all_zero & (true_endpoint[:, 6] > net_deadband[6]),
        "allzero_Tp_cooling": all_zero & (true_endpoint[:, 6] < -net_deadband[6]),
        "powered_T5_cooling": (~all_zero) & (true_endpoint[:, 4] < -net_deadband[4]),
        "powered_T5_rising": (~all_zero) & (true_endpoint[:, 4] > net_deadband[4]),
        "powered_Tp_cooling": (~all_zero) & (true_endpoint[:, 6] < -net_deadband[6]),
        "powered_Tp_rising": (~all_zero) & (true_endpoint[:, 6] > net_deadband[6]),
    }
    result: dict[str, Any] = {}
    for name, mask in scenarios.items():
        point_index = 4 if "T5" in name else 6
        count = int(mask.sum())
        predicted_sign = _signed(predicted_endpoint[mask, point_index], 0.002)
        true_sign = _signed(true_endpoint[mask, point_index], net_deadband[point_index])
        wrong = int((predicted_sign != true_sign).sum())
        target_segments = fitted_segment_changes(target[mask])[:, :, point_index]
        predicted_segments = fitted_segment_changes(prediction[mask])[:, :, point_index]
        target_signs = _signed(target_segments, segment_deadband[point_index])
        predicted_signs = _signed(predicted_segments, 0.002)
        target_shapes = np.asarray([_curve_shape_name(row) for row in target_signs])
        predicted_shapes = np.asarray([_curve_shape_name(row) for row in predicted_signs])
        shape_correct = target_shapes == predicted_shapes
        result[name] = {
            "windows": count,
            "wrong_sign_count": wrong,
            "direction_agreement": float(1.0 - wrong / count) if count else 1.0,
            "true_endpoint_mean_C": float(true_endpoint[mask, point_index].mean()) if count else 0.0,
            "predicted_endpoint_mean_C": float(predicted_endpoint[mask, point_index].mean()) if count else 0.0,
            "future_curve_shape_agreement": float(shape_correct.mean()) if count else 1.0,
            "target_curve_shape_counts": {
                shape: int((target_shapes == shape).sum())
                for shape in (
                    "sustained_rise", "sustained_cooling", "rise_to_cool",
                    "cool_to_rise", "flat", "multi_turn"
                )
            },
            "direction_definition": "complete T10-T120 curve; no history-origin sign",
        }
    return result


@torch.no_grad()
def evaluate_dataset(
    model: MultiHeaterStructuralLSTM,
    dataset: EvaluationSet,
    device: torch.device,
    maximum_windows: int,
    batch_size: int,
    deadbands: dict[str, np.ndarray],
) -> dict[str, Any]:
    indices = _balanced_indices(dataset, maximum_windows)
    predictions = []
    targets = []
    model.eval()
    for start in range(0, len(indices), batch_size):
        chosen = indices[start : start + batch_size]
        history = to_device(dataset.history[chosen], device)
        future = to_device(dataset.future[chosen], device)
        prediction = model.rollout_native_120s(history, future)
        predictions.append(prediction.cpu().numpy())
        targets.append(dataset.target[chosen])
    predicted = np.concatenate(predictions).astype(np.float64)
    target = np.concatenate(targets).astype(np.float64)
    error = predicted - target
    point_mae = np.abs(error).mean(axis=(0, 1))
    point_rmse = np.sqrt(np.square(error).mean(axis=(0, 1)))

    worst_source = {point: float(value) for point, value in zip(POINTS, point_mae)}
    p90_source = dict(worst_source)
    if dataset.source_labels is not None:
        selected_labels = dataset.source_labels[indices]
        source_mae = []
        for label in np.unique(selected_labels):
            source_mask = selected_labels == label
            source_mae.append(np.abs(error[source_mask]).mean(axis=(0, 1)))
        source_mae_array = np.stack(source_mae)
        worst_source = {
            point: float(value) for point, value in zip(POINTS, source_mae_array.max(axis=0))
        }
        p90_source = {
            point: float(value) for point, value in zip(POINTS, np.quantile(source_mae_array, 0.90, axis=0))
        }

    step_agreement, step_by_point = _agreement(
        np.diff(target, axis=1), np.diff(predicted, axis=1), deadbands["step"]
    )
    target_segments = fitted_segment_changes(target)
    predicted_segments = fitted_segment_changes(predicted)
    segment_agreement, segment_by_point = _agreement(
        target_segments, predicted_segments, deadbands["segment"], 0.002
    )
    target_net = target[:, -1] - target[:, 0]
    predicted_net = predicted[:, -1] - predicted[:, 0]
    net_agreement, net_by_point = _agreement(
        target_net, predicted_net, deadbands["net"], 0.002
    )
    turning_by_point, turning_counts = _turning_agreement(
        target_segments, predicted_segments, deadbands["segment"]
    )
    whole_shape, whole_shape_by_point, shape_counts = _whole_curve_shape_agreement(
        target_segments, predicted_segments, deadbands["segment"]
    )
    return {
        "windows": int(len(indices)),
        "source_balanced": dataset.source_labels is not None,
        "source_count": int(len(np.unique(dataset.source_labels))) if dataset.source_labels is not None else None,
        "external6_MAE_C": float(point_mae[:6].mean()),
        "Tp_MAE_C": float(point_mae[6]),
        "all7_MAE_C": float(point_mae.mean()),
        "first_10s_MAE_C": float(np.abs(error[:, 0]).mean()),
        "endpoint_MAE_C": float(np.abs(error[:, -1]).mean()),
        "point_MAE_C": {point: float(value) for point, value in zip(POINTS, point_mae)},
        "point_RMSE_C": {point: float(value) for point, value in zip(POINTS, point_rmse)},
        "worst_source_point_MAE_C": worst_source,
        "p90_source_point_MAE_C": p90_source,
        "step_direction_agreement": step_agreement,
        "step_direction_by_point": step_by_point,
        "segment_direction_agreement": segment_agreement,
        "segment_direction_by_point": segment_by_point,
        "net_10_120_direction_agreement": net_agreement,
        "net_10_120_direction_by_point": net_by_point,
        "turning_agreement_by_point": turning_by_point,
        "turning_case_count_by_point": turning_counts,
        "future_curve_shape_agreement": whole_shape,
        "future_curve_shape_by_point": whole_shape_by_point,
        "future_curve_shape_target_counts_by_point": shape_counts,
        "direction_definition": (
            "future-only: all adjacent changes, three fitted segments, complete curve "
            "shape, and T120-T10; history origin is excluded"
        ),
        "critical_scenarios": _critical_scenario_metrics(
            dataset.future[indices], target, predicted, deadbands["net"], deadbands["segment"]
        ),
    }


@torch.no_grad()
def candidate_audit(
    model: MultiHeaterStructuralLSTM,
    dataset: EvaluationSet,
    device: torch.device,
    histories: int,
    tiny_tolerance_c: float = 1.0e-5,
    meaningful_tolerance_c: float = 2.0e-3,
) -> dict[str, Any]:
    indices = _balanced_indices(dataset, histories)
    tiny_inversions = 0
    meaningful_inversions = 0
    tp_meaningful_inversions = 0
    by_point = np.zeros(OUTPUTS, dtype=np.int64)
    maximum_inversion = 0.0
    minimum_adjacent_difference = float("inf")
    total_comparisons = 0
    first_step_max_spread = 0.0
    all_cooling = np.zeros(OUTPUTS, dtype=np.int64)
    cooling_to_rising = np.zeros(OUTPUTS, dtype=np.int64)
    all_rising = np.zeros(OUTPUTS, dtype=np.int64)
    other_regime = np.zeros(OUTPUTS, dtype=np.int64)
    model.eval()
    for start in range(0, len(indices), 8):
        chosen = indices[start : start + 8]
        history = to_device(dataset.history[chosen], device)
        future = to_device(dataset.future[chosen], device)
        candidates = []
        for level_w in UH_LEVELS_W:
            candidate_future = future.clone()
            candidate_future[:, :, UH_FUTURE_INDEX] = level_w * 5.0
            candidates.append(model.rollout_native_120s(history, candidate_future))
        stacked = torch.stack(candidates, dim=1)
        differences = stacked[:, 1:] - stacked[:, :-1]
        endpoint = stacked[:, :, -1] - stacked[:, :, 0]
        minimum_endpoint = endpoint.amin(dim=1)
        maximum_endpoint = endpoint.amax(dim=1)
        cooling_mask = maximum_endpoint <= -0.03
        rising_mask = minimum_endpoint >= 0.03
        crossover_mask = (minimum_endpoint < -0.03) & (maximum_endpoint > 0.03)
        other_mask = ~(cooling_mask | rising_mask | crossover_mask)
        all_cooling += cooling_mask.sum(dim=0).cpu().numpy().astype(np.int64)
        all_rising += rising_mask.sum(dim=0).cpu().numpy().astype(np.int64)
        cooling_to_rising += crossover_mask.sum(dim=0).cpu().numpy().astype(np.int64)
        other_regime += other_mask.sum(dim=0).cpu().numpy().astype(np.int64)
        first_spread = stacked[:, :, 0].amax(dim=1) - stacked[:, :, 0].amin(dim=1)
        first_step_max_spread = max(
            first_step_max_spread, float(first_spread.abs().max().cpu())
        )
        tiny = differences < -tiny_tolerance_c
        meaningful = differences < -meaningful_tolerance_c
        tiny_inversions += int(tiny.sum().cpu())
        meaningful_inversions += int(meaningful.sum().cpu())
        tp_meaningful_inversions += int(meaningful[..., 6].sum().cpu())
        by_point += meaningful.sum(dim=(0, 1, 2)).cpu().numpy().astype(np.int64)
        total_comparisons += differences.numel()
        maximum_inversion = max(maximum_inversion, float((-differences).clamp_min(0).max().cpu()))
        minimum_adjacent_difference = min(minimum_adjacent_difference, float(differences.min().cpu()))
    return {
        "histories": int(len(indices)),
        "candidate_levels_W": list(UH_LEVELS_W),
        "meaningful_tolerance_C": meaningful_tolerance_c,
        "tiny_tolerance_C": tiny_tolerance_c,
        "total_comparisons": int(total_comparisons),
        "all7_inversion_count": meaningful_inversions,
        "meaningful_inversion_count": meaningful_inversions,
        "tiny_inversion_count": tiny_inversions,
        "tiny_inversion_ratio": float(tiny_inversions / max(total_comparisons, 1)),
        "Tp_inversion_count": tp_meaningful_inversions,
        "inversion_count_by_point": {point: int(value) for point, value in zip(POINTS, by_point)},
        "maximum_inversion_C": maximum_inversion,
        "minimum_adjacent_difference_C": minimum_adjacent_difference,
        "first_10s_max_candidate_spread_C": first_step_max_spread,
        "endpoint_direction_regimes_by_point": {
            point: {
                "all_candidates_cooling": int(all_cooling[index]),
                "low_cooling_high_rising": int(cooling_to_rising[index]),
                "all_candidates_rising": int(all_rising[index]),
                "contains_flat_or_boundary": int(other_regime[index]),
            }
            for index, point in enumerate(POINTS)
        },
    }


def _window_digest(*arrays: np.ndarray) -> bytes:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.view(np.uint8))
    return digest.digest()


def validation_overlap_audit(
    evaluation: dict[str, EvaluationSet],
) -> dict[str, Any]:
    digests: dict[str, set[bytes]] = {}
    for name, dataset in evaluation.items():
        digests[name] = {
            _window_digest(history, future, target, origin)
            for history, future, target, origin in zip(
                dataset.history, dataset.future, dataset.target, dataset.origin
            )
        }
    overlaps: dict[str, int] = {}
    names = tuple(evaluation)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlaps[f"{left}__{right}"] = len(digests[left] & digests[right])
    return {
        "dataset_windows": {
            name: int(len(dataset.history)) for name, dataset in evaluation.items()
        },
        "exact_window_overlap_between_validation_blocks": overlaps,
        "critical_gate_population": "18-source field selection block",
        "formula_selection_sources_are_file_isolated_from_formula_fit": True,
        "frozen_29_opened": False,
    }


def evaluate_suite(
    model: MultiHeaterStructuralLSTM,
    evaluation: dict[str, EvaluationSet],
    device: torch.device,
    maximum_windows: int,
    batch_size: int,
    candidate_histories: int,
    deadbands: dict[str, np.ndarray],
    full_field: bool = False,
) -> dict[str, Any]:
    metrics = {}
    for name, dataset in evaluation.items():
        full_field_evaluation = full_field and name == "field"
        limit = len(dataset.history) if full_field_evaluation else maximum_windows
        metrics[name] = evaluate_dataset(
            model, dataset, device, limit, batch_size, deadbands
        )
    metrics["candidate_audit"] = candidate_audit(
        model,
        evaluation["paired"],
        device,
        histories=candidate_histories,
    )
    metrics["validation_overlap_audit"] = validation_overlap_audit(evaluation)
    metrics["curve_deadbands_C"] = curve_deadbands_json(deadbands)
    return metrics
