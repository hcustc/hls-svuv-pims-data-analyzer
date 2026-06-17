from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal
import math

import numpy as np


ParameterValue = float | int
ParameterKind = Literal["float", "int", "log_float"]


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    low: float
    high: float
    kind: ParameterKind = "float"


@dataclass(frozen=True)
class PeakOptimizationTrial:
    trial: int
    strategy: str
    parameters: dict[str, ParameterValue]
    score: float
    metrics: dict[str, Any]


@dataclass(frozen=True)
class PeakOptimizationResult:
    analysis_type: str
    folder: str
    best_parameters: dict[str, ParameterValue]
    best_score: float
    best_metrics: dict[str, Any]
    trials: list[PeakOptimizationTrial]
    search_space: list[ParameterSpec]


def _normalise_parameters(
    parameters: dict[str, ParameterValue],
    search_space: list[ParameterSpec] | tuple[ParameterSpec, ...],
) -> dict[str, ParameterValue]:
    specs = {spec.name: spec for spec in search_space}
    normalised = dict(parameters)
    for name, value in list(normalised.items()):
        spec = specs.get(name)
        if spec is None:
            continue
        clipped = min(max(float(value), float(spec.low)), float(spec.high))
        normalised[name] = int(round(clipped)) if spec.kind == "int" else float(clipped)
    return normalised


def _sample_value(spec: ParameterSpec, rng: np.random.Generator) -> ParameterValue:
    if spec.kind == "int":
        return int(rng.integers(int(round(spec.low)), int(round(spec.high)) + 1))
    if spec.kind == "log_float":
        low = math.log(max(float(spec.low), np.finfo(float).tiny))
        high = math.log(max(float(spec.high), np.finfo(float).tiny))
        return float(math.exp(rng.uniform(low, high)))
    return float(rng.uniform(float(spec.low), float(spec.high)))


def _sample_parameters(
    base_parameters: dict[str, ParameterValue],
    search_space: list[ParameterSpec],
    rng: np.random.Generator,
) -> dict[str, ParameterValue]:
    parameters = dict(base_parameters)
    for spec in search_space:
        parameters[spec.name] = _sample_value(spec, rng)
    return _normalise_parameters(parameters, search_space)


def _jitter_parameters(
    base_parameters: dict[str, ParameterValue],
    search_space: list[ParameterSpec],
    rng: np.random.Generator,
    scale: float,
) -> dict[str, ParameterValue]:
    parameters = dict(base_parameters)
    for spec in search_space:
        current = float(parameters.get(spec.name, (spec.low + spec.high) / 2.0))
        span = float(spec.high) - float(spec.low)
        if spec.kind == "log_float":
            current = max(current, np.finfo(float).tiny)
            log_low = math.log(max(float(spec.low), np.finfo(float).tiny))
            log_high = math.log(max(float(spec.high), np.finfo(float).tiny))
            log_current = math.log(current)
            value = math.exp(log_current + rng.normal(0.0, (log_high - log_low) * scale))
        else:
            value = current + rng.normal(0.0, span * scale)
        parameters[spec.name] = int(round(value)) if spec.kind == "int" else float(value)
    return _normalise_parameters(parameters, search_space)


def run_peak_parameter_calibration(
    *,
    analysis_type: str,
    folder: str,
    base_parameters: dict[str, ParameterValue],
    evaluator: Callable[[dict[str, ParameterValue]], tuple[float, dict[str, Any]]],
    search_space: list[ParameterSpec] | tuple[ParameterSpec, ...],
    method: str = "bayesian",
    n_trials: int = 30,
    initial_random: int = 8,
    random_state: int = 42,
) -> PeakOptimizationResult:
    """Run lightweight parameter search for peak-detection workflows.

    The historical API accepted method="bayesian"; this dependency-free
    implementation keeps that interface and uses random exploration followed by
    local jitter around the best result.
    """
    specs = list(search_space)
    rng = np.random.default_rng(random_state)
    n_trials = max(1, int(n_trials))
    initial_random = max(0, min(int(initial_random), n_trials))

    trials: list[PeakOptimizationTrial] = []
    best_parameters = _normalise_parameters(dict(base_parameters), specs)
    best_score = -float("inf")
    best_metrics: dict[str, Any] = {}

    for trial_index in range(n_trials):
        if trial_index == 0:
            strategy = "base"
            parameters = best_parameters
        elif trial_index <= initial_random:
            strategy = "random"
            parameters = _sample_parameters(base_parameters, specs, rng)
        else:
            strategy = "local"
            scale = 0.20 if method.lower() in {"bayesian", "adaptive"} else 0.35
            parameters = _jitter_parameters(best_parameters, specs, rng, scale=scale)

        score, metrics = evaluator(parameters)
        trial = PeakOptimizationTrial(
            trial=trial_index + 1,
            strategy=strategy,
            parameters=dict(parameters),
            score=float(score),
            metrics=dict(metrics),
        )
        trials.append(trial)
        if trial.score > best_score:
            best_score = trial.score
            best_parameters = dict(parameters)
            best_metrics = dict(metrics)

    return PeakOptimizationResult(
        analysis_type=analysis_type,
        folder=folder,
        best_parameters=best_parameters,
        best_score=float(best_score),
        best_metrics=best_metrics,
        trials=trials,
        search_space=specs,
    )
