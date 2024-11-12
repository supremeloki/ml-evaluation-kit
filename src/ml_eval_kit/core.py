from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence


class EvalError(Exception):
    pass


class LengthMismatchError(EvalError):
    def __init__(self, truth: int, predicted: int) -> None:
        super().__init__(f"length mismatch: {truth} labels vs {predicted} predictions")


class EmptyInputError(EvalError):
    pass


def _validate_pair(y_true: Sequence[float], y_pred: Sequence[float]) -> None:
    if len(y_true) != len(y_pred):
        raise LengthMismatchError(len(y_true), len(y_pred))
    if not y_true:
        raise EmptyInputError("empty evaluation input")


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float
    higher_is_better: bool = True

    def __str__(self) -> str:
        return f"{self.name}={self.value:.4f}"


Metric = Callable[[Sequence[float], Sequence[float]], float]


def mean_squared_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    _validate_pair(y_true, y_pred)
    return sum((t - p) ** 2 for t, p in zip(y_true, y_pred)) / len(y_true)


def root_mean_squared_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return math.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    _validate_pair(y_true, y_pred)
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)


def r2_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    _validate_pair(y_true, y_pred)
    mean = sum(y_true) / len(y_true)
    ss_res = sum((t - p) ** 2 for t, p in zip(y_true, y_pred))
    ss_tot = sum((t - mean) ** 2 for t in y_true)
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def accuracy(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    _validate_pair(y_true, y_pred)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def precision_binary(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    _validate_pair(y_true, y_pred)
    predicted_positive = sum(1 for p in y_pred if p == 1)
    if predicted_positive == 0:
        return 0.0
    true_positive = sum(1 for t, p in zip(y_true, y_pred) if p == 1 and t == 1)
    return true_positive / predicted_positive


def recall_binary(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    _validate_pair(y_true, y_pred)
    actual_positive = sum(1 for t in y_true if t == 1)
    if actual_positive == 0:
        return 0.0
    true_positive = sum(1 for t, p in zip(y_true, y_pred) if p == 1 and t == 1)
    return true_positive / actual_positive


def f1_binary(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    prec = precision_binary(y_true, y_pred)
    rec = recall_binary(y_true, y_pred)
    if prec + rec == 0.0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


REGRESSION_METRICS: dict[str, Metric] = {
    "mse": mean_squared_error,
    "rmse": root_mean_squared_error,
    "mae": mean_absolute_error,
    "r2": r2_score,
}

CLASSIFICATION_METRICS: dict[str, Metric] = {
    "accuracy": accuracy,
    "precision": precision_binary,
    "recall": recall_binary,
    "f1": f1_binary,
}

HIGHER_IS_BETTER = {"r2", "accuracy", "precision", "recall", "f1"}


@dataclass(frozen=True)
class EvaluationReport:
    task: str
    results: tuple[MetricResult, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        parts = [str(result) for result in self.results]
        return f"[{self.task}] " + " | ".join(parts) if parts else f"[{self.task}] no metrics"

    def best(self, name: str) -> MetricResult | None:
        return next((r for r in self.results if r.name == name), None)


def evaluate(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    metrics: dict[str, Metric],
) -> EvaluationReport:
    results = []
    for name, metric in metrics.items():
        value = metric(y_true, y_pred)
        results.append(MetricResult(
            name=name,
            value=value,
            higher_is_better=name in HIGHER_IS_BETTER,
        ))
    return EvaluationReport(task="evaluation", results=tuple(results))


def evaluate_regression(y_true: Sequence[float], y_pred: Sequence[float]) -> EvaluationReport:
    report = evaluate(y_true, y_pred, REGRESSION_METRICS)
    return replace_task(report, "regression")


def evaluate_classification(y_true: Sequence[float], y_pred: Sequence[float]) -> EvaluationReport:
    report = evaluate(y_true, y_pred, CLASSIFICATION_METRICS)
    return replace_task(report, "classification")


def replace_task(report: EvaluationReport, task: str) -> EvaluationReport:
    return EvaluationReport(task=task, results=report.results)


def compare_models(
    baseline: EvaluationReport,
    candidate: EvaluationReport,
    metric_name: str,
) -> dict[str, float]:
    base = baseline.best(metric_name)
    cand = candidate.best(metric_name)
    if base is None or cand is None:
        raise EvalError(f"metric {metric_name!r} missing from a report")
    delta = (cand.value - base.value) * (1.0 if cand.higher_is_better else -1.0)
    return {
        "baseline": base.value,
        "candidate": cand.value,
        "improvement": round(delta, 6),
        "improved": float(delta > 0),
    }
