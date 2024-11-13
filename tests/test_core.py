import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from ml_eval_kit import (
    EmptyInputError,
    EvalError,
    LengthMismatchError,
    accuracy,
    compare_models,
    evaluate_regression,
    f1_binary,
    mean_absolute_error,
    mean_squared_error,
    precision_binary,
    r2_score,
    recall_binary,
    root_mean_squared_error,
)


def test_mse_known_value():
    assert mean_squared_error([1, 2, 3], [1, 2, 3]) == 0.0
    assert mean_squared_error([0, 0], [1, 1]) == pytest.approx(1.0)


def test_rmse_is_sqrt_of_mse():
    truth, pred = [0, 0, 4], [1, 2, 0]
    assert root_mean_squared_error(truth, pred) == pytest.approx(
        mean_squared_error(truth, pred) ** 0.5
    )


def test_mae_basic():
    assert mean_absolute_error([2.0], [1.0]) == pytest.approx(1.0)


def test_r2_perfect_fit():
    assert r2_score([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_r2_constant_truth_edge():
    assert r2_score([5, 5], [5, 4]) == 0.0
    assert r2_score([5, 5], [5, 5]) == 1.0


def test_accuracy_counts_exact_matches():
    assert accuracy([1, 0, 1, 1], [1, 0, 0, 1]) == pytest.approx(0.75)


def test_precision_recall_f1():
    truth = [1, 0, 1, 1, 0]
    pred = [1, 1, 1, 0, 0]
    assert precision_binary(truth, pred) == pytest.approx(2 / 3)
    assert recall_binary(truth, pred) == pytest.approx(2 / 3)
    assert f1_binary(truth, pred) == pytest.approx(2 / 3)


def test_precision_zero_division_safe():
    assert precision_binary([0, 0], [1, 1]) == 0.0
    assert recall_binary([0, 0], [1, 1]) == 0.0
    assert f1_binary([0, 0], [1, 1]) == 0.0


def test_length_mismatch_raises():
    with pytest.raises(LengthMismatchError):
        mean_squared_error([1, 2], [1])


def test_empty_input_raises():
    with pytest.raises(EmptyInputError):
        accuracy([], [])


def test_regression_report_structure():
    report = evaluate_regression([1, 2, 3], [1.1, 1.9, 3.2])
    names = {result.name for result in report.results}
    assert {"mse", "rmse", "mae", "r2"} <= names
    assert report.summary().startswith("[regression]")


def test_compare_models_detects_improvement():
    baseline = evaluate_regression([1, 2, 3], [1.5, 2.5, 3.5])
    candidate = evaluate_regression([1, 2, 3], [1.1, 2.1, 3.1])
    comparison = compare_models(baseline, candidate, "mae")
    assert comparison["improved"] == 1.0
    assert comparison["improvement"] > 0


def test_compare_missing_metric_raises():
    baseline = evaluate_regression([1], [1])
    with pytest.raises(EvalError):
        compare_models(baseline, baseline, "f1")
