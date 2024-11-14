# ml-evaluation-kit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Regression and binary-classification metrics with typed errors, frozen reports, and model-vs-model comparison — the evaluation layer that doesn't silently average garbage.

## 🚀 Overview

Evaluation code fails quietly: mismatched lengths broadcast, empty inputs return NaN, and zero-division turns into mystery numbers. `ml-evaluation-kit` validates every pair up front (`LengthMismatchError`, `EmptyInputError`), computes standard metrics exactly (MSE/RMSE/MAE/R² · accuracy/precision/recall/F1), wraps results in a frozen `EvaluationReport` with a one-line summary, and compares two reports to answer "did candidate actually beat baseline?"

## ✨ Features

- **Regression metrics:** MSE, RMSE, MAE, R² — including constant-truth edge handling
- **Classification metrics:** accuracy, binary precision/recall/F1 with explicit zero-division → 0.0
- **Fail-fast validation:** length mismatch and empty input raise before any math
- **Frozen reports:** `MetricResult(name, value, higher_is_better)` + `summary()` one-liner
- **Model comparison:** `compare_models()` respects metric direction; returns signed improvement
- **Zero dependencies**

## 🚧 Structure

```
ml-evaluation-kit/
├── src/ml_eval_kit/
│   ├── __init__.py
│   └── core.py
├── tests/
│   └── test_core.py
├── README.md
└── pyproject.toml
```

## 📦 Installation

```bash
git clone https://github.com/supremeloki/ml-evaluation-kit.git
cd ml-evaluation-kit
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## 📋 Requirements

- Python 3.11+
- No runtime dependencies

## 🏃 Quick Start

```python
from ml_eval_kit import compare_models, evaluate_classification, evaluate_regression

reg = evaluate_regression([1.0, 2.0, 3.0], [1.1, 1.9, 3.2])
print(reg.summary())

cls = evaluate_classification([1, 0, 1, 1], [1, 1, 1, 0])
print(cls.best("f1"))

verdict = compare_models(baseline_report, candidate_report, metric_name="mae")
print(verdict["improved"])
```

## 🔧 Error Handling

```text
EvalError
├── LengthMismatchError    # truth/prediction counts differ
└── EmptyInputError        # nothing to evaluate
```

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📝 Code Quality

- Full type hints (`X | None` style), frozen dataclasses throughout
- Zero comments — names carry the meaning
- Hand-computed expected values in every test

## 📄 License

MIT — see [LICENSE](LICENSE).

## 👤 Author

**Kooroush Masoumi**

---

⭐ Star this repo if you find it useful!
