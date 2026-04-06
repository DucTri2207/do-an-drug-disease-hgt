"""Evaluation helpers for drug-disease link prediction."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    import torch
except ImportError:  # pragma: no cover - for tooling environments without torch
    torch = None


@dataclass(slots=True)
class BinaryClassificationMetrics:
    """Standard metrics for binary link prediction."""

    auc: float
    aupr: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    mcc: float
    loss: float | None = None
    threshold: float = 0.5
    num_samples: int = 0
    num_positive: int = 0
    num_negative: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_binary_classification(
    y_true: np.ndarray | list[int],
    *,
    logits: np.ndarray | list[float] | None = None,
    probabilities: np.ndarray | list[float] | None = None,
    threshold: float = 0.5,
    loss: float | None = None,
) -> BinaryClassificationMetrics:
    """Compute binary classification metrics from logits or probabilities."""

    y_true_np = _to_numpy_1d(y_true, dtype=np.int64)
    if logits is None and probabilities is None:
        raise ValueError("Either logits or probabilities must be provided.")

    if probabilities is None:
        logits_np = _to_numpy_1d(logits, dtype=np.float32)
        probabilities_np = sigmoid_numpy(logits_np)
    else:
        probabilities_np = _to_numpy_1d(probabilities, dtype=np.float32)

    if y_true_np.shape[0] != probabilities_np.shape[0]:
        raise ValueError(
            f"Label count {y_true_np.shape[0]} does not match prediction count {probabilities_np.shape[0]}."
        )

    y_pred_np = (probabilities_np >= threshold).astype(np.int64)

    auc = _safe_roc_auc(y_true_np, probabilities_np)
    aupr = _safe_average_precision(y_true_np, probabilities_np)

    metrics = BinaryClassificationMetrics(
        auc=auc,
        aupr=aupr,
        accuracy=float(accuracy_score(y_true_np, y_pred_np)),
        precision=float(precision_score(y_true_np, y_pred_np, zero_division=0)),
        recall=float(recall_score(y_true_np, y_pred_np, zero_division=0)),
        f1=float(f1_score(y_true_np, y_pred_np, zero_division=0)),
        mcc=float(matthews_corrcoef(y_true_np, y_pred_np)),
        loss=loss,
        threshold=float(threshold),
        num_samples=int(y_true_np.shape[0]),
        num_positive=int(np.count_nonzero(y_true_np == 1)),
        num_negative=int(np.count_nonzero(y_true_np == 0)),
    )
    return metrics


def summarize_metrics(metrics: BinaryClassificationMetrics) -> dict[str, Any]:
    """Return a flat metric summary for logging, JSON, or tables."""

    return metrics.to_dict()


def sigmoid_numpy(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for numpy arrays."""

    logits = np.asarray(logits, dtype=np.float32)
    positive_mask = logits >= 0
    negative_mask = ~positive_mask

    result = np.empty_like(logits, dtype=np.float32)
    result[positive_mask] = 1.0 / (1.0 + np.exp(-logits[positive_mask]))
    exp_logits = np.exp(logits[negative_mask])
    result[negative_mask] = exp_logits / (1.0 + exp_logits)
    return result


def tensor_to_numpy(array_like: Any) -> np.ndarray:
    """Convert a torch tensor or array-like object into a numpy array."""

    if torch is not None and isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def _to_numpy_1d(values: Any, dtype: np.dtype) -> np.ndarray:
    array = tensor_to_numpy(values).astype(dtype, copy=False)
    array = np.ravel(array)
    return array


def _safe_roc_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    if np.unique(y_true).shape[0] < 2:
        return float("nan")
    return float(roc_auc_score(y_true, probabilities))


def _safe_average_precision(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    if np.unique(y_true).shape[0] < 2:
        return float("nan")
    return float(average_precision_score(y_true, probabilities))
