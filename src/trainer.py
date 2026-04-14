"""Training utilities for baseline and future graph models."""

from __future__ import annotations

import copy
from dataclasses import asdict, is_dataclass, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from .baseline import DrugDiseaseMLPBaseline
    from .evaluator import BinaryClassificationMetrics, evaluate_binary_classification
    from .model_hgt import DrugDiseaseHGT
    from .split import PairDataset
except ImportError:  # pragma: no cover - allows direct script execution
    from baseline import DrugDiseaseMLPBaseline
    from evaluator import BinaryClassificationMetrics, evaluate_binary_classification
    from model_hgt import DrugDiseaseHGT
    from split import PairDataset


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    """Hyperparameters shared by training loops."""

    device: str | None = None
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 20
    monitor_metric: str = "aupr"
    monitor_mode: str = "max"
    checkpoint_path: str | Path | None = None
    num_workers: int = 0
    loss_name: str = "bce"
    pos_weight: float | None = None
    focal_gamma: float = 2.0
    emulate_paper_leakage: bool = False
    artifact_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EpochRecord:
    """Metrics collected for one epoch."""

    epoch: int
    train_loss: float
    val_metrics: BinaryClassificationMetrics
    monitor_value: float


@dataclass(slots=True)
class TrainingResult:
    """Summary returned after training finishes or early-stops."""

    device: str
    best_epoch: int
    best_monitor_value: float
    best_val_metrics: BinaryClassificationMetrics
    epochs_completed: int
    checkpoint_path: str | None
    history: list[EpochRecord] = field(default_factory=list)


def train_baseline_model(
    model: DrugDiseaseMLPBaseline,
    train_pair_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_pair_features: torch.Tensor,
    val_labels: torch.Tensor,
    config: TrainerConfig = TrainerConfig(),
    test_pair_features: torch.Tensor | None = None,
    test_labels: torch.Tensor | None = None,
) -> TrainingResult:
    """Train the MLP baseline with early stopping on validation AUPR by default."""

    device = resolve_device(config.device)
    model = model.to(device)

    train_loader = _make_pair_loader(
        pair_features=train_pair_features,
        labels=train_labels,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = _make_pair_loader(
        pair_features=val_pair_features,
        labels=val_labels,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    criterion = _make_loss_fn(config, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_state_dict = copy.deepcopy(model.state_dict())
    best_metrics = BinaryClassificationMetrics(
        auc=float("nan"),
        aupr=float("nan"),
        accuracy=float("nan"),
        precision=float("nan"),
        recall=float("nan"),
        f1=float("nan"),
        mcc=float("nan"),
    )
    best_monitor_value = _initial_monitor_value(config.monitor_mode)
    best_epoch = 0
    patience_counter = 0
    history: list[EpochRecord] = []

    checkpoint_path = str(config.checkpoint_path) if config.checkpoint_path is not None else None

    for epoch in range(1, config.epochs + 1):
        train_loss = _train_baseline_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        if config.emulate_paper_leakage:
            if test_pair_features is None or test_labels is None:
                raise ValueError("test features and labels must be provided if emulate_paper_leakage is True.")
            eval_features = test_pair_features
            eval_labels = test_labels
        else:
            eval_features = val_pair_features
            eval_labels = val_labels

        val_metrics = evaluate_baseline_model(
            model=model,
            pair_features=eval_features,
            labels=eval_labels,
            batch_size=config.batch_size,
            device=device,
            loss_fn=criterion,
            num_workers=config.num_workers,
        )

        monitor_value = _extract_metric_value(val_metrics, config.monitor_metric)
        history.append(
            EpochRecord(
                epoch=epoch,
                train_loss=float(train_loss),
                val_metrics=val_metrics,
                monitor_value=float(monitor_value),
            )
        )

        if _is_improvement(
            current_value=monitor_value,
            best_value=best_monitor_value,
            mode=config.monitor_mode,
        ):
            best_monitor_value = float(monitor_value)
            best_metrics = val_metrics
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
            if checkpoint_path is not None:
                _save_checkpoint(
                    model=model,
                    checkpoint_path=Path(checkpoint_path),
                    epoch=epoch,
                    metrics=val_metrics,
                    config=config,
                )
        else:
            patience_counter += 1

        if patience_counter >= config.early_stopping_patience:
            break

    model.load_state_dict(best_state_dict)
    return TrainingResult(
        device=str(device),
        best_epoch=best_epoch,
        best_monitor_value=float(best_monitor_value),
        best_val_metrics=best_metrics,
        epochs_completed=len(history),
        checkpoint_path=checkpoint_path,
        history=history,
    )


@torch.no_grad()
def evaluate_baseline_model(
    model: DrugDiseaseMLPBaseline,
    pair_features: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int = 1024,
    device: str | torch.device | None = None,
    loss_fn: nn.Module | None = None,
    num_workers: int = 0,
) -> BinaryClassificationMetrics:
    """Evaluate the MLP baseline on a labeled pair tensor."""

    target_device = resolve_device(device)
    loader = _make_pair_loader(
        pair_features=pair_features,
        labels=labels,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model.eval()
    model = model.to(target_device)

    logits_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    total_loss = 0.0
    total_samples = 0

    for batch_features, batch_labels in loader:
        batch_features = batch_features.to(target_device)
        batch_labels = batch_labels.to(target_device)
        logits = model(batch_features)

        if loss_fn is not None:
            batch_loss = loss_fn(logits, batch_labels)
            batch_size_actual = int(batch_labels.shape[0])
            total_loss += float(batch_loss.item()) * batch_size_actual
            total_samples += batch_size_actual

        logits_batches.append(logits.detach().cpu())
        label_batches.append(batch_labels.detach().cpu())

    logits_all = torch.cat(logits_batches, dim=0)
    labels_all = torch.cat(label_batches, dim=0)
    average_loss = (total_loss / total_samples) if total_samples > 0 else None

    return evaluate_binary_classification(
        y_true=labels_all.numpy(),
        logits=logits_all.numpy(),
        loss=average_loss,
    )


def resolve_device(device: str | torch.device | None) -> torch.device:
    """Resolve the runtime device in a safe, portable way."""

    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def summarize_training_result(result: TrainingResult) -> dict[str, Any]:
    """Return a flat summary for logging or reporting."""

    return {
        "device": result.device,
        "best_epoch": result.best_epoch,
        "best_monitor_value": result.best_monitor_value,
        "epochs_completed": result.epochs_completed,
        "checkpoint_path": result.checkpoint_path,
        "best_val_metrics": result.best_val_metrics.to_dict(),
        "history": [
            {
                "epoch": record.epoch,
                "train_loss": record.train_loss,
                "monitor_value": record.monitor_value,
                "val_metrics": record.val_metrics.to_dict(),
            }
            for record in result.history
        ],
    }


def train_hgt_model(
    model: DrugDiseaseHGT,
    train_graph: Any,
    train_pairs: PairDataset,
    val_pairs: PairDataset,
    config: TrainerConfig = TrainerConfig(),
    test_pairs: PairDataset | None = None,
) -> TrainingResult:
    """Train the HGT model on the training graph with early stopping on validation."""

    device = resolve_device(config.device)
    model = model.to(device)
    train_graph = train_graph.to(device)

    train_loader = _make_index_loader(
        pairs=train_pairs,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = _make_index_loader(
        pairs=val_pairs,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    criterion = _make_loss_fn(config, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_state_dict = copy.deepcopy(model.state_dict())
    best_metrics = BinaryClassificationMetrics(
        auc=float("nan"),
        aupr=float("nan"),
        accuracy=float("nan"),
        precision=float("nan"),
        recall=float("nan"),
        f1=float("nan"),
        mcc=float("nan"),
    )
    best_monitor_value = _initial_monitor_value(config.monitor_mode)
    best_epoch = 0
    patience_counter = 0
    history: list[EpochRecord] = []

    checkpoint_path = str(config.checkpoint_path) if config.checkpoint_path is not None else None

    for epoch in range(1, config.epochs + 1):
        train_loss = _train_hgt_epoch(
            model=model,
            graph=train_graph,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        if config.emulate_paper_leakage:
            if test_pairs is None:
                raise ValueError("test_pairs must be provided if emulate_paper_leakage is True.")
            eval_pairs = test_pairs
        else:
            eval_pairs = val_pairs

        val_metrics = evaluate_hgt_model(
            model=model,
            graph=train_graph,
            pairs=eval_pairs,
            batch_size=config.batch_size,
            device=device,
            loss_fn=criterion,
            num_workers=config.num_workers,
        )

        monitor_value = _extract_metric_value(val_metrics, config.monitor_metric)
        history.append(
            EpochRecord(
                epoch=epoch,
                train_loss=float(train_loss),
                val_metrics=val_metrics,
                monitor_value=float(monitor_value),
            )
        )

        if _is_improvement(
            current_value=monitor_value,
            best_value=best_monitor_value,
            mode=config.monitor_mode,
        ):
            best_monitor_value = float(monitor_value)
            best_metrics = val_metrics
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0
            if checkpoint_path is not None:
                _save_checkpoint(
                    model=model,
                    checkpoint_path=Path(checkpoint_path),
                    epoch=epoch,
                    metrics=val_metrics,
                    config=config,
                )
        else:
            patience_counter += 1

        if patience_counter >= config.early_stopping_patience:
            break

    model.load_state_dict(best_state_dict)
    return TrainingResult(
        device=str(device),
        best_epoch=best_epoch,
        best_monitor_value=float(best_monitor_value),
        best_val_metrics=best_metrics,
        epochs_completed=len(history),
        checkpoint_path=checkpoint_path,
        history=history,
    )


@torch.no_grad()
def evaluate_hgt_model(
    model: DrugDiseaseHGT,
    graph: Any,
    pairs: PairDataset,
    *,
    batch_size: int = 1024,
    device: str | torch.device | None = None,
    loss_fn: nn.Module | None = None,
    num_workers: int = 0,
) -> BinaryClassificationMetrics:
    """Evaluate the HGT model on a labeled pair split using a fixed graph."""

    target_device = resolve_device(device)
    model.eval()
    model = model.to(target_device)
    graph = graph.to(target_device)

    loader = _make_index_loader(
        pairs=pairs,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    logits_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    total_loss = 0.0
    total_samples = 0

    for drug_index, disease_index, labels in loader:
        drug_index = drug_index.to(target_device)
        disease_index = disease_index.to(target_device)
        labels = labels.to(target_device)
        logits = model(graph, drug_index, disease_index)

        if loss_fn is not None:
            batch_loss = loss_fn(logits, labels)
            batch_size_actual = int(labels.shape[0])
            total_loss += float(batch_loss.item()) * batch_size_actual
            total_samples += batch_size_actual

        logits_batches.append(logits.detach().cpu())
        label_batches.append(labels.detach().cpu())

    logits_all = torch.cat(logits_batches, dim=0)
    labels_all = torch.cat(label_batches, dim=0)
    average_loss = (total_loss / total_samples) if total_samples > 0 else None

    return evaluate_binary_classification(
        y_true=labels_all.numpy(),
        logits=logits_all.numpy(),
        loss=average_loss,
    )


def _make_pair_loader(
    pair_features: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(
        pair_features.detach().cpu().float(),
        labels.detach().cpu().float(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def _make_index_loader(
    pairs: PairDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.as_tensor(pairs.drug_index, dtype=torch.long),
        torch.as_tensor(pairs.disease_index, dtype=torch.long),
        torch.as_tensor(pairs.labels, dtype=torch.float32),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def _train_baseline_epoch(
    model: DrugDiseaseMLPBaseline,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_features, batch_labels in loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        logits = model(batch_features)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()

        batch_size_actual = int(batch_labels.shape[0])
        total_loss += float(loss.item()) * batch_size_actual
        total_samples += batch_size_actual

    return total_loss / total_samples


def _train_hgt_epoch(
    model: DrugDiseaseHGT,
    graph: Any,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for drug_index, disease_index, labels in loader:
        drug_index = drug_index.to(device)
        disease_index = disease_index.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(graph, drug_index, disease_index)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size_actual = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size_actual
        total_samples += batch_size_actual

    return total_loss / total_samples


def _make_loss_fn(config: TrainerConfig, device: torch.device) -> nn.Module:
    normalized = config.loss_name.strip().lower()
    if normalized == "bce":
        if config.pos_weight is None:
            return nn.BCEWithLogitsLoss()
        pos_weight = torch.tensor(float(config.pos_weight), dtype=torch.float32, device=device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if normalized == "focal":
        return FocalBCEWithLogitsLoss(gamma=config.focal_gamma)
    raise ValueError(f"Unsupported loss_name '{config.loss_name}'.")


class FocalBCEWithLogitsLoss(nn.Module):
    """Binary focal loss built on top of BCE-with-logits."""

    def __init__(self, gamma: float = 2.0, alpha: float | None = None) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_factor = (1.0 - p_t).pow(self.gamma)
        loss = bce * focal_factor

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = loss * alpha_t

        return loss.mean()


def _extract_metric_value(metrics: BinaryClassificationMetrics, metric_name: str) -> float:
    if not hasattr(metrics, metric_name):
        raise ValueError(f"Metric '{metric_name}' does not exist on BinaryClassificationMetrics.")
    return float(getattr(metrics, metric_name))


def _initial_monitor_value(mode: str) -> float:
    normalized = mode.strip().lower()
    if normalized == "max":
        return float("-inf")
    if normalized == "min":
        return float("inf")
    raise ValueError(f"Unsupported monitor mode '{mode}'.")


def _is_improvement(current_value: float, best_value: float, mode: str) -> bool:
    normalized = mode.strip().lower()
    if normalized == "max":
        return current_value > best_value
    if normalized == "min":
        return current_value < best_value
    raise ValueError(f"Unsupported monitor mode '{mode}'.")


def _save_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    epoch: int,
    metrics: BinaryClassificationMetrics,
    config: TrainerConfig,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_class": model.__class__.__name__,
        "model_state_dict": model.state_dict(),
        "metrics": metrics.to_dict(),
        "trainer_config": {
            "device": config.device,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "early_stopping_patience": config.early_stopping_patience,
            "monitor_metric": config.monitor_metric,
            "monitor_mode": config.monitor_mode,
        },
    }
    if config.artifact_metadata:
        payload["artifact_metadata"] = dict(config.artifact_metadata)
    if hasattr(model, "config"):
        model_config = getattr(model, "config")
        if is_dataclass(model_config):
            payload["model_config"] = asdict(model_config)
    if hasattr(model, "input_dims"):
        payload["input_dims"] = dict(getattr(model, "input_dims"))
    if hasattr(model, "metadata"):
        payload["metadata"] = getattr(model, "metadata")
    if hasattr(model, "drug_feature_dim"):
        payload["drug_feature_dim"] = int(getattr(model, "drug_feature_dim"))
    if hasattr(model, "disease_feature_dim"):
        payload["disease_feature_dim"] = int(getattr(model, "disease_feature_dim"))
    torch.save(payload, checkpoint_path)
