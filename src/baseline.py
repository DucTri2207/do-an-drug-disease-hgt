"""MLP baseline for drug-disease link prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

try:
    from .data_loader import RawDataset
    from .split import PairDataset
except ImportError:  # pragma: no cover - allows direct script execution
    from data_loader import RawDataset
    from split import PairDataset


@dataclass(frozen=True, slots=True)
class BaselineMLPConfig:
    """Hyperparameters for the MLP baseline."""

    hidden_dims: tuple[int, ...] = (256, 128)
    dropout: float = 0.2
    activation: str = "relu"


class DrugDiseaseMLPBaseline(nn.Module):
    """A simple baseline that scores drug-disease pairs from raw features only."""

    def __init__(
        self,
        drug_feature_dim: int,
        disease_feature_dim: int,
        config: BaselineMLPConfig = BaselineMLPConfig(),
    ) -> None:
        super().__init__()
        self.drug_feature_dim = int(drug_feature_dim)
        self.disease_feature_dim = int(disease_feature_dim)
        self.input_dim = self.drug_feature_dim + self.disease_feature_dim
        self.config = config

        activation = _make_activation(config.activation)
        layers: list[nn.Module] = []
        in_dim = self.input_dim
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation)
            layers.append(nn.Dropout(config.dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, pair_features: torch.Tensor) -> torch.Tensor:
        """Return one logit per drug-disease pair."""

        logits = self.mlp(pair_features)
        return logits.squeeze(-1)

    def score_pairs(
        self,
        drug_features: torch.Tensor,
        disease_features: torch.Tensor,
        drug_index: torch.Tensor,
        disease_index: torch.Tensor,
    ) -> torch.Tensor:
        pair_features = gather_pair_features(
            drug_features=drug_features,
            disease_features=disease_features,
            drug_index=drug_index,
            disease_index=disease_index,
        )
        return self(pair_features)


def build_baseline_model(
    dataset: RawDataset,
    config: BaselineMLPConfig = BaselineMLPConfig(),
) -> DrugDiseaseMLPBaseline:
    """Instantiate the baseline directly from dataset feature dimensions."""

    return DrugDiseaseMLPBaseline(
        drug_feature_dim=dataset.feature_dims["drug"],
        disease_feature_dim=dataset.feature_dims["disease"],
        config=config,
    )


def gather_pair_features(
    drug_features: torch.Tensor,
    disease_features: torch.Tensor,
    drug_index: torch.Tensor,
    disease_index: torch.Tensor,
) -> torch.Tensor:
    """Gather drug and disease features and concatenate them into pair features."""

    if drug_index.ndim != 1 or disease_index.ndim != 1:
        raise ValueError("drug_index and disease_index must be 1D tensors.")
    if drug_index.shape[0] != disease_index.shape[0]:
        raise ValueError("drug_index and disease_index must have the same batch size.")

    drug_batch = drug_features[drug_index]
    disease_batch = disease_features[disease_index]
    return torch.cat((drug_batch, disease_batch), dim=-1)


def build_pair_feature_tensor(
    dataset: RawDataset,
    pairs: PairDataset,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a labeled pair split into dense features and labels for the baseline."""

    target_device = torch.device(device) if device is not None else torch.device("cpu")
    drug_features = torch.as_tensor(dataset.drugs.features, dtype=torch.float32, device=target_device)
    disease_features = torch.as_tensor(
        dataset.diseases.features,
        dtype=torch.float32,
        device=target_device,
    )
    drug_index = torch.as_tensor(pairs.drug_index, dtype=torch.long, device=target_device)
    disease_index = torch.as_tensor(pairs.disease_index, dtype=torch.long, device=target_device)
    labels = torch.as_tensor(pairs.labels, dtype=torch.float32, device=target_device)

    pair_features = gather_pair_features(
        drug_features=drug_features,
        disease_features=disease_features,
        drug_index=drug_index,
        disease_index=disease_index,
    )
    return pair_features, labels


def build_all_candidate_pair_features(
    dataset: RawDataset,
    drug_index: int,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create pair features for one drug against all diseases, for top-k inference later."""

    target_device = torch.device(device) if device is not None else torch.device("cpu")
    num_diseases = dataset.node_counts["disease"]

    if not 0 <= int(drug_index) < dataset.node_counts["drug"]:
        raise ValueError(f"drug_index {drug_index} exceeds available drug nodes.")

    drug_indices = torch.full((num_diseases,), int(drug_index), dtype=torch.long, device=target_device)
    disease_indices = torch.arange(num_diseases, dtype=torch.long, device=target_device)
    drug_features = torch.as_tensor(dataset.drugs.features, dtype=torch.float32, device=target_device)
    disease_features = torch.as_tensor(
        dataset.diseases.features,
        dtype=torch.float32,
        device=target_device,
    )

    pair_features = gather_pair_features(
        drug_features=drug_features,
        disease_features=disease_features,
        drug_index=drug_indices,
        disease_index=disease_indices,
    )
    return pair_features, drug_indices, disease_indices


def summarize_pair_tensor(pair_features: torch.Tensor, labels: torch.Tensor) -> dict[str, int]:
    """Return a compact summary useful for smoke tests and logging."""

    return {
        "num_pairs": int(pair_features.shape[0]),
        "pair_feature_dim": int(pair_features.shape[1]),
        "num_positive": int(torch.count_nonzero(labels == 1).item()),
        "num_negative": int(torch.count_nonzero(labels == 0).item()),
    }


def _make_activation(name: str) -> nn.Module:
    normalized = name.strip().lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation '{name}'.")
