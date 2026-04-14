"""Sparse top-k similarity graph builders for fusion-based drug-disease models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

try:
    from .data_loader import RawDataset
except ImportError:  # pragma: no cover - allows direct script execution
    from data_loader import RawDataset


@dataclass(frozen=True, slots=True)
class SimilarityGraphConfig:
    """Controls how dense AMDGT similarity matrices are converted into sparse graphs."""

    top_k: int = 20
    symmetric: bool = True
    drug_fingerprint_weight: float = 0.5
    drug_gip_weight: float = 0.5
    disease_ps_weight: float = 0.5
    disease_gip_weight: float = 0.5


@dataclass(slots=True)
class SimilarityGraphData:
    """One sparse homogeneous similarity graph."""

    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    num_nodes: int
    name: str

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])


@dataclass(slots=True)
class SimilarityGraphBundle:
    """Drug and disease similarity graphs built from AMDGT matrices."""

    drug: SimilarityGraphData
    disease: SimilarityGraphData
    config: SimilarityGraphConfig


def build_similarity_graph_bundle(
    dataset: RawDataset,
    config: SimilarityGraphConfig = SimilarityGraphConfig(),
) -> SimilarityGraphBundle:
    """Build sparse top-k similarity graphs for drug and disease nodes."""

    drug_fingerprint = _require_similarity_matrix(dataset, "DrugFingerprint")
    drug_gip = _require_similarity_matrix(dataset, "DrugGIP")
    disease_ps = _require_similarity_matrix(dataset, "DiseasePS")
    disease_gip = _require_similarity_matrix(dataset, "DiseaseGIP")

    drug_matrix = (
        config.drug_fingerprint_weight * drug_fingerprint
        + config.drug_gip_weight * drug_gip
    )
    disease_matrix = (
        config.disease_ps_weight * disease_ps
        + config.disease_gip_weight * disease_gip
    )

    return SimilarityGraphBundle(
        drug=_build_topk_similarity_graph(
            matrix=drug_matrix,
            top_k=config.top_k,
            symmetric=config.symmetric,
            name="drug_similarity",
        ),
        disease=_build_topk_similarity_graph(
            matrix=disease_matrix,
            top_k=config.top_k,
            symmetric=config.symmetric,
            name="disease_similarity",
        ),
        config=config,
    )


def summarize_similarity_graph_bundle(bundle: SimilarityGraphBundle) -> dict[str, Any]:
    """Return a compact summary useful for logging and JSON output."""

    return {
        "config": {
            "top_k": bundle.config.top_k,
            "symmetric": bundle.config.symmetric,
            "drug_fingerprint_weight": bundle.config.drug_fingerprint_weight,
            "drug_gip_weight": bundle.config.drug_gip_weight,
            "disease_ps_weight": bundle.config.disease_ps_weight,
            "disease_gip_weight": bundle.config.disease_gip_weight,
        },
        "drug": {
            "num_nodes": bundle.drug.num_nodes,
            "num_edges": bundle.drug.num_edges,
        },
        "disease": {
            "num_nodes": bundle.disease.num_nodes,
            "num_edges": bundle.disease.num_edges,
        },
    }


def _require_similarity_matrix(dataset: RawDataset, name: str) -> np.ndarray:
    if name not in dataset.similarity:
        raise ValueError(
            f"Dataset '{dataset.dataset_name}' does not contain similarity matrix '{name}'."
        )
    return np.asarray(dataset.similarity[name], dtype=np.float32)


def _build_topk_similarity_graph(
    *,
    matrix: np.ndarray,
    top_k: int,
    symmetric: bool,
    name: str,
) -> SimilarityGraphData:
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} matrix must be square, got shape {matrix.shape}.")

    values = np.asarray(matrix, dtype=np.float32).copy()
    np.fill_diagonal(values, 0.0)
    num_nodes = int(values.shape[0])

    directed_edges: dict[tuple[int, int], float] = {}
    for source_index in range(num_nodes):
        row = values[source_index]
        candidate_indices = np.flatnonzero(row > 0)
        if candidate_indices.size == 0:
            continue

        selected_indices = _select_topk_indices(row[candidate_indices], top_k=top_k)
        neighbor_indices = candidate_indices[selected_indices]
        neighbor_indices = neighbor_indices[np.argsort(row[neighbor_indices])[::-1]]

        for target_index in neighbor_indices.tolist():
            if source_index == int(target_index):
                continue
            weight = float(row[int(target_index)])
            if weight <= 0:
                continue
            key = (int(source_index), int(target_index))
            directed_edges[key] = max(directed_edges.get(key, 0.0), weight)

    if symmetric:
        pair_weights: dict[tuple[int, int], float] = {}
        for (source_index, target_index), weight in directed_edges.items():
            pair_key = (
                (source_index, target_index)
                if source_index < target_index
                else (target_index, source_index)
            )
            pair_weights[pair_key] = max(pair_weights.get(pair_key, 0.0), weight)

        sources: list[int] = []
        targets: list[int] = []
        weights: list[float] = []
        for (left, right), weight in sorted(pair_weights.items()):
            sources.extend((left, right))
            targets.extend((right, left))
            weights.extend((weight, weight))
    else:
        items = sorted(directed_edges.items())
        sources = [source for (source, _), _ in items]
        targets = [target for (_, target), _ in items]
        weights = [weight for _, weight in items]

    if not sources:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float32)
    else:
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        edge_attr = torch.tensor(weights, dtype=torch.float32).unsqueeze(-1)

    return SimilarityGraphData(
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_nodes,
        name=name,
    )


def _select_topk_indices(values: np.ndarray, *, top_k: int) -> np.ndarray:
    if values.size <= top_k:
        return np.arange(values.size, dtype=np.int64)
    partition_indices = np.argpartition(values, -top_k)[-top_k:]
    return partition_indices.astype(np.int64, copy=False)
