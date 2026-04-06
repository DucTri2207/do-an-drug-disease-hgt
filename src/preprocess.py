"""Dataset validation and lightweight preprocessing for AMDGT data."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

try:
    from .data_loader import EdgeData, NodeData, RawDataset
except ImportError:  # pragma: no cover - allows direct script execution
    from data_loader import EdgeData, NodeData, RawDataset


NormalizeMode = Literal["none", "zscore", "l2"]


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    """Controls non-destructive cleanup before graph building."""

    normalize_features: NormalizeMode = "none"
    replace_non_finite_with: float = 0.0
    deduplicate_edges: bool = True
    validate_auxiliary: bool = True


@dataclass(slots=True)
class ValidationReport:
    """Compact summary of dataset integrity after preprocessing."""

    node_counts: dict[str, int]
    feature_dims: dict[str, int]
    edge_counts: dict[str, int]
    similarity_shapes: dict[str, tuple[int, int]]
    checks_passed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def preprocess_dataset(
    dataset: RawDataset,
    config: PreprocessConfig = PreprocessConfig(),
) -> tuple[RawDataset, ValidationReport]:
    """Clean features, canonicalize edges, and validate index alignment."""

    drugs = _preprocess_node_data(
        dataset.drugs,
        normalize_mode=config.normalize_features,
        replace_non_finite_with=config.replace_non_finite_with,
    )
    diseases = _preprocess_node_data(
        dataset.diseases,
        normalize_mode=config.normalize_features,
        replace_non_finite_with=config.replace_non_finite_with,
    )
    proteins = _preprocess_node_data(
        dataset.proteins,
        normalize_mode=config.normalize_features,
        replace_non_finite_with=config.replace_non_finite_with,
    )

    edges = {
        name: _preprocess_edge_data(edge, deduplicate=config.deduplicate_edges)
        for name, edge in dataset.edges.items()
    }

    similarity = {
        name: _sanitize_feature_matrix(matrix, config.replace_non_finite_with)
        for name, matrix in dataset.similarity.items()
    }

    processed = RawDataset(
        dataset_name=dataset.dataset_name,
        dataset_dir=dataset.dataset_dir,
        drugs=drugs,
        diseases=diseases,
        proteins=proteins,
        edges=edges,
        similarity=similarity,
        auxiliary=dataset.auxiliary,
    )
    report = validate_dataset_alignment(processed, validate_auxiliary=config.validate_auxiliary)
    return processed, report


def validate_dataset_alignment(
    dataset: RawDataset,
    validate_auxiliary: bool = True,
) -> ValidationReport:
    """Raise on alignment issues and return a compact integrity summary."""

    checks: list[str] = []
    warnings: list[str] = []

    node_lookup = {
        "drug": dataset.drugs,
        "disease": dataset.diseases,
        "protein": dataset.proteins,
    }
    node_counts = dataset.node_counts
    feature_dims = dataset.feature_dims
    edge_counts = {name: edge.num_edges for name, edge in dataset.edges.items()}
    similarity_shapes = {
        name: (int(matrix.shape[0]), int(matrix.shape[1]))
        for name, matrix in dataset.similarity.items()
    }

    for node_type, node_data in node_lookup.items():
        _validate_node_data(node_type, node_data)
        checks.append(f"{node_type} features/labels/metadata aligned")

    for name, edge_data in dataset.edges.items():
        _validate_edge_bounds(edge_data, node_counts)
        checks.append(f"{name} edge indices within node ranges")

    for name, matrix in dataset.similarity.items():
        expected = _expected_similarity_size(name, node_counts)
        if matrix.shape != (expected, expected):
            raise ValueError(
                f"Similarity matrix '{name}' has shape {matrix.shape}, "
                f"expected {(expected, expected)}."
            )
        checks.append(f"{name} similarity shape matches node count")

    if validate_auxiliary:
        _validate_auxiliary_exports(dataset, checks, warnings)

    return ValidationReport(
        node_counts=node_counts,
        feature_dims=feature_dims,
        edge_counts=edge_counts,
        similarity_shapes=similarity_shapes,
        checks_passed=checks,
        warnings=warnings,
    )


def _preprocess_node_data(
    node_data: NodeData,
    normalize_mode: NormalizeMode,
    replace_non_finite_with: float,
) -> NodeData:
    features = _sanitize_feature_matrix(node_data.features, replace_non_finite_with)
    features = _normalize_feature_matrix(features, normalize_mode)
    metadata = {key: list(values) for key, values in node_data.metadata.items()}
    return NodeData(features=features, labels=list(node_data.labels), metadata=metadata)


def _preprocess_edge_data(edge_data: EdgeData, deduplicate: bool) -> EdgeData:
    source_index = np.asarray(edge_data.source_index, dtype=np.int64)
    target_index = np.asarray(edge_data.target_index, dtype=np.int64)
    if deduplicate:
        unique_pairs = np.unique(
            np.column_stack((source_index, target_index)),
            axis=0,
        )
        source_index = unique_pairs[:, 0]
        target_index = unique_pairs[:, 1]

    return EdgeData(
        source_index=source_index,
        target_index=target_index,
        source_type=edge_data.source_type,
        relation=edge_data.relation,
        target_type=edge_data.target_type,
    )


def _sanitize_feature_matrix(matrix: np.ndarray, replace_non_finite_with: float) -> np.ndarray:
    clean = np.asarray(matrix, dtype=np.float32).copy()
    non_finite_mask = ~np.isfinite(clean)
    if np.any(non_finite_mask):
        clean[non_finite_mask] = np.float32(replace_non_finite_with)
    return clean


def _normalize_feature_matrix(matrix: np.ndarray, mode: NormalizeMode) -> np.ndarray:
    if mode == "none":
        return matrix
    if mode == "zscore":
        mean = matrix.mean(axis=0, keepdims=True)
        std = matrix.std(axis=0, keepdims=True)
        std[std == 0] = 1.0
        return (matrix - mean) / std
    if mode == "l2":
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms
    raise ValueError(f"Unsupported normalize mode: {mode}")


def _validate_node_data(node_type: str, node_data: NodeData) -> None:
    if node_data.features.ndim != 2:
        raise ValueError(f"{node_type} features must be a 2D matrix.")
    if node_data.num_nodes != len(node_data.labels):
        raise ValueError(
            f"{node_type} label count {len(node_data.labels)} does not match "
            f"feature rows {node_data.num_nodes}."
        )
    for key, values in node_data.metadata.items():
        if len(values) != node_data.num_nodes:
            raise ValueError(
                f"{node_type} metadata column '{key}' has {len(values)} rows, "
                f"expected {node_data.num_nodes}."
            )

    if node_type in {"drug", "protein"}:
        try:
            labels_as_int = np.asarray([int(label) for label in node_data.labels], dtype=np.int64)
        except ValueError as exc:
            raise ValueError(
                f"{node_type} labels should be local integer indices in AMDGT data."
            ) from exc

        expected = np.arange(node_data.num_nodes, dtype=np.int64)
        if not np.array_equal(labels_as_int, expected):
            raise ValueError(
                f"{node_type} labels are not sequential local indices 0..{node_data.num_nodes - 1}."
            )


def _validate_edge_bounds(edge_data: EdgeData, node_counts: dict[str, int]) -> None:
    source_count = node_counts[edge_data.source_type]
    target_count = node_counts[edge_data.target_type]

    if edge_data.source_index.ndim != 1 or edge_data.target_index.ndim != 1:
        raise ValueError(f"{edge_data.relation} edge indices must be 1D arrays.")
    if edge_data.source_index.shape[0] != edge_data.target_index.shape[0]:
        raise ValueError(f"{edge_data.relation} edge sources and targets have mismatched lengths.")
    if np.any(edge_data.source_index < 0) or np.any(edge_data.source_index >= source_count):
        raise ValueError(
            f"{edge_data.relation} source indices exceed {edge_data.source_type} node bounds."
        )
    if np.any(edge_data.target_index < 0) or np.any(edge_data.target_index >= target_count):
        raise ValueError(
            f"{edge_data.relation} target indices exceed {edge_data.target_type} node bounds."
        )


def _expected_similarity_size(name: str, node_counts: dict[str, int]) -> int:
    if name.startswith("Drug"):
        return node_counts["drug"]
    if name.startswith("Disease"):
        return node_counts["disease"]
    raise ValueError(f"Unknown similarity matrix prefix for '{name}'.")


def _validate_auxiliary_exports(
    dataset: RawDataset,
    checks: list[str],
    warnings: list[str],
) -> None:
    auxiliary = dataset.auxiliary
    node_counts = dataset.node_counts

    if "drug_disease_adjacency" in auxiliary:
        adjacency = auxiliary["drug_disease_adjacency"]["matrix"]
        expected_shape = (node_counts["drug"], node_counts["disease"])
        if adjacency.shape != expected_shape:
            raise ValueError(
                f"adj.csv matrix has shape {adjacency.shape}, expected {expected_shape}."
            )
        edge_total = int(dataset.edges["drug_disease"].num_edges)
        adjacency_nonzero = int(np.count_nonzero(adjacency))
        if adjacency_nonzero == edge_total:
            checks.append("adj.csv matches canonical drug-disease edges")
        else:
            warnings.append(
                "adj.csv non-zero count does not match canonical drug-disease "
                f"edges ({adjacency_nonzero} vs {edge_total}); edge list will be treated as source of truth."
            )

    if "all_nodes" in auxiliary:
        all_nodes = auxiliary["all_nodes"]
        expected_total = sum(node_counts.values())
        indices = np.asarray(all_nodes["indices"], dtype=np.int64)
        if indices.shape[0] != expected_total:
            raise ValueError(
                f"Allnode.csv has {indices.shape[0]} rows, expected {expected_total}."
            )
        if not np.array_equal(indices, np.arange(expected_total, dtype=np.int64)):
            raise ValueError("Allnode.csv global indices are not sequential 0..N-1.")
        checks.append("Allnode.csv global node indexing is sequential")

    if "all_edges" in auxiliary:
        all_edges = np.asarray(auxiliary["all_edges"], dtype=np.int64)
        unique_all_edges = np.unique(all_edges, axis=0)
        expected_total = sum(edge.num_edges for edge in dataset.edges.values())
        if unique_all_edges.shape[0] != expected_total:
            raise ValueError(
                f"Alledge.csv has {unique_all_edges.shape[0]} unique rows, expected {expected_total}."
            )
        if unique_all_edges.shape[0] != all_edges.shape[0]:
            warnings.append(
                f"Alledge.csv contains {all_edges.shape[0] - unique_all_edges.shape[0]} duplicated edges."
            )

        offsets = {
            "drug": 0,
            "disease": node_counts["drug"],
            "protein": node_counts["drug"] + node_counts["disease"],
        }
        type_counts = Counter()
        for source, target in unique_all_edges:
            source_type = _classify_global_index(source, node_counts, offsets)
            target_type = _classify_global_index(target, node_counts, offsets)
            type_counts[(source_type, target_type)] += 1

        expected_drug_disease = dataset.edges["drug_disease"].num_edges
        expected_drug_protein = dataset.edges["drug_protein"].num_edges
        expected_protein_disease = dataset.edges["protein_disease"].num_edges

        if type_counts.get(("drug", "disease"), 0) != expected_drug_disease:
            raise ValueError("Alledge.csv drug-disease edges do not match canonical counts.")
        if type_counts.get(("drug", "protein"), 0) != expected_drug_protein:
            raise ValueError("Alledge.csv drug-protein edges do not match canonical counts.")

        protein_disease_like = (
            type_counts.get(("protein", "disease"), 0)
            + type_counts.get(("disease", "protein"), 0)
        )
        if protein_disease_like != expected_protein_disease:
            raise ValueError(
                "Alledge.csv protein-disease style edges do not match canonical counts."
            )
        checks.append("Alledge.csv matches canonical edge partitions up to stored direction")


def _classify_global_index(
    index: int,
    node_counts: dict[str, int],
    offsets: dict[str, int],
) -> str:
    if 0 <= index < node_counts["drug"]:
        return "drug"
    if offsets["disease"] <= index < offsets["protein"]:
        return "disease"
    if offsets["protein"] <= index < offsets["protein"] + node_counts["protein"]:
        return "protein"
    raise ValueError(f"Global node index {index} exceeds the merged graph size.")
