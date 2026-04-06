"""Build PyG heterogeneous graphs for the HGT pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch_geometric.data import HeteroData

try:
    from .data_loader import EdgeData, RawDataset
    from .split import DrugDiseaseSplit, build_train_graph_edges
except ImportError:  # pragma: no cover - allows direct script execution
    from data_loader import EdgeData, RawDataset
    from split import DrugDiseaseSplit, build_train_graph_edges


RELATION_MAPPING = {
    "drug_disease": ("drug", "treats", "disease"),
    "drug_protein": ("drug", "targets", "protein"),
    "protein_disease": ("protein", "associates", "disease"),
}

REVERSE_RELATION_MAPPING = {
    "drug_disease": ("disease", "treated_by", "drug"),
    "drug_protein": ("protein", "targeted_by", "drug"),
    "protein_disease": ("disease", "associated_by", "protein"),
}


@dataclass(frozen=True, slots=True)
class GraphBuildConfig:
    """Controls how the PyG heterograph is constructed."""

    add_reverse_edges: bool = True
    add_self_loops: bool = False
    self_loop_relation: str = "self_loop"
    attach_node_ids: bool = True
    validate_graph: bool = True


@dataclass(slots=True)
class GraphBuildReport:
    """Summary of the constructed heterograph."""

    dataset_name: str
    node_counts: dict[str, int]
    feature_dims: dict[str, int]
    edge_counts: dict[str, int]
    isolated_node_counts: dict[str, int]
    checks_passed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def build_full_hetero_graph(
    dataset: RawDataset,
    config: GraphBuildConfig = GraphBuildConfig(),
) -> tuple[HeteroData, GraphBuildReport]:
    """Build a full heterograph using every canonical relation."""

    return build_hetero_graph(dataset=dataset, edge_map=dataset.edges, config=config)


def build_train_hetero_graph(
    dataset: RawDataset,
    split: DrugDiseaseSplit,
    config: GraphBuildConfig = GraphBuildConfig(),
) -> tuple[HeteroData, GraphBuildReport]:
    """Build a training graph with only training positive drug-disease edges."""

    edge_map = build_train_graph_edges(dataset, split)
    return build_hetero_graph(dataset=dataset, edge_map=edge_map, config=config)


def build_hetero_graph(
    dataset: RawDataset,
    edge_map: dict[str, EdgeData],
    config: GraphBuildConfig = GraphBuildConfig(),
) -> tuple[HeteroData, GraphBuildReport]:
    """Build a PyG ``HeteroData`` object from canonical node features and edges."""

    data = HeteroData()
    _add_node_stores(data, dataset, config)

    edge_counts: dict[str, int] = {}
    for edge_key, edge_data in edge_map.items():
        if edge_key not in RELATION_MAPPING:
            raise ValueError(f"Unsupported relation key '{edge_key}'.")

        source_type, relation, target_type = RELATION_MAPPING[edge_key]
        edge_index = _to_edge_index(edge_data)
        data[(source_type, relation, target_type)].edge_index = edge_index
        edge_counts[_edge_type_key((source_type, relation, target_type))] = int(edge_index.shape[1])

        if config.add_reverse_edges:
            reverse_source, reverse_relation, reverse_target = REVERSE_RELATION_MAPPING[edge_key]
            reverse_edge_index = torch.stack((edge_index[1], edge_index[0]), dim=0)
            data[(reverse_source, reverse_relation, reverse_target)].edge_index = reverse_edge_index
            edge_counts[_edge_type_key((reverse_source, reverse_relation, reverse_target))] = int(
                reverse_edge_index.shape[1]
            )

    if config.add_self_loops:
        _add_self_loops(data, dataset, edge_counts, config)

    isolated_counts = _count_isolated_nodes(data, dataset)
    report = GraphBuildReport(
        dataset_name=dataset.dataset_name,
        node_counts=dataset.node_counts,
        feature_dims=dataset.feature_dims,
        edge_counts=edge_counts,
        isolated_node_counts=isolated_counts,
    )

    if config.validate_graph:
        _validate_hetero_graph(data, dataset, edge_map, config, report)

    return data, report


def summarize_graph_report(report: GraphBuildReport) -> dict[str, Any]:
    """Return a flat graph summary for logging or serialization."""

    return {
        "dataset_name": report.dataset_name,
        "node_counts": dict(report.node_counts),
        "feature_dims": dict(report.feature_dims),
        "edge_counts": dict(report.edge_counts),
        "isolated_node_counts": dict(report.isolated_node_counts),
        "checks_passed": list(report.checks_passed),
        "warnings": list(report.warnings),
    }


def _add_node_stores(data: HeteroData, dataset: RawDataset, config: GraphBuildConfig) -> None:
    for node_type, node_data in (
        ("drug", dataset.drugs),
        ("disease", dataset.diseases),
        ("protein", dataset.proteins),
    ):
        x = torch.as_tensor(node_data.features, dtype=torch.float32)
        data[node_type].x = x
        data[node_type].num_nodes = x.shape[0]
        if config.attach_node_ids:
            data[node_type].node_id = torch.arange(x.shape[0], dtype=torch.long)


def _to_edge_index(edge_data: EdgeData) -> torch.Tensor:
    return torch.stack(
        (
            torch.as_tensor(edge_data.source_index, dtype=torch.long),
            torch.as_tensor(edge_data.target_index, dtype=torch.long),
        ),
        dim=0,
    )


def _add_self_loops(
    data: HeteroData,
    dataset: RawDataset,
    edge_counts: dict[str, int],
    config: GraphBuildConfig,
) -> None:
    for node_type, num_nodes in dataset.node_counts.items():
        node_ids = torch.arange(num_nodes, dtype=torch.long)
        edge_index = torch.stack((node_ids, node_ids), dim=0)
        edge_type = (node_type, config.self_loop_relation, node_type)
        data[edge_type].edge_index = edge_index
        edge_counts[_edge_type_key(edge_type)] = int(num_nodes)


def _count_isolated_nodes(data: HeteroData, dataset: RawDataset) -> dict[str, int]:
    incident = {
        node_type: torch.zeros(num_nodes, dtype=torch.bool)
        for node_type, num_nodes in dataset.node_counts.items()
    }

    for source_type, relation, target_type in data.edge_types:
        edge_index = data[(source_type, relation, target_type)].edge_index
        incident[source_type][edge_index[0]] = True
        incident[target_type][edge_index[1]] = True

    return {
        node_type: int((~mask).sum().item())
        for node_type, mask in incident.items()
    }


def _validate_hetero_graph(
    data: HeteroData,
    dataset: RawDataset,
    edge_map: dict[str, EdgeData],
    config: GraphBuildConfig,
    report: GraphBuildReport,
) -> None:
    for node_type, node_data in (
        ("drug", dataset.drugs),
        ("disease", dataset.diseases),
        ("protein", dataset.proteins),
    ):
        x = data[node_type].x
        if x.ndim != 2:
            raise ValueError(f"{node_type} features in HeteroData must be a 2D tensor.")
        if x.shape[0] != node_data.num_nodes or x.shape[1] != node_data.feature_dim:
            raise ValueError(
                f"{node_type} feature shape mismatch: got {tuple(x.shape)}, "
                f"expected {(node_data.num_nodes, node_data.feature_dim)}."
            )
        report.checks_passed.append(f"{node_type} node store shape matches preprocessed dataset")

    for edge_key, edge_data in edge_map.items():
        edge_type = RELATION_MAPPING[edge_key]
        edge_index = data[edge_type].edge_index
        _validate_edge_index(edge_index, edge_type, dataset.node_counts)
        if int(edge_index.shape[1]) != edge_data.num_edges:
            raise ValueError(
                f"Edge count mismatch for {edge_type}: got {edge_index.shape[1]}, "
                f"expected {edge_data.num_edges}."
            )

        if config.add_reverse_edges:
            reverse_type = REVERSE_RELATION_MAPPING[edge_key]
            reverse_edge_index = data[reverse_type].edge_index
            _validate_edge_index(reverse_edge_index, reverse_type, dataset.node_counts)
            if reverse_edge_index.shape[1] != edge_index.shape[1]:
                raise ValueError(f"Reverse edge count mismatch for {reverse_type}.")
        report.checks_passed.append(
            f"{_edge_type_key(edge_type)} edge index validated"
        )

    if config.add_reverse_edges:
        report.checks_passed.append("reverse edges added for every canonical relation")
    else:
        report.warnings.append(
            "Reverse edges are disabled; this deviates from the project graph design."
        )

    if config.add_self_loops:
        report.checks_passed.append("self loops added for every node type")

    isolated_total = sum(report.isolated_node_counts.values())
    if isolated_total == 0:
        report.checks_passed.append("no isolated nodes detected in the constructed heterograph")
    else:
        report.warnings.append(
            "Isolated nodes detected: "
            + ", ".join(
                f"{node_type}={count}"
                for node_type, count in report.isolated_node_counts.items()
                if count > 0
            )
        )


def _validate_edge_index(
    edge_index: torch.Tensor,
    edge_type: tuple[str, str, str],
    node_counts: dict[str, int],
) -> None:
    source_type, _, target_type = edge_type
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"Edge index for {edge_type} must have shape [2, num_edges].")

    if edge_index.numel() == 0:
        return

    source_max = int(edge_index[0].max().item())
    target_max = int(edge_index[1].max().item())
    source_min = int(edge_index[0].min().item())
    target_min = int(edge_index[1].min().item())

    if source_min < 0 or target_min < 0:
        raise ValueError(f"Negative node indices found in edge type {edge_type}.")
    if source_max >= node_counts[source_type]:
        raise ValueError(f"Source indices exceed node count for {edge_type}.")
    if target_max >= node_counts[target_type]:
        raise ValueError(f"Target indices exceed node count for {edge_type}.")


def _edge_type_key(edge_type: tuple[str, str, str]) -> str:
    return "__".join(edge_type)
