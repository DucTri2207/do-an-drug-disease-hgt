"""Prepare tensor features and lookup tables for training and inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch

try:
    from .data_loader import RawDataset
except ImportError:  # pragma: no cover - allows direct script execution
    from data_loader import RawDataset


NodeType = Literal["drug", "disease", "protein"]

QUERY_METADATA_FIELDS: dict[str, tuple[str, ...]] = {
    "drug": ("name", "drugbank_id"),
    "disease": (),
    "protein": ("uniprot_id",),
}

DISPLAY_METADATA_FIELDS: dict[str, tuple[str, ...]] = {
    "drug": ("name", "drugbank_id"),
    "disease": (),
    "protein": ("uniprot_id",),
}


@dataclass(frozen=True, slots=True)
class FeatureBuilderConfig:
    """Controls tensor placement and string normalization for lookups."""

    device: str | torch.device | None = None
    normalize_queries: bool = True


@dataclass(slots=True)
class NodeLookupTable:
    """Lookup helpers for one node type."""

    node_type: str
    labels: list[str]
    metadata: dict[str, list[str]]
    normalize_queries: bool = True
    query_to_indices: dict[str, list[int]] = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return len(self.labels)


@dataclass(slots=True)
class FeatureBundle:
    """Tensor and lookup bundle reused by inference and future web export."""

    dataset_name: str
    device: str
    tensors: dict[str, torch.Tensor]
    lookups: dict[str, NodeLookupTable]

    @property
    def node_counts(self) -> dict[str, int]:
        return {
            node_type: int(tensor.shape[0])
            for node_type, tensor in self.tensors.items()
        }

    @property
    def feature_dims(self) -> dict[str, int]:
        return {
            node_type: int(tensor.shape[1])
            for node_type, tensor in self.tensors.items()
        }


def build_feature_tensor_dict(
    dataset: RawDataset,
    device: str | torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Convert dataset feature matrices into torch tensors by node type."""

    target_device = torch.device(device) if device is not None else torch.device("cpu")
    return {
        "drug": torch.as_tensor(dataset.drugs.features, dtype=torch.float32, device=target_device),
        "disease": torch.as_tensor(
            dataset.diseases.features,
            dtype=torch.float32,
            device=target_device,
        ),
        "protein": torch.as_tensor(
            dataset.proteins.features,
            dtype=torch.float32,
            device=target_device,
        ),
    }


def build_feature_bundle(
    dataset: RawDataset,
    config: FeatureBuilderConfig = FeatureBuilderConfig(),
) -> FeatureBundle:
    """Build reusable feature tensors plus lookup tables for every node type."""

    tensors = build_feature_tensor_dict(dataset, device=config.device)
    lookups = {
        "drug": _build_node_lookup(
            node_type="drug",
            labels=dataset.drugs.labels,
            metadata=dataset.drugs.metadata,
            normalize_queries=config.normalize_queries,
        ),
        "disease": _build_node_lookup(
            node_type="disease",
            labels=dataset.diseases.labels,
            metadata=dataset.diseases.metadata,
            normalize_queries=config.normalize_queries,
        ),
        "protein": _build_node_lookup(
            node_type="protein",
            labels=dataset.proteins.labels,
            metadata=dataset.proteins.metadata,
            normalize_queries=config.normalize_queries,
        ),
    }
    return FeatureBundle(
        dataset_name=dataset.dataset_name,
        device=str(next(iter(tensors.values())).device),
        tensors=tensors,
        lookups=lookups,
    )


def resolve_node_query(
    bundle: FeatureBundle,
    node_type: NodeType,
    query: str | int,
) -> int:
    """Resolve a user or API query into a local node index."""

    lookup = bundle.lookups[node_type]
    if isinstance(query, int):
        index = int(query)
        _validate_node_index(lookup, node_type, index)
        return index

    normalized_query = _normalize_lookup_value(
        str(query),
        normalize_queries=lookup.normalize_queries,
    )
    if normalized_query in lookup.query_to_indices:
        indices = lookup.query_to_indices[normalized_query]
        if len(indices) > 1:
            raise ValueError(
                f"Query '{query}' matches multiple {node_type} indices: {indices[:5]}"
            )
        return int(indices[0])

    if normalized_query.isdigit():
        index = int(normalized_query)
        _validate_node_index(lookup, node_type, index)
        return index

    raise KeyError(f"Could not resolve {node_type} query '{query}'.")


def get_node_record(
    bundle: FeatureBundle,
    node_type: NodeType,
    index: int,
) -> dict[str, Any]:
    """Return a serializable record for one node."""

    lookup = bundle.lookups[node_type]
    _validate_node_index(lookup, node_type, index)

    record: dict[str, Any] = {
        "node_type": node_type,
        "index": int(index),
        "label": lookup.labels[index],
    }
    for key, values in lookup.metadata.items():
        record[key] = values[index]
    record["display_name"] = _select_display_name(node_type, lookup, index)
    return record


def export_node_table(bundle: FeatureBundle, node_type: NodeType) -> list[dict[str, Any]]:
    """Return every node record as a serializable table for web lookup exports."""

    lookup = bundle.lookups[node_type]
    return [get_node_record(bundle, node_type, index) for index in range(lookup.num_nodes)]


def summarize_feature_bundle(bundle: FeatureBundle) -> dict[str, Any]:
    """Return a compact bundle summary for debugging and logs."""

    return {
        "dataset_name": bundle.dataset_name,
        "device": bundle.device,
        "node_counts": bundle.node_counts,
        "feature_dims": bundle.feature_dims,
    }


def _build_node_lookup(
    node_type: str,
    labels: list[str],
    metadata: dict[str, list[str]],
    *,
    normalize_queries: bool,
) -> NodeLookupTable:
    query_to_indices: dict[str, list[int]] = {}

    for index, label in enumerate(labels):
        _register_query_value(
            query_to_indices,
            label,
            index,
            normalize_queries=normalize_queries,
        )
        _register_query_value(
            query_to_indices,
            str(index),
            index,
            normalize_queries=normalize_queries,
        )

        for field_name in QUERY_METADATA_FIELDS.get(node_type, ()):
            values = metadata.get(field_name)
            if values is None:
                continue
            _register_query_value(
                query_to_indices,
                values[index],
                index,
                normalize_queries=normalize_queries,
            )

    return NodeLookupTable(
        node_type=node_type,
        labels=list(labels),
        metadata={key: list(values) for key, values in metadata.items()},
        normalize_queries=normalize_queries,
        query_to_indices=query_to_indices,
    )


def _register_query_value(
    query_to_indices: dict[str, list[int]],
    value: str,
    index: int,
    *,
    normalize_queries: bool,
) -> None:
    normalized = _normalize_lookup_value(value, normalize_queries=normalize_queries)
    if not normalized:
        return
    existing = query_to_indices.setdefault(normalized, [])
    if index not in existing:
        existing.append(index)


def _normalize_lookup_value(value: str, *, normalize_queries: bool) -> str:
    normalized = value.strip()
    if normalize_queries:
        normalized = normalized.casefold()
    return normalized


def _validate_node_index(lookup: NodeLookupTable, node_type: str, index: int) -> None:
    if not 0 <= int(index) < lookup.num_nodes:
        raise ValueError(
            f"{node_type}_index {index} exceeds available node range 0..{lookup.num_nodes - 1}."
        )


def _select_display_name(node_type: str, lookup: NodeLookupTable, index: int) -> str:
    for field_name in DISPLAY_METADATA_FIELDS.get(node_type, ()):
        values = lookup.metadata.get(field_name)
        if values is None:
            continue
        value = values[index].strip()
        if value:
            return value
    return lookup.labels[index]
