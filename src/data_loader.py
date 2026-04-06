"""Load AMDGT benchmark datasets into a canonical Python structure.

This module focuses on raw IO only:
    - read node features and metadata
    - read canonical edge lists
    - optionally read similarity matrices and auxiliary graph exports

Feature projection, graph construction, and tensor conversion happen later
in the pipeline.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


AVAILABLE_DATASETS = ("B-dataset", "C-dataset", "F-dataset")


@dataclass(slots=True)
class NodeData:
    """Raw node-level inputs for one node type."""

    features: np.ndarray
    labels: list[str]
    metadata: dict[str, list[str]] = field(default_factory=dict)

    @property
    def num_nodes(self) -> int:
        return int(self.features.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])


@dataclass(slots=True)
class EdgeData:
    """Canonical edge list for one relation."""

    source_index: np.ndarray
    target_index: np.ndarray
    source_type: str
    relation: str
    target_type: str

    @property
    def num_edges(self) -> int:
        return int(self.source_index.shape[0])

    def as_pairs(self) -> np.ndarray:
        return np.column_stack((self.source_index, self.target_index))


@dataclass(slots=True)
class RawDataset:
    """Raw AMDGT dataset after file loading, before preprocessing."""

    dataset_name: str
    dataset_dir: Path
    drugs: NodeData
    diseases: NodeData
    proteins: NodeData
    edges: dict[str, EdgeData]
    similarity: dict[str, np.ndarray] = field(default_factory=dict)
    auxiliary: dict[str, Any] = field(default_factory=dict)

    @property
    def node_counts(self) -> dict[str, int]:
        return {
            "drug": self.drugs.num_nodes,
            "disease": self.diseases.num_nodes,
            "protein": self.proteins.num_nodes,
        }

    @property
    def feature_dims(self) -> dict[str, int]:
        return {
            "drug": self.drugs.feature_dim,
            "disease": self.diseases.feature_dim,
            "protein": self.proteins.feature_dim,
        }


def load_dataset(
    dataset: str | Path,
    data_root: str | Path = "data",
    include_similarity: bool = True,
    include_auxiliary: bool = True,
) -> RawDataset:
    """Load one AMDGT dataset directory.

    Args:
        dataset: Dataset name like ``"C-dataset"`` or a concrete directory path.
        data_root: Root directory containing the AMDGT datasets.
        include_similarity: Whether to load similarity matrices used in Tier 2.
        include_auxiliary: Whether to load duplicate graph exports like
            ``Allnode.csv``, ``Alledge.csv``, and ``adj.csv`` for validation.
    """

    dataset_dir = _resolve_dataset_dir(dataset, data_root)

    drug_labels, drug_features = _read_feature_csv(dataset_dir / "Drug_mol2vec.csv")
    disease_labels, disease_features = _read_feature_csv(
        dataset_dir / "DiseaseFeature.csv"
    )
    protein_labels, protein_features = _read_feature_csv(dataset_dir / "Protein_ESM.csv")

    drugs = NodeData(
        features=drug_features,
        labels=drug_labels,
        metadata=_load_drug_information(
            dataset_dir / "DrugInformation.csv",
            expected_indices=drug_labels,
        ),
    )
    diseases = NodeData(features=disease_features, labels=disease_labels)
    proteins = NodeData(
        features=protein_features,
        labels=protein_labels,
        metadata=_load_protein_information(dataset_dir / "ProteinInformation.csv"),
    )

    edges = {
        "drug_disease": _load_edge_list(
            dataset_dir / "DrugDiseaseAssociationNumber.csv",
            source_column="drug",
            target_column="disease",
            source_type="drug",
            relation="treats",
            target_type="disease",
        ),
        "drug_protein": _load_edge_list(
            dataset_dir / "DrugProteinAssociationNumber.csv",
            source_column="drug",
            target_column="protein",
            source_type="drug",
            relation="targets",
            target_type="protein",
        ),
        "protein_disease": _load_edge_list(
            dataset_dir / "ProteinDiseaseAssociationNumber.csv",
            source_column="protein",
            target_column="disease",
            source_type="protein",
            relation="associates",
            target_type="disease",
        ),
    }

    similarity: dict[str, np.ndarray] = {}
    if include_similarity:
        for name in ("DrugFingerprint", "DrugGIP", "DiseasePS", "DiseaseGIP"):
            _, _, matrix = _read_labeled_matrix(dataset_dir / f"{name}.csv")
            similarity[name] = matrix

    auxiliary: dict[str, Any] = {}
    if include_auxiliary:
        if (dataset_dir / "adj.csv").exists():
            row_labels, col_labels, matrix = _read_labeled_matrix(dataset_dir / "adj.csv")
            auxiliary["drug_disease_adjacency"] = {
                "row_labels": row_labels,
                "column_labels": col_labels,
                "matrix": matrix,
            }
        if (dataset_dir / "Allnode.csv").exists():
            auxiliary["all_nodes"] = _load_index_label_table(dataset_dir / "Allnode.csv")
        if (dataset_dir / "Alledge.csv").exists():
            auxiliary["all_edges"] = _load_pair_csv(
                dataset_dir / "Alledge.csv", has_header=False
            )

    return RawDataset(
        dataset_name=dataset_dir.name,
        dataset_dir=dataset_dir,
        drugs=drugs,
        diseases=diseases,
        proteins=proteins,
        edges=edges,
        similarity=similarity,
        auxiliary=auxiliary,
    )


def _resolve_dataset_dir(dataset: str | Path, data_root: str | Path) -> Path:
    candidate = Path(dataset)
    if candidate.exists():
        return candidate.resolve()

    dataset_dir = Path(data_root) / str(dataset)
    if not dataset_dir.exists():
        available = ", ".join(AVAILABLE_DATASETS)
        raise FileNotFoundError(
            f"Dataset '{dataset}' was not found under '{data_root}'. "
            f"Expected one of: {available}"
        )
    return dataset_dir.resolve()


def _read_csv_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def _read_feature_csv(path: Path) -> tuple[list[str], np.ndarray]:
    rows = _read_csv_rows(path)
    if not rows:
        raise ValueError(f"Feature file '{path}' is empty.")

    labels = [row[0].strip() for row in rows]
    values = np.asarray([[float(value) for value in row[1:]] for row in rows], dtype=np.float32)
    return labels, values


def _load_drug_information(
    path: Path,
    expected_indices: list[str] | None = None,
) -> dict[str, list[str]]:
    rows = _read_csv_rows(path)
    if len(rows) < 2:
        raise ValueError(f"Drug metadata file '{path}' is empty.")

    header = {name.strip().lower(): idx for idx, name in enumerate(rows[0])}
    required = {"id", "name", "smiles"}
    missing = required.difference(header)
    if missing:
        raise ValueError(f"Drug metadata file '{path}' is missing columns: {sorted(missing)}")

    data_rows = rows[1:]
    if expected_indices is None:
        return {
            "drugbank_id": [row[header["id"]].strip() for row in data_rows],
            "name": [row[header["name"]].strip() for row in data_rows],
            "smiles": [row[header["smiles"]].strip() for row in data_rows],
        }

    local_index_column = _find_local_index_column(rows[0], data_rows)
    if local_index_column is None:
        if len(data_rows) != len(expected_indices):
            raise ValueError(
                f"Drug metadata file '{path}' has {len(data_rows)} rows, expected {len(expected_indices)}."
            )
        return {
            "drugbank_id": [row[header["id"]].strip() for row in data_rows],
            "name": [row[header["name"]].strip() for row in data_rows],
            "smiles": [row[header["smiles"]].strip() for row in data_rows],
        }

    row_by_index = {int(row[local_index_column]): row for row in data_rows}
    ordered_indices = [int(index) for index in expected_indices]
    missing_indices = [idx for idx in ordered_indices if idx not in row_by_index]
    if missing_indices:
        raise ValueError(
            f"Drug metadata file '{path}' is missing local indices: {missing_indices[:5]}"
        )

    return {
        "drugbank_id": [row_by_index[idx][header["id"]].strip() for idx in ordered_indices],
        "name": [row_by_index[idx][header["name"]].strip() for idx in ordered_indices],
        "smiles": [row_by_index[idx][header["smiles"]].strip() for idx in ordered_indices],
    }


def _load_protein_information(path: Path) -> dict[str, list[str]]:
    rows = _read_csv_rows(path)
    if len(rows) < 2:
        raise ValueError(f"Protein metadata file '{path}' is empty.")

    header = {name.strip().lower(): idx for idx, name in enumerate(rows[0])}
    required = {"id", "sequence"}
    missing = required.difference(header)
    if missing:
        raise ValueError(
            f"Protein metadata file '{path}' is missing columns: {sorted(missing)}"
        )

    return {
        "uniprot_id": [row[header["id"]].strip() for row in rows[1:]],
        "sequence": [row[header["sequence"]].strip() for row in rows[1:]],
    }


def _find_local_index_column(header_row: list[str], data_rows: list[list[str]]) -> int | None:
    if not data_rows:
        return None

    for column_idx, column_name in enumerate(header_row):
        normalized_name = column_name.strip().lower()
        if normalized_name in {"id", "name", "smiles"}:
            continue
        sample = data_rows[0][column_idx].strip()
        if sample.isdigit():
            return column_idx
    return None


def _load_edge_list(
    path: Path,
    source_column: str,
    target_column: str,
    source_type: str,
    relation: str,
    target_type: str,
) -> EdgeData:
    rows = _read_csv_rows(path)
    if len(rows) < 2:
        raise ValueError(f"Edge file '{path}' is empty.")

    header = {name.strip().lower(): idx for idx, name in enumerate(rows[0])}
    missing = {source_column, target_column}.difference(header)
    if missing:
        raise ValueError(f"Edge file '{path}' is missing columns: {sorted(missing)}")

    source_index = np.asarray(
        [int(row[header[source_column]]) for row in rows[1:]],
        dtype=np.int64,
    )
    target_index = np.asarray(
        [int(row[header[target_column]]) for row in rows[1:]],
        dtype=np.int64,
    )

    return EdgeData(
        source_index=source_index,
        target_index=target_index,
        source_type=source_type,
        relation=relation,
        target_type=target_type,
    )


def _read_labeled_matrix(path: Path) -> tuple[list[str], list[str], np.ndarray]:
    rows = _read_csv_rows(path)
    if len(rows) < 2:
        raise ValueError(f"Matrix file '{path}' is empty.")

    column_labels = [value.strip() for value in rows[0][1:]]
    row_labels = [row[0].strip() for row in rows[1:]]
    matrix = np.asarray(
        [[float(value) for value in row[1:]] for row in rows[1:]],
        dtype=np.float32,
    )
    return row_labels, column_labels, matrix


def _load_index_label_table(path: Path) -> dict[str, Any]:
    rows = _read_csv_rows(path)
    if not rows:
        raise ValueError(f"Index-label file '{path}' is empty.")

    if all(len(row) == 1 for row in rows):
        indices = np.arange(len(rows), dtype=np.int64)
        labels = [row[0].strip() for row in rows]
        return {"indices": indices, "labels": labels}

    if not all(len(row) >= 2 for row in rows):
        raise ValueError(f"Index-label file '{path}' has inconsistent column counts.")

    try:
        int(rows[0][0])
    except ValueError:
        rows = rows[1:]
        if not rows:
            raise ValueError(f"Index-label file '{path}' only contains a header row.")

    indices = np.asarray([int(row[0]) for row in rows], dtype=np.int64)
    labels = [row[1].strip() for row in rows]
    return {"indices": indices, "labels": labels}


def _load_pair_csv(path: Path, has_header: bool) -> np.ndarray:
    rows = _read_csv_rows(path)
    if has_header:
        rows = rows[1:]
    if not rows:
        raise ValueError(f"Pair file '{path}' is empty.")

    return np.asarray([[int(row[0]), int(row[1])] for row in rows], dtype=np.int64)
