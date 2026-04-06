"""Train/validation/test splitting for drug-disease link prediction."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

try:
    import torch
except ImportError:  # pragma: no cover - torch may be unavailable in some tooling
    torch = None

try:
    from .data_loader import EdgeData, RawDataset
except ImportError:  # pragma: no cover - allows direct script execution
    from data_loader import EdgeData, RawDataset


@dataclass(frozen=True, slots=True)
class SplitConfig:
    """Controls negative sampling and stratified train/val/test splitting."""

    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    negative_ratio: float = 1.0
    random_seed: int = 42
    deduplicate_positive_edges: bool = True
    stratify: bool = True
    shuffle: bool = True


@dataclass(slots=True)
class PairDataset:
    """A labeled set of drug-disease candidate pairs."""

    drug_index: np.ndarray
    disease_index: np.ndarray
    labels: np.ndarray

    @property
    def num_samples(self) -> int:
        return int(self.labels.shape[0])

    @property
    def num_positive(self) -> int:
        return int(np.count_nonzero(self.labels == 1))

    @property
    def num_negative(self) -> int:
        return int(np.count_nonzero(self.labels == 0))

    def positive_pairs(self) -> np.ndarray:
        mask = self.labels == 1
        return np.column_stack((self.drug_index[mask], self.disease_index[mask]))

    def negative_pairs(self) -> np.ndarray:
        mask = self.labels == 0
        return np.column_stack((self.drug_index[mask], self.disease_index[mask]))

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "drug_index": self.drug_index,
            "disease_index": self.disease_index,
            "labels": self.labels,
        }


@dataclass(slots=True)
class DrugDiseaseSplit:
    """Split bundle used by later training and evaluation stages."""

    dataset_name: str
    train: PairDataset
    val: PairDataset
    test: PairDataset
    train_positive_edges: EdgeData
    sampled_negative_pairs: np.ndarray
    config: SplitConfig


@dataclass(slots=True)
class SplitReport:
    """Compact summary of the generated split."""

    dataset_name: str
    num_drugs: int
    num_diseases: int
    total_positive_pairs: int
    total_unknown_pairs: int
    sampled_negative_pairs: int
    train_samples: int
    val_samples: int
    test_samples: int
    train_positive: int
    val_positive: int
    test_positive: int
    train_negative: int
    val_negative: int
    test_negative: int
    checks_passed: list[str] = field(default_factory=list)


def create_drug_disease_splits(
    dataset: RawDataset,
    config: SplitConfig = SplitConfig(),
) -> tuple[DrugDiseaseSplit, SplitReport]:
    """Sample negatives globally, then create stratified train/val/test splits."""

    _validate_split_config(config)
    _set_random_seed(config.random_seed)

    positive_edge = dataset.edges["drug_disease"]
    positive_pairs = positive_edge.as_pairs()
    if config.deduplicate_positive_edges:
        positive_pairs = np.unique(positive_pairs, axis=0)

    num_drugs = dataset.node_counts["drug"]
    num_diseases = dataset.node_counts["disease"]

    unknown_pairs = _enumerate_unknown_pairs(
        num_drugs=num_drugs,
        num_diseases=num_diseases,
        positive_pairs=positive_pairs,
    )
    negative_pairs = _sample_negative_pairs(
        unknown_pairs=unknown_pairs,
        num_positive=positive_pairs.shape[0],
        negative_ratio=config.negative_ratio,
        random_seed=config.random_seed,
    )

    all_pairs = np.vstack((positive_pairs, negative_pairs))
    labels = np.concatenate(
        (
            np.ones(positive_pairs.shape[0], dtype=np.int64),
            np.zeros(negative_pairs.shape[0], dtype=np.int64),
        )
    )

    train_pairs, val_pairs, test_pairs = _stratified_train_val_test_split(
        pairs=all_pairs,
        labels=labels,
        config=config,
    )

    train_dataset = PairDataset(
        drug_index=train_pairs[:, 0],
        disease_index=train_pairs[:, 1],
        labels=train_pairs[:, 2],
    )
    val_dataset = PairDataset(
        drug_index=val_pairs[:, 0],
        disease_index=val_pairs[:, 1],
        labels=val_pairs[:, 2],
    )
    test_dataset = PairDataset(
        drug_index=test_pairs[:, 0],
        disease_index=test_pairs[:, 1],
        labels=test_pairs[:, 2],
    )

    train_positive_pairs = train_dataset.positive_pairs()
    train_positive_edges = EdgeData(
        source_index=train_positive_pairs[:, 0],
        target_index=train_positive_pairs[:, 1],
        source_type="drug",
        relation="treats",
        target_type="disease",
    )

    split = DrugDiseaseSplit(
        dataset_name=dataset.dataset_name,
        train=train_dataset,
        val=val_dataset,
        test=test_dataset,
        train_positive_edges=train_positive_edges,
        sampled_negative_pairs=negative_pairs,
        config=config,
    )
    report = _build_split_report(dataset, split, positive_pairs, unknown_pairs)
    return split, report


def build_train_graph_edges(dataset: RawDataset, split: DrugDiseaseSplit) -> dict[str, EdgeData]:
    """Return graph edges for message passing during training.

    Drug-protein and protein-disease relations stay fixed.
    Drug-disease edges are reduced to training positives only to avoid leakage.
    """

    return {
        "drug_disease": split.train_positive_edges,
        "drug_protein": dataset.edges["drug_protein"],
        "protein_disease": dataset.edges["protein_disease"],
    }


def _validate_split_config(config: SplitConfig) -> None:
    ratios = (config.train_ratio, config.val_ratio, config.test_ratio)
    if any(ratio <= 0 for ratio in ratios):
        raise ValueError("train/val/test ratios must all be positive.")

    ratio_sum = sum(ratios)
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(
            f"train/val/test ratios must sum to 1.0, received {ratio_sum:.6f}."
        )
    if config.negative_ratio <= 0:
        raise ValueError("negative_ratio must be positive.")


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _enumerate_unknown_pairs(
    num_drugs: int,
    num_diseases: int,
    positive_pairs: np.ndarray,
) -> np.ndarray:
    known_mask = np.zeros((num_drugs, num_diseases), dtype=bool)
    known_mask[positive_pairs[:, 0], positive_pairs[:, 1]] = True
    unknown_pairs = np.argwhere(~known_mask)
    return unknown_pairs.astype(np.int64, copy=False)


def _sample_negative_pairs(
    unknown_pairs: np.ndarray,
    num_positive: int,
    negative_ratio: float,
    random_seed: int,
) -> np.ndarray:
    requested = int(round(num_positive * negative_ratio))
    if requested > unknown_pairs.shape[0]:
        raise ValueError(
            f"Requested {requested} negatives but only {unknown_pairs.shape[0]} unknown pairs exist."
        )

    rng = np.random.default_rng(random_seed)
    sampled_indices = rng.choice(unknown_pairs.shape[0], size=requested, replace=False)
    return unknown_pairs[sampled_indices]


def _stratified_train_val_test_split(
    pairs: np.ndarray,
    labels: np.ndarray,
    config: SplitConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pair_table = np.column_stack((pairs, labels))
    stratify_labels = labels if config.stratify else None

    train_rows, temp_rows = train_test_split(
        pair_table,
        test_size=config.val_ratio + config.test_ratio,
        random_state=config.random_seed,
        shuffle=config.shuffle,
        stratify=stratify_labels,
    )

    temp_labels = temp_rows[:, 2] if config.stratify else None
    val_fraction_within_temp = config.val_ratio / (config.val_ratio + config.test_ratio)

    val_rows, test_rows = train_test_split(
        temp_rows,
        test_size=1.0 - val_fraction_within_temp,
        random_state=config.random_seed,
        shuffle=config.shuffle,
        stratify=temp_labels,
    )

    return (
        train_rows.astype(np.int64, copy=False),
        val_rows.astype(np.int64, copy=False),
        test_rows.astype(np.int64, copy=False),
    )


def _build_split_report(
    dataset: RawDataset,
    split: DrugDiseaseSplit,
    positive_pairs: np.ndarray,
    unknown_pairs: np.ndarray,
) -> SplitReport:
    checks: list[str] = []

    _validate_split_labels(split.train, "train")
    _validate_split_labels(split.val, "val")
    _validate_split_labels(split.test, "test")
    checks.append("each split contains only binary labels {0, 1}")

    _validate_no_overlap(split.train, split.val, split.test)
    checks.append("train/val/test pair sets do not overlap")

    _validate_train_positive_edges(split)
    checks.append("training graph uses only training positive drug-disease edges")

    _validate_negative_pairs(split, positive_pairs)
    checks.append("sampled negatives do not intersect known positive pairs")

    report = SplitReport(
        dataset_name=dataset.dataset_name,
        num_drugs=dataset.node_counts["drug"],
        num_diseases=dataset.node_counts["disease"],
        total_positive_pairs=int(positive_pairs.shape[0]),
        total_unknown_pairs=int(unknown_pairs.shape[0]),
        sampled_negative_pairs=int(split.sampled_negative_pairs.shape[0]),
        train_samples=split.train.num_samples,
        val_samples=split.val.num_samples,
        test_samples=split.test.num_samples,
        train_positive=split.train.num_positive,
        val_positive=split.val.num_positive,
        test_positive=split.test.num_positive,
        train_negative=split.train.num_negative,
        val_negative=split.val.num_negative,
        test_negative=split.test.num_negative,
        checks_passed=checks,
    )
    return report


def _validate_split_labels(split_data: PairDataset, split_name: str) -> None:
    valid_values = np.unique(split_data.labels)
    if not np.array_equal(valid_values, np.array([0, 1], dtype=np.int64)) and not np.array_equal(
        valid_values, np.array([0], dtype=np.int64)
    ) and not np.array_equal(valid_values, np.array([1], dtype=np.int64)):
        raise ValueError(f"{split_name} split contains non-binary labels: {valid_values.tolist()}")


def _validate_no_overlap(train: PairDataset, val: PairDataset, test: PairDataset) -> None:
    train_pairs = set(map(tuple, np.column_stack((train.drug_index, train.disease_index)).tolist()))
    val_pairs = set(map(tuple, np.column_stack((val.drug_index, val.disease_index)).tolist()))
    test_pairs = set(map(tuple, np.column_stack((test.drug_index, test.disease_index)).tolist()))

    if train_pairs & val_pairs:
        raise ValueError("Train and validation splits overlap.")
    if train_pairs & test_pairs:
        raise ValueError("Train and test splits overlap.")
    if val_pairs & test_pairs:
        raise ValueError("Validation and test splits overlap.")


def _validate_train_positive_edges(split: DrugDiseaseSplit) -> None:
    train_positive_pairs = set(map(tuple, split.train_positive_edges.as_pairs().tolist()))
    train_label_positive_pairs = set(map(tuple, split.train.positive_pairs().tolist()))
    if train_positive_pairs != train_label_positive_pairs:
        raise ValueError("Training graph positives do not match training positive labels.")


def _validate_negative_pairs(split: DrugDiseaseSplit, positive_pairs: np.ndarray) -> None:
    positive_set = set(map(tuple, positive_pairs.tolist()))
    negative_set = set(map(tuple, split.sampled_negative_pairs.tolist()))
    if positive_set & negative_set:
        raise ValueError("Sampled negatives overlap with known positive drug-disease pairs.")


def summarize_split_report(report: SplitReport) -> dict[str, Any]:
    """Return a flat summary that is easy to log or serialize."""

    return {
        "dataset_name": report.dataset_name,
        "num_drugs": report.num_drugs,
        "num_diseases": report.num_diseases,
        "total_positive_pairs": report.total_positive_pairs,
        "total_unknown_pairs": report.total_unknown_pairs,
        "sampled_negative_pairs": report.sampled_negative_pairs,
        "train_samples": report.train_samples,
        "val_samples": report.val_samples,
        "test_samples": report.test_samples,
        "train_positive": report.train_positive,
        "val_positive": report.val_positive,
        "test_positive": report.test_positive,
        "train_negative": report.train_negative,
        "val_negative": report.val_negative,
        "test_negative": report.test_negative,
        "checks_passed": list(report.checks_passed),
    }
