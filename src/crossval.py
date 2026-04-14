"""K-fold cross-validation runner for the HGT drug-disease pipeline.

This module is intentionally separate from ``src.main`` because the current
training CLI is built around a single train/val/test split. The cross-validation
runner reuses the same loader, preprocessing, graph builder, HGT model, and
trainer, but wraps them in an outer Stratified K-Fold loop.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

try:
    import torch
except ImportError:  # pragma: no cover - tooling environments may not have torch
    torch = None

try:
    from .data_loader import AVAILABLE_DATASETS, EdgeData, RawDataset, load_dataset
    from .evaluator import BinaryClassificationMetrics, summarize_metrics
    from .graph_builder import GraphBuildConfig, build_train_hetero_graph, summarize_graph_report
    from .model_fusion_hgt import (
        FusionHGTModelConfig,
        build_fusion_hgt_model,
        summarize_fusion_hgt_model,
    )
    from .model_hgt import HGTModelConfig, build_hgt_model, summarize_hgt_model
    from .preprocess import PreprocessConfig, preprocess_dataset
    from .similarity_graph import (
        SimilarityGraphConfig,
        build_similarity_graph_bundle,
        summarize_similarity_graph_bundle,
    )
    from .split import DrugDiseaseSplit, PairDataset, SplitConfig
    from .trainer import TrainerConfig, evaluate_hgt_model, summarize_training_result, train_hgt_model
except ImportError:  # pragma: no cover - allows direct script execution
    from data_loader import AVAILABLE_DATASETS, EdgeData, RawDataset, load_dataset
    from evaluator import BinaryClassificationMetrics, summarize_metrics
    from graph_builder import GraphBuildConfig, build_train_hetero_graph, summarize_graph_report
    from model_fusion_hgt import (
        FusionHGTModelConfig,
        build_fusion_hgt_model,
        summarize_fusion_hgt_model,
    )
    from model_hgt import HGTModelConfig, build_hgt_model, summarize_hgt_model
    from preprocess import PreprocessConfig, preprocess_dataset
    from similarity_graph import (
        SimilarityGraphConfig,
        build_similarity_graph_bundle,
        summarize_similarity_graph_bundle,
    )
    from split import DrugDiseaseSplit, PairDataset, SplitConfig
    from trainer import TrainerConfig, evaluate_hgt_model, summarize_training_result, train_hgt_model


@dataclass(frozen=True, slots=True)
class KFoldConfig:
    """Configuration for the outer cross-validation protocol."""

    folds: int = 10
    val_ratio_within_train: float = 0.1
    negative_ratio: float = 1.0
    hard_negative_ratio: float = 0.0
    random_seed: int = 42
    deduplicate_positive_edges: bool = True
    shuffle: bool = True


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, data_root=args.data_root)
    preprocess_config = PreprocessConfig(
        normalize_features=args.normalize_features,
        validate_auxiliary=not args.skip_auxiliary_validation,
    )
    processed_dataset, preprocess_report = preprocess_dataset(dataset, preprocess_config)

    resolved_hard_negative_ratio = (
        0.5 if args.model == "fusion_hgt" and args.hard_negative_ratio is None else args.hard_negative_ratio
    )
    if resolved_hard_negative_ratio is None:
        resolved_hard_negative_ratio = 0.0

    kfold_config = KFoldConfig(
        folds=args.folds,
        val_ratio_within_train=args.val_ratio_within_train,
        negative_ratio=args.negative_ratio,
        hard_negative_ratio=resolved_hard_negative_ratio,
        random_seed=args.random_seed,
    )
    _validate_kfold_config(kfold_config)
    _set_random_seed(kfold_config.random_seed)

    pair_payload = _build_labeled_pairs(processed_dataset, kfold_config)
    all_pairs = pair_payload["all_pairs"]
    labels = pair_payload["labels"]
    positive_pairs = pair_payload["positive_pairs"]
    negative_pairs = pair_payload["negative_pairs"]
    unknown_pairs = pair_payload["unknown_pairs"]

    graph_config = GraphBuildConfig(
        add_reverse_edges=not args.disable_reverse_edges,
        add_self_loops=args.add_self_loops,
    )
    similarity_graphs = None
    similarity_summary = None
    if args.model == "fusion_hgt":
        similarity_graphs = build_similarity_graph_bundle(
            processed_dataset,
            SimilarityGraphConfig(
                top_k=args.similarity_topk,
                symmetric=not args.disable_similarity_symmetry,
            ),
        )
        similarity_summary = summarize_similarity_graph_bundle(similarity_graphs)

    hgt_model_config = HGTModelConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.hgt_layers,
        num_heads=args.hgt_heads,
        dropout=args.dropout,
        decoder_hidden_dims=tuple(args.hgt_decoder_hidden_dims),
        decoder_mode=args.hgt_decoder_mode,
        activation=args.activation,
        use_layer_norm=not args.disable_layer_norm,
    )
    fusion_model_config = FusionHGTModelConfig(
        hidden_dim=args.hidden_dim,
        num_layers=args.hgt_layers,
        num_heads=args.hgt_heads,
        dropout=args.dropout,
        decoder_hidden_dims=tuple(args.hgt_decoder_hidden_dims),
        decoder_mode=args.hgt_decoder_mode,
        activation=args.activation,
        use_layer_norm=not args.disable_layer_norm,
        similarity_topk=args.similarity_topk,
        sim_layers=args.sim_layers,
        sim_heads=args.sim_heads,
        sim_dropout=args.sim_dropout,
        symmetric_similarity=not args.disable_similarity_symmetry,
    )
    trainer_config_base = TrainerConfig(
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        loss_name=args.loss_name,
        pos_weight=args.pos_weight,
        focal_gamma=args.focal_gamma,
    )

    result_json_path = Path(args.result_json) if args.result_json is not None else None
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir is not None else None
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    splitter = StratifiedKFold(
        n_splits=kfold_config.folds,
        shuffle=kfold_config.shuffle,
        random_state=kfold_config.random_seed,
    )

    fold_reports: list[dict[str, Any]] = []
    fold_test_metrics: list[BinaryClassificationMetrics] = []
    total_start = time.perf_counter()

    for fold_index, (train_val_indices, test_indices) in enumerate(
        splitter.split(all_pairs, labels),
        start=1,
    ):
        fold_start = time.perf_counter()
        print(f"[Fold {fold_index}/{kfold_config.folds}] preparing splits...")

        train_val_pairs = all_pairs[train_val_indices]
        train_val_labels = labels[train_val_indices]
        test_pairs = all_pairs[test_indices]
        test_labels = labels[test_indices]

        train_rows, val_rows = _split_train_val_rows(
            train_val_pairs=train_val_pairs,
            train_val_labels=train_val_labels,
            val_ratio_within_train=kfold_config.val_ratio_within_train,
            random_seed=kfold_config.random_seed + fold_index,
            shuffle=kfold_config.shuffle,
        )

        test_rows = np.column_stack((test_pairs, test_labels)).astype(np.int64, copy=False)
        fold_split = _build_fold_split(
            dataset=processed_dataset,
            train_rows=train_rows,
            val_rows=val_rows,
            test_rows=test_rows,
            sampled_negative_pairs=negative_pairs,
            kfold_config=kfold_config,
        )

        train_graph, graph_report = build_train_hetero_graph(
            processed_dataset,
            fold_split,
            graph_config,
        )
        if args.model == "fusion_hgt":
            model = build_fusion_hgt_model(
                processed_dataset,
                train_graph,
                similarity_graphs,
                fusion_model_config,
            )
        else:
            model = build_hgt_model(processed_dataset, train_graph, hgt_model_config)

        checkpoint_path = None
        if checkpoint_dir is not None:
            checkpoint_path = checkpoint_dir / f"{processed_dataset.dataset_name.lower()}_fold_{fold_index:02d}.pt"

        trainer_config_payload = asdict(trainer_config_base)
        trainer_config_payload["checkpoint_path"] = checkpoint_path
        trainer_config_payload["artifact_metadata"] = {
            "checkpoint_schema_version": 1,
            "model_type": args.model,
            "protocol": "kfold_cross_validation",
            "dataset_name": processed_dataset.dataset_name,
            "fold_index": fold_index,
            "kfold_config": asdict(kfold_config),
            "preprocess_config": asdict(preprocess_config),
            "graph_build_config": asdict(graph_config),
        }
        if similarity_summary is not None:
            trainer_config_payload["artifact_metadata"]["similarity_graph_config"] = similarity_summary["config"]
        trainer_config = TrainerConfig(**trainer_config_payload)

        print(
            f"[Fold {fold_index}/{kfold_config.folds}] "
            f"train={fold_split.train.num_samples}, "
            f"val={fold_split.val.num_samples}, "
            f"test={fold_split.test.num_samples}"
        )
        training_result = train_hgt_model(
            model,
            train_graph,
            fold_split.train,
            fold_split.val,
            trainer_config,
        )
        test_metrics = evaluate_hgt_model(
            model,
            train_graph,
            fold_split.test,
            batch_size=trainer_config.batch_size,
            device=trainer_config.device,
        )

        elapsed_seconds = time.perf_counter() - fold_start
        fold_test_metrics.append(test_metrics)
        fold_report = {
            "fold_index": fold_index,
            "elapsed_seconds": elapsed_seconds,
            "split": {
                "train_samples": fold_split.train.num_samples,
                "val_samples": fold_split.val.num_samples,
                "test_samples": fold_split.test.num_samples,
                "train_positive": fold_split.train.num_positive,
                "val_positive": fold_split.val.num_positive,
                "test_positive": fold_split.test.num_positive,
                "train_negative": fold_split.train.num_negative,
                "val_negative": fold_split.val.num_negative,
                "test_negative": fold_split.test.num_negative,
            },
            "train_graph": summarize_graph_report(graph_report),
            "training": summarize_training_result(training_result),
            "test_metrics": summarize_metrics(test_metrics),
        }
        if similarity_summary is not None:
            fold_report["similarity_graphs"] = similarity_summary
        fold_reports.append(fold_report)
        print(
            f"[Fold {fold_index}/{kfold_config.folds}] "
            f"done in {elapsed_seconds:.1f}s | "
            f"test AUC={test_metrics.auc:.4f} | "
            f"test AUPR={test_metrics.aupr:.4f}"
        )

    total_elapsed_seconds = time.perf_counter() - total_start
    aggregate_test_metrics = _aggregate_metrics(fold_test_metrics)
    payload = {
        "dataset": processed_dataset.dataset_name,
        "model": args.model,
        "protocol": {
            "name": "outer_stratified_kfold_with_inner_validation",
            "notes": (
                "Outer folds follow StratifiedKFold over sampled labeled pairs. "
                "Inside each training fold, a stratified validation split is created "
                "so early stopping and model selection do not look at the fold test set."
            ),
            "kfold_config": asdict(kfold_config),
        },
        "preprocess": {
            "checks_passed": list(preprocess_report.checks_passed),
            "warnings": list(preprocess_report.warnings),
        },
        "pair_sampling": {
            "total_positive_pairs": int(positive_pairs.shape[0]),
            "total_unknown_pairs": int(unknown_pairs.shape[0]),
            "sampled_negative_pairs": int(negative_pairs.shape[0]),
        },
        "model_config": (
            summarize_hgt_model(model)
            if args.model == "hgt"
            else summarize_fusion_hgt_model(model)
        ),
        "trainer_config": {
            "device": trainer_config_base.device,
            "epochs": trainer_config_base.epochs,
            "batch_size": trainer_config_base.batch_size,
            "learning_rate": trainer_config_base.learning_rate,
            "weight_decay": trainer_config_base.weight_decay,
            "early_stopping_patience": trainer_config_base.early_stopping_patience,
            "loss_name": trainer_config_base.loss_name,
            "pos_weight": trainer_config_base.pos_weight,
            "focal_gamma": trainer_config_base.focal_gamma,
        },
        "folds": fold_reports,
        "aggregate_test_metrics": aggregate_test_metrics,
        "total_elapsed_seconds": total_elapsed_seconds,
    }
    if similarity_summary is not None:
        payload["similarity_graphs"] = similarity_summary

    print(json.dumps(payload["aggregate_test_metrics"], indent=2))
    if result_json_path is not None:
        result_json_path.parent.mkdir(parents=True, exist_ok=True)
        with result_json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Saved k-fold result JSON to {result_json_path}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run HGT or fusion-HGT drug-disease link prediction with outer k-fold cross-validation.",
    )
    parser.add_argument("--model", choices=("hgt", "fusion_hgt"), default="hgt")
    parser.add_argument("--dataset", choices=AVAILABLE_DATASETS, default="C-dataset")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=50)
    parser.add_argument("--result-json", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--val-ratio-within-train", type=float, default=0.1)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--hard-negative-ratio", type=float, default=None)
    parser.add_argument(
        "--normalize-features",
        choices=("none", "zscore", "l2"),
        default="none",
    )
    parser.add_argument("--skip-auxiliary-validation", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--activation", choices=("relu", "gelu"), default="gelu")
    parser.add_argument("--loss-name", choices=("bce", "focal"), default="bce")
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hgt-layers", type=int, default=3)
    parser.add_argument("--hgt-heads", type=int, default=4)
    parser.add_argument(
        "--hgt-decoder-hidden-dims",
        type=int,
        nargs="+",
        default=(256, 128),
    )
    parser.add_argument("--hgt-decoder-mode", choices=("product", "concat", "hybrid"), default="product")
    parser.add_argument("--disable-layer-norm", action="store_true")
    parser.add_argument("--disable-reverse-edges", action="store_true")
    parser.add_argument("--add-self-loops", action="store_true")
    parser.add_argument("--similarity-topk", type=int, default=20)
    parser.add_argument("--sim-layers", type=int, default=2)
    parser.add_argument("--sim-heads", type=int, default=4)
    parser.add_argument("--sim-dropout", type=float, default=0.2)
    parser.add_argument("--disable-similarity-symmetry", action="store_true")
    return parser


def _validate_kfold_config(config: KFoldConfig) -> None:
    if config.folds < 2:
        raise ValueError("folds must be at least 2.")
    if config.val_ratio_within_train <= 0 or config.val_ratio_within_train >= 1:
        raise ValueError("val_ratio_within_train must be in (0, 1).")
    if config.negative_ratio <= 0:
        raise ValueError("negative_ratio must be positive.")
    if config.hard_negative_ratio < 0 or config.hard_negative_ratio > 1:
        raise ValueError("hard_negative_ratio must be in [0, 1].")


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _build_labeled_pairs(dataset: RawDataset, config: KFoldConfig) -> dict[str, np.ndarray]:
    positive_pairs = dataset.edges["drug_disease"].as_pairs()
    if config.deduplicate_positive_edges:
        positive_pairs = np.unique(positive_pairs, axis=0)

    unknown_pairs = _enumerate_unknown_pairs(
        num_drugs=dataset.node_counts["drug"],
        num_diseases=dataset.node_counts["disease"],
        positive_pairs=positive_pairs,
    )
    negative_pairs = _sample_negative_pairs(
        unknown_pairs=unknown_pairs,
        positive_pairs=positive_pairs,
        negative_ratio=config.negative_ratio,
        hard_negative_ratio=config.hard_negative_ratio,
        random_seed=config.random_seed,
    )
    all_pairs = np.vstack((positive_pairs, negative_pairs)).astype(np.int64, copy=False)
    labels = np.concatenate(
        (
            np.ones(positive_pairs.shape[0], dtype=np.int64),
            np.zeros(negative_pairs.shape[0], dtype=np.int64),
        )
    )
    return {
        "positive_pairs": positive_pairs,
        "negative_pairs": negative_pairs,
        "unknown_pairs": unknown_pairs,
        "all_pairs": all_pairs,
        "labels": labels,
    }


def _enumerate_unknown_pairs(
    *,
    num_drugs: int,
    num_diseases: int,
    positive_pairs: np.ndarray,
) -> np.ndarray:
    known_mask = np.zeros((num_drugs, num_diseases), dtype=bool)
    known_mask[positive_pairs[:, 0], positive_pairs[:, 1]] = True
    return np.argwhere(~known_mask).astype(np.int64, copy=False)


def _sample_negative_pairs(
    *,
    unknown_pairs: np.ndarray,
    positive_pairs: np.ndarray,
    negative_ratio: float,
    hard_negative_ratio: float,
    random_seed: int,
) -> np.ndarray:
    requested_total = int(round(positive_pairs.shape[0] * negative_ratio))
    if requested_total > unknown_pairs.shape[0]:
        raise ValueError(
            f"Requested {requested_total} negatives but only {unknown_pairs.shape[0]} unknown pairs exist."
        )

    rng = np.random.default_rng(random_seed)
    requested_hard = int(round(requested_total * hard_negative_ratio))
    requested_random = requested_total - requested_hard

    if requested_hard == 0:
        sampled_indices = rng.choice(unknown_pairs.shape[0], size=requested_total, replace=False)
        return unknown_pairs[sampled_indices]

    pos_drugs = set(positive_pairs[:, 0].tolist())
    pos_diseases = set(positive_pairs[:, 1].tolist())
    hard_mask = np.array(
        [
            (int(drug_idx) in pos_drugs) or (int(disease_idx) in pos_diseases)
            for drug_idx, disease_idx in unknown_pairs.tolist()
        ],
        dtype=bool,
    )
    hard_candidates = unknown_pairs[hard_mask]
    random_candidates = unknown_pairs[~hard_mask]

    hard_take = min(requested_hard, hard_candidates.shape[0])
    random_take = requested_random

    if random_take > random_candidates.shape[0]:
        shortfall = random_take - random_candidates.shape[0]
        random_take = random_candidates.shape[0]
        hard_take = min(hard_take + shortfall, hard_candidates.shape[0])

    hard_sample = (
        hard_candidates[rng.choice(hard_candidates.shape[0], size=hard_take, replace=False)]
        if hard_take > 0
        else np.empty((0, 2), dtype=np.int64)
    )
    random_sample = (
        random_candidates[rng.choice(random_candidates.shape[0], size=random_take, replace=False)]
        if random_take > 0
        else np.empty((0, 2), dtype=np.int64)
    )
    merged = np.vstack((hard_sample, random_sample))
    rng.shuffle(merged)
    return merged


def _split_train_val_rows(
    *,
    train_val_pairs: np.ndarray,
    train_val_labels: np.ndarray,
    val_ratio_within_train: float,
    random_seed: int,
    shuffle: bool,
) -> tuple[np.ndarray, np.ndarray]:
    train_val_rows = np.column_stack((train_val_pairs, train_val_labels)).astype(np.int64, copy=False)
    train_rows, val_rows = train_test_split(
        train_val_rows,
        test_size=val_ratio_within_train,
        random_state=random_seed,
        shuffle=shuffle,
        stratify=train_val_labels,
    )
    return (
        train_rows.astype(np.int64, copy=False),
        val_rows.astype(np.int64, copy=False),
    )


def _build_fold_split(
    *,
    dataset: RawDataset,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    test_rows: np.ndarray,
    sampled_negative_pairs: np.ndarray,
    kfold_config: KFoldConfig,
) -> DrugDiseaseSplit:
    train_dataset = PairDataset(
        drug_index=train_rows[:, 0],
        disease_index=train_rows[:, 1],
        labels=train_rows[:, 2],
    )
    val_dataset = PairDataset(
        drug_index=val_rows[:, 0],
        disease_index=val_rows[:, 1],
        labels=val_rows[:, 2],
    )
    test_dataset = PairDataset(
        drug_index=test_rows[:, 0],
        disease_index=test_rows[:, 1],
        labels=test_rows[:, 2],
    )

    train_positive_pairs = train_dataset.positive_pairs()
    train_positive_edges = EdgeData(
        source_index=train_positive_pairs[:, 0],
        target_index=train_positive_pairs[:, 1],
        source_type="drug",
        relation="treats",
        target_type="disease",
    )
    effective_split_config = SplitConfig(
        train_ratio=((kfold_config.folds - 1) / kfold_config.folds) * (1.0 - kfold_config.val_ratio_within_train),
        val_ratio=((kfold_config.folds - 1) / kfold_config.folds) * kfold_config.val_ratio_within_train,
        test_ratio=1.0 / kfold_config.folds,
        negative_ratio=kfold_config.negative_ratio,
        hard_negative_ratio=kfold_config.hard_negative_ratio,
        random_seed=kfold_config.random_seed,
    )
    return DrugDiseaseSplit(
        dataset_name=dataset.dataset_name,
        train=train_dataset,
        val=val_dataset,
        test=test_dataset,
        train_positive_edges=train_positive_edges,
        sampled_negative_pairs=sampled_negative_pairs,
        config=effective_split_config,
    )


def _aggregate_metrics(metrics_list: list[BinaryClassificationMetrics]) -> dict[str, dict[str, float]]:
    metric_names = ("auc", "aupr", "accuracy", "precision", "recall", "f1", "mcc")
    mean: dict[str, float] = {}
    std: dict[str, float] = {}

    for metric_name in metric_names:
        values = np.asarray([getattr(metrics, metric_name) for metrics in metrics_list], dtype=np.float64)
        mean[metric_name] = float(np.nanmean(values))
        std[metric_name] = float(np.nanstd(values))

    return {
        "mean": mean,
        "std": std,
    }


if __name__ == "__main__":
    main()
