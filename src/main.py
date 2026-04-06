"""CLI entry point for training baseline and HGT models."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

try:
    from .baseline import BaselineMLPConfig, build_baseline_model, build_pair_feature_tensor
    from .data_loader import AVAILABLE_DATASETS, load_dataset
    from .evaluator import summarize_metrics
    from .graph_builder import GraphBuildConfig, build_train_hetero_graph, summarize_graph_report
    from .model_hgt import HGTModelConfig, build_hgt_model, summarize_hgt_model
    from .preprocess import PreprocessConfig, preprocess_dataset
    from .split import SplitConfig, create_drug_disease_splits, summarize_split_report
    from .trainer import (
        TrainerConfig,
        evaluate_baseline_model,
        evaluate_hgt_model,
        summarize_training_result,
        train_baseline_model,
        train_hgt_model,
    )
except ImportError:  # pragma: no cover - allows `python src/main.py`
    from baseline import BaselineMLPConfig, build_baseline_model, build_pair_feature_tensor
    from data_loader import AVAILABLE_DATASETS, load_dataset
    from evaluator import summarize_metrics
    from graph_builder import GraphBuildConfig, build_train_hetero_graph, summarize_graph_report
    from model_hgt import HGTModelConfig, build_hgt_model, summarize_hgt_model
    from preprocess import PreprocessConfig, preprocess_dataset
    from split import SplitConfig, create_drug_disease_splits, summarize_split_report
    from trainer import (
        TrainerConfig,
        evaluate_baseline_model,
        evaluate_hgt_model,
        summarize_training_result,
        train_baseline_model,
        train_hgt_model,
    )


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    dataset = load_dataset(args.dataset, data_root=args.data_root)
    preprocess_config = PreprocessConfig(
        normalize_features=args.normalize_features,
        validate_auxiliary=not args.skip_auxiliary_validation,
    )
    processed_dataset, preprocess_report = preprocess_dataset(dataset, preprocess_config)

    split_config = SplitConfig(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        negative_ratio=args.negative_ratio,
        random_seed=args.random_seed,
    )
    split_bundle, split_report = create_drug_disease_splits(processed_dataset, split_config)

    trainer_config = TrainerConfig(
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_path=args.checkpoint_path,
        artifact_metadata={
            "checkpoint_schema_version": 1,
            "model_type": args.model,
            "dataset_name": processed_dataset.dataset_name,
            "preprocess_config": asdict(preprocess_config),
            "split_config": asdict(split_config),
        },
    )

    run_summary: dict[str, Any] = {
        "dataset": processed_dataset.dataset_name,
        "model": args.model,
        "preprocess": {
            "checks_passed": list(preprocess_report.checks_passed),
            "warnings": list(preprocess_report.warnings),
        },
        "split": summarize_split_report(split_report),
    }

    if args.model == "baseline":
        model, training_result, test_metrics = _run_baseline_training(
            processed_dataset,
            split_bundle,
            trainer_config,
            args,
        )
        run_summary["model_config"] = asdict(model.config)
    else:
        model, training_result, test_metrics, graph_report = _run_hgt_training(
            processed_dataset,
            split_bundle,
            trainer_config,
            args,
        )
        run_summary["model_config"] = summarize_hgt_model(model)
        run_summary["train_graph"] = summarize_graph_report(graph_report)

    run_summary["training"] = summarize_training_result(training_result)
    run_summary["test_metrics"] = summarize_metrics(test_metrics)

    _print_run_summary(run_summary)
    if args.result_json is not None:
        _write_json(Path(args.result_json), run_summary)


def _run_baseline_training(
    dataset,
    split_bundle,
    trainer_config: TrainerConfig,
    args,
):
    model = build_baseline_model(
        dataset,
        BaselineMLPConfig(
            hidden_dims=tuple(args.baseline_hidden_dims),
            dropout=args.dropout,
            activation=args.activation,
        ),
    )

    train_features, train_labels = build_pair_feature_tensor(dataset, split_bundle.train)
    val_features, val_labels = build_pair_feature_tensor(dataset, split_bundle.val)
    test_features, test_labels = build_pair_feature_tensor(dataset, split_bundle.test)

    baseline_trainer_config = replace(
        trainer_config,
        artifact_metadata={
            **trainer_config.artifact_metadata,
            "graph_mode": "none",
        },
    )

    training_result = train_baseline_model(
        model,
        train_features,
        train_labels,
        val_features,
        val_labels,
        baseline_trainer_config,
    )
    test_metrics = evaluate_baseline_model(
        model,
        test_features,
        test_labels,
        batch_size=trainer_config.batch_size,
        device=trainer_config.device,
    )
    return model, training_result, test_metrics


def _run_hgt_training(
    dataset,
    split_bundle,
    trainer_config: TrainerConfig,
    args,
):
    graph_config = GraphBuildConfig(
        add_reverse_edges=not args.disable_reverse_edges,
        add_self_loops=args.add_self_loops,
    )
    train_graph, graph_report = build_train_hetero_graph(
        dataset,
        split_bundle,
        graph_config,
    )

    model = build_hgt_model(
        dataset,
        train_graph,
        HGTModelConfig(
            hidden_dim=args.hidden_dim,
            num_layers=args.hgt_layers,
            num_heads=args.hgt_heads,
            dropout=args.dropout,
            decoder_hidden_dims=tuple(args.hgt_decoder_hidden_dims),
            activation=args.activation,
            use_layer_norm=not args.disable_layer_norm,
        ),
    )

    hgt_trainer_config = replace(
        trainer_config,
        artifact_metadata={
            **trainer_config.artifact_metadata,
            "graph_mode": "train",
            "graph_build_config": asdict(graph_config),
        },
    )

    training_result = train_hgt_model(
        model,
        train_graph,
        split_bundle.train,
        split_bundle.val,
        hgt_trainer_config,
    )
    test_metrics = evaluate_hgt_model(
        model,
        train_graph,
        split_bundle.test,
        batch_size=trainer_config.batch_size,
        device=trainer_config.device,
    )
    return model, training_result, test_metrics, graph_report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train baseline or HGT models for drug-disease link prediction.",
    )
    parser.add_argument("--model", choices=("baseline", "hgt"), default="baseline")
    parser.add_argument("--dataset", choices=AVAILABLE_DATASETS, default="C-dataset")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--result-json", default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument(
        "--normalize-features",
        choices=("none", "zscore", "l2"),
        default="none",
    )
    parser.add_argument("--skip-auxiliary-validation", action="store_true")

    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--activation", choices=("relu", "gelu"), default="gelu")
    parser.add_argument(
        "--baseline-hidden-dims",
        type=int,
        nargs="+",
        default=(256, 128),
    )

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hgt-layers", type=int, default=2)
    parser.add_argument("--hgt-heads", type=int, default=4)
    parser.add_argument(
        "--hgt-decoder-hidden-dims",
        type=int,
        nargs="+",
        default=(256, 128),
    )
    parser.add_argument("--disable-layer-norm", action="store_true")
    parser.add_argument("--disable-reverse-edges", action="store_true")
    parser.add_argument("--add-self-loops", action="store_true")
    return parser


def _print_run_summary(run_summary: dict[str, Any]) -> None:
    print(json.dumps(run_summary, indent=2))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
