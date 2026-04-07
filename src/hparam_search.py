"""Random search utility for Tier-1 HGT improvements."""

from __future__ import annotations

import argparse
import itertools
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.main import main as _unused  # noqa: F401  # ensure package import works when run as module
from src.baseline import BaselineMLPConfig, build_baseline_model, build_pair_feature_tensor
from src.data_loader import AVAILABLE_DATASETS, load_dataset
from src.evaluator import summarize_metrics
from src.graph_builder import GraphBuildConfig, build_train_hetero_graph
from src.model_hgt import HGTModelConfig, build_hgt_model
from src.preprocess import PreprocessConfig, preprocess_dataset
from src.split import SplitConfig, create_drug_disease_splits
from src.trainer import (
    TrainerConfig,
    evaluate_hgt_model,
    train_hgt_model,
)


def run_trial(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    dataset = load_dataset(config["dataset"], data_root=config["data_root"])
    processed_dataset, _ = preprocess_dataset(
        dataset,
        PreprocessConfig(normalize_features=config["normalize_features"]),
    )

    split_cfg = SplitConfig(
        train_ratio=config["train_ratio"],
        val_ratio=config["val_ratio"],
        test_ratio=config["test_ratio"],
        negative_ratio=config["negative_ratio"],
        hard_negative_ratio=config["hard_negative_ratio"],
        random_seed=config["seed"],
    )
    split_bundle, _ = create_drug_disease_splits(processed_dataset, split_cfg)

    graph, _ = build_train_hetero_graph(
        processed_dataset,
        split_bundle,
        GraphBuildConfig(add_reverse_edges=True, add_self_loops=False),
    )

    model = build_hgt_model(
        processed_dataset,
        graph,
        HGTModelConfig(
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            dropout=config["dropout"],
            decoder_hidden_dims=tuple(config["decoder_hidden_dims"]),
            decoder_mode=config["decoder_mode"],
            activation=config["activation"],
            use_layer_norm=True,
        ),
    )

    trial_name = config["trial_name"]
    ckpt_path = output_dir / "checkpoints" / f"{trial_name}.pt"
    trainer_cfg = TrainerConfig(
        device=config["device"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        early_stopping_patience=config["early_stopping_patience"],
        checkpoint_path=ckpt_path,
        loss_name=config["loss_name"],
        pos_weight=config["pos_weight"],
        focal_gamma=config["focal_gamma"],
        artifact_metadata={"trial_name": trial_name},
    )

    train_result = train_hgt_model(
        model,
        graph,
        split_bundle.train,
        split_bundle.val,
        trainer_cfg,
    )
    test_metrics = evaluate_hgt_model(
        model,
        graph,
        split_bundle.test,
        batch_size=config["batch_size"],
        device=config["device"],
    )

    return {
        "trial_name": trial_name,
        "search_config": dict(config),
        "best_epoch": train_result.best_epoch,
        "best_val": train_result.best_val_metrics.to_dict(),
        "test_metrics": summarize_metrics(test_metrics),
        "checkpoint_path": str(ckpt_path),
    }


def sample_trials(args: argparse.Namespace) -> list[dict[str, Any]]:
    rng = random.Random(args.search_seed)

    hidden_dim_space = [64, 96, 128, 192, 256]
    num_layers_space = [2, 3]
    num_heads_space = [2, 4, 8]
    dropout_space = [0.1, 0.2, 0.3, 0.4]
    lr_space = [1e-3, 5e-4, 2e-4]
    wd_space = [1e-4, 5e-5, 1e-5]
    decoder_mode_space = ["product", "concat", "hybrid"]
    hard_negative_space = [0.0, 0.25, 0.5]
    loss_space = ["bce", "focal"]

    candidates = list(
        itertools.product(
            hidden_dim_space,
            num_layers_space,
            num_heads_space,
            dropout_space,
            lr_space,
            wd_space,
            decoder_mode_space,
            hard_negative_space,
            loss_space,
        )
    )
    rng.shuffle(candidates)

    selected = candidates[: args.num_trials]
    trials: list[dict[str, Any]] = []
    for idx, (hidden_dim, num_layers, num_heads, dropout, lr, wd, decoder_mode, hard_neg, loss_name) in enumerate(selected):
        if hidden_dim % num_heads != 0:
            continue
        trial = {
            "trial_name": f"tier1_trial_{idx:03d}",
            "dataset": args.dataset,
            "data_root": args.data_root,
            "normalize_features": args.normalize_features,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "negative_ratio": args.negative_ratio,
            "hard_negative_ratio": hard_neg,
            "seed": args.seed + idx,
            "device": args.device,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": lr,
            "weight_decay": wd,
            "early_stopping_patience": args.early_stopping_patience,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "dropout": dropout,
            "decoder_hidden_dims": [256, 128],
            "decoder_mode": decoder_mode,
            "activation": "gelu",
            "loss_name": loss_name,
            "pos_weight": args.pos_weight,
            "focal_gamma": args.focal_gamma,
        }
        trials.append(trial)
        if len(trials) >= args.num_trials:
            break
    return trials


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tier-1 random search for HGT")
    parser.add_argument("--dataset", choices=AVAILABLE_DATASETS, default="C-dataset")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="results/tier1_search")
    parser.add_argument("--num-trials", type=int, default=12)
    parser.add_argument("--search-seed", type=int, default=123)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--normalize-features", choices=("none", "zscore", "l2"), default="none")
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    trials = sample_trials(args)
    trial_results: list[dict[str, Any]] = []
    for trial in trials:
        result = run_trial(trial, output_dir)
        trial_results.append(result)
        print(
            f"[{result['trial_name']}] val_aupr={result['best_val']['aupr']:.4f} "
            f"test_aupr={result['test_metrics']['aupr']:.4f}"
        )

    trial_results = sorted(
        trial_results,
        key=lambda x: float(x["best_val"].get("aupr", float("nan"))),
        reverse=True,
    )

    summary = {
        "search_args": vars(args),
        "num_trials": len(trial_results),
        "best_trial": trial_results[0] if trial_results else None,
        "trials": trial_results,
    }

    out_path = output_dir / "search_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved summary: {out_path}")


if __name__ == "__main__":
    main()
