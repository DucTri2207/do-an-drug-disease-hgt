"""Inference and lightweight export helpers for deployment-oriented top-k ranking."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

try:
    from .baseline import (
        BaselineMLPConfig,
        DrugDiseaseMLPBaseline,
        build_all_candidate_pair_features,
    )
    from .data_loader import AVAILABLE_DATASETS, RawDataset, load_dataset
    from .feature_builder import (
        FeatureBuilderConfig,
        FeatureBundle,
        export_node_table,
        build_feature_bundle,
        get_node_record,
        resolve_node_query,
    )
    from .graph_builder import (
        GraphBuildConfig,
        GraphBuildReport,
        build_full_hetero_graph,
        build_train_hetero_graph,
        summarize_graph_report,
    )
    from .model_hgt import DrugDiseaseHGT, HGTModelConfig
    from .preprocess import NormalizeMode, PreprocessConfig, preprocess_dataset
    from .split import SplitConfig, create_drug_disease_splits
    from .trainer import resolve_device
except ImportError:  # pragma: no cover - allows direct script execution
    from baseline import (
        BaselineMLPConfig,
        DrugDiseaseMLPBaseline,
        build_all_candidate_pair_features,
    )
    from data_loader import AVAILABLE_DATASETS, RawDataset, load_dataset
    from feature_builder import (
        FeatureBuilderConfig,
        FeatureBundle,
        export_node_table,
        build_feature_bundle,
        get_node_record,
        resolve_node_query,
    )
    from graph_builder import (
        GraphBuildConfig,
        GraphBuildReport,
        build_full_hetero_graph,
        build_train_hetero_graph,
        summarize_graph_report,
    )
    from model_hgt import DrugDiseaseHGT, HGTModelConfig
    from preprocess import NormalizeMode, PreprocessConfig, preprocess_dataset
    from split import SplitConfig, create_drug_disease_splits
    from trainer import resolve_device


ModelType = Literal["baseline", "hgt"]
GraphMode = Literal["none", "train", "full"]


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """Controls how inference assets are rebuilt and how ranking is filtered."""

    device: str | None = None
    top_k: int = 10
    graph_mode: GraphMode | None = None
    normalize_features: NormalizeMode | None = None
    validate_auxiliary: bool = False
    add_reverse_edges: bool | None = None
    add_self_loops: bool | None = None
    exclude_known_associations: bool = True


@dataclass(slots=True)
class PredictionRecord:
    """One ranked disease candidate for a query drug."""

    rank: int
    disease_index: int
    disease_record: dict[str, Any]
    logit: float
    probability: float
    known_association: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "disease_index": self.disease_index,
            "disease": dict(self.disease_record),
            "logit": self.logit,
            "probability": self.probability,
            "known_association": self.known_association,
        }


@dataclass(slots=True)
class TopKPredictionResult:
    """Serializable top-k prediction payload for logging or web use."""

    dataset_name: str
    model_type: ModelType
    checkpoint_path: str
    graph_mode: GraphMode
    top_k: int
    num_candidates_considered: int
    num_known_filtered: int
    drug_record: dict[str, Any]
    predictions: list[PredictionRecord]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "model_type": self.model_type,
            "checkpoint_path": self.checkpoint_path,
            "graph_mode": self.graph_mode,
            "top_k": self.top_k,
            "num_candidates_considered": self.num_candidates_considered,
            "num_known_filtered": self.num_known_filtered,
            "drug": dict(self.drug_record),
            "predictions": [prediction.to_dict() for prediction in self.predictions],
        }


@dataclass(slots=True)
class InferenceSession:
    """Runtime bundle used by API or CLI inference."""

    checkpoint_path: str
    checkpoint_payload: dict[str, Any]
    model_type: ModelType
    dataset: RawDataset
    feature_bundle: FeatureBundle
    model: torch.nn.Module
    device: str
    graph_mode: GraphMode
    graph: Any | None = None
    graph_report: GraphBuildReport | None = None
    known_drug_disease_mask: np.ndarray | None = None


def load_inference_session(
    checkpoint_path: str | Path,
    *,
    dataset: str | None = None,
    data_root: str | Path = "data",
    model_type: ModelType | None = None,
    config: InferenceConfig = InferenceConfig(),
) -> InferenceSession:
    """Load a trained model plus the minimum data needed for deployment inference."""

    payload = torch.load(Path(checkpoint_path), map_location="cpu")
    resolved_model_type = _resolve_model_type(payload, explicit_model_type=model_type)
    artifact_metadata = payload.get("artifact_metadata", {})
    dataset_name = dataset or artifact_metadata.get("dataset_name")
    if dataset_name is None:
        raise ValueError(
            "Dataset name was not provided and checkpoint does not store artifact_metadata.dataset_name."
        )

    preprocess_config = _resolve_preprocess_config(
        payload,
        normalize_features=config.normalize_features,
        validate_auxiliary=config.validate_auxiliary,
    )
    raw_dataset = load_dataset(dataset_name, data_root=data_root)
    processed_dataset, _ = preprocess_dataset(raw_dataset, preprocess_config)
    device = resolve_device(config.device)
    feature_bundle = build_feature_bundle(
        processed_dataset,
        FeatureBuilderConfig(device=device),
    )

    resolved_graph_mode = _resolve_graph_mode(resolved_model_type, config.graph_mode)
    known_mask = _build_known_drug_disease_mask(processed_dataset)

    graph = None
    graph_report = None
    if resolved_model_type == "baseline":
        model = _load_baseline_model(payload, processed_dataset)
    else:
        graph_config = _resolve_graph_build_config(
            payload,
            add_reverse_edges=config.add_reverse_edges,
            add_self_loops=config.add_self_loops,
        )
        if resolved_graph_mode == "train":
            split_config = _resolve_split_config(payload)
            split_bundle, _ = create_drug_disease_splits(processed_dataset, split_config)
            graph, graph_report = build_train_hetero_graph(
                processed_dataset,
                split_bundle,
                graph_config,
            )
        elif resolved_graph_mode == "full":
            graph, graph_report = build_full_hetero_graph(processed_dataset, graph_config)
        else:
            raise ValueError(f"HGT inference requires graph_mode 'train' or 'full', got '{resolved_graph_mode}'.")

        model = _load_hgt_model(payload, processed_dataset, graph)
        graph = graph.to(device)

    model = model.to(device)
    model.eval()

    return InferenceSession(
        checkpoint_path=str(Path(checkpoint_path).resolve()),
        checkpoint_payload=payload,
        model_type=resolved_model_type,
        dataset=processed_dataset,
        feature_bundle=feature_bundle,
        model=model,
        device=str(device),
        graph_mode=resolved_graph_mode,
        graph=graph,
        graph_report=graph_report,
        known_drug_disease_mask=known_mask,
    )


@torch.no_grad()
def predict_top_k_diseases(
    session: InferenceSession,
    drug_query: str | int,
    *,
    top_k: int | None = None,
    exclude_known_associations: bool | None = None,
) -> TopKPredictionResult:
    """Score one drug against every disease and return the top-ranked candidates."""

    resolved_top_k = int(top_k or session.checkpoint_payload.get("artifact_metadata", {}).get("default_top_k", 0) or 10)
    if resolved_top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    if exclude_known_associations is None:
        exclude_known_associations = True

    drug_index = resolve_node_query(session.feature_bundle, "drug", drug_query)
    drug_record = get_node_record(session.feature_bundle, "drug", drug_index)
    known_mask = session.known_drug_disease_mask[drug_index]

    if session.model_type == "baseline":
        disease_indices, logits = _score_with_baseline(session, drug_index)
    else:
        disease_indices, logits = _score_with_hgt(session, drug_index)

    logits_np = logits.detach().cpu().numpy().astype(np.float64, copy=False)
    probabilities = 1.0 / (1.0 + np.exp(-logits_np))
    disease_indices_np = disease_indices.detach().cpu().numpy().astype(np.int64, copy=False)

    candidate_rows: list[dict[str, Any]] = []
    filtered_known = 0
    for disease_index, logit, probability in zip(
        disease_indices_np.tolist(),
        logits_np.tolist(),
        probabilities.tolist(),
        strict=True,
    ):
        is_known = bool(known_mask[disease_index])
        if exclude_known_associations and is_known:
            filtered_known += 1
            continue
        candidate_rows.append(
            {
                "disease_index": int(disease_index),
                "logit": float(logit),
                "probability": float(probability),
                "known_association": is_known,
            }
        )

    candidate_rows.sort(
        key=lambda row: (-row["probability"], -row["logit"], row["disease_index"])
    )
    top_rows = candidate_rows[:resolved_top_k]

    predictions = [
        PredictionRecord(
            rank=rank,
            disease_index=row["disease_index"],
            disease_record=get_node_record(session.feature_bundle, "disease", row["disease_index"]),
            logit=row["logit"],
            probability=row["probability"],
            known_association=row["known_association"],
        )
        for rank, row in enumerate(top_rows, start=1)
    ]

    return TopKPredictionResult(
        dataset_name=session.dataset.dataset_name,
        model_type=session.model_type,
        checkpoint_path=session.checkpoint_path,
        graph_mode=session.graph_mode,
        top_k=resolved_top_k,
        num_candidates_considered=len(candidate_rows),
        num_known_filtered=filtered_known,
        drug_record=drug_record,
        predictions=predictions,
    )


@torch.no_grad()
def score_drug_disease_pair(
    session: InferenceSession,
    drug_query: str | int,
    disease_query: str | int,
) -> dict[str, Any]:
    """Score one concrete drug-disease pair for API-style single-pair inference."""

    drug_index = resolve_node_query(session.feature_bundle, "drug", drug_query)
    disease_index = resolve_node_query(session.feature_bundle, "disease", disease_query)

    if session.model_type == "baseline":
        pair_features, _, _ = build_all_candidate_pair_features(
            session.dataset,
            drug_index=drug_index,
            device=session.device,
        )
        logits = session.model(pair_features)
        logit = logits[disease_index]
    else:
        drug_indices = torch.tensor([drug_index], dtype=torch.long, device=session.device)
        disease_indices = torch.tensor([disease_index], dtype=torch.long, device=session.device)
        logit = session.model(session.graph, drug_indices, disease_indices).squeeze(0)

    probability = torch.sigmoid(logit).item()
    is_known = bool(session.known_drug_disease_mask[drug_index, disease_index])

    return {
        "dataset_name": session.dataset.dataset_name,
        "model_type": session.model_type,
        "checkpoint_path": session.checkpoint_path,
        "graph_mode": session.graph_mode,
        "drug": get_node_record(session.feature_bundle, "drug", drug_index),
        "disease": get_node_record(session.feature_bundle, "disease", disease_index),
        "logit": float(logit.item()),
        "probability": float(probability),
        "known_association": is_known,
    }


def export_web_artifacts(
    session: InferenceSession,
    output_dir: str | Path,
) -> dict[str, str]:
    """Export lightweight lookup tables and metadata for a future web frontend."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "dataset_name": session.dataset.dataset_name,
        "model_type": session.model_type,
        "checkpoint_path": session.checkpoint_path,
        "graph_mode": session.graph_mode,
        "node_counts": session.dataset.node_counts,
        "feature_dims": session.dataset.feature_dims,
        "artifact_metadata": session.checkpoint_payload.get("artifact_metadata", {}),
        "graph_report": (
            summarize_graph_report(session.graph_report)
            if session.graph_report is not None
            else None
        ),
    }

    metadata_path = output_path / "metadata.json"
    drug_lookup_path = output_path / "drug_lookup.json"
    disease_lookup_path = output_path / "disease_lookup.json"
    protein_lookup_path = output_path / "protein_lookup.json"

    _write_json(metadata_path, metadata)
    _write_json(drug_lookup_path, export_node_table(session.feature_bundle, "drug"))
    _write_json(disease_lookup_path, export_node_table(session.feature_bundle, "disease"))
    _write_json(protein_lookup_path, export_node_table(session.feature_bundle, "protein"))

    return {
        "metadata": str(metadata_path.resolve()),
        "drug_lookup": str(drug_lookup_path.resolve()),
        "disease_lookup": str(disease_lookup_path.resolve()),
        "protein_lookup": str(protein_lookup_path.resolve()),
    }


def summarize_inference_session(session: InferenceSession) -> dict[str, Any]:
    """Return a compact summary useful for logging."""

    return {
        "checkpoint_path": session.checkpoint_path,
        "model_type": session.model_type,
        "dataset_name": session.dataset.dataset_name,
        "device": session.device,
        "graph_mode": session.graph_mode,
        "node_counts": session.dataset.node_counts,
        "feature_dims": session.dataset.feature_dims,
        "graph_report": (
            summarize_graph_report(session.graph_report)
            if session.graph_report is not None
            else None
        ),
    }


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    session = load_inference_session(
        checkpoint_path=args.checkpoint_path,
        dataset=args.dataset,
        data_root=args.data_root,
        model_type=args.model,
        config=InferenceConfig(
            device=args.device,
            top_k=args.top_k,
            graph_mode=args.graph_mode,
            normalize_features=args.normalize_features,
            validate_auxiliary=not args.skip_auxiliary_validation,
            add_reverse_edges=None if args.disable_reverse_edges is None else (not args.disable_reverse_edges),
            add_self_loops=args.add_self_loops,
            exclude_known_associations=not args.include_known_associations,
        ),
    )

    prediction_result = predict_top_k_diseases(
        session,
        drug_query=args.drug_query,
        top_k=args.top_k,
        exclude_known_associations=not args.include_known_associations,
    )

    cli_summary = {
        "session": summarize_inference_session(session),
        "prediction": prediction_result.to_dict(),
    }

    if args.export_dir is not None:
        cli_summary["exported_artifacts"] = export_web_artifacts(session, args.export_dir)

    print(json.dumps(cli_summary, indent=2))
    if args.result_json is not None:
        _write_json(Path(args.result_json), cli_summary)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run deployment-oriented top-k inference for trained drug-disease models.",
    )
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--model", choices=("baseline", "hgt"), default=None)
    parser.add_argument("--dataset", choices=AVAILABLE_DATASETS, default=None)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--drug-query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument("--graph-mode", choices=("none", "train", "full"), default=None)
    parser.add_argument(
        "--normalize-features",
        choices=("none", "zscore", "l2"),
        default=None,
    )
    parser.add_argument("--skip-auxiliary-validation", action="store_true")
    parser.add_argument("--include-known-associations", action="store_true")
    parser.add_argument("--disable-reverse-edges", action="store_true", default=None)
    parser.add_argument("--add-self-loops", action="store_true", default=None)
    parser.add_argument("--result-json", default=None)
    parser.add_argument("--export-dir", default=None)
    return parser


def _resolve_model_type(
    payload: dict[str, Any],
    *,
    explicit_model_type: ModelType | None,
) -> ModelType:
    if explicit_model_type is not None:
        return explicit_model_type

    artifact_metadata = payload.get("artifact_metadata", {})
    if artifact_metadata.get("model_type") in {"baseline", "hgt"}:
        return artifact_metadata["model_type"]

    model_class = str(payload.get("model_class", "")).casefold()
    if "hgt" in model_class:
        return "hgt"
    if "baseline" in model_class or "mlp" in model_class:
        return "baseline"
    raise ValueError("Could not infer model type from checkpoint payload.")


def _resolve_graph_mode(model_type: ModelType, graph_mode: GraphMode | None) -> GraphMode:
    if graph_mode is not None:
        return graph_mode
    return "none" if model_type == "baseline" else "full"


def _resolve_preprocess_config(
    payload: dict[str, Any],
    *,
    normalize_features: NormalizeMode | None,
    validate_auxiliary: bool,
) -> PreprocessConfig:
    artifact_metadata = payload.get("artifact_metadata", {})
    saved_config = artifact_metadata.get("preprocess_config", {})
    config_dict = {
        "normalize_features": saved_config.get("normalize_features", "none"),
        "replace_non_finite_with": saved_config.get("replace_non_finite_with", 0.0),
        "deduplicate_edges": saved_config.get("deduplicate_edges", True),
        "validate_auxiliary": validate_auxiliary,
    }
    if normalize_features is not None:
        config_dict["normalize_features"] = normalize_features
    return PreprocessConfig(**config_dict)


def _resolve_graph_build_config(
    payload: dict[str, Any],
    *,
    add_reverse_edges: bool | None,
    add_self_loops: bool | None,
) -> GraphBuildConfig:
    artifact_metadata = payload.get("artifact_metadata", {})
    saved_config = artifact_metadata.get("graph_build_config", {})
    config_dict = {
        "add_reverse_edges": saved_config.get("add_reverse_edges", True),
        "add_self_loops": saved_config.get("add_self_loops", False),
        "self_loop_relation": saved_config.get("self_loop_relation", "self_loop"),
        "attach_node_ids": saved_config.get("attach_node_ids", True),
        "validate_graph": saved_config.get("validate_graph", True),
    }
    if add_reverse_edges is not None:
        config_dict["add_reverse_edges"] = add_reverse_edges
    if add_self_loops is not None:
        config_dict["add_self_loops"] = add_self_loops
    return GraphBuildConfig(**config_dict)


def _resolve_split_config(payload: dict[str, Any]) -> SplitConfig:
    artifact_metadata = payload.get("artifact_metadata", {})
    saved_config = artifact_metadata.get("split_config")
    if saved_config is None:
        raise ValueError(
            "Checkpoint does not store split_config, so graph_mode='train' cannot be reconstructed safely."
        )
    return SplitConfig(**saved_config)


def _load_baseline_model(
    payload: dict[str, Any],
    dataset: RawDataset,
) -> DrugDiseaseMLPBaseline:
    model_config = payload.get("model_config", {})
    model = DrugDiseaseMLPBaseline(
        drug_feature_dim=int(payload.get("drug_feature_dim", dataset.feature_dims["drug"])),
        disease_feature_dim=int(payload.get("disease_feature_dim", dataset.feature_dims["disease"])),
        config=BaselineMLPConfig(**model_config),
    )
    model.load_state_dict(payload["model_state_dict"])
    return model


def _load_hgt_model(
    payload: dict[str, Any],
    dataset: RawDataset,
    graph: Any,
) -> DrugDiseaseHGT:
    model_config = dict(payload.get("model_config", {}))

    # Backward compatibility: older checkpoints may save `hidden_dims`
    # while current HGTModelConfig expects `hidden_dim`.
    if "hidden_dim" not in model_config and "hidden_dims" in model_config:
        legacy_hidden = model_config.pop("hidden_dims")
        if isinstance(legacy_hidden, (list, tuple)):
            model_config["hidden_dim"] = int(legacy_hidden[0]) if len(legacy_hidden) > 0 else 128
        else:
            model_config["hidden_dim"] = int(legacy_hidden)

    input_dims = payload.get("input_dims", dataset.feature_dims)
    model = DrugDiseaseHGT(
        metadata=graph.metadata(),
        input_dims=input_dims,
        config=HGTModelConfig(**model_config),
    )
    model.load_state_dict(payload["model_state_dict"])
    return model


def _build_known_drug_disease_mask(dataset: RawDataset) -> np.ndarray:
    mask = np.zeros(
        (dataset.node_counts["drug"], dataset.node_counts["disease"]),
        dtype=bool,
    )
    edge = dataset.edges["drug_disease"]
    mask[edge.source_index, edge.target_index] = True
    return mask


@torch.no_grad()
def _score_with_baseline(
    session: InferenceSession,
    drug_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    pair_features, _, disease_indices = build_all_candidate_pair_features(
        session.dataset,
        drug_index=drug_index,
        device=session.device,
    )
    logits = session.model(pair_features)
    return disease_indices, logits


@torch.no_grad()
def _score_with_hgt(
    session: InferenceSession,
    drug_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_diseases = session.dataset.node_counts["disease"]
    drug_indices = torch.full(
        (num_diseases,),
        int(drug_index),
        dtype=torch.long,
        device=session.device,
    )
    disease_indices = torch.arange(
        num_diseases,
        dtype=torch.long,
        device=session.device,
    )
    logits = session.model(session.graph, drug_indices, disease_indices)
    return disease_indices, logits


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


if __name__ == "__main__":
    main()
