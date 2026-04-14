"""Core package exports for the HGT drug-protein-disease pipeline."""

from .baseline import (
    BaselineMLPConfig,
    DrugDiseaseMLPBaseline,
    build_all_candidate_pair_features,
    build_baseline_model,
    build_pair_feature_tensor,
    summarize_pair_tensor,
)
from .data_loader import AVAILABLE_DATASETS, RawDataset, load_dataset
from .evaluator import (
    BinaryClassificationMetrics,
    evaluate_binary_classification,
    summarize_metrics,
)
from .feature_builder import (
    FeatureBuilderConfig,
    FeatureBundle,
    NodeLookupTable,
    build_feature_bundle,
    build_feature_tensor_dict,
    export_node_table,
    get_node_record,
    resolve_node_query,
    summarize_feature_bundle,
)
from .graph_builder import (
    GraphBuildConfig,
    GraphBuildReport,
    build_full_hetero_graph,
    build_hetero_graph,
    build_train_hetero_graph,
    summarize_graph_report,
)
from .model_fusion_hgt import (
    DrugDiseaseFusionHGT,
    FusionHGTModelConfig,
    build_fusion_hgt_model,
    summarize_fusion_hgt_model,
)
from .model_hgt import (
    DrugDiseaseHGT,
    HGTModelConfig,
    build_hgt_model,
    score_all_diseases_for_drug,
    summarize_hgt_model,
)
from .preprocess import PreprocessConfig, ValidationReport, preprocess_dataset
from .split import (
    DrugDiseaseSplit,
    PairDataset,
    SplitConfig,
    SplitReport,
    build_train_graph_edges,
    create_drug_disease_splits,
    summarize_split_report,
)
from .trainer import (
    TrainerConfig,
    TrainingResult,
    evaluate_baseline_model,
    evaluate_hgt_model,
    resolve_device,
    summarize_training_result,
    train_baseline_model,
    train_hgt_model,
)
from .similarity_graph import (
    SimilarityGraphBundle,
    SimilarityGraphConfig,
    SimilarityGraphData,
    build_similarity_graph_bundle,
    summarize_similarity_graph_bundle,
)

__all__ = [
    "AVAILABLE_DATASETS",
    "BaselineMLPConfig",
    "BinaryClassificationMetrics",
    "DrugDiseaseSplit",
    "DrugDiseaseFusionHGT",
    "DrugDiseaseMLPBaseline",
    "DrugDiseaseHGT",
    "FeatureBuilderConfig",
    "FeatureBundle",
    "FusionHGTModelConfig",
    "GraphBuildConfig",
    "GraphBuildReport",
    "HGTModelConfig",
    "InferenceConfig",
    "InferenceSession",
    "NodeLookupTable",
    "PairDataset",
    "PredictionRecord",
    "PreprocessConfig",
    "RawDataset",
    "SimilarityGraphBundle",
    "SimilarityGraphConfig",
    "SimilarityGraphData",
    "SplitConfig",
    "SplitReport",
    "TopKPredictionResult",
    "TrainerConfig",
    "TrainingResult",
    "ValidationReport",
    "build_all_candidate_pair_features",
    "build_baseline_model",
    "build_feature_bundle",
    "build_feature_tensor_dict",
    "build_fusion_hgt_model",
    "build_hgt_model",
    "build_pair_feature_tensor",
    "build_full_hetero_graph",
    "build_hetero_graph",
    "build_similarity_graph_bundle",
    "build_train_hetero_graph",
    "build_train_graph_edges",
    "create_drug_disease_splits",
    "evaluate_baseline_model",
    "evaluate_binary_classification",
    "evaluate_hgt_model",
    "export_node_table",
    "export_web_artifacts",
    "get_node_record",
    "load_inference_session",
    "load_dataset",
    "preprocess_dataset",
    "predict_top_k_diseases",
    "resolve_device",
    "resolve_node_query",
    "score_drug_disease_pair",
    "score_all_diseases_for_drug",
    "summarize_feature_bundle",
    "summarize_fusion_hgt_model",
    "summarize_graph_report",
    "summarize_hgt_model",
    "summarize_inference_session",
    "summarize_metrics",
    "summarize_pair_tensor",
    "summarize_similarity_graph_bundle",
    "summarize_split_report",
    "summarize_training_result",
    "train_baseline_model",
    "train_hgt_model",
]

_LAZY_INFERENCE_EXPORTS = {
    "InferenceConfig",
    "InferenceSession",
    "PredictionRecord",
    "TopKPredictionResult",
    "export_web_artifacts",
    "load_inference_session",
    "predict_top_k_diseases",
    "score_drug_disease_pair",
    "summarize_inference_session",
}


def __getattr__(name: str):
    if name in _LAZY_INFERENCE_EXPORTS:
        from . import inference as _inference

        value = getattr(_inference, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'src' has no attribute '{name}'")
