"""Fusion HGT model with drug/disease similarity branches for AMDGT-lite++."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, TransformerConv

try:
    from .data_loader import RawDataset
    from .similarity_graph import SimilarityGraphBundle
except ImportError:  # pragma: no cover - allows direct script execution
    from data_loader import RawDataset
    from similarity_graph import SimilarityGraphBundle


@dataclass(frozen=True, slots=True)
class FusionHGTModelConfig:
    """Hyperparameters for the heterograph encoder, similarity encoders, and fusion decoder."""

    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.2
    decoder_hidden_dims: tuple[int, ...] = (256, 128)
    decoder_mode: str = "hybrid"
    activation: str = "gelu"
    use_layer_norm: bool = True
    similarity_topk: int = 20
    sim_layers: int = 2
    sim_heads: int = 4
    sim_dropout: float = 0.2
    drug_fingerprint_weight: float = 0.5
    drug_gip_weight: float = 0.5
    disease_ps_weight: float = 0.5
    disease_gip_weight: float = 0.5
    symmetric_similarity: bool = True


class DrugDiseaseFusionHGT(nn.Module):
    """Three-branch graph encoder with learnable gated fusion for drug-disease scoring."""

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        input_dims: dict[str, int],
        similarity_graphs: SimilarityGraphBundle,
        config: FusionHGTModelConfig = FusionHGTModelConfig(),
    ) -> None:
        super().__init__()
        self.metadata = metadata
        self.input_dims = dict(input_dims)
        self.config = config

        self.input_projection = nn.ModuleDict(
            {
                node_type: nn.Linear(input_dim, config.hidden_dim)
                for node_type, input_dim in self.input_dims.items()
            }
        )

        self.hetero_convs = nn.ModuleList(
            [
                HGTConv(
                    in_channels=config.hidden_dim,
                    out_channels=config.hidden_dim,
                    metadata=metadata,
                    heads=config.num_heads,
                )
                for _ in range(config.num_layers)
            ]
        )

        if config.use_layer_norm:
            self.hetero_layer_norms = nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            node_type: nn.LayerNorm(config.hidden_dim)
                            for node_type in metadata[0]
                        }
                    )
                    for _ in range(config.num_layers)
                ]
            )
            self.sim_layer_norms = nn.ModuleDict(
                {
                    node_type: nn.ModuleList(
                        [nn.LayerNorm(config.hidden_dim) for _ in range(config.sim_layers)]
                    )
                    for node_type in ("drug", "disease")
                }
            )
        else:
            self.hetero_layer_norms = None
            self.sim_layer_norms = None

        self.drug_sim_convs = nn.ModuleList(
            [
                TransformerConv(
                    in_channels=config.hidden_dim,
                    out_channels=config.hidden_dim,
                    heads=config.sim_heads,
                    concat=False,
                    dropout=config.sim_dropout,
                    edge_dim=1,
                    beta=True,
                )
                for _ in range(config.sim_layers)
            ]
        )
        self.disease_sim_convs = nn.ModuleList(
            [
                TransformerConv(
                    in_channels=config.hidden_dim,
                    out_channels=config.hidden_dim,
                    heads=config.sim_heads,
                    concat=False,
                    dropout=config.sim_dropout,
                    edge_dim=1,
                    beta=True,
                )
                for _ in range(config.sim_layers)
            ]
        )

        gate_input_dim = config.hidden_dim * 4
        self.fusion_gates = nn.ModuleDict(
            {
                node_type: nn.Sequential(
                    nn.Linear(gate_input_dim, config.hidden_dim),
                    _make_activation(config.activation),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                )
                for node_type in ("drug", "disease")
            }
        )

        decoder_layers: list[nn.Module] = []
        decoder_input_dim = _decoder_input_dim(config.hidden_dim, config.decoder_mode)
        for hidden_dim in config.decoder_hidden_dims:
            decoder_layers.append(nn.Linear(decoder_input_dim, hidden_dim))
            decoder_layers.append(_make_activation(config.activation))
            decoder_layers.append(nn.Dropout(config.dropout))
            decoder_input_dim = hidden_dim
        decoder_layers.append(nn.Linear(decoder_input_dim, 1))
        self.decoder = nn.Sequential(*decoder_layers)

        self.register_buffer(
            "drug_similarity_edge_index",
            similarity_graphs.drug.edge_index.clone(),
            persistent=False,
        )
        self.register_buffer(
            "drug_similarity_edge_attr",
            similarity_graphs.drug.edge_attr.clone(),
            persistent=False,
        )
        self.register_buffer(
            "disease_similarity_edge_index",
            similarity_graphs.disease.edge_index.clone(),
            persistent=False,
        )
        self.register_buffer(
            "disease_similarity_edge_attr",
            similarity_graphs.disease.edge_attr.clone(),
            persistent=False,
        )

    def encode(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """Encode all node types, then fuse heterograph and similarity views."""

        projected = {
            node_type: self.input_projection[node_type](data[node_type].x)
            for node_type in self.metadata[0]
        }
        hetero_embeddings = self._encode_heterograph(projected, data)
        drug_similarity_embeddings = self._encode_similarity_branch(
            projected["drug"],
            self.drug_similarity_edge_index,
            self.drug_similarity_edge_attr,
            self.drug_sim_convs,
            node_type="drug",
        )
        disease_similarity_embeddings = self._encode_similarity_branch(
            projected["disease"],
            self.disease_similarity_edge_index,
            self.disease_similarity_edge_attr,
            self.disease_sim_convs,
            node_type="disease",
        )

        fused_embeddings = dict(hetero_embeddings)
        fused_embeddings["drug"] = self._fuse_embeddings(
            hetero_embeddings["drug"],
            drug_similarity_embeddings,
            node_type="drug",
        )
        fused_embeddings["disease"] = self._fuse_embeddings(
            hetero_embeddings["disease"],
            disease_similarity_embeddings,
            node_type="disease",
        )
        return fused_embeddings

    def decode(
        self,
        embeddings: dict[str, torch.Tensor],
        drug_index: torch.Tensor,
        disease_index: torch.Tensor,
    ) -> torch.Tensor:
        """Score drug-disease pairs from fused embeddings."""

        drug_embeddings = embeddings["drug"][drug_index]
        disease_embeddings = embeddings["disease"][disease_index]
        pair_embeddings = _build_pair_embeddings(
            drug_embeddings,
            disease_embeddings,
            mode=self.config.decoder_mode,
        )
        logits = self.decoder(pair_embeddings)
        return logits.squeeze(-1)

    def forward(
        self,
        data: HeteroData,
        drug_index: torch.Tensor,
        disease_index: torch.Tensor,
        *,
        return_embeddings: bool = False,
    ) -> torch.Tensor | tuple[dict[str, torch.Tensor], torch.Tensor]:
        embeddings = self.encode(data)
        logits = self.decode(embeddings, drug_index, disease_index)
        if return_embeddings:
            return embeddings, logits
        return logits

    def _encode_heterograph(
        self,
        projected: dict[str, torch.Tensor],
        data: HeteroData,
    ) -> dict[str, torch.Tensor]:
        x_dict = dict(projected)
        for layer_index, conv in enumerate(self.hetero_convs):
            conv_out = conv(x_dict, data.edge_index_dict)
            next_x: dict[str, torch.Tensor] = {}
            for node_type in self.metadata[0]:
                previous = x_dict[node_type]
                updated = conv_out.get(node_type)
                if updated is None:
                    updated = previous
                else:
                    updated = updated + previous
                    if self.hetero_layer_norms is not None:
                        updated = self.hetero_layer_norms[layer_index][node_type](updated)
                    updated = _apply_activation(updated, self.config.activation)
                    updated = F.dropout(
                        updated,
                        p=self.config.dropout,
                        training=self.training,
                    )
                next_x[node_type] = updated
            x_dict = next_x
        return x_dict

    def _encode_similarity_branch(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        convs: nn.ModuleList,
        *,
        node_type: str,
    ) -> torch.Tensor:
        if edge_index.numel() == 0:
            return node_features

        x = node_features
        for layer_index, conv in enumerate(convs):
            updated = conv(x, edge_index, edge_attr=edge_attr)
            updated = updated + x
            if self.sim_layer_norms is not None:
                updated = self.sim_layer_norms[node_type][layer_index](updated)
            updated = _apply_activation(updated, self.config.activation)
            updated = F.dropout(
                updated,
                p=self.config.sim_dropout,
                training=self.training,
            )
            x = updated
        return x

    def _fuse_embeddings(
        self,
        hetero_embedding: torch.Tensor,
        similarity_embedding: torch.Tensor,
        *,
        node_type: str,
    ) -> torch.Tensor:
        gate_input = torch.cat(
            (
                hetero_embedding,
                similarity_embedding,
                hetero_embedding * similarity_embedding,
                torch.abs(hetero_embedding - similarity_embedding),
            ),
            dim=-1,
        )
        gate = torch.sigmoid(self.fusion_gates[node_type](gate_input))
        return gate * hetero_embedding + (1.0 - gate) * similarity_embedding


def build_fusion_hgt_model(
    dataset: RawDataset,
    graph: HeteroData,
    similarity_graphs: SimilarityGraphBundle,
    config: FusionHGTModelConfig = FusionHGTModelConfig(),
) -> DrugDiseaseFusionHGT:
    """Build the fusion model directly from dataset metadata and sparse similarity graphs."""

    return DrugDiseaseFusionHGT(
        metadata=graph.metadata(),
        input_dims=dataset.feature_dims,
        similarity_graphs=similarity_graphs,
        config=config,
    )


def summarize_fusion_hgt_model(model: DrugDiseaseFusionHGT) -> dict[str, Any]:
    """Return a compact, serializable summary of the fusion architecture."""

    return {
        "metadata": model.metadata,
        "input_dims": dict(model.input_dims),
        "hidden_dim": model.config.hidden_dim,
        "num_layers": model.config.num_layers,
        "num_heads": model.config.num_heads,
        "dropout": model.config.dropout,
        "decoder_hidden_dims": list(model.config.decoder_hidden_dims),
        "decoder_mode": model.config.decoder_mode,
        "activation": model.config.activation,
        "use_layer_norm": model.config.use_layer_norm,
        "similarity_topk": model.config.similarity_topk,
        "sim_layers": model.config.sim_layers,
        "sim_heads": model.config.sim_heads,
        "sim_dropout": model.config.sim_dropout,
        "drug_similarity_edges": int(model.drug_similarity_edge_index.shape[1]),
        "disease_similarity_edges": int(model.disease_similarity_edge_index.shape[1]),
    }


@torch.no_grad()
def score_all_diseases_for_drug(
    model: DrugDiseaseFusionHGT,
    graph: HeteroData,
    drug_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return logits for one drug against every disease in the graph."""

    num_diseases = int(graph["disease"].num_nodes)
    if not 0 <= int(drug_index) < int(graph["drug"].num_nodes):
        raise ValueError(f"drug_index {drug_index} exceeds available drug nodes.")

    device = graph["drug"].x.device
    drug_indices = torch.full((num_diseases,), int(drug_index), dtype=torch.long, device=device)
    disease_indices = torch.arange(num_diseases, dtype=torch.long, device=device)
    logits = model(graph, drug_indices, disease_indices)
    return disease_indices, logits


def _decoder_input_dim(hidden_dim: int, mode: str) -> int:
    normalized = mode.strip().lower()
    if normalized == "product":
        return hidden_dim
    if normalized == "concat":
        return hidden_dim * 2
    if normalized == "hybrid":
        return hidden_dim * 4
    raise ValueError(f"Unsupported decoder_mode '{mode}'.")


def _build_pair_embeddings(
    drug_embeddings: torch.Tensor,
    disease_embeddings: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    normalized = mode.strip().lower()
    if normalized == "product":
        return drug_embeddings * disease_embeddings
    if normalized == "concat":
        return torch.cat((drug_embeddings, disease_embeddings), dim=-1)
    if normalized == "hybrid":
        return torch.cat(
            (
                drug_embeddings,
                disease_embeddings,
                drug_embeddings * disease_embeddings,
                torch.abs(drug_embeddings - disease_embeddings),
            ),
            dim=-1,
        )
    raise ValueError(f"Unsupported decoder_mode '{mode}'.")


def _make_activation(name: str) -> nn.Module:
    normalized = name.strip().lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation '{name}'.")


def _apply_activation(tensor: torch.Tensor, name: str) -> torch.Tensor:
    normalized = name.strip().lower()
    if normalized == "relu":
        return F.relu(tensor)
    if normalized == "gelu":
        return F.gelu(tensor)
    raise ValueError(f"Unsupported activation '{name}'.")
