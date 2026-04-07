"""PyG Heterogeneous Graph Transformer for drug-disease link prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv

try:
    from .data_loader import RawDataset
except ImportError:  # pragma: no cover - allows direct script execution
    from data_loader import RawDataset


@dataclass(frozen=True, slots=True)
class HGTModelConfig:
    """Hyperparameters for the HGT encoder and pair decoder."""

    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.2
    decoder_hidden_dims: tuple[int, ...] = (256, 128)
    decoder_mode: str = "product"
    activation: str = "gelu"
    use_layer_norm: bool = True


class DrugDiseaseHGT(nn.Module):
    """HGT encoder plus an MLP decoder over drug-disease pairs."""

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        input_dims: dict[str, int],
        config: HGTModelConfig = HGTModelConfig(),
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

        self.convs = nn.ModuleList(
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
            self.layer_norms = nn.ModuleList(
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
        else:
            self.layer_norms = None

        activation = _make_activation(config.activation)
        decoder_layers: list[nn.Module] = []
        decoder_in_dim = _decoder_input_dim(config.hidden_dim, config.decoder_mode)
        for hidden_dim in config.decoder_hidden_dims:
            decoder_layers.append(nn.Linear(decoder_in_dim, hidden_dim))
            decoder_layers.append(activation)
            decoder_layers.append(nn.Dropout(config.dropout))
            decoder_in_dim = hidden_dim
        decoder_layers.append(nn.Linear(decoder_in_dim, 1))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """Encode all node types into contextual embeddings."""

        x_dict = {
            node_type: self.input_projection[node_type](data[node_type].x)
            for node_type in self.metadata[0]
        }

        for layer_idx, conv in enumerate(self.convs):
            conv_out = conv(x_dict, data.edge_index_dict)
            next_x: dict[str, torch.Tensor] = {}
            for node_type in self.metadata[0]:
                previous = x_dict[node_type]
                updated = conv_out.get(node_type)
                if updated is None:
                    updated = previous
                else:
                    updated = updated + previous
                    if self.layer_norms is not None:
                        updated = self.layer_norms[layer_idx][node_type](updated)
                    updated = _apply_activation(updated, self.config.activation)
                    updated = F.dropout(
                        updated,
                        p=self.config.dropout,
                        training=self.training,
                    )
                next_x[node_type] = updated
            x_dict = next_x

        return x_dict

    def decode(
        self,
        embeddings: dict[str, torch.Tensor],
        drug_index: torch.Tensor,
        disease_index: torch.Tensor,
    ) -> torch.Tensor:
        """Score drug-disease pairs from encoded node embeddings."""

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


def build_hgt_model(
    dataset: RawDataset,
    graph: HeteroData,
    config: HGTModelConfig = HGTModelConfig(),
) -> DrugDiseaseHGT:
    """Build the HGT model directly from dataset dimensions and graph metadata."""

    return DrugDiseaseHGT(
        metadata=graph.metadata(),
        input_dims=dataset.feature_dims,
        config=config,
    )


@torch.no_grad()
def score_all_diseases_for_drug(
    model: DrugDiseaseHGT,
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


def summarize_hgt_model(model: DrugDiseaseHGT) -> dict[str, Any]:
    """Return a compact, serializable summary of the HGT architecture."""

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
    }


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
