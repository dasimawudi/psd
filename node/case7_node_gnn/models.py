from __future__ import annotations

import torch
from torch import nn

from case7_node_mlp.models import _activation


def _require_transformer_conv() -> type[nn.Module]:
    try:
        from torch_geometric.nn import TransformerConv
    except ImportError as exc:
        raise ImportError(
            "TransformerConv requires torch_geometric. Install PyG in the training environment before running "
            "node/train_node_transformerconv.py."
        ) from exc
    return TransformerConv


class NodeTransformerConv(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_layers: int = 2,
        heads: int = 4,
        edge_dim: int = 4,
        dropout: float = 0.1,
        activation: str = "silu",
        use_layer_norm: bool = True,
        beta: bool = True,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads when concat=True.")

        TransformerConv = _require_transformer_conv()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            _activation(activation),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )
        convs: list[nn.Module] = []
        norms: list[nn.Module] = []
        for _ in range(num_layers):
            convs.append(
                TransformerConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // heads,
                    heads=heads,
                    concat=True,
                    beta=beta,
                    dropout=dropout,
                    edge_dim=edge_dim,
                )
            )
            norms.append(nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity())
        self.convs = nn.ModuleList(convs)
        self.norms = nn.ModuleList(norms)
        self.activation = _activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            _activation(activation),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(features)
        for conv, norm in zip(self.convs, self.norms):
            update = conv(hidden, edge_index=edge_index, edge_attr=edge_attr)
            hidden = norm(hidden + self.dropout(self.activation(update)))
        return self.decoder(hidden)

