from __future__ import annotations

import torch
from torch import nn


def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int = 2,
    dropout: float = 0.0,
    final_activation: bool = False,
) -> nn.Sequential:
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")

    layers: list[nn.Module] = []
    dims = [input_dim]
    if num_layers == 1:
        dims.append(output_dim)
    else:
        dims.extend([hidden_dim] * (num_layers - 1))
        dims.append(output_dim)

    for idx in range(len(dims) - 1):
        in_dim = dims[idx]
        out_dim = dims[idx + 1]
        layers.append(nn.Linear(in_dim, out_dim))
        is_last = idx == len(dims) - 2
        if (not is_last) or final_activation:
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class EdgeMessagePassingLayer(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, global_dim: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.message_mlp = build_mlp(
            input_dim=hidden_dim + edge_dim + global_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
        )
        self.update_mlp = build_mlp(
            input_dim=hidden_dim + hidden_dim + global_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        node_state: torch.Tensor,
        edge_index: torch.Tensor,
        edge_state: torch.Tensor,
        global_state: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index
        num_nodes = node_state.size(0)
        num_edges = edge_index.size(1)

        global_edges = global_state.unsqueeze(0).expand(num_edges, -1)
        message_input = torch.cat([node_state[src], edge_state, global_edges], dim=-1)
        messages = self.message_mlp(message_input)

        aggregated = torch.zeros(num_nodes, self.hidden_dim, device=node_state.device, dtype=node_state.dtype)
        aggregated.index_add_(0, dst, messages)

        degree = torch.zeros(num_nodes, 1, device=node_state.device, dtype=node_state.dtype)
        degree.index_add_(0, dst, torch.ones(num_edges, 1, device=node_state.device, dtype=node_state.dtype))
        aggregated = aggregated / degree.clamp_min(1.0)

        global_nodes = global_state.unsqueeze(0).expand(num_nodes, -1)
        update_input = torch.cat([node_state, aggregated, global_nodes], dim=-1)
        delta = self.update_mlp(update_input)
        return self.norm(node_state + delta)


class GraphEncoder(nn.Module):
    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        global_input_dim: int,
        hidden_dim: int,
        global_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.node_encoder = build_mlp(
            input_dim=node_input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
        )
        self.edge_encoder = build_mlp(
            input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
        )
        self.global_encoder = build_mlp(
            input_dim=global_input_dim,
            hidden_dim=global_dim,
            output_dim=global_dim,
            num_layers=2,
            dropout=dropout,
        )
        self.layers = nn.ModuleList(
            [
                EdgeMessagePassingLayer(
                    hidden_dim=hidden_dim,
                    edge_dim=hidden_dim,
                    global_dim=global_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_state = self.node_encoder(node_features)
        edge_state = self.edge_encoder(edge_features)
        global_state = self.global_encoder(global_features.unsqueeze(0)).squeeze(0)

        for layer in self.layers:
            node_state = layer(node_state, edge_index, edge_state, global_state)

        pooled_mean = node_state.mean(dim=0)
        pooled_max = node_state.max(dim=0).values
        graph_state = torch.cat([pooled_mean, pooled_max, global_state], dim=-1)
        return node_state, graph_state, global_state


class FrequencyGNN(nn.Module):
    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        global_input_dim: int,
        hidden_dim: int,
        global_dim: int,
        num_layers: int,
        dropout: float,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.encoder = GraphEncoder(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            global_input_dim=global_input_dim,
            hidden_dim=hidden_dim,
            global_dim=global_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.readout = build_mlp(
            input_dim=hidden_dim * 2 + global_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=3,
            dropout=dropout,
            final_activation=False,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        _, graph_state, _ = self.encoder(node_features, edge_index, edge_features, global_features)
        return self.readout(graph_state)


class FieldGNN(nn.Module):
    def __init__(
        self,
        node_input_dim: int,
        edge_input_dim: int,
        global_input_dim: int,
        hidden_dim: int,
        global_dim: int,
        num_layers: int,
        dropout: float,
        output_dim: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = GraphEncoder(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            global_input_dim=global_input_dim,
            hidden_dim=hidden_dim,
            global_dim=global_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.decoder = build_mlp(
            input_dim=hidden_dim + global_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=3,
            dropout=dropout,
            final_activation=False,
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        node_state, _, global_state = self.encoder(node_features, edge_index, edge_features, global_features)
        global_nodes = global_state.unsqueeze(0).expand(node_state.size(0), -1)
        return self.decoder(torch.cat([node_state, global_nodes], dim=-1))
