from __future__ import annotations

import torch
from torch import nn


def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    dropout: float,
    final_activation: bool = False,
) -> nn.Sequential:
    if num_layers < 1:
        raise ValueError("num_layers must be >= 1")

    dims = [input_dim]
    if num_layers == 1:
        dims.append(output_dim)
    else:
        dims.extend([hidden_dim] * (num_layers - 1))
        dims.append(output_dim)

    layers: list[nn.Module] = []
    for idx in range(len(dims) - 1):
        layers.append(nn.Linear(dims[idx], dims[idx + 1]))
        is_last = idx == len(dims) - 2
        if (not is_last) or final_activation:
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class FeatureModulation(nn.Module):
    def __init__(self, feature_dim: int, conditioning_dim: int) -> None:
        super().__init__()
        self.affine = nn.Linear(conditioning_dim, feature_dim * 2)

    def forward(self, features: torch.Tensor, conditioning_state: torch.Tensor) -> torch.Tensor:
        scale_shift = self.affine(conditioning_state)
        scale, shift = torch.chunk(scale_shift, chunks=2, dim=-1)
        while scale.dim() < features.dim():
            scale = scale.unsqueeze(0)
            shift = shift.unsqueeze(0)
        return features * (1.0 + torch.tanh(scale)) + shift


class ConditionalResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, conditioning_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.modulation = FeatureModulation(hidden_dim, conditioning_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, node_state: torch.Tensor, conditioning_nodes: torch.Tensor) -> torch.Tensor:
        hidden = self.modulation(self.norm(node_state), conditioning_nodes)
        return node_state + self.ffn(hidden).to(dtype=node_state.dtype)


class ConditionalFieldMLP(nn.Module):
    """Per-node stress model that reuses the GNN trainer/evaluator interface.

    The edge tensors are accepted for compatibility with the existing training
    loop, but this model intentionally ignores mesh message passing.
    """

    def __init__(
        self,
        node_input_dim: int,
        global_input_dim: int,
        hidden_dim: int,
        global_dim: int,
        num_layers: int,
        dropout: float,
        conditioning_dim: int,
        output_dim: int,
        use_two_stage_rmises: bool,
        use_peak_relative_stress: bool,
        head_layers: int = 3,
    ) -> None:
        super().__init__()
        expected_output_dim = (1 if use_two_stage_rmises else 0) + (2 if use_peak_relative_stress else 1)
        if output_dim != expected_output_dim:
            raise ValueError("ConditionalFieldMLP output_dim does not match the requested stress heads.")

        self.use_two_stage_rmises = use_two_stage_rmises
        self.use_peak_relative_stress = use_peak_relative_stress
        self.node_encoder = build_mlp(
            input_dim=node_input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=2,
            dropout=dropout,
        )
        self.case_encoder = build_mlp(
            input_dim=global_input_dim,
            hidden_dim=hidden_dim,
            output_dim=conditioning_dim,
            num_layers=3,
            dropout=dropout,
        )
        self.global_encoder = build_mlp(
            input_dim=conditioning_dim,
            hidden_dim=global_dim,
            output_dim=global_dim,
            num_layers=2,
            dropout=dropout,
        )
        self.node_blocks = nn.ModuleList(
            [ConditionalResidualBlock(hidden_dim, conditioning_dim, dropout) for _ in range(num_layers)]
        )

        node_context_dim = hidden_dim + global_dim
        graph_context_dim = hidden_dim * 2 + global_dim
        if self.use_two_stage_rmises:
            self.hotspot_decoder = build_mlp(
                input_dim=node_context_dim,
                hidden_dim=hidden_dim,
                output_dim=1,
                num_layers=head_layers,
                dropout=dropout,
            )
            self.stress_decoder = build_mlp(
                input_dim=node_context_dim + 1,
                hidden_dim=hidden_dim,
                output_dim=1,
                num_layers=head_layers,
                dropout=dropout,
            )
        else:
            self.hotspot_decoder = None
            self.stress_decoder = build_mlp(
                input_dim=node_context_dim,
                hidden_dim=hidden_dim,
                output_dim=1,
                num_layers=head_layers,
                dropout=dropout,
            )

        self.peak_decoder = (
            build_mlp(
                input_dim=graph_context_dim,
                hidden_dim=hidden_dim,
                output_dim=1,
                num_layers=head_layers,
                dropout=dropout,
            )
            if self.use_peak_relative_stress
            else None
        )

    def _expand_graph_state_to_nodes(
        self,
        graph_state: torch.Tensor,
        node_count: int,
        node_graph_index: torch.Tensor | None,
    ) -> torch.Tensor:
        if graph_state.dim() == 1:
            return graph_state.unsqueeze(0).expand(node_count, -1)
        if node_graph_index is None:
            raise ValueError("Batched global features require node_graph_index.")
        return graph_state[node_graph_index]

    def _pool_graph_state(
        self,
        node_state: torch.Tensor,
        global_state: torch.Tensor,
        node_graph_index: torch.Tensor | None,
    ) -> torch.Tensor:
        if global_state.dim() == 1:
            return torch.cat([node_state.mean(dim=0), node_state.max(dim=0).values, global_state], dim=-1)
        if node_graph_index is None:
            raise ValueError("Batched global features require node_graph_index.")

        pooled_parts: list[torch.Tensor] = []
        for graph_idx in range(global_state.size(0)):
            mask = node_graph_index == graph_idx
            graph_nodes = node_state[mask]
            pooled_parts.append(
                torch.cat(
                    [graph_nodes.mean(dim=0), graph_nodes.max(dim=0).values, global_state[graph_idx]],
                    dim=-1,
                )
            )
        return torch.stack(pooled_parts, dim=0)

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        global_features: torch.Tensor,
        node_graph_index: torch.Tensor | None = None,
        edge_graph_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del edge_index, edge_features, edge_graph_index

        if global_features.dim() == 1:
            conditioning_state = self.case_encoder(global_features.unsqueeze(0)).squeeze(0)
            global_state = self.global_encoder(conditioning_state.unsqueeze(0)).squeeze(0)
        else:
            conditioning_state = self.case_encoder(global_features)
            global_state = self.global_encoder(conditioning_state)

        conditioning_nodes = self._expand_graph_state_to_nodes(
            conditioning_state,
            node_count=node_features.size(0),
            node_graph_index=node_graph_index,
        )
        global_nodes = self._expand_graph_state_to_nodes(
            global_state,
            node_count=node_features.size(0),
            node_graph_index=node_graph_index,
        )

        node_state = self.node_encoder(node_features)
        for block in self.node_blocks:
            node_state = block(node_state, conditioning_nodes)

        node_context = torch.cat([node_state, global_nodes], dim=-1)
        outputs: list[torch.Tensor] = []
        if self.use_two_stage_rmises:
            assert self.hotspot_decoder is not None
            hotspot_logit = self.hotspot_decoder(node_context)
            stress_input = torch.cat([node_context, hotspot_logit], dim=-1)
            stress_prediction = self.stress_decoder(stress_input)
            outputs.extend([hotspot_logit, stress_prediction])
        else:
            outputs.append(self.stress_decoder(node_context))

        if self.use_peak_relative_stress:
            assert self.peak_decoder is not None
            graph_context = self._pool_graph_state(node_state, global_state, node_graph_index)
            if graph_context.dim() == 1:
                peak_prediction = self.peak_decoder(graph_context.unsqueeze(0)).expand(node_state.size(0), -1)
            else:
                if node_graph_index is None:
                    raise ValueError("Batched peak prediction requires node_graph_index.")
                peak_prediction = self.peak_decoder(graph_context)[node_graph_index]
            outputs.append(peak_prediction)

        return torch.cat(outputs, dim=-1)

