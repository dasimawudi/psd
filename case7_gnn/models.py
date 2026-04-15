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


class EdgeMessagePassingLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        global_dim: int,
        dropout: float,
        conditioning_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conditioning_dim = conditioning_dim
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
        self.message_modulation = (
            FeatureModulation(feature_dim=hidden_dim, conditioning_dim=conditioning_dim)
            if conditioning_dim is not None
            else None
        )
        self.update_modulation = (
            FeatureModulation(feature_dim=hidden_dim, conditioning_dim=conditioning_dim)
            if conditioning_dim is not None
            else None
        )

    def forward(
        self,
        node_state: torch.Tensor,
        edge_index: torch.Tensor,
        edge_state: torch.Tensor,
        global_state: torch.Tensor,
        conditioning_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        src, dst = edge_index
        num_nodes = node_state.size(0)
        num_edges = edge_index.size(1)

        global_edges = global_state.unsqueeze(0).expand(num_edges, -1)
        message_input = torch.cat([node_state[src], edge_state, global_edges], dim=-1)
        messages = self.message_mlp(message_input)
        if self.message_modulation is not None:
            if conditioning_state is None:
                raise ValueError("conditioning_state is required when message conditioning is enabled.")
            messages = self.message_modulation(messages, conditioning_state)

        aggregated = torch.zeros(num_nodes, self.hidden_dim, device=node_state.device, dtype=node_state.dtype)
        aggregated.index_add_(0, dst, messages)

        degree = torch.zeros(num_nodes, 1, device=node_state.device, dtype=node_state.dtype)
        degree.index_add_(0, dst, torch.ones(num_edges, 1, device=node_state.device, dtype=node_state.dtype))
        aggregated = aggregated / degree.clamp_min(1.0)

        global_nodes = global_state.unsqueeze(0).expand(num_nodes, -1)
        update_input = torch.cat([node_state, aggregated, global_nodes], dim=-1)
        delta = self.update_mlp(update_input)
        if self.update_modulation is not None:
            if conditioning_state is None:
                raise ValueError("conditioning_state is required when update conditioning is enabled.")
            delta = self.update_modulation(delta, conditioning_state)
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
        conditioning_dim: int | None = None,
        enable_conditioning: bool = False,
    ) -> None:
        super().__init__()
        self.enable_conditioning = enable_conditioning
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
        if self.enable_conditioning:
            if conditioning_dim is None:
                raise ValueError("conditioning_dim must be provided when conditioning is enabled.")
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
        else:
            self.case_encoder = None
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
                    conditioning_dim=conditioning_dim if enable_conditioning else None,
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        node_state = self.node_encoder(node_features)
        edge_state = self.edge_encoder(edge_features)
        if self.enable_conditioning:
            assert self.case_encoder is not None
            conditioning_state = self.case_encoder(global_features.unsqueeze(0)).squeeze(0)
            global_state = self.global_encoder(conditioning_state.unsqueeze(0)).squeeze(0)
        else:
            conditioning_state = None
            global_state = self.global_encoder(global_features.unsqueeze(0)).squeeze(0)

        for layer in self.layers:
            node_state = layer(node_state, edge_index, edge_state, global_state, conditioning_state)

        pooled_mean = node_state.mean(dim=0)
        pooled_max = node_state.max(dim=0).values
        graph_state = torch.cat([pooled_mean, pooled_max, global_state], dim=-1)
        return node_state, edge_state, graph_state, global_state, conditioning_state


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
        conditioning_dim: int | None = None,
        enable_conditioning: bool = False,
    ) -> None:
        super().__init__()
        self.enable_conditioning = enable_conditioning
        self.encoder = GraphEncoder(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            global_input_dim=global_input_dim,
            hidden_dim=hidden_dim,
            global_dim=global_dim,
            num_layers=num_layers,
            dropout=dropout,
            conditioning_dim=conditioning_dim,
            enable_conditioning=enable_conditioning,
        )
        self.readout = build_mlp(
            input_dim=hidden_dim * 2 + global_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=3,
            dropout=dropout,
            final_activation=False,
        )
        self.graph_modulation = (
            FeatureModulation(feature_dim=hidden_dim * 2 + global_dim, conditioning_dim=conditioning_dim)
            if enable_conditioning and conditioning_dim is not None
            else None
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        _, _, graph_state, _, conditioning_state = self.encoder(
            node_features,
            edge_index,
            edge_features,
            global_features,
        )
        if self.graph_modulation is not None:
            assert conditioning_state is not None
            graph_state = self.graph_modulation(graph_state, conditioning_state)
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
        rmises_refine_layers: int | None = None,
        conditioning_dim: int | None = None,
        enable_conditioning: bool = False,
        use_two_stage_rmises: bool = False,
    ) -> None:
        super().__init__()
        if output_dim != 2:
            raise ValueError("FieldGNN expects exactly two outputs: RTA and RMises.")

        self.enable_conditioning = enable_conditioning
        self.use_two_stage_rmises = use_two_stage_rmises
        self.encoder = GraphEncoder(
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            global_input_dim=global_input_dim,
            hidden_dim=hidden_dim,
            global_dim=global_dim,
            num_layers=num_layers,
            dropout=dropout,
            conditioning_dim=conditioning_dim,
            enable_conditioning=enable_conditioning,
        )
        self.rta_decoder = build_mlp(
            input_dim=hidden_dim + global_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=3,
            dropout=dropout,
            final_activation=False,
        )
        refine_layers = rmises_refine_layers if rmises_refine_layers is not None else max(1, num_layers - 1)
        self.rmises_layers = nn.ModuleList(
            [
                EdgeMessagePassingLayer(
                    hidden_dim=hidden_dim,
                    edge_dim=hidden_dim,
                    global_dim=global_dim,
                    dropout=dropout,
                    conditioning_dim=conditioning_dim if enable_conditioning else None,
                )
                for _ in range(refine_layers)
            ]
        )
        self.rta_context_modulation = (
            FeatureModulation(feature_dim=hidden_dim + global_dim, conditioning_dim=conditioning_dim)
            if enable_conditioning and conditioning_dim is not None
            else None
        )
        if self.use_two_stage_rmises:
            self.hotspot_decoder = build_mlp(
                input_dim=hidden_dim + hidden_dim + global_dim + 1,
                hidden_dim=hidden_dim,
                output_dim=1,
                num_layers=3,
                dropout=dropout,
                final_activation=False,
            )
            self.rmises_decoder = build_mlp(
                input_dim=hidden_dim + hidden_dim + global_dim + 1 + 1,
                hidden_dim=hidden_dim,
                output_dim=1,
                num_layers=3,
                dropout=dropout,
                final_activation=False,
            )
            self.hotspot_context_modulation = (
                FeatureModulation(
                    feature_dim=hidden_dim + hidden_dim + global_dim + 1,
                    conditioning_dim=conditioning_dim,
                )
                if enable_conditioning and conditioning_dim is not None
                else None
            )
            self.rmises_context_modulation = (
                FeatureModulation(
                    feature_dim=hidden_dim + hidden_dim + global_dim + 1 + 1,
                    conditioning_dim=conditioning_dim,
                )
                if enable_conditioning and conditioning_dim is not None
                else None
            )
        else:
            self.hotspot_decoder = None
            self.rmises_decoder = build_mlp(
                input_dim=hidden_dim + hidden_dim + global_dim + 1,
                hidden_dim=hidden_dim,
                output_dim=1,
                num_layers=3,
                dropout=dropout,
                final_activation=False,
            )
            self.hotspot_context_modulation = None
            self.rmises_context_modulation = (
                FeatureModulation(feature_dim=hidden_dim + hidden_dim + global_dim + 1, conditioning_dim=conditioning_dim)
                if enable_conditioning and conditioning_dim is not None
                else None
            )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
        global_features: torch.Tensor,
    ) -> torch.Tensor:
        node_state, edge_state, _, global_state, conditioning_state = self.encoder(
            node_features,
            edge_index,
            edge_features,
            global_features,
        )
        global_nodes = global_state.unsqueeze(0).expand(node_state.size(0), -1)
        shared_context = torch.cat([node_state, global_nodes], dim=-1)
        if self.rta_context_modulation is not None:
            assert conditioning_state is not None
            shared_context = self.rta_context_modulation(shared_context, conditioning_state)
        rta_prediction = self.rta_decoder(shared_context)

        rmises_state = node_state
        for layer in self.rmises_layers:
            rmises_state = layer(rmises_state, edge_index, edge_state, global_state, conditioning_state)

        base_rmises_context = torch.cat([node_state, rmises_state, global_nodes, rta_prediction], dim=-1)
        if self.use_two_stage_rmises:
            hotspot_input = base_rmises_context
            if self.hotspot_context_modulation is not None:
                assert conditioning_state is not None
                hotspot_input = self.hotspot_context_modulation(hotspot_input, conditioning_state)
            assert self.hotspot_decoder is not None
            hotspot_logit = self.hotspot_decoder(hotspot_input)

            rmises_input = torch.cat([base_rmises_context, hotspot_logit], dim=-1)
            if self.rmises_context_modulation is not None:
                assert conditioning_state is not None
                rmises_input = self.rmises_context_modulation(rmises_input, conditioning_state)
            rmises_prediction = self.rmises_decoder(rmises_input)
            return torch.cat([rta_prediction, hotspot_logit, rmises_prediction], dim=-1)

        rmises_input = base_rmises_context
        if self.rmises_context_modulation is not None:
            assert conditioning_state is not None
            rmises_input = self.rmises_context_modulation(rmises_input, conditioning_state)
        rmises_prediction = self.rmises_decoder(rmises_input)
        return torch.cat([rta_prediction, rmises_prediction], dim=-1)
