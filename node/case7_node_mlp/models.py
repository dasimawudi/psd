from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    activation_name = name.strip().lower()
    if activation_name == "relu":
        return nn.ReLU()
    if activation_name == "silu":
        return nn.SiLU()
    if activation_name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


class PointMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float = 0.0,
        activation: str = "silu",
        use_layer_norm: bool = True,
        low_rank_curve_head: bool = False,
        curve_rank: int = 3,
        frequency_feature_indices: list[int] | None = None,
        node_feature_indices: list[int] | None = None,
        residual_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must not be empty.")

        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, int(hidden_dim)))
            if use_layer_norm:
                layers.append(nn.LayerNorm(int(hidden_dim)))
            layers.append(_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            previous_dim = int(hidden_dim)
        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)
        self.low_rank_curve_head = bool(low_rank_curve_head)
        self.curve_rank = max(1, int(curve_rank))
        self.residual_weight = float(residual_weight)
        self.frequency_feature_indices = list(frequency_feature_indices or [])
        self.node_feature_indices = list(node_feature_indices or [])
        if self.low_rank_curve_head:
            if not self.frequency_feature_indices:
                raise ValueError("frequency_feature_indices must not be empty when low_rank_curve_head is enabled.")
            if not self.node_feature_indices:
                raise ValueError("node_feature_indices must not be empty when low_rank_curve_head is enabled.")
            freq_hidden = max(32, min(128, int(hidden_dims[0])))
            node_hidden = max(32, min(128, int(hidden_dims[0])))
            self.frequency_head = _make_mlp(
                input_dim=len(self.frequency_feature_indices),
                hidden_dim=freq_hidden,
                output_dim=self.curve_rank,
                dropout=dropout,
                activation=activation,
                use_layer_norm=use_layer_norm,
            )
            self.node_mixing_head = _make_mlp(
                input_dim=len(self.node_feature_indices),
                hidden_dim=node_hidden,
                output_dim=self.curve_rank,
                dropout=dropout,
                activation=activation,
                use_layer_norm=use_layer_norm,
            )
            self.node_scale_head = _make_mlp(
                input_dim=len(self.node_feature_indices),
                hidden_dim=node_hidden,
                output_dim=1,
                dropout=dropout,
                activation=activation,
                use_layer_norm=use_layer_norm,
            )
        else:
            self.frequency_head = None
            self.node_mixing_head = None
            self.node_scale_head = None

    def forward(self, features: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        pointwise = self.network(features)
        if not self.low_rank_curve_head:
            return pointwise
        frequency_features = features[:, self.frequency_feature_indices]
        node_features = features[:, self.node_feature_indices]
        curve_values = self.frequency_head(frequency_features)
        mixing_logits = self.node_mixing_head(node_features)
        mixing_weights = torch.softmax(mixing_logits, dim=-1)
        node_scale = self.node_scale_head(node_features)
        low_rank = node_scale + (mixing_weights * curve_values).sum(dim=-1, keepdim=True)
        regression = low_rank + self.residual_weight * pointwise
        return {
            "regression": regression,
            "pointwise": pointwise,
            "low_rank": low_rank,
            "node_scale": node_scale,
            "curve_values": curve_values,
            "mixing_weights": mixing_weights,
        }


def _make_mlp(
    *,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    dropout: float,
    activation: str,
    use_layer_norm: bool,
) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(int(input_dim), int(hidden_dim))]
    if use_layer_norm:
        layers.append(nn.LayerNorm(int(hidden_dim)))
    layers.append(_activation(activation))
    if dropout > 0.0:
        layers.append(nn.Dropout(float(dropout)))
    layers.append(nn.Linear(int(hidden_dim), int(output_dim)))
    return nn.Sequential(*layers)


def regression_output(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(output, dict):
        return output["regression"]
    return output


def aux_output(output: torch.Tensor | dict[str, torch.Tensor], name: str) -> torch.Tensor | None:
    if isinstance(output, dict):
        value: Any = output.get(name)
        return value if isinstance(value, torch.Tensor) else None
    return None
