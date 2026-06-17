from __future__ import annotations

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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

