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
        output_mode: str = "regression",
        zero_gate: dict[str, Any] | None = None,
        peak_relative: dict[str, Any] | None = None,
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
        self.output_mode = output_mode.strip().lower()
        self.zero_gate_enabled = bool((zero_gate or {}).get("enabled", False))
        self.peak_relative_enabled = bool((peak_relative or {}).get("enabled", False))
        if self.output_mode in {"auto", ""}:
            if self.peak_relative_enabled:
                self.output_mode = "peak_relative"
            elif self.zero_gate_enabled:
                self.output_mode = "zero_gate"
            else:
                self.output_mode = "regression"
        if self.output_mode not in {"regression", "zero_gate", "peak_relative"}:
            raise ValueError(f"Unsupported PointMLP output_mode: {output_mode}")

        self.zero_logit_head = nn.Linear(previous_dim, 1) if self.zero_gate_enabled else None
        self.peak_log_head = nn.Linear(previous_dim, 1) if self.peak_relative_enabled else None
        self.drop_head = nn.Linear(previous_dim, 1) if self.peak_relative_enabled else None

    def forward(self, features: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        embedding = features
        for layer in self.network[:-1]:
            embedding = layer(embedding)
        stress_scaled = self.network[-1](embedding)
        outputs: dict[str, torch.Tensor] = {"stress_scaled": stress_scaled}
        if self.zero_logit_head is not None:
            outputs["nonzero_logit"] = self.zero_logit_head(embedding)
        if self.peak_log_head is not None and self.drop_head is not None:
            outputs["peak_log_scaled"] = self.peak_log_head(embedding)
            outputs["drop_log"] = self.drop_head(embedding)
        if self.output_mode == "regression":
            return stress_scaled
        return outputs
