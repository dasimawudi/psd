from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class StandardScaler:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, tensors: Iterable[torch.Tensor], eps: float = 1e-6) -> "StandardScaler":
        flattened = []
        for tensor in tensors:
            if tensor.dim() == 1:
                flattened.append(tensor.unsqueeze(0))
            else:
                flattened.append(tensor.reshape(-1, tensor.shape[-1]))

        if not flattened:
            raise ValueError("Cannot fit scaler on an empty tensor collection.")

        data = torch.cat(flattened, dim=0)
        mean = data.mean(dim=0)
        std = data.std(dim=0, unbiased=False).clamp_min(eps)
        return cls(mean=mean, std=std)

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std

    def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.std + self.mean

    def to(self, device: torch.device | str) -> "StandardScaler":
        return StandardScaler(mean=self.mean.to(device), std=self.std.to(device))

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"mean": self.mean.detach().cpu(), "std": self.std.detach().cpu()}

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> "StandardScaler":
        return cls(mean=state["mean"], std=state["std"])


def signed_log1p(tensor: torch.Tensor) -> torch.Tensor:
    return torch.sign(tensor) * torch.log1p(torch.abs(tensor))


def signed_expm1(tensor: torch.Tensor) -> torch.Tensor:
    return torch.sign(tensor) * torch.expm1(torch.abs(tensor))


def encode_field_targets(targets: torch.Tensor, clamp_negative_rmises: bool) -> torch.Tensor:
    rta = torch.log1p(targets[:, 0].clamp_min(0.0))
    rmises_raw = targets[:, 1]
    if clamp_negative_rmises:
        rmises = torch.log1p(rmises_raw.clamp_min(0.0))
    else:
        rmises = signed_log1p(rmises_raw)
    return torch.stack([rta, rmises], dim=-1)


def decode_field_targets(encoded: torch.Tensor, clamp_negative_rmises: bool) -> torch.Tensor:
    rta = torch.expm1(encoded[:, 0]).clamp_min(0.0)
    if clamp_negative_rmises:
        rmises = torch.expm1(encoded[:, 1]).clamp_min(0.0)
    else:
        rmises = signed_expm1(encoded[:, 1])
    return torch.stack([rta, rmises], dim=-1)


def metric_field_targets(targets: torch.Tensor, clamp_negative_rmises: bool) -> torch.Tensor:
    rta = targets[:, 0].clamp_min(0.0)
    rmises = targets[:, 1].clamp_min(0.0) if clamp_negative_rmises else targets[:, 1]
    return torch.stack([rta, rmises], dim=-1)
