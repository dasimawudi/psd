from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class StandardScaler:
    mean: torch.Tensor
    std: torch.Tensor

    @staticmethod
    def _flatten_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 1:
            return tensor.unsqueeze(0)
        return tensor.reshape(-1, tensor.shape[-1])

    @classmethod
    def fit(cls, tensors: Iterable[torch.Tensor], eps: float = 1e-6) -> "StandardScaler":
        stats = RunningTensorStats()
        for tensor in tensors:
            stats.update(tensor)
        return stats.finalize(eps=eps)

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


@dataclass
class RunningTensorStats:
    count: int = 0
    total_sum: torch.Tensor | None = None
    total_sum_sq: torch.Tensor | None = None

    def update(self, tensor: torch.Tensor) -> None:
        matrix = StandardScaler._flatten_tensor(tensor).to(dtype=torch.float64)

        if self.total_sum is None or self.total_sum_sq is None:
            feature_dim = int(matrix.size(-1))
            self.total_sum = torch.zeros(feature_dim, dtype=torch.float64)
            self.total_sum_sq = torch.zeros(feature_dim, dtype=torch.float64)
        elif matrix.size(-1) != self.total_sum.size(0):
            raise ValueError("Cannot accumulate tensors with inconsistent feature dimensions.")

        self.total_sum += matrix.sum(dim=0)
        self.total_sum_sq += matrix.pow(2).sum(dim=0)
        self.count += int(matrix.size(0))

    def finalize(self, eps: float = 1e-6) -> StandardScaler:
        if self.count == 0 or self.total_sum is None or self.total_sum_sq is None:
            raise ValueError("Cannot fit scaler on an empty tensor collection.")

        mean = self.total_sum / self.count
        variance = (self.total_sum_sq / self.count) - mean.pow(2)
        std = variance.clamp_min(0.0).sqrt().clamp_min(eps)
        return StandardScaler(mean=mean.to(dtype=torch.float32), std=std.to(dtype=torch.float32))


def signed_log1p(tensor: torch.Tensor) -> torch.Tensor:
    return torch.sign(tensor) * torch.log1p(torch.abs(tensor))


def signed_expm1(tensor: torch.Tensor) -> torch.Tensor:
    return torch.sign(tensor) * torch.expm1(torch.abs(tensor))


def metric_rmises_targets(targets: torch.Tensor, clamp_negative_rmises: bool) -> torch.Tensor:
    rmises = targets[:, 1]
    if clamp_negative_rmises:
        rmises = rmises.clamp_min(0.0)
    return rmises


def encode_rmises_targets(
    rmises_targets: torch.Tensor,
    clamp_negative_rmises: bool,
    as_excess: bool = False,
    threshold: float = 0.0,
) -> torch.Tensor:
    if as_excess:
        rmises_metric = (rmises_targets - threshold).clamp_min(0.0)
        return torch.log1p(rmises_metric)

    if clamp_negative_rmises:
        return torch.log1p(rmises_targets.clamp_min(0.0))
    return signed_log1p(rmises_targets)


def decode_rmises_targets(
    encoded_rmises: torch.Tensor,
    clamp_negative_rmises: bool,
    as_excess: bool = False,
    threshold: float = 0.0,
) -> torch.Tensor:
    if as_excess:
        return torch.expm1(encoded_rmises).clamp_min(0.0)

    if clamp_negative_rmises:
        return torch.expm1(encoded_rmises).clamp_min(0.0)
    return signed_expm1(encoded_rmises)


def build_rmises_hotspot_targets(rmises_targets: torch.Tensor, threshold: float) -> torch.Tensor:
    return (rmises_targets >= threshold).to(dtype=torch.float32)


def encode_field_targets(
    targets: torch.Tensor,
    clamp_negative_rmises: bool,
    rmises_as_excess: bool = False,
    rmises_threshold: float = 0.0,
) -> torch.Tensor:
    rta = torch.log1p(targets[:, 0].clamp_min(0.0))
    rmises = encode_rmises_targets(
        metric_rmises_targets(targets, clamp_negative_rmises=clamp_negative_rmises),
        clamp_negative_rmises=clamp_negative_rmises,
        as_excess=rmises_as_excess,
        threshold=rmises_threshold,
    )
    return torch.stack([rta, rmises], dim=-1)


def decode_field_targets(
    encoded: torch.Tensor,
    clamp_negative_rmises: bool,
    rmises_as_excess: bool = False,
    rmises_threshold: float = 0.0,
) -> torch.Tensor:
    rta = torch.expm1(encoded[:, 0]).clamp_min(0.0)
    rmises = decode_rmises_targets(
        encoded[:, 1],
        clamp_negative_rmises=clamp_negative_rmises,
        as_excess=rmises_as_excess,
        threshold=rmises_threshold,
    )
    return torch.stack([rta, rmises], dim=-1)


def metric_field_targets(targets: torch.Tensor, clamp_negative_rmises: bool) -> torch.Tensor:
    rta = targets[:, 0].clamp_min(0.0)
    rmises = metric_rmises_targets(targets, clamp_negative_rmises=clamp_negative_rmises)
    return torch.stack([rta, rmises], dim=-1)
