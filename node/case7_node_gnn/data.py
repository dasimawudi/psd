from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from case7_node_mlp.data import load_raw_point_sample
from case7_node_mlp.scalers import StandardScaler
from case7_node_mlp.trainer import PreparedPointSample, prepare_point_sample


@dataclass
class GraphSample:
    name: str
    case_name: str
    frequency_hz: float
    features: torch.Tensor
    target_scaled: torch.Tensor
    target_log: torch.Tensor
    target_raw: torch.Tensor
    node_indices: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    point_weights: torch.Tensor

    @property
    def num_points(self) -> int:
        return int(self.target_raw.numel())

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.size(1))


@dataclass
class GraphBatch:
    names: list[str]
    case_names: list[str]
    frequency_hz: torch.Tensor
    features: torch.Tensor
    target_scaled: torch.Tensor
    target_log: torch.Tensor
    target_raw: torch.Tensor
    node_indices: torch.Tensor
    graph_index: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    point_weights: torch.Tensor

    @property
    def num_points(self) -> int:
        return int(self.target_raw.numel())

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.size(1))

    def to(self, device: torch.device) -> "GraphBatch":
        return GraphBatch(
            names=self.names,
            case_names=self.case_names,
            frequency_hz=self.frequency_hz.to(device),
            features=self.features.to(device),
            target_scaled=self.target_scaled.to(device),
            target_log=self.target_log.to(device),
            target_raw=self.target_raw.to(device),
            node_indices=self.node_indices.to(device),
            graph_index=self.graph_index.to(device),
            edge_index=self.edge_index.to(device),
            edge_attr=self.edge_attr.to(device),
            point_weights=self.point_weights.to(device),
        )


def _edge_feature_columns(edge_cfg: dict[str, Any]) -> tuple[str, ...]:
    columns = edge_cfg.get("edge_attr", ["dx", "dy", "dz", "dist"])
    return tuple(str(column) for column in columns)


@lru_cache(maxsize=128)
def _load_case_edges_cached(case_dir_str: str, edge_columns: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    case_dir = Path(case_dir_str)
    edge_path = case_dir / "edges.csv"
    if not edge_path.exists():
        raise FileNotFoundError(f"Graph edge file does not exist: {edge_path}")

    required = {"src", "dst", *edge_columns}
    header = pd.read_csv(edge_path, nrows=0)
    missing = sorted(required.difference(header.columns))
    if missing:
        raise KeyError(f"{edge_path} is missing edge columns: {missing}")

    edge_df = pd.read_csv(edge_path, usecols=["src", "dst", *edge_columns])
    src = edge_df["src"].to_numpy(dtype=np.int64, copy=False)
    dst = edge_df["dst"].to_numpy(dtype=np.int64, copy=False)
    edge_attr = edge_df[list(edge_columns)].to_numpy(dtype=np.float32, copy=False)
    return src, dst, edge_attr


@lru_cache(maxsize=128)
def _load_case_node_ids_cached(case_dir_str: str) -> np.ndarray:
    case_dir = Path(case_dir_str)
    nodes_df = pd.read_csv(case_dir / "nodes.csv", usecols=["node_index"])
    return nodes_df["node_index"].to_numpy(dtype=np.int64, copy=False)


def _standardize_edge_attr(edge_attr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if edge_attr.size == 0:
        return edge_attr
    mean = edge_attr.mean(axis=0, keepdims=True)
    std = np.maximum(edge_attr.std(axis=0, keepdims=True), eps)
    return (edge_attr - mean) / std


def _build_subgraph_edges(
    case_dir: Path,
    node_indices: torch.Tensor,
    edge_cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    node_count = int(node_indices.numel())
    edge_columns = _edge_feature_columns(edge_cfg)
    edge_attr_dim = len(edge_columns)
    if node_count <= 0:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0, edge_attr_dim), dtype=torch.float32)

    case_dir_str = str(case_dir.resolve())
    src, dst, attr = _load_case_edges_cached(case_dir_str, edge_columns)
    node_ids = _load_case_node_ids_cached(case_dir_str)
    selected_rows = node_indices.cpu().numpy().astype(np.int64, copy=False)
    selected = node_ids[selected_rows]
    selected_mask = np.zeros(int(max(src.max(initial=0), dst.max(initial=0), selected.max(initial=0))) + 1, dtype=bool)
    selected_mask[selected] = True
    edge_mask = selected_mask[src] & selected_mask[dst]
    src = src[edge_mask]
    dst = dst[edge_mask]
    attr = attr[edge_mask]

    local_lookup = np.full(selected_mask.shape[0], -1, dtype=np.int64)
    local_lookup[selected] = np.arange(node_count, dtype=np.int64)
    local_src = local_lookup[src]
    local_dst = local_lookup[dst]
    valid = (local_src >= 0) & (local_dst >= 0)
    local_src = local_src[valid]
    local_dst = local_dst[valid]
    attr = attr[valid]

    add_reverse = bool(edge_cfg.get("add_reverse", True))
    add_self_loops = bool(edge_cfg.get("add_self_loops", True))
    if add_reverse and local_src.size:
        reverse_attr = attr.copy()
        for column_index, column_name in enumerate(edge_columns):
            if column_name in {"dx", "dy", "dz"}:
                reverse_attr[:, column_index] *= -1.0
        original_src = local_src
        original_dst = local_dst
        local_src = np.concatenate([local_src, local_dst], axis=0)
        local_dst = np.concatenate([original_dst, original_src], axis=0)
        attr = np.concatenate([attr, reverse_attr], axis=0)

    if bool(edge_cfg.get("standardize_per_graph", True)) and attr.size:
        attr = _standardize_edge_attr(attr)

    if add_self_loops:
        loop_nodes = np.arange(node_count, dtype=np.int64)
        loop_attr = np.zeros((node_count, edge_attr_dim), dtype=np.float32)
        local_src = np.concatenate([local_src, loop_nodes], axis=0)
        local_dst = np.concatenate([local_dst, loop_nodes], axis=0)
        attr = np.concatenate([attr, loop_attr], axis=0)

    edge_index = torch.tensor(np.stack([local_src, local_dst], axis=0), dtype=torch.long)
    edge_attr = torch.tensor(attr, dtype=torch.float32)
    return edge_index, edge_attr


def graph_from_prepared_sample(
    sample_path: str | Path,
    prepared: PreparedPointSample,
    edge_cfg: dict[str, Any],
) -> GraphSample:
    target_path = Path(sample_path)
    case_dir = target_path.parent.parent
    edge_index, edge_attr = _build_subgraph_edges(case_dir, prepared.node_indices, edge_cfg)
    return GraphSample(
        name=prepared.name,
        case_name=prepared.case_name,
        frequency_hz=prepared.frequency_hz,
        features=prepared.features,
        target_scaled=prepared.target_scaled,
        target_log=prepared.target_log,
        target_raw=prepared.target_raw,
        node_indices=prepared.node_indices,
        edge_index=edge_index,
        edge_attr=edge_attr,
        point_weights=prepared.point_weights,
    )


class GraphSampleDataset(Dataset[GraphSample]):
    def __init__(
        self,
        sample_paths: list[Path],
        dataset_cfg: dict[str, Any],
        feature_cfg: dict[str, Any],
        graph_cfg: dict[str, Any],
        x_scaler: StandardScaler,
        y_scaler: StandardScaler,
        feature_schema: dict[str, Any],
        target_cfg: dict[str, Any],
        loss_cfg: dict[str, Any],
    ) -> None:
        self.sample_paths = list(sample_paths)
        self.dataset_cfg = dict(dataset_cfg)
        self.feature_cfg = dict(feature_cfg)
        self.graph_cfg = dict(graph_cfg)
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.feature_schema = dict(feature_schema)
        self.target_cfg = dict(target_cfg)
        self.loss_cfg = dict(loss_cfg)

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, index: int) -> GraphSample:
        sample_path = self.sample_paths[index]
        raw = load_raw_point_sample(
            sample_path,
            dataset_cfg=self.dataset_cfg,
            feature_cfg=self.feature_cfg,
        )
        prepared = prepare_point_sample(
            raw,
            x_scaler=self.x_scaler,
            y_scaler=self.y_scaler,
            feature_schema=self.feature_schema,
            target_cfg=self.target_cfg,
            loss_cfg=self.loss_cfg,
        )
        return graph_from_prepared_sample(sample_path, prepared, edge_cfg=self.graph_cfg)


def collate_graph_samples(samples: list[GraphSample]) -> GraphBatch:
    if not samples:
        raise ValueError("Cannot collate an empty graph sample list.")

    names: list[str] = []
    case_names: list[str] = []
    frequencies: list[float] = []
    features: list[torch.Tensor] = []
    target_scaled: list[torch.Tensor] = []
    target_log: list[torch.Tensor] = []
    target_raw: list[torch.Tensor] = []
    node_indices: list[torch.Tensor] = []
    graph_indices: list[torch.Tensor] = []
    edge_indices: list[torch.Tensor] = []
    edge_attrs: list[torch.Tensor] = []
    point_weights: list[torch.Tensor] = []
    node_offset = 0

    for graph_idx, sample in enumerate(samples):
        names.append(sample.name)
        case_names.append(sample.case_name)
        frequencies.append(float(sample.frequency_hz))
        features.append(sample.features)
        target_scaled.append(sample.target_scaled)
        target_log.append(sample.target_log)
        target_raw.append(sample.target_raw)
        node_indices.append(sample.node_indices)
        point_weights.append(sample.point_weights)
        graph_indices.append(torch.full((sample.num_points,), graph_idx, dtype=torch.long))
        edge_indices.append(sample.edge_index + node_offset)
        edge_attrs.append(sample.edge_attr)
        node_offset += sample.num_points

    return GraphBatch(
        names=names,
        case_names=case_names,
        frequency_hz=torch.tensor(frequencies, dtype=torch.float32),
        features=torch.cat(features, dim=0),
        target_scaled=torch.cat(target_scaled, dim=0),
        target_log=torch.cat(target_log, dim=0),
        target_raw=torch.cat(target_raw, dim=0),
        node_indices=torch.cat(node_indices, dim=0),
        graph_index=torch.cat(graph_indices, dim=0),
        edge_index=torch.cat(edge_indices, dim=1),
        edge_attr=torch.cat(edge_attrs, dim=0),
        point_weights=torch.cat(point_weights, dim=0),
    )


def make_graph_loader(
    sample_paths: list[Path],
    dataset_cfg: dict[str, Any],
    feature_cfg: dict[str, Any],
    graph_cfg: dict[str, Any],
    x_scaler: StandardScaler,
    y_scaler: StandardScaler,
    feature_schema: dict[str, Any],
    target_cfg: dict[str, Any],
    loss_cfg: dict[str, Any],
    graph_batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader[GraphBatch]:
    dataset = GraphSampleDataset(
        sample_paths=sample_paths,
        dataset_cfg=dataset_cfg,
        feature_cfg=feature_cfg,
        graph_cfg=graph_cfg,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        feature_schema=feature_schema,
        target_cfg=target_cfg,
        loss_cfg=loss_cfg,
    )
    workers = max(0, int(num_workers))
    return DataLoader(
        dataset,
        batch_size=max(1, int(graph_batch_size)),
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=collate_graph_samples,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )
