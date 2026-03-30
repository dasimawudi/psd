from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import json

import numpy as np
import pandas as pd
import torch


DEFAULT_NODE_COLUMNS = ("x", "y", "z", "bc_mask")
DEFAULT_EDGE_COLUMNS = ("dx", "dy", "dz", "dist")
DEFAULT_NODE_TARGET_COLUMNS = ("RTA", "RMises")


@dataclass
class CaseGraph:
    name: str
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    params: torch.Tensor
    psd: torch.Tensor
    freq_target: torch.Tensor
    node_targets: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.node_features.size(0))

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.size(1))


def discover_complete_cases(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_path}")

    case_dirs: list[Path] = []
    for case_dir in sorted(root_path.iterdir()):
        if not case_dir.is_dir():
            continue
        if (
            (case_dir / "nodes.csv").exists()
            and (case_dir / "edges.csv").exists()
            and (case_dir / "global.json").exists()
        ):
            case_dirs.append(case_dir)
    return case_dirs


def _flatten_psd_points(psd_points: Sequence[Sequence[float]]) -> list[float]:
    flattened: list[float] = []
    for triple in psd_points:
        flattened.extend(float(value) for value in triple)
    return flattened


def _load_global_json(case_dir: Path, target_freq_key: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    payload = json.loads((case_dir / "global.json").read_text(encoding="utf-8"))

    if "params_list" not in payload:
        raise KeyError(f"{case_dir} is missing params_list")
    if "psd_points" not in payload:
        raise KeyError(f"{case_dir} is missing psd_points")
    if target_freq_key not in payload:
        raise KeyError(f"{case_dir} is missing {target_freq_key}")

    params = torch.tensor(payload["params_list"], dtype=torch.float32)
    psd = torch.tensor(_flatten_psd_points(payload["psd_points"]), dtype=torch.float32)
    freq_target = torch.tensor(payload[target_freq_key], dtype=torch.float32)
    return params, psd, freq_target


def _load_nodes(case_dir: Path, node_columns: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
    usecols = list(node_columns) + list(DEFAULT_NODE_TARGET_COLUMNS)
    nodes_df = pd.read_csv(case_dir / "nodes.csv", usecols=usecols)
    node_features = torch.tensor(nodes_df[list(node_columns)].to_numpy(dtype=np.float32), dtype=torch.float32)
    node_targets = torch.tensor(
        nodes_df[list(DEFAULT_NODE_TARGET_COLUMNS)].to_numpy(dtype=np.float32),
        dtype=torch.float32,
    )
    return node_features, node_targets


def _load_edges(case_dir: Path, edge_columns: Sequence[str], make_undirected: bool) -> tuple[torch.Tensor, torch.Tensor]:
    usecols = ["src", "dst"] + list(edge_columns)
    edges_df = pd.read_csv(case_dir / "edges.csv", usecols=usecols)

    src = edges_df["src"].to_numpy(dtype=np.int64)
    dst = edges_df["dst"].to_numpy(dtype=np.int64)
    edge_attr = edges_df[list(edge_columns)].to_numpy(dtype=np.float32)

    if make_undirected:
        reverse_attr = edge_attr.copy()
        for axis in range(min(3, reverse_attr.shape[1])):
            reverse_attr[:, axis] *= -1.0

        src_all = np.concatenate([src, dst], axis=0)
        dst_all = np.concatenate([dst, src], axis=0)
        edge_all = np.concatenate([edge_attr, reverse_attr], axis=0)
    else:
        src_all = src
        dst_all = dst
        edge_all = edge_attr

    edge_index = torch.tensor(np.stack([src_all, dst_all], axis=0), dtype=torch.long)
    edge_features = torch.tensor(edge_all, dtype=torch.float32)
    return edge_index, edge_features


def load_case_graph(
    case_dir: str | Path,
    node_columns: Sequence[str] = DEFAULT_NODE_COLUMNS,
    edge_columns: Sequence[str] = DEFAULT_EDGE_COLUMNS,
    target_freq_key: str = "freq_top3",
    make_undirected: bool = True,
) -> CaseGraph:
    case_path = Path(case_dir)
    params, psd, freq_target = _load_global_json(case_path, target_freq_key=target_freq_key)
    node_features, node_targets = _load_nodes(case_path, node_columns=node_columns)
    edge_index, edge_features = _load_edges(
        case_path,
        edge_columns=edge_columns,
        make_undirected=make_undirected,
    )

    return CaseGraph(
        name=case_path.name,
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        params=params,
        psd=psd,
        freq_target=freq_target,
        node_targets=node_targets,
    )


def load_selected_cases(
    root: str | Path,
    selected_names: Iterable[str],
    node_columns: Sequence[str] = DEFAULT_NODE_COLUMNS,
    edge_columns: Sequence[str] = DEFAULT_EDGE_COLUMNS,
    target_freq_key: str = "freq_top3",
    make_undirected: bool = True,
) -> dict[str, CaseGraph]:
    selected_set = set(selected_names)
    available = {path.name: path for path in discover_complete_cases(root)}

    missing = selected_set.difference(available)
    if missing:
        raise ValueError(f"Requested cases are missing or incomplete: {sorted(missing)}")

    loaded: dict[str, CaseGraph] = {}
    for name in sorted(selected_set):
        loaded[name] = load_case_graph(
            available[name],
            node_columns=node_columns,
            edge_columns=edge_columns,
            target_freq_key=target_freq_key,
            make_undirected=make_undirected,
        )
    return loaded


def build_global_features(case: CaseGraph, use_psd: bool) -> torch.Tensor:
    if use_psd:
        return torch.cat([case.params, case.psd], dim=0)
    return case.params.clone()
