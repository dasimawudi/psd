from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import json
import math
import random

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


def discover_case_index(root: str | Path) -> dict[str, Path]:
    return {case_dir.name: case_dir for case_dir in discover_complete_cases(root)}


def _normalize_case_names(names: Iterable[str] | None) -> list[str]:
    if names is None:
        return []
    return list(dict.fromkeys(str(name) for name in names))


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


def _sanitize_cache_part(value: str) -> str:
    sanitized = "".join(ch if ch.isalnum() else "-" for ch in value)
    sanitized = sanitized.strip("-")
    return sanitized or "default"


def _cache_signature(
    node_columns: Sequence[str],
    edge_columns: Sequence[str],
    target_freq_key: str,
    make_undirected: bool,
) -> str:
    node_part = "_".join(_sanitize_cache_part(column) for column in node_columns)
    edge_part = "_".join(_sanitize_cache_part(column) for column in edge_columns)
    freq_part = _sanitize_cache_part(target_freq_key)
    undir_part = "1" if make_undirected else "0"
    return f"nodes-{node_part}__edges-{edge_part}__freq-{freq_part}__undir-{undir_part}"


def _cache_path_for_case(
    cache_dir: str | Path,
    case_dir: Path,
    node_columns: Sequence[str],
    edge_columns: Sequence[str],
    target_freq_key: str,
    make_undirected: bool,
) -> Path:
    signature = _cache_signature(
        node_columns=node_columns,
        edge_columns=edge_columns,
        target_freq_key=target_freq_key,
        make_undirected=make_undirected,
    )
    return Path(cache_dir) / signature / f"{case_dir.name}.pt"


def _save_case_cache(cache_path: Path, case: CaseGraph) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "name": case.name,
            "node_features": case.node_features,
            "edge_index": case.edge_index,
            "edge_features": case.edge_features,
            "params": case.params,
            "psd": case.psd,
            "freq_target": case.freq_target,
            "node_targets": case.node_targets,
        },
        cache_path,
    )


def _load_case_cache(cache_path: Path) -> CaseGraph:
    payload = torch.load(cache_path, map_location="cpu")
    return CaseGraph(
        name=payload["name"],
        node_features=payload["node_features"],
        edge_index=payload["edge_index"],
        edge_features=payload["edge_features"],
        params=payload["params"],
        psd=payload["psd"],
        freq_target=payload["freq_target"],
        node_targets=payload["node_targets"],
    )


def _allocate_split_counts(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    ratios = [float(train_ratio), float(val_ratio), float(test_ratio)]
    if any(ratio < 0.0 for ratio in ratios):
        raise ValueError("Split ratios must be non-negative.")

    ratio_sum = sum(ratios)
    if ratio_sum <= 0.0:
        raise ValueError("At least one split ratio must be positive.")

    normalized = [ratio / ratio_sum for ratio in ratios]
    positive_splits = sum(1 for ratio in normalized if ratio > 0.0)
    if total < positive_splits:
        raise ValueError(
            f"Need at least {positive_splits} cases to satisfy non-empty ratio splits, but found {total}."
        )

    raw_counts = [ratio * total for ratio in normalized]
    counts = [int(math.floor(value)) for value in raw_counts]
    remainder = total - sum(counts)
    fractional_order = sorted(
        range(3),
        key=lambda index: raw_counts[index] - counts[index],
        reverse=True,
    )
    for index in fractional_order[:remainder]:
        counts[index] += 1

    for index, ratio in enumerate(normalized):
        if ratio <= 0.0 or counts[index] > 0:
            continue
        donor = max(
            (candidate for candidate, candidate_count in enumerate(counts) if candidate_count > 1),
            key=lambda candidate: counts[candidate],
            default=None,
        )
        if donor is None:
            raise ValueError("Could not allocate non-empty dataset splits from the available cases.")
        counts[donor] -= 1
        counts[index] += 1

    return counts[0], counts[1], counts[2]


def _validate_splits(splits: dict[str, list[str]], available: dict[str, Path]) -> dict[str, list[str]]:
    validated: dict[str, list[str]] = {}
    seen_names: dict[str, str] = {}

    for split_name, names in splits.items():
        normalized_names = _normalize_case_names(names)
        missing = [name for name in normalized_names if name not in available]
        if missing:
            raise ValueError(f"Requested cases are missing or incomplete for split '{split_name}': {sorted(missing)}")

        for name in normalized_names:
            if name in seen_names:
                raise ValueError(
                    f"Case '{name}' appears in both '{seen_names[name]}' and '{split_name}' splits."
                )
            seen_names[name] = split_name

        validated[split_name] = sorted(normalized_names)

    if not validated.get("train"):
        raise ValueError("Training split is empty.")
    if not validated.get("val"):
        raise ValueError("Validation split is empty.")
    return validated


def resolve_case_splits(root: str | Path, dataset_cfg: dict[str, Any]) -> dict[str, list[str]]:
    available = discover_case_index(root)

    explicit_lists_present = any(
        dataset_cfg.get(key) is not None for key in ("train_cases", "val_cases", "test_cases")
    )
    split_mode = str(dataset_cfg.get("split_mode", "explicit" if explicit_lists_present else "ratio")).lower()

    if split_mode == "explicit":
        splits = {
            "train": _normalize_case_names(dataset_cfg.get("train_cases")),
            "val": _normalize_case_names(dataset_cfg.get("val_cases")),
            "test": _normalize_case_names(dataset_cfg.get("test_cases")),
        }
        return _validate_splits(splits, available)

    if split_mode != "ratio":
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    selected_names = sorted(available)
    include_cases = _normalize_case_names(dataset_cfg.get("include_cases"))
    if include_cases:
        missing = [name for name in include_cases if name not in available]
        if missing:
            raise ValueError(f"Included cases are missing or incomplete: {sorted(missing)}")
        selected_names = include_cases

    exclude_cases = set(_normalize_case_names(dataset_cfg.get("exclude_cases")))
    selected_names = [name for name in selected_names if name not in exclude_cases]

    if not selected_names:
        raise ValueError("No complete cases remain after applying include/exclude filters.")

    shuffled_names = list(selected_names)
    random.Random(int(dataset_cfg.get("split_seed", 42))).shuffle(shuffled_names)

    max_cases = dataset_cfg.get("max_cases")
    if max_cases is not None:
        max_cases_int = int(max_cases)
        if max_cases_int <= 0:
            raise ValueError("max_cases must be positive when provided.")
        shuffled_names = shuffled_names[:max_cases_int]

    train_count, val_count, test_count = _allocate_split_counts(
        len(shuffled_names),
        train_ratio=float(dataset_cfg.get("train_ratio", 0.8)),
        val_ratio=float(dataset_cfg.get("val_ratio", 0.1)),
        test_ratio=float(dataset_cfg.get("test_ratio", 0.1)),
    )

    train_end = train_count
    val_end = train_end + val_count
    splits = {
        "train": sorted(shuffled_names[:train_end]),
        "val": sorted(shuffled_names[train_end:val_end]),
        "test": sorted(shuffled_names[val_end : val_end + test_count]),
    }
    return _validate_splits(splits, available)


def load_case_graph(
    case_dir: str | Path,
    node_columns: Sequence[str] = DEFAULT_NODE_COLUMNS,
    edge_columns: Sequence[str] = DEFAULT_EDGE_COLUMNS,
    target_freq_key: str = "freq_top3",
    make_undirected: bool = True,
    cache_dir: str | Path | None = None,
) -> CaseGraph:
    case_path = Path(case_dir)

    if cache_dir is not None:
        cache_path = _cache_path_for_case(
            cache_dir=cache_dir,
            case_dir=case_path,
            node_columns=node_columns,
            edge_columns=edge_columns,
            target_freq_key=target_freq_key,
            make_undirected=make_undirected,
        )
        if cache_path.exists():
            return _load_case_cache(cache_path)

    params, psd, freq_target = _load_global_json(case_path, target_freq_key=target_freq_key)
    node_features, node_targets = _load_nodes(case_path, node_columns=node_columns)
    edge_index, edge_features = _load_edges(
        case_path,
        edge_columns=edge_columns,
        make_undirected=make_undirected,
    )

    case = CaseGraph(
        name=case_path.name,
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        params=params,
        psd=psd,
        freq_target=freq_target,
        node_targets=node_targets,
    )

    if cache_dir is not None:
        _save_case_cache(cache_path, case)

    return case


def load_selected_cases(
    root: str | Path,
    selected_names: Iterable[str],
    node_columns: Sequence[str] = DEFAULT_NODE_COLUMNS,
    edge_columns: Sequence[str] = DEFAULT_EDGE_COLUMNS,
    target_freq_key: str = "freq_top3",
    make_undirected: bool = True,
    cache_dir: str | Path | None = None,
) -> dict[str, CaseGraph]:
    selected_set = set(selected_names)
    available = discover_case_index(root)

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
            cache_dir=cache_dir,
        )
    return loaded


def build_global_features(case: CaseGraph, use_psd: bool) -> torch.Tensor:
    if use_psd:
        return torch.cat([case.params, case.psd], dim=0)
    return case.params.clone()
