from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence
from uuid import uuid4

import json
import math
import random
import re
import warnings

import numpy as np
import pandas as pd
import torch


DEFAULT_NODE_COLUMNS = ("x", "y", "z", "bc_mask")
DEFAULT_EDGE_COLUMNS = ("dx", "dy", "dz", "dist")
DEFAULT_NODE_TARGET_COLUMNS = ("RMises",)
PER_FREQUENCY_DIRNAME = "per_frequency_mises"
PER_FREQUENCY_TARGET_COLUMN = "MISES_psd_density"
FINAL_RMISES_FILENAME = "final_rmises.csv"
FINAL_RMISES_COLUMN = "RMises_native"
FRAME_FREQUENCY_PATTERN = re.compile(r"_(\d+(?:\.\d+)?)Hz$")
MODE_SHAPES_DIRNAME = "mode_shapes"
MODAL_FREQUENCIES_FILENAME = "modal_frequencies.csv"
MODE_SHAPE_PATTERN = re.compile(r"mode_(\d+)_(\d+(?:\.\d+)?)Hz$", re.IGNORECASE)
DEFAULT_MODE_SHAPE_COLUMNS = ("U1", "U2", "U3", "U_mag")
MODAL_TABLE_COLUMN_ALIASES = {
    "mode_index": (
        "mode_index",
        "mode",
        "mode_id",
        "mode_no",
        "mode_number",
        "modal_index",
        "modal_no",
        "frame_index",
        "eigenmode",
    ),
    "frequency_hz": (
        "frequency_hz",
        "freq_hz",
        "frequency",
        "frequency_hz.",
        "frequency (hz)",
        "frequency [hz]",
        "freq",
        "freq.",
        "hz",
        "eigenfrequency",
        "eigenfrequency_hz",
        "eigenfrequency (hz)",
    ),
    "file": (
        "file",
        "filename",
        "file_name",
        "path",
        "relative_path",
        "mode_file",
        "mode_shape_file",
        "csv",
        "csv_file",
    ),
}
NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

DEFAULT_FIXED_GEOMETRY = {
    "plate_thickness": 15.0,
    "plate_HoleRadius": 4.0,
    "plate_HoleDist": 25.0,
    "plate_HoleCount": 4,
    "earpiece_HoleRadius": 4.0,
    "earpiece_Count_default": 3,
    "mass_couple_radius": 65.0,
}


@dataclass
class CaseGraph:
    name: str
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    params: torch.Tensor
    psd: torch.Tensor
    freq_target: torch.Tensor
    frequency_scalar: torch.Tensor
    fixed_geometry: dict[str, float]
    node_targets: torch.Tensor
    modal_frequencies: torch.Tensor | None = None
    mode_shapes: torch.Tensor | None = None
    mode_shape_columns: tuple[str, ...] = DEFAULT_MODE_SHAPE_COLUMNS
    source_node_indices: torch.Tensor | None = None

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


def _load_global_payload(case_dir: Path) -> dict[str, Any]:
    return json.loads((case_dir / "global.json").read_text(encoding="utf-8"))


def _load_fixed_geometry(payload: dict[str, Any]) -> dict[str, float]:
    fixed_geometry = dict(DEFAULT_FIXED_GEOMETRY)
    raw_fixed_geometry = payload.get("fixed_geometry", {})
    for key, default_value in DEFAULT_FIXED_GEOMETRY.items():
        fixed_geometry[key] = float(raw_fixed_geometry.get(key, default_value))
    return fixed_geometry


def _load_global_json(
    payload: dict[str, Any],
    case_dir: Path,
    target_freq_key: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def _extract_param_value(payload: dict[str, Any], key: str, fallback_index: int | None = None) -> float | None:
    params = payload.get("params", {})
    if key in params:
        return float(params[key])
    params_list = payload.get("params_list", [])
    if fallback_index is not None and fallback_index < len(params_list):
        return float(params_list[fallback_index])
    return None


def _generate_boundary_mask(nodes_df: pd.DataFrame, payload: dict[str, Any]) -> np.ndarray:
    fixed_geometry = payload.get("fixed_geometry", {})
    earpiece_count = int(fixed_geometry.get("earpiece_Count_default", 0))
    earpiece_hole_radius = float(fixed_geometry.get("earpiece_HoleRadius", 0.0))
    plate_thickness = float(fixed_geometry.get("plate_thickness", 0.0))
    earpiece_radial_dist = _extract_param_value(payload, "earpiece_RadialDist", fallback_index=1)

    if earpiece_count <= 0 or earpiece_hole_radius <= 0.0 or earpiece_radial_dist is None:
        return np.zeros(len(nodes_df), dtype=np.float32)

    xy = nodes_df[["x", "y"]].to_numpy(dtype=np.float32)
    z = nodes_df["z"].to_numpy(dtype=np.float32)
    angles = np.linspace(0.0, 2.0 * math.pi, num=earpiece_count, endpoint=False, dtype=np.float32)
    centers = np.stack(
        [
            -float(earpiece_radial_dist) * np.sin(angles),
            float(earpiece_radial_dist) * np.cos(angles),
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    xy_delta = xy[:, None, :] - centers[None, :, :]
    radial_distance = np.sqrt(np.sum(np.square(xy_delta), axis=-1))
    hole_radius = np.float32(earpiece_hole_radius * 1.1)
    in_hole = radial_distance <= hole_radius
    in_z_range = (z >= -1.0) & (z <= (plate_thickness + 1.0))
    return (in_hole.any(axis=1) & in_z_range).astype(np.float32, copy=False)


def _load_aligned_target_column(
    target_df: pd.DataFrame,
    target_column: str,
    nodes_df: pd.DataFrame,
    target_path: Path,
) -> np.ndarray:
    if target_column not in target_df.columns:
        raise KeyError(f"{target_path} is missing {target_column}")

    if "node_index" in nodes_df.columns and "node_index" in target_df.columns:
        merged = nodes_df[["node_index"]].merge(
            target_df[["node_index", target_column]],
            on="node_index",
            how="left",
            sort=False,
        )
        if merged[target_column].isna().any():
            missing_count = int(merged[target_column].isna().sum())
            raise ValueError(f"{target_path} is missing {missing_count} node targets after aligning by node_index")
        return merged[target_column].to_numpy(dtype=np.float32)

    values = target_df[target_column].to_numpy(dtype=np.float32)
    if values.shape[0] != len(nodes_df):
        raise ValueError(f"{target_path} row count {values.shape[0]} does not match nodes.csv row count {len(nodes_df)}")
    return values


def _stack_field_targets(primary: np.ndarray) -> torch.Tensor:
    return torch.tensor(primary[:, None], dtype=torch.float32)


def _load_final_rmises_targets(case_dir: Path, nodes_df: pd.DataFrame) -> torch.Tensor:
    target_path = case_dir / FINAL_RMISES_FILENAME
    target_df = pd.read_csv(target_path)
    rmises = _load_aligned_target_column(target_df, FINAL_RMISES_COLUMN, nodes_df=nodes_df, target_path=target_path)
    return _stack_field_targets(rmises)


def _load_per_frequency_targets(target_path: Path, nodes_df: pd.DataFrame) -> torch.Tensor:
    target_df = pd.read_csv(target_path)
    mises_psd_density = _load_aligned_target_column(
        target_df,
        PER_FREQUENCY_TARGET_COLUMN,
        nodes_df=nodes_df,
        target_path=target_path,
    )
    return _stack_field_targets(mises_psd_density)


def _parse_mode_shape_frequency(mode_path: Path) -> tuple[int, float]:
    match = MODE_SHAPE_PATTERN.search(mode_path.stem)
    if match is None:
        raise ValueError(f"Could not parse modal index/frequency from mode shape file: {mode_path.name}")
    return int(match.group(1)), float(match.group(2))


def _canonical_modal_column(column: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(column).strip().lower())


def _resolve_modal_table_columns(modal_df: pd.DataFrame) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for column in modal_df.columns:
        canonical = _canonical_modal_column(column)
        if canonical and canonical not in lookup:
            lookup[canonical] = str(column)

    resolved: dict[str, str] = {}
    for expected_column, aliases in MODAL_TABLE_COLUMN_ALIASES.items():
        for alias in aliases:
            matched_column = lookup.get(_canonical_modal_column(alias))
            if matched_column is not None:
                resolved[expected_column] = matched_column
                break
    return resolved


def _coerce_mode_index(value: object, modal_table_path: Path) -> int:
    if pd.isna(value):
        raise ValueError(f"{modal_table_path} contains an empty mode index")
    try:
        return int(float(value))
    except (TypeError, ValueError):
        match = NUMBER_PATTERN.search(str(value))
        if match is not None:
            return int(float(match.group(0)))
    raise ValueError(f"{modal_table_path} contains an invalid mode index: {value!r}")


def _coerce_frequency_hz(value: object, modal_table_path: Path) -> float:
    if pd.isna(value):
        raise ValueError(f"{modal_table_path} contains an empty modal frequency")
    try:
        return float(value)
    except (TypeError, ValueError):
        match = NUMBER_PATTERN.search(str(value))
        if match is not None:
            return float(match.group(0))
    raise ValueError(f"{modal_table_path} contains an invalid modal frequency: {value!r}")


def _resolve_mode_shape_file(case_dir: Path, raw_file: object, modal_table_path: Path) -> Path:
    if pd.isna(raw_file):
        raise ValueError(f"{modal_table_path} contains an empty mode shape file path")
    raw_path = Path(str(raw_file).strip().replace("\\", "/"))
    candidates = [raw_path if raw_path.is_absolute() else case_dir / raw_path]
    if raw_path.name:
        candidates.append(case_dir / MODE_SHAPES_DIRNAME / raw_path.name)
    for mode_file in candidates:
        if mode_file.exists():
            return mode_file
    raise FileNotFoundError(f"Mode shape file listed in {modal_table_path} does not exist: {candidates[0]}")


def _discover_mode_shape_entries_from_directory(case_dir: Path) -> list[tuple[int, float, Path]]:
    mode_dir = case_dir / MODE_SHAPES_DIRNAME
    entries: list[tuple[int, float, Path]] = []

    if not mode_dir.exists():
        raise FileNotFoundError(f"Mode shape directory does not exist: {mode_dir}")
    for mode_path in sorted(mode_dir.glob("*.csv")):
        mode_index, frequency_hz = _parse_mode_shape_frequency(mode_path)
        entries.append((mode_index, frequency_hz, mode_path))
    if not entries:
        raise FileNotFoundError(f"No mode shape CSV files found in {mode_dir}")
    return sorted(entries, key=lambda item: item[0])


def _discover_mode_shape_entries_from_table(case_dir: Path, modal_table_path: Path) -> list[tuple[int, float, Path]]:
    modal_df = pd.read_csv(modal_table_path)
    resolved_columns = _resolve_modal_table_columns(modal_df)
    missing = sorted({"mode_index", "frequency_hz", "file"}.difference(resolved_columns))

    entries: list[tuple[int, float, Path]] = []
    if "file" in resolved_columns:
        file_column = resolved_columns["file"]
        for _, row in modal_df.iterrows():
            mode_file = _resolve_mode_shape_file(case_dir, row[file_column], modal_table_path)
            if "mode_index" in resolved_columns and "frequency_hz" in resolved_columns:
                mode_index = _coerce_mode_index(row[resolved_columns["mode_index"]], modal_table_path)
                frequency_hz = _coerce_frequency_hz(row[resolved_columns["frequency_hz"]], modal_table_path)
            else:
                mode_index, frequency_hz = _parse_mode_shape_frequency(mode_file)
            entries.append((mode_index, frequency_hz, mode_file))
        return sorted(entries, key=lambda item: item[0])

    if "mode_index" in resolved_columns and "frequency_hz" in resolved_columns:
        directory_entries = _discover_mode_shape_entries_from_directory(case_dir)
        files_by_mode_index = {mode_index: mode_file for mode_index, _, mode_file in directory_entries}
        for position, row in modal_df.iterrows():
            mode_index = _coerce_mode_index(row[resolved_columns["mode_index"]], modal_table_path)
            frequency_hz = _coerce_frequency_hz(row[resolved_columns["frequency_hz"]], modal_table_path)
            mode_file = files_by_mode_index.get(mode_index)
            if mode_file is None and int(position) < len(directory_entries):
                mode_file = directory_entries[int(position)][2]
            if mode_file is None:
                raise FileNotFoundError(f"Could not match mode {mode_index} in {modal_table_path} to a mode shape CSV")
            entries.append((mode_index, frequency_hz, mode_file))
        return sorted(entries, key=lambda item: item[0])

    raise KeyError(f"{modal_table_path} is missing columns: {missing}")


def _discover_mode_shape_entries(case_dir: Path) -> list[tuple[int, float, Path]]:
    modal_table_path = case_dir / MODAL_FREQUENCIES_FILENAME
    if modal_table_path.exists():
        try:
            return _discover_mode_shape_entries_from_table(case_dir, modal_table_path)
        except KeyError:
            if (case_dir / MODE_SHAPES_DIRNAME).exists():
                return _discover_mode_shape_entries_from_directory(case_dir)
            raise
    return _discover_mode_shape_entries_from_directory(case_dir)


def _normalize_mode_shape_array(values: np.ndarray, columns: tuple[str, ...], normalization: str) -> np.ndarray:
    if normalization == "none":
        return values.astype(np.float32, copy=False)

    if normalization != "max_umag":
        raise ValueError(f"Unsupported mode shape normalization: {normalization}")

    normalized = values.astype(np.float32, copy=True)
    if "U_mag" in columns:
        denom_source = np.abs(normalized[:, columns.index("U_mag")])
    else:
        denom_source = np.abs(normalized)
    denom = float(np.nanmax(denom_source)) if denom_source.size else 0.0
    if not np.isfinite(denom) or denom <= 0.0:
        denom = 1.0
    normalized /= np.float32(denom)
    return normalized


def _load_single_mode_shape(
    mode_path: Path,
    nodes_df: pd.DataFrame,
    columns: tuple[str, ...],
    normalization: str,
) -> np.ndarray:
    header = pd.read_csv(mode_path, nrows=0)
    missing = [column for column in columns if column not in header.columns]
    if missing:
        raise KeyError(f"{mode_path} is missing requested mode shape columns: {missing}")

    usecols = list(columns)
    if "node_index" in header.columns:
        usecols = ["node_index"] + usecols
    mode_df = pd.read_csv(mode_path, usecols=usecols)

    if "node_index" in nodes_df.columns and "node_index" in mode_df.columns:
        merged = nodes_df[["node_index"]].merge(
            mode_df[["node_index", *columns]],
            on="node_index",
            how="left",
            sort=False,
        )
        if merged[list(columns)].isna().any().any():
            missing_count = int(merged[list(columns)].isna().any(axis=1).sum())
            raise ValueError(f"{mode_path} is missing {missing_count} nodes after aligning by node_index")
        values = merged[list(columns)].to_numpy(dtype=np.float32)
    else:
        values = mode_df[list(columns)].to_numpy(dtype=np.float32)
        if values.shape[0] != len(nodes_df):
            raise ValueError(
                f"{mode_path} row count {values.shape[0]} does not match nodes.csv row count {len(nodes_df)}"
            )

    return _normalize_mode_shape_array(values, columns=columns, normalization=normalization)


@lru_cache(maxsize=512)
def _load_mode_shapes_cached(
    case_dir_str: str,
    columns: tuple[str, ...],
    mode_count_limit: int | None,
    normalization: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    case_dir = Path(case_dir_str)
    node_header = pd.read_csv(case_dir / "nodes.csv", nrows=0)
    node_usecols = ["node_index"] if "node_index" in node_header.columns else None
    nodes_df = pd.read_csv(case_dir / "nodes.csv", usecols=node_usecols)

    entries = _discover_mode_shape_entries(case_dir)
    if mode_count_limit is not None:
        entries = entries[: int(mode_count_limit)]
    if not entries:
        raise FileNotFoundError(f"No mode shapes selected for {case_dir}")

    frequencies = np.array([frequency_hz for _, frequency_hz, _ in entries], dtype=np.float32)
    mode_arrays = [
        _load_single_mode_shape(
            mode_path=mode_path,
            nodes_df=nodes_df,
            columns=columns,
            normalization=normalization,
        )
        for _, _, mode_path in entries
    ]
    mode_shapes = np.stack(mode_arrays, axis=1).astype(np.float32, copy=False)
    return torch.tensor(frequencies, dtype=torch.float32), torch.tensor(mode_shapes, dtype=torch.float32)


def _load_mode_shapes(
    case_dir: Path,
    columns: Sequence[str],
    mode_count_limit: int | None,
    normalization: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _load_mode_shapes_cached(
        str(case_dir.resolve()),
        tuple(str(column) for column in columns),
        mode_count_limit,
        normalization,
    )


def _resolve_case_and_target_path(sample_path: Path) -> tuple[Path, Path | None]:
    if sample_path.is_dir():
        return sample_path, None
    if sample_path.is_file() and sample_path.parent.name == PER_FREQUENCY_DIRNAME:
        return sample_path.parent.parent, sample_path
    raise ValueError(f"Unsupported sample path: {sample_path}")


def _load_frequency_scalar(target_path: Path | None) -> torch.Tensor:
    if target_path is None:
        return torch.empty(0, dtype=torch.float32)
    match = FRAME_FREQUENCY_PATTERN.search(target_path.stem)
    if match is None:
        raise ValueError(f"Could not parse frequency value from frame filename: {target_path.name}")
    return torch.tensor([float(match.group(1))], dtype=torch.float32)


def _load_nodes(
    case_dir: Path,
    node_columns: Sequence[str],
    global_payload: dict[str, Any],
    target_path: Path | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    nodes_df = pd.read_csv(case_dir / "nodes.csv")

    feature_arrays: list[np.ndarray] = []
    for column in node_columns:
        if column in nodes_df.columns:
            feature_arrays.append(nodes_df[column].to_numpy(dtype=np.float32))
            continue
        if column == "bc_mask":
            feature_arrays.append(_generate_boundary_mask(nodes_df, global_payload))
            continue
        raise KeyError(f"{case_dir / 'nodes.csv'} is missing requested node column: {column}")

    node_features = torch.tensor(np.stack(feature_arrays, axis=-1), dtype=torch.float32)

    if all(column in nodes_df.columns for column in DEFAULT_NODE_TARGET_COLUMNS):
        node_targets = torch.tensor(
            nodes_df[list(DEFAULT_NODE_TARGET_COLUMNS)].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )
    elif target_path is not None:
        node_targets = _load_per_frequency_targets(target_path, nodes_df)
    elif (case_dir / FINAL_RMISES_FILENAME).exists():
        node_targets = _load_final_rmises_targets(case_dir, nodes_df)
    else:
        raise KeyError(
            f"{case_dir} must provide either {list(DEFAULT_NODE_TARGET_COLUMNS)} in nodes.csv, "
            f"{FINAL_RMISES_FILENAME}, or a per-frequency target file"
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


def _canonical_node_region_cfg(node_region: dict[str, Any] | str | None) -> dict[str, Any]:
    if node_region is None:
        return {"type": "all"}
    if isinstance(node_region, str):
        return {"type": node_region}
    if not bool(node_region.get("enabled", True)):
        return {"type": "all"}
    return dict(node_region)


def _node_region_cache_part(node_region: dict[str, Any] | str | None) -> str:
    cfg = _canonical_node_region_cfg(node_region)
    if str(cfg.get("type", "all")).lower() in {"", "all", "none"}:
        return "all"
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return _sanitize_cache_part(payload)


def _build_earpiece_region_mask(
    nodes_df: pd.DataFrame,
    global_payload: dict[str, Any],
    cfg: dict[str, Any],
) -> np.ndarray:
    fixed_geometry = global_payload.get("fixed_geometry", {})
    earpiece_count = max(1, int(float(fixed_geometry.get("earpiece_Count_default", 3))))
    earpiece_hole_radius = float(fixed_geometry.get("earpiece_HoleRadius", 4.0))

    earpiece_radial_dist = _extract_param_value(global_payload, "earpiece_RadialDist", fallback_index=1)
    earpiece_top_width = _extract_param_value(global_payload, "earpiece_TopWidth", fallback_index=2)
    earpiece_hole_top_dist = _extract_param_value(global_payload, "earpiece_HoleTopDist", fallback_index=3)
    earpiece_top_fillet = _extract_param_value(global_payload, "earpiece_TopFilletRadius", fallback_index=4) or 0.0
    earpiece_bottom_fillet = _extract_param_value(global_payload, "earpiece_BottomFilletRadius", fallback_index=5) or 0.0
    plate_radius = _extract_param_value(global_payload, "plate_radius", fallback_index=6)

    required = {
        "earpiece_RadialDist": earpiece_radial_dist,
        "earpiece_TopWidth": earpiece_top_width,
        "earpiece_HoleTopDist": earpiece_hole_top_dist,
        "plate_radius": plate_radius,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"Cannot build earpiece node region; global.json is missing {missing}")

    root_margin = float(cfg.get("root_margin", max(earpiece_hole_radius, earpiece_bottom_fillet, 4.0)))
    top_margin = float(cfg.get("top_margin", max(earpiece_hole_radius, earpiece_top_fillet, 4.0)))
    width_margin = float(cfg.get("width_margin", earpiece_hole_radius))
    width_scale = float(cfg.get("width_scale", 1.5))
    radial_min = float(cfg.get("radial_min", float(plate_radius) - root_margin))
    axial_min = float(cfg.get("axial_min", float(plate_radius) - root_margin))
    axial_max = float(cfg.get("axial_max", float(earpiece_radial_dist) + float(earpiece_hole_top_dist) + top_margin))
    half_width = float(cfg.get("half_width", 0.5 * float(earpiece_top_width) * width_scale + width_margin))

    xy = nodes_df[["x", "y"]].to_numpy(dtype=np.float32)
    radius = np.sqrt(np.sum(np.square(xy), axis=1))
    angles = np.linspace(0.0, 2.0 * math.pi, num=earpiece_count, endpoint=False, dtype=np.float32)
    axial_axes = np.stack([-np.sin(angles), np.cos(angles)], axis=1).astype(np.float32, copy=False)
    tangent_axes = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32, copy=False)

    axial = xy @ axial_axes.T
    transverse = np.abs(xy @ tangent_axes.T)
    in_corridor = (axial >= axial_min) & (axial <= axial_max) & (transverse <= half_width)
    return ((radius >= radial_min) & in_corridor.any(axis=1)).astype(bool, copy=False)


def _build_node_region_indices(
    case_dir: Path,
    global_payload: dict[str, Any],
    node_region: dict[str, Any] | str | None,
) -> torch.Tensor | None:
    cfg = _canonical_node_region_cfg(node_region)
    region_type = str(cfg.get("type", "all")).lower()
    if region_type in {"", "all", "none"}:
        return None
    if region_type not in {"earpiece", "earpiece_region"}:
        raise ValueError(f"Unsupported dataset.node_region.type: {region_type}")

    nodes_df = pd.read_csv(case_dir / "nodes.csv", usecols=["x", "y"])
    mask = _build_earpiece_region_mask(nodes_df, global_payload=global_payload, cfg=cfg)
    if not mask.any():
        raise ValueError(f"Node region '{region_type}' selected zero nodes for {case_dir}")
    return torch.tensor(np.flatnonzero(mask), dtype=torch.long)


def _filter_graph_to_node_indices(
    node_features: torch.Tensor,
    node_targets: torch.Tensor,
    edge_index: torch.Tensor,
    edge_features: torch.Tensor,
    node_indices: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if node_indices is None:
        return node_features, node_targets, edge_index, edge_features

    node_indices = node_indices.to(dtype=torch.long)
    original_node_count = int(node_features.size(0))
    keep_nodes = torch.zeros(original_node_count, dtype=torch.bool)
    keep_nodes[node_indices] = True

    src, dst = edge_index
    keep_edges = keep_nodes[src] & keep_nodes[dst]
    if not bool(keep_edges.any()):
        raise ValueError("Node region filtering removed all edges; widen the region margins.")

    old_to_new = torch.full((original_node_count,), -1, dtype=torch.long)
    old_to_new[node_indices] = torch.arange(node_indices.numel(), dtype=torch.long)
    return (
        node_features[node_indices],
        node_targets[node_indices],
        old_to_new[edge_index[:, keep_edges]],
        edge_features[keep_edges],
    )


def _sanitize_cache_part(value: str) -> str:
    sanitized = "".join(ch if ch.isalnum() else "-" for ch in value)
    sanitized = sanitized.strip("-")
    return sanitized or "default"


def _cache_signature(
    node_columns: Sequence[str],
    edge_columns: Sequence[str],
    target_freq_key: str,
    make_undirected: bool,
    node_region: dict[str, Any] | str | None = None,
) -> str:
    node_part = "_".join(_sanitize_cache_part(column) for column in node_columns)
    edge_part = "_".join(_sanitize_cache_part(column) for column in edge_columns)
    freq_part = _sanitize_cache_part(target_freq_key)
    undir_part = "1" if make_undirected else "0"
    region_part = _node_region_cache_part(node_region)
    base = f"nodes-{node_part}__edges-{edge_part}__freq-{freq_part}__undir-{undir_part}"
    return base if region_part == "all" else f"{base}__region-{region_part}"


def _cache_sample_name(sample_path: Path) -> str:
    if sample_path.is_dir():
        return sample_path.name
    if sample_path.parent.name == PER_FREQUENCY_DIRNAME:
        return f"{sample_path.parent.parent.name}__{sample_path.stem}"
    return sample_path.stem


def _cache_path_for_case(
    cache_dir: str | Path,
    sample_path: Path,
    node_columns: Sequence[str],
    edge_columns: Sequence[str],
    target_freq_key: str,
    make_undirected: bool,
    node_region: dict[str, Any] | str | None = None,
) -> Path:
    signature = _cache_signature(
        node_columns=node_columns,
        edge_columns=edge_columns,
        target_freq_key=target_freq_key,
        make_undirected=make_undirected,
        node_region=node_region,
    )
    return Path(cache_dir) / signature / f"{_cache_sample_name(sample_path)}.pt"


def _remove_cache_file(cache_path: Path) -> None:
    try:
        cache_path.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        warnings.warn(f"Could not remove invalid cache file {cache_path}: {exc}", RuntimeWarning, stacklevel=2)


def _save_case_cache(cache_path: Path, case: CaseGraph) -> None:
    payload = {
        "name": case.name,
        "node_features": case.node_features,
        "edge_index": case.edge_index,
        "edge_features": case.edge_features,
        "params": case.params,
        "psd": case.psd,
        "freq_target": case.freq_target,
        "frequency_scalar": case.frequency_scalar,
        "fixed_geometry": case.fixed_geometry,
        "node_targets": case.node_targets,
        "source_node_indices": case.source_node_indices,
    }
    temp_path = cache_path.with_name(f".{cache_path.name}.{uuid4().hex}.tmp")
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, temp_path)
        temp_path.replace(cache_path)
    except Exception as exc:
        _remove_cache_file(temp_path)
        warnings.warn(f"Skipping case cache write for {cache_path}: {exc}", RuntimeWarning, stacklevel=2)


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
        frequency_scalar=payload.get("frequency_scalar", torch.empty(0, dtype=torch.float32)),
        fixed_geometry=payload.get("fixed_geometry", dict(DEFAULT_FIXED_GEOMETRY)),
        node_targets=payload["node_targets"],
        source_node_indices=payload.get("source_node_indices"),
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
    load_mode_shapes: bool = False,
    mode_shape_columns: Sequence[str] = DEFAULT_MODE_SHAPE_COLUMNS,
    mode_shape_count: int | None = None,
    mode_shape_normalization: str = "max_umag",
    node_region: dict[str, Any] | str | None = None,
) -> CaseGraph:
    sample_path = Path(case_dir)
    case_path, target_path = _resolve_case_and_target_path(sample_path)
    mode_columns = tuple(str(column) for column in mode_shape_columns)

    if cache_dir is not None:
        cache_path = _cache_path_for_case(
            cache_dir=cache_dir,
            sample_path=sample_path,
            node_columns=node_columns,
            edge_columns=edge_columns,
            target_freq_key=target_freq_key,
            make_undirected=make_undirected,
            node_region=node_region,
        )
        if cache_path.exists():
            try:
                cached_case = _load_case_cache(cache_path)
            except Exception as exc:
                warnings.warn(
                    f"Ignoring invalid case cache {cache_path}; rebuilding from source CSVs: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                _remove_cache_file(cache_path)
            else:
                if load_mode_shapes:
                    modal_frequencies, mode_shapes = _load_mode_shapes(
                        case_path,
                        columns=mode_columns,
                        mode_count_limit=mode_shape_count,
                        normalization=mode_shape_normalization,
                    )
                    if cached_case.source_node_indices is not None:
                        mode_shapes = mode_shapes[cached_case.source_node_indices]
                    cached_case.modal_frequencies = modal_frequencies
                    cached_case.mode_shapes = mode_shapes
                    cached_case.mode_shape_columns = mode_columns
                return cached_case

    global_payload = _load_global_payload(case_path)
    params, psd, freq_target = _load_global_json(global_payload, case_path, target_freq_key=target_freq_key)
    fixed_geometry = _load_fixed_geometry(global_payload)
    node_features, node_targets = _load_nodes(
        case_path,
        node_columns=node_columns,
        global_payload=global_payload,
        target_path=target_path,
    )
    edge_index, edge_features = _load_edges(
        case_path,
        edge_columns=edge_columns,
        make_undirected=make_undirected,
    )
    source_node_indices = _build_node_region_indices(
        case_path,
        global_payload=global_payload,
        node_region=node_region,
    )
    node_features, node_targets, edge_index, edge_features = _filter_graph_to_node_indices(
        node_features,
        node_targets,
        edge_index,
        edge_features,
        source_node_indices,
    )
    frequency_scalar = _load_frequency_scalar(target_path)
    sample_name = case_path.name if target_path is None else f"{case_path.name}/{target_path.name}"
    modal_frequencies = None
    mode_shapes = None
    if load_mode_shapes:
        modal_frequencies, mode_shapes = _load_mode_shapes(
            case_path,
            columns=mode_columns,
            mode_count_limit=mode_shape_count,
            normalization=mode_shape_normalization,
        )
        if source_node_indices is not None:
            mode_shapes = mode_shapes[source_node_indices]

    case = CaseGraph(
        name=sample_name,
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        params=params,
        psd=psd,
        freq_target=freq_target,
        frequency_scalar=frequency_scalar,
        fixed_geometry=fixed_geometry,
        node_targets=node_targets,
        modal_frequencies=modal_frequencies,
        mode_shapes=mode_shapes,
        mode_shape_columns=mode_columns,
        source_node_indices=source_node_indices,
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
    load_mode_shapes: bool = False,
    mode_shape_columns: Sequence[str] = DEFAULT_MODE_SHAPE_COLUMNS,
    mode_shape_count: int | None = None,
    mode_shape_normalization: str = "max_umag",
    node_region: dict[str, Any] | str | None = None,
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
            load_mode_shapes=load_mode_shapes,
            mode_shape_columns=mode_shape_columns,
            mode_shape_count=mode_shape_count,
            mode_shape_normalization=mode_shape_normalization,
            node_region=node_region,
        )
    return loaded


def expand_case_sample_paths(case_paths: Sequence[Path], dataset_cfg: dict[str, Any]) -> list[Path]:
    sample_mode = str(dataset_cfg.get("sample_mode", "case")).lower()
    if sample_mode == "case":
        return list(case_paths)
    if sample_mode != "per_frequency":
        raise ValueError(f"Unsupported dataset.sample_mode: {sample_mode}")

    include_zero_frequency = bool(dataset_cfg.get("include_zero_frequency", False))
    min_frequency = dataset_cfg.get("min_frequency_hz")
    max_frequency = dataset_cfg.get("max_frequency_hz")
    expanded: list[Path] = []
    for case_path in case_paths:
        frame_dir = Path(case_path) / PER_FREQUENCY_DIRNAME
        if not frame_dir.exists():
            raise FileNotFoundError(f"Per-frequency target directory does not exist: {frame_dir}")

        frame_paths: list[tuple[float, Path]] = []
        for frame_path in sorted(frame_dir.glob("*.csv")):
            frequency_scalar = float(_load_frequency_scalar(frame_path).item())
            if not include_zero_frequency and abs(frequency_scalar) < 1e-9:
                continue
            if min_frequency is not None and frequency_scalar < float(min_frequency):
                continue
            if max_frequency is not None and frequency_scalar > float(max_frequency):
                continue
            frame_paths.append((frequency_scalar, frame_path))
        expanded.extend(path for _, path in sorted(frame_paths, key=lambda item: item[0]))
    return expanded


def build_global_features(
    case: CaseGraph,
    use_psd: bool,
    use_freq_top3: bool = False,
    use_frequency_scalar: bool = False,
    use_frequency_relations: bool = False,
) -> torch.Tensor:
    features = [case.params]
    if use_psd:
        features.append(case.psd)
    if use_freq_top3:
        features.append(case.freq_target)
    if use_frequency_scalar:
        if case.frequency_scalar.numel() == 0:
            raise ValueError("Requested use_frequency_scalar=True but this sample does not provide a frequency value.")
        features.append(case.frequency_scalar)
    if use_frequency_relations:
        if case.frequency_scalar.numel() == 0:
            raise ValueError("Requested use_frequency_relations=True but this sample does not provide a frequency value.")
        if case.freq_target.numel() < 1:
            raise ValueError("Requested use_frequency_relations=True but this sample has no modal frequency targets.")
        modes = case.freq_target[:3].clamp_min(1e-6)
        current_frequency = case.frequency_scalar[:1]
        signed_delta = (current_frequency - modes) / modes
        abs_delta = signed_delta.abs()
        nearest_delta = abs_delta.min().reshape(1)
        first_mode_ratio = current_frequency / modes[:1]
        features.append(torch.cat([signed_delta, abs_delta, nearest_delta, first_mode_ratio], dim=0))
    if len(features) == 1:
        return case.params.clone()
    return torch.cat(features, dim=0)
