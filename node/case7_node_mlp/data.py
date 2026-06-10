from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import json
import math
import random
import re

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree


PER_FREQUENCY_DIRNAME = "per_frequency_mises"
PER_FREQUENCY_TARGET_COLUMN = "MISES_psd_density"
FRAME_FREQUENCY_PATTERN = re.compile(r"_(\d+(?:\.\d+)?)Hz$")
MODE_SHAPES_DIRNAME = "mode_shapes"
MODAL_FREQUENCIES_FILENAME = "modal_frequencies.csv"
MODE_SHAPE_PATTERN = re.compile(r"mode_(\d+)_(\d+(?:\.\d+)?)Hz$", re.IGNORECASE)
DEFAULT_MODE_SHAPE_COLUMNS = ("U1", "U2", "U3", "U_mag")
NODE_REGION_MASK_COLUMNS = (
    "center_node_mask",
    "center_couple_mask",
    "plate_hole_wall_mask",
    "ear_hole_wall_mask",
    "ear_connection_fillet_mask",
    "ear_connection_earside_mask",
    "ear_connection_mask",
)
STRESS_REGION_MASK_COLUMNS = (
    "center_couple_mask",
    "plate_hole_wall_mask",
    "ear_hole_wall_mask",
    "ear_connection_fillet_mask",
    "ear_connection_earside_mask",
    "ear_connection_mask",
)
STRESS_REGION_GROUPS = {
    "center_couple_region": ("center_couple_mask",),
    "plate_hole_region": ("plate_hole_wall_mask",),
    "ear_hole_region": ("ear_hole_wall_mask",),
    "ear_connection_region": (
        "ear_connection_fillet_mask",
        "ear_connection_earside_mask",
        "ear_connection_mask",
    ),
}

DEFAULT_FIXED_GEOMETRY = {
    "plate_thickness": 15.0,
    "plate_HoleRadius": 4.0,
    "plate_HoleDist": 25.0,
    "plate_HoleCount": 4,
    "earpiece_HoleRadius": 4.0,
    "earpiece_Count_default": 3,
    "mass_couple_radius": 65.0,
}

BASE_GEOMETRY_FEATURE_NAMES = [
    "x_norm",
    "y_norm",
    "z_norm",
    "r_norm",
    "dist_to_edge",
    "sin_theta",
    "cos_theta",
    "dist_to_ear_hole_edge_local",
    "dist_to_ear_hole_edge_global",
    "earpiece_width_over_plate_radius",
    "earpiece_width_over_hole_radius",
    "center_radius_local",
    "center_couple_signed",
]
EARPIECE_LOCAL_FEATURE_NAMES = [
    "ear_local_u_over_hole_radius",
    "ear_local_v_over_half_width",
    "ear_local_r_over_hole_radius",
    "ear_local_sin",
    "ear_local_cos",
    "dist_to_ear_axis_over_half_width",
    "dist_to_ear_root_signed_over_hole_radius",
    "dist_to_ear_root_abs_over_hole_radius",
    "near_ear_root",
    "dist_to_hole_center_over_hole_radius",
    "near_hole_root_bridge",
]
DISK_CENTER_FEATURE_NAMES = [
    "center_region_r_over_mask_radius",
    "center_region_signed",
    "near_center_region_exp",
]
PLATE_HOLE_FEATURE_NAMES = [
    "dist_to_plate_hole_center_over_radius",
    "dist_to_plate_hole_edge_over_radius",
    "dist_to_plate_hole_edge_over_plate_radius",
    "near_plate_hole_wall",
]
ANGULAR_PERIODIC_FEATURE_NAMES = [
    "sin_ear_period_theta",
    "cos_ear_period_theta",
    "sin_plate_hole_period_theta",
    "cos_plate_hole_period_theta",
]
STRESS_REGION_DISTANCE_FEATURE_NAMES = [
    "dist_to_center_couple_region_over_plate_radius",
    "dist_to_plate_hole_region_over_plate_radius",
    "dist_to_ear_hole_region_over_plate_radius",
    "dist_to_ear_connection_region_over_plate_radius",
    "dist_to_nearest_stress_region_over_plate_radius",
]
CENTER_MODAL_INTERACTION_SPECS = (
    (
        "center_couple_mask",
        "modal_baseline_log_grad_umag_max",
        "center_couple_mask_x_modal_baseline_log_grad_umag_max",
    ),
    (
        "near_center_region_exp",
        "modal_baseline_log_grad_umag_max",
        "near_center_region_exp_x_modal_baseline_log_grad_umag_max",
    ),
    (
        "near_center_region_exp",
        "weighted_umag_frf",
        "near_center_region_exp_x_weighted_umag_frf",
    ),
    (
        "near_center_region_exp",
        "weighted_grad_umag_max_frf",
        "near_center_region_exp_x_weighted_grad_umag_max_frf",
    ),
    (
        "near_center_region_exp",
        "active1_modal_weight_frf",
        "near_center_region_exp_x_active1_modal_weight_frf",
    ),
    (
        "near_center_region_exp",
        "active1_log_modal_gain_frf",
        "near_center_region_exp_x_active1_log_modal_gain_frf",
    ),
)
GEOMETRY_FEATURE_NAMES = list(BASE_GEOMETRY_FEATURE_NAMES)
MASK_FEATURE_NAMES = ["bc_mask", "near_ear_hole", "near_center_couple"]


@dataclass
class RawPointSample:
    name: str
    case_name: str
    frequency_hz: float
    geometry_features: torch.Tensor
    scaled_features: torch.Tensor
    mask_features: torch.Tensor
    target_log: torch.Tensor
    target_raw: torch.Tensor
    node_indices: torch.Tensor
    geometry_feature_names: list[str]
    scaled_feature_names: list[str]
    mask_feature_names: list[str]
    region_masks: dict[str, torch.Tensor] | None = None

    @property
    def num_points(self) -> int:
        return int(self.target_raw.numel())


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


def _allocate_split_counts(total: int, train_ratio: float, val_ratio: float, test_ratio: float) -> tuple[int, int, int]:
    ratios = [float(train_ratio), float(val_ratio), float(test_ratio)]
    if any(ratio < 0.0 for ratio in ratios):
        raise ValueError("Split ratios must be non-negative.")
    ratio_sum = sum(ratios)
    if ratio_sum <= 0.0:
        raise ValueError("At least one split ratio must be positive.")

    normalized = [ratio / ratio_sum for ratio in ratios]
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
            (candidate for candidate, count in enumerate(counts) if count > 1),
            key=lambda candidate: counts[candidate],
            default=None,
        )
        if donor is None:
            raise ValueError("Could not allocate non-empty dataset splits.")
        counts[donor] -= 1
        counts[index] += 1
    return counts[0], counts[1], counts[2]


def _validate_splits(splits: dict[str, list[str]], available: dict[str, Path]) -> dict[str, list[str]]:
    validated: dict[str, list[str]] = {}
    seen: dict[str, str] = {}
    for split_name, names in splits.items():
        normalized_names = _normalize_case_names(names)
        missing = [name for name in normalized_names if name not in available]
        if missing:
            raise ValueError(f"Missing cases for split '{split_name}': {sorted(missing)}")
        for name in normalized_names:
            if name in seen:
                raise ValueError(f"Case '{name}' appears in both '{seen[name]}' and '{split_name}'.")
            seen[name] = split_name
        validated[split_name] = sorted(normalized_names)
    if not validated.get("train"):
        raise ValueError("Training split is empty.")
    if not validated.get("val"):
        raise ValueError("Validation split is empty.")
    return validated


def resolve_case_splits(root: str | Path, dataset_cfg: dict[str, Any]) -> dict[str, list[str]]:
    available = discover_case_index(root)
    explicit_lists_present = any(dataset_cfg.get(key) is not None for key in ("train_cases", "val_cases", "test_cases"))
    split_mode = str(dataset_cfg.get("split_mode", "explicit" if explicit_lists_present else "ratio")).lower()

    if split_mode == "explicit":
        return _validate_splits(
            {
                "train": _normalize_case_names(dataset_cfg.get("train_cases")),
                "val": _normalize_case_names(dataset_cfg.get("val_cases")),
                "test": _normalize_case_names(dataset_cfg.get("test_cases")),
            },
            available,
        )
    if split_mode != "ratio":
        raise ValueError(f"Unsupported split_mode: {split_mode}")

    selected_names = sorted(available)
    include_cases = _normalize_case_names(dataset_cfg.get("include_cases"))
    if include_cases:
        missing = [name for name in include_cases if name not in available]
        if missing:
            raise ValueError(f"Included cases are missing: {sorted(missing)}")
        selected_names = include_cases

    exclude_cases = set(_normalize_case_names(dataset_cfg.get("exclude_cases")))
    selected_names = [name for name in selected_names if name not in exclude_cases]
    if not selected_names:
        raise ValueError("No complete cases remain after include/exclude filters.")

    shuffled_names = list(selected_names)
    random.Random(int(dataset_cfg.get("split_seed", 42))).shuffle(shuffled_names)
    max_cases = dataset_cfg.get("max_cases")
    if max_cases is not None:
        shuffled_names = shuffled_names[: int(max_cases)]

    train_count, val_count, test_count = _allocate_split_counts(
        len(shuffled_names),
        train_ratio=float(dataset_cfg.get("train_ratio", 0.8)),
        val_ratio=float(dataset_cfg.get("val_ratio", 0.1)),
        test_ratio=float(dataset_cfg.get("test_ratio", 0.1)),
    )
    train_end = train_count
    val_end = train_end + val_count
    return _validate_splits(
        {
            "train": sorted(shuffled_names[:train_end]),
            "val": sorted(shuffled_names[train_end:val_end]),
            "test": sorted(shuffled_names[val_end : val_end + test_count]),
        },
        available,
    )


def _load_global_payload(case_dir: Path) -> dict[str, Any]:
    return json.loads((case_dir / "global.json").read_text(encoding="utf-8"))


def _region_cache_key(region_cfg: dict[str, Any] | None) -> str:
    return json.dumps(region_cfg or {}, sort_keys=True, separators=(",", ":"))


@lru_cache(maxsize=128)
def _load_case_static_cached(case_dir_str: str, region_key: str) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray]:
    case_dir = Path(case_dir_str)
    payload = _load_global_payload(case_dir)
    # Keep targets out of this cache, but preserve nodes.csv region labels for
    # full-part diagnostics and feature construction.
    nodes_df = pd.read_csv(case_dir / "nodes.csv")
    region_cfg = json.loads(region_key)
    earpiece_mask = _build_earpiece_region_mask(nodes_df, global_payload=payload, region_cfg=region_cfg)
    return nodes_df, payload, earpiece_mask


def _load_case_static(case_dir: Path, region_cfg: dict[str, Any] | None) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray]:
    nodes_df, payload, earpiece_mask = _load_case_static_cached(
        str(case_dir.resolve()),
        _region_cache_key(region_cfg),
    )
    return nodes_df, dict(payload), earpiece_mask


def _load_fixed_geometry(payload: dict[str, Any]) -> dict[str, float]:
    fixed_geometry = dict(DEFAULT_FIXED_GEOMETRY)
    raw_fixed_geometry = payload.get("fixed_geometry", {})
    for key, default_value in DEFAULT_FIXED_GEOMETRY.items():
        fixed_geometry[key] = float(raw_fixed_geometry.get(key, default_value))
    return fixed_geometry


def _center_couple_mask_radius(payload: dict[str, Any], default: float = 15.0) -> float:
    mask_definition = payload.get("nodes_csv_mask_definition", {})
    raw_value = mask_definition.get("center_couple_mask_radius", default)
    try:
        radius = float(raw_value)
    except (TypeError, ValueError):
        radius = float(default)
    if not math.isfinite(radius) or radius <= 0.0:
        radius = float(default)
    return max(radius, 1e-6)


def _node_mask_values(nodes_df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in nodes_df.columns:
        return np.zeros(len(nodes_df), dtype=bool)
    return nodes_df[column].to_numpy(dtype=np.float32) > 0.5


def _combined_node_mask(nodes_df: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    mask = np.zeros(len(nodes_df), dtype=bool)
    for column in columns:
        mask |= _node_mask_values(nodes_df, str(column))
    return mask


def _build_disk_center_region_mask(nodes_df: pd.DataFrame) -> np.ndarray:
    return _combined_node_mask(nodes_df, ("center_couple_mask", "center_node_mask"))


def _pointset_cache_key(pointset_cfg: dict[str, Any] | None) -> str:
    return json.dumps(pointset_cfg or {}, sort_keys=True, separators=(",", ":"))


@lru_cache(maxsize=16)
def _load_pointset_nodes_cached(pointset_key: str) -> dict[str, set[int]]:
    cfg = json.loads(pointset_key)
    raw_path = cfg.get("path") or cfg.get("csv_path") or cfg.get("file")
    if not raw_path:
        return {}
    pointset_path = Path(str(raw_path))
    if not pointset_path.exists():
        raise FileNotFoundError(f"Pointset filter file does not exist: {pointset_path}")

    case_column = str(cfg.get("case_column", "case_name"))
    node_column = str(cfg.get("node_column", "node_index"))
    type_column = str(cfg.get("type_column", "pointset_type"))
    raw_pointset_type = cfg.get("pointset_type", cfg.get("type"))
    df = pd.read_csv(pointset_path)
    missing = [column for column in (case_column, node_column) if column not in df.columns]
    if missing:
        raise KeyError(f"{pointset_path} is missing pointset columns: {missing}")
    if raw_pointset_type is not None:
        if type_column not in df.columns:
            raise KeyError(f"{pointset_path} is missing pointset type column: {type_column}")
        allowed_types = {str(item) for item in raw_pointset_type} if isinstance(raw_pointset_type, list) else {str(raw_pointset_type)}
        df = df[df[type_column].astype(str).isin(allowed_types)]

    grouped: dict[str, set[int]] = {}
    for case_name, group in df.groupby(case_column, sort=False):
        grouped[str(case_name)] = {int(value) for value in group[node_column].dropna().to_numpy()}
    return grouped


def _build_pointset_mask(nodes_df: pd.DataFrame, case_name: str, pointset_cfg: dict[str, Any] | None) -> np.ndarray:
    if not pointset_cfg:
        return np.ones(len(nodes_df), dtype=bool)
    grouped = _load_pointset_nodes_cached(_pointset_cache_key(pointset_cfg))
    allowed_nodes = grouped.get(str(case_name), set())
    if not allowed_nodes:
        return np.zeros(len(nodes_df), dtype=bool)
    if "node_index" in nodes_df.columns:
        node_values = nodes_df["node_index"].to_numpy(dtype=np.int64)
    else:
        node_values = np.arange(len(nodes_df), dtype=np.int64)
    return np.isin(node_values, np.fromiter(allowed_nodes, dtype=np.int64)).astype(bool, copy=False)


def _build_node_scope_mask(
    nodes_df: pd.DataFrame,
    payload: dict[str, Any],
    earpiece_mask: np.ndarray,
    dataset_cfg: dict[str, Any],
    case_name: str,
) -> np.ndarray:
    raw_scope = dataset_cfg.get("node_scope")
    if raw_scope is None:
        earpiece_cfg = dataset_cfg.get("earpiece_region", {})
        raw_scope = "earpiece" if bool(earpiece_cfg.get("enabled", True)) else "all_nodes"
    scope = str(raw_scope).strip().lower()
    if scope in {"earpiece", "earpiece_only", "ear"}:
        mask = earpiece_mask.astype(bool, copy=True)
    elif scope in {"disk_center", "center", "center_region", "disk_center_region"}:
        mask = _build_disk_center_region_mask(nodes_df)
    elif scope in {"all", "all_nodes", "full", "full_part", "fullpart"}:
        mask = np.ones(len(nodes_df), dtype=bool)
    else:
        raise ValueError(f"Unsupported dataset.node_scope: {raw_scope}")

    if bool(dataset_cfg.get("exclude_bc_nodes", False)):
        if "bc_mask" in nodes_df.columns:
            mask &= nodes_df["bc_mask"].to_numpy(dtype=np.float32) <= 0.5
        else:
            mask &= _build_boundary_mask(nodes_df, payload) <= 0.5
    if bool(dataset_cfg.get("exclude_center_node", False)):
        mask &= ~_node_mask_values(nodes_df, "center_node_mask")

    pointset_cfg = dataset_cfg.get("pointset") or dataset_cfg.get("pointset_filter")
    mask &= _build_pointset_mask(nodes_df, case_name=case_name, pointset_cfg=pointset_cfg)
    return mask


def build_node_selection_mask(
    case_dir: Path,
    dataset_cfg: dict[str, Any],
    target_values: np.ndarray | None = None,
) -> np.ndarray:
    nodes_df, payload, earpiece_mask = _load_case_static(
        case_dir,
        region_cfg=dataset_cfg.get("earpiece_region"),
    )
    mask = _build_node_scope_mask(
        nodes_df=nodes_df,
        payload=payload,
        earpiece_mask=earpiece_mask,
        dataset_cfg=dataset_cfg,
        case_name=case_dir.name,
    )
    if target_values is not None:
        mask &= np.isfinite(target_values) & (target_values >= 0.0)
    return mask


def _min_distance_to_mask(points: np.ndarray, region_points: np.ndarray, chunk_size: int = 4096) -> np.ndarray:
    if region_points.size == 0:
        return np.full(points.shape[0], np.nan, dtype=np.float32)
    distances, _ = cKDTree(region_points).query(points, k=1, workers=-1)
    return distances.astype(np.float32, copy=False)


@lru_cache(maxsize=64)
def _load_stress_region_distances_cached(case_dir_str: str) -> dict[str, torch.Tensor]:
    case_dir = Path(case_dir_str)
    nodes_df = pd.read_csv(case_dir / "nodes.csv")
    points = nodes_df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    output: dict[str, torch.Tensor] = {}
    nearest: np.ndarray | None = None
    for region_name, columns in STRESS_REGION_GROUPS.items():
        mask = _combined_node_mask(nodes_df, columns)
        distances = _min_distance_to_mask(points, points[mask])
        if np.isnan(distances).all():
            distances = np.full(points.shape[0], 1e6, dtype=np.float32)
        else:
            finite_max = float(np.nanmax(distances[np.isfinite(distances)])) if np.isfinite(distances).any() else 1e6
            distances = np.nan_to_num(distances, nan=max(finite_max, 1e6)).astype(np.float32, copy=False)
        output[region_name] = torch.tensor(distances, dtype=torch.float32)
        nearest = distances if nearest is None else np.minimum(nearest, distances)
    if nearest is None:
        nearest = np.full(points.shape[0], 1e6, dtype=np.float32)
    output["nearest_stress_region"] = torch.tensor(nearest.astype(np.float32, copy=False), dtype=torch.float32)
    return output


def _extract_param_value(payload: dict[str, Any], key: str, fallback_index: int | None = None) -> float | None:
    params = payload.get("params", {})
    if key in params:
        return float(params[key])
    params_list = payload.get("params_list", [])
    if fallback_index is not None and fallback_index < len(params_list):
        return float(params_list[fallback_index])
    return None


def _frequency_from_path(target_path: Path) -> float:
    match = FRAME_FREQUENCY_PATTERN.search(target_path.stem)
    if match is None:
        raise ValueError(f"Could not parse frequency from target file name: {target_path.name}")
    return float(match.group(1))


def expand_case_sample_paths(case_paths: Sequence[Path], dataset_cfg: dict[str, Any]) -> list[Path]:
    sample_mode = str(dataset_cfg.get("sample_mode", "per_frequency")).lower()
    if sample_mode != "per_frequency":
        raise ValueError(f"Node MLP currently supports only dataset.sample_mode=per_frequency, got {sample_mode}")

    pointset_cfg = dataset_cfg.get("pointset") or dataset_cfg.get("pointset_filter")
    pointset_case_names: set[str] | None = None
    if pointset_cfg:
        pointset_case_names = set(_load_pointset_nodes_cached(_pointset_cache_key(pointset_cfg)))

    include_zero_frequency = bool(dataset_cfg.get("include_zero_frequency", False))
    min_frequency = dataset_cfg.get("min_frequency_hz")
    max_frequency = dataset_cfg.get("max_frequency_hz")
    max_frames_per_case = dataset_cfg.get("max_frames_per_case")

    expanded: list[Path] = []
    for case_path in case_paths:
        if pointset_case_names is not None and case_path.name not in pointset_case_names:
            continue
        frame_dir = case_path / PER_FREQUENCY_DIRNAME
        if not frame_dir.exists():
            raise FileNotFoundError(f"Per-frequency target directory does not exist: {frame_dir}")
        frame_paths: list[tuple[float, Path]] = []
        for frame_path in sorted(frame_dir.glob("*.csv")):
            frequency_hz = _frequency_from_path(frame_path)
            if not include_zero_frequency and abs(frequency_hz) < 1e-9:
                continue
            if min_frequency is not None and frequency_hz < float(min_frequency):
                continue
            if max_frequency is not None and frequency_hz > float(max_frequency):
                continue
            frame_paths.append((frequency_hz, frame_path))
        frame_paths = sorted(frame_paths, key=lambda item: item[0])
        if max_frames_per_case is not None:
            frame_paths = frame_paths[: int(max_frames_per_case)]
        expanded.extend(path for _, path in frame_paths)
    return expanded


def _load_aligned_target_column(
    target_df: pd.DataFrame,
    target_column: str,
    nodes_df: pd.DataFrame,
    target_path: Path,
) -> np.ndarray:
    if target_column not in target_df.columns:
        raise KeyError(f"{target_path} is missing {target_column}")
    if "node_index" in nodes_df.columns and "node_index" in target_df.columns:
        node_index = nodes_df["node_index"].to_numpy()
        target_node_index = target_df["node_index"].to_numpy()
        if target_node_index.shape[0] == node_index.shape[0] and np.array_equal(target_node_index, node_index):
            return target_df[target_column].to_numpy(dtype=np.float32)

        merged = nodes_df[["node_index"]].merge(
            target_df[["node_index", target_column]],
            on="node_index",
            how="left",
            sort=False,
        )
        if merged[target_column].isna().any():
            missing_count = int(merged[target_column].isna().sum())
            raise ValueError(f"{target_path} is missing {missing_count} node targets after node_index alignment")
        return merged[target_column].to_numpy(dtype=np.float32)

    values = target_df[target_column].to_numpy(dtype=np.float32)
    if values.shape[0] != len(nodes_df):
        raise ValueError(f"{target_path} row count {values.shape[0]} does not match nodes.csv row count {len(nodes_df)}")
    return values


def _build_boundary_mask(nodes_df: pd.DataFrame, payload: dict[str, Any]) -> np.ndarray:
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
        [-float(earpiece_radial_dist) * np.sin(angles), float(earpiece_radial_dist) * np.cos(angles)],
        axis=1,
    ).astype(np.float32, copy=False)
    radial_distance = np.sqrt(np.sum(np.square(xy[:, None, :] - centers[None, :, :]), axis=-1))
    in_hole = radial_distance <= np.float32(earpiece_hole_radius * 1.1)
    in_z_range = (z >= -1.0) & (z <= (plate_thickness + 1.0))
    return (in_hole.any(axis=1) & in_z_range).astype(np.float32, copy=False)


def _build_earpiece_region_mask(
    nodes_df: pd.DataFrame,
    global_payload: dict[str, Any],
    region_cfg: dict[str, Any] | None = None,
) -> np.ndarray:
    cfg = region_cfg or {}
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
        raise ValueError(f"Cannot build earpiece region; global.json is missing {missing}")

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


def _parse_mode_shape_frequency(mode_path: Path) -> tuple[int, float]:
    match = MODE_SHAPE_PATTERN.search(mode_path.stem)
    if match is None:
        raise ValueError(f"Could not parse modal index/frequency from mode shape file: {mode_path.name}")
    return int(match.group(1)), float(match.group(2))


def _discover_mode_shape_entries(case_dir: Path) -> list[tuple[int, float, Path]]:
    modal_table_path = case_dir / MODAL_FREQUENCIES_FILENAME
    entries: list[tuple[int, float, Path]] = []
    if modal_table_path.exists():
        modal_df = pd.read_csv(modal_table_path)
        if {"mode_index", "frequency_hz", "file"}.issubset(set(modal_df.columns)):
            for _, row in modal_df.iterrows():
                raw_path = Path(str(row["file"]).replace("\\", "/"))
                mode_path = raw_path if raw_path.is_absolute() else case_dir / raw_path
                if not mode_path.exists():
                    mode_path = case_dir / MODE_SHAPES_DIRNAME / raw_path.name
                if not mode_path.exists():
                    raise FileNotFoundError(f"Mode shape file does not exist: {mode_path}")
                entries.append((int(row["mode_index"]), float(row["frequency_hz"]), mode_path))
            return sorted(entries, key=lambda item: item[0])

    mode_dir = case_dir / MODE_SHAPES_DIRNAME
    if not mode_dir.exists():
        raise FileNotFoundError(f"Mode shape directory does not exist: {mode_dir}")
    for mode_path in sorted(mode_dir.glob("*.csv")):
        mode_index, frequency_hz = _parse_mode_shape_frequency(mode_path)
        entries.append((mode_index, frequency_hz, mode_path))
    if not entries:
        raise FileNotFoundError(f"No mode shape CSV files found in {mode_dir}")
    return sorted(entries, key=lambda item: item[0])


def _normalize_mode_shape_array(values: np.ndarray, columns: tuple[str, ...], normalization: str) -> np.ndarray:
    if normalization == "none":
        return values.astype(np.float32, copy=False)
    if normalization not in {"max_umag", "rms_umag"}:
        raise ValueError(f"Unsupported mode_shape_normalization: {normalization}")
    normalized = values.astype(np.float32, copy=True)
    if "U_mag" in columns:
        denom_source = np.abs(normalized[:, columns.index("U_mag")])
    else:
        denom_source = np.abs(normalized)
    if normalization == "max_umag":
        denom = float(np.nanmax(denom_source)) if denom_source.size else 0.0
    else:
        denom = float(np.sqrt(np.nanmean(np.square(denom_source, dtype=np.float64)))) if denom_source.size else 0.0
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
        raise KeyError(f"{mode_path} is missing mode shape columns: {missing}")

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
            raise ValueError(f"{mode_path} is missing {missing_count} nodes after node_index alignment")
        values = merged[list(columns)].to_numpy(dtype=np.float32)
    else:
        values = mode_df[list(columns)].to_numpy(dtype=np.float32)
        if values.shape[0] != len(nodes_df):
            raise ValueError(f"{mode_path} row count {values.shape[0]} does not match nodes.csv row count {len(nodes_df)}")
    return _normalize_mode_shape_array(values, columns=columns, normalization=normalization)


@lru_cache(maxsize=32)
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
        _load_single_mode_shape(mode_path, nodes_df=nodes_df, columns=columns, normalization=normalization)
        for _, _, mode_path in entries
    ]
    mode_shapes = np.stack(mode_arrays, axis=1).astype(np.float32, copy=False)
    return torch.tensor(frequencies, dtype=torch.float32), torch.tensor(mode_shapes, dtype=torch.float32)


def _load_mode_shapes(
    case_dir: Path,
    columns: Sequence[str],
    mode_count_limit: int | None,
    normalization: str,
) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
    mode_columns = tuple(str(column) for column in columns)
    modal_frequencies, mode_shapes = _load_mode_shapes_cached(
        str(case_dir.resolve()),
        mode_columns,
        mode_count_limit,
        normalization,
    )
    return modal_frequencies, mode_shapes, mode_columns


def _resolve_edge_columns(edge_header: pd.Index) -> tuple[str, str, str | None]:
    columns = set(edge_header)
    src_column = "src" if "src" in columns else "source" if "source" in columns else None
    dst_column = "dst" if "dst" in columns else "target" if "target" in columns else None
    if src_column is None or dst_column is None:
        raise KeyError(f"edges.csv must contain src/dst or source/target columns, got {sorted(columns)}")
    dist_column = "dist" if "dist" in columns else None
    return src_column, dst_column, dist_column


@lru_cache(maxsize=32)
def _load_edge_index_cached(case_dir_str: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    case_dir = Path(case_dir_str)
    edge_header = pd.read_csv(case_dir / "edges.csv", nrows=0).columns
    src_column, dst_column, dist_column = _resolve_edge_columns(edge_header)
    usecols = [src_column, dst_column]
    if dist_column is not None:
        usecols.append(dist_column)
    else:
        usecols.extend([column for column in ("dx", "dy", "dz") if column in edge_header])
    edges_df = pd.read_csv(case_dir / "edges.csv", usecols=usecols)

    src_raw = edges_df[src_column].to_numpy(dtype=np.int64)
    dst_raw = edges_df[dst_column].to_numpy(dtype=np.int64)
    node_header = pd.read_csv(case_dir / "nodes.csv", nrows=0).columns
    node_usecols = ["node_index"] if "node_index" in node_header else None
    node_count = len(pd.read_csv(case_dir / "nodes.csv", usecols=node_usecols))

    if src_raw.size and (src_raw.min() < 0 or dst_raw.min() < 0 or src_raw.max() >= node_count or dst_raw.max() >= node_count):
        if "node_index" not in node_header:
            raise ValueError(f"{case_dir / 'edges.csv'} edge indices are outside node row range and nodes.csv has no node_index")
        node_indices = pd.read_csv(case_dir / "nodes.csv", usecols=["node_index"])["node_index"].to_numpy(dtype=np.int64)
        node_position = {int(node_index): position for position, node_index in enumerate(node_indices)}
        try:
            src = np.array([node_position[int(value)] for value in src_raw], dtype=np.int64)
            dst = np.array([node_position[int(value)] for value in dst_raw], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"{case_dir / 'edges.csv'} references a node_index not present in nodes.csv") from exc
    else:
        src = src_raw
        dst = dst_raw

    if dist_column is not None:
        dist = edges_df[dist_column].to_numpy(dtype=np.float32)
    else:
        deltas = [edges_df[column].to_numpy(dtype=np.float32) for column in ("dx", "dy", "dz") if column in edges_df.columns]
        if not deltas:
            raise KeyError(f"{case_dir / 'edges.csv'} must contain dist or dx/dy/dz columns")
        dist = np.sqrt(np.sum(np.square(np.stack(deltas, axis=1), dtype=np.float64), axis=1)).astype(np.float32)

    valid = np.isfinite(dist) & (dist > 1e-12) & (src >= 0) & (dst >= 0) & (src < node_count) & (dst < node_count)
    return src[valid].astype(np.int64, copy=False), dst[valid].astype(np.int64, copy=False), dist[valid].astype(np.float32, copy=False)


def _aggregate_edge_gradient(edge_values: np.ndarray, src: np.ndarray, dst: np.ndarray, dist: np.ndarray, node_count: int) -> tuple[np.ndarray, np.ndarray]:
    mode_count = int(edge_values.shape[-1])
    sum_values = np.zeros((node_count, mode_count), dtype=np.float32)
    max_values = np.zeros((node_count, mode_count), dtype=np.float32)
    counts = np.zeros((node_count, 1), dtype=np.float32)

    scaled = edge_values / np.maximum(dist[:, None], np.float32(1e-6))
    np.add.at(sum_values, src, scaled)
    np.add.at(sum_values, dst, scaled)
    np.maximum.at(max_values, src, scaled)
    np.maximum.at(max_values, dst, scaled)
    np.add.at(counts, src, 1.0)
    np.add.at(counts, dst, 1.0)

    mean_values = sum_values / np.maximum(counts, np.float32(1.0))
    return mean_values.astype(np.float32, copy=False), max_values.astype(np.float32, copy=False)


@lru_cache(maxsize=16)
def _load_mode_edge_gradients_cached(
    case_dir_str: str,
    columns: tuple[str, ...],
    mode_count_limit: int | None,
    normalization: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    case_dir = Path(case_dir_str)
    _, mode_shapes = _load_mode_shapes_cached(case_dir_str, columns, mode_count_limit, normalization)
    values = mode_shapes.numpy()
    node_count = int(values.shape[0])
    src, dst, dist = _load_edge_index_cached(case_dir_str)
    if src.size == 0:
        empty = torch.zeros((node_count, int(values.shape[1])), dtype=torch.float32)
        return empty, empty, empty, empty

    u1 = _mode_column_index(columns, "U1")
    u2 = _mode_column_index(columns, "U2")
    u3 = _mode_column_index(columns, "U3")
    umag_idx = _mode_column_index(columns, "U_mag")

    umag = np.maximum(values[:, :, umag_idx], 0.0)
    umag_edge = np.abs(umag[src] - umag[dst]).astype(np.float32, copy=False)
    umag_mean, umag_max = _aggregate_edge_gradient(umag_edge, src=src, dst=dst, dist=dist, node_count=node_count)

    displacement = values[:, :, [u1, u2, u3]]
    vector_edge = np.sqrt(
        np.sum(np.square(displacement[src] - displacement[dst], dtype=np.float64), axis=-1)
    ).astype(np.float32, copy=False)
    vector_mean, vector_max = _aggregate_edge_gradient(vector_edge, src=src, dst=dst, dist=dist, node_count=node_count)

    return (
        torch.tensor(umag_mean, dtype=torch.float32),
        torch.tensor(umag_max, dtype=torch.float32),
        torch.tensor(vector_mean, dtype=torch.float32),
        torch.tensor(vector_max, dtype=torch.float32),
    )


def _load_mode_edge_gradients(
    case_dir: Path,
    columns: Sequence[str],
    mode_count_limit: int | None,
    normalization: str,
) -> dict[str, torch.Tensor]:
    mode_columns = tuple(str(column) for column in columns)
    umag_mean, umag_max, vector_mean, vector_max = _load_mode_edge_gradients_cached(
        str(case_dir.resolve()),
        mode_columns,
        mode_count_limit,
        normalization,
    )
    return {
        "grad_umag_mean": umag_mean,
        "grad_umag_max": umag_max,
        "grad_vector_mean": vector_mean,
        "grad_vector_max": vector_max,
    }


def _modal_response_weights(
    frequency_hz: float,
    modal_frequencies: torch.Tensor,
    damping_ratio: float,
    weighting: str,
    modal_frequency_power: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frequency = torch.tensor(float(frequency_hz), dtype=modal_frequencies.dtype)
    modal_frequencies = modal_frequencies.clamp_min(1e-6)
    safe_frequency = frequency.clamp_min(1e-6)
    log_gap = torch.abs(torch.log(safe_frequency / modal_frequencies))
    weighting = weighting.lower()
    if weighting == "resonance":
        ratio = safe_frequency / modal_frequencies
        damping = max(float(damping_ratio), 1e-6)
        weights = torch.rsqrt((1.0 - ratio.pow(2)).pow(2) + (2.0 * damping * ratio).pow(2) + 1e-12)
        frequency_power = max(float(modal_frequency_power), 0.0)
        if frequency_power > 0.0:
            weights = weights / modal_frequencies.pow(frequency_power)
    elif weighting == "log_gaussian":
        sigma = max(float(damping_ratio), 1e-6)
        weights = torch.exp(-0.5 * (log_gap / sigma).pow(2))
    elif weighting == "inverse_log_gap":
        weights = 1.0 / (log_gap + max(float(damping_ratio), 1e-6))
    else:
        raise ValueError(f"Unsupported mode_shape_weighting: {weighting}")
    weights = weights / weights.sum().clamp_min(1e-12)
    nearest_index = torch.argmin(log_gap)
    return weights, nearest_index, log_gap


def _modal_frf_amplitude(
    frequency_hz: float,
    modal_frequencies: torch.Tensor,
    damping_ratio: float,
    amp_clip: float | None = None,
) -> torch.Tensor:
    frequency = torch.tensor(float(frequency_hz), dtype=modal_frequencies.dtype)
    modal_frequencies = modal_frequencies.clamp_min(1e-6)
    ratio = frequency.clamp_min(1e-6) / modal_frequencies
    damping = max(float(damping_ratio), 1e-6)
    amplitude = torch.rsqrt((1.0 - ratio.pow(2)).pow(2) + (2.0 * damping * ratio).pow(2) + 1e-12)
    if amp_clip is not None and float(amp_clip) > 0.0:
        amplitude = amplitude.clamp_max(float(amp_clip))
    return amplitude


def _modal_frf_gain(
    frequency_hz: float,
    modal_frequencies: torch.Tensor,
    damping_ratio: float,
    amp_clip: float | None,
    gain_power: float,
    gain_clip: float | None,
) -> torch.Tensor:
    amplitude = _modal_frf_amplitude(
        frequency_hz=frequency_hz,
        modal_frequencies=modal_frequencies,
        damping_ratio=damping_ratio,
        amp_clip=amp_clip,
    )
    power = max(float(gain_power), 1e-6)
    gain = amplitude.pow(power)
    if gain_clip is not None and float(gain_clip) > 0.0:
        gain = gain.clamp_max(float(gain_clip))
    return gain


def _modal_frf_weight_and_topk(
    frequency_hz: float,
    modal_frequencies: torch.Tensor,
    feature_cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    damping = float(feature_cfg.get("modal_frf_shape_damping_ratio", feature_cfg.get("mode_shape_damping_ratio", 0.02)))
    amp_clip_raw = feature_cfg.get("modal_frf_shape_amp_clip", feature_cfg.get("modal_frf_amp_clip", 1000.0))
    gain_clip_raw = feature_cfg.get("modal_frf_shape_gain_clip", None)
    gain_power = float(feature_cfg.get("modal_frf_shape_gain_power", 2.0))
    topk = max(1, int(feature_cfg.get("modal_frf_shape_topk", feature_cfg.get("modal_frf_topk", 3))))
    gain = _modal_frf_gain(
        frequency_hz=frequency_hz,
        modal_frequencies=modal_frequencies,
        damping_ratio=damping,
        amp_clip=None if amp_clip_raw is None else float(amp_clip_raw),
        gain_power=gain_power,
        gain_clip=None if gain_clip_raw is None else float(gain_clip_raw),
    )
    weight = gain / gain.sum().clamp_min(1e-12)
    active_indices = torch.topk(gain, k=min(topk, int(gain.numel()))).indices
    return gain, weight, active_indices


def _damping_feature_tag(damping_ratio: float) -> str:
    return f"zeta{float(damping_ratio):.4g}".replace(".", "p").replace("-", "m")


def _build_modal_response_features(
    frequency_hz: float,
    modal_frequencies: torch.Tensor,
    nearest_index: torch.Tensor,
    feature_cfg: dict[str, Any],
) -> tuple[torch.Tensor, list[str]]:
    parts: list[torch.Tensor] = []
    names: list[str] = []
    modal_frequencies = modal_frequencies.clamp_min(1e-6)
    mode_count = int(modal_frequencies.numel())
    safe_frequency = torch.tensor(float(frequency_hz), dtype=modal_frequencies.dtype).clamp_min(1e-6)

    if bool(feature_cfg.get("include_modal_ratio_features", False)):
        ratio = safe_frequency / modal_frequencies
        detuning = 1.0 - ratio.pow(2)
        parts.extend([ratio, detuning])
        names.extend([f"freq_ratio_mode_{idx + 1}" for idx in range(mode_count)])
        names.extend([f"modal_detuning_mode_{idx + 1}" for idx in range(mode_count)])
        nearest = int(nearest_index.item())
        parts.extend([ratio[nearest].reshape(1), detuning[nearest].reshape(1)])
        names.extend(["nearest_freq_ratio", "nearest_modal_detuning"])

    if bool(feature_cfg.get("include_modal_frf_features", False)):
        damping_values = feature_cfg.get("modal_frf_damping_ratios", [feature_cfg.get("mode_shape_damping_ratio", 0.02)])
        topk = max(1, int(feature_cfg.get("modal_frf_topk", 3)))
        amp_clip_raw = feature_cfg.get("modal_frf_amp_clip", 1000.0)
        amp_clip = None if amp_clip_raw is None else float(amp_clip_raw)
        for raw_damping in damping_values:
            damping = float(raw_damping)
            tag = _damping_feature_tag(damping)
            amplitude = _modal_frf_amplitude(
                frequency_hz=frequency_hz,
                modal_frequencies=modal_frequencies,
                damping_ratio=damping,
                amp_clip=amp_clip,
            )
            modal_weight = amplitude / amplitude.sum().clamp_min(1e-12)
            log_amplitude = torch.log1p(amplitude)
            nearest = int(nearest_index.item())
            parts.extend(
                [
                    log_amplitude,
                    modal_weight,
                    log_amplitude[nearest].reshape(1),
                    torch.log1p(amplitude.sum()).reshape(1),
                    torch.log1p(torch.topk(amplitude, k=min(topk, mode_count)).values.sum()).reshape(1),
                    modal_weight[nearest].reshape(1),
                ]
            )
            names.extend([f"log_modal_amp_{tag}_mode_{idx + 1}" for idx in range(mode_count)])
            names.extend([f"modal_weight_{tag}_mode_{idx + 1}" for idx in range(mode_count)])
            names.extend(
                [
                    f"nearest_log_modal_amp_{tag}",
                    f"sum_log_modal_amp_{tag}",
                    f"top{min(topk, mode_count)}_log_modal_amp_sum_{tag}",
                    f"nearest_modal_weight_{tag}",
                ]
            )

    if not parts:
        return torch.empty((0,), dtype=torch.float32), []
    return torch.cat([part.reshape(-1) for part in parts], dim=0).to(dtype=torch.float32), names


def _mode_column_index(columns: tuple[str, ...], name: str) -> int:
    try:
        return columns.index(name)
    except ValueError as exc:
        raise ValueError(f"Mode shape feature requires column {name}") from exc


def _build_mode_shape_features(
    case_dir: Path,
    selected_indices: torch.Tensor,
    frequency_hz: float,
    feature_cfg: dict[str, Any],
    psd_value_at_frequency: float = 0.0,
) -> tuple[torch.Tensor, list[str]]:
    if not bool(feature_cfg.get("use_mode_shapes", True)):
        return torch.empty((selected_indices.numel(), 0), dtype=torch.float32), []

    modal_frequencies, mode_shapes, mode_columns = _load_mode_shapes(
        case_dir,
        columns=feature_cfg.get("mode_shape_columns", list(DEFAULT_MODE_SHAPE_COLUMNS)),
        mode_count_limit=feature_cfg.get("mode_shape_count", 10),
        normalization=str(feature_cfg.get("mode_shape_normalization", "max_umag")),
    )
    mode_shapes = mode_shapes[selected_indices]
    weights, nearest_index, log_gap = _modal_response_weights(
        frequency_hz=frequency_hz,
        modal_frequencies=modal_frequencies,
        damping_ratio=float(feature_cfg.get("mode_shape_damping_ratio", 0.02)),
        weighting=str(feature_cfg.get("mode_shape_weighting", "resonance")),
        modal_frequency_power=float(feature_cfg.get("mode_shape_modal_frequency_power", 2.0)),
    )

    u1 = _mode_column_index(mode_columns, "U1")
    u2 = _mode_column_index(mode_columns, "U2")
    u3 = _mode_column_index(mode_columns, "U3")
    umag_idx = _mode_column_index(mode_columns, "U_mag")
    abs_xyz = mode_shapes[:, :, [u1, u2, u3]].abs()
    umag = mode_shapes[:, :, umag_idx : umag_idx + 1].clamp_min(0.0)

    weighted_abs_xyz = (abs_xyz * weights.view(1, -1, 1)).sum(dim=1)
    weighted_umag = (umag * weights.view(1, -1, 1)).sum(dim=1)
    nearest_abs_xyz = abs_xyz[:, int(nearest_index.item()), :]
    nearest_umag = umag[:, int(nearest_index.item()), :]
    node_count = selected_indices.numel()
    nearest_gap = log_gap[nearest_index].view(1, 1).expand(node_count, 1)
    nearest_weight = weights[nearest_index].view(1, 1).expand(node_count, 1)
    features = torch.cat(
        [
            weighted_abs_xyz,
            weighted_umag,
            nearest_abs_xyz,
            nearest_umag,
            nearest_gap,
            nearest_weight,
        ],
        dim=-1,
    )
    names = [
        "weighted_abs_u1",
        "weighted_abs_u2",
        "weighted_abs_u3",
        "weighted_umag",
        "nearest_abs_u1",
        "nearest_abs_u2",
        "nearest_abs_u3",
        "nearest_umag",
        "nearest_gap",
        "nearest_weight",
    ]

    frf_gain: torch.Tensor | None = None
    frf_weight: torch.Tensor | None = None
    active_indices: torch.Tensor | None = None

    def get_frf_state() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        nonlocal frf_gain, frf_weight, active_indices
        if frf_gain is None or frf_weight is None or active_indices is None:
            frf_gain, frf_weight, active_indices = _modal_frf_weight_and_topk(
                frequency_hz=frequency_hz,
                modal_frequencies=modal_frequencies,
                feature_cfg=feature_cfg,
            )
        return frf_gain, frf_weight, active_indices

    if bool(feature_cfg.get("include_modal_frf_shape_features", False)):
        gain, frf_weights, active = get_frf_state()
        weighted_abs_xyz_frf = (abs_xyz * frf_weights.view(1, -1, 1)).sum(dim=1)
        weighted_umag_frf = (umag * frf_weights.view(1, -1, 1)).sum(dim=1)
        features = torch.cat(
            [
                features,
                weighted_abs_xyz_frf,
                weighted_umag_frf,
            ],
            dim=-1,
        )
        names.extend(
            [
                "weighted_abs_u1_frf",
                "weighted_abs_u2_frf",
                "weighted_abs_u3_frf",
                "weighted_umag_frf",
            ]
        )
        safe_frequency = torch.tensor(float(frequency_hz), dtype=modal_frequencies.dtype).clamp_min(1e-6)
        for rank, mode_idx_tensor in enumerate(active, start=1):
            mode_idx = int(mode_idx_tensor.item())
            log_gain = torch.log1p(gain[mode_idx]).view(1, 1).expand(node_count, 1)
            mode_weight = frf_weights[mode_idx].view(1, 1).expand(node_count, 1)
            freq_ratio = (safe_frequency / modal_frequencies[mode_idx].clamp_min(1e-6)).view(1, 1).expand(node_count, 1)
            features = torch.cat(
                [
                    features,
                    abs_xyz[:, mode_idx, :],
                    umag[:, mode_idx, :],
                    log_gain,
                    mode_weight,
                    freq_ratio,
                ],
                dim=-1,
            )
            names.extend(
                [
                    f"active{rank}_abs_u1_frf",
                    f"active{rank}_abs_u2_frf",
                    f"active{rank}_abs_u3_frf",
                    f"active{rank}_umag_frf",
                    f"active{rank}_log_modal_gain_frf",
                    f"active{rank}_modal_weight_frf",
                    f"active{rank}_freq_ratio_frf",
                ]
            )

    gradient_features: dict[str, torch.Tensor] | None = None

    def get_gradient_features() -> dict[str, torch.Tensor]:
        nonlocal gradient_features
        if gradient_features is None:
            gradient_features = _load_mode_edge_gradients(
                case_dir,
                columns=feature_cfg.get("mode_shape_columns", list(DEFAULT_MODE_SHAPE_COLUMNS)),
                mode_count_limit=feature_cfg.get("mode_shape_count", 10),
                normalization=str(feature_cfg.get("modal_gradient_normalization", feature_cfg.get("mode_shape_normalization", "max_umag"))),
            )
        return gradient_features

    if bool(feature_cfg.get("include_modal_gradient_features", False)):
        _, frf_weights, active = get_frf_state()
        gradients = get_gradient_features()
        raw_gradient_keys = feature_cfg.get(
            "modal_gradient_keys",
            ["grad_umag_mean", "grad_umag_max", "grad_vector_mean", "grad_vector_max"],
        )
        gradient_keys = [str(key) for key in raw_gradient_keys]
        for key in gradient_keys:
            if key not in gradients:
                raise ValueError(f"Unsupported modal_gradient_key: {key}")
            values = gradients[key][selected_indices]
            weighted = (values * frf_weights.view(1, -1)).sum(dim=1, keepdim=True)
            nearest = values[:, int(nearest_index.item())].reshape(node_count, 1)
            feature_parts = [weighted, nearest]
            feature_names = [f"weighted_{key}_frf", f"nearest_{key}"]
            for rank, mode_idx_tensor in enumerate(active, start=1):
                mode_idx = int(mode_idx_tensor.item())
                feature_parts.append(values[:, mode_idx].reshape(node_count, 1))
                feature_names.append(f"active{rank}_{key}_frf")
            features = torch.cat([features, *feature_parts], dim=-1)
            names.extend(feature_names)

    if bool(feature_cfg.get("include_modal_baseline_feature", False)):
        gain, _, _ = get_frf_state()
        gradients = get_gradient_features()
        baseline_sources: dict[str, torch.Tensor] = {
            "umag": umag.squeeze(-1),
            **{key: value[selected_indices] for key, value in gradients.items()},
        }
        raw_baseline_keys = feature_cfg.get("modal_baseline_keys", ["grad_umag_mean", "grad_umag_max"])
        baseline_keys = [str(key) for key in raw_baseline_keys]
        psd_scale = max(float(psd_value_at_frequency), 0.0) if bool(feature_cfg.get("modal_baseline_use_psd", True)) else 1.0
        psd_scale_tensor = torch.tensor(psd_scale, dtype=torch.float32)
        for key in baseline_keys:
            if key not in baseline_sources:
                raise ValueError(f"Unsupported modal_baseline_key: {key}")
            source = baseline_sources[key].clamp_min(0.0)
            baseline_raw = (source * gain.view(1, -1)).sum(dim=1, keepdim=True) * psd_scale_tensor
            features = torch.cat([features, torch.log1p(baseline_raw.clamp_min(0.0))], dim=-1)
            names.append(f"modal_baseline_log_{key}")

    if bool(feature_cfg.get("include_shape_normalized_mode_features", False)):
        raw_normalizations = feature_cfg.get("mode_shape_extra_normalizations", ["rms_umag"])
        if isinstance(raw_normalizations, str):
            if raw_normalizations.strip().lower() == "both":
                extra_normalizations = ["max_umag", "rms_umag"]
            else:
                extra_normalizations = [raw_normalizations]
        else:
            extra_normalizations = [str(item) for item in raw_normalizations]

        base_normalization = str(feature_cfg.get("mode_shape_normalization", "max_umag"))
        for extra_normalization in extra_normalizations:
            extra_normalization = extra_normalization.strip().lower()
            if extra_normalization == base_normalization:
                continue
            _, extra_mode_shapes, extra_mode_columns = _load_mode_shapes(
                case_dir,
                columns=feature_cfg.get("mode_shape_columns", list(DEFAULT_MODE_SHAPE_COLUMNS)),
                mode_count_limit=feature_cfg.get("mode_shape_count", 10),
                normalization=extra_normalization,
            )
            extra_mode_shapes = extra_mode_shapes[selected_indices]
            extra_u1 = _mode_column_index(extra_mode_columns, "U1")
            extra_u2 = _mode_column_index(extra_mode_columns, "U2")
            extra_u3 = _mode_column_index(extra_mode_columns, "U3")
            extra_umag_idx = _mode_column_index(extra_mode_columns, "U_mag")
            shape_abs_xyz = extra_mode_shapes[:, :, [extra_u1, extra_u2, extra_u3]].abs()
            shape_umag = extra_mode_shapes[:, :, extra_umag_idx : extra_umag_idx + 1].clamp_min(0.0)
            norm_name = extra_normalization.replace("_umag", "")
            features = torch.cat(
                [
                    features,
                    (shape_abs_xyz * weights.view(1, -1, 1)).sum(dim=1),
                    (shape_umag * weights.view(1, -1, 1)).sum(dim=1),
                    shape_abs_xyz[:, int(nearest_index.item()), :],
                    shape_umag[:, int(nearest_index.item()), :],
                ],
                dim=-1,
            )
            names.extend(
                [
                    f"weighted_abs_u1_shape_{norm_name}",
                    f"weighted_abs_u2_shape_{norm_name}",
                    f"weighted_abs_u3_shape_{norm_name}",
                    f"weighted_umag_shape_{norm_name}",
                    f"nearest_abs_u1_shape_{norm_name}",
                    f"nearest_abs_u2_shape_{norm_name}",
                    f"nearest_abs_u3_shape_{norm_name}",
                    f"nearest_umag_shape_{norm_name}",
                ]
            )

    modal_response_values, modal_response_names = _build_modal_response_features(
        frequency_hz=frequency_hz,
        modal_frequencies=modal_frequencies,
        nearest_index=nearest_index,
        feature_cfg=feature_cfg,
    )
    if modal_response_values.numel() > 0:
        features = torch.cat([features, modal_response_values.unsqueeze(0).expand(node_count, -1)], dim=-1)
        names.extend(modal_response_names)
    return features.to(dtype=torch.float32), names


def _flatten_processed_psd_points(psd_points: Sequence[Sequence[float]]) -> tuple[list[float], list[str], list[tuple[float, float]]]:
    values: list[float] = []
    names: list[str] = []
    value_frequency_pairs: list[tuple[float, float]] = []
    for idx, point in enumerate(psd_points):
        if len(point) >= 1:
            raw_value = max(float(point[0]), 0.0)
            values.append(math.log1p(raw_value))
            names.append(f"psd_{idx + 1}_value_log")
        if len(point) >= 2:
            values.append(float(point[1]))
            names.append(f"psd_{idx + 1}_imag")
        if len(point) >= 3:
            frequency = float(point[2])
            values.append(frequency)
            names.append(f"psd_{idx + 1}_freq")
            value_frequency_pairs.append((frequency, max(float(point[0]), 0.0)))
    return values, names, value_frequency_pairs


def _interpolate_psd_value(psd_pairs: Sequence[tuple[float, float]], frequency_hz: float) -> float:
    if not psd_pairs:
        return 0.0
    pairs = sorted(psd_pairs, key=lambda item: item[0])
    frequencies = np.array([freq for freq, _ in pairs], dtype=np.float64)
    values = np.array([value for _, value in pairs], dtype=np.float64)
    return float(np.interp(float(frequency_hz), frequencies, values))


def _repeat_vector(values: Sequence[float], node_count: int) -> torch.Tensor:
    vector = torch.tensor(list(values), dtype=torch.float32)
    return vector.unsqueeze(0).expand(node_count, -1)


def _build_center_modal_interaction_features(
    *,
    center_couple_mask: torch.Tensor,
    near_center_region_exp: torch.Tensor,
    mode_features: torch.Tensor,
    mode_names: list[str],
) -> tuple[torch.Tensor, list[str]]:
    mode_indices = {name: idx for idx, name in enumerate(mode_names)}
    missing = sorted(
        {
            source_name
            for _, source_name, _ in CENTER_MODAL_INTERACTION_SPECS
            if source_name not in mode_indices
        }
    )
    if missing:
        raise ValueError(
            "Center-modal interaction features require missing modal features: "
            f"{missing}"
        )

    gates = {
        "center_couple_mask": center_couple_mask,
        "near_center_region_exp": near_center_region_exp,
    }
    parts = [
        gates[gate_name] * mode_features[:, mode_indices[source_name] : mode_indices[source_name] + 1]
        for gate_name, source_name, _ in CENTER_MODAL_INTERACTION_SPECS
    ]
    names = [output_name for _, _, output_name in CENTER_MODAL_INTERACTION_SPECS]
    return torch.cat(parts, dim=-1).to(dtype=torch.float32), names


def _build_base_features(
    nodes_df: pd.DataFrame,
    payload: dict[str, Any],
    selected_indices: torch.Tensor,
    frequency_hz: float,
    feature_cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[str], list[str]]:
    fixed_geometry = _load_fixed_geometry(payload)
    params = torch.tensor(payload["params_list"], dtype=torch.float32)
    freq_top3 = torch.tensor(payload.get("freq_top3", payload.get("frequencies", [0.0, 0.0, 0.0])[:3]), dtype=torch.float32)
    if freq_top3.numel() < 3:
        freq_top3 = torch.cat([freq_top3, torch.zeros(3 - freq_top3.numel(), dtype=torch.float32)], dim=0)
    freq_top3 = freq_top3[:3]

    x_all = torch.tensor(nodes_df["x"].to_numpy(dtype=np.float32), dtype=torch.float32)
    y_all = torch.tensor(nodes_df["y"].to_numpy(dtype=np.float32), dtype=torch.float32)
    z_all = torch.tensor(nodes_df["z"].to_numpy(dtype=np.float32), dtype=torch.float32)
    x = x_all[selected_indices].unsqueeze(-1)
    y = y_all[selected_indices].unsqueeze(-1)
    z = z_all[selected_indices].unsqueeze(-1)
    node_count = selected_indices.numel()

    plate_radius = torch.tensor(float(params[6].item()), dtype=torch.float32).clamp_min(1e-6)
    plate_thickness = torch.tensor(float(fixed_geometry["plate_thickness"]), dtype=torch.float32).clamp_min(1e-6)
    earpiece_radial_dist = torch.tensor(float(params[1].item()), dtype=torch.float32).clamp_min(1e-6)
    earpiece_width = torch.tensor(float(params[2].item()), dtype=torch.float32).clamp_min(1e-6)
    earpiece_hole_radius = torch.tensor(float(fixed_geometry["earpiece_HoleRadius"]), dtype=torch.float32).clamp_min(1e-6)
    earpiece_count = max(1, int(float(fixed_geometry["earpiece_Count_default"])))
    mass_couple_radius = torch.tensor(float(fixed_geometry["mass_couple_radius"]), dtype=torch.float32).clamp_min(1e-6)
    plate_hole_radius = torch.tensor(float(fixed_geometry["plate_HoleRadius"]), dtype=torch.float32).clamp_min(1e-6)
    plate_hole_dist = torch.tensor(float(fixed_geometry["plate_HoleDist"]), dtype=torch.float32).clamp_min(1e-6)
    plate_hole_count = max(1, int(float(fixed_geometry["plate_HoleCount"])))

    r = torch.sqrt(x.pow(2) + y.pow(2) + 1e-12)
    theta = torch.atan2(y, x)

    angles = torch.linspace(0.0, 2.0 * math.pi, steps=earpiece_count + 1, dtype=torch.float32)[:-1]
    centers = torch.stack(
        [-earpiece_radial_dist * torch.sin(angles), earpiece_radial_dist * torch.cos(angles)],
        dim=-1,
    )
    axial_axes = torch.stack([-torch.sin(angles), torch.cos(angles)], dim=-1)
    tangent_axes = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    node_xy = torch.cat([x, y], dim=-1)
    distance_to_centers = torch.cdist(node_xy, centers)
    nearest_ear_index = distance_to_centers.argmin(dim=1)
    dist_to_earpiece_center = distance_to_centers.gather(1, nearest_ear_index.unsqueeze(-1))
    dist_to_ear_hole_edge = dist_to_earpiece_center - earpiece_hole_radius
    center_couple_signed = (mass_couple_radius - r) / mass_couple_radius

    geometry_parts = [
        x / plate_radius,
        y / plate_radius,
        z / plate_thickness,
        r / plate_radius,
        (plate_radius - r) / plate_radius,
        torch.sin(theta),
        torch.cos(theta),
        dist_to_ear_hole_edge / earpiece_hole_radius,
        dist_to_ear_hole_edge / plate_radius,
        earpiece_width / plate_radius * torch.ones_like(x),
        earpiece_width / earpiece_hole_radius * torch.ones_like(x),
        r / mass_couple_radius,
        center_couple_signed,
    ]
    geometry_names = list(BASE_GEOMETRY_FEATURE_NAMES)
    near_center_region_exp: torch.Tensor | None = None

    if bool(feature_cfg.get("include_disk_center_features", False)):
        center_mask_radius = torch.tensor(_center_couple_mask_radius(payload), dtype=torch.float32).clamp_min(1e-6)
        center_region_signed = (center_mask_radius - r) / center_mask_radius
        center_region_tau = max(float(feature_cfg.get("center_region_tau", 8.0)), 1e-6)
        near_center_region_exp = torch.exp((r - center_mask_radius).clamp_min(0.0).neg() / center_region_tau)
        geometry_parts.extend(
            [
                r / center_mask_radius,
                center_region_signed,
                near_center_region_exp,
            ]
        )
        geometry_names.extend(DISK_CENTER_FEATURE_NAMES)

    if bool(feature_cfg.get("include_plate_hole_features", False)):
        plate_hole_angles = torch.linspace(0.0, 2.0 * math.pi, steps=plate_hole_count + 1, dtype=torch.float32)[:-1]
        plate_hole_centers = torch.stack(
            [
                -plate_hole_dist * torch.sin(plate_hole_angles),
                plate_hole_dist * torch.cos(plate_hole_angles),
            ],
            dim=-1,
        )
        dist_to_plate_hole_center = torch.cdist(node_xy, plate_hole_centers).min(dim=1, keepdim=True).values
        dist_to_plate_hole_edge = dist_to_plate_hole_center - plate_hole_radius
        plate_hole_wall_tau = max(float(feature_cfg.get("plate_hole_wall_tau", 2.0)), 1e-6)
        near_plate_hole_wall = torch.exp(-dist_to_plate_hole_edge.abs() / plate_hole_wall_tau)
        geometry_parts.extend(
            [
                dist_to_plate_hole_center / plate_hole_radius,
                dist_to_plate_hole_edge / plate_hole_radius,
                dist_to_plate_hole_edge / plate_radius,
                near_plate_hole_wall,
            ]
        )
        geometry_names.extend(PLATE_HOLE_FEATURE_NAMES)

    if bool(feature_cfg.get("include_angular_periodic_features", False)):
        ear_period = torch.tensor(float(earpiece_count), dtype=torch.float32)
        plate_hole_period = torch.tensor(float(plate_hole_count), dtype=torch.float32)
        geometry_parts.extend(
            [
                torch.sin(ear_period * theta),
                torch.cos(ear_period * theta),
                torch.sin(plate_hole_period * theta),
                torch.cos(plate_hole_period * theta),
            ]
        )
        geometry_names.extend(ANGULAR_PERIODIC_FEATURE_NAMES)

    if bool(feature_cfg.get("include_earpiece_local_features", False)):
        nearest_axial = axial_axes[nearest_ear_index]
        nearest_tangent = tangent_axes[nearest_ear_index]
        local_u_absolute = (node_xy * nearest_axial).sum(dim=1, keepdim=True)
        local_v = (node_xy * nearest_tangent).sum(dim=1, keepdim=True)
        local_u = local_u_absolute - earpiece_radial_dist
        local_r = torch.sqrt(local_u.pow(2) + local_v.pow(2) + 1e-12)
        local_angle = torch.atan2(local_v, local_u)

        half_width = (0.5 * earpiece_width).clamp_min(1e-6)
        root_signed = local_u_absolute - plate_radius
        root_abs = root_signed.abs()
        bottom_fillet = float(params[5].item()) if params.numel() > 5 else 0.0
        root_band_default = max(float(earpiece_hole_radius.item()), bottom_fillet, 0.15 * float(earpiece_width.item()))
        root_band = max(float(feature_cfg.get("earpiece_root_band", root_band_default)), 1e-6)
        bridge_tau = max(float(feature_cfg.get("earpiece_bridge_tau", 0.5 * float(earpiece_hole_radius.item()))), 1e-6)
        bridge_width_default = max(float(earpiece_hole_radius.item()), 0.35 * float(earpiece_width.item()))
        bridge_width = max(float(feature_cfg.get("earpiece_bridge_width", bridge_width_default)), 1e-6)
        between_root_and_hole = torch.sigmoid(root_signed / bridge_tau) * torch.sigmoid(
            (earpiece_radial_dist - local_u_absolute) / bridge_tau
        )
        near_hole_root_bridge = between_root_and_hole * torch.exp(-local_v.abs() / bridge_width)

        geometry_parts.extend(
            [
                local_u / earpiece_hole_radius,
                local_v / half_width,
                local_r / earpiece_hole_radius,
                torch.sin(local_angle),
                torch.cos(local_angle),
                local_v.abs() / half_width,
                root_signed / earpiece_hole_radius,
                root_abs / earpiece_hole_radius,
                torch.exp(-root_abs / root_band),
                dist_to_earpiece_center / earpiece_hole_radius,
                near_hole_root_bridge,
            ]
        )
        geometry_names.extend(EARPIECE_LOCAL_FEATURE_NAMES)

    if bool(feature_cfg.get("include_stress_region_distance_features", False)):
        distances = _load_stress_region_distances_cached(str(Path(str(payload.get("__case_dir__", ""))).resolve()))
        distance_parts = [
            distances["center_couple_region"][selected_indices].unsqueeze(-1) / plate_radius,
            distances["plate_hole_region"][selected_indices].unsqueeze(-1) / plate_radius,
            distances["ear_hole_region"][selected_indices].unsqueeze(-1) / plate_radius,
            distances["ear_connection_region"][selected_indices].unsqueeze(-1) / plate_radius,
            distances["nearest_stress_region"][selected_indices].unsqueeze(-1) / plate_radius,
        ]
        geometry_parts.extend(distance_parts)
        geometry_names.extend(STRESS_REGION_DISTANCE_FEATURE_NAMES)

    geometry = torch.cat(
        geometry_parts,
        dim=-1,
    ).to(dtype=torch.float32)

    if "bc_mask" in nodes_df.columns:
        bc_mask_all = torch.tensor(nodes_df["bc_mask"].to_numpy(dtype=np.float32), dtype=torch.float32)
    else:
        bc_mask_all = torch.tensor(_build_boundary_mask(nodes_df, payload), dtype=torch.float32)
    bc_mask = bc_mask_all[selected_indices].unsqueeze(-1)
    near_ear_hole = torch.exp(-dist_to_ear_hole_edge.clamp_min(0.0) / earpiece_hole_radius)
    center_couple_sigmoid_gain = float(feature_cfg.get("center_couple_sigmoid_gain", 5.0))
    near_center_couple = torch.sigmoid(center_couple_sigmoid_gain * center_couple_signed)
    mask_parts = [bc_mask, near_ear_hole, near_center_couple]
    mask_names = list(MASK_FEATURE_NAMES)
    if bool(feature_cfg.get("include_node_region_masks", False)):
        for column in NODE_REGION_MASK_COLUMNS:
            if column in nodes_df.columns:
                values = torch.tensor(nodes_df[column].to_numpy(dtype=np.float32), dtype=torch.float32)
            else:
                values = torch.zeros(len(nodes_df), dtype=torch.float32)
            mask_parts.append(values[selected_indices].unsqueeze(-1))
            mask_names.append(column)
        stress_mask = torch.zeros(len(nodes_df), dtype=torch.float32)
        for column in STRESS_REGION_MASK_COLUMNS:
            if column in nodes_df.columns:
                stress_mask = torch.maximum(
                    stress_mask,
                    torch.tensor(nodes_df[column].to_numpy(dtype=np.float32), dtype=torch.float32),
                )
        mask_parts.append(stress_mask[selected_indices].unsqueeze(-1))
        mask_names.append("stress_region_mask")
    masks = torch.cat(mask_parts, dim=-1).to(dtype=torch.float32)

    scaled_parts: list[torch.Tensor] = []
    scaled_names: list[str] = []

    scaled_parts.append(params.unsqueeze(0).expand(node_count, -1))
    scaled_names.extend(
        [
            "earpiece_thickness",
            "earpiece_RadialDist",
            "earpiece_TopWidth",
            "earpiece_HoleTopDist",
            "earpiece_TopFilletRadius",
            "earpiece_BottomFilletRadius",
            "plate_radius",
            "Add_mass",
        ]
    )
    fixed_values = [fixed_geometry["plate_thickness"], fixed_geometry["earpiece_HoleRadius"], fixed_geometry["mass_couple_radius"]]
    scaled_parts.append(_repeat_vector(fixed_values, node_count))
    scaled_names.extend(["plate_thickness", "earpiece_HoleRadius", "mass_couple_radius"])

    if bool(feature_cfg.get("use_psd", True)):
        psd_values, psd_names, psd_pairs = _flatten_processed_psd_points(payload.get("psd_points", []))
        if psd_values:
            scaled_parts.append(_repeat_vector(psd_values, node_count))
            scaled_names.extend(psd_names)
        psd_at_frequency = _interpolate_psd_value(psd_pairs, frequency_hz)
        if bool(feature_cfg.get("include_psd_value_at_frequency", True)):
            scaled_parts.append(_repeat_vector([max(psd_at_frequency, 0.0)], node_count))
            scaled_names.append("psd_value_at_frequency")
        if bool(feature_cfg.get("include_log_psd_value_at_frequency", True)):
            scaled_parts.append(_repeat_vector([math.log1p(max(psd_at_frequency, 0.0))], node_count))
            scaled_names.append("log_psd_value_at_frequency")
    else:
        psd_at_frequency = 0.0

    current_frequency = torch.tensor([float(frequency_hz)], dtype=torch.float32)
    safe_frequency = current_frequency.clamp_min(1e-6)
    modes = freq_top3.clamp_min(1e-6)
    signed_delta = (safe_frequency - modes) / modes
    abs_delta = signed_delta.abs()
    frequency_values = torch.cat(
        [
            current_frequency,
            safe_frequency.log(),
            freq_top3,
            signed_delta,
            abs_delta,
            abs_delta.min().reshape(1),
            safe_frequency / modes[:1],
        ],
        dim=0,
    )
    scaled_parts.append(frequency_values.unsqueeze(0).expand(node_count, -1))
    scaled_names.extend(
        [
            "frequency",
            "log_frequency",
            "freq_top1",
            "freq_top2",
            "freq_top3",
            "signed_delta_to_mode_1",
            "signed_delta_to_mode_2",
            "signed_delta_to_mode_3",
            "abs_delta_to_mode_1",
            "abs_delta_to_mode_2",
            "abs_delta_to_mode_3",
            "nearest_delta",
            "first_mode_ratio",
        ]
    )

    mode_features, mode_names = _build_mode_shape_features(
        case_dir=Path(str(payload.get("__case_dir__", ""))),
        selected_indices=selected_indices,
        frequency_hz=frequency_hz,
        feature_cfg=feature_cfg,
        psd_value_at_frequency=psd_at_frequency,
    )
    if mode_names:
        scaled_parts.append(mode_features)
        scaled_names.extend(mode_names)

    if bool(feature_cfg.get("include_center_modal_interaction_features", False)):
        if near_center_region_exp is None:
            raise ValueError(
                "Center-modal interaction features require features.include_disk_center_features=true."
            )
        if "center_couple_mask" not in nodes_df.columns:
            raise ValueError(
                "Center-modal interaction features require center_couple_mask in nodes.csv."
            )
        center_couple_mask = torch.tensor(
            nodes_df["center_couple_mask"].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )[selected_indices].unsqueeze(-1).clamp(0.0, 1.0)
        interaction_features, interaction_names = _build_center_modal_interaction_features(
            center_couple_mask=center_couple_mask,
            near_center_region_exp=near_center_region_exp,
            mode_features=mode_features,
            mode_names=mode_names,
        )
        scaled_parts.append(interaction_features)
        scaled_names.extend(interaction_names)

    scaled = torch.cat(scaled_parts, dim=-1).to(dtype=torch.float32)
    return geometry, scaled, masks, geometry_names, scaled_names, mask_names


def load_raw_point_sample(
    sample_path: str | Path,
    dataset_cfg: dict[str, Any],
    feature_cfg: dict[str, Any],
) -> RawPointSample:
    target_path = Path(sample_path)
    if target_path.parent.name != PER_FREQUENCY_DIRNAME:
        raise ValueError(f"Expected a per-frequency target file, got {target_path}")
    case_dir = target_path.parent.parent
    frequency_hz = _frequency_from_path(target_path)

    nodes_df, payload, earpiece_mask = _load_case_static(
        case_dir,
        region_cfg=dataset_cfg.get("earpiece_region"),
    )
    payload["__case_dir__"] = str(case_dir)
    target_df = pd.read_csv(
        target_path,
        usecols=lambda column: column in {"node_index", PER_FREQUENCY_TARGET_COLUMN},
    )
    target_values = _load_aligned_target_column(
        target_df,
        target_column=PER_FREQUENCY_TARGET_COLUMN,
        nodes_df=nodes_df,
        target_path=target_path,
    )

    selected_mask = build_node_selection_mask(case_dir, dataset_cfg=dataset_cfg, target_values=target_values)
    selected_indices_np = np.flatnonzero(selected_mask).astype(np.int64, copy=False)
    selected_indices = torch.tensor(selected_indices_np, dtype=torch.long)
    target_raw = torch.tensor(target_values[selected_indices_np], dtype=torch.float32)
    target_log = torch.log1p(target_raw.clamp_min(0.0)).unsqueeze(-1)
    selected_earpiece_mask = earpiece_mask[selected_indices_np].astype(bool, copy=False)
    disk_center_mask = _build_disk_center_region_mask(nodes_df)[selected_indices_np]
    region_masks = {
        "fullpart_region": torch.ones(selected_indices.numel(), dtype=torch.bool),
        "earpiece_region": torch.tensor(selected_earpiece_mask, dtype=torch.bool),
        "disk_region": torch.tensor(~selected_earpiece_mask, dtype=torch.bool),
        "disk_center_region": torch.tensor(disk_center_mask, dtype=torch.bool),
    }

    geometry, scaled, masks, geometry_names, scaled_names, mask_names = _build_base_features(
        nodes_df=nodes_df,
        payload=payload,
        selected_indices=selected_indices,
        frequency_hz=frequency_hz,
        feature_cfg=feature_cfg,
    )
    sample_name = f"{case_dir.name}/{target_path.name}"
    return RawPointSample(
        name=sample_name,
        case_name=case_dir.name,
        frequency_hz=frequency_hz,
        geometry_features=geometry,
        scaled_features=scaled,
        mask_features=masks,
        target_log=target_log,
        target_raw=target_raw,
        node_indices=selected_indices,
        geometry_feature_names=geometry_names,
        scaled_feature_names=scaled_names,
        mask_feature_names=mask_names,
        region_masks=region_masks,
    )
