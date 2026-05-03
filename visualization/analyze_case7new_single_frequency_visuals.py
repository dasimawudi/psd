from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import argparse
import json
import math
import struct

import numpy as np
import pandas as pd

from visualize_nodes import _project_points, _write_projection_svg


DEFAULT_VIEWS = ("xy", "xz", "yz")
DEFAULT_OUTPUT_DIR = Path("outputs") / "visualizations" / "case7new_single_frequency_visual_check"


@dataclass
class SelectedFrame:
    label: str
    file_name: str
    frequency_hz: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze and visualize single-frequency case7new MISES PSD slices."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("case7new"),
        help="Root directory containing case*_mises_full_export folders.",
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        default=None,
        help="Optional explicit case directory. Defaults to the case with the largest single-frame peak.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for reports and SVG outputs.",
    )
    return parser.parse_args()


def _safe_slug(text: str) -> str:
    keep = []
    for char in text:
        if char.isalnum():
            keep.append(char.lower())
        elif char in {"-", "_"}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def _format_num(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        return "n/a"
    if abs(value) >= 1e6 or (abs(value) > 0 and abs(value) < 1e-3):
        return f"{value:.3e}"
    return f"{value:.6g}"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def select_representative_case(dataset_root: Path) -> Path:
    best_case: Path | None = None
    best_peak = float("-inf")
    for case_dir in sorted(dataset_root.glob("case*_mises_full_export")):
        summary_path = case_dir / "summary.json"
        if not summary_path.exists():
            continue
        summary = _load_json(summary_path)
        frames = summary.get("frames", [])
        for frame in frames:
            freq = float(frame.get("frequency_hz", 0.0))
            peak = float(frame.get("mises_max", float("-inf")))
            if freq >= 20.0 and peak > best_peak:
                best_peak = peak
                best_case = case_dir
    if best_case is None:
        raise FileNotFoundError(f"No complete case folders found under {dataset_root}")
    return best_case


def scan_frame_stats(case_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    frame_dir = case_dir / "per_frequency_mises"
    for frame_path in sorted(frame_dir.glob("frame_*.csv")):
        frame = pd.read_csv(frame_path, usecols=["MISES_psd_density"])
        values = pd.to_numeric(frame["MISES_psd_density"], errors="coerce").dropna()
        rows.append(
            {
                "file_name": frame_path.name,
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "q99": float(values.quantile(0.99)),
                "q999": float(values.quantile(0.999)),
                "negative_count": int((values < 0).sum()),
            }
        )
    stats = pd.DataFrame(rows)
    frequency_map = pd.read_csv(case_dir / "frequencies.csv")[
        ["export_frame_index", "frequency_hz"]
    ].copy()
    frequency_map["file_name"] = frequency_map["export_frame_index"].map(
        lambda idx: f"frame_{int(idx):04d}_{float(frequency_map.loc[frequency_map['export_frame_index'] == idx, 'frequency_hz'].iloc[0]):010.4f}Hz.csv"
    )
    merged = frequency_map.merge(stats, on="file_name", how="inner")
    return merged.sort_values("export_frame_index").reset_index(drop=True)


def _closest_frame(stats: pd.DataFrame, target_frequency_hz: float) -> pd.Series:
    idx = (stats["frequency_hz"] - float(target_frequency_hz)).abs().idxmin()
    return stats.loc[idx]


def select_frames(stats: pd.DataFrame, global_payload: dict) -> list[SelectedFrame]:
    chosen: list[SelectedFrame] = []
    seen: set[str] = set()

    def add(label: str, row: pd.Series) -> None:
        file_name = str(row["file_name"])
        if file_name in seen:
            return
        seen.add(file_name)
        chosen.append(
            SelectedFrame(
                label=label,
                file_name=file_name,
                frequency_hz=float(row["frequency_hz"]),
            )
        )

    usable = stats[stats["frequency_hz"] >= 20.0].copy()
    add("low_band", usable.iloc[0])

    freq_top3 = list(global_payload.get("freq_top3", []))
    if freq_top3:
        add("first_mode", _closest_frame(usable, freq_top3[0]))
    if len(freq_top3) >= 2:
        add("second_mode", _closest_frame(usable, freq_top3[1]))

    negative_sorted = usable.sort_values(["negative_count", "min"], ascending=[False, True])
    add("most_negative", negative_sorted.iloc[0])

    add("upper_band", usable.iloc[-1])
    return chosen


def build_geometry_features(nodes: pd.DataFrame, global_payload: dict) -> pd.DataFrame:
    nodes = nodes.copy()
    params = global_payload["params"]
    fixed_geometry = global_payload["fixed_geometry"]
    radial_dist = float(params["earpiece_RadialDist"])
    earpiece_count = int(fixed_geometry["earpiece_Count_default"])
    plate_thickness = float(fixed_geometry["plate_thickness"])

    angles = np.linspace(0.0, 2.0 * math.pi, num=earpiece_count, endpoint=False, dtype=np.float64)
    centers = np.stack(
        [
            -radial_dist * np.sin(angles),
            radial_dist * np.cos(angles),
        ],
        axis=1,
    )
    xy = nodes[["x", "y"]].to_numpy(dtype=np.float64)
    hole_distances = np.sqrt(np.sum((xy[:, None, :] - centers[None, :, :]) ** 2, axis=2))

    nodes["r"] = np.sqrt(np.square(nodes["x"]) + np.square(nodes["y"]))
    nodes["theta_deg"] = np.degrees(np.arctan2(nodes["y"], nodes["x"]))
    nodes["nearest_hole_idx"] = hole_distances.argmin(axis=1)
    nodes["nearest_hole_dist"] = hole_distances.min(axis=1)
    nodes["midplane_dist"] = np.abs(nodes["z"] - plate_thickness / 2.0)
    return nodes


def region_masks(df: pd.DataFrame, global_payload: dict) -> dict[str, pd.Series]:
    params = global_payload["params"]
    fixed_geometry = global_payload["fixed_geometry"]
    plate_radius = float(params["plate_radius"])
    mass_radius = float(fixed_geometry["mass_couple_radius"])
    plate_thickness = float(fixed_geometry["plate_thickness"])

    return {
        "center_core_r_lt_10": df["r"] < 10.0,
        "mass_region_r_lt_65": df["r"] < mass_radius,
        "plate_outer_band": (df["r"] >= plate_radius - 5.0) & (df["r"] <= plate_radius + 5.0),
        "ear_hole_vicinity_lt_6": df["nearest_hole_dist"] < 6.0,
        "ear_hole_band_6_20": (df["nearest_hole_dist"] >= 6.0) & (df["nearest_hole_dist"] < 20.0),
        "far_from_holes_gt_40": df["nearest_hole_dist"] > 40.0,
        "top_or_bottom_surface": (df["z"] < 1.0) | (df["z"] > (plate_thickness - 1.0)),
    }


def summarize_subset(df: pd.DataFrame, value_column: str, global_payload: dict, subset_name: str) -> dict:
    result: dict[str, object] = {
        "count": int(len(df)),
        "subset_name": subset_name,
    }
    if df.empty:
        result["empty"] = True
        return result

    counts = df["nearest_hole_idx"].value_counts().sort_index()
    sector_counts = {str(int(idx)): int(count) for idx, count in counts.items()}
    symmetry_balance = float(counts.min() / counts.max()) if not counts.empty and counts.max() else None

    masks = region_masks(df, global_payload)
    result.update(
        {
            "min": float(df[value_column].min()),
            "max": float(df[value_column].max()),
            "mean": float(df[value_column].mean()),
            "r_stats": {
                "min": float(df["r"].min()),
                "max": float(df["r"].max()),
                "mean": float(df["r"].mean()),
            },
            "nearest_hole_dist_stats": {
                "min": float(df["nearest_hole_dist"].min()),
                "max": float(df["nearest_hole_dist"].max()),
                "mean": float(df["nearest_hole_dist"].mean()),
            },
            "z_stats": {
                "min": float(df["z"].min()),
                "max": float(df["z"].max()),
                "mean": float(df["z"].mean()),
            },
            "sector_counts": sector_counts,
            "symmetry_balance": symmetry_balance,
            "region_counts": {name: int(mask.sum()) for name, mask in masks.items()},
            "region_ratios": {name: float(mask.mean()) for name, mask in masks.items()},
        }
    )
    return result


def merge_frame(nodes: pd.DataFrame, case_dir: Path, file_name: str) -> pd.DataFrame:
    frame = pd.read_csv(case_dir / "per_frequency_mises" / file_name)
    merged = nodes.merge(frame[["node_index", "MISES_psd_density"]], on="node_index", how="inner")
    merged["MISES_psd_density"] = pd.to_numeric(merged["MISES_psd_density"], errors="coerce")
    merged = merged.dropna(subset=["MISES_psd_density"]).reset_index(drop=True)
    return merged


def add_visual_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    frame = frame.copy()
    values = frame["MISES_psd_density"].to_numpy(dtype=np.float64)
    frame["signed_log10_mises"] = np.sign(values) * np.log10(1.0 + np.abs(values))
    q99 = float(np.quantile(values, 0.99))
    q999 = float(np.quantile(values, 0.999))
    frame["overlay_code"] = 0.0
    frame.loc[values >= q99, "overlay_code"] = 1.0
    frame.loc[values >= q999, "overlay_code"] = 2.0
    frame.loc[values < 0.0, "overlay_code"] = -1.0
    return frame, q99, q999


def axis_columns(view: str) -> tuple[str, str]:
    mapping = {
        "xy": ("x", "y"),
        "xz": ("x", "z"),
        "yz": ("y", "z"),
    }
    return mapping[view]


def write_overlay_svg(
    output_path: Path,
    frame: pd.DataFrame,
    title: str,
    subtitle: str,
    view: str,
) -> None:
    x_column, y_column = axis_columns(view)
    width = 1120
    height = 820
    left_margin = 88
    right_margin = 240
    top_margin = 92
    bottom_margin = 82

    x_values = frame[x_column].to_numpy(dtype=np.float64)
    y_values = frame[y_column].to_numpy(dtype=np.float64)
    x_px, y_px = _project_points(
        x_values=x_values,
        y_values=y_values,
        width=width,
        height=height,
        left_margin=left_margin,
        right_margin=right_margin,
        top_margin=top_margin,
        bottom_margin=bottom_margin,
    )

    base_style = 'fill="#c7ced6" fill-opacity="0.28" stroke="none"'
    negative_style = 'fill="#2b6cb0" fill-opacity="0.78" stroke="none"'
    high_style = 'fill="#f59e0b" fill-opacity="0.80" stroke="none"'
    spike_style = 'fill="#dc2626" fill-opacity="0.92" stroke="#111827" stroke-width="0.5"'

    overlay_values = frame["overlay_code"].to_numpy(dtype=np.float64)
    negative_mask = overlay_values < 0
    high_mask = overlay_values >= 1
    spike_mask = overlay_values >= 2

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        f'<text x="{left_margin}" y="38" font-size="24" font-weight="700" fill="#111827">{title}</text>',
        f'<text x="{left_margin}" y="64" font-size="14" fill="#4b5563">{subtitle}</text>',
        f'<rect x="{left_margin}" y="{top_margin}" width="{width - left_margin - right_margin}" height="{height - top_margin - bottom_margin}" fill="#fafafa" stroke="#d1d5db" />',
    ]

    for x, y in zip(x_px, y_px):
        parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.05" {base_style} />')

    def draw_subset(mask: np.ndarray, radius: float, style: str) -> None:
        for x, y in zip(x_px[mask], y_px[mask]):
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" {style} />')

    draw_subset(negative_mask, radius=1.5, style=negative_style)
    draw_subset(high_mask, radius=1.55, style=high_style)
    draw_subset(spike_mask, radius=2.2, style=spike_style)

    legend_x = width - right_margin + 26
    legend_y = top_margin + 24
    legend_items = [
        ("背景节点", "#c7ced6", 0.28, len(frame)),
        ("负值", "#2b6cb0", 0.78, int(negative_mask.sum())),
        ("高值 q99+", "#f59e0b", 0.80, int(high_mask.sum())),
        ("尖峰 q999+", "#dc2626", 0.92, int(spike_mask.sum())),
    ]
    for idx, (label, color, opacity, count) in enumerate(legend_items):
        y = legend_y + idx * 26
        parts.append(
            f'<circle cx="{legend_x}" cy="{y}" r="6" fill="{color}" fill-opacity="{opacity}" stroke="#111827" stroke-width="0.4" />'
        )
        parts.append(
            f'<text x="{legend_x + 16}" y="{y + 5}" font-size="14" fill="#111827">{label}: {count}</text>'
        )

    axis_label_x = x_column.upper()
    axis_label_y = y_column.upper()
    parts.extend(
        [
            f'<text x="{left_margin + (width - left_margin - right_margin) / 2:.1f}" y="{height - 22}" font-size="15" text-anchor="middle" fill="#374151">{axis_label_x}</text>',
            f'<text x="24" y="{top_margin + (height - top_margin - bottom_margin) / 2:.1f}" font-size="15" text-anchor="middle" transform="rotate(-90 24 {top_margin + (height - top_margin - bottom_margin) / 2:.1f})" fill="#374151">{axis_label_y}</text>',
            "</svg>",
        ]
    )
    output_path.write_text("\n".join(parts), encoding="utf-8")


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def _draw_disc(canvas: np.ndarray, cx: float, cy: float, radius: float, color: tuple[int, int, int]) -> None:
    x_center = int(round(cx))
    y_center = int(round(cy))
    radius_int = max(1, int(math.ceil(radius)))
    x_min = max(0, x_center - radius_int)
    x_max = min(canvas.shape[1] - 1, x_center + radius_int)
    y_min = max(0, y_center - radius_int)
    y_max = min(canvas.shape[0] - 1, y_center + radius_int)
    radius_sq = float(radius * radius)
    for y in range(y_min, y_max + 1):
        dy_sq = float((y - y_center) ** 2)
        for x in range(x_min, x_max + 1):
            if float((x - x_center) ** 2) + dy_sq <= radius_sq:
                canvas[y, x] = color


def _write_bmp(path: Path, image: np.ndarray) -> None:
    height, width, channels = image.shape
    if channels != 3:
        raise ValueError("BMP writer expects an RGB image.")
    row_stride = (width * 3 + 3) & ~3
    pixel_array_size = row_stride * height
    file_size = 14 + 40 + pixel_array_size
    with path.open("wb") as f:
        f.write(b"BM")
        f.write(struct.pack("<IHHI", file_size, 0, 0, 54))
        f.write(
            struct.pack(
                "<IIIHHIIIIII",
                40,
                width,
                height,
                1,
                24,
                0,
                pixel_array_size,
                2835,
                2835,
                0,
                0,
            )
        )
        padding = b"\x00" * (row_stride - width * 3)
        for row in image[::-1]:
            f.write(row[:, ::-1].tobytes())
            f.write(padding)


def write_overlay_bmp(
    output_path: Path,
    frame: pd.DataFrame,
    view: str,
) -> None:
    x_column, y_column = axis_columns(view)
    width = 1120
    height = 820
    left_margin = 88
    right_margin = 240
    top_margin = 92
    bottom_margin = 82
    background = np.full((height, width, 3), fill_value=255, dtype=np.uint8)
    background[top_margin : height - bottom_margin, left_margin : width - right_margin] = np.array(
        [250, 250, 250], dtype=np.uint8
    )

    x_values = frame[x_column].to_numpy(dtype=np.float64)
    y_values = frame[y_column].to_numpy(dtype=np.float64)
    x_px, y_px = _project_points(
        x_values=x_values,
        y_values=y_values,
        width=width,
        height=height,
        left_margin=left_margin,
        right_margin=right_margin,
        top_margin=top_margin,
        bottom_margin=bottom_margin,
    )

    overlay_values = frame["overlay_code"].to_numpy(dtype=np.float64)
    negative_mask = overlay_values < 0
    high_mask = overlay_values >= 1
    spike_mask = overlay_values >= 2

    colors = {
        "base": (199, 206, 214),
        "negative": _hex_to_rgb("#2b6cb0"),
        "high": _hex_to_rgb("#f59e0b"),
        "spike": _hex_to_rgb("#dc2626"),
    }

    for x, y in zip(x_px, y_px):
        _draw_disc(background, x, y, radius=1.0, color=colors["base"])
    for x, y in zip(x_px[negative_mask], y_px[negative_mask]):
        _draw_disc(background, x, y, radius=1.3, color=colors["negative"])
    for x, y in zip(x_px[high_mask], y_px[high_mask]):
        _draw_disc(background, x, y, radius=1.4, color=colors["high"])
    for x, y in zip(x_px[spike_mask], y_px[spike_mask]):
        _draw_disc(background, x, y, radius=2.0, color=colors["spike"])

    _write_bmp(output_path, background)


def write_markdown_report(
    output_path: Path,
    case_dir: Path,
    selected_frames: list[SelectedFrame],
    report_payload: dict,
) -> None:
    lines = [
        f"# case7new 单频可视化检查",
        "",
        f"- 代表 case: `{case_dir.name}`",
        f"- 规则: 自动选择峰值最强 case，并抽取低频/一阶模态/二阶模态/负值最多/高频末端单频帧。",
        "",
    ]

    for frame in selected_frames:
        payload = report_payload["frames"][frame.label]
        stats = payload["stats"]
        lines.extend(
            [
                f"## {frame.label}",
                "",
                f"- 文件: `{frame.file_name}`",
                f"- 频率: `{frame.frequency_hz:.4f} Hz`",
                f"- 原始统计: min=`{_format_num(stats['min'])}`, max=`{_format_num(stats['max'])}`, mean=`{_format_num(stats['mean'])}`, q99=`{_format_num(stats['q99'])}`, q999=`{_format_num(stats['q999'])}`, 负值节点=`{stats['negative_count']}`",
                "",
            ]
        )
        for subset_key in ("high_q99", "spike_q999", "negative"):
            subset = payload[subset_key]
            region_text = ", ".join(
                f"{name}={subset['region_counts'][name]} ({subset['region_ratios'][name] * 100:.1f}%)"
                for name in subset["region_counts"]
            )
            lines.extend(
                [
                    f"### {subset_key}",
                    "",
                    f"- count=`{subset['count']}`, symmetry_balance=`{_format_num(subset.get('symmetry_balance'))}`, sector_counts=`{subset['sector_counts']}`",
                    f"- r stats=`{subset['r_stats']}`",
                    f"- nearest hole stats=`{subset['nearest_hole_dist_stats']}`",
                    f"- z stats=`{subset['z_stats']}`",
                    f"- region split: {region_text}",
                    "",
                ]
            )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    case_dir = args.case_dir.resolve() if args.case_dir is not None else select_representative_case(dataset_root)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    global_payload = _load_json(case_dir / "global.json")
    nodes = pd.read_csv(case_dir / "nodes.csv")
    nodes = build_geometry_features(nodes, global_payload)
    stats = scan_frame_stats(case_dir)
    selected_frames = select_frames(stats, global_payload)

    report_payload: dict[str, object] = {
        "case_dir": str(case_dir),
        "selected_frames": [
            {"label": frame.label, "file_name": frame.file_name, "frequency_hz": frame.frequency_hz}
            for frame in selected_frames
        ],
        "frames": {},
    }

    for frame in selected_frames:
        merged = merge_frame(nodes, case_dir, frame.file_name)
        merged, q99, q999 = add_visual_columns(merged)
        frame_stats_row = stats.loc[stats["file_name"] == frame.file_name].iloc[0]

        high = merged[merged["MISES_psd_density"] >= q99]
        spike = merged[merged["MISES_psd_density"] >= q999]
        negative = merged[merged["MISES_psd_density"] < 0.0]

        frame_slug = _safe_slug(f"{frame.label}_{frame.frequency_hz:.4f}Hz")
        frame_dir = output_dir / frame_slug
        frame_dir.mkdir(parents=True, exist_ok=True)
        merged.to_csv(frame_dir / f"{frame_slug}_merged.csv", index=False)

        for view in DEFAULT_VIEWS:
            x_column, y_column = axis_columns(view)
            _write_projection_svg(
                output_path=frame_dir / f"{frame_slug}_signedlog_{view}.svg",
                frame=merged,
                x_column=x_column,
                y_column=y_column,
                value_column="signed_log10_mises",
                value_min=float(merged["signed_log10_mises"].min()),
                value_max=float(merged["signed_log10_mises"].max()),
                center_zero=True,
                point_radius=1.2,
                point_opacity=0.78,
                title=f"{case_dir.name} | {frame.label} | {frame.frequency_hz:.4f} Hz",
                subtitle="signed log10(1 + |MISES_psd_density|), centered at 0",
            )
            write_overlay_svg(
                output_path=frame_dir / f"{frame_slug}_overlay_{view}.svg",
                frame=merged,
                title=f"{case_dir.name} | {frame.label} | {frame.frequency_hz:.4f} Hz",
                subtitle="gray=all nodes, orange=q99+, red=q999+, blue=negative",
                view=view,
            )
            write_overlay_bmp(
                output_path=frame_dir / f"{frame_slug}_overlay_{view}.bmp",
                frame=merged,
                view=view,
            )

        report_payload["frames"][frame.label] = {
            "file_name": frame.file_name,
            "frequency_hz": frame.frequency_hz,
            "stats": {
                "min": float(frame_stats_row["min"]),
                "max": float(frame_stats_row["max"]),
                "mean": float(frame_stats_row["mean"]),
                "q99": float(frame_stats_row["q99"]),
                "q999": float(frame_stats_row["q999"]),
                "negative_count": int(frame_stats_row["negative_count"]),
            },
            "high_q99": summarize_subset(high, "MISES_psd_density", global_payload, "high_q99"),
            "spike_q999": summarize_subset(spike, "MISES_psd_density", global_payload, "spike_q999"),
            "negative": summarize_subset(negative, "MISES_psd_density", global_payload, "negative"),
        }

    with (output_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)
    write_markdown_report(output_dir / "report.md", case_dir, selected_frames, report_payload)

    print(json.dumps(report_payload["selected_frames"], ensure_ascii=False, indent=2))
    print(f"Saved report to {output_dir}")


if __name__ == "__main__":
    main()
