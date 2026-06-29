from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from case7_node_mlp.data import _load_case_static, discover_case_index
from case7_node_mlp.runtime import ensure_dir, read_config, write_json


REGIONS = {
    "fullpart": "全零件",
    "earpiece_region": "耳片区域",
    "disk_region": "圆盘区域",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate final RMises top-percent metrics by fullpart/earpiece/disk region."
    )
    parser.add_argument("--input-dir", required=True, type=str, help="Directory from summarize_final_rmises_topk.py.")
    parser.add_argument("--config", required=True, type=str, help="Resolved config used by the evaluated checkpoint.")
    parser.add_argument("--output-dir", default=None, type=str)
    parser.add_argument("--top-fractions", default="0.01,0.05,0.10,0.15,0.25", type=str)
    return parser.parse_args()


def _parse_fractions(value: str) -> list[float]:
    fractions: list[float] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        fraction = float(part)
        if fraction <= 0.0 or fraction > 1.0:
            raise ValueError(f"top fraction must be in (0, 1], got {fraction}")
        fractions.append(fraction)
    if not fractions:
        raise ValueError("At least one top fraction is required.")
    return sorted(set(fractions))


def _fraction_label(fraction: float) -> str:
    percent = fraction * 100.0
    if abs(percent - round(percent)) < 1e-8:
        return f"top{int(round(percent))}pct"
    return f"top{percent:g}pct"


def _safe_relative_error(pred: pd.Series, target: pd.Series) -> pd.Series:
    denom = target.abs()
    return (pred - target).abs() / denom.where(denom > 1e-12, np.nan)


def _summarize(rows: pd.DataFrame, group: str, mean_nodes_per_case: float | None = None) -> dict[str, Any]:
    if rows.empty:
        return {
            "group": group,
            "points": 0,
            "mean_nodes_per_case": mean_nodes_per_case if mean_nodes_per_case is not None else 0.0,
            "within25_ratio": float("nan"),
            "relative_mae": float("nan"),
            "pred_target_ratio": float("nan"),
        }
    rel = _safe_relative_error(rows["pred_rmises"], rows["target_rmises"])
    return {
        "group": group,
        "points": int(len(rows)),
        "mean_nodes_per_case": mean_nodes_per_case if mean_nodes_per_case is not None else "",
        "target_rmises_mean": float(rows["target_rmises"].mean()),
        "target_rmises_max": float(rows["target_rmises"].max()),
        "pred_rmises_mean": float(rows["pred_rmises"].mean()),
        "pred_rmises_max": float(rows["pred_rmises"].max()),
        "mae": float((rows["pred_rmises"] - rows["target_rmises"]).abs().mean()),
        "relative_mae": float(rel.mean(skipna=True)),
        "within25_ratio": float((rel <= 0.25).mean()),
        "pred_target_ratio": float(rows["pred_rmises"].mean() / max(abs(float(rows["target_rmises"].mean())), 1e-12)),
    }


def _load_case_rows(path: Path) -> pd.DataFrame:
    rows = pd.read_csv(path)
    rows = rows[np.isfinite(rows["target_rmises"]) & np.isfinite(rows["pred_rmises"])].copy()
    if "case_name" not in rows.columns:
        raise ValueError(f"Missing case_name column in {path}")
    if "node_index" not in rows.columns:
        raise ValueError(f"Missing node_index column in {path}")
    return rows


def _append_region_labels(rows: pd.DataFrame, case_dir: Path, dataset_cfg: dict[str, Any]) -> pd.DataFrame:
    nodes_df, _payload, earpiece_mask = _load_case_static(
        case_dir,
        region_cfg=dataset_cfg.get("earpiece_region"),
    )
    if "node_index" in nodes_df.columns:
        node_values = nodes_df["node_index"].to_numpy(dtype=np.int64, copy=False)
    else:
        node_values = np.arange(len(nodes_df), dtype=np.int64)
    mask_by_node = pd.Series(earpiece_mask.astype(bool, copy=False), index=node_values)
    rows = rows.copy()
    rows["is_earpiece_region"] = rows["node_index"].map(mask_by_node).fillna(False).astype(bool)
    rows["region"] = np.where(rows["is_earpiece_region"], "earpiece_region", "disk_region")
    return rows


def _top_rows_by_case(rows: pd.DataFrame, fraction: float) -> tuple[pd.DataFrame, float]:
    chunks: list[pd.DataFrame] = []
    counts: list[int] = []
    for _case_name, case_rows in rows.groupby("case_name", sort=False):
        if case_rows.empty:
            continue
        count = max(1, int(math.ceil(len(case_rows) * fraction)))
        counts.append(count)
        chunks.append(case_rows.sort_values("target_rmises", ascending=False, kind="mergesort").head(count))
    if not chunks:
        return rows.iloc[:0].copy(), 0.0
    return pd.concat(chunks, ignore_index=True), float(np.mean(counts))


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    config = read_config(args.config)
    dataset_cfg = dict(config["dataset"])
    output_dir = ensure_dir(args.output_dir or input_dir / "rmises_region_top_percent")
    fractions = _parse_fractions(args.top_fractions)

    case_index = discover_case_index(dataset_cfg["root"])
    paths = sorted((input_dir / "cases").glob("*_final_rmises_per_node.csv"))
    if not paths:
        raise FileNotFoundError(f"No per-node RMises CSVs found under {input_dir / 'cases'}")

    case_frames: list[pd.DataFrame] = []
    for path in paths:
        rows = _load_case_rows(path)
        case_name = str(rows["case_name"].iloc[0])
        case_dir = case_index.get(case_name)
        if case_dir is None:
            raise KeyError(f"Case {case_name!r} from {path} is not present under dataset root.")
        case_frames.append(_append_region_labels(rows, case_dir=case_dir, dataset_cfg=dataset_cfg))

    all_rows = pd.concat(case_frames, ignore_index=True)
    summary: dict[str, Any] = {
        "input_dir": str(input_dir),
        "config": str(args.config),
        "dimension": "case-node-final-rmises",
        "topk_definition": "per case and per region, top fraction of nodes ranked by target final RMises",
        "regions": {},
    }
    csv_rows: list[dict[str, Any]] = []

    region_filters = {
        "fullpart": np.ones(len(all_rows), dtype=bool),
        "earpiece_region": all_rows["region"].to_numpy() == "earpiece_region",
        "disk_region": all_rows["region"].to_numpy() == "disk_region",
    }
    for region_name, mask in region_filters.items():
        region_rows = all_rows.loc[mask].copy()
        region_summary: dict[str, Any] = {
            "display_name": REGIONS[region_name],
            "overall": _summarize(region_rows, "overall"),
            "top_fraction_by_case_target_rmises": {},
        }
        csv_rows.append({"region": region_name, **region_summary["overall"]})
        for fraction in fractions:
            key = _fraction_label(fraction)
            selected, mean_count = _top_rows_by_case(region_rows, fraction)
            metrics = _summarize(selected, key, mean_nodes_per_case=mean_count)
            region_summary["top_fraction_by_case_target_rmises"][key] = metrics
            csv_rows.append({"region": region_name, **metrics})
        summary["regions"][region_name] = region_summary

    write_json(output_dir / "final_rmises_region_top_percent_metrics.json", summary)
    pd.DataFrame(csv_rows).to_csv(output_dir / "final_rmises_region_top_percent_metrics.csv", index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
