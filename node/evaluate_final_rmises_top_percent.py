from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from case7_node_mlp.runtime import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate final RMises metrics for per-case target top percentages.")
    parser.add_argument("--input-dir", required=True, type=str, help="Directory from summarize_final_rmises_topk.py with --write-per-node.")
    parser.add_argument("--output-dir", default=None, type=str)
    parser.add_argument("--top-fractions", default="0.01,0.05,0.10,0.15,0.25", type=str)
    return parser.parse_args()


def _parse_fractions(value: str) -> list[float]:
    fractions = []
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


def _load_per_node_rows(input_dir: Path) -> pd.DataFrame:
    plot_csv = input_dir / "rmises_by_case_and_quantile_plots" / "final_rmises_per_node_error_metrics.csv"
    if plot_csv.exists():
        rows = pd.read_csv(plot_csv)
    else:
        paths = sorted((input_dir / "cases").glob("*_final_rmises_per_node.csv"))
        if not paths:
            raise FileNotFoundError(f"No per-node RMises CSVs found under {input_dir / 'cases'}; rerun summarize_final_rmises_topk.py with --write-per-node")
        rows = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    rows = rows[np.isfinite(rows["target_rmises"]) & np.isfinite(rows["pred_rmises"])].copy()
    if "abs_error" not in rows.columns:
        rows["abs_error"] = (rows["pred_rmises"] - rows["target_rmises"]).abs()
    if "relative_error" not in rows.columns:
        rows["relative_error"] = rows["abs_error"] / rows["target_rmises"].abs().where(rows["target_rmises"].abs() > 1e-12)
    if "within25" not in rows.columns:
        rows["within25"] = rows["relative_error"] <= 0.25
    return rows


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = ensure_dir(args.output_dir or input_dir / "rmises_by_case_and_quantile_plots")
    fractions = _parse_fractions(args.top_fractions)
    rows = _load_per_node_rows(input_dir)

    overall_rel = rows["relative_error"].dropna()
    top = {}
    csv_rows = [
        {
            "group": "overall",
            "points": int(len(rows)),
            "mean_nodes_per_case": "",
            "within25_ratio": float(rows["within25"].mean()),
            "relative_mae": float(overall_rel.mean()),
            "pred_target_ratio": float(rows["pred_rmises"].mean() / max(abs(rows["target_rmises"].mean()), 1e-12)),
        }
    ]
    for fraction in fractions:
        chunks = []
        counts = []
        for _case_name, case_rows in rows.groupby("case_name", sort=False):
            count = max(1, int(math.ceil(len(case_rows) * fraction)))
            counts.append(count)
            chunks.append(case_rows.sort_values("target_rmises", ascending=False).head(count))
        selected = pd.concat(chunks, ignore_index=True)
        rel = selected["relative_error"].dropna()
        key = _fraction_label(fraction)
        top[key] = {
            "points": int(len(selected)),
            "mean_nodes_per_case": float(np.mean(counts)),
            "within25_ratio": float(selected["within25"].mean()),
            "relative_mae": float(rel.mean()),
            "pred_target_ratio": float(selected["pred_rmises"].mean() / max(abs(selected["target_rmises"].mean()), 1e-12)),
        }
        csv_rows.append({"group": key, **top[key]})

    summary = {
        "input_dir": str(input_dir),
        "dimension": "case-node-final-rmises",
        "topk_definition": "per case, top fraction of nodes ranked by target final RMises",
        "overall": csv_rows[0],
        "top_fraction_by_case_target_rmises": top,
    }
    write_json(output_dir / "final_rmises_top_percent_metrics.json", summary)
    pd.DataFrame(csv_rows).to_csv(output_dir / "final_rmises_top_percent_metrics.csv", index=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
