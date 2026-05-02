from __future__ import annotations

from pathlib import Path
from typing import Any

import argparse
import json

import pandas as pd
import torch

from case7_gnn_stress_only.data import load_case_graph
from case7_gnn_stress_only.runtime import ensure_dir, read_config, resolve_device, write_json
from case7_gnn_stress_only.scalers import StandardScaler
from case7_gnn_stress_only.trainer import (
    PreparedCase,
    build_model,
    decode_field_prediction,
    get_two_stage_rmises_cfg,
    prepare_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a trained case7 GNN checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--case-dir", type=str, required=True, help="Path to a case directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save predictions")
    parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda")
    parser.add_argument("--config", type=str, default=None, help="Optional config override")
    return parser.parse_args()


def _load_scalers(payload: dict[str, Any]) -> dict[str, StandardScaler]:
    return {
        name: StandardScaler.from_state_dict(state)
        for name, state in payload["scalers"].items()
    }


def _prepare_case(
    case_dir: str | Path,
    config: dict[str, Any],
    scalers: dict[str, StandardScaler],
) -> tuple[Any, PreparedCase]:
    dataset_cfg = config["dataset"]
    two_stage_rmises_cfg = get_two_stage_rmises_cfg(config)
    use_psd = bool(config["features"]["use_psd"])
    use_freq_top3 = bool(config["features"].get("use_freq_top3", False))
    use_frequency_scalar = bool(config["features"].get("use_frequency_scalar", False))
    use_frequency_relations = bool(config["features"].get("use_frequency_relations", False))
    feature_cfg = dict(config.get("features", {}))
    clamp_negative_rmises = bool(dataset_cfg.get("clamp_negative_rmises", True))

    case = load_case_graph(
        case_dir=case_dir,
        node_columns=dataset_cfg["node_columns"],
        edge_columns=dataset_cfg["edge_columns"],
        target_freq_key=dataset_cfg["target_freq_key"],
        make_undirected=bool(dataset_cfg["make_undirected"]),
        cache_dir=dataset_cfg.get("cache_dir"),
    )
    prepared = prepare_case(
        case=case,
        node_scaler=scalers["node"],
        edge_scaler=scalers["edge"],
        global_scaler=scalers["global"],
        target_scaler=scalers["target"],
        task=config["task"],
        use_psd=use_psd,
        use_freq_top3=use_freq_top3,
        use_frequency_scalar=use_frequency_scalar,
        use_frequency_relations=use_frequency_relations,
        clamp_negative_rmises=clamp_negative_rmises,
        node_columns=dataset_cfg["node_columns"],
        edge_columns=dataset_cfg["edge_columns"],
        feature_cfg=feature_cfg,
        two_stage_rmises_cfg=two_stage_rmises_cfg,
    )
    return case, prepared


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"] if args.config is None else read_config(args.config)
    scalers = _load_scalers(checkpoint)
    device = resolve_device(args.device)

    raw_case, prepared_case = _prepare_case(
        case_dir=args.case_dir,
        config=config,
        scalers=scalers,
    )

    model = build_model(config, sample_case=prepared_case)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    batch = prepared_case.to(device)
    with torch.no_grad():
        prediction = model(
            batch.node_features,
            batch.edge_index,
            batch.edge_features,
            batch.global_features,
        )

    output_dir = ensure_dir(args.output_dir)
    target_scaler = scalers["target"].to(device)

    if config["task"] == "frequency":
        prediction_hz = target_scaler.inverse_transform(prediction).detach().cpu().tolist()
        result = {
            "case": raw_case.name,
            "task": "frequency",
            "predicted_freq_top3": prediction_hz,
            "actual_freq_top3": raw_case.freq_target.tolist(),
        }
        write_json(output_dir / "frequency_prediction.json", result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    clamp_negative_rmises = bool(config["dataset"].get("clamp_negative_rmises", True))
    two_stage_rmises_cfg = get_two_stage_rmises_cfg(config)
    prediction_raw, hotspot_prob, hotspot_mask = decode_field_prediction(
        prediction=prediction,
        target_scaler=target_scaler,
        clamp_negative_rmises=clamp_negative_rmises,
        two_stage_rmises_cfg=two_stage_rmises_cfg,
    )
    prediction_raw = prediction_raw.detach().cpu()

    node_count = raw_case.node_features.size(0)
    columns: dict[str, Any] = {
        "node_id": list(range(node_count)),
        "x": raw_case.node_features[:, 0].cpu().numpy(),
        "y": raw_case.node_features[:, 1].cpu().numpy(),
        "z": raw_case.node_features[:, 2].cpu().numpy(),
        "pred_stress": prediction_raw[:, 0].cpu().numpy(),
        "actual_stress": raw_case.node_targets[:, -1].cpu().numpy(),
    }
    if bool(two_stage_rmises_cfg["enabled"]):
        assert hotspot_prob is not None and hotspot_mask is not None
        columns["pred_hotspot_prob"] = hotspot_prob.detach().cpu().numpy()
        columns["pred_hotspot_label"] = hotspot_mask.detach().cpu().to(dtype=torch.int32).numpy()
    output_df = pd.DataFrame(columns)
    output_df.to_csv(output_dir / "field_prediction.csv", index=False)
    summary = {
        "case": raw_case.name,
        "task": "field",
        "rows": int(len(output_df)),
        "output_csv": str(output_dir / "field_prediction.csv"),
        "two_stage_hotspot": bool(two_stage_rmises_cfg["enabled"]),
    }
    write_json(output_dir / "field_prediction_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
