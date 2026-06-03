from __future__ import annotations

import argparse
from pathlib import Path

import torch

from case7_node_gnn.data import make_graph_loader
from case7_node_gnn.models import NodeTransformerConv
from case7_node_gnn.trainer import evaluate_graph
from case7_node_mlp.data import discover_case_index, expand_case_sample_paths, resolve_case_splits
from case7_node_mlp.runtime import ensure_dir, make_logger, read_config, resolve_device, write_json
from case7_node_mlp.scalers import StandardScaler
from case7_node_mlp.trainer import write_evaluation_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained TransformerConv node checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt.")
    parser.add_argument("--config", type=str, default=None, help="Optional config override. Defaults to checkpoint config.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val", help="Dataset split to evaluate.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for metrics and diagnostics.")
    parser.add_argument("--device", type=str, default="auto", help="Device name, e.g. auto, cuda, cpu.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader workers.")
    parser.add_argument("--graph-batch-size", type=int, default=None, help="Override graph batch size.")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional quick-eval case limit.")
    parser.add_argument("--max-frames-per-case", type=int, default=None, help="Optional quick-eval frame limit per case.")
    parser.add_argument("--no-diagnostics", action="store_true", help="Do not write per case-frequency diagnostics.")
    return parser.parse_args()


def _load_checkpoint(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {path}. Run node/train_node_transformerconv.py first, "
            "or pass --checkpoint to an existing best.pt."
        )
    return torch.load(path, map_location="cpu")


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = _load_checkpoint(checkpoint_path)
    config = read_config(args.config) if args.config is not None else dict(checkpoint["config"])
    dataset_cfg = dict(config["dataset"])
    feature_cfg = dict(config.get("features", {}))
    graph_cfg = dict(config.get("graph", {}))
    training_cfg = dict(config.get("training", {}))
    target_cfg = dict(config.get("target", {}))
    loss_cfg = dict(config.get("loss", {}))

    if args.max_cases is not None:
        dataset_cfg["max_cases"] = int(args.max_cases)
        if dataset_cfg.get("split_mode") == "explicit":
            dataset_cfg[args.split + "_cases"] = list(dataset_cfg.get(args.split + "_cases", []))[: int(args.max_cases)]
    if args.max_frames_per_case is not None:
        dataset_cfg["max_frames_per_case"] = int(args.max_frames_per_case)

    output_dir = ensure_dir(args.output_dir or checkpoint_path.parent / f"eval_{args.split}")
    logger = make_logger(output_dir, logger_name="case7_node_gnn.evaluate", log_file="evaluate.log")
    device = resolve_device(args.device)

    feature_schema = dict(checkpoint["feature_schema"])
    edge_attr_names = list(feature_schema.get("graph_edge_attr_names", graph_cfg.get("edge_attr", ["dx", "dy", "dz", "dist"])))
    x_scaler = StandardScaler.from_state_dict(checkpoint["x_scaler"])
    y_scaler = StandardScaler.from_state_dict(checkpoint["y_scaler"])
    model_cfg = dict(config.get("model", {}))
    model = NodeTransformerConv(
        input_dim=int(feature_schema["input_dim"]),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        output_dim=1,
        num_layers=int(model_cfg.get("num_layers", 2)),
        heads=int(model_cfg.get("heads", 4)),
        edge_dim=len(edge_attr_names),
        dropout=float(model_cfg.get("dropout", 0.0)),
        activation=str(model_cfg.get("activation", "silu")),
        use_layer_norm=bool(model_cfg.get("layer_norm", True)),
        beta=bool(model_cfg.get("beta", True)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    case_index = discover_case_index(dataset_cfg["root"])
    split_names = resolve_case_splits(dataset_cfg["root"], dataset_cfg)
    case_dirs = [case_index[name] for name in split_names[args.split]]
    sample_paths = expand_case_sample_paths(case_dirs, dataset_cfg)
    logger.info(
        "Evaluating TransformerConv checkpoint | split=%s | cases=%s | samples=%s | device=%s",
        args.split,
        len(case_dirs),
        len(sample_paths),
        device,
    )

    loader = make_graph_loader(
        sample_paths=sample_paths,
        dataset_cfg=dataset_cfg,
        feature_cfg=feature_cfg,
        graph_cfg=graph_cfg,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        feature_schema=feature_schema,
        target_cfg=target_cfg,
        loss_cfg=loss_cfg,
        graph_batch_size=int(args.graph_batch_size or training_cfg.get("graph_batch_size", 1)),
        num_workers=int(args.num_workers if args.num_workers is not None else training_cfg.get("num_workers", 0)),
        shuffle=False,
    )
    result = evaluate_graph(
        model=model,
        loader=loader,
        y_scaler=y_scaler,
        device=device,
        split_name=args.split,
        logger=logger,
        progress_every_steps=int(training_cfg.get("eval_progress_every_steps", training_cfg.get("progress_every_steps", 50))),
        collect_diagnostics=not args.no_diagnostics,
    )
    metrics_path = output_dir / f"{args.split}_metrics.json"
    write_json(metrics_path, result.metrics)
    logger.info("Metrics: %s", result.metrics)
    if not args.no_diagnostics:
        diagnostics_path = output_dir / f"{args.split}_diagnostics.csv"
        write_evaluation_diagnostics(diagnostics_path, result.diagnostics)
        logger.info("Saved diagnostics: %s", diagnostics_path)


if __name__ == "__main__":
    main()
