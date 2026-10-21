from __future__ import annotations

import argparse
from pathlib import Path

import torch

from train7.data import discover_case_index, expand_case_sample_paths, resolve_case_splits
from train7.runtime import ensure_dir, make_logger, read_config, resolve_device, write_json
from train7.scalers import StandardScaler
from train7.trainer import build_model, evaluate, make_loader, write_evaluation_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained node MLP checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt.")
    parser.add_argument("--config", type=str, default=None, help="Optional config override. Defaults to checkpoint config.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val", help="Dataset split to evaluate.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for metrics and diagnostics.")
    parser.add_argument("--device", type=str, default="auto", help="Device name, e.g. auto, cuda, cpu.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader workers.")
    parser.add_argument("--sample-batch-size", type=int, default=None, help="Override case-frequency batch size.")
    parser.add_argument("--point-batch-size", type=int, default=None, help="Override point batch size.")
    parser.add_argument("--max-cases", type=int, default=None, help="Optional quick-eval case limit.")
    parser.add_argument("--max-frames-per-case", type=int, default=None, help="Optional quick-eval frame limit per case.")
    parser.add_argument("--no-diagnostics", action="store_true", help="Do not write per case-frequency diagnostics.")
    parser.add_argument(
        "--no-curve-diagnostics",
        action="store_true",
        help="Do not write sampled full-frequency hotspot/background curve diagnostics.",
    )
    parser.add_argument(
        "--curve-diagnostics-cases",
        type=int,
        default=20,
        help="Number of cases sampled for full-frequency curve diagnostics. Set to 0 to disable.",
    )
    parser.add_argument("--curve-diagnostics-seed", type=int, default=42)
    parser.add_argument("--curve-diagnostics-output-dir", type=str, default=None)
    parser.add_argument("--curve-diagnostics-background-quantile", type=float, default=0.10)
    parser.add_argument("--curve-diagnostics-point-batch-size", type=int, default=None)
    parser.add_argument(
        "--curve-diagnostics-respect-selection",
        action="store_true",
        help="Trace only frequencies where selected nodes survive dataset selection. Default forces each selected node across all frequencies.",
    )
    return parser.parse_args()


def _load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = _load_checkpoint(checkpoint_path)
    config = read_config(args.config) if args.config is not None else dict(checkpoint["config"])
    dataset_cfg = dict(config["dataset"])
    feature_cfg = dict(config.get("features", {}))
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
    logger = make_logger(output_dir, logger_name="train7.evaluate", log_file="evaluate.log")
    device = resolve_device(args.device)

    feature_schema = dict(checkpoint["feature_schema"])
    x_scaler = StandardScaler.from_state_dict(checkpoint["x_scaler"])
    y_scaler = StandardScaler.from_state_dict(checkpoint["y_scaler"])
    model = build_model(config, input_dim=int(feature_schema["input_dim"])).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=False)

    case_index = discover_case_index(dataset_cfg["root"])
    split_names = resolve_case_splits(dataset_cfg["root"], dataset_cfg)
    case_dirs = [case_index[name] for name in split_names[args.split]]
    sample_paths = expand_case_sample_paths(case_dirs, dataset_cfg)
    logger.info(
        "Evaluating checkpoint | split=%s | cases=%s | samples=%s | device=%s",
        args.split,
        len(case_dirs),
        len(sample_paths),
        device,
    )

    loader = make_loader(
        sample_paths=sample_paths,
        dataset_cfg=dataset_cfg,
        feature_cfg=feature_cfg,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        feature_schema=feature_schema,
        target_cfg=target_cfg,
        loss_cfg=loss_cfg,
        sample_batch_size=int(args.sample_batch_size or training_cfg.get("sample_batch_size", 1)),
        num_workers=int(args.num_workers if args.num_workers is not None else training_cfg.get("num_workers", 0)),
        shuffle=False,
    )
    result = evaluate(
        model=model,
        loader=loader,
        y_scaler=y_scaler,
        device=device,
        point_batch_size=int(args.point_batch_size or training_cfg.get("batch_size", 32768)),
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

    curve_cases = int(args.curve_diagnostics_cases)
    if not args.no_curve_diagnostics and curve_cases > 0:
        from train7.sample_frequency_curves import run_frequency_curve_diagnostics

        curve_output_dir = ensure_dir(
            args.curve_diagnostics_output_dir
            or output_dir / f"frequency_curve_diagnostics_{args.split}_{curve_cases}cases"
        )
        curve_summary = run_frequency_curve_diagnostics(
            checkpoint_path=checkpoint_path,
            checkpoint=checkpoint,
            config=config,
            dataset_cfg=dataset_cfg,
            feature_cfg=feature_cfg,
            target_cfg=target_cfg,
            loss_cfg=loss_cfg,
            split=args.split,
            output_dir=curve_output_dir,
            num_cases=curve_cases,
            seed=int(args.curve_diagnostics_seed),
            device=device,
            model=model,
            x_scaler=x_scaler,
            y_scaler=y_scaler,
            feature_schema=feature_schema,
            background_quantile=float(args.curve_diagnostics_background_quantile),
            respect_selection=bool(args.curve_diagnostics_respect_selection),
            case_response_point_batch_size=int(args.curve_diagnostics_point_batch_size or args.point_batch_size or training_cfg.get("batch_size", 32768)),
            logger=logger,
        )
        write_json(output_dir / f"{args.split}_frequency_curve_summary.json", curve_summary)
        logger.info("Saved frequency curve diagnostics: %s", curve_output_dir)


if __name__ == "__main__":
    main()
