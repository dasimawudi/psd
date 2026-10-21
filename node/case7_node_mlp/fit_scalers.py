from __future__ import annotations

import argparse
from pathlib import Path

from case7_node_mlp.data import discover_case_index, expand_case_sample_paths, resolve_case_splits
from case7_node_mlp.runtime import make_logger, read_config, set_seed, write_json
from case7_node_mlp.trainer import _resolve_threshold_from_config, _save_scaler_cache, estimate_target_quantiles, fit_scalers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute node MLP feature/target scalers.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--output", type=str, default=None, help="Output .pt path. Defaults to scaler.cache_path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_config(args.config)
    set_seed(int(config["training"].get("seed", 42)))
    dataset_cfg = dict(config["dataset"])
    feature_cfg = dict(config.get("features", {}))
    scaler_cfg = dict(config.get("scaler", {}))
    target_cfg = dict(config.get("target", {}))

    output = args.output or scaler_cfg.get("cache_path")
    if output is None:
        raise ValueError("Provide --output or set scaler.cache_path in the config.")
    output_path = Path(output)
    logger = make_logger(output_path.parent, logger_name="case7_node_mlp.fit_scalers")

    case_index = discover_case_index(dataset_cfg["root"])
    split_names = resolve_case_splits(dataset_cfg["root"], dataset_cfg)
    train_case_dirs = [case_index[name] for name in split_names["train"]]
    train_sample_paths = expand_case_sample_paths(train_case_dirs, dataset_cfg)
    logger.info("Precomputing scalers | train_cases=%s | train_samples=%s", len(train_case_dirs), len(train_sample_paths))

    target_stats = estimate_target_quantiles(
        train_sample_paths=train_sample_paths,
        dataset_cfg=dataset_cfg,
        sample_limit=scaler_cfg.get("target_stats_sample_limit", scaler_cfg.get("sample_limit")),
        sample_seed=int(scaler_cfg.get("sample_seed", config["training"].get("seed", 42))),
        num_workers=int(scaler_cfg.get("num_workers", config["training"].get("num_workers", 0))),
        max_values=scaler_cfg.get("target_stats_max_values", 2_000_000),
        logger=logger,
    )
    target_zero_threshold = _resolve_threshold_from_config(target_cfg, target_stats, "zero_below")
    target_stats["target_zero_threshold"] = float(target_zero_threshold)

    x_scaler, y_scaler, feature_schema = fit_scalers(
        train_sample_paths=train_sample_paths,
        dataset_cfg=dataset_cfg,
        feature_cfg=feature_cfg,
        num_workers=int(scaler_cfg.get("num_workers", config["training"].get("num_workers", 0))),
        sample_limit=scaler_cfg.get("sample_limit"),
        sample_seed=int(scaler_cfg.get("sample_seed", config["training"].get("seed", 42))),
        prefetch_factor=int(scaler_cfg.get("prefetch_factor", 4)),
        target_zero_threshold=target_zero_threshold,
        target_stats=target_stats,
        logger=logger,
    )
    _save_scaler_cache(
        output_path,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        feature_schema=feature_schema,
        config=config,
    )
    write_json(output_path.with_suffix(".schema.json"), feature_schema)
    logger.info("Saved scaler cache: %s", output_path)


if __name__ == "__main__":
    main()
