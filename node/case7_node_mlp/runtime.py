from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import logging
import random

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _deep_merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def read_config(path: str | Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    config_path = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if config_path in seen:
        raise ValueError(f"Recursive config base_config reference: {config_path}")
    seen.add(config_path)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")

    base_ref = config.pop("base_config", None) or config.pop("inherits", None)
    if base_ref is None:
        return config

    base_refs = base_ref if isinstance(base_ref, list) else [base_ref]
    merged: dict[str, Any] = {}
    for item in base_refs:
        base_path = Path(str(item))
        if not base_path.is_absolute():
            base_path = config_path.parent / base_path
        merged = _deep_merge_config(merged, read_config(base_path, _seen=seen))
    return _deep_merge_config(merged, config)


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def make_logger(
    log_dir: str | Path,
    logger_name: str = "case7_node_mlp",
    log_file: str = "train.log",
) -> logging.Logger:
    log_dir = ensure_dir(log_dir)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_dir / log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
