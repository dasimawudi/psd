from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import csv
import json
import random

import torch
import torch.nn.functional as F

from case7_gnn.data import build_global_features, discover_case_index, load_case_graph, resolve_case_splits
from case7_gnn.models import FieldGNN, FrequencyGNN
from case7_gnn.runtime import ensure_dir, make_logger, write_json, write_yaml
from case7_gnn.scalers import (
    RunningTensorStats,
    StandardScaler,
    decode_field_targets,
    encode_field_targets,
    metric_field_targets,
)


@dataclass
class PreparedCase:
    name: str
    node_features: torch.Tensor
    edge_index: torch.Tensor
    edge_features: torch.Tensor
    global_features: torch.Tensor
    target_normalized: torch.Tensor
    target_metric: torch.Tensor

    def to(self, device: torch.device) -> "PreparedCase":
        return PreparedCase(
            name=self.name,
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            edge_features=self.edge_features.to(device),
            global_features=self.global_features.to(device),
            target_normalized=self.target_normalized.to(device),
            target_metric=self.target_metric.to(device),
        )


def prepare_case(
    case: Any,
    node_scaler: StandardScaler,
    edge_scaler: StandardScaler,
    global_scaler: StandardScaler,
    target_scaler: StandardScaler,
    task: str,
    use_psd: bool,
    use_freq_top3: bool,
    clamp_negative_rmises: bool,
) -> PreparedCase:
    global_features = global_scaler.transform(
        build_global_features(case, use_psd=use_psd, use_freq_top3=use_freq_top3)
    )
    node_features = node_scaler.transform(case.node_features)
    edge_features = edge_scaler.transform(case.edge_features)

    if task == "frequency":
        target_metric = case.freq_target
        target_normalized = target_scaler.transform(case.freq_target)
    elif task == "field":
        encoded_target = encode_field_targets(case.node_targets, clamp_negative_rmises=clamp_negative_rmises)
        target_metric = metric_field_targets(case.node_targets, clamp_negative_rmises=clamp_negative_rmises)
        target_normalized = target_scaler.transform(encoded_target)
    else:
        raise ValueError(f"Unsupported task: {task}")

    return PreparedCase(
        name=case.name,
        node_features=node_features,
        edge_index=case.edge_index,
        edge_features=edge_features,
        global_features=global_features,
        target_normalized=target_normalized,
        target_metric=target_metric,
    )


def fit_feature_scalers(
    train_case_paths: list[Path],
    dataset_cfg: dict[str, Any],
    task: str,
    use_psd: bool,
    use_freq_top3: bool,
    clamp_negative_rmises: bool,
    case_limit: int | None = None,
) -> tuple[StandardScaler, StandardScaler, StandardScaler, StandardScaler]:
    selected_paths = list(train_case_paths)
    if case_limit is not None:
        case_limit_int = int(case_limit)
        if case_limit_int <= 0:
            raise ValueError("scaler_fit_case_limit must be positive when provided.")
        selected_paths = selected_paths[:case_limit_int]

    if not selected_paths:
        raise ValueError("Training split is empty; cannot fit feature scalers.")

    cache_dir = dataset_cfg.get("cache_dir")
    node_stats = RunningTensorStats()
    edge_stats = RunningTensorStats()
    global_stats = RunningTensorStats()
    target_stats = RunningTensorStats()

    for case_path in selected_paths:
        case = load_case_graph(
            case_path,
            node_columns=dataset_cfg["node_columns"],
            edge_columns=dataset_cfg["edge_columns"],
            target_freq_key=dataset_cfg["target_freq_key"],
            make_undirected=bool(dataset_cfg["make_undirected"]),
            cache_dir=cache_dir,
        )

        node_stats.update(case.node_features)
        edge_stats.update(case.edge_features)
        global_stats.update(build_global_features(case, use_psd=use_psd, use_freq_top3=use_freq_top3))

        if task == "frequency":
            target_stats.update(case.freq_target)
        elif task == "field":
            target_stats.update(
                encode_field_targets(case.node_targets, clamp_negative_rmises=clamp_negative_rmises)
            )
        else:
            raise ValueError(f"Unsupported task: {task}")

    return (
        node_stats.finalize(),
        edge_stats.finalize(),
        global_stats.finalize(),
        target_stats.finalize(),
    )


def build_model(config: dict[str, Any], sample_case: PreparedCase) -> torch.nn.Module:
    task = config["task"]
    model_cfg = config["model"]

    common_kwargs = dict(
        node_input_dim=int(sample_case.node_features.size(-1)),
        edge_input_dim=int(sample_case.edge_features.size(-1)),
        global_input_dim=int(sample_case.global_features.numel()),
        hidden_dim=int(model_cfg["hidden_dim"]),
        global_dim=int(model_cfg["global_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
    )

    if task == "frequency":
        return FrequencyGNN(output_dim=int(sample_case.target_normalized.numel()), **common_kwargs)
    if task == "field":
        return FieldGNN(output_dim=int(sample_case.target_normalized.size(-1)), **common_kwargs)
    raise ValueError(f"Unsupported task: {task}")


def compute_loss(prediction: torch.Tensor, target: torch.Tensor, loss_name: str) -> torch.Tensor:
    if loss_name == "mse":
        return F.mse_loss(prediction, target)
    if loss_name == "smooth_l1":
        return F.smooth_l1_loss(prediction, target)
    raise ValueError(f"Unsupported loss: {loss_name}")


def evaluate_frequency(
    model: torch.nn.Module,
    case_paths: list[Path],
    case_loader: Callable[[Path], PreparedCase],
    target_scaler: StandardScaler,
    device: torch.device,
    loss_name: str,
) -> dict[str, float]:
    scaler = target_scaler.to(device)
    total_loss = 0.0
    total_count = 0
    abs_sum = 0.0
    sq_sum = 0.0
    value_count = 0

    model.eval()
    with torch.no_grad():
        for case_path in case_paths:
            batch = case_loader(case_path).to(device)
            prediction = model(
                batch.node_features,
                batch.edge_index,
                batch.edge_features,
                batch.global_features,
            )
            loss = compute_loss(prediction, batch.target_normalized, loss_name=loss_name)
            total_loss += loss.item()
            total_count += 1

            prediction_hz = scaler.inverse_transform(prediction).detach().cpu()
            target_hz = batch.target_metric.detach().cpu()
            error = prediction_hz - target_hz
            abs_sum += error.abs().sum().item()
            sq_sum += error.pow(2).sum().item()
            value_count += int(target_hz.numel())

    denom = max(total_count, 1)
    metric_denom = max(value_count, 1)
    return {
        "loss": total_loss / denom,
        "mae_hz": abs_sum / metric_denom,
        "rmse_hz": (sq_sum / metric_denom) ** 0.5,
    }


def evaluate_field(
    model: torch.nn.Module,
    case_paths: list[Path],
    case_loader: Callable[[Path], PreparedCase],
    target_scaler: StandardScaler,
    clamp_negative_rmises: bool,
    device: torch.device,
    loss_name: str,
) -> dict[str, float]:
    scaler = target_scaler.to(device)
    total_loss = 0.0
    total_nodes = 0
    total_rta_abs = 0.0
    total_rmises_abs = 0.0

    model.eval()
    with torch.no_grad():
        for case_path in case_paths:
            batch = case_loader(case_path).to(device)
            prediction = model(
                batch.node_features,
                batch.edge_index,
                batch.edge_features,
                batch.global_features,
            )
            loss = compute_loss(prediction, batch.target_normalized, loss_name=loss_name)
            node_count = int(batch.node_features.size(0))
            total_loss += loss.item() * node_count
            total_nodes += node_count

            prediction_encoded = scaler.inverse_transform(prediction).detach().cpu()
            prediction_raw = decode_field_targets(
                prediction_encoded,
                clamp_negative_rmises=clamp_negative_rmises,
            )
            target_raw = batch.target_metric.detach().cpu()
            error = (prediction_raw - target_raw).abs()
            total_rta_abs += error[:, 0].sum().item()
            total_rmises_abs += error[:, 1].sum().item()

    denom = max(total_nodes, 1)
    return {
        "loss": total_loss / denom,
        "rta_mae": total_rta_abs / denom,
        "rmises_mae": total_rmises_abs / denom,
    }


def train_one_epoch(
    model: torch.nn.Module,
    case_paths: list[Path],
    case_loader: Callable[[Path], PreparedCase],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_name: str,
    grad_clip: float,
) -> float:
    model.train()
    shuffled = list(case_paths)
    random.shuffle(shuffled)

    total_loss = 0.0
    total_weight = 0

    for case_path in shuffled:
        batch = case_loader(case_path).to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(
            batch.node_features,
            batch.edge_index,
            batch.edge_features,
            batch.global_features,
        )
        loss = compute_loss(prediction, batch.target_normalized, loss_name=loss_name)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        weight = int(batch.node_features.size(0)) if batch.target_normalized.dim() == 2 else 1
        total_loss += loss.item() * weight
        total_weight += weight

    return total_loss / max(total_weight, 1)


def _save_checkpoint(
    save_dir: Path,
    config: dict[str, Any],
    model: torch.nn.Module,
    scalers: dict[str, StandardScaler],
    metrics: dict[str, Any],
) -> None:
    checkpoint = {
        "model_state": model.state_dict(),
        "config": config,
        "scalers": {name: scaler.state_dict() for name, scaler in scalers.items()},
        "metrics": metrics,
    }
    torch.save(checkpoint, save_dir / "best.pt")


class Case7Trainer:
    def __init__(self, config: dict[str, Any], device: torch.device) -> None:
        self.config = config
        self.device = device

        self.task = config["task"]
        self.dataset_cfg = config["dataset"]
        self.training_cfg = config["training"]
        self.use_psd = bool(config["features"]["use_psd"])
        self.use_freq_top3 = bool(config["features"].get("use_freq_top3", False))
        self.clamp_negative_rmises = bool(self.dataset_cfg.get("clamp_negative_rmises", True))
        self.cache_dir = self.dataset_cfg.get("cache_dir")

        self.save_dir = ensure_dir(self.training_cfg["save_dir"])
        self.logger = make_logger(self.save_dir, logger_name=f"case7_gnn.{self.task}")
        self.history_path = self.save_dir / "history.csv"

        self.model: torch.nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scalers: dict[str, StandardScaler] = {}
        self.case_index: dict[str, Path] = {}
        self.split_names: dict[str, list[str]] = {}
        self.train_case_paths: list[Path] = []
        self.val_case_paths: list[Path] = []
        self.test_case_paths: list[Path] = []
        self.resolved_config: dict[str, Any] = {}

        self._prepare()

    def _load_prepared_case(self, case_path: Path) -> PreparedCase:
        case = load_case_graph(
            case_path,
            node_columns=self.dataset_cfg["node_columns"],
            edge_columns=self.dataset_cfg["edge_columns"],
            target_freq_key=self.dataset_cfg["target_freq_key"],
            make_undirected=bool(self.dataset_cfg["make_undirected"]),
            cache_dir=self.cache_dir,
        )
        return prepare_case(
            case,
            node_scaler=self.scalers["node"],
            edge_scaler=self.scalers["edge"],
            global_scaler=self.scalers["global"],
            target_scaler=self.scalers["target"],
            task=self.task,
            use_psd=self.use_psd,
            use_freq_top3=self.use_freq_top3,
            clamp_negative_rmises=self.clamp_negative_rmises,
        )

    def _prepare(self) -> None:
        self.case_index = discover_case_index(self.dataset_cfg["root"])
        self.split_names = resolve_case_splits(self.dataset_cfg["root"], self.dataset_cfg)

        self.train_case_paths = [self.case_index[name] for name in self.split_names["train"]]
        self.val_case_paths = [self.case_index[name] for name in self.split_names["val"]]
        self.test_case_paths = [self.case_index[name] for name in self.split_names.get("test", [])]

        node_scaler, edge_scaler, global_scaler, target_scaler = fit_feature_scalers(
            train_case_paths=self.train_case_paths,
            dataset_cfg=self.dataset_cfg,
            task=self.task,
            use_psd=self.use_psd,
            use_freq_top3=self.use_freq_top3,
            clamp_negative_rmises=self.clamp_negative_rmises,
            case_limit=self.dataset_cfg.get("scaler_fit_case_limit"),
        )

        self.scalers = {
            "node": node_scaler,
            "edge": edge_scaler,
            "global": global_scaler,
            "target": target_scaler,
        }

        sample_case = self._load_prepared_case(self.train_case_paths[0])
        self.model = build_model(self.config, sample_case=sample_case).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.training_cfg["lr"]),
            weight_decay=float(self.training_cfg["weight_decay"]),
        )

        resolved_dataset = dict(self.dataset_cfg)
        resolved_dataset["train_cases"] = list(self.split_names["train"])
        resolved_dataset["val_cases"] = list(self.split_names["val"])
        resolved_dataset["test_cases"] = list(self.split_names.get("test", []))
        self.resolved_config = dict(self.config)
        self.resolved_config["dataset"] = resolved_dataset
        write_yaml(self.save_dir / "resolved_config.yaml", self.resolved_config)

    def _evaluate(self, case_paths: list[Path], loss_name: str) -> dict[str, float]:
        assert self.model is not None
        target_scaler = self.scalers["target"]
        if self.task == "frequency":
            return evaluate_frequency(
                model=self.model,
                case_paths=case_paths,
                case_loader=self._load_prepared_case,
                target_scaler=target_scaler,
                device=self.device,
                loss_name=loss_name,
            )
        return evaluate_field(
            model=self.model,
            case_paths=case_paths,
            case_loader=self._load_prepared_case,
            target_scaler=target_scaler,
            clamp_negative_rmises=self.clamp_negative_rmises,
            device=self.device,
            loss_name=loss_name,
        )

    def _append_history_row(self, row: dict[str, Any]) -> None:
        fieldnames = list(row.keys())
        file_exists = self.history_path.exists()
        with self.history_path.open("a", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def fit(self) -> dict[str, Any]:
        assert self.model is not None
        assert self.optimizer is not None

        eval_every = int(self.training_cfg.get("eval_every", 1))
        if eval_every <= 0:
            raise ValueError("training.eval_every must be positive.")

        self.logger.info("Task: %s", self.task)
        self.logger.info("Device: %s", self.device)
        self.logger.info(
            "Train/Val/Test graphs: %s/%s/%s",
            len(self.train_case_paths),
            len(self.val_case_paths),
            len(self.test_case_paths),
        )
        self.logger.info("Using PSD features: %s", self.use_psd)
        self.logger.info("Using freq_top3 features: %s", self.use_freq_top3)
        self.logger.info("Split mode: %s", self.dataset_cfg.get("split_mode", "explicit"))
        if self.cache_dir:
            self.logger.info("Raw case cache: %s", self.cache_dir)
        if self.dataset_cfg.get("scaler_fit_case_limit") is not None:
            self.logger.info("Scaler fit case limit: %s", self.dataset_cfg["scaler_fit_case_limit"])

        best_val_loss = float("inf")
        best_payload: dict[str, Any] | None = None
        patience = int(self.training_cfg["early_stopping_patience"])
        wait = 0

        for epoch in range(1, int(self.training_cfg["epochs"]) + 1):
            train_loss = train_one_epoch(
                model=self.model,
                case_paths=self.train_case_paths,
                case_loader=self._load_prepared_case,
                optimizer=self.optimizer,
                device=self.device,
                loss_name=self.training_cfg["loss"],
                grad_clip=float(self.training_cfg["grad_clip"]),
            )

            history_row = {
                "epoch": epoch,
                "train_loss": round(float(train_loss), 8),
            }

            if epoch % eval_every != 0:
                self._append_history_row(history_row)
                if epoch == 1 or epoch % int(self.training_cfg["print_every"]) == 0:
                    self.logger.info("Epoch %04d | train_loss=%.6f | eval=skipped", epoch, train_loss)
                continue

            val_metrics = self._evaluate(self.val_case_paths, loss_name=self.training_cfg["loss"])
            test_metrics = self._evaluate(self.test_case_paths, loss_name=self.training_cfg["loss"])

            history_row["val_loss"] = round(float(val_metrics["loss"]), 8)
            history_row["test_loss"] = round(float(test_metrics["loss"]), 8)
            for key, value in val_metrics.items():
                if key != "loss":
                    history_row[f"val_{key}"] = round(float(value), 8)
            for key, value in test_metrics.items():
                if key != "loss":
                    history_row[f"test_{key}"] = round(float(value), 8)
            self._append_history_row(history_row)

            if epoch == 1 or epoch % int(self.training_cfg["print_every"]) == 0:
                self.logger.info(
                    "Epoch %04d | train_loss=%.6f | val=%s | test=%s",
                    epoch,
                    train_loss,
                    json.dumps(val_metrics, ensure_ascii=False),
                    json.dumps(test_metrics, ensure_ascii=False),
                )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                wait = 0
                best_payload = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                }
                _save_checkpoint(
                    save_dir=self.save_dir,
                    config=self.resolved_config,
                    model=self.model,
                    scalers=self.scalers,
                    metrics=best_payload,
                )
                write_json(self.save_dir / "metrics.json", best_payload)
            else:
                wait += 1
                if wait >= patience:
                    self.logger.info("Early stopping at epoch %s.", epoch)
                    break

        if best_payload is None:
            raise RuntimeError("Training finished without a saved checkpoint.")

        self.logger.info("Best run summary:\n%s", json.dumps(best_payload, indent=2, ensure_ascii=False))
        return best_payload
