from __future__ import annotations

from typing import Any

import torch

from case7_gnn_stress_only.data import (
    discover_case_index,
    expand_case_sample_paths,
    load_case_graph,
    resolve_case_splits,
)
from case7_gnn_stress_only.runtime import write_yaml
from case7_gnn_stress_only.trainer import (
    Case7Trainer as BaseCase7Trainer,
    PreparedCase,
    build_augmented_global_features,
    build_augmented_node_features,
    get_mode_shape_loader_kwargs,
    get_stress_peak_relative_cfg,
    get_two_stage_rmises_cfg,
    prepare_case,
)
from case7_gnn_stress_only.scalers import RunningTensorStats, StandardScaler, encode_field_targets

from case7_mlp_stress.models import ConditionalFieldMLP


def fit_mlp_feature_scalers(
    train_case_paths: list[Any],
    dataset_cfg: dict[str, Any],
    task: str,
    use_psd: bool,
    use_freq_top3: bool,
    use_frequency_scalar: bool,
    use_frequency_relations: bool,
    clamp_negative_rmises: bool,
    feature_cfg: dict[str, Any] | None = None,
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

    node_stats = RunningTensorStats()
    global_stats = RunningTensorStats()
    target_stats = RunningTensorStats()
    cache_dir = dataset_cfg.get("cache_dir")

    for case_path in selected_paths:
        case = load_case_graph(
            case_path,
            node_columns=dataset_cfg["node_columns"],
            edge_columns=dataset_cfg["edge_columns"],
            target_freq_key=dataset_cfg["target_freq_key"],
            make_undirected=bool(dataset_cfg["make_undirected"]),
            cache_dir=cache_dir,
            node_region=dataset_cfg.get("node_region"),
            **get_mode_shape_loader_kwargs(feature_cfg),
        )
        node_stats.update(
            build_augmented_node_features(
                case,
                node_columns=dataset_cfg["node_columns"],
                feature_cfg=feature_cfg,
            )
        )
        global_stats.update(
            build_augmented_global_features(
                case,
                use_psd=use_psd,
                use_freq_top3=use_freq_top3,
                use_frequency_scalar=use_frequency_scalar,
                use_frequency_relations=use_frequency_relations,
                feature_cfg=feature_cfg,
            )
        )
        if task != "field":
            raise ValueError("The MLP stress trainer currently supports task: field only.")
        target_stats.update(
            encode_field_targets(
                case.node_targets,
                clamp_negative_rmises=clamp_negative_rmises,
                rmises_as_excess=False,
                rmises_threshold=0.0,
            )
        )

    edge_dim = len(dataset_cfg.get("edge_columns", []))
    edge_scaler = StandardScaler(
        mean=torch.zeros(edge_dim, dtype=torch.float32),
        std=torch.ones(edge_dim, dtype=torch.float32),
    )
    return node_stats.finalize(), edge_scaler, global_stats.finalize(), target_stats.finalize()


def build_model(config: dict[str, Any], sample_case: PreparedCase) -> torch.nn.Module:
    if config["task"] != "field":
        raise ValueError("The MLP stress trainer currently supports task: field only.")

    model_cfg = config["model"]
    model_type = str(model_cfg.get("type", "conditional_node_mlp")).lower()
    if model_type not in {"conditional_node_mlp", "conditional_mlp", "mlp"}:
        raise ValueError(f"Unsupported MLP model.type: {model_type}")

    conditioning_cfg = dict(model_cfg.get("conditioning", {}))
    two_stage_cfg = get_two_stage_rmises_cfg(config)
    peak_cfg = get_stress_peak_relative_cfg(config)
    stress_output_dim = 2 if bool(peak_cfg["enabled"]) else int(sample_case.target_normalized.size(-1))

    return ConditionalFieldMLP(
        node_input_dim=int(sample_case.node_features.size(-1)),
        global_input_dim=int(sample_case.global_features.size(-1)),
        hidden_dim=int(model_cfg["hidden_dim"]),
        global_dim=int(model_cfg["global_dim"]),
        num_layers=int(model_cfg["num_layers"]),
        dropout=float(model_cfg["dropout"]),
        conditioning_dim=int(conditioning_cfg.get("case_dim", model_cfg["global_dim"])),
        output_dim=(1 if bool(two_stage_cfg["enabled"]) else 0) + stress_output_dim,
        use_two_stage_rmises=bool(two_stage_cfg["enabled"]),
        use_peak_relative_stress=bool(peak_cfg["enabled"]),
        head_layers=int(model_cfg.get("head_layers", 3)),
    )


class Case7MLPTrainer(BaseCase7Trainer):
    def _load_prepared_case(self, case_path: Any) -> PreparedCase:
        case = load_case_graph(
            case_path,
            node_columns=self.dataset_cfg["node_columns"],
            edge_columns=self.dataset_cfg["edge_columns"],
            target_freq_key=self.dataset_cfg["target_freq_key"],
            make_undirected=bool(self.dataset_cfg["make_undirected"]),
            cache_dir=self.cache_dir,
            node_region=self.dataset_cfg.get("node_region"),
            **get_mode_shape_loader_kwargs(self.feature_cfg),
        )
        prepared = prepare_case(
            case,
            node_scaler=self.scalers["node"],
            edge_scaler=self.scalers["edge"],
            global_scaler=self.scalers["global"],
            target_scaler=self.scalers["target"],
            task=self.task,
            use_psd=self.use_psd,
            use_freq_top3=self.use_freq_top3,
            use_frequency_scalar=self.use_frequency_scalar,
            use_frequency_relations=self.use_frequency_relations,
            clamp_negative_rmises=self.clamp_negative_rmises,
            node_columns=self.dataset_cfg["node_columns"],
            edge_columns=self.dataset_cfg["edge_columns"],
            feature_cfg=self.feature_cfg,
            two_stage_rmises_cfg=self.two_stage_rmises_cfg,
        )

        if not bool(self.config.get("model", {}).get("drop_edges", True)):
            return prepared
        if float(self.field_loss_cfg.get("physics_stress_smoothness_weight", 0.0)) > 0.0:
            raise ValueError("model.drop_edges=True is incompatible with physics_stress_smoothness_weight > 0.")

        return PreparedCase(
            name=prepared.name,
            frequency_hz=prepared.frequency_hz,
            node_features=prepared.node_features,
            edge_index=prepared.edge_index.new_empty((2, 0)),
            edge_features=prepared.edge_features.new_empty((0, prepared.edge_features.size(-1))),
            node_bc_mask=prepared.node_bc_mask,
            edge_distance=prepared.edge_distance.new_empty((0,)),
            global_features=prepared.global_features,
            target_normalized=prepared.target_normalized,
            target_metric=prepared.target_metric,
            node_graph_index=prepared.node_graph_index,
            edge_graph_index=prepared.edge_graph_index,
            graph_count=prepared.graph_count,
        )

    def _prepare(self) -> None:
        self.case_index = discover_case_index(self.dataset_cfg["root"])
        self.split_names = resolve_case_splits(self.dataset_cfg["root"], self.dataset_cfg)

        train_case_dirs = [self.case_index[name] for name in self.split_names["train"]]
        val_case_dirs = [self.case_index[name] for name in self.split_names["val"]]
        test_case_dirs = [self.case_index[name] for name in self.split_names.get("test", [])]

        self.train_case_paths = expand_case_sample_paths(train_case_dirs, self.dataset_cfg)
        self.val_case_paths = expand_case_sample_paths(val_case_dirs, self.dataset_cfg)
        self.test_case_paths = expand_case_sample_paths(test_case_dirs, self.dataset_cfg)

        node_scaler, edge_scaler, global_scaler, target_scaler = fit_mlp_feature_scalers(
            train_case_paths=self.train_case_paths,
            dataset_cfg=self.dataset_cfg,
            task=self.task,
            use_psd=self.use_psd,
            use_freq_top3=self.use_freq_top3,
            use_frequency_scalar=self.use_frequency_scalar,
            use_frequency_relations=self.use_frequency_relations,
            clamp_negative_rmises=self.clamp_negative_rmises,
            feature_cfg=self.feature_cfg,
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
        self.resolved_config["stress_peak_relative"] = dict(self.stress_peak_relative_cfg)
        self.resolved_config["stress_hotspot_metric"] = dict(self.hotspot_metric_cfg)
        write_yaml(self.save_dir / "resolved_config.yaml", self.resolved_config)
