# 去掉模态/频率特征实验(训练用模态、推理不用)实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **执行状态(2026-10-21):** 代码/配置已完成并提交(数据门控 + 蒸馏 + 配置 A/B),已验证 278/89 维正确。**训练暂缓执行**——当时机器 CPU 已过载(load 135 > 128 核,被其他用户 OpenROAD/RL 任务占用),用户决定"后面再跑"。启动时用**保守 worker(每实验训练 num_workers=2 + eval 1,合计约 6 核)**。剩余待办:**Task 5 预 fit 89 维 scaler → Task 6 启动 A/B → Task 7 评估**。

**Goal:** 训练出两个"推理时只用 89 维推理特征"的模型——A(无模态基线,89 维从零训练)和 B(从 floor200 teacher 蒸馏,log1p 空间 λ=0.5),各 100 epoch,与 floor200 对比量化模态信息价值与蒸馏有效性。

**Architecture:** 这是 privileged-features 问题。data.py 增加 `include_modal_frequency_features` 门控,使数据集可构建 89 维推理特征;A 直接用 89 维训练。B 的 batch 保留完整 278 维(teacher 吃全量),student 模型 input_dim=89,在 train_one_epoch 里按 student_feature_indices 从 278 维 batch 切片后前向;teacher 以 no_grad 前向,在 log1p 空间(经 y_scaler.inverse_transform)加蒸馏损失。

**Tech Stack:** Python + PyTorch 2.2.1(cu121,env `ci2n`)、numpy、pandas、sklearn StandardScaler。训练入口 `node/case7_node_mlp/train.py --config <yaml>`,预 fit scaler 入口 `node/case7_node_mlp/fit_scalers.py --config <yaml>`。均在 `node/` 目录下运行(数据根 `.cache/mises_psd_next1000_full_export_step0p5_balanced_v1`)。

## Global Constraints

- CPU:**两个实验合计 ≤8 核**(用户明确要求;机器共享、其他用户负载高)→ 每实验 `training.num_workers: 2` + `eval_num_workers: 1`(保守,合计约 6 核);预 fit scaler 也用低 worker 数(`scaler.num_workers: 2`)。
- GPU:每个实验独占一张空闲 A800(tmux 中训练便于用户观察)。
- 每实验 100 epoch,与 floor200 对齐。
- 蒸馏 λ=0.5,log1p 空间。
- A/B 均**从零训练**(特征维度从 278 变 89,不能续训 278 维 checkpoint)→ 两个 config 都清空 `training.init_checkpoint`、`aux_init_checkpoints`。
- `read_config` 深合并(base_config 链),顶层新增 `distill:` 键会原样传递。
- 数据/特征构造逻辑改动后,**先用 `_probe_current_feature_schema` 验证 89 维计数**,再进训练。

---

### Task 1: data.py 增加 `include_modal_frequency_features` 门控

**Files:**
- Modify: `node/case7_node_mlp/data.py:1775-1809`(`_build_base_features` 内的频率块)

**Interfaces:**
- Produces: `feature_cfg["include_modal_frequency_features"]`(默认 `True`)。`False` 时频率块只输出 `frequency`、`log_frequency` 两维。

- [ ] **Step 1: 修改频率块**

把现在的无条件频率块(`freq_top3`/`signed_delta`/`abs_delta`/`nearest_delta`/`first_mode_ratio`)包进 `if bool(feature_cfg.get("include_modal_frequency_features", True))`,基础两维(`frequency`、`log_frequency`)始终保留:

```python
    current_frequency = torch.tensor([float(frequency_hz)], dtype=torch.float32)
    safe_frequency = current_frequency.clamp_min(1e-6)
    frequency_parts = [current_frequency, safe_frequency.log()]
    frequency_names = ["frequency", "log_frequency"]
    if bool(feature_cfg.get("include_modal_frequency_features", True)):
        modes = freq_top3.clamp_min(1e-6)
        signed_delta = (safe_frequency - modes) / modes
        abs_delta = signed_delta.abs()
        frequency_parts.extend(
            [
                freq_top3,
                signed_delta,
                abs_delta,
                abs_delta.min().reshape(1),
                safe_frequency / modes[:1],
            ]
        )
        frequency_names.extend(
            [
                "freq_top1",
                "freq_top2",
                "freq_top3",
                "signed_delta_to_mode_1",
                "signed_delta_to_mode_2",
                "signed_delta_to_mode_3",
                "abs_delta_to_mode_1",
                "abs_delta_to_mode_2",
                "abs_delta_to_mode_3",
                "nearest_delta",
                "first_mode_ratio",
            ]
        )
    scaled_parts.append(torch.cat(frequency_parts, dim=0).unsqueeze(0).expand(node_count, -1))
    scaled_names.extend(frequency_names)
```

- [ ] **Step 2: 验证 89 维计数**

用探针脚本确认门控后特征数=89(并打印 89 个名字核对分类 51 几何 + 11 全局 + 12 PSD + 4 频率 + 11 mask):

```bash
cd /data-ssd/libo/psd/node
/data1/libo/miniconda3/envs/ci2n/bin/python3 - <<'EOF'
import sys; sys.path.insert(0, '.')
from case7_node_mlp.runtime import read_config
from case7_node_mlp.data import discover_case_index, expand_case_sample_paths, resolve_case_splits
from case7_node_mlp.trainer import _probe_current_feature_schema
c = read_config('configs/node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32_remove_inner_disk_floor200.yaml')
ds, fc = dict(c['dataset']), dict(c['features'])
ci = discover_case_index(ds['root'])
splits = resolve_case_splits(ds['root'], ds)
train_paths = expand_case_sample_paths([ci[n] for n in splits['train']], ds)
fc_on = dict(fc)
fc_off = dict(fc); fc_off['use_mode_shapes'] = False; fc_off['include_modal_frequency_features'] = False
s_full = _probe_current_feature_schema(train_paths, ds, fc_on)
s_student = _probe_current_feature_schema(train_paths, ds, fc_off)
print('full dim:', s_full['input_dim'], '| student dim:', s_student['input_dim'])
assert s_full['input_dim'] == 278, s_full['input_dim']
assert s_student['input_dim'] == 89, s_student['input_dim']
for n in s_student['feature_names']:
    print(n)
EOF
```

Expected: `full dim: 278 | student dim: 89`,并列出 89 个特征名,与 spec 分类核对。

- [ ] **Step 3: 提交**

```bash
cd /data-ssd/libo/psd && git add node/case7_node_mlp/data.py && git commit -m "feat: gate modal-frequency features behind include_modal_frequency_features"
```

---

### Task 2: trainer.py 增加蒸馏支持

**Files:**
- Modify: `node/case7_node_mlp/trainer.py`:
  - `NodeMLPTrainer.__init__`(~2720-2746):读取并保存 `self.distill_cfg = dict(config.get("distill", {}) or {})`。
  - `_prepare`(~2867 模型构建区):distill 时计算 student schema/indices、构建 student 模型(89 维)、加载 teacher(278 维);否则照旧。
  - `fit`(~3056):把 teacher/distill 参数传给 `train_one_epoch`。
  - `train_one_epoch`(2097-2292):distill 时从 278 维 batch 切片 student 输入、teacher no_grad 前向、加 log1p 空间蒸馏损失。

**Interfaces:**
- Consumes: `config["distill"]` = `{enabled: bool, teacher_checkpoint: str, weight: float}`;`self.feature_schema["feature_names"]`;`_probe_current_feature_schema(paths, dataset_cfg, feature_cfg)`(已存在)。
- Produces: `self.distill_enabled`、`self.teacher`(或 None)、`self.student_feature_indices`(list[int],89 个,索引到 278 维 batch)。`train_one_epoch` 新增可选参数 `teacher=None, distill_weight=0.0, student_feature_indices=None`。

- [ ] **Step 1: `__init__` 记录 distill 配置**

在 `NodeMLPTrainer.__init__` 里(读取其它 config 键的位置附近)加:

```python
        self.distill_cfg = dict(config.get("distill", {}) or {})
        self.distill_enabled = bool(self.distill_cfg.get("enabled", False))
        self.teacher: torch.nn.Module | None = None
        self.student_feature_indices: list[int] | None = None
```

- [ ] **Step 2: `_prepare` 中按 distill 分支构建模型与 teacher**

把 `_prepare` 里这段(构建模型 + 加载 init_checkpoint 之前):

```python
        input_dim = int(self.feature_schema["input_dim"])
        self.model = build_model(self.config, input_dim=input_dim, feature_schema=self.feature_schema).to(self.device)
```

替换为:

```python
        if self.distill_enabled:
            student_feature_cfg = dict(self.feature_cfg)
            student_feature_cfg["use_mode_shapes"] = False
            student_feature_cfg["include_modal_frequency_features"] = False
            student_schema = _probe_current_feature_schema(
                self.train_sample_paths, self.dataset_cfg, student_feature_cfg
            )
            student_names = [str(name) for name in student_schema.get("feature_names", [])]
            full_names = [str(name) for name in self.feature_schema.get("feature_names", [])]
            name_to_index = {name: idx for idx, name in enumerate(full_names)}
            missing = [name for name in student_names if name not in name_to_index]
            if missing:
                raise ValueError(f"Student features missing from full schema: {missing}")
            self.student_feature_indices = [name_to_index[name] for name in student_names]
            student_schema["input_dim"] = len(student_names)
            self.model = build_model(
                self.config, input_dim=len(student_names), feature_schema=student_schema
            ).to(self.device)
            teacher_path = Path(str(self.distill_cfg["teacher_checkpoint"]))
            if not teacher_path.exists():
                raise FileNotFoundError(f"distill.teacher_checkpoint does not exist: {teacher_path}")
            self.logger.info("Loading distillation teacher: %s", teacher_path)
            teacher_checkpoint = torch.load(teacher_path, map_location="cpu")
            teacher = build_model(
                self.config,
                input_dim=int(self.feature_schema["input_dim"]),
                feature_schema=self.feature_schema,
            ).to(self.device)
            teacher.load_state_dict(teacher_checkpoint["model_state"])
            teacher.eval()
            self.teacher = teacher
        else:
            input_dim = int(self.feature_schema["input_dim"])
            self.model = build_model(self.config, input_dim=input_dim, feature_schema=self.feature_schema).to(self.device)
```

注意:两个 no_modal config 的 `training.init_checkpoint` 清空,后续 init_checkpoint 加载块自然跳过。

- [ ] **Step 3: `fit` 传入蒸馏参数**

把 `fit` 里的 `train_one_epoch(...)` 调用加三个参数:

```python
            train_metrics = train_one_epoch(
                model=self.model,
                loader=train_loader,
                optimizer=self.optimizer,
                y_scaler=self.y_scaler,
                device=self.device,
                point_batch_size=point_batch_size,
                grad_clip=grad_clip,
                loss_cfg=self.loss_cfg,
                logger=self.logger,
                epoch=epoch,
                progress_every_steps=progress_every_steps,
                teacher=self.teacher,
                distill_weight=float(self.distill_cfg.get("weight", 0.5)) if self.distill_enabled else 0.0,
                student_feature_indices=self.student_feature_indices,
            )
```

- [ ] **Step 4: `train_one_epoch` 加蒸馏前向与损失**

在 `train_one_epoch` 签名加三个可选参数(默认值保持非蒸馏行为不变):

```python
    teacher: torch.nn.Module | None = None,
    distill_weight: float = 0.0,
    student_feature_indices: list[int] | None = None,
```

chunk 循环开头,把:

```python
            features = host_batch.features[chunk].to(device, non_blocking=True)
            target = host_batch.target_scaled[chunk].to(device, non_blocking=True)
            weights = host_batch.point_weights[chunk].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = model(features)
            prediction = regression_output(output)
```

替换为(teacher 吃全量 278,student 吃切片 89;`host_batch.features` 在 distill 时是 278 维):

```python
            features_full = host_batch.features[chunk].to(device, non_blocking=True)
            target = host_batch.target_scaled[chunk].to(device, non_blocking=True)
            weights = host_batch.point_weights[chunk].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if teacher is not None and student_feature_indices is not None:
                with torch.no_grad():
                    teacher_output = teacher(features_full)
                teacher_pred = regression_output(teacher_output)
                features = features_full[:, student_feature_indices]
            else:
                teacher_pred = None
                features = features_full
            output = model(features)
            prediction = regression_output(output)
```

在 `loss = loss + aux_loss + low_target_loss + censored_background_loss + rank_loss` 之后、`loss.backward()` 之前,加:

```python
            if teacher_pred is not None and distill_weight > 0.0:
                pred_log = y_scaler.inverse_transform(prediction)
                teacher_log = y_scaler.inverse_transform(teacher_pred)
                distill_values = F.smooth_l1_loss(pred_log, teacher_log, reduction="none").squeeze(-1)
                distill_term = (distill_values * weights).sum() / weights.sum().clamp_min(1e-12)
                loss = loss + float(distill_weight) * distill_term
```

说明:`y_scaler.inverse_transform` 把 scaled 预测转回 log1p 空间,与 `_decode_prediction` 一致(见 trainer.py:2295)。

- [ ] **Step 5: 语法/导入自检**

```bash
cd /data-ssd/libo/psd/node && /data1/libo/miniconda3/envs/ci2n/bin/python3 -c "import ast; ast.parse(open('case7_node_mlp/trainer.py').read()); print('trainer.py syntax OK')" && /data1/libo/miniconda3/envs/ci2n/bin/python3 -c "import sys; sys.path.insert(0,'.'); import case7_node_mlp.trainer; print('import OK')"
```

Expected: `trainer.py syntax OK` 与 `import OK`。

- [ ] **Step 6: 提交**

```bash
cd /data-ssd/libo/psd && git add node/case7_node_mlp/trainer.py && git commit -m "feat: add teacher-student distillation in log1p space for no-modal student"
```

---

### Task 3: 创建配置 A(89 维无模态基线)

**Files:**
- Create: `node/configs/node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32_remove_inner_disk_floor200_no_modal.yaml`

**Interfaces:**
- Consumes: floor200 配置链;Task 1 的 `include_modal_frequency_features` 门控。
- Produces: A 实验配置——数据集 89 维,模型 input_dim=89,从零训练,100 epoch,新 89 维 scaler cache。

- [ ] **Step 1: 写配置文件**

```yaml
base_config: node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32_remove_inner_disk_floor200.yaml

features:
  # Inference-time only: no modal analysis. Drop 178 modal node features + 11 modal-frequency features.
  use_mode_shapes: false
  include_modal_frequency_features: false

scaler:
  cache_path: node/cache/node_mlp_v6_fullpart_fusion_low_rank_80g_remove_inner_disk_floor200_no_modal_scalers.pt
  target_stats_cache_path: node/cache/node_mlp_v6_fullpart_fusion_low_rank_80g_remove_inner_disk_floor200_no_modal_scalers.pt
  overwrite_cache: false
  reuse_target_stats_from_cache: true
  num_workers: 4
  sample_limit: 1024
  prefetch_factor: 2

training:
  epochs: 100
  num_workers: 4
  persistent_workers: true
  prefetch_factor: 2
  eval_num_workers: 1
  eval_prefetch_factor: 1
  # 89-dim student: cannot continue from the 278-dim floor200 checkpoint.
  init_checkpoint: null
  allow_partial_init_checkpoint: false
  aux_init_checkpoints: []
  selection_metric: composite_v6_fullpart_fusion_low_rank_80g_fast_eval_remove_inner_disk_floor200_no_modal
  selection:
    - - mode
      - weighted_sum
    - - name
      - composite_v6_fullpart_fusion_low_rank_80g_fast_eval_remove_inner_disk_floor200_no_modal
    - - components
      - fullpart_stress_miss25_rate: 0.30
        earpiece_stress_top1_miss25_rate: 0.20
        earpiece_stress_top5_miss25_rate: 0.25
        disk_stress_miss25_rate: 0.15
        earpiece_region_stress_miss25_rate: 0.35
        earpiece_stress_log_mae: 0.15
  save_dir: node/outputs/node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32_remove_inner_disk_floor200_no_modal
```

注意:替换 `training.epochs` 为 100、`num_workers` 为 4、`eval_num_workers` 为 1 是因为本配置显式覆盖 base(floor200 链里 continue32 用 num_workers:8)。

- [ ] **Step 2: 验证 config 可读且 input_dim 推导正确**

```bash
cd /data-ssd/libo/psd/node && /data1/libo/miniconda3/envs/ci2n/bin/python3 -c "
import sys; sys.path.insert(0,'.')
from case7_node_mlp.runtime import read_config
c = read_config('configs/node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32_remove_inner_disk_floor200_no_modal.yaml')
print('epochs', c['training']['epochs'], '| num_workers', c['training']['num_workers'], '| eval_num_workers', c['training']['eval_num_workers'])
print('use_mode_shapes', c['features']['use_mode_shapes'], '| include_modal_frequency_features', c['features']['include_modal_frequency_features'])
print('init_checkpoint', c['training']['init_checkpoint'])
print('save_dir', c['training']['save_dir'])
"
```

Expected: epochs=100、num_workers=4、eval_num_workers=1、use_mode_shapes=False、include_modal_frequency_features=False、init_checkpoint=None、save_dir 指向 ..._no_modal。

- [ ] **Step 3: 提交**

```bash
cd /data-ssd/libo/psd && git add node/configs/node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32_remove_inner_disk_floor200_no_modal.yaml && git commit -m "config: add 89-dim no-modal baseline (A) from scratch, 100 epoch, low workers"
```

---

### Task 4: 创建配置 B(89 维 + 蒸馏)

**Files:**
- Create: `node/configs/node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32_remove_inner_disk_floor200_no_modal_distill.yaml`

**Interfaces:**
- Consumes: Task 2 的 distill 支持;Task 3 的 A 配置;floor200 278 维 scaler cache 与 teacher checkpoint。
- Produces: B 实验配置——数据集 278 维(teacher 用),student 89 维,λ=0.5。

- [ ] **Step 1: 写配置文件**

```yaml
base_config: node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32_remove_inner_disk_floor200_no_modal.yaml

features:
  # Distillation: dataset keeps FULL 278-dim so the teacher can run; the student
  # consumes the 89-dim inference subset (derived at runtime in trainer).
  use_mode_shapes: true
  include_modal_frequency_features: true

distill:
  enabled: true
  teacher_checkpoint: node/outputs/node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32_remove_inner_disk_floor200/best.pt
  weight: 0.5

scaler:
  # Reuse the existing 278-dim floor200 scaler cache (matches full schema).
  cache_path: node/cache/node_mlp_v6_fullpart_fusion_low_rank_80g_remove_inner_disk_floor200_scalers.pt
  target_stats_cache_path: node/cache/node_mlp_v6_fullpart_fusion_low_rank_80g_remove_inner_disk_floor200_scalers.pt
  overwrite_cache: false
  reuse_target_stats_from_cache: true

training:
  selection_metric: composite_v6_fullpart_fusion_low_rank_80g_fast_eval_remove_inner_disk_floor200_no_modal_distill
  selection:
    - - mode
      - weighted_sum
    - - name
      - composite_v6_fullpart_fusion_low_rank_80g_fast_eval_remove_inner_disk_floor200_no_modal_distill
    - - components
      - fullpart_stress_miss25_rate: 0.30
        earpiece_stress_top1_miss25_rate: 0.20
        earpiece_stress_top5_miss25_rate: 0.25
        disk_stress_miss25_rate: 0.15
        earpiece_region_stress_miss25_rate: 0.35
        earpiece_stress_log_mae: 0.15
  save_dir: node/outputs/node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32_remove_inner_disk_floor200_no_modal_distill
```

注意:继承 A 的 num_workers=4 / eval_num_workers=1 / epochs=100 / init_checkpoint=null;仅覆盖 features 为全量、加 distill 块、改 scaler cache、换 save_dir 与 selection 名。

- [ ] **Step 2: 验证 config 可读**

```bash
cd /data-ssd/libo/psd/node && /data1/libo/miniconda3/envs/ci2n/bin/python3 -c "
import sys; sys.path.insert(0,'.')
from case7_node_mlp.runtime import read_config
c = read_config('configs/node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32_remove_inner_disk_floor200_no_modal_distill.yaml')
print('distill', c['distill'])
print('use_mode_shapes', c['features']['use_mode_shapes'], '| include_modal_frequency_features', c['features']['include_modal_frequency_features'])
print('epochs', c['training']['epochs'], '| num_workers', c['training']['num_workers'])
print('init_checkpoint', c['training']['init_checkpoint'])
print('save_dir', c['training']['save_dir'])
"
```

Expected: distill={enabled:True, weight:0.5, teacher_checkpoint:...floor200/best.pt}、use_mode_shapes=True、epochs=100、num_workers=4、init_checkpoint=None。

- [ ] **Step 3: 提交**

```bash
cd /data-ssd/libo/psd && git add node/configs/node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32_remove_inner_disk_floor200_no_modal_distill.yaml && git commit -m "config: add no-modal distill experiment (B) from floor200 teacher, lambda=0.5"
```

---

### Task 5: 预 fit 89 维 scaler cache

**Files:**
- Run: `node/case7_node_mlp/fit_scalers.py`(已存在)
- Produce: `node/cache/node_mlp_v6_fullpart_fusion_low_rank_80g_remove_inner_disk_floor200_no_modal_scalers.pt`

**Interfaces:**
- Consumes: Task 3 的 A 配置(`scaler.cache_path` 指向新 89 维 cache)。
- Produces: 89 维 x_scaler/y_scaler/feature_schema cache,A/B 训练加载后因 schema 匹配而跳过重 fit。

- [ ] **Step 1: 单次低 worker 预 fit(避免并发 fit 冲突)**

```bash
cd /data-ssd/libo/psd/node && /data1/libo/miniconda3/envs/ci2n/bin/python3 -m case7_node_mlp.fit_scalers \
  --config configs/node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32_remove_inner_disk_floor200_no_modal.yaml
```

(低 worker 数已由 A 配置 scaler.num_workers=4 控制;`--output` 缺省用 `scaler.cache_path`。)

- [ ] **Step 2: 验证 cache 已生成且 input_dim=89**

```bash
cd /data-ssd/libo/psd && /data1/libo/miniconda3/envs/ci2n/bin/python3 -c "
import torch
ck = torch.load('node/cache/node_mlp_v6_fullpart_fusion_low_rank_80g_remove_inner_disk_floor200_no_modal_scalers.pt', map_location='cpu')
print('input_dim:', ck['feature_schema']['input_dim'])
assert ck['feature_schema']['input_dim'] == 89
print('OK')
"
```

Expected: `input_dim: 89` 与 `OK`。

- [ ] **Step 3: 提交**

```bash
cd /data-ssd/libo/psd && git add -f node/cache/node_mlp_v6_fullpart_fusion_low_rank_80g_remove_inner_disk_floor200_no_modal_scalers.pt && git commit -m "cache: pre-fit 89-dim no-modal scalers"
```

---

### Task 6: 启动 A/B 训练(tmux、各占一张 GPU、CPU≤8 核)

**Files:**
- Run: `node/case7_node_mlp/train.py --config <yaml>`

**Interfaces:**
- Consumes: Task 3/4 的 A/B 配置;Task 5 的 89 维 scaler cache;B 还需 floor200 278 维 cache 与 teacher best.pt(均已存在)。

- [ ] **Step 1: 选两张空闲 GPU 与确认 CPU 现状**

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
uptime
```

Expected: 挑两张 memory.used≈0 的卡;记录当前 load,若远超核数需先与用户确认启动/降低 worker。

- [ ] **Step 2: 在两个 tmux 会话中分别启动**

```bash
cd /data-ssd/libo/psd/node
tmux new-session -d -s train_no_modal \
  "CUDA_VISIBLE_DEVICES=<空闲gpu_a> /data1/libo/miniconda3/envs/ci2n/bin/python3 -m case7_node_mlp.train --config configs/node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32_remove_inner_disk_floor200_no_modal.yaml 2>&1 | tee ../node/outputs/..._no_modal/train.log"
tmux new-session -d -s train_no_modal_distill \
  "CUDA_VISIBLE_DEVICES=<空闲gpu_b> /data1/libo/miniconda3/envs/ci2n/bin/python3 -m case7_node_mlp.train --config configs/node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32_remove_inner_disk_floor200_no_modal_distill.yaml 2>&1 | tee ../node/outputs/..._no_modal_distill/train.log"
tmux ls
```

Expected: 两个 session 出现;各自独占一张 GPU;每训练进程 4 worker + 主进程 ≈5 核,合计 ≤10 核(训练期),eval 时各 +1 worker。

- [ ] **Step 3: 启动后 2 分钟内确认无异常**

```bash
tmux capture-pane -t train_no_modal -p | tail -20
tmux capture-pane -t train_no_modal_distill -p | tail -20
```

Expected: 各自日志出现 "Loading scaler cache" / "Building model" 与首个训练 step;B 额外出现 "Loading distillation teacher"。若报特征维度/schema mismatch,回到 Task 1/2 排查。

---

### Task 7: 评估与对比(训练完成后)

**Files:**
- Run: 现有 eval 入口对 floor200 / A / B 三组各跑同一套评估(逐频 within25 + 最终 RMises + 分区域 + topk)。
- Produce: `node/doc/` 下对比报告。

- [ ] **Step 1: 对 A、B 各跑最终评估**

对 `..._floor200_no_modal` 与 `..._floor200_no_modal_distill` 的 best.pt 分别跑与 floor200 报告相同的 eval(最终 RMises within25、逐频 within25、分区域、topk)。

- [ ] **Step 2: 汇总三组对比表**

| 指标 | floor200(teacher,278) | A(89,无模态) | B(89,蒸馏) |
|---|---|---|---|

- [ ] **Step 3: 结论与报告**

A vs floor200 → 模态信息价值;A vs B → 蒸馏有效性。写报告并更新 `node/README.md`。

---

## Self-Review

- **Spec 覆盖**:data.py 门控(Task 1)、trainer 蒸馏(Task 2)、配置 A(Task 3)、配置 B(Task 4)、89 维 scaler(Task 5)、双实验并行启动+CPU≤8 核+tmux+每卡(Task 6)、评估对比(Task 7)——覆盖 spec 全部交付物。
- **占位符扫描**:无 TBD/TODO;所有步骤含具体代码/命令与期望输出。
- **类型/命名一致性**:`include_modal_frequency_features` 在 Task 1/3/4 一致;`distill.enabled/teacher_checkpoint/weight`、`teacher`/`distill_weight`/`student_feature_indices` 在 Task 2/4 一致;`_probe_current_feature_schema(paths, dataset_cfg, feature_cfg)` 签名与 trainer.py:2824 现有调用一致。
- **关键风险标注**:若 89 维计数不是 89,优先怀疑 data.py 其它无条件模态特征 → 回查 Task 1 Step 2 探针输出。
