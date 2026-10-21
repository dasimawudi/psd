# 去掉模态/频率特征实验设计(训练用模态、推理不用)

日期:2026-10-21

## 目标

当前终版模型(`node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32_remove_inner_disk_floor200`,278 维特征)依赖模态分析结果(固有频率、振型、FRF)。未来部署场景不做模态分析、只有网格几何+载荷(含激励频率),因此需要一个**推理时只用 89 维推理特征**的模型。

关键认知:**训练时全量模态信息可用(低秩曲线头照常用),推理时无模态信息**。这是典型的"训练有特权特征、部署无特权特征"(privileged features)问题。

## 特征分类(278 维 → 89 维)

**推理可用(89 维,保留)**:
- 纯几何(51):坐标/极坐标/距离/角度/耳片局部/圆盘中心/板孔/角向周期/区域距离
- 全局结构参数(11):`earpiece_thickness`~`mass_couple_radius`
- PSD 载荷谱(12):`psd_1..4_value_log/imag/freq`
- 工况频率(4):`frequency`、`log_frequency`、`psd_value_at_frequency`、`log_psd_value_at_frequency`
- mask(11)

**模态分析相关(189 维,去掉)**:
- 频率块(11):`freq_top1/2/3`、`signed_delta_to_mode_1/2/3`、`abs_delta_to_mode_1/2/3`、`nearest_delta`、`first_mode_ratio`
- 模态特征块(178):`weighted_/nearest_/active*_umag`、`*_frf`、`*_grad_*`、`freq_ratio_mode_*`、`modal_detuning_mode_*`、`log_modal_amp_*`、`modal_weight_*`、`*_shape_rms`、`center_inner/outer_*` 等

## 方案选择

| 方案 | 内容 | 结论 |
|---|---|---|
| A | 直接用 89 维推理特征训练(不用模态信息),保留低秩曲线头 | **采用**,作为"无模态下限"基线 |
| B | Teacher(floor200)→Student(89 维)蒸馏,log1p 空间,λ=0.5 | **采用**,迁移模态知识 |
| C | 几何→固有频率预测器 + 主模型 | 备选,暂不做 |

A、B **并行训练**,各 100 epoch,与 floor200 对齐,便于对比。

## 代码改动(最小集)

1. **`node/case7_node_mlp/data.py`**:频率块加开关 `include_modal_frequency_features`(默认 true)。关闭时去掉上述 11 维模态频率特征,保留 `frequency/log_frequency`。
2. **`node/case7_node_mlp/trainer.py`**:加蒸馏支持——`distill_enabled`、`distill_teacher_checkpoint`、`distill_weight`;每个 batch 前向 teacher(no grad),在 log1p 空间加 `L = L_hard + λ·L_distill(student_pred_log, teacher_pred_log)`。
3. **scaler**:89 维特征重新 fit。先单独跑一次 `fit_scalers` 生成 89 维 cache(低 worker 数),A/B 复用,`overwrite_cache: false`,避免并发 fit 冲突。

## 配置

- `node/configs/node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32_remove_inner_disk_floor200_no_modal.yaml`(A)
  - base: floor200;features: `use_mode_shapes: false`、`include_modal_frequency_features: false`
  - 模型:89 维,`low_rank_curve_head: true`(frequency_head 输入自动变为 geometry+psd+激励频率)
  - init:从零(维度变,不能续训);epochs: 100;`num_workers: 4`;`eval_num_workers: 1`
- `node/configs/..._no_modal_distill.yaml`(B)
  - base: A;加 `distill: {enabled: true, teacher_checkpoint: <floor200 best.pt>, weight: 0.5}`
  - 其余同 A

## 训练执行

- 每个实验独占一个空闲 GPU,在 tmux 中运行以便观察(GPU id 在启动时用 `nvidia-smi` 选定空闲卡):
  - `tmux new-session -d -s train_no_modal 'CUDA_VISIBLE_DEVICES=<空闲gpu_a> python ...'`
  - `tmux new-session -d -s train_no_modal_distill 'CUDA_VISIBLE_DEVICES=<空闲gpu_b> python ...'`
- CPU 约束:**两个实验合计 ≤8 核** → 每实验训练 `num_workers: 4`、`eval_num_workers: 1`;预 fit scaler 也用低 worker 数。
- 预估:各 ~1 天(100 epoch,参考 floor200 约 29h)。

## 评估与对比

训练完后用同一套 eval(逐频 within25 + 最终 RMises + 分区域 + topk)对比三组:
- floor200(teacher,278 维)
- A(89 维,无模态)
- B(89 维,蒸馏)

关注指标:
- **A vs floor200**:模态信息价值(预期 A 明显下降)
- **B vs A**:蒸馏有效性(B 应追回部分差距)

## 风险

- A 可能明显退化(正是要量化的事实)。
- 蒸馏 student 受 teacher 质量约束;teacher 有 floor200 口径潜在偏差。
- 89 维时频率头输入只剩 `frequency/log_frequency/psd_value_at_frequency`,低秩部分表达变弱,模型更依赖点残差分支——保留低秩头仍可运行,但贡献会变小。
- 若 B 效果不佳,备选方案 C(几何→固有频率预测器)。

## 交付物

- 2 个新 config
- data.py / trainer.py 的最小改动
- 89 维 scaler cache
- A、B 训练产出(history.csv / best.pt / metrics)
- 三组对比评估报告
