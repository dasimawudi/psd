# Node 模型运行说明

本文档记录 `node/` 目录下点模型 MLP 和图模型 TransformerConv 的训练、评估、分桶分析和当前版本约定。任务目标是基于耳片区域节点，预测 per-frequency `MISES_psd_density`。

## 维护约定

以后只要修改 `node/` 下的训练逻辑、评估逻辑、特征处理、配置含义、输出文件或运行命令，都必须同步更新本 README。这个文件作为 node 模型分支的运行入口文档。

## 当前分支和版本

当前新增的全零件三类点诊断实验工作线应在分支：

```bash
git checkout dali/fullpart-diagnostic-specialists
```

上一条 v5 特征工程工作线在分支：

```bash
git checkout dali/mlp-v5-modal-gradient-baseline
```

当前已完成训练并作为主模型对比基准的 MLP 版本是：

```text
node_mlp_earpiece_case1000_v4_local_frf_mode_norm_80g
```

该版本已完成 70 epoch 训练，best checkpoint 来自 epoch 68。它基于已经训练完成的 v3 checkpoint 继续训练，并加入：

- 耳片局部坐标、根部和孔-根部 bridge 几何先验
- FRF 模态响应和频率-模态关系特征
- 额外的 `rms_umag` 模态振型归一化特征
- v3 的 sample 级 peak/top tail 辅助 loss 和 composite 组合选模指标

当前正在训练的下一版 MLP 是：

```text
node_mlp_earpiece_case1000_v5_modal_gradient_baseline
```

v4 在 v3 基础上新增三类输入先验：

- 耳片局部坐标、耳片根部距离和孔-根部连接带 soft mask。
- FRF 模态放大因子、模态权重和频率-模态比值特征。
- 额外的 `rms_umag` 模态振型内部归一化特征。

v5 在 v4 基础上继续新增三类特征，目标是改善高响应桶、top1/top5 热点和耳片孔-根部连接区误差：

- FRF-weighted 节点振型形状特征：用 2% 阻尼下的 `|H(f)|^2` 权重重新加权节点振型，并保留 active top-3 模态的节点形状。
- 模态边差分梯度 proxy：基于 `edges.csv` 计算每阶振型的 `grad_umag_mean/max` 和 `grad_vector_mean/max`，作为应变/应力集中 proxy。
- `modal_baseline_log_*` 特征：用 `psd(f) * Σ modal_gain_i * gradient_i(node)` 构造物理 baseline，先作为输入特征使用，暂不改成 residual 输出结构。

## 任务范围

- 默认历史配置只训练耳片区域点。
- 全零件诊断配置使用 `dataset.node_scope: all_nodes`，并通过 pointset CSV 过滤 D1/D2/D3 三类点。
- 使用 `per_frequency_mises/*.csv` 中的频点级 `MISES_psd_density` 标签。
- 模型预测的是标准化后的 log target。
- 特征构造前会过滤 `MISES_psd_density < 0` 的点。
- 不把 `case_id`、`node_id`、原始 `x/y/z`、`nearest_mode_index` 作为模型输入。
- 特征按 case-frequency 在线构造，不提前展开成完整点表。
- checkpoint 中保存 train split 上拟合得到的输入 scaler 和目标 scaler。

## 主要入口脚本

| 文件 | 用途 |
|---|---|
| `node/train_node_mlp.py` | 训练点级 MLP |
| `node/evaluate_node_mlp.py` | 评估 MLP checkpoint |
| `node/fit_node_scalers.py` | 预计算 MLP scaler |
| `node/bucket_eval_node_mlp.py` | 对已有 diagnostics CSV 做分桶评估 |
| `node/plot_training_curves.py` | 从 `history.csv` 绘制 loss 和指标趋势 |
| `node/sample_node_frequency_curves.py` | 抽样热点/非热点节点，绘制同一节点全频率 target/pred 曲线 |
| `node/build_fullpart_diagnostic_pointsets.py` | 构造全零件 D1/D2/D3 诊断点集、数据说明和可视化 |
| `node/train_node_transformerconv.py` | 训练 TransformerConv 图模型 |
| `node/evaluate_node_transformerconv.py` | 评估 TransformerConv checkpoint |

默认使用项目环境：

```bash
PYTHONPATH=node /data1/libo/miniconda3/envs/ci2n/bin/python ...
```

## 配置文件

| 配置 | 输出目录 | 说明 |
|---|---|---|
| `node/configs/node_mlp_earpiece_case1000.yaml` | `node/outputs/node_mlp_earpiece_case1000_v1` | 第一版 MLP baseline |
| `node/configs/node_mlp_earpiece_case1000_v2_weighted_tail.yaml` | `node/outputs/node_mlp_earpiece_case1000_v2_weighted_tail` | 加入 top tail 加权的 v2 |
| `node/configs/node_mlp_earpiece_case1000_v3_peak_tail_composite.yaml` | `node/outputs/node_mlp_earpiece_case1000_v3_peak_tail_composite` | 从 v2 继续训练，加入 peak/tail 辅助 loss 和组合选模 |
| `node/configs/node_mlp_earpiece_case1000_v4_local_frf_mode_norm.yaml` | `node/outputs/node_mlp_earpiece_case1000_v4_local_frf_mode_norm` | 从 v3 继续训练，加入耳片局部坐标、FRF 模态响应和模态归一化增强 |
| `node/configs/node_mlp_earpiece_case1000_v4_local_frf_mode_norm_80g.yaml` | `node/outputs/node_mlp_earpiece_case1000_v4_local_frf_mode_norm_80g` | v4 的 80GB 显存高并发训练配置 |
| `node/configs/node_mlp_earpiece_case1000_v4_local_frf_mode_norm_80g_shuffle.yaml` | `node/outputs/node_mlp_earpiece_case1000_v4_local_frf_mode_norm_80g_shuffle` | v4 高并发 + sample shuffle |
| `node/configs/node_mlp_earpiece_case1000_v4_local_frf_mode_norm_80g_peak_focus.yaml` | `node/outputs/node_mlp_earpiece_case1000_v4_local_frf_mode_norm_80g_peak_focus` | v4 高并发 + 更强 peak/top1 loss 和 selection 权重 |
| `node/configs/node_mlp_earpiece_case1000_v4_ablation_local_only_80g.yaml` | `node/outputs/node_mlp_earpiece_case1000_v4_ablation_local_only_80g` | 只加耳片局部坐标的 v4 消融 |
| `node/configs/node_mlp_earpiece_case1000_v4_ablation_frf_only_80g.yaml` | `node/outputs/node_mlp_earpiece_case1000_v4_ablation_frf_only_80g` | 只加 FRF 模态响应的 v4 消融 |
| `node/configs/node_mlp_earpiece_case1000_v4_ablation_mode_norm_only_80g.yaml` | `node/outputs/node_mlp_earpiece_case1000_v4_ablation_mode_norm_only_80g` | 只加 rms 模态归一化的 v4 消融 |
| `node/configs/node_mlp_earpiece_case1000_v5_modal_gradient_baseline_80g.yaml` | `node/outputs/node_mlp_earpiece_case1000_v5_modal_gradient_baseline_80g` | 从 v4 继续训练，加入 FRF-weighted 节点振型、模态梯度 proxy 和 modal baseline 特征 |
| `node/configs/node_mlp_v5_exp_region_distance_within25_50ep.yaml` | `node/outputs/node_mlp_v5_exp_region_distance_within25_50ep` | region distance + within25 选模实验，从 v5 best 初始化，跑满 50 epoch 观察长训练走势 |
| `node/configs/node_mlp_v5_diag_full_flat_background_only.yaml` | `node/outputs/node_mlp_v5_diag_full_flat_background_only_100ep` | 全零件 D1 平坦背景点 specialist，100 epoch 诊断训练 |
| `node/configs/node_mlp_v5_diag_full_sensitive_flat_only.yaml` | `node/outputs/node_mlp_v5_diag_full_sensitive_flat_only_100ep` | 全零件 D2 标注敏感区平坦点 specialist，100 epoch 诊断训练 |
| `node/configs/node_mlp_v5_diag_full_hotspot_only.yaml` | `node/outputs/node_mlp_v5_diag_full_hotspot_only_100ep` | 全零件 D3 真实热点点 specialist，100 epoch 诊断训练 |
| `node/configs/node_mlp_v5_diag_full_flat_background_only_unlimited_30ep.yaml` | `node/outputs/node_mlp_v5_diag_full_flat_background_only_unlimited_30ep` | D1 使用 unlimited 全量点集重训 30 epoch |
| `node/configs/node_mlp_v5_diag_full_sensitive_flat_only_unlimited_30ep.yaml` | `node/outputs/node_mlp_v5_diag_full_sensitive_flat_only_unlimited_30ep` | D2 使用 unlimited 全量点集重训 30 epoch |
| `node/configs/node_mlp_v5_diag_full_hotspot_only_unlimited_30ep.yaml` | `node/outputs/node_mlp_v5_diag_full_hotspot_only_unlimited_30ep` | D3 使用 unlimited 全量点集重训 30 epoch |
| `node/configs/node_transformerconv_earpiece_case1000_v1.yaml` | `node/outputs/node_transformerconv_earpiece_case1000_v1` | 基于图结构的 TransformerConv 版本 |

## 训练 MLP

训练 v1：

```bash
PYTHONPATH=node /data1/libo/miniconda3/envs/ci2n/bin/python node/train_node_mlp.py \
  --config node/configs/node_mlp_earpiece_case1000.yaml
```

训练 v2 weighted-tail：

```bash
PYTHONPATH=node /data1/libo/miniconda3/envs/ci2n/bin/python node/train_node_mlp.py \
  --config node/configs/node_mlp_earpiece_case1000_v2_weighted_tail.yaml
```

训练 v3 peak-tail-composite：

```bash
PYTHONPATH=node /data1/libo/miniconda3/envs/ci2n/bin/python node/train_node_mlp.py \
  --config node/configs/node_mlp_earpiece_case1000_v3_peak_tail_composite.yaml
```

v3 的关键训练配置：

```yaml
training:
  init_checkpoint: node/outputs/node_mlp_earpiece_case1000_v2_weighted_tail/best.pt
  epochs: 70
  lr: 0.0005
  selection_metric: composite_log_peak_tail
```

v3 的组合选模指标：

```text
score =
  1.00 * earpiece_stress_log_mae
  + 0.20 * earpiece_stress_peak_relative_error
  + 0.30 * earpiece_stress_top5_log_mae
  + 0.20 * earpiece_stress_top1_log_mae
  + 0.15 * earpiece_stress_miss25_rate
```

该指标越低越好。

训练 v4 local-frf-mode-norm：

```bash
PYTHONPATH=node /data1/libo/miniconda3/envs/ci2n/bin/python node/train_node_mlp.py \
  --config node/configs/node_mlp_earpiece_case1000_v4_local_frf_mode_norm.yaml
```

80GB 显存高并发训练 v4：

```bash
PYTHONPATH=node /data1/libo/miniconda3/envs/ci2n/bin/python node/train_node_mlp.py \
  --config node/configs/node_mlp_earpiece_case1000_v4_local_frf_mode_norm_80g.yaml
```

该配置使用：

```yaml
training:
  sample_batch_size: 64
  batch_size: 1048576
  num_workers: 16
```

其中 `sample_batch_size` 控制一个 DataLoader batch 拼多少个 case-frequency 样本，`batch_size` 控制每次送入 GPU 的最大点数。80GB 显存下可先用该配置跑；如果显存占用仍明显偏低，可继续把 `batch_size` 调到 `2097152`，再把 `sample_batch_size` 试到 `96` 或 `128`。

多卡并行实验建议：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=node /data1/libo/miniconda3/envs/ci2n/bin/python node/train_node_mlp.py \
  --config node/configs/node_mlp_earpiece_case1000_v4_local_frf_mode_norm_80g_shuffle.yaml

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=node /data1/libo/miniconda3/envs/ci2n/bin/python node/train_node_mlp.py \
  --config node/configs/node_mlp_earpiece_case1000_v4_local_frf_mode_norm_80g_peak_focus.yaml

CUDA_VISIBLE_DEVICES=2 PYTHONPATH=node /data1/libo/miniconda3/envs/ci2n/bin/python node/train_node_mlp.py \
  --config node/configs/node_mlp_earpiece_case1000_v4_ablation_local_only_80g.yaml

CUDA_VISIBLE_DEVICES=3 PYTHONPATH=node /data1/libo/miniconda3/envs/ci2n/bin/python node/train_node_mlp.py \
  --config node/configs/node_mlp_earpiece_case1000_v4_ablation_frf_only_80g.yaml

CUDA_VISIBLE_DEVICES=4 PYTHONPATH=node /data1/libo/miniconda3/envs/ci2n/bin/python node/train_node_mlp.py \
  --config node/configs/node_mlp_earpiece_case1000_v4_ablation_mode_norm_only_80g.yaml
```

这五组实验用途：

- `80g_shuffle`：验证 sample 顺序打散是否改善泛化。
- `80g_peak_focus`：验证更强 peak/top1 约束是否继续降低峰值误差。
- `ablation_local_only`：验证耳片局部坐标的单独贡献。
- `ablation_frf_only`：验证 FRF 模态响应的单独贡献。
- `ablation_mode_norm_only`：验证 rms 模态归一化的单独贡献。

v4 会改变输入特征维度，所以使用独立 scaler cache：

```yaml
scaler:
  cache_path: node/cache/node_mlp_earpiece_case1000_v4_local_frf_mode_norm_scalers.pt
  overwrite_cache: true
```

v4 从 v3 best checkpoint 初始化，并允许 feature schema 变化时做 partial warm start：

```yaml
training:
  init_checkpoint: node/outputs/node_mlp_earpiece_case1000_v3_peak_tail_composite/best.pt
  allow_partial_init_checkpoint: true
```

partial warm start 的规则是：后续层和同形状参数直接加载；第一层按相同 feature name 对齐复用旧权重；新增特征列保持新模型随机初始化。

训练 v5 modal-gradient-baseline，当前正式训练指定卡 1：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=node /data1/libo/miniconda3/envs/ci2n/bin/python node/train_node_mlp.py \
  --config node/configs/node_mlp_earpiece_case1000_v5_modal_gradient_baseline_80g.yaml
```

v5 当前配置：

```yaml
features:
  include_modal_frf_shape_features: true
  include_modal_gradient_features: true
  include_modal_baseline_feature: true

training:
  init_checkpoint: node/outputs/node_mlp_earpiece_case1000_v4_local_frf_mode_norm_80g/best.pt
  allow_partial_init_checkpoint: true
  lr: 0.0003
  sample_batch_size: 64
  batch_size: 1048576
  num_workers: 16
```

v5 会改变输入特征维度，所以使用独立 scaler cache：

```yaml
scaler:
  cache_path: node/cache/node_mlp_earpiece_case1000_v5_modal_gradient_baseline_80g_scalers.pt
  overwrite_cache: true
```

## 全零件三类点诊断实验

目的：验证当前 `case + frequency + node -> MISES_psd_density` 建模方式在三类典型区域上的表达能力。

三类点：

- D1 `full_flat_background`：全零件中远离 `nodes.csv` 标注应力集中区域、真实频谱平坦的背景点。
- D2 `full_sensitive_flat`：位于标注应力集中区域，但真实频谱平坦的点，用来检查是否乱造热点。
- D3 `full_hotspot`：真实 peak/top5%/谱动态明显的热点点，用来检查是否能拟合峰值和峰形。

`nodes.csv` 使用的区域标注：

```text
center_couple_mask
plate_hole_wall_mask
ear_hole_wall_mask
ear_connection_fillet_mask
ear_connection_earside_mask
ear_connection_mask
```

当前 pilot 点集构造命令：

```bash
PYTHONPATH=node /data1/libo/miniconda3/envs/ci2n/bin/python node/build_fullpart_diagnostic_pointsets.py \
  --config node/configs/node_mlp_earpiece_case1000_v5_modal_gradient_baseline_80g.yaml \
  --output-dir node/outputs/diagnostics/fullpart_three_point_specialists_pilot \
  --max-cases-per-split 40 \
  --max-nodes-per-case-pointset 800 \
  --seed 42
```

全量构造时去掉 `--max-cases-per-split 40` 即可。

输出文件：

```text
node/outputs/diagnostics/fullpart_three_point_specialists_pilot/
  DATA说明.md
  pointsets/selected_points_all.csv
  pointsets/selected_points_train.csv
  pointsets/selected_points_val.csv
  pointsets/selected_points_test.csv
  metrics/diagnostic_pointset_summary.csv
  plots/pointset_counts_by_split.png
  plots/pointset_region_counts.png
  plots/pointset_response_distributions.png
```

训练配置通过下面字段使用点集：

```yaml
dataset:
  node_scope: all_nodes
  exclude_bc_nodes: true
  exclude_center_node: true
  pointset:
    path: node/outputs/diagnostics/fullpart_three_point_specialists_pilot/pointsets/selected_points_all.csv
    pointset_type: full_flat_background
```

`expand_case_sample_paths` 会自动跳过 pointset 中不存在的 case，避免 pilot 点集训练时遍历完整 1000 case 的空样本。

三组诊断训练命令：

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=node /data1/libo/miniconda3/envs/ci2n/bin/python node/train_node_mlp.py \
  --config node/configs/node_mlp_v5_diag_full_flat_background_only.yaml

CUDA_VISIBLE_DEVICES=3 PYTHONPATH=node /data1/libo/miniconda3/envs/ci2n/bin/python node/train_node_mlp.py \
  --config node/configs/node_mlp_v5_diag_full_sensitive_flat_only.yaml

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=node /data1/libo/miniconda3/envs/ci2n/bin/python node/train_node_mlp.py \
  --config node/configs/node_mlp_v5_diag_full_hotspot_only.yaml
```

当前三组诊断配置均设置为 `epochs: 100`，`early_stopping_patience: 100`，用于完整观察三类点 specialist 的长训练收敛曲线。100 轮实验输出目录使用 `*_100ep` 后缀，避免覆盖早期 20 轮短跑结果。

这三个模型是 specialist 诊断模型，不作为最终全零件主模型。重点看它们在自己点集上的误差是否明显优于全零件主模型。

## Scaler 说明

训练时通常会自动拟合或加载 scaler。也可以手动预计算：

```bash
PYTHONPATH=node /data1/libo/miniconda3/envs/ci2n/bin/python node/fit_node_scalers.py \
  --config node/configs/node_mlp_earpiece_case1000.yaml
```

缓存逻辑：

- `scaler.cache_path` 存在且兼容时，直接加载已有 scaler。
- 如果 cache 不兼容，会重新计算 scaler。
- 默认不会把重新计算出来的 scaler 写回 cache。
- 只有显式设置下面参数时才覆盖 cache：

```yaml
scaler:
  overwrite_cache: true
```

这样可以避免 smoke test 或小样本实验误覆盖正式训练用的全量 scaler。

## 特征处理

当前 MLP 使用以下特征组：

- 节点归一化几何特征：
  - `x_norm`, `y_norm`, `z_norm`, `r_norm`, `dist_to_edge`, `sin_theta`, `cos_theta`
  - 耳孔距离相关特征
  - 中心耦合区距离相关特征
  - v4 可选开启耳片局部坐标、根部距离和孔-根部连接带 soft mask
- case 全局特征：
  - `params_list` 中的几何/质量参数
  - `plate_thickness`
  - `earpiece_HoleRadius`
  - `mass_couple_radius`
- PSD 特征：
  - 每个谱点的 log PSD value
  - 每个谱点的频率
  - `psd_value_at_frequency`
  - `log_psd_value_at_frequency`
- 频率和模态关系特征：
  - `frequency`
  - `log_frequency`
  - `freq_top1..3`
  - 当前频率到前三阶模态的 signed/absolute gap
  - `nearest_delta`
  - `first_mode_ratio`
  - v4 可选开启前 10 阶 `freq_ratio`、`modal_detuning`、FRF `log_modal_amp` 和 `modal_weight`
- 节点模态振型特征：
  - 加权绝对振型分量
  - 最近模态的绝对振型分量
  - `nearest_gap`
  - `nearest_weight`
  - v4 可选开启额外 `rms_umag` 内部归一化后的 weighted/nearest 振型特征
  - v5 可选开启 FRF-weighted 节点振型和 active top-3 模态节点形状
  - v5 可选开启基于 `edges.csv` 的模态边差分梯度 proxy
  - v5 可选开启 `modal_baseline_log_*` 物理 baseline 输入特征
- mask 特征：
  - `bc_mask`
  - `near_ear_hole`
  - `near_center_couple`

标准化方式：

- 归一化几何特征保持物理比例。
- v4 的耳片局部几何也保持物理无量纲比例，不进入 mean/std scaler。
- PSD、频率、全局参数和模态特征使用 train split 的 `mean/std` 标准化。
- 目标值先做：

```text
y_log = log1p(max(MISES_psd_density, 0))
```

然后再做：

```text
y_scaled = (y_log - train_mean) / train_std
```

预测时反标准化后再 `expm1` 回到原始 `MISES_psd_density` 尺度。

## Loss 版本

v1：

- scaled log target 空间上的 SmoothL1。

v2：

- SmoothL1 + target quantile 权重。
- top 5% target 点提高权重。
- top 1% target 点进一步提高权重。
- 小目标值 floor：

```yaml
target:
  zero_below: "p99*1e-8"
```

v3：

- 继承 v2 的点级权重。
- 增加 sample 级辅助 loss：
  - `sample_peak_loss_weight`
  - `sample_top5_loss_weight`
  - `sample_top1_loss_weight`
  - `sample_mean_loss_weight`

这些辅助 loss 在每个 case-frequency 样本内计算，用来缓解 peak 和 top tail 拟合不足。

v4：

- loss 与 v3 保持一致。
- 主要改动在输入特征，不改变 target transform 和 composite 选模指标。
- 训练时从 v3 checkpoint 做 partial warm start，用于验证新增物理先验是否改善最高响应桶、top1 和 top5。

## 评估 MLP

评估一个 checkpoint：

```bash
PYTHONPATH=node /data1/libo/miniconda3/envs/ci2n/bin/python node/evaluate_node_mlp.py \
  --checkpoint node/outputs/node_mlp_earpiece_case1000_v2_weighted_tail/best.pt \
  --split test \
  --output-dir node/outputs/node_mlp_earpiece_case1000_v2_weighted_tail/eval_test
```

常用快速评估参数：

```bash
--max-cases 5
--max-frames-per-case 5
--num-workers 0
--sample-batch-size 1
--point-batch-size 262144
```

默认评估会额外抽样 20 个 case，输出频率曲线诊断。每个 case 生成 2 张图：

- 1 张 `selected_nodes_frequency_curves`：同一个 case 内 2 个 hotspot 节点和 2 个 background 节点的全频率 target/pred 曲线放在同一张图中。
- 1 张 `response_buckets`：该 case 内全部有效点按 target 响应水平分桶后的 `within25` 统计图，同时显示每个桶的数据量占比和 `pred/target mean`。

曲线图的误差图例会标出该曲线的 `within25` 占比。

正式评估的 `<split>_metrics.json` 和 `<split>_diagnostics.csv` 还会输出响应排名桶指标：

- `top1`：每个 case-frequency 内真实 target 排名前 1% 的点。
- `top1_5`：真实 target 排名 1%-5% 的点。
- `top5`：真实 target 排名前 5% 的点。
- `top5_10`：真实 target 排名 5%-10% 的点。
- `top10`：真实 target 排名前 10% 的点。

这些桶主要看热点和次热点区域是否预测到位，字段包括 `within25_ratio`、`mae`、`log_mae`、`relative_mae`、`under_pred_ratio`、`over_pred_ratio` 和 `pred_target_ratio`。

```text
eval_<split>/
  <split>_metrics.json
  <split>_diagnostics.csv
  <split>_frequency_curve_summary.json
  frequency_curve_diagnostics_<split>_20cases/
    frequency_curve_points.csv
    curve_summary.csv
    case_response_buckets.csv
    summary_by_type.csv
    plots/*.png
```

可调参数：

```bash
--curve-diagnostics-cases 20
--curve-diagnostics-seed 42
--curve-diagnostics-output-dir <dir>
--curve-diagnostics-point-batch-size 1048576
--no-curve-diagnostics
```

如果只是快速 smoke test，可用 `--curve-diagnostics-cases 2` 或 `--no-curve-diagnostics`。

## 训练曲线

训练曲线脚本读取已有 `history.csv`，不重新跑模型。默认会画：

- 左图：`train_loss`、`val_loss`、`train_weighted_loss`、`val_weighted_loss`
- 右图：指定 metric 的 train/val 曲线

示例：

```bash
PYTHONPATH=node /data1/libo/miniconda3/envs/ci2n/bin/python node/plot_training_curves.py \
  --run-dir node/outputs/node_mlp_earpiece_case1000_v4_local_frf_mode_norm_80g \
  --metric earpiece_stress_log_mae
```

指定输出路径：

```bash
PYTHONPATH=node /data1/libo/miniconda3/envs/ci2n/bin/python node/plot_training_curves.py \
  --history node/outputs/node_mlp_earpiece_case1000_v4_local_frf_mode_norm_80g/history.csv \
  --metric earpiece_stress_within25_ratio \
  --output node/outputs/node_mlp_earpiece_case1000_v4_local_frf_mode_norm_80g/training_curves_within25.png
```

常用 metric：

```text
earpiece_stress_log_mae
earpiece_stress_within25_ratio
earpiece_stress_peak_relative_error
earpiece_stress_top5_log_mae
earpiece_stress_top1_log_mae
selection_score
```

## 频率曲线诊断

频率曲线诊断用于检查同一个 case、同一个节点在全部频点上的预测曲线是否跟 target 曲线一致。脚本会随机抽 case，每个 case 自动选：

- `hotspot`：该 case 内全频率 target peak 最大的耳片节点。
- `background`：该 case 内低响应非热点节点，用来检查模型是否在背景区乱造峰。

运行当前 v5 test split 的 20 个 case：

```bash
PYTHONPATH=node /data1/libo/miniconda3/envs/ci2n/bin/python node/sample_node_frequency_curves.py \
  --checkpoint node/outputs/node_mlp_earpiece_case1000_v5_modal_gradient_baseline_80g/best.pt \
  --split test \
  --num-cases 20 \
  --seed 42
```

默认输出到 checkpoint 目录下：

```text
frequency_curve_diagnostics_test_20cases/
  frequency_curve_points.csv
  curve_summary.csv
  case_response_buckets.csv
  summary_by_type.csv
  plots/*.png
```

重点看：

- `log_mae`：整条 log 曲线的平均误差。
- `within25_ratio`：非零 target 点中 25% 相对误差内的比例。
- `peak_relative_error`：该节点全频率峰值幅值误差。
- `peak_frequency_delta_hz`：预测峰值频率相对真实峰值频率的偏移。
- `freq_delta_log_mae`：相邻频点 log 斜率误差，用于检查频率方向曲线形状是否破碎。
- `log_curve_corr`：预测和 target 的 log 曲线相关性。
- `case_response_buckets.csv`：同一个 case 内按 target 响应水平分桶后的 `within25`、`pred/target mean` 和点数分布。

## 分桶评估

分桶评估读取已有 diagnostics CSV，不重新跑模型推理。

运行示例：

```bash
PYTHONPATH=node /data1/libo/miniconda3/envs/ci2n/bin/python node/bucket_eval_node_mlp.py \
  --diagnostics node/outputs/node_mlp_earpiece_case1000_v2_weighted_tail/best_test_diagnostics.csv \
  --output-dir node/outputs/node_mlp_earpiece_case1000_v2_weighted_tail/bucket_eval_test
```

输出文件：

```text
frequency_buckets.csv
target_peak_buckets.csv
target_mean_buckets.csv
bucket_summary.json
```

三个分桶含义：

- `frequency_bucket`：按当前频率分桶，用来看模型在哪些频段预测差。
- `target_peak_bucket`：按当前 case-frequency 的真实峰值分桶，用来看不同峰值强度下的热点拟合能力。
- `target_mean_bucket`：按当前 case-frequency 的真实平均应力分桶，用来看整体响应强弱下的系统性偏差。

分桶结果重点看：

- `log_mae_mean`
- `within25_ratio_mean`
- `peak_relative_error_mean`
- `peak_relative_error_p90`
- `top1_mae_mean`
- `top5_mae_mean`
- `pred_target_ratio_mean`
- `under_pred_ratio_mean`

其中：

```text
pred_target_ratio_mean < 1
```

表示该桶整体偏低估；

```text
pred_target_ratio_mean > 1
```

表示该桶整体偏高估。

## 已完成训练结果

详细结果汇总见：

```text
node/模型训练结果汇总.md
node/doc/三类点诊断实验结果汇总_20260521.md
```

已经训练完成：

```text
node/outputs/node_mlp_earpiece_case1000_v1
node/outputs/node_mlp_earpiece_case1000_v2_weighted_tail
node/outputs/node_mlp_earpiece_case1000_v3_peak_tail_composite
node/outputs/node_transformerconv_earpiece_case1000_v1
node/outputs/node_mlp_v5_diag_full_flat_background_only_100ep
node/outputs/node_mlp_v5_diag_full_sensitive_flat_only_100ep
node/outputs/node_mlp_v5_diag_full_hotspot_only_100ep
```

待训练 / 当前开发：

```text
node/outputs/node_mlp_earpiece_case1000_v4_local_frf_mode_norm
```

test 集整体指标：

| 指标 | v1 | v2 weighted-tail | v3 peak-tail | TransformerConv |
|---|---:|---:|---:|---:|
| `log_mae` | 0.4210 | 0.4163 | 0.3546 | 0.4654 |
| `within25_ratio` | 0.3972 | 0.4089 | 0.4871 | 0.3384 |
| `peak_relative_error_mean` | 0.3793 | 0.3582 | 0.1858 | 0.4230 |
| `top1_mae` | 4.021e7 | 4.049e7 | 2.400e7 | 4.179e7 |
| `top5_mae` | 1.333e7 | 1.369e7 | 8.200e6 | 1.752e7 |
| `pred_mean / target_mean` | 0.648 | 0.598 | 0.874 | 1.466 |

当前判断：

- v3 是当前主模型，best epoch 为 70。
- v3 相比 v2 显著改善了整体误差、peak relative error 和 top1/top5 tail 误差。
- TransformerConv 当前版本 early stopping 于 epoch 17，best epoch 为 9；后期过拟合并明显高估，不作为当前主模型。
- v3 仍需关注最高 `target_peak_bucket` 和最高 `target_mean_bucket` 的极端高响应样本。

## TransformerConv 图模型

训练：

```bash
PYTHONPATH=node /data1/libo/miniconda3/envs/ci2n/bin/python node/train_node_transformerconv.py \
  --config node/configs/node_transformerconv_earpiece_case1000_v1.yaml
```

评估：

```bash
PYTHONPATH=node /data1/libo/miniconda3/envs/ci2n/bin/python node/evaluate_node_transformerconv.py \
  --checkpoint node/outputs/node_transformerconv_earpiece_case1000_v1/best.pt \
  --split test
```

该版本依赖 `torch_geometric`。它复用 MLP 的 target transform、scaler、split、metrics 和 diagnostics，但会根据 `edges.csv` 为每个 case-frequency 构建子图。

## 输出文件

训练输出：

```text
best.pt
metrics.json
history.csv
resolved_config.yaml
feature_schema.json
train.log
best_val_diagnostics.csv
best_test_diagnostics.csv
```

checkpoint 内容：

```text
model_state
config
x_scaler
y_scaler
feature_schema
metrics
```

diagnostics 是 case-frequency 级别，每行对应一个频点样本，包含：

```text
frequency_hz
points
loss
mae / rmse
log_mae / log_rmse
bias
relative_mae
symmetric_relative_mae
within25_ratio
target_mean / pred_mean
target_peak / pred_peak
peak_relative_error
top1_mae / top1_log_mae
top5_mae / top5_log_mae
top1_within25_ratio
top1_5_within25_ratio
top5_within25_ratio
top5_10_within25_ratio
top10_within25_ratio
under_pred_ratio / over_pred_ratio
```

注意：旧的 v1/v2 diagnostics 没有 `top1_log_mae`、`top5_log_mae` 和响应排名桶 `within25` 字段，v3 以及之后新评估的版本会有。

## Smoke Test

快速检查时可以复制一份配置到 `/tmp`，加上：

```yaml
dataset:
  max_cases: 3
  max_frames_per_case: 2
training:
  epochs: 1
  num_workers: 0
  sample_batch_size: 1
  write_diagnostics: false
```

smoke test 不要设置：

```yaml
scaler:
  overwrite_cache: true
```

避免误覆盖正式 scaler cache。
