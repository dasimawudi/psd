# 第一版 MLP 点模型特征训练方案

## 1. 目标

第一版 MLP 点模型用于预测单个节点在单个频点下的 `MISES_psd_density`：

```text
(case_id, frequency_hz, node_id) -> MISES_psd_density
```

它不是替代所有 GNN 能力的最终方案，而是一个强 baseline，用来验证：

- 只用点级特征、全局工况特征和模态振型特征，能否学到主要应力分布规律。
- 相比当前 GNN，耳片区域应力、峰值应力和空间连续性损失有多大。

训练目标建议使用 log 空间：

```text
y_log = log1p(max(MISES_psd_density, 0))
y_train = (y_log - y_mean_train) / y_std_train
```

预测后反变换：

```text
y_log_pred = y_pred * y_std_train + y_mean_train
MISES_psd_density_pred = expm1(y_log_pred)
```

---

## 2. 训练集选择

### 2.1 切分原则

必须按 `case_id` 切分训练集、验证集和测试集，不允许把所有 node-frequency 点随机打散后切分。

错误方式：

```text
所有点随机切分 train / val / test
```

原因是同一个 case 的不同节点和不同频点会同时出现在训练集和测试集，造成严重数据泄漏。

正确方式：

```text
先按 case_id 切分 case，再展开每个 split 内的 frequency-node 点
```

建议沿用当前 `stress_only_v1` 的比例和随机种子，方便和 GNN 对比：

```yaml
split_mode: ratio
split_seed: 42
train_ratio: 0.8
val_ratio: 0.1
test_ratio: 0.1
```

### 2.2 频率样本范围

第一版建议和现有 GNN 配置保持一致：

```yaml
sample_mode: per_frequency
include_zero_frequency: false
min_frequency_hz: 20.0
max_frequency_hz: 2000.0
```

每个训练样本来自：

```text
case_dir/per_frequency_mises/*Hz.csv
```

标签列：

```text
MISES_psd_density
```

### 2.3 scaler 拟合范围

所有需要 `mean/std` 标准化的输入 scaler 和目标 scaler 只能用训练 split 拟合。

验证、测试和预测必须复用训练集保存下来的 scaler：

```text
fit scaler: train cases only
apply scaler: train / val / test / predict
```

---

## 3. 训练点选择

第一版先不做 hotspot/top-k/background 分层采样，而是只训练耳片区域上的所有节点。

这样做的目的：

- 先把建模范围收敛到最关心的耳片区域。
- 避免全场背景节点数量过多，稀释耳片区域的训练信号。
- 不依赖真实 hotspot 标签或动态阈值，训练点选择只由几何区域决定。

### 3.1 单条点样本定义

一条点样本为：

```text
case_id
frequency_hz
node_id
node features
case-frequency features
label: MISES_psd_density
```

训练点选择的基本单位是 `case-frequency`，也就是每个频点图独立筛选耳片节点。

### 3.2 耳片区域定义

第一版训练集只使用耳片上的点：

```text
train_nodes = nodes where earpiece_mask == true
```

优先使用和现有代码一致的耳片区域筛选逻辑：

- 如果后续实现点模型数据生成脚本，建议复用 `stress_only_v1/case7_gnn_stress_only/data.py` 里的 `dataset.node_region.type: earpiece` 思路。
- 耳片区域由 `global.json` 中的耳片径向距离、耳片宽度、孔距、板半径和固定几何共同确定。
- 选择的是耳片 corridor 内的节点，而不是依赖真实应力值筛选节点。

如果需要一个简化的几何口径，可以先定义为：

```text
earpiece_mask =
  节点位于任一耳片方向的宽度 corridor 内
  且径向位置靠近或超过 plate_radius
```

但最终建议以代码中的固定 region mask 为准，避免不同脚本里的耳片定义不一致。

### 3.3 训练节点选择策略

对每个 `case-frequency`：

```text
1. 计算或加载 earpiece_mask
2. 保留所有 earpiece_mask == true 的节点
3. 不按真实应力做 hotspot/top-k 采样
4. 不从非耳片区域采样背景点
```

也就是说，第一版训练数据是：

```text
train samples =
  all train cases
  x all selected frequencies
  x all earpiece nodes
```

### 3.4 验证和测试点选择

验证和测试也优先在耳片区域全量节点上评估，保持训练和验证范围一致。

推荐：

- 训练：每个 `case-frequency` 使用耳片区域全量节点。
- 验证：每个 `case-frequency` 使用耳片区域全量节点。
- 测试：每个 `case-frequency` 使用耳片区域全量节点。

这样得到的是耳片区域内真实的：

- 耳片区域误差
- peak 误差
- top1/top5 误差，top-k 只在耳片区域内部计算
- 空间连续性误差

如果需要和 GNN 的全场结果对比，可以额外跑全场评估，但第一版 MLP 的训练和选模先以耳片区域为准。

### 3.5 训练权重

第一版可以先使用均匀权重：

```text
earpiece node weight = 1.0
```

如果后续发现耳片区域内峰值仍然偏低，再加入应力分位加权，例如：

```text
normal active node weight = 1.0
top 5% weight = 2.0
top 1% weight = 5.0
```

如果使用 `SmoothL1Loss(reduction="none")`，最终 loss：

```text
loss = mean(point_weight * smooth_l1(pred, target))
```

---

## 4. 第一版 MLP 特征选择

### 4.1 不作为输入的字段

以下字段只能用于分组、追踪、采样或评估，不进入 MLP：

| 字段 | 用途 |
|---|---|
| `case_id` | split、分组、追踪 |
| `node_id` | 对齐标签和导出结果 |
| 原始 `x/y/z` | 用于计算几何特征和导出结果，不直接作为 MLP 输入 |
| 原始 `MISES_psd_density` | 标签 |
| `earpiece_mask` | 训练点筛选和评估分组，不作为输入 |

`node_id` 不建议输入 MLP，避免模型记住节点编号。

原始 `x/y/z` 也不直接输入 MLP。第一版只输入物理归一化后的坐标特征，避免 `x/y/z` 和 `x_norm/y_norm/z_norm` 重复表达同一位置信息。

### 4.2 输入特征总表

第一版 MLP 建议使用以下特征。

#### 节点基础特征

| 特征 | 计算方式 | 归一化 / 变换方法 |
|---|---|---|
| `bc_mask` | 来自 `nodes.csv` 或由边界条件几何推导 | 0/1 特征，保留原值，不做 mean/std |

说明：

- `x/y/z` 只作为中间字段，用于计算归一化坐标、径向距离、角度和区域距离。
- MLP 输入不包含原始 `x/y/z`。

#### 节点几何特征

| 特征 | 计算方式 | 归一化 / 变换方法 |
|---|---|---|
| `x_norm` | `x / plate_radius` | 已按 `plate_radius` 物理归一化，保留原值 |
| `y_norm` | `y / plate_radius` | 已按 `plate_radius` 物理归一化，保留原值 |
| `z_norm` | `z / plate_thickness` | 已按 `plate_thickness` 物理归一化，保留原值 |
| `r_norm` | `sqrt(x^2 + y^2) / plate_radius` | 已按 `plate_radius` 物理归一化，保留原值 |
| `dist_to_edge` | `(plate_radius - r) / plate_radius` | 已按 `plate_radius` 物理归一化，保留原值 |
| `sin_theta` | `sin(atan2(y, x))` | 范围约为 `[-1, 1]`，保留原值 |
| `cos_theta` | `cos(atan2(y, x))` | 范围约为 `[-1, 1]`，保留原值 |

说明：

- 这里的归一化是由几何尺寸定义的物理归一化，不是训练集 `mean/std` 标准化。
- 第一版不再对 `x_norm/y_norm/z_norm/r_norm/dist_to_edge/sin_theta/cos_theta` 做额外 z-score，避免破坏这些特征本身的物理尺度。

#### 工程区域特征

| 特征 | 计算方式 | 归一化 / 变换方法 |
|---|---|---|
| `dist_to_ear_hole_edge_local` | `(dist_to_nearest_ear_hole_center - earpiece_HoleRadius) / earpiece_HoleRadius` | 已按 `earpiece_HoleRadius` 物理归一化，保留原值 |
| `dist_to_ear_hole_edge_global` | `(dist_to_nearest_ear_hole_center - earpiece_HoleRadius) / plate_radius` | 已按 `plate_radius` 物理归一化，保留原值 |
| `near_ear_hole` | `exp(-max(dist_to_ear_hole_edge, 0) / earpiece_HoleRadius)` | 0-1 soft mask，保留原值 |
| `earpiece_width_over_plate_radius` | `earpiece_TopWidth / plate_radius` | 几何比例，保留原值 |
| `earpiece_width_over_hole_radius` | `earpiece_TopWidth / earpiece_HoleRadius` | 几何比例，保留原值 |
| `center_radius_local` | `r / mass_couple_radius` | 已按 `mass_couple_radius` 物理归一化，保留原值 |
| `center_couple_signed` | `(mass_couple_radius - r) / mass_couple_radius` | 已按 `mass_couple_radius` 物理归一化，保留原值 |
| `near_center_couple` | `sigmoid(5 * center_couple_signed)` | 0-1 soft mask，保留原值 |

说明：

- `earpiece_width_over_plate_radius` 和 `earpiece_width_over_hole_radius` 建议保留，这两个特征在现有 `stress_only_v1` 高可靠性增强特征中已经使用。
- `near_*` 特征本身是 soft mask，不需要强制标准化。

#### Case 全局几何与质量特征

| 特征 | 计算方式 | 归一化 / 变换方法 |
|---|---|---|
| `params_list` 8 维 | 来自 `global.json -> params_list` | 每一维用训练集耳片点上的 `mean/std` 标准化 |
| `plate_thickness` | 来自 `global.json -> fixed_geometry.plate_thickness` | 用训练集耳片点上的 `mean/std` 标准化 |
| `earpiece_HoleRadius` | 来自 `global.json -> fixed_geometry.earpiece_HoleRadius` | 用训练集耳片点上的 `mean/std` 标准化 |
| `mass_couple_radius` | 来自 `global.json -> fixed_geometry.mass_couple_radius` | 用训练集耳片点上的 `mean/std` 标准化 |

`params_list` 顺序：

1. `earpiece_thickness`
2. `earpiece_RadialDist`
3. `earpiece_TopWidth`
4. `earpiece_HoleTopDist`
5. `earpiece_TopFilletRadius`
6. `earpiece_BottomFilletRadius`
7. `plate_radius`
8. `Add_mass`

#### PSD 特征

| 特征 | 计算方式 | 归一化 / 变换方法 |
|---|---|---|
| `psd_points` 展平向量 | `global.json -> psd_points` 按原顺序展开 | PSD value 列先 `log1p(max(value, 0))` 再用训练集 `mean/std` 标准化；PSD 频率列直接用训练集 `mean/std` 标准化 |
| `psd_value_at_frequency` | 由 `psd_points` 对当前 `frequency` 插值得到 | 建议先 `log1p(max(value, 0))`，再用训练集 `mean/std` 标准化 |
| `log_psd_value_at_frequency` | `log1p(max(psd_value_at_frequency, 0))` | 用训练集 `mean/std` 标准化 |

注意：

- `psd_value_at_frequency` 的插值方式必须和数据生成时的 PSD 曲线定义一致。
- 如果不确定 PSD 是线性插值还是 log-log 插值，第一版可以先保留原始 `psd_points`，把 `psd_value_at_frequency` 作为可开关特征做消融。

#### 频率与模态关系特征

| 特征 | 计算方式 | 归一化 / 变换方法 |
|---|---|---|
| `frequency` | 当前频点 Hz | 用训练集耳片点上的 `mean/std` 标准化 |
| `log_frequency` | `log(max(frequency, 1e-6))` | 用训练集耳片点上的 `mean/std` 标准化 |
| `freq_top3` | `global.json` 中前三阶模态频率 | 每一阶单独用训练集 `mean/std` 标准化 |
| `signed_delta_to_top3` | `(frequency - mode_i) / mode_i`，`i=1..3` | 已按模态频率形成无量纲比值，再用训练集 `mean/std` 标准化 |
| `abs_delta_to_top3` | `abs(signed_delta_to_top3)` | 用训练集 `mean/std` 标准化 |
| `nearest_delta` | `min(abs_delta_to_top3)` | 用训练集 `mean/std` 标准化 |
| `first_mode_ratio` | `frequency / mode_1` | 已形成无量纲比值，再用训练集 `mean/std` 标准化 |

#### 节点模态振型特征

| 特征 | 计算方式 | 归一化 / 变换方法 |
|---|---|---|
| `weighted_abs_u1` | 选中模态 `abs(U1)` 按当前频率响应权重加权求和 | mode shape 先按每阶 `max_umag` 归一化；该特征再用训练集 `mean/std` 标准化 |
| `weighted_abs_u2` | 选中模态 `abs(U2)` 按当前频率响应权重加权求和 | mode shape 先按每阶 `max_umag` 归一化；该特征再用训练集 `mean/std` 标准化 |
| `weighted_abs_u3` | 选中模态 `abs(U3)` 按当前频率响应权重加权求和 | mode shape 先按每阶 `max_umag` 归一化；该特征再用训练集 `mean/std` 标准化 |
| `weighted_umag` | 选中模态 `U_mag` 按当前频率响应权重加权求和 | mode shape 先按每阶 `max_umag` 归一化；该特征再用训练集 `mean/std` 标准化 |
| `nearest_abs_u1` | 当前频率最近模态的 `abs(U1)` | mode shape 先按每阶 `max_umag` 归一化；该特征再用训练集 `mean/std` 标准化 |
| `nearest_abs_u2` | 当前频率最近模态的 `abs(U2)` | mode shape 先按每阶 `max_umag` 归一化；该特征再用训练集 `mean/std` 标准化 |
| `nearest_abs_u3` | 当前频率最近模态的 `abs(U3)` | mode shape 先按每阶 `max_umag` 归一化；该特征再用训练集 `mean/std` 标准化 |
| `nearest_umag` | 当前频率最近模态的 `U_mag` | mode shape 先按每阶 `max_umag` 归一化；该特征再用训练集 `mean/std` 标准化 |
| `nearest_gap` | 当前频率与最近模态频率的 log gap | 用训练集 `mean/std` 标准化 |
| `nearest_weight` | 最近模态对应的响应权重 | 权重归一化后通常在 `[0, 1]`，第一版仍用训练集 `mean/std` 标准化 |

模态振型建议使用绝对值特征，不建议第一版使用有符号振型：

```text
不用 signed U1/U2/U3
```

原因是模态振型的整体符号可能翻转，有符号特征会给 MLP 引入不稳定性。

---

## 5. 标准化方案

### 5.1 输入特征分类

建议把输入分成三类处理：

```text
geometry_normalized_features: 已按几何尺寸物理归一化，保留原值
scaled_continuous_features: 连续特征，用训练集 mean/std 标准化
binary_or_mask_features: 0/1 或 0-1 soft mask，保留原值
```

#### geometry_normalized_features

这类特征不再做训练集 `mean/std` 标准化：

```text
x_norm, y_norm, z_norm
r_norm, dist_to_edge
sin_theta, cos_theta
dist_to_ear_hole_edge_local, dist_to_ear_hole_edge_global
earpiece_width_over_plate_radius, earpiece_width_over_hole_radius
center_radius_local, center_couple_signed
```

这些特征已经通过 `plate_radius`、`plate_thickness`、`earpiece_HoleRadius`、`mass_couple_radius` 等几何量完成物理归一化。第一版保持原值输入 MLP。

#### scaled_continuous_features

包括：

```text
params_list 8维
plate_thickness, earpiece_HoleRadius, mass_couple_radius
psd_points 处理后的连续列
psd_value_at_frequency / log_psd_value_at_frequency
frequency, log_frequency
freq_top3
signed_delta_to_top3, abs_delta_to_top3
nearest_delta, first_mode_ratio
weighted_abs_u1/u2/u3, weighted_umag
nearest_abs_u1/u2/u3, nearest_umag
nearest_gap, nearest_weight
```

这类特征使用训练集上的 `mean/std` 标准化：

```text
x_scaled = (x - mean_train) / std_train
std_train = max(std_train, 1e-6)
```

#### binary_or_mask_features

包括：

```text
bc_mask
near_ear_hole
near_center_couple
```

第一版建议保留原始值，不参与标准化。

### 5.2 目标标准化

目标单独做 scaler，不和输入共用。

```text
y_raw = MISES_psd_density
y_log = log1p(max(y_raw, 0))
y_scaled = (y_log - y_log_mean_train) / y_log_std_train
```

保存：

```text
y_log_mean_train
y_log_std_train
```

预测反变换：

```text
y_log_pred = y_scaled_pred * y_log_std_train + y_log_mean_train
y_raw_pred = expm1(y_log_pred)
y_raw_pred = max(y_raw_pred, 0)
```

### 5.3 scaler 拟合细节

点模型中，同一个 `case-frequency` 的全局特征会在所有节点上重复。如果直接用全量训练点拟合 scaler，节点数多的 case-frequency 会更影响统计量。

第一版训练点已经限定为耳片区域，`scaled_continuous_features` 的输入 scaler 建议直接用训练 split 内所有耳片训练点拟合：

```text
fit scaled_continuous scaler on:
  train cases
  x selected frequencies
  x all earpiece nodes
```

目标 scaler 使用训练 split 内耳片区域的训练点拟合，和第一版训练分布保持一致。

---

## 6. 第一版 MLP 输入拼接顺序

建议固定 feature schema，并保存到训练产物中。

推荐顺序：

```text
1. 几何归一化特征
2. 工程区域几何特征
3. mask 特征
4. case 全局连续特征
5. PSD 连续特征
6. 频率-模态连续特征
7. 节点模态连续特征
```

训练时必须保存：

```text
feature_names.json
geometry_normalized_feature_names.json
scaled_continuous_feature_names.json
mask_feature_names.json
x_scaler.json 或 x_scaler.pt
y_scaler.json 或 y_scaler.pt
split_cases.json
earpiece_region_config.json
```

其中 `x_scaler` 只对应 `scaled_continuous_feature_names`，不作用于 `geometry_normalized_features` 和 `mask_feature_names`。

这样预测和复现实验时不会因为列顺序变化导致结果错误。

---

## 7. 第一版 MLP 建议配置

### 7.1 模型结构

第一版建议使用普通 MLP，不加复杂多头：

```text
input_dim -> 256 -> 256 -> 128 -> 1
activation: ReLU or SiLU
dropout: 0.05-0.10
normalization: LayerNorm or BatchNorm1d 可选
```

### 7.2 优化器

```text
optimizer: AdamW
lr: 1e-3
weight_decay: 1e-4
loss: SmoothL1
batch_size: 8192-65536 points
```

如果训练不稳定：

```text
lr 降到 5e-4
gradient_clip = 1.0
```

### 7.3 选模指标

不要按全场 loss 选 best。第一版建议按耳片区域验证误差选 best：

```text
selection_metric = earpiece_stress_log_mae
```

也可以同时记录但不用于第一版选模：

```text
earpiece_stress_mae
earpiece_stress_peak_relative_error
earpiece_stress_top1_mae
earpiece_stress_top5_mae
```

---

## 8. 评估指标

第一版至少输出：

```text
earpiece_stress_mae
earpiece_stress_log_mae
earpiece_stress_top1_mae
earpiece_stress_top5_mae
earpiece_stress_peak_relative_error
earpiece_stress_bias
```

点模型额外建议输出空间一致性指标：

```text
edge_smoothness_error =
  mean over edges |(pred_i - pred_j) - (true_i - true_j)|
```

该指标用于检查点模型是否预测出破碎、不连续的空间场。

---

## 9. 第一版不做的事情

为了让第一版更可控，暂时不建议加入：

- 有符号模态振型 `signed U1/U2/U3`
- 真实应力场邻域均值
- 真实 hotspot label 作为输入
- `node_id` 或 `case_id` 作为输入
- `nearest_mode_index`
- peak-relative 双模型
- edge smoothness loss
- 大量 k-hop 图统计

这些可以作为第二版或第三版增强实验。

---

## 10. 推荐落地顺序

```text
1. 固定 case split
2. 展开 per-frequency 样本
3. 对每个 case-frequency 生成节点特征、earpiece_mask 和标签
4. 保留耳片区域全量节点作为训练点
5. 用训练点拟合输入 scaler 和目标 scaler
6. 训练 SmoothL1 MLP
7. 在验证集耳片区域全量节点上计算指标并选 best
8. 在测试集耳片区域全量节点上和 GNN 区域指标对比
```

第一版是否成功，主要看：

```text
earpiece_stress_log_mae
earpiece_stress_peak_relative_error
earpiece_stress_top1/top5_mae
edge_smoothness_error
```

如果耳片区域平均误差好但 peak 差，优先调整 loss 权重或加入耳片区域内部 top-k 加权，而不是继续堆更多普通特征。
