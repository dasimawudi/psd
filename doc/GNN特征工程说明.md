# GNN特征工程说明

## 1. 文档范围

本文档汇总当前仓库内两套 GNN 实现的输入特征定义：

- 主线实现：`case7_gnn/`
- 独立分支：`stress_only_v1/case7_gnn_stress_only/`

重点只覆盖输入特征：

- 点特征（node features）
- 边特征（edge features）
- 全局特征（global features）

不展开监督标签、损失函数和模型结构细节。

---

## 2. 特征来源总览

两套实现的原始输入来源基本一致：

- `nodes.csv`
  - 提供节点级几何坐标和边界标记
- `edges.csv`
  - 提供网格拓扑和边几何
- `global.json`
  - 提供 case 级几何参数、模态频率、PSD 和固定几何信息
- `mode_shapes/*.csv` 或 `modal_frequencies.csv`
  - 仅 `stress_only_v1` 的可选模态振型特征会用到

原始字段定义可参考 [数据说明.md](./数据说明.md)。

---

## 3. 主线 `case7_gnn`

实现位置：

- `case7_gnn/data.py`
- `case7_gnn/trainer.py`

### 3.1 点特征

#### 3.1.1 基础点特征

默认节点列由 `dataset.node_columns` 控制，默认值为：

```yaml
node_columns: [x, y, z, bc_mask]
```

对应特征如下：


| 特征名       | 含义                          | 来源                                |
| --------- | --------------------------- | --------------------------------- |
| `x`       | 节点 x 坐标，单位 `mm`             | `nodes.csv`                       |
| `y`       | 节点 y 坐标，单位 `mm`             | `nodes.csv`                       |
| `z`       | 节点 z 坐标，单位 `mm`             | `nodes.csv`                       |
| `bc_mask` | 边界条件掩码，`1` 为约束节点，`0` 为非约束节点 | `nodes.csv`，或由 `global.json` 推导生成 |


说明：

- 如果 `nodes.csv` 中存在 `bc_mask`，直接使用。
- 如果请求了 `bc_mask` 但 `nodes.csv` 中没有，则代码会根据 `global.json` 中的耳孔几何信息自动生成边界掩码。

#### 3.1.2 物理增强点特征

当下面开关打开时：

```yaml
features:
  augment_node_physics: true
```

会在基础点特征后追加以下派生特征：


| 特征名                         | 定义                                                                                           |
| --------------------------- | -------------------------------------------------------------------------------------------- |
| `x_norm`                    | `x / plate_radius`                                                                           |
| `y_norm`                    | `y / plate_radius`                                                                           |
| `z_norm`                    | `z / plate_radius`                                                                           |
| `z_thickness_norm`          | `z / earpiece_thickness`                                                                     |
| `r_norm`                    | `sqrt(x^2 + y^2) / plate_radius`                                                             |
| `dist_to_edge`              | `(plate_radius - sqrt(x^2 + y^2)) / plate_radius`                                            |
| `sin_theta`                 | `sin(atan2(y, x))`                                                                           |
| `cos_theta`                 | `cos(atan2(y, x))`                                                                           |
| `dist_to_earpiece`          | 节点到最近耳片中心的距离，再除以 `plate_radius`                                                              |
| `radial_offset_to_earpiece` | `(r - mean_earpiece_radius) / plate_radius`                                                  |
| `near_boundary`             | `dist_to_edge <= boundary_band_ratio` 时为 `1`，否则为 `0`                                         |
| `near_earpiece`             | `dist_to_earpiece <= max(earpiece_band_ratio, earpiece_width / plate_radius)` 时为 `1`，否则为 `0` |


相关参数来源：

- `plate_radius = params_list[6]`
- `earpiece_thickness = params_list[0]`
- `earpiece_RadialDist = params_list[1]`
- `earpiece_TopWidth = params_list[2]`
- `earpiece_HoleTopDist = params_list[3]`

`params_list` 顺序定义见 [数据说明.md](./数据说明.md)。

补充说明：

- 耳片中心优先通过 `bc_mask` 节点分簇反推。
- 如果无法从约束节点反推，则回退到 `params_list` 中的几何参数。

### 3.2 边特征

默认边列由 `dataset.edge_columns` 控制，默认值为：

```yaml
edge_columns: [dx, dy, dz, dist]
```

对应特征如下：


| 特征名    | 含义                         | 来源          |
| ------ | -------------------------- | ----------- |
| `dx`   | 从 `src` 指向 `dst` 的 x 方向坐标差 | `edges.csv` |
| `dy`   | 从 `src` 指向 `dst` 的 y 方向坐标差 | `edges.csv` |
| `dz`   | 从 `src` 指向 `dst` 的 z 方向坐标差 | `edges.csv` |
| `dist` | 边长度，单位 `mm`                | `edges.csv` |


图结构说明：

- `src` 和 `dst` 决定边连接关系，本身不作为数值特征参与 MLP 编码。
- 默认会把图转为无向图，即每条边补一条反向边。
- 反向边特征为：

```text
[-dx, -dy, -dz, dist]
```

### 3.3 全局特征

全局特征由 `case7_gnn/data.py::build_global_features()` 拼接得到。

#### 3.3.1 基础全局特征

无论什么任务，基础全局特征始终包含：


| 特征组           | 维度  | 来源            |
| ------------- | --- | ------------- |
| `params_list` | 8   | `global.json` |


`params_list` 的 8 个元素顺序为：

1. `earpiece_thickness`
2. `earpiece_RadialDist`
3. `earpiece_TopWidth`
4. `earpiece_HoleTopDist`
5. `earpiece_TopFilletRadius`
6. `earpiece_BottomFilletRadius`
7. `plate_radius`
8. `Add_mass`

#### 3.3.2 可选全局特征

根据配置，基础全局特征后还可以继续拼接：


| 开关                                    | 特征组                | 维度          | 说明                               |
| ------------------------------------- | ------------------ | ----------- | -------------------------------- |
| `features.use_psd: true`              | `psd_points` 展平向量  | 当前数据下通常为 12 | 每个谱点按 `[PSD值, 虚部, 频率]` 展平        |
| `features.use_freq_top3: true`        | `freq_top3`        | 3           | 前三阶非零固有频率                        |
| `features.use_frequency_scalar: true` | `frequency_scalar` | 1           | 当前样本频点，仅 `per_frequency` 样本模式下可用 |


说明：

- `psd_points` 当前常见是 4 个谱点，因此展平后是 `4 x 3 = 12` 维。
- `frequency_scalar` 仅在 `dataset.sample_mode: per_frequency` 时存在。

### 3.4 主线默认配置下的特征组合

#### `configs/frequency.yaml`

- 点特征：`[x, y, z, bc_mask]`
- 边特征：`[dx, dy, dz, dist]`
- 全局特征：`params_list`

#### `configs/field.yaml`

- 点特征：`[x, y, z, bc_mask]`
- 边特征：`[dx, dy, dz, dist]`
- 全局特征：`params_list + psd_points + freq_top3`

#### `configs/field_physics.yaml`

- 点特征：`[x, y, z, bc_mask] + 物理增强特征`
- 边特征：`[dx, dy, dz, dist]`
- 全局特征：`params_list + psd_points + freq_top3`

#### `configs/field_case7new_mises_psd.yaml`

- 点特征：`[x, y, z, bc_mask] + 物理增强特征`
- 边特征：`[dx, dy, dz, dist]`
- 全局特征：`params_list + psd_points + frequency_scalar`

### 3.5 标准化方式

主线实现会分别对三类输入特征单独做标准化：

- 点特征 scaler
- 边特征 scaler
- 全局特征 scaler

标准化形式为：

```text
(feature - mean) / std
```

均值和方差只在训练集上拟合。

---

## 4. `stress_only_v1`

实现位置：

- `stress_only_v1/case7_gnn_stress_only/data.py`
- `stress_only_v1/case7_gnn_stress_only/trainer.py`

这套实现面向热点应力预测，特征工程更重，除了基础几何外，还加入了频率关系和可选模态振型特征。

### 4.1 点特征

#### 4.1.1 基础点特征

基础点特征与主线一致：

```yaml
node_columns: [x, y, z, bc_mask]
```

即：

- `x`
- `y`
- `z`
- `bc_mask`

#### 4.1.2 高可靠性增强点特征

当下面开关打开时：

```yaml
features:
  augment_high_reliability_features: true
```

会在基础点特征后追加以下派生特征：


| 特征名                                | 定义                                                                      |
| ---------------------------------- | ----------------------------------------------------------------------- |
| `x_norm`                           | `x / plate_radius`                                                      |
| `y_norm`                           | `y / plate_radius`                                                      |
| `z_norm`                           | `z / plate_thickness`                                                   |
| `r_norm`                           | `sqrt(x^2 + y^2) / plate_radius`                                        |
| `dist_to_edge`                     | `(plate_radius - r) / plate_radius`                                     |
| `sin_theta`                        | `sin(atan2(y, x))`                                                      |
| `cos_theta`                        | `cos(atan2(y, x))`                                                      |
| `earpiece_width_over_plate_radius` | `earpiece_width / plate_radius`                                         |
| `earpiece_width_over_hole_radius`  | `earpiece_width / earpiece_HoleRadius`                                  |
| `dist_to_ear_hole_edge_local`      | `(dist_to_earpiece_center - earpiece_HoleRadius) / earpiece_HoleRadius` |
| `dist_to_ear_hole_edge_global`     | `(dist_to_earpiece_center - earpiece_HoleRadius) / plate_radius`        |
| `near_ear_hole`                    | `exp(-max(dist_to_ear_hole_edge, 0) / earpiece_HoleRadius)`             |
| `center_radius_local`              | `r / mass_couple_radius`                                                |
| `center_couple_signed`             | `(mass_couple_radius - r) / mass_couple_radius`                         |
| `near_center_couple`               | `sigmoid(5 * center_couple_signed)`                                     |


这里除 `params_list` 外，还会显式使用 `global.json -> fixed_geometry` 中的固定几何：

- `plate_thickness`
- `earpiece_HoleRadius`
- `earpiece_Count_default`
- `mass_couple_radius`

#### 4.1.3 可选模态振型点特征

当下面开关打开时：

```yaml
features:
  use_mode_shapes: true
```

会从 `mode_shapes/*.csv` 中加载振型，并根据当前频率与各阶模态频率的关系计算权重，再把模态摘要拼到节点特征后面。

默认配置下常见的模态点特征包括：


| 特征名               | 说明                     |
| ----------------- | ---------------------- |
| `weighted_abs_u1` | 按模态权重加权后的 `            |
| `weighted_abs_u2` | 按模态权重加权后的 `            |
| `weighted_abs_u3` | 按模态权重加权后的 `            |
| `weighted_umag`   | 按模态权重加权后的 `U_mag`      |
| `nearest_abs_u1`  | 与当前频率最接近那一阶模态的 `       |
| `nearest_abs_u2`  | 与当前频率最接近那一阶模态的 `       |
| `nearest_abs_u3`  | 与当前频率最接近那一阶模态的 `       |
| `nearest_umag`    | 与当前频率最接近那一阶模态的 `U_mag` |
| `nearest_gap`     | 当前频率与最近模态频率的对数间隔       |
| `nearest_weight`  | 最近模态对应的模态权重            |


按配置还可以继续追加：


| 开关                                         | 追加特征               |
| ------------------------------------------ | ------------------ |
| `mode_shape_include_signed_weighted: true` | 加权后的有符号 `U1/U2/U3` |
| `mode_shape_include_all_umag: true`        | 所有选中模态的 `U_mag` 向量 |


说明：

- 模态权重由当前频率与模态频率关系计算得到。
- 支持的权重方式有：`resonance`、`log_gaussian`、`inverse_log_gap`。

### 4.2 边特征

`stress_only_v1` 的边特征与主线一致：


| 特征名    | 含义                         | 来源          |
| ------ | -------------------------- | ----------- |
| `dx`   | 从 `src` 指向 `dst` 的 x 方向坐标差 | `edges.csv` |
| `dy`   | 从 `src` 指向 `dst` 的 y 方向坐标差 | `edges.csv` |
| `dz`   | 从 `src` 指向 `dst` 的 z 方向坐标差 | `edges.csv` |
| `dist` | 边长度                        | `edges.csv` |


也同样默认转为无向图，反向边特征为：

```text
[-dx, -dy, -dz, dist]
```

### 4.3 全局特征

`stress_only_v1` 的全局特征分两层：

- 基础全局特征
- 可选模态全局特征

#### 4.3.1 基础全局特征

基础拼接逻辑与主线相似，但额外支持频率关系特征：


| 开关                              | 特征组                   | 维度          | 说明              |
| ------------------------------- | --------------------- | ----------- | --------------- |
| 固定包含                            | `params_list`         | 8           | case 几何和质量参数    |
| `use_psd: true`                 | `psd_points` 展平向量     | 当前数据下通常为 12 | `4 x 3`         |
| `use_freq_top3: true`           | `freq_top3`           | 3           | 前三阶模态频率         |
| `use_frequency_scalar: true`    | `frequency_scalar`    | 1           | 当前频点            |
| `use_frequency_relations: true` | `frequency_relations` | 8           | 当前频点与前三阶模态频率的关系 |


其中 `frequency_relations` 由以下 8 个量组成：


| 特征名                    | 维度  | 定义                                      |
| ---------------------- | --- | --------------------------------------- |
| `signed_delta_to_top3` | 3   | `(current_frequency - mode_i) / mode_i` |
| `abs_delta_to_top3`    | 3   | `abs(signed_delta_to_top3)`             |
| `nearest_delta`        | 1   | `abs_delta_to_top3` 中的最小值               |
| `first_mode_ratio`     | 1   | `current_frequency / first_mode`        |


#### 4.3.2 可选模态全局特征

当下面开关同时满足时：

```yaml
features:
  use_mode_shapes: true
  mode_shape_global_features: true
```

会把模态摘要继续追加到全局特征末尾。

这部分特征包括：


| 特征组                  | 说明                     |
| -------------------- | ---------------------- |
| `modal_frequencies`  | 选中模态的频率向量              |
| `weights`            | 当前频率对应的模态权重向量          |
| `log_gap`            | 当前频率与各阶模态频率的对数间隔       |
| `per_mode_umag_mean` | 每一阶模态的全节点平均 `U_mag`    |
| `per_mode_umag_rms`  | 每一阶模态的全节点 RMS `U_mag`  |
| `node_mean`          | 加权节点模态特征的全图均值，4 维      |
| `node_std`           | 加权节点模态特征的全图标准差，4 维     |
| `node_max`           | 加权节点模态特征的全图最大值，4 维     |
| `axis_energy`        | 三个方向的归一化模态能量占比，3 维     |
| `nearest_features`   | 最近模态的频率、gap、weight，3 维 |


如果选中了 `M` 个模态，则这部分维度为：

```text
5 * M + 18
```

其中：

- `5 * M` 来自 `modal_frequencies + weights + log_gap + per_mode_umag_mean + per_mode_umag_rms`
- `18` 来自 `node_mean(4) + node_std(4) + node_max(4) + axis_energy(3) + nearest_features(3)`

### 4.4 `stress_only_v1` 常见默认组合

以配置文件：

- `stress_only_v1/configs/stress_two_stage_peak_aux_modes_node_nearest_case7new.yaml`

为例，常见组合是：

- 点特征：
  - `[x, y, z, bc_mask]`
  - `+ 高可靠性增强点特征`
  - `+ 模态振型点特征`
- 边特征：
  - `[dx, dy, dz, dist]`
- 全局特征：
  - `params_list + psd_points + freq_top3 + frequency_scalar + frequency_relations`
  - 该配置下 `mode_shape_global_features: false`，因此模态特征默认不并入全局向量

### 4.5 标准化方式

`stress_only_v1` 与主线一样，也会分别拟合并应用：

- 点特征 scaler
- 边特征 scaler
- 全局特征 scaler

标准化方式同样是：

```text
(feature - mean) / std
```

---

## 5. 两套实现的差异总结

### 主线 `case7_gnn`

- 以基础图几何特征为主
- 可选物理增强较轻
- 全局特征主要是 `params_list`、`psd_points`、`freq_top3`、`frequency_scalar`
- 更偏通用全场任务

### `stress_only_v1`

- 在基础图几何外，显式加入更多高可靠性几何先验
- 增加当前频率与模态频率的关系特征
- 可选加入模态振型特征到节点和全局侧
- 更偏热点应力预测

---

## 6. 代码定位

如果后续需要继续追代码，建议优先看以下函数：

### 主线 `case7_gnn`

- `case7_gnn/data.py::load_case_graph`
- `case7_gnn/data.py::build_global_features`
- `case7_gnn/trainer.py::build_augmented_node_features`
- `case7_gnn/trainer.py::prepare_case`

### `stress_only_v1`

- `stress_only_v1/case7_gnn_stress_only/data.py::load_case_graph`
- `stress_only_v1/case7_gnn_stress_only/data.py::build_global_features`
- `stress_only_v1/case7_gnn_stress_only/trainer.py::build_augmented_global_features`
- `stress_only_v1/case7_gnn_stress_only/trainer.py::build_augmented_node_features`
- `stress_only_v1/case7_gnn_stress_only/trainer.py::build_mode_shape_node_features`
- `stress_only_v1/case7_gnn_stress_only/trainer.py::build_mode_shape_global_features`
