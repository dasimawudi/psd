# v6_fullpart_disk_center 特征方案

> 目标：基于 `node_mlp_v6_earpiece_allpoints` 做全区域点级实验，不再只看耳片区域；同时显式建模圆盘中心和圆盘四个通孔孔壁应力集中区。  
> 状态：方案对齐文档，尚未绑定代码实现。

---

## 1. 实验定位

`v6_fullpart_disk_center` 不是简单把 `node_scope: earpiece` 改成 `all_nodes`。

本实验的核心假设是：

```text
全区域预测需要同时表达：
1. v6 已验证的频率/PSD/模态/FRF 特征；
2. case7_gnn 中的全场圆盘几何和邻域思想；
3. 圆盘中心、圆盘四个通孔孔壁、耳片孔、耳片根部这几类应力集中区域的显式区域先验。
```

其中“圆盘中心”和“四个圆盘通孔孔壁”只作为应力集中位置先验，不直接等价于高应力标签。是否高响应仍由当前频率、PSD、模态 FRF、模态梯度和目标数据共同决定。

---

## 2. 点域与数据范围

建议配置：

```yaml
dataset:
  node_scope: all_nodes
  exclude_bc_nodes: true
  exclude_center_node: false
```

说明：

- `all_nodes`：覆盖圆盘、圆盘四个通孔、中心耦合区、耳片连接区、耳片孔附近和其它背景点。
- `exclude_bc_nodes: true`：固定边界节点通常属于耳片孔约束区，响应物理含义不同，第一版先排除。
- `exclude_center_node: false`：中心是本实验重点区域，不能排除中心代表点。

如果后续发现 `center_node_mask` 的单点标签噪声过强，再单独做 ablation。

---

## 3. 特征总览

第一版特征分 7 组：

| 组别 | 特征主题 | 来源/借鉴 |
| --- | --- | --- |
| A | v6 基础几何和 mask | 当前 `case7_node_mlp` |
| B | 圆盘中心应力集中特征 | 本实验新增，基于 `nodes.csv` mask 和固定几何 |
| C | 圆盘四通孔/圆盘角向周期特征 | 借鉴 `case7_gnn` 全场几何，再扩展 |
| D | 耳片局部特征 | 沿用 v6 |
| E | 区域 mask 和到应力区距离 | 沿用 v6/fullpart 诊断 |
| F | PSD、频率、FRF、模态振型特征 | 沿用 v6 |
| G | 模态梯度/模态 baseline 特征 | 沿用 v6，等价于部分边邻域信息 |

第一版不直接加入 edge aggregate 特征。edge aggregate 放到下一组 `v6_fullpart_disk_center_edgeagg` 做 ablation，避免一次性扩大变量。

---

## 4. A 组：基础几何特征

沿用当前 v6 的基础几何：

| 特征 | 定义 | 作用 |
| --- | --- | --- |
| `x_norm` | `x / plate_radius` | 全局 x 相对位置 |
| `y_norm` | `y / plate_radius` | 全局 y 相对位置 |
| `z_norm` | `z / plate_thickness` | 厚度方向相对位置 |
| `r_norm` | `sqrt(x^2 + y^2) / plate_radius` | 圆盘径向位置 |
| `dist_to_edge` | `(plate_radius - r) / plate_radius` | 到圆盘外边缘的 signed 距离 |
| `sin_theta` | `sin(atan2(y, x))` | 角向位置，避免角度断点 |
| `cos_theta` | `cos(atan2(y, x))` | 角向位置，避免角度断点 |

这组与 `case7_gnn` 的 `augment_node_physics` 基础圆盘几何一致，是全区域实验的底座。

---

## 5. B 组：圆盘中心应力集中特征

中心区域来源：

- `global.json -> fixed_geometry.mass_couple_radius`
- `global.json -> nodes_csv_mask_definition.center_couple_mask_radius`
- `nodes.csv -> center_node_mask`
- `nodes.csv -> center_couple_mask`

建议新增或明确保留：

| 特征 | 定义 | 作用 |
| --- | --- | --- |
| `center_radius_local` | `r / mass_couple_radius` | 相对中心耦合半径的位置 |
| `center_couple_signed` | `(mass_couple_radius - r) / mass_couple_radius` | 是否位于中心耦合半径内 |
| `near_center_couple` | `sigmoid(k * center_couple_signed)` | 中心耦合 soft prior，沿用 stress_only/case7 思路 |
| `center_region_r_over_mask_radius` | `r / center_couple_mask_radius` | 相对导出中心 mask 半径的位置 |
| `center_region_signed` | `(center_couple_mask_radius - r) / center_couple_mask_radius` | 是否靠近导出中心应力区域 |
| `near_center_region_exp` | `exp(-max(r - center_couple_mask_radius, 0) / tau_center)` | 中心应力集中 soft distance |
| `center_node_mask` | `nodes.csv` 原始 mask | 标记中心代表节点 |
| `center_couple_mask` | `nodes.csv` 原始 mask | 标记中心导出区域 |

建议默认参数：

```yaml
features:
  center_region_tau: 8.0
  center_couple_sigmoid_gain: 5.0
```

注意：

- `mass_couple_radius` 表示仿真耦合半径，通常大于 `center_couple_mask_radius`。
- `center_couple_mask_radius` 表示导出/诊断用的小中心区域，更接近“中心应力集中区域”的局部标注。
- 两者都保留，避免把大耦合区和真实中心热点混成一个概念。

---

## 6. C 组：圆盘四通孔与角向周期特征

### 6.1 圆盘四通孔应力集中特征

圆盘四个通孔也是圆盘侧明确的应力集中区域，第一版必须和中心区一样显式建模，不能只依赖 `x/y/r/theta` 让模型自己学习。

通孔参数来源：

```text
fixed_geometry.plate_HoleCount
fixed_geometry.plate_HoleDist
fixed_geometry.plate_HoleRadius
```

孔中心按现有区域评估逻辑生成：

```text
angle_i = 2π * i / plate_HoleCount
center_i = [-plate_HoleDist * sin(angle_i), plate_HoleDist * cos(angle_i)]
```

建议新增：

| 特征 | 定义 | 作用 |
| --- | --- | --- |
| `dist_to_plate_hole_center_over_radius` | `min_dist(node, plate_hole_centers) / plate_HoleRadius` | 到最近通孔中心的局部距离 |
| `dist_to_plate_hole_edge_over_radius` | `(min_dist - plate_HoleRadius) / plate_HoleRadius` | 到最近通孔孔壁的 signed/local 距离 |
| `dist_to_plate_hole_edge_over_plate_radius` | `(min_dist - plate_HoleRadius) / plate_radius` | 到最近通孔孔壁的全局尺度距离 |
| `near_plate_hole_wall` | `exp(-abs(min_dist - plate_HoleRadius) / tau_plate_hole)` | 通孔孔壁 soft prior，两侧都衰减 |
| `plate_hole_wall_mask` | `nodes.csv` 原始 mask | 标记四个通孔孔壁区域 |
| `dist_to_plate_hole_region_over_plate_radius` | 到 `plate_hole_wall_mask` 最近点距离 / `plate_radius` | 到已标注通孔应力区的距离 |

建议默认参数：

```yaml
features:
  plate_hole_wall_tau: 2.0
```

### 6.2 角向周期特征

当前已有：

```text
sin_theta
cos_theta
```

建议新增：

| 特征 | 含义 |
| --- | --- |
| `sin_ear_period_theta` | `sin(earpiece_Count_default * theta)` |
| `cos_ear_period_theta` | `cos(earpiece_Count_default * theta)` |
| `sin_plate_hole_period_theta` | `sin(plate_HoleCount * theta)` |
| `cos_plate_hole_period_theta` | `cos(plate_HoleCount * theta)` |

当前固定几何下一般是：

```text
earpiece_Count_default = 3
plate_HoleCount = 4
```

即：

```text
sin(3θ), cos(3θ)
sin(4θ), cos(4θ)
```

作用：

- `sin(3θ)/cos(3θ)`：表达三耳片 120° 周期。
- `sin(4θ)/cos(4θ)`：表达四个圆盘通孔 90° 周期。
- 让相同重复结构位置在特征空间更接近。

中心点附近 `theta` 物理意义较弱，因此中心区域仍以 B 组中心距离/mask 特征为主。

---

## 7. D 组：耳片局部特征

沿用 v6：

| 特征 | 作用 |
| --- | --- |
| `ear_local_u_over_hole_radius` | 最近耳片轴向局部坐标 |
| `ear_local_v_over_half_width` | 最近耳片横向局部坐标 |
| `ear_local_r_over_hole_radius` | 相对耳片孔中心的局部半径 |
| `ear_local_sin` / `ear_local_cos` | 耳片局部角向位置 |
| `dist_to_ear_axis_over_half_width` | 到耳片中心线距离 |
| `dist_to_ear_root_signed_over_hole_radius` | 到耳片根部 signed 距离 |
| `dist_to_ear_root_abs_over_hole_radius` | 到耳片根部绝对距离 |
| `near_ear_root` | 耳片根部 soft prior |
| `dist_to_hole_center_over_hole_radius` | 到耳孔中心距离 |
| `near_hole_root_bridge` | 耳孔到根部传力通道 prior |

说明：

- 全区域实验中，圆盘内部点也会有“最近耳片局部坐标”，但其物理含义弱于耳片区域。
- 第一版保留这些特征，因为耳片根部/耳孔仍是重要热点区域。

---

## 8. E 组：区域 mask 与应力区距离

必须保留：

```text
center_node_mask
center_couple_mask
plate_hole_wall_mask
ear_hole_wall_mask
ear_connection_fillet_mask
ear_connection_earside_mask
ear_connection_mask
stress_region_mask
```

必须保留距离特征：

```text
dist_to_center_couple_region_over_plate_radius
dist_to_plate_hole_region_over_plate_radius
dist_to_ear_hole_region_over_plate_radius
dist_to_ear_connection_region_over_plate_radius
dist_to_nearest_stress_region_over_plate_radius
```

这些来自 fullpart 诊断实验已有逻辑，用于告诉点模型不同应力集中区域的位置。

本实验中至少需要单独关注两个圆盘侧区域：

```text
disk_center_region = center_couple_mask OR center_node_mask
disk_plate_hole_region = plate_hole_wall_mask
```

这两个区域不能只合并进 `stress_region_mask` 后丢失身份，因为中心耦合区和四个通孔孔壁的应力成因不同。

---

## 9. F 组：PSD、频率、FRF 和模态振型

沿用 v6，不做降级：

```text
psd_points
psd_value_at_frequency
log_psd_value_at_frequency
frequency
log_frequency
freq_top1/top2/top3
signed_delta_to_mode_1/2/3
abs_delta_to_mode_1/2/3
nearest_delta
first_mode_ratio
modal_frf_damping_ratios
modal_frf_shape_features
active top-k mode shape features
nearest mode features
```

原因：

- 全区域热点不是只由几何决定。
- 中心区和四个通孔孔壁是否产生高应力，需要当前频率与模态响应共同决定。
- v6 已经验证 FRF 和模态梯度特征比单纯频率差值更有效。

---

## 10. G 组：模态梯度和模态 baseline

沿用 v6：

```text
weighted_grad_umag_mean_frf
weighted_grad_umag_max_frf
weighted_grad_vector_mean_frf
weighted_grad_vector_max_frf
active*_grad_*_frf
modal_baseline_log_grad_umag_mean
modal_baseline_log_grad_umag_max
modal_baseline_log_grad_vector_mean
```

这部分是对 `case7_gnn` 边/邻域思想的第一层融合：

- GNN 通过 `edges.csv` 做 message passing。
- v6 MLP 没有 message passing。
- 模态梯度特征利用 `edges.csv` 计算节点邻边上的振型差分，给 MLP 补充“局部形变剧烈程度”。

因此第一版先保留模态梯度，不额外加 edge aggregate。

---

## 11. 暂不加入第一版的特征

以下特征建议放到下一组 ablation：

```text
node_degree
edge_dist_mean/min/max/std
edge_dx_abs_mean
edge_dy_abs_mean
edge_dz_abs_mean
neighbor_center_couple_mask_ratio
neighbor_plate_hole_wall_mask_ratio
neighbor_stress_region_mask_ratio
```

原因：

- 这些更直接借鉴 `case7_gnn` 的 edge attr 和邻域传播。
- 但会明显改变输入维度和 scaler。
- 第一版应先验证“全区域 + 圆盘中心显式特征”是否有效。

下一版命名建议：

```text
v6_fullpart_disk_center_edgeagg
```

---

## 12. 推荐配置差异

相对 `node_mlp_v6_earpiece_allpoints.yaml`：

```yaml
dataset:
  node_scope: all_nodes
  exclude_bc_nodes: true
  exclude_center_node: false

features:
  include_earpiece_local_features: true
  include_node_region_masks: true
  include_stress_region_distance_features: true
  include_disk_center_features: true
  include_plate_hole_features: true
  include_angular_periodic_features: true
  center_region_tau: 8.0
  center_couple_sigmoid_gain: 5.0
  plate_hole_wall_tau: 2.0
```

scaler 必须重算：

```yaml
scaler:
  cache_path: node/cache/node_mlp_v6_fullpart_disk_center_scalers.pt
  overwrite_cache: true
  reuse_target_stats_from_cache: false
```

初始化建议：

```yaml
training:
  init_checkpoint: node/outputs/node_mlp_v6_earpiece_allpoints/best.pt
  allow_partial_init_checkpoint: true
  save_dir: node/outputs/node_mlp_v6_fullpart_disk_center
```

说明：

- 全区域目标分布和耳片目标分布不同，不能复用耳片 target stats。
- 新特征导致 input schema 变化，必须 partial init。

---

## 13. 需要对齐的决策点

1. `exclude_center_node` 是否保持 `false`。
   - 建议：先保持 `false`，因为中心是目标区域。

2. 是否把 `center_couple_mask_radius` 作为独立尺度。
   - 建议：保留。它和 `mass_couple_radius` 不是同一个物理概念。

3. `near_plate_hole_wall` 使用 `abs(edge distance)` 还是只对孔外距离衰减。
   - 建议：使用 `abs(min_dist - plate_HoleRadius)`，因为孔壁两侧都可能是应力集中过渡区。

4. 第一版是否加入 edge aggregate。
   - 建议：不加，放到下一组 ablation。

5. 是否把圆盘中心和四个通孔孔壁都作为独立评估桶输出。
   - 建议：是。至少输出 `disk_center_region` 和 `disk_plate_hole_region` 的 log MAE、top5 log MAE、within25。
