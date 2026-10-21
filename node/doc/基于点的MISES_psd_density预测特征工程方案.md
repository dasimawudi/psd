# 基于点的 MISES_psd_density 预测特征工程方案

## 1. 建模目标

本方案用于将原来的 GNN 全场预测建模方式，改造成基于单节点样本的 `MISES_psd_density` 预测方式。

原 GNN 建模方式为：

```text
一个 case + 一个频率下的整张网格图
→ 当前频率下所有节点的 MISES_psd_density
```

新的点模型建模方式为：

```text
一个 case + 一个频率 + 一个节点
→ 该节点在该频率下的 MISES_psd_density
```

即每一条训练样本为：

```text
(case_id, frequency, node_id) -> MISES_psd_density
```

推荐训练目标为：

```text
y = log1p(max(MISES_psd_density, 0))
```

预测后再反变换：

```text
MISES_psd_density = expm1(y_pred)
```

这样可以缓解应力值动态范围过大、高应力热点主导训练不稳定的问题。

---

## 2. 样本表结构

建议最终构造一张点级训练表，例如：

```text
point_stress_dataset.parquet
```

每一行是一条 node-frequency 样本。

| 字段 | 是否作为特征 | 说明 |
|---|---:|---|
| `case_id` | 否 | 用于分组、划分训练集、追踪来源，不建议作为模型输入 |
| `node_id` | 否 / 可选 | 不建议直接作为特征，避免模型记忆节点编号 |
| `frequency` | 是 | 当前频点 |
| `x, y, z` | 是 | 节点坐标 |
| `bc_mask` | 是 | 是否约束节点 |
| `global_features` | 是 | case 几何、PSD、模态频率等 |
| `modal_node_features` | 强烈建议 | 当前频率相关的节点振型特征 |
| `neighbor_features` | 建议 | 用来弥补不用 GNN 后丢失的邻域信息 |
| `MISES_psd_density` | 标签 | 原始目标 |
| `log_mises_psd_density` | 标签 | 训练目标 |

---

## 3. 特征分组设计

### A. 节点基础特征

来源：`nodes.csv`

| 特征 | 说明 | 是否必选 |
|---|---|---:|
| `x` | 节点 x 坐标 | 必选 |
| `y` | 节点 y 坐标 | 必选 |
| `z` | 节点 z 坐标 | 必选 |
| `bc_mask` | 是否为约束节点，`1` 为约束节点，`0` 为非约束节点 | 必选 |

作用：

- 提供节点在三维空间中的基础位置。
- 标记该节点是否位于约束区域。
- 是所有后续几何增强特征的基础。

注意：

- 不建议只使用 `x, y, z, frequency` 作为输入，这样会丢失大量全局几何、PSD 和模态信息。

---

### B. 节点归一化几何特征

这部分用于告诉模型节点在整个结构中的相对位置，而不是只依赖绝对坐标。

| 特征 | 计算方式 | 作用 |
|---|---|---|
| `x_norm` | `x / plate_radius` | 归一化 x 位置 |
| `y_norm` | `y / plate_radius` | 归一化 y 位置 |
| `z_norm` | `z / plate_thickness` | 厚度方向归一化 |
| `r` | `sqrt(x^2 + y^2)` | 节点径向位置 |
| `r_norm` | `r / plate_radius` | 归一化径向位置 |
| `theta` | `atan2(y, x)` | 极角 |
| `sin_theta` | `sin(theta)` | 周向位置编码 |
| `cos_theta` | `cos(theta)` | 周向位置编码 |
| `dist_to_edge` | `(plate_radius - r) / plate_radius` | 到外边界距离 |

作用：

- 帮助模型识别节点的全局空间位置。
- 让不同尺寸 case 之间的节点位置具有可比性。
- 对盘状结构、耳片结构、边界区域尤其重要。

建议：

- 这组特征建议作为第一版点模型的必选特征。
- 对于角度变量，建议使用 `sin_theta` 和 `cos_theta`，不要直接只用 `theta`，以避免角度周期断点问题。

---

### C. 工程区域距离特征

这部分是点模型最重要的特征之一，目的是帮助模型识别高应力热点区域。

| 特征 | 计算方式 | 作用 |
|---|---|---|
| `dist_to_earpiece_center` | 节点到最近耳片中心距离 | 定位耳片区域 |
| `dist_to_ear_hole_edge_local` | `(dist_to_earpiece_center - earpiece_HoleRadius) / earpiece_HoleRadius` | 到耳孔边缘的局部距离 |
| `dist_to_ear_hole_edge_global` | `(dist_to_earpiece_center - earpiece_HoleRadius) / plate_radius` | 到耳孔边缘的全局归一化距离 |
| `near_ear_hole` | `exp(-max(dist_to_ear_hole_edge, 0) / earpiece_HoleRadius)` | 耳孔附近 soft mask |
| `earpiece_width_over_plate_radius` | `earpiece_width / plate_radius` | 耳片宽度相对板半径的比例 |
| `earpiece_width_over_hole_radius` | `earpiece_width / earpiece_HoleRadius` | 耳片宽度相对耳孔半径的比例 |
| `center_radius_local` | `r / mass_couple_radius` | 中心耦合区相对位置 |
| `center_couple_signed` | `(mass_couple_radius - r) / mass_couple_radius` | 是否在中心耦合区内外 |
| `near_center_couple` | `sigmoid(5 * center_couple_signed)` | 中心耦合区 soft mask |
| `near_boundary` | `dist_to_edge < threshold` | 是否靠近外边界 |

作用：

- 显式告诉模型哪些节点靠近耳孔、耳片根部、中心耦合区和外边界。
- 这些区域往往是应力集中和热点出现的位置。
- 对不用 GNN 的点模型尤其重要，因为模型无法通过消息传递自动识别局部几何区域。

建议：

- 第一版模型必须保留耳孔、中心耦合区、外边界相关特征。
- 距离类特征尽量使用归一化距离。
- `near_*` 特征建议使用 soft mask，而不是只使用硬阈值，避免边界附近特征突变。

---

### D. 局部邻域统计特征

这组特征用于补偿不用 GNN 后丢失的节点间拓扑信息。

来源：`edges.csv`

| 特征 | 计算方式 | 作用 |
|---|---|---|
| `node_degree` | 与该节点相连的边数量 | 局部拓扑密度 |
| `neighbor_dist_mean` | 相邻边长度均值 | 局部网格尺度 |
| `neighbor_dist_min` | 相邻边长度最小值 | 是否存在很近邻居 |
| `neighbor_dist_max` | 相邻边长度最大值 | 局部网格是否拉伸 |
| `neighbor_dist_std` | 相邻边长度标准差 | 局部网格规则性 |
| `neighbor_dx_mean` | 邻接边 `dx` 均值 | 局部方向偏置 |
| `neighbor_dy_mean` | 邻接边 `dy` 均值 | 局部方向偏置 |
| `neighbor_dz_mean` | 邻接边 `dz` 均值 | 厚度方向邻接关系 |
| `neighbor_coord_std_x` | 邻居节点 x 坐标标准差 | 局部几何跨度 |
| `neighbor_coord_std_y` | 邻居节点 y 坐标标准差 | 局部几何跨度 |
| `neighbor_coord_std_z` | 邻居节点 z 坐标标准差 | 局部厚度变化 |
| `distance_to_nearest_bc_node` | 到最近约束节点距离 | 约束影响强弱 |
| `k_hop_bc_ratio` | k-hop 邻域内约束节点比例 | 约束邻域影响 |

作用：

- 点模型本身不知道相邻节点是谁。
- 通过邻域统计量，可以让模型获得局部网格尺度、局部拓扑密度和约束邻域信息。
- 有助于改善热点区域形状破碎、局部峰值不稳定的问题。

建议：

- 第一版可以先不加，作为增强实验。
- 如果发现不用 GNN 后热点形状不连续、局部误差较大，应优先加入这组特征。

---

### E. Case 全局几何与质量特征

来源：`global.json`

| 特征 | 说明 | 是否必选 |
|---|---|---:|
| `earpiece_thickness` | 耳片厚度 | 必选 |
| `earpiece_RadialDist` | 耳片径向距离 | 必选 |
| `earpiece_TopWidth` | 耳片宽度 | 必选 |
| `earpiece_HoleTopDist` | 孔距相关参数 | 必选 |
| `earpiece_TopFilletRadius` | 上倒角半径 | 必选 |
| `earpiece_BottomFilletRadius` | 下倒角半径 | 必选 |
| `plate_radius` | 板半径 | 必选 |
| `Add_mass` | 附加质量 | 必选 |
| `plate_thickness` | 板厚 | 建议 |
| `earpiece_HoleRadius` | 耳孔半径 | 建议 |
| `mass_couple_radius` | 中心耦合半径 | 建议 |

作用：

- 告诉模型当前节点属于什么几何 case。
- 同一个节点相对位置，在不同几何参数和附加质量下，应力响应可能完全不同。
- 对跨 case 泛化非常关键。

建议：

- `params_list` 8 维必须保留。
- 如果 `fixed_geometry` 中有板厚、耳孔半径、中心耦合半径，也建议显式加入。
- 除原始参数外，建议额外构造比例特征，例如耳片宽度 / 板半径、耳孔半径 / 板半径等。

---

### F. PSD 激励特征

来源：`global.json -> psd_points`

| 特征 | 说明 | 是否必选 |
|---|---|---:|
| `psd_1_value` | 第 1 个谱点 PSD 值 | 必选 |
| `psd_1_imag` | 第 1 个谱点虚部，如果存在 | 可保留 |
| `psd_1_freq` | 第 1 个谱点频率 | 必选 |
| `psd_2_value` | 第 2 个谱点 PSD 值 | 必选 |
| `psd_2_freq` | 第 2 个谱点频率 | 必选 |
| `...` | 其他谱点 | 必选 |
| `psd_band_energy` | 对 PSD 曲线积分近似 | 建议 |
| `psd_value_at_frequency` | 当前频率插值得到的 PSD 值 | 强烈建议 |
| `log_psd_value_at_frequency` | `log1p(psd_value_at_frequency)` | 强烈建议 |

作用：

- PSD 描述输入激励能量。
- `MISES_psd_density` 是频率点上的应力 PSD 响应，因此当前频率附近的 PSD 输入能量很重要。

建议：

- 保留原始 `psd_points` 展平向量。
- 额外加入 `psd_value_at_frequency` 和 `log_psd_value_at_frequency`。
- 如果 PSD 曲线是分段线性谱，可以对当前频率进行插值。
- 如果后续要预测积分后的 RMises，则还应加入 PSD 频带能量特征。

---

### G. 当前频率与模态频率关系特征

这部分是频域应力预测的核心特征。

| 特征 | 计算方式 | 作用 |
|---|---|---|
| `frequency` | 当前频点 | 基础频率输入 |
| `log_frequency` | `log(frequency)` | 缓解频率尺度问题 |
| `freq_top1` | 第一阶模态频率 | 模态位置 |
| `freq_top2` | 第二阶模态频率 | 模态位置 |
| `freq_top3` | 第三阶模态频率 | 模态位置 |
| `signed_delta_to_mode_1` | `(f - mode_1) / mode_1` | 到一阶模态的有符号距离 |
| `signed_delta_to_mode_2` | `(f - mode_2) / mode_2` | 到二阶模态的有符号距离 |
| `signed_delta_to_mode_3` | `(f - mode_3) / mode_3` | 到三阶模态的有符号距离 |
| `abs_delta_to_mode_1` | `abs(signed_delta_to_mode_1)` | 到一阶模态的距离 |
| `abs_delta_to_mode_2` | `abs(signed_delta_to_mode_2)` | 到二阶模态的距离 |
| `abs_delta_to_mode_3` | `abs(signed_delta_to_mode_3)` | 到三阶模态的距离 |
| `nearest_delta` | `min(abs_delta_to_top3)` | 最近模态距离 |
| `nearest_mode_index` | 最近模态编号 | 当前频率靠近哪阶模态 |
| `first_mode_ratio` | `f / mode_1` | 当前频率相对一阶模态位置 |

作用：

- 同样的几何和 PSD，在不同频率下的应力响应可能完全不同。
- 模型需要知道当前频率是在模态附近、模态之间，还是远离主要模态。
- 这组特征直接帮助模型识别共振邻域和响应峰值区域。

建议：

- 第一版必须保留 `frequency`、`log_frequency`、`freq_top3` 和 `frequency_relations`。
- 如果有更多阶模态频率，建议扩展到前 5 阶或前 10 阶。
- 对随机振动频域响应而言，这组特征通常比单纯的 `frequency` 更重要。

---

### H. 节点模态振型特征

这组特征非常关键。没有 GNN 后，节点模态振型特征可以补充很多全局结构信息。

来源：`mode_shapes/*.csv`

| 特征 | 说明 | 是否建议 |
|---|---|---:|
| `weighted_abs_u1` | 当前频率加权后的 `|U1|` | 强烈建议 |
| `weighted_abs_u2` | 当前频率加权后的 `|U2|` | 强烈建议 |
| `weighted_abs_u3` | 当前频率加权后的 `|U3|` | 强烈建议 |
| `weighted_umag` | 当前频率加权后的位移幅值 | 强烈建议 |
| `nearest_abs_u1` | 最近模态的 `|U1|` | 强烈建议 |
| `nearest_abs_u2` | 最近模态的 `|U2|` | 强烈建议 |
| `nearest_abs_u3` | 最近模态的 `|U3|` | 强烈建议 |
| `nearest_umag` | 最近模态的 `U_mag` | 强烈建议 |
| `nearest_gap` | 当前频率与最近模态的 gap | 强烈建议 |
| `nearest_weight` | 最近模态权重 | 强烈建议 |
| `weighted_signed_u1` | 加权有符号 `U1` | 可选 |
| `weighted_signed_u2` | 加权有符号 `U2` | 可选 |
| `weighted_signed_u3` | 加权有符号 `U3` | 可选 |
| `mode_i_umag` | 每一阶模态的节点 `U_mag` | 可选，维度较高 |

作用：

- 告诉模型当前频率下该节点在模态空间中大概如何运动。
- 将“当前频点可能激发出什么空间形态”作为节点侧先验输入。
- 弥补点模型没有图消息传递、难以从全局几何中间接推断振型影响的问题。

建议：

- 如果点模型只能保留一类高级物理特征，优先保留节点模态振型特征。
- 第一版可以使用前 10 阶模态，采用 `resonance`、`log_gaussian` 或 `inverse_log_gap` 权重。
- 建议至少保留加权模态特征和最近模态特征。

---

### I. 可选全局模态摘要特征

这组特征不是每个节点不同，而是每个 case-frequency 共享。

| 特征 | 说明 |
|---|---|
| `modal_frequencies` | 选中模态频率向量 |
| `mode_weights` | 当前频率对应的模态权重 |
| `mode_log_gap` | 当前频率与每阶模态的 log gap |
| `per_mode_umag_mean` | 每阶模态全节点平均位移幅值 |
| `per_mode_umag_rms` | 每阶模态全节点 RMS 位移幅值 |
| `node_umag_mean_weighted` | 加权模态位移全图均值 |
| `node_umag_std_weighted` | 加权模态位移全图标准差 |
| `node_umag_max_weighted` | 加权模态位移全图最大值 |
| `axis_energy_u1` | U1 方向模态能量占比 |
| `axis_energy_u2` | U2 方向模态能量占比 |
| `axis_energy_u3` | U3 方向模态能量占比 |

作用：

- 给模型提供当前 case-frequency 的整体模态响应强弱。
- 有助于预测整体幅值和峰值水平。
- 对峰值应力预测不稳定的情况尤其有帮助。

建议：

- 第一版可以先不加入。
- 如果发现峰值幅值预测偏低或偏高，再加入该组特征。

---

## 4. 最终推荐特征表

### 4.1 第一版必选特征

建议第一版使用以下特征，作为基于点的强 baseline。

```text
节点基础：
- x
- y
- z
- bc_mask

节点几何：
- x_norm
- y_norm
- z_norm
- r_norm
- dist_to_edge
- sin_theta
- cos_theta

工程区域：
- dist_to_ear_hole_edge_local
- dist_to_ear_hole_edge_global
- near_ear_hole
- center_radius_local
- center_couple_signed
- near_center_couple

case 全局：
- params_list 8维
- plate_thickness
- earpiece_HoleRadius
- mass_couple_radius

PSD：
- psd_points 展平向量
- psd_value_at_frequency
- log_psd_value_at_frequency

频率-模态：
- frequency
- log_frequency
- freq_top3
- signed_delta_to_top3
- abs_delta_to_top3
- nearest_delta
- nearest_mode_index
- first_mode_ratio

节点模态：
- weighted_abs_u1
- weighted_abs_u2
- weighted_abs_u3
- weighted_umag
- nearest_abs_u1
- nearest_abs_u2
- nearest_abs_u3
- nearest_umag
- nearest_gap
- nearest_weight
```

---

### 4.2 第二版增强特征

如果第一版发现热点预测不稳定、局部误差较大，建议加入：

```text
局部邻域：
- node_degree
- neighbor_dist_mean
- neighbor_dist_min
- neighbor_dist_max
- neighbor_dist_std
- neighbor_coord_std_x
- neighbor_coord_std_y
- neighbor_coord_std_z
- distance_to_nearest_bc_node
- k_hop_bc_ratio

全局模态摘要：
- mode_weights
- mode_log_gap
- per_mode_umag_mean
- per_mode_umag_rms
- axis_energy_u1
- axis_energy_u2
- axis_energy_u3
- weighted_umag_global_mean
- weighted_umag_global_max
```

---

## 5. 不建议使用的特征

| 特征 | 原因 |
|---|---|
| `node_id` | 容易让模型记忆节点编号，不利于泛化 |
| `case_id` | 只能用于分组，不应作为模型输入 |
| 只使用原始未归一化的几何参数 | 不利于跨尺寸 case 泛化，应配合比例特征 |
| 只用 `x, y, z, frequency` | 信息严重不足，无法表达全局几何、PSD、模态响应 |
| 真实 hotspot 标签作为输入 | 会造成标签泄漏 |
| 由目标 `MISES_psd_density` 推出来的统计量 | 会造成标签泄漏 |
| 当前频率下真实应力场的邻域均值 | 会造成标签泄漏 |

---

## 6. 训练数据划分原则

必须按 `case_id` 划分训练集、验证集和测试集。

错误做法：

```text
把所有 node-frequency 样本随机打散后划分 train / val / test
```

这样会导致同一个 case 的不同节点、不同频率同时出现在训练集和测试集，造成严重数据泄漏。

正确做法：

```text
按 case_id 划分 train / val / test
```

例如：

```text
train: 70% cases
val:   15% cases
test:  15% cases
```

测试集中的 case 在训练过程中必须完全不可见。

---

## 7. 标准化方案

| 对象 | 标准化方式 |
|---|---|
| 连续输入特征 | `(x - mean) / std` |
| 目标值 | `log1p(MISES_psd_density)` 后再标准化 |
| `bc_mask` / `near_*` mask | 可不标准化，或随整体 scaler 处理 |
| `frequency` | 同时保留 `frequency` 和 `log_frequency` |
| PSD 值 | 建议加 `log_psd_value_at_frequency` |
| 距离类特征 | 尽量使用归一化距离 |

注意：

- 所有 scaler 只能在训练集上拟合。
- 验证集、测试集和预测集必须复用训练集 scaler。
- 目标值建议先 `log1p`，再标准化。

---

## 8. 推荐建模版本

### Version 1：强 baseline

```text
输入：
节点基础特征
+ 节点几何区域特征
+ case 全局参数
+ PSD
+ 当前频率
+ frequency_relations

模型：
LightGBM / XGBoost / CatBoost / MLP

目标：
log1p(MISES_psd_density)
```

目的：

- 快速判断不用 GNN 后，单点模型是否已经能学到主要规律。
- 作为后续 GNN 或 hybrid 模型的对照基线。

---

### Version 2：物理增强点模型

```text
Version 1
+ 节点模态振型特征
+ 当前频率加权模态特征
+ 最近模态特征
```

目的：

- 引入当前频率相关的空间响应先验。
- 弥补点模型无法通过图消息传递自动学习模态形状的问题。

这是最推荐重点实验的版本。

---

### Version 3：补偿 GNN 信息损失

```text
Version 2
+ 局部邻域统计特征
+ 到最近约束节点距离
+ k-hop 区域统计
```

目的：

- 弥补不用 GNN 后丢失的节点间拓扑信息。
- 改善热点区域形状破碎、空间场不连续的问题。

---

### Version 4：峰值辅助版本

增加一个 case-frequency 级别的辅助预测：

```text
case + frequency -> peak_MISES_psd_density
```

然后点模型预测：

```text
node_drop_from_peak
```

最终组合为：

```text
node_stress = peak_stress - softplus(drop_from_peak)
```

目的：

- 提升热点峰值幅值预测。
- 避免点模型逐点预测时整体峰值偏低或偏高。
- 与原来的 peak-relative 思路保持一致。

---

## 9. 推荐评估指标

不要只看全场平均误差。点模型尤其需要检查热点和空间一致性。

| 指标 | 说明 |
|---|---|
| `stress_mae` | 全场平均绝对误差 |
| `stress_log_mae` | log 空间平均绝对误差 |
| `stress_top1_mae` | top 1% 高应力节点误差 |
| `stress_top5_mae` | top 5% 高应力节点误差 |
| `stress_peak_relative_error` | 峰值相对误差 |
| `stress_hotspot_mae` | 热点节点 MAE |
| `stress_hotspot_within25_ratio` | 热点节点中相对误差 ≤ 25% 的比例 |
| `stress_hotspot_miss25_rate` | `1 - hotspot_within25_ratio` |
| `stress_non_hotspot_mae` | 非热点区域 MAE |
| `edge_smoothness_error` | 相邻节点预测差异与真实差异的一致性 |
| 区域误差 | 耳孔、耳片根部、中心耦合区、外圈等区域误差 |

建议新增空间一致性指标：

```text
edge_smoothness_error = mean over edges |(pred_i - pred_j) - (true_i - true_j)|
```

作用：

- 检查点模型是否把全场预测得过于破碎。
- 评估不用 GNN 后空间连续性损失是否严重。

---

## 10. 总体建议

基于点的 `MISES_psd_density` 预测可以作为 GNN 的轻量化替代方案和强 baseline。

但它不能简化成：

```text
节点坐标 + 当前频率 -> MISES_psd_density
```

更合理的输入结构应该是：

```text
节点空间位置
+ 工程区域距离
+ 局部邻域统计
+ case 几何/质量参数
+ PSD 激励
+ 当前频率
+ 频率-模态关系
+ 节点模态振型特征
→ 当前节点 MISES_psd_density
```

可以把特征理解成三层：

```text
第一层：节点在哪里
x, y, z, 归一化坐标，区域距离，near mask

第二层：这个 case 是什么
几何参数，Add_mass，PSD，固定几何参数

第三层：当前频率下它怎么响应
frequency_relations，模态频率，节点振型，加权模态特征
```

最终判断：

```text
点模型是合理的，但前提是必须把节点放回 case、频率、PSD 和模态响应的上下文中。
```

如果后续实验发现：

- 热点峰值误差偏大；
- 热点区域形状破碎；
- 新几何 case 泛化变差；
- 空间场不连续；

则应优先尝试：

```text
加入局部邻域统计特征
加入更多阶模态振型特征
加入全局峰值辅助模型
或回到轻量 hybrid GNN
```
