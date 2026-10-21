# 全零件三类点 Specialist 诊断实验方案

更新日期：2026-05-20  
适用任务：全零件点级 `MISES_psd_density` 预测  
当前主线：点模型，`case + frequency + node -> MISES_psd_density`

## 1. 实验目的

本实验不是为了直接训练最终主模型，而是为了诊断当前建模方式是否正确：

```text
当前点模型特征是否足够表达：
1. 平坦背景区；
2. 靠近应力集中标注区但真实平坦的点；
3. 真实热点峰值点。
```

要回答的问题是：

```text
如果只训练某一类物理场景，模型能不能学好？
```

解释方式：

```text
specialist 模型在自己点集上明显好：
    说明特征表达能力基本够，主模型问题更可能来自样本竞争、loss 权重或 selection。

specialist 模型在自己点集上也不好：
    说明当前输入特征对这类场景表达不足，需要补物理特征、频率连续性或空间约束。
```

## 2. 三个诊断实验

训练三个只看单类点的 specialist 模型：

| 实验 | 模型名建议 | 训练点集 | 诊断目标 |
|---|---|---|---|
| D1 | `node_mlp_v5_diag_full_flat_background_only` | 全零件中远离标注集中区的平坦背景点 | 判断模型能否在普通背景区保持平坦，不造假峰 |
| D2 | `node_mlp_v5_diag_full_sensitive_flat_only` | 全零件中位于标注集中区但真实平坦的点 | 判断区域先验是否过强，是否会乱报热点 |
| D3 | `node_mlp_v5_diag_full_hotspot_only` | 全零件真实热点 top1/top5 和峰值点 | 判断当前特征是否足够预测峰值和热点形态 |

对照模型：

```text
全零件主模型 best：优先使用后续全零件 V5/V6 主模型
当前耳片 V4/V5 best：只能作为耳片区域预训练/参考，不能代表全零件泛化
D1/D2/D3 specialist
```

注意：

```text
本诊断实验使用全零件节点，不再只使用耳片区域节点。
如果当前训练代码默认筛选耳片节点，需要先增加 all_nodes / full_part 点选择开关。
```

## 3. 全零件区域标注来源

`nodes.csv` 中已经有应力集中相关区域标注，应优先使用这些标注定义 D1/D2/D3 的区域语义。

当前样例列名：

```text
node_index
node_label
x
y
z
bc_mask
center_node_mask
center_couple_mask
plate_hole_wall_mask
ear_hole_wall_mask
ear_connection_fillet_mask
ear_connection_earside_mask
ear_connection_mask
```

建议定义全零件敏感区 mask：

```text
stress_region_mask =
    center_couple_mask
 or plate_hole_wall_mask
 or ear_hole_wall_mask
 or ear_connection_fillet_mask
 or ear_connection_earside_mask
 or ear_connection_mask
```

各标注区含义：

| mask | 建议含义 | 是否纳入敏感区 |
|---|---|---|
| `center_couple_mask` | 中心耦合/加载连接区域 | 是 |
| `plate_hole_wall_mask` | 圆盘中心孔壁区域 | 是 |
| `ear_hole_wall_mask` | 耳片通孔孔壁区域 | 是 |
| `ear_connection_fillet_mask` | 耳片到圆盘连接圆角区域 | 是，重点 |
| `ear_connection_earside_mask` | 耳片连接区靠耳片侧 | 是，重点 |
| `ear_connection_mask` | 耳片连接整体区域 | 是，重点 |
| `center_node_mask` | 中心节点/参考节点 | 单独统计，不默认作为热点训练点 |
| `bc_mask` | 约束边界 | 默认不作为 D1/D2/D3 训练点，除非专门做边界诊断 |

重点应力集中区域建议分组：

```text
center_couple_region      = center_couple_mask
plate_hole_region         = plate_hole_wall_mask
ear_hole_region           = ear_hole_wall_mask
ear_connection_region     = ear_connection_fillet_mask
                         or ear_connection_earside_mask
                         or ear_connection_mask
```

后续统计和评估需要按这些区域分别输出，不能只给一个 overall。

## 4. 关键前提：先做数据量统计

这三个实验可能出现训练数据不足，尤其是 D2 和 D3。

因此训练前必须先构造点集统计表：

```text
diagnostic_pointset_summary.csv
```

按 split 输出：

```text
split
pointset_type
region_type
case_count
case_node_count
case_frequency_point_count
frequency_count_mean
frequency_count_min
mean_log_p50
std_log_p50
peak_ratio_p50
max_raw_p90
```

其中：

```text
case_node_count = 选中的唯一 case-node 数
case_frequency_point_count = case-node 展开到所有频率后的训练点数
```

注意：模型训练样本最终仍是 `(case, frequency, node)`，所以一个 case-node 如果有 100 个频点，会展开成约 100 个训练点。

必须按区域额外统计：

```text
region_type = full_part
region_type = center_couple_region
region_type = plate_hole_region
region_type = ear_hole_region
region_type = ear_connection_region
region_type = background_non_stress_region
```

原因：

```text
全零件点很多，但 D2/D3 在某个具体应力集中区域可能很少。
如果只看全局数量，可能掩盖某个关键区域样本不足。
```

## 5. 数据量门槛

建议使用以下最低门槛判断是否值得单独训练：

| 点集 | train case-node 最低数量 | train 展开点最低数量 | val/test case-node 最低数量 | 说明 |
|---|---:|---:|---:|---|
| D1 full flat background | 30,000 | 1,500,000 | 3,000 | 全零件背景点应充足，需要覆盖不同空间区域 |
| D2 full sensitive flat | 5,000 | 250,000 | 500 | 需要每个主要应力集中区域都有样本 |
| D3 full hotspot | 5,000 | 250,000 | 500 | 如果只选 peak node 会不足，必须包含 top1/top5 |

更稳的推荐数量：

```text
D1: train case-node >= 100,000
D2: train case-node >= 20,000
D3: train case-node >= 20,000
```

如果低于最低门槛，不建议训练 specialist 模型，因为结论容易变成采样噪声。

区域级最低要求：

```text
D2 每个主要应力集中区域 train case-node >= 1,000
D3 每个主要应力集中区域 train case-node >= 1,000
```

如果某个区域低于该门槛，该区域不单独训练 specialist，只保留为评估子桶。

## 6. 点集构造总流程

对每个 split 独立构造点集：

```text
train cases -> 只能构造 train pointset
val cases   -> 只能构造 val pointset
test cases  -> 只能构造 test pointset
```

不能用全量 case 统计阈值后再切 split，避免 test 信息泄漏。

对每个 `case + node`，沿频率计算真实响应谱：

```text
raw_y(f) = MISES_psd_density(node, f)
log_y(f) = log1p(raw_y(f))
```

节点级谱统计：

```text
mean_log = mean(log_y)
std_log = std(log_y)
range_log = max(log_y) - min(log_y)
peak_ratio = max(raw_y) / (mean(raw_y) + eps)
max_raw = max(raw_y)
peak_frequency = frequency at max(raw_y)
```

全零件区域标注直接来自 `nodes.csv`：

```text
center_couple_mask
plate_hole_wall_mask
ear_hole_wall_mask
ear_connection_fillet_mask
ear_connection_earside_mask
ear_connection_mask
```

同时保留 V5/V6 几何和模态特征：

```text
全局坐标/归一化坐标
bc_mask
各区域 mask
到各标注区域的距离特征
频率/模态/FRF 特征
模态梯度特征
```

如果当前特征只覆盖耳片局部距离，建议在全零件版本中补齐：

```text
dist_to_center_couple
dist_to_plate_hole_wall
dist_to_ear_hole_wall
dist_to_ear_connection
nearest_stress_region_type
nearest_stress_region_distance
```

## 7. D1：全零件平坦背景区点集

### 7.1 目标

验证模型能否在普通背景区保持平坦：

```text
真实平坦 -> 预测也平坦
```

不希望看到：

```text
真实没有峰，但预测在模态频率附近凭空起峰
```

### 7.2 候选条件

背景区候选：

```text
bc_mask = 0
center_node_mask = 0
stress_region_mask = 0
center_couple_mask = 0
plate_hole_wall_mask = 0
ear_hole_wall_mask = 0
ear_connection_fillet_mask = 0
ear_connection_earside_mask = 0
ear_connection_mask = 0
dist_to_nearest_stress_region >= P60
```

平坦条件：

```text
std_log <= 背景候选 P20
range_log <= 背景候选 P20
peak_ratio <= 背景候选 P20
```

为避免只选低响应点，按 `mean_log` 分三档：

```text
low_flat:  P10 <= mean_log < P30
mid_flat:  P40 <= mean_log < P60
high_flat: P70 <= mean_log < P85
```

每档按 case 均衡抽样。

### 7.3 数据不足降级

如果 D1 数据不足：

1. 放宽 `std_log/range_log/peak_ratio` 到 P30。
2. 放宽 `dist_to_nearest_stress_region >= P50`。
3. 合并 `low/mid/high` 分层，但保留每个 case 的最大采样数限制。

如果仍不足，说明背景平坦点定义过严，不建议训练 D1。

## 8. D2：标注敏感区但真实平坦点集

### 8.1 目标

验证模型不会因为物理先验过强而乱报热点：

```text
位于应力集中标注区，但真实平坦 -> 预测也应平坦
```

D2 是最重要的假峰诊断实验。

### 8.2 候选条件

敏感区候选满足至少一个：

```text
center_couple_mask = 1
plate_hole_wall_mask = 1
ear_hole_wall_mask = 1
ear_connection_fillet_mask = 1
ear_connection_earside_mask = 1
ear_connection_mask = 1
weighted_grad_umag_mean_frf >= P80
weighted_grad_vector_mean_frf >= P80
```

建议拆成四个子类：

```text
center_couple_flat
plate_hole_flat
ear_hole_flat
ear_connection_flat
```

真实平坦条件：

```text
std_log <= 对应敏感候选 P20
range_log <= 对应敏感候选 P20
peak_ratio <= 对应敏感候选 P20
```

### 8.3 数据不足降级

D2 可能明显不足。降级顺序：

1. 子类单独不足时，合并为一个 `sensitive_flat` 点集。
2. 对没有足够 mask 点的区域，增加 `dist_to_region <= P20` 的近邻节点。
3. 将平坦阈值从 P20 放宽到 P30。
4. 优先保留 `ear_hole_flat + ear_connection_flat + plate_hole_flat`，`center_couple_flat` 可作为单独评估桶。

如果 train case-node 仍低于 3,000，不建议训练 D2 specialist，只保留为评估点集。

## 9. D3：全零件真实热点点集

### 9.1 目标

验证模型是否具备预测真实热点峰值和峰形的能力：

```text
真实有峰 -> 预测应在相近频率和相近区域起峰
```

### 9.2 候选条件

热点候选：

```text
max_raw >= 全零件节点 P90
peak_ratio >= 全零件节点 P80
std_log >= 全零件节点 P80
```

同时包含每个 case-frequency 内：

```text
true peak node
true top1% nodes
true top5% nodes
```

并按热点所在区域打标签：

```text
hotspot_region_type =
    center_couple_region
  / plate_hole_region
  / ear_hole_region
  / ear_connection_region
  / other_region
```

注意：

```text
选中热点 node 后，训练时使用该 node 的完整频率谱线，
不是只使用峰值频点。
```

这样可以判断模型是否学到峰形，而不是只学到某个高值点。

### 9.3 数据不足降级

D3 如果只选 peak node，一定不足。降级策略：

1. 必须包含 top5% 节点，而不是只选 peak/top1。
2. 将 `max_raw` 从 P90 放宽到 P85。
3. 将 `peak_ratio/std_log` 从 P80 放宽到 P70。
4. 对每个 case 限制最大热点节点数，防止少数 case 占满训练集。

如果 D3 仍低于最低门槛，说明热点定义过严；优先扩大 top5 或按 `target_peak_bucket` 高档 case 增加采样。

## 10. 训练设计

三个 specialist 模型统一使用全零件版本特征。

必须确认训练数据选择为：

```yaml
dataset:
  node_scope: all_nodes
```

不要使用：

```yaml
dataset:
  node_scope: earpiece_only
```

特征建议：

```text
nodes.csv 区域 mask
到各标注应力集中区域的距离
include_modal_frf_shape_features: true
include_modal_gradient_features: true
include_modal_baseline_feature: true
```

初始化建议：

如果已经有全零件主模型 best，优先从全零件主模型初始化：

```yaml
training:
  init_checkpoint: node/outputs/<full_part_main_model>/best.pt
  allow_partial_init_checkpoint: true
  lr: 0.0001
  epochs: 20
```

如果暂时没有全零件主模型，可以从耳片 V5/V4 做部分 warm-start，但它只能复用共享输入层和 MLP 表达，不能作为全零件效果基准：

```yaml
training:
  init_checkpoint: node/outputs/node_mlp_earpiece_case1000_v5_modal_gradient_baseline_80g/best.pt
  allow_partial_init_checkpoint: true
  lr: 0.0001
  epochs: 20
```

如果 V5 尚未训练完成，可以临时从 V4 best 部分 warm-start：

```yaml
training:
  init_checkpoint: node/outputs/node_mlp_earpiece_case1000_v4_local_frf_mode_norm_80g/best.pt
  allow_partial_init_checkpoint: true
  lr: 0.0002
  epochs: 20
```

不建议把耳片 V4/V5 的 full_part_test 结果作为正式结论，因为它们没有见过全零件点分布。

### 10.1 三个配置

建议新增：

```text
node/configs/node_mlp_v5_diag_full_flat_background_only.yaml
node/configs/node_mlp_v5_diag_full_sensitive_flat_only.yaml
node/configs/node_mlp_v5_diag_full_hotspot_only.yaml
```

输出目录：

```text
node/outputs/node_mlp_v5_diag_full_flat_background_only
node/outputs/node_mlp_v5_diag_full_sensitive_flat_only
node/outputs/node_mlp_v5_diag_full_hotspot_only
```

## 11. 评估矩阵

每个模型都评估四套 test 点集：

```text
full_flat_background_test
full_sensitive_flat_test
full_hotspot_test
full_part_test
```

形成矩阵：

| 模型 | full_flat_background_test | full_sensitive_flat_test | full_hotspot_test | full_part_test |
|---|---|---|---|---|
| 全零件主模型 best | eval | eval | eval | eval |
| D1 full-flat specialist | eval | eval | eval | eval |
| D2 full-sensitive-flat specialist | eval | eval | eval | eval |
| D3 full-hotspot specialist | eval | eval | eval | eval |

重点不是 specialist 的 full_part_test，而是：

```text
specialist 在自己点集上能否明显超过全零件主模型。
```

每个 test 点集都要额外按区域拆分：

```text
center_couple_region
plate_hole_region
ear_hole_region
ear_connection_region
other_region
```

## 12. 诊断指标

### 12.1 平坦类指标

用于 D1/D2：

```text
log_mae
mae
bias_log = mean(pred_log - true_log)
true_std_log
pred_std_log
std_ratio = pred_std_log / (true_std_log + eps)
shape_error = mean |diff(pred_log) - diff(true_log)|
true_peak_ratio = max(true_raw) / mean(true_raw)
pred_peak_ratio = max(pred_raw) / mean(pred_raw)
false_peak_over_true = pred_peak_ratio / (true_peak_ratio + eps)
```

判断假峰：

```text
false_peak_over_true > 1.5
或 pred_std_log / true_std_log > 1.5
```

### 12.2 热点类指标

用于 D3：

```text
peak_relative_error
peak_amp_error
peak_freq_error
top1_mae
top1_log_mae
top5_mae
top5_log_mae
pred_peak_node_in_true_top1
pred_peak_node_in_true_top5
shape_error
```

区域拆分指标：

```text
region_type
region_case_node_count
region_log_mae
region_false_peak_over_true
region_peak_relative_error
region_top1_mae
region_top5_mae
```

## 13. 可视化输出

每类点集抽样 30 个 case-node 画频谱：

```text
true_log(f)
V4_pred_log(f)
V5_pred_log(f)
full_main_pred_log(f)
D*_pred_log(f)
modal frequencies 竖线
```

输出目录建议：

```text
node/outputs/diagnostics/three_point_specialists/
  pointsets/
  metrics/
  plots/
```

关键文件：

```text
selected_points_train.csv
selected_points_val.csv
selected_points_test.csv
diagnostic_pointset_summary.csv
diagnostic_eval_matrix.csv
diagnostic_eval_by_region.csv
```

## 14. 结果解释

### 14.1 D1 解释

```text
D1 在 full_flat_background 上明显优于全零件主模型：
    平坦背景特征是可表达的，主模型若造峰，多半是训练目标被热点/FRF 牵引。

D1 仍然在 full_flat_background 上造峰：
    当前频率/模态特征可能过强，或者缺少频率连续性约束。
```

### 14.2 D2 解释

```text
D2 在 full_sensitive_flat 上明显优于全零件主模型：
    标注应力集中区域的假峰可以通过训练约束解决，后续主模型应加 sensitive-flat 权重或 flat penalty。

D2 仍然乱报热点：
    当前特征无法区分“标注敏感但平坦”和“标注敏感且热点”，需要补 modal stress/strain、区域距离特征或更强空间/频率约束。
```

### 14.3 D3 解释

```text
D3 在 full_hotspot 上明显优于全零件主模型：
    当前特征足够表达热点，主模型 peak 问题主要是 loss/样本竞争。

D3 也预测不出峰：
    当前输入缺关键物理信息，优先考虑 modal stress/strain、全零件区域距离、频率连续性或局部图约束。
```

## 15. 最终决策

实验结束后给出三条结论：

```text
1. 平坦背景区是否能被当前特征正确建模？
2. 标注应力集中区域的平坦点假峰是否来自区域先验过强？
3. 热点峰值预测差是训练权重问题，还是特征表达能力问题？
```

后续动作：

```text
如果 D1/D2 specialist 明显好：
    主模型加 flat/sensitive-flat 权重或 false-peak penalty。

如果 D3 specialist 明显好：
    主模型加 peak/top1 权重或 peak-focused selection。

如果 specialist 也不好：
    不优先调 loss，应补 modal stress/strain、frequency shape loss 或空间约束。
```
