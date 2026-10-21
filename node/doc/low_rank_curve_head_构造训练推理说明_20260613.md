# low_rank_curve_head 构造、训练与推理说明

本文档说明 `low_rank_curve_head: true` 这一路圆盘中心模型的真实数据流、训练 loss 和前向计算。对应配置入口是 `node/configs/node_mlp_v6_disk_center_low_rank_curves.yaml`，快速评估实验入口是 `node/configs/node_mlp_v6_disk_center_low_rank_curves_fast_eval.yaml`。

## 1. 结论先行

`low_rank_curve_head` 不是一个离线预先计算好的输入特征，也不是从训练集目标里提前 SVD/PCA 得到的固定三条曲线。

它是 `PointMLP` 里的一个模型结构开关：

- 输入仍然是每个点、每个频率对应的一行特征。
- 模型把输入特征按名称模式拆成两组：`frequency_features` 和 `node_features`。
- `frequency_head` 在线输出 `center_curve_rank` 维 latent curve value，当前 rank 为 3。
- `node_mixing_head` 为每个节点输出 3 个 mixing 权重。
- `node_scale_head` 为每个节点输出整体 offset。
- 最终预测由低秩曲线混合结果加一个小权重 pointwise residual 得到。

核心公式为：

```text
G(f, case) = frequency_head(frequency_features)       # shape: [rank]
w(i) = softmax(node_mixing_head(node_features_i))     # shape: [rank]
s(i) = node_scale_head(node_features_i)               # shape: [1]
p(i, f) = base_pointwise_mlp(all_features_i,f)        # shape: [1]

low_rank_i(f) = s(i) + sum_k w_k(i) * G_k(f, case)
prediction_i(f) = low_rank_i(f) + center_residual_weight * p(i, f)
```

当前配置：

```yaml
model:
  low_rank_curve_head: true
  center_curve_rank: 3
  center_residual_weight: 0.10
```

所以每个输入点都会产生 3 个 latent curve value，即 `[G1, G2, G3]`。

## 2. 数据样本如何构造

当前圆盘中心模型使用 per-frequency 点样本。一个训练样本对应：

```text
一个 case + 一个频率文件 + 该频率下被选中的 disk_center 节点集合
```

数据入口在 `node/case7_node_mlp/data.py` 的 `load_raw_point_sample`：

1. 从 per-frequency target CSV 读取当前频率下所有节点的 `MISES_psd_density`。
2. 根据 `dataset.node_scope: disk_center` 和相关过滤条件选择圆盘中心节点。
3. 对选中节点构造：
   - `target_raw`: 原始 `MISES_psd_density`
   - `target_log`: `log1p(target_raw)`
   - `geometry_features`: 几何、位置、区域等不走 scaler 的特征
   - `scaled_features`: 需要标准化的连续特征
   - `mask_features`: 区域 mask / one-hot 类特征
   - `region_masks`: `fullpart_region`、`earpiece_region`、`disk_region`、`disk_center_region`

进入训练前，`prepare_point_sample` 会把原始样本变成模型输入：

```python
scaled_features = x_scaler.transform(raw.scaled_features)
features = torch.cat([raw.geometry_features, scaled_features, raw.mask_features], dim=-1)
target_raw = _apply_target_floor(raw.target_raw, target_floor)
target_log = torch.log1p(target_raw).unsqueeze(-1)
target_scaled = y_scaler.transform(target_log)
point_weights = _build_point_weights(target_raw, feature_schema, loss_cfg, region_masks)
```

也就是说模型真实训练目标不是 raw PSD，而是标准化后的 `log1p(PSD)`：

```text
y_train = y_scaler(log1p(MISES_psd_density))
```

## 3. 训练集、验证集和测试集如何构造

基础配置在 `node/configs/node_mlp_v6_disk_center_baseline.yaml`：

```yaml
dataset:
  root: .cache/mises_psd_next1000_full_export_step0p5_balanced_v1
  sample_mode: per_frequency
  include_zero_frequency: false
  min_frequency_hz: 20.0
  max_frequency_hz: 2000.0
  split_mode: ratio
  split_seed: 42
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  node_scope: disk_center
```

`Trainer._prepare` 会调用 `resolve_case_splits`：

1. 扫描数据根目录下可用 case。
2. 用 `split_seed: 42` 打乱 case 名称。
3. 按 `0.8 / 0.1 / 0.1` 划分 train / val / test。
4. 再对每个 split 展开 per-frequency 样本路径。
5. 写入 `resolved_config.yaml` 时把 split 固化成 explicit case 列表，后续复现实验不会重新随机切分。

所以切分单位是 case，不是单个频率文件，也不是单个节点。这样可以避免同一个 case 的不同频率泄漏到训练和测试两侧。

## 4. 输入特征构造

`low_rank_curve_head` 复用的是 `hotspot_features` 的输入特征配置。配置继承链为：

```text
node_mlp_v6_disk_center_baseline.yaml
  -> node_mlp_v6_disk_center_hotspot_features.yaml
    -> node_mlp_v6_disk_center_low_rank_curves.yaml
      -> node_mlp_v6_disk_center_low_rank_curves_fast_eval.yaml
```

baseline 已包含：

- PSD 当前频率值和 log PSD 当前频率值
- 圆盘中心几何与板孔特征
- 角度周期特征
- 节点区域 mask
- stress region distance
- mode shape 特征
- modal ratio 特征
- modal FRF 特征
- modal gradient 特征
- modal baseline 特征

`hotspot_features` 在 baseline 上增加圆盘中心 hotspot 相关特征：

```yaml
features:
  include_disk_center_band_features: true
  disk_center_inner_radius_mm: 3.0
  disk_center_outer_radius_mm: 12.0
  disk_center_ring_softness_mm: 1.0
  include_disk_center_rbf_features: true
  disk_center_rbf_count: 8
  disk_center_rbf_bandwidth_ratio: 0.14
  include_disk_center_modal_region_features: true
  disk_center_modal_region_feature_keys:
    - nearest_umag
    - weighted_umag_frf
    - weighted_grad_umag_max_frf
```

这些新增特征主要表达两类信息：

- 节点相对圆盘中心的空间位置：内核、环带、RBF 径向位置编码。
- 模态/FRF 在圆盘中心内外区域的聚合差异：inner、outer、outer-minus-inner 等区域统计。

在当前实验里，baseline feature schema 是 238 维，打开 hotspot 特征后是 267 维。`low_rank_curve_head` 本身不再增加输入维度，而是在这 267 维输入上增加结构化 head。

## 5. low-rank head 如何选择 frequency / node 子特征

构造代码在 `node/case7_node_mlp/trainer.py` 的 `_low_rank_feature_indices`。

如果配置里没有显式给 `curve_frequency_feature_indices` 和 `curve_node_feature_indices`，代码会按 feature name pattern 自动选两组索引。

`frequency_features` 偏向描述当前 case、当前频率、PSD、模态频率关系和 FRF 响应，例如：

- 全局几何：`earpiece_thickness`、`plate_radius`、`plate_thickness`、`Add_mass`
- 当前频率：`frequency`、`log_frequency`
- 激励 PSD：`psd_`、`psd_value_at_frequency`、`log_psd_value_at_frequency`
- 模态 detuning：`signed_delta_to_mode`、`abs_delta_to_mode`、`nearest_delta`
- 模态比例：`first_mode_ratio`、`freq_ratio_mode_`、`modal_detuning_mode_`
- 模态/FRF gain：`log_modal_amp_`、`modal_weight_`、`log_modal_gain_frf`、`modal_weight_frf`

`node_features` 偏向描述节点位置、中心区域、局部形状、模态形状和 mask，例如：

- 归一化坐标与距离：`_norm`、`dist_to_`、`sin_theta`、`cos_theta`
- 中心区域：`center_`
- 邻近/加权振型：`weighted_umag`、`nearest_umag`、`weighted_abs_`、`nearest_abs_`
- 梯度特征：`weighted_grad_`、`nearest_grad_`
- active mode shape：`active1_`、`active2_`、`active3_`
- 区域和形状统计：`modal_baseline_log_`、`shape_rms`、`mask`、`stress_region_mask`

直观上：

- `frequency_head` 学“这个 case 在这个频率附近的整体响应形态”。
- `node_mixing_head` 学“这个节点更像哪一种响应形态”。
- `node_scale_head` 学“这个节点整体偏高还是偏低”。

## 6. 前向计算过程

实现代码在 `node/case7_node_mlp/models.py` 的 `PointMLP.forward`。

不开 `low_rank_curve_head` 时，模型就是普通 MLP：

```python
pointwise = self.network(features)
return pointwise
```

打开后，前向会返回一个 dict：

```python
pointwise = self.network(features)
frequency_features = features[:, self.frequency_feature_indices]
node_features = features[:, self.node_feature_indices]

curve_values = self.frequency_head(frequency_features)
mixing_logits = self.node_mixing_head(node_features)
mixing_weights = torch.softmax(mixing_logits, dim=-1)
node_scale = self.node_scale_head(node_features)

low_rank = node_scale + (mixing_weights * curve_values).sum(dim=-1, keepdim=True)
regression = low_rank + self.residual_weight * pointwise
```

返回内容：

```python
{
    "regression": regression,
    "pointwise": pointwise,
    "low_rank": low_rank,
    "node_scale": node_scale,
    "curve_values": curve_values,
    "mixing_weights": mixing_weights,
}
```

训练和评估统一通过 `regression_output(output)` 取最终预测：

```python
def regression_output(output):
    if isinstance(output, dict):
        return output["regression"]
    return output
```

因此 loss 和指标都只监督最终的 `regression`，不是单独监督 `curve_values`、`mixing_weights` 或 `node_scale`。

### 6.1 对同一个 case 扫频时的解释

对同一个 case，如果固定节点集合，把频率从 20 Hz 扫到 2000 Hz，`frequency_head` 会在每个频率点输出：

```text
f=20.0 Hz   -> [G1, G2, G3]
f=20.5 Hz   -> [G1, G2, G3]
f=21.0 Hz   -> [G1, G2, G3]
...
```

把第 1 维沿频率连起来就是 `G1(f)`，第 2 维是 `G2(f)`，第 3 维是 `G3(f)`。这三条曲线是模型训练出来的 latent basis curve，不带显式物理命名。

每个节点的预测曲线是三条 latent curve 的加权混合：

```text
low_rank_i(f)
  = node_scale_i
    + w_i1 * G1(f)
    + w_i2 * G2(f)
    + w_i3 * G3(f)
```

由于 `mixing_weights = softmax(mixing_logits)`：

```text
w_i1 + w_i2 + w_i3 = 1
w_ik >= 0
```

最终再加 residual：

```text
prediction_i(f) = low_rank_i(f) + 0.10 * pointwise_i(f)
```

这个 residual 的作用是保留少量逐点自由度，避免 3 条低秩曲线无法表达局部异常时模型完全受限。

## 7. 这个 head 跟模型其他部分如何交互

`low_rank_curve_head` 在当前模型里不是替代主干 MLP，而是和主干 MLP 并联后融合。整条链路可以理解成：

```text
同一行 features
  ├─ base PointMLP network  -> pointwise residual
  ├─ frequency feature slice -> frequency_head -> curve_values
  └─ node feature slice      -> node_mixing_head / node_scale_head

curve_values + mixing_weights + node_scale -> low_rank
low_rank + 0.10 * pointwise -> regression
regression -> loss / eval / metrics / checkpoint selection
```

### 7.1 与输入 feature schema 的交互

所有分支都吃同一个 `features` 张量，只是使用方式不同：

- 主干 `self.network(features)` 使用完整输入维度。
- `frequency_head` 使用 `features[:, self.frequency_feature_indices]`。
- `node_mixing_head` 和 `node_scale_head` 使用 `features[:, self.node_feature_indices]`。

这些 index 不是硬编码在模型里的，而是在 `build_model` 时根据 `feature_schema["feature_names"]` 匹配出来。因此它和 feature schema 强绑定：

- 如果新增、删除、重排特征，需要用 checkpoint 里的 feature schema 或当前 scaler cache 重新解析。
- 如果某个 feature name 被 frequency/node pattern 匹配到，它就会进入对应 head。
- `low_rank_curve_head` 本身不改变 scaler，也不单独维护一套输入标准化。

需要注意：`frequency_features` 是工程上的命名，表示这组特征主要描述频率、PSD、模态 detuning、FRF gain 和全局几何；它不是数学上强制“只随频率变化”。如果 pattern 匹配到带节点条件的 modal/FRF 特征，那么 `curve_values` 也会带有节点条件。当前使用方式的目标仍然是让它主要承担频率响应基函数的角色。

### 7.2 与主干 MLP 的交互

主干 MLP 始终存在：

```python
pointwise = self.network(features)
```

打开 low-rank 后，主干输出不再直接作为最终预测，而是作为 residual：

```python
regression = low_rank + self.residual_weight * pointwise
```

当前 `center_residual_weight = 0.10`，含义是：

- `low_rank` 是主预测通道。
- `pointwise` 是小权重修正通道。
- loss 的梯度会同时回传到 low-rank 三个 head 和主干 MLP。
- 主干 MLP 对最终输出的直接影响被 0.10 缩放，因此更偏向修局部误差，而不是重新完全自由地拟合每个点。

这也是它和普通 MLP 的核心差异：普通 MLP 每个点独立回归，low-rank 版本把主要预测约束成“频率 latent curve + 节点混合”的形式，再用主干 residual 做补偿。

### 7.3 与 loss 和辅助 loss 的交互

训练代码不知道也不关心内部是普通 tensor 还是 dict。它统一调用：

```python
output = model(features)
prediction = regression_output(output)
```

`regression_output` 只取 dict 里的 `"regression"`。所以主 loss、sample peak loss、top1/top5 loss、低响应过预测惩罚、背景假峰惩罚等辅助项，看到的都是最终融合后的预测：

```text
prediction = low_rank + 0.10 * pointwise
```

这带来两个结果：

- 所有 loss 都会通过 `regression` 反向传播到 `frequency_head`、`node_mixing_head`、`node_scale_head` 和主干 MLP。
- 默认没有任何 loss 直接要求 `G1/G2/G3` 长成某种固定曲线，也没有直接要求某个节点的 mixing weight 等于某个标签。

因此 head 内部的分解方式是 latent decomposition，只要最终 `regression` 能降低 loss，模型可以自行决定三条 curve 和节点混合权重的语义。

### 7.4 与 optimizer 和 checkpoint 的交互

`frequency_head`、`node_mixing_head`、`node_scale_head` 都是 `PointMLP` 的子模块，所以它们的参数会自动进入同一个 optimizer。当前没有给 low-rank head 设置单独学习率或单独 weight decay。

当前 low-rank 实验从 hotspot 模型初始化：

```yaml
training:
  init_checkpoint: node/outputs/node_mlp_v6_disk_center_hotspot_features/best.pt
  allow_partial_init_checkpoint: true
```

交互方式是：

- 旧 checkpoint 里已有的主干 MLP 参数尽量加载。
- 新增的 `frequency_head`、`node_mixing_head`、`node_scale_head` 没有旧权重，对应参数随机初始化。
- 后续训练中，主干 residual 和新增 head 一起更新。

保存 checkpoint 时会保存完整 `model_state`，包括主干和 low-rank head。后续推理必须用同样的 model 配置和 feature schema 来恢复这些参数。

### 7.5 与评估、推理和诊断的交互

评估和推理阶段同样只使用最终 `regression`：

```python
predictions_scaled.append(regression_output(model(features)).detach())
```

所以对于外部评估逻辑来说，low-rank head 是透明的：

- overall / top1 / top5 / top10 / within25 等指标都基于最终 `regression`。
- `curve_values` 和 `mixing_weights` 不参与指标计算。
- 如果需要解释模型，可以额外读取 forward 返回 dict 中的 `curve_values`、`mixing_weights`、`node_scale`、`pointwise`，但这属于诊断，不是标准推理输出。

也就是说，这个 head 对外暴露的行为仍然是“输入一批点特征，输出一批 PSD 预测”；它改变的是模型内部如何组织预测自由度。

## 8. Loss 如何计算

主 loss 在 `node/case7_node_mlp/trainer.py` 的 `train_one_epoch`：

```python
output = model(features)
prediction = regression_output(output)
loss_values = F.smooth_l1_loss(prediction, target, reduction="none").squeeze(-1)
loss = (loss_values * weights).sum() / weights.sum().clamp_min(1e-12)
```

其中：

- `prediction`: 标准化 log 空间的最终预测，即 `regression`
- `target`: `target_scaled = y_scaler(log1p(target_raw))`
- `weights`: 每个点的训练权重

主 loss 可以写成：

```text
L_main =
  sum_i weight_i * SmoothL1(pred_scaled_i, target_scaled_i)
  / sum_i weight_i
```

### 8.1 点权重

当前 baseline / low-rank 配置使用：

```yaml
loss:
  weighting: target_quantile
  top5_weight: 4.0
  top1_weight: 12.0
  target_bucket_weights:
    - {min: 0.0, max: 1.0, weight: 1.35}
    - {min: 1.0, max: 10.0, weight: 1.20}
    - {min: 10.0, max: 100.0, weight: 1.10}
```

`_build_point_weights` 会根据训练集 target 统计量设置：

- `target_raw >= target_positive_p95` 的点权重至少为 `top5_weight = 4.0`
- `target_raw >= target_positive_p99` 的点权重至少为 `top1_weight = 12.0`
- 低值 bucket 额外给 1.35 / 1.20 / 1.10 的权重，用于稳定背景和小响应区域

这里的 top1 / top5 是按 target 分位数定义的尾部区域，不是分类任务里的 top-k accuracy。

### 8.2 辅助 loss

主 loss 后面还会叠加若干辅助项，具体是否生效由配置中的权重决定：

- `sample_peak_loss_weight`: 每个频率样本的峰值约束
- `sample_top5_loss_weight`: 每个频率样本 top5 区域约束
- `sample_top1_loss_weight`: 每个频率样本 top1 区域约束
- `sample_mean_loss_weight`: 样本均值约束
- `low_target_overprediction_weight`: 低响应区过预测惩罚
- `background_false_peak_weight`: 背景假峰惩罚
- `participation_residual_weight`: participation baseline residual 约束
- `participation_residual_band_weight`: rank band residual 约束

标准 `low_rank_curves_fast_eval` 这条实验没有对 `curve_values` 本身加直接监督。也就是说：

```text
G1/G2/G3 的形状只通过最终 prediction 的 loss 间接学习。
```

如果某些 calibrated 变体打开了 `center_curve_consistency_weight`，才会额外约束同 case、同节点或相邻结构下的曲线一致性。但这不是 low-rank head 的必要条件。

## 9. 训练过程

训练入口整体流程在 `Trainer._prepare` 和 `train_one_epoch`：

1. 解析配置继承，得到最终 dataset / feature / target / loss / model / training 配置。
2. 按 case 划分 train / val / test，并展开 per-frequency 样本路径。
3. 构造或读取 scaler cache：
   - `x_scaler`: 标准化连续输入特征
   - `y_scaler`: 标准化 `log1p(target_raw)`
   - `feature_schema`: 保存 feature name、target quantile 统计等
4. 构建 `PointSampleDataset`。
5. `DataLoader` 每次取一批 frequency samples。
6. `collate_point_samples` 把多个样本的点拼成一个大 `PointBatch`。
7. `_point_chunks` 再按 `training.batch_size` 把点切成 GPU chunk。
8. 每个 chunk 做一次 forward、loss、backward、optimizer step。

当前 low-rank 实验还会从 hotspot 模型初始化：

```yaml
training:
  init_checkpoint: node/outputs/node_mlp_v6_disk_center_hotspot_features/best.pt
  allow_partial_init_checkpoint: true
```

因为 low-rank head 新增了 `frequency_head`、`node_mixing_head`、`node_scale_head`，这些新参数不能从旧 checkpoint 完整加载，所以使用 partial init：已有主干网络权重尽量继承，新 head 随机初始化后训练。

## 10. 推理和评估过程

评估时不会使用 teacher forcing，也不会从目标里反推曲线。流程是：

1. 对 val/test 的每个 per-frequency 样本构造同样的输入 features。
2. 调用 `model(features)`。
3. 用 `regression_output(output)` 取最终 `regression`。
4. 用 `y_scaler.inverse_transform` 还原到 log 空间。
5. 用 `expm1` 还原到 raw PSD 空间。

可写成：

```text
pred_scaled = regression_output(model(features))
pred_log = y_scaler.inverse_transform(pred_scaled)
pred_raw = expm1(pred_log).clamp_min(0)
```

指标计算包括：

- overall log MAE / MAE / within25
- 每个样本 top1 / top5 / top10 等尾部区域的 log MAE / MAE / within25
- peak_relative_error
- 可选 region metrics

`within25` 在 raw 空间按相对误差计算：

```text
abs(pred_raw - target_raw) / abs(target_raw) <= 0.25
```

top1 / top5 在评估时按每个 frequency sample 内的 target 分位数选点：

```text
top1: target_raw >= sample q0.99
top5: target_raw >= sample q0.95
```

所以 top1 / top5 指标衡量的是高响应尾部区域的预测质量。

## 11. 这个结构为什么适合圆盘中心

圆盘中心区域的主要困难是：不同节点的频响曲线高度相关，但每个节点又有不同的幅值、相位近似和局部峰值差异。普通 pointwise MLP 对每个点独立预测，容易在高响应尾部出现局部不连续或峰值错位。

`low_rank_curve_head` 给模型加了一个归纳偏置：

- 频率方向只允许先生成少数几条全局 latent curve。
- 节点方向主要学习如何混合这些曲线。
- 小 residual 保留局部修正能力。

因此模型自由度从“每个节点、每个频率完全独立预测”变成“少量频率基函数 + 节点混合 + 小残差”。这个约束对圆盘中心这种空间上强相关、频率上有共振结构的问题更合适。

## 12. 需要注意的边界

1. `G1/G2/G3` 没有固定物理含义，不应解释成一阶/二阶/三阶模态。
2. 三条曲线不是数据预处理产物，而是模型参数和输入共同决定的在线输出。
3. `curve_values` 没有直接 label，训练只约束最终 `regression`。
4. 如果输入 feature schema 改变，frequency/node feature indices 会重新按名称匹配，需要确认匹配结果是否仍符合预期。
5. 推理阶段只需要输入 features 和模型参数，不需要训练集目标或任何离线曲线表。
