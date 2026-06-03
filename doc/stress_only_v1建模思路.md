# stress_only_v1 建模思路

## 1. 项目定位

`stress_only_v1` 是从主线 `case7_gnn` 中独立出来的应力预测分支，目标不是同时预测频率、RTA 等多任务结果，而是专门优化节点级应力场，尤其是高应力热点区域的预测质量。

核心变化可以概括为：

- 去掉 RTA 分支，只保留 stress 相关任务。
- 将每个 case 拆成按频点组织的样本，在给定几何、PSD、模态信息和当前频率的条件下预测该频点的节点应力。
- 输出目标聚焦 `MISES_psd_density` 或最终响应应力 `RMises_native/RMises`。
- 训练目标、验证指标和模型头都围绕 hotspot stress 设计，而不是追求全场平均误差最低。

代码主入口：

- 训练入口：`stress_only_v1/train_stress_only.py`
- 预测入口：`stress_only_v1/predict_stress_only.py`
- 核心模型：`stress_only_v1/case7_gnn_stress_only/models.py`
- 数据和特征：`stress_only_v1/case7_gnn_stress_only/data.py`
- 训练与损失：`stress_only_v1/case7_gnn_stress_only/trainer.py`

---

## 2. 数据建模方式

### 2.1 图表示

每个有限元 case 被建模成一个图：

- 节点：网格节点，基础特征来自 `nodes.csv`，通常为 `[x, y, z, bc_mask]`。
- 边：网格邻接关系，来自 `edges.csv`，边特征通常为 `[dx, dy, dz, dist]`。
- 全局条件：来自 `global.json`，包含几何参数、PSD、模态频率等。

默认会把图转成无向图。反向边会复制一份，并把 `dx/dy/dz` 取反，`dist` 保持不变。

### 2.2 样本组织

`stress_only_v1` 支持两种样本模式：

- `case`：一个 case 对应一个最终响应应力样本。
- `per_frequency`：一个 case 下的每个频点文件对应一个样本。

当前主要配置采用：

```yaml
dataset:
  sample_mode: per_frequency
  include_zero_frequency: false
  min_frequency_hz: 20.0
  max_frequency_hz: 2000.0
```

也就是说，同一个几何 case 会展开为多个频率样本。模型看到的是：

```text
几何图结构 + 全局几何/PSD/模态条件 + 当前频率 -> 当前频点的节点应力场
```

### 2.3 监督目标

应力目标只保留一列：

- per-frequency 样本：`MISES_psd_density`
- final-response 样本：`RMises_native` 或 `RMises`

训练前会先做应力变换：

```text
stress_encoded = log1p(max(stress, 0))
```

然后在训练集上拟合 target scaler：

```text
stress_normalized = (stress_encoded - mean) / std
```

模型实际回归的是标准化后的 log-stress，评估和导出时再反变换回真实应力尺度。

---

## 3. 输入特征思路

`stress_only_v1` 的特征设计原则是：只加入可靠、可稳定复现、和热点位置或响应幅值强相关的物理先验。

详细字段可参考 `doc/GNN特征工程说明.md`。这里按建模作用总结。

### 3.1 节点侧：定位热点

节点特征除了基础坐标和边界掩码，还会追加高可靠性几何增强特征：

- 归一化坐标和极坐标：`x/radius`、`y/radius`、`z/thickness`、`r/radius`、`sin(theta)`、`cos(theta)`。
- 盘边界距离：帮助模型识别外缘、耳片根部等容易产生高应力的位置。
- 耳孔相关距离：节点到最近耳孔边缘的局部/全局归一化距离，以及 soft near-ear-hole mask。
- 中心耦合区相关距离：节点到中心质量耦合半径的相对位置和 soft mask。

这些特征的作用是把“热点可能在哪里”显式暴露给网络，减少模型只靠消息传递从坐标中反推几何关系的难度。

### 3.2 全局侧：描述工况和响应条件

全局特征用于告诉模型当前样本对应什么几何和激励：

- `params_list`：耳片厚度、耳片径向距离、耳片宽度、孔距、倒角、板半径、附加质量等。
- `psd_points`：输入谱点展开后的向量。
- `freq_top3`：前三阶模态频率。
- `frequency_scalar`：当前频点。
- `frequency_relations`：当前频率相对前三阶模态频率的距离、最近模态距离和一阶频率比值。

其中 `frequency_relations` 很关键，因为同样的几何和 PSD，在不同频率处的响应可能完全不同。模型需要知道当前频点是在模态附近、模态之间，还是远离主要模态。

### 3.3 模态振型融合：引入频率相关的空间形状先验

当 `features.use_mode_shapes: true` 时，模型会加载 `mode_shapes/*.csv`，用当前频率与各阶模态频率的关系计算模态权重，再生成节点级或全局级模态摘要。

常用配置是：

```yaml
features:
  use_mode_shapes: true
  mode_shape_count: 10
  mode_shape_weighting: resonance
  mode_shape_include_nearest: true
  mode_shape_include_frequency_context: true
```

节点侧会拼入：

- 按模态权重加权后的 `|U1|/|U2|/|U3|/U_mag`
- 与当前频率最近的模态振型
- 最近模态的频率 gap 和权重

这相当于把“当前频点可能激发出什么空间形态”作为先验输入。模型仍然学习应力分布，但不必完全从几何和频率标量中间接推断振型影响。

---

## 4. 网络结构

核心模型是 `FieldGNN`。它由三部分组成：

```text
输入编码 -> 条件化图消息传递 -> 应力/热点/峰值解码头
```

### 4.1 输入编码

`GraphEncoder` 分别编码三类输入：

- `node_encoder`：节点特征 MLP。
- `edge_encoder`：边特征 MLP。
- `global_encoder`：全局特征 MLP。

如果开启：

```yaml
model:
  conditioning:
    enabled: true
```

则全局特征会先经过 `case_encoder` 得到一个 case conditioning state，再用于调制消息和解码过程。

### 4.2 条件化消息传递

每一层 `EdgeMessagePassingLayer` 的逻辑是：

```text
message = MLP([source_node_state, edge_state, global_state])
aggregated = mean(message grouped by dst node)
delta = MLP([node_state, aggregated, global_state])
next_node_state = LayerNorm(node_state + delta)
```

如果开启 case conditioning，message 和 update 都会经过类似 FiLM 的调制：

```text
features = features * (1 + tanh(scale)) + shift
```

其中 `scale/shift` 由全局 case conditioning state 生成。这样做的含义是：同一套图卷积参数可以根据不同几何、PSD、频率和模态条件动态改变消息传递行为。

### 4.3 图级状态

消息传递后，模型会做图级池化：

```text
graph_state = [node_mean_pool, node_max_pool, global_state]
```

这个 graph state 主要供峰值应力头使用，也用于需要图级输出的任务。

### 4.4 应力解码头

`FieldGNN` 在共享 encoder 之后，又增加了 `rmises_refine_layers`，专门对节点状态做应力预测前的局部细化。

最终节点解码上下文为：

```text
stress_context = [encoder_node_state, refined_stress_state, global_state_for_node]
```

模型根据配置有三种输出形态。

#### 单阶段应力回归

```text
output = [stress]
```

直接对每个节点回归标准化后的 log-stress。

#### 两阶段 hotspot + stress

```text
output = [hotspot_logit, stress]
```

第一头预测节点是否属于热点区域，第二头在 stress context 后额外拼入 `hotspot_logit` 再预测应力：

```text
stress = stress_decoder([stress_context, hotspot_logit])
```

这样热点分类结果会直接影响应力回归头，使模型更关注高应力节点。

#### 峰值辅助头 / peak-relative 结构

当 `stress_peak_relative.enabled: true` 时，模型增加图级峰值应力头：

```text
peak = stress_peak_decoder(graph_state)
```

输出形态为：

```text
[stress_node_head, stress_peak]
```

或两阶段模式下：

```text
[hotspot_logit, stress_node_head, stress_peak]
```

如果：

```yaml
stress_peak_relative:
  combine_prediction: true
```

最终节点预测不是直接用 `stress_node_head`，而是：

```text
pred_stress = pred_peak - softplus(pred_drop_from_peak)
```

也就是让模型学习“每个节点距离全图峰值差多少”。这会强制节点预测不超过图级峰值，适合峰值主导的热点场景。

当前常用配置中 `combine_prediction: false`，表示节点应力仍走直接回归路径，峰值头作为辅助监督帮助模型学习响应幅值。

---

## 5. 损失函数设计

`stress_only_v1` 的损失不是均匀对所有节点做 MSE/SmoothL1，而是围绕热点区域加权。

### 5.1 基础应力回归

基础点级损失支持：

- `mse`
- `smooth_l1`

常用配置是：

```yaml
training:
  loss: smooth_l1
```

对每个节点先计算标准化 log-stress 空间下的回归误差。

### 5.2 热点加权

节点权重由真实应力决定：

- 低应力背景节点使用较低权重，例如 `stress_low_value_weight: 0.05`。
- 应力越接近当前样本高值区，权重越高。
- 可通过 `stress_hotspot_quantile` 和 `stress_hotspot_boost` 对高分位节点额外加权。
- 可通过 `stress_topk_ratio` 和 `stress_topk_weight` 对 top-k 高应力节点再加一项专门损失。

这一部分的核心目的是避免训练被大量低应力背景节点主导。

### 5.3 两阶段 hotspot 损失

当 `stress_two_stage.enabled: true` 时，训练目标包括：

```text
总损失 =
  hotspot 分类损失
  + hotspot 节点回归损失
  + 少量 background 回归损失
  + 加权全场回归损失
  + 峰值辅助损失
  + 可选平滑损失
```

hotspot 标签由真实应力动态生成。常用定义是：

```text
stress >= min(q0.999, 0.05 * peak_stress)
```

训练配置里通过下面参数控制：

```yaml
stress_two_stage:
  threshold_quantile: 0.999
  threshold_peak_ratio: 0.05
  threshold_combine: min
```

分类损失使用 `binary_cross_entropy_with_logits`，并通过 `positive_class_weight` 提高热点正样本权重。

### 5.4 峰值一致性损失

峰值相关损失让预测场的最大值或图级 peak head 接近真实峰值：

```text
loss_peak = loss(pred_peak, true_peak)
```

这对热点应力预测很重要，因为即使平均误差较低，只要峰值幅值偏掉，工程上仍然不可接受。

### 5.5 可选平滑约束

代码支持基于边的应力平滑项：

```text
(pred_i - pred_j)^2 / edge_dist^power
```

并且可以排除边界边和高应力热点边。这样做是为了只约束背景区域的物理平滑性，不压平真正的热点尖峰。

当前常见配置里该项通常为 0：

```yaml
field_loss:
  physics_stress_smoothness_weight: 0.0
```

---

## 6. 训练与评估策略

### 6.1 标准化

模型训练前会在训练集上分别拟合：

- node scaler
- edge scaler
- global scaler
- target scaler

所有验证、测试、预测都复用 checkpoint 中保存的 scaler。

### 6.2 batch 方式

支持多个频点图样本合并成一个 batch。合并时会维护：

- `node_graph_index`
- `edge_graph_index`

用于区分不同图的全局条件、图级峰值和池化结果。

常见大数据配置会使用：

```yaml
training:
  batch_size: 24
  batch_same_case_only: true
  amp: bf16
```

`batch_same_case_only` 会优先把同一 case 的频点样本组织到一起，减少图结构和模态数据重复加载带来的开销。

### 6.3 选模指标

默认 checkpoint selection metric 不是全场 MAE，而是：

```text
stress_hotspot_miss25_rate = 1 - stress_hotspot_within25_ratio
```

其中：

```text
stress_hotspot_within25_ratio =
  真实热点节点中，相对误差 <= 25% 的节点比例
```

也就是说，验证集上越多热点节点落入 25% 相对误差带，模型越好。

这和项目目标一致：宁愿牺牲一些背景节点平均误差，也要优先提升 hotspot stress 的可用性。

### 6.4 输出诊断

训练和评估会输出多类指标：

- 全场：`stress_mae`、`stress_log_mae`、`stress_log_rmse`
- 高分位区域：`stress_top1_mae`、`stress_top5_mae`
- 峰值：`stress_peak_relative_error`
- 热点：`stress_hotspot_mae`、`stress_hotspot_within25_ratio`、`stress_hotspot_miss25_rate`
- 非热点：`stress_non_hotspot_mae`、`stress_non_hotspot_bias`
- 两阶段分类：`hotspot_precision`、`hotspot_recall`、`hotspot_f1`

此外，`evaluate_stress_regions.py` 还可以按工程区域输出更细的区域误差，例如中心耦合区、盘外圈、耳片根部、耳孔附近等。

---

## 7. 总体建模闭环

`stress_only_v1` 的整体思路可以串成下面这条链路：

```text
有限元网格
  -> 图结构节点/边建模
  -> 几何、PSD、当前频率、模态关系作为全局条件
  -> 可选振型特征给出频率相关空间先验
  -> 条件化 GNN 做消息传递
  -> 节点 stress head 预测 log-stress
  -> hotspot head 强化高应力区域识别
  -> peak head 约束全图响应幅值
  -> hotspot-focused loss 和 hotspot validation metric 选模
```

一句话总结：

```text
stress_only_v1 不是均匀拟合全场应力的通用 GNN，
而是一个用频率/模态条件驱动、用热点分类和峰值辅助约束校准的热点应力预测模型。
```

---

## 8. 典型配置解读

以 `stress_two_stage_peak_aux_modes_node_nearest_case1000_cls05_reg25_pos50_bf16_eval5_batch24.yaml` 为例：

- 数据：`case1000`，按 `20-2000 Hz` 的 per-frequency 样本训练。
- 特征：使用 PSD、前三阶频率、当前频点、频率关系、高可靠性几何增强、前 10 阶振型。
- 模型：`hidden_dim=64`、`global_dim=32`、3 层主消息传递、2 层 stress refine。
- 条件化：开启 `conditioning.enabled=true`，case conditioning 维度为 48。
- 头部：开启两阶段 hotspot head，开启 peak auxiliary head。
- 损失：`smooth_l1`，热点区域强加权，背景回归权重很低。
- 训练：`bf16` 混合精度，batch size 24，每 5 个 epoch 评估一次。
- 选模：使用 `stress_hotspot_miss25_rate`，即优先选择热点 25% 相对误差命中率高的 checkpoint。
