# node_mlp 中心区域预测优化方案

## 1. 背景

当前模型：

```text
node/outputs/node_mlp_v6_fullpart_disk_center
```

已经从耳片区域扩展到全零件数据，并补充了全零件、耳片、圆盘、圆盘中心区域的 TopK within25 指标。当前最突出的问题不是全场平均误差，而是圆盘中心区域的峰值和曲线形状预测不稳定。

中心区域对应：

```text
disk_center_region = center_couple_mask OR center_node_mask
```

## 2. 当前现象

### 2.1 区域指标差异

在测试集区域指标中，中心区域明显弱于耳片和全零件：

| 指标 | 数值 | 说明 |
| --- | ---: | --- |
| `fullpart_stress_top5_within25_ratio` | 0.557 | 全零件 top5 within25 |
| `earpiece_region_stress_top5_within25_ratio` | 0.650 | 真实耳片区域 top5 within25 |
| `disk_stress_top5_within25_ratio` | 0.368 | 圆盘区域 top5 within25 |
| `disk_center_stress_top5_within25_ratio` | 0.224 | 圆盘中心 top5 within25 |
| `disk_center_stress_top1_within25_ratio` | 0.139 | 圆盘中心 top1 within25 |
| `disk_center_stress_top5_pred_target_ratio` | 0.097 | 圆盘中心 top5 预测峰值明显偏低 |
| `disk_center_stress_top1_pred_target_ratio` | 0.047 | 圆盘中心 top1 预测峰值更严重偏低 |

这说明中心区域问题集中在高响应热点，尤其是 top1/top5 峰值幅值，而不是简单的全场偏差。

### 2.2 中心区域曲线拟合现象

对 8 个测试 case、每个 case 4 个中心高响应节点画频率响应曲线后，汇总结果如下：

| 指标 | 数值 |
| --- | ---: |
| 曲线数量 | 32 |
| 真实峰值中位数 | 1.628e9 |
| 预测峰值中位数 | 1.416e7 |
| 峰值 pred/target 中位数 | 0.0085 |
| 曲线 within25 均值 | 0.057 |
| 曲线 within25 中位数 | 0.000 |
| log 曲线相关均值 | 0.960 |
| 峰值频率偏差中位数 | 0 Hz |
| 欠预测比例均值 | 0.890 |

进一步看同一个 case 内的中心节点曲线：

| 对比项 | 真实曲线 | 预测曲线 |
| --- | ---: | ---: |
| 同 case 节点间 log 曲线相关性 | 0.999995 ~ 1.000000 | 0.900 ~ 0.979 |
| 最差节点对相关性 | 约 1.000 | 0.755 |
| 去掉节点幅值偏置后的形状离散 | 0.0004 ~ 0.0069 log | 0.382 ~ 0.812 log |

结论：

- 真实中心区域曲线基本是同一条频率响应曲线乘以不同节点系数。
- 预测曲线不仅幅值偏低，而且同一 case 内不同中心节点的走势被模型预测得过于分散。
- 模型通常能找到峰值频率，但严重低估峰值幅值。

## 3. 原因分析

### 3.1 当前 Point-wise MLP 缺少中心区域一致性约束

当前模型形式可以概括为：

```text
stress(node_i, f) = MLP(node_i 特征, frequency 特征, PSD 特征, 模态特征, 几何参数...)
```

每个节点都独立预测。模型没有显式约束：

```text
同一个 case 的中心区域节点应该共享主要频率响应形状
```

因此，模型可以把节点局部几何、中心 mask、mode shape、modal gradient 等特征和频率特征自由组合，学出每个中心点不同的频率走势。真实数据里这些局部特征主要影响幅值比例，但模型把它们误学成了曲线形状差异。

### 3.2 全零件训练目标稀释了中心峰值

全零件样本中，中心区域高峰值节点数量少，且峰值极端。常规 log-stress 回归和全场平均损失更容易优化大量中低响应点。

这会导致模型学到保守预测：

```text
频率位置大致正确，但峰值幅值被压低 1 到 2 个数量级以上
```

### 3.3 现有 TopK 约束不够区域化

现有 sample top1/top5 loss 更偏向全样本最高点。如果全零件或耳片区域高响应更稳定，中心区域 top1/top5 的梯度贡献仍然不足。

中心区域需要独立的区域 TopK 约束，否则模型很容易在全局指标还可以的情况下继续忽略中心热点。

## 4. 优化目标

优化目标不是单纯提高中心区域训练集指标，而是提高中心区域可泛化的热点预测能力。

核心目标：

- 提升 `disk_center_stress_top1/top5_within25_ratio`。
- 提升 `disk_center_stress_top1/top5_pred_target_ratio`，降低系统性欠预测。
- 保持峰值频率定位能力，不破坏已有 `log_curve_corr`。
- 降低同 case 中心节点预测曲线的无意义分叉。
- 不明显牺牲 `fullpart`、`disk`、`earpiece_region` 指标。

建议初始验收目标：

| 指标 | 当前 | 第一阶段目标 |
| --- | ---: | ---: |
| `disk_center_stress_top5_within25_ratio` | 0.224 | >= 0.35 |
| `disk_center_stress_top1_within25_ratio` | 0.139 | >= 0.25 |
| `disk_center_stress_top5_pred_target_ratio` | 0.097 | >= 0.30 |
| 中心曲线峰值 pred/target 中位数 | 0.0085 | >= 0.10 |
| 中心曲线 `log_curve_corr_mean` | 0.960 | >= 0.94 |
| `fullpart_stress_top5_within25_ratio` | 0.557 | 下降不超过 0.03 |
| `earpiece_region_stress_top5_within25_ratio` | 0.650 | 下降不超过 0.03 |

## 5. 推荐方案

### 5.1 第一阶段：低风险损失优化

优先做区域化 loss，不改模型结构。

#### 方案 A：增加 `disk_center_top1/top5` auxiliary loss

在每个 batch 的每个样本内，基于 `disk_center_region` mask 单独计算中心区域 top1/top5 loss：

```text
loss = base_loss
     + w_center_top5 * loss(center_region_top5)
     + w_center_top1 * loss(center_region_top1)
```

建议初始权重：

```yaml
disk_center_top5_loss_weight: 0.15
disk_center_top1_loss_weight: 0.25
```

如果中心区域仍然明显欠预测，再逐步提升到：

```yaml
disk_center_top5_loss_weight: 0.30
disk_center_top1_loss_weight: 0.50
```

注意：

- target top1/top5 只用于训练 loss 和验证指标。
- 不把 target 排名、target top mask、真实峰值作为模型输入。
- 这不是标签泄露，和当前 sample top1/top5 loss 的性质一致。

#### 方案 B：中心区域高响应样本加权

对中心区域高 target 分位点增加 point-wise 权重：

```text
center_peak_weight = 1 + alpha * I(node in center_region and target in center_top_quantile)
```

建议从小权重开始：

```yaml
center_region_point_weight: 1.5
center_region_top_quantile: 0.95
```

如果只加 top loss 不稳定，再叠加该项。

#### 方案 C：验证集模型选择改成区域联合指标

不要只用全局验证 loss 或全零件 top 指标选 checkpoint。建议增加 composite score：

```text
score =
  0.35 * disk_center_top5_within25
+ 0.20 * disk_center_top1_within25
+ 0.15 * disk_top5_within25
+ 0.15 * fullpart_top5_within25
+ 0.15 * earpiece_region_top5_within25
```

这样能避免模型为了全局平均表现继续牺牲中心区域。

### 5.2 第二阶段：中心区域曲线一致性正则

如果第一阶段后中心峰值提升，但同 case 中心曲线仍然分叉，可以增加弱一致性约束。

目标不是强制所有中心点完全一样，而是限制频率走势无意义分叉。

可选实现：

```text
对同一个 case 的中心区域节点，
先去掉每个节点自己的平均 log 偏置，
再约束剩余的频率响应形状接近。
```

形式：

```text
normalized_log_pred_i(f) = log_pred_i(f) - mean_f(log_pred_i(f))
consistency_loss = variance_i(normalized_log_pred_i(f))
```

实现要求：

- 需要 grouped mini-batch，使同一个 case 的多个频点可以同时进入 batch。
- 或者离线构造中心区域 curve batch，作为辅助训练 batch。
- 权重要小，避免抹掉真实局部差异。

建议权重：

```yaml
center_curve_consistency_weight: 0.01 ~ 0.05
```

### 5.3 第三阶段：`common frequency response × node scale` 结构

如果损失优化和一致性正则仍不够，再考虑中心区域专用结构。

中心区域真实响应更接近：

```text
stress_i(f) ≈ common_response(case, f) * node_scale_i(case)
```

log 空间为：

```text
log_stress_i(f) ≈ common_log_response(case, f)
                + node_log_scale_i(case)
                + small_residual_i(case, f)
```

含义：

- `common_log_response(case, f)`：负责学这个 case 的频率响应走势和峰值位置。
- `node_log_scale_i(case)`：负责学中心区域不同节点之间的幅值比例。
- `small_residual_i(case, f)`：保留少量自由度，避免过强约束。

建议只对 `disk_center_region` 启用该结构，其他区域仍保持原来的 point-wise MLP 或共享 trunk。

推荐形式：

```text
if node in disk_center_region:
    pred_log = common_head(global_frequency_features)
             + scale_head(node_geometry_and_modal_shape_features)
             + residual_weight * residual_head(all_features)
else:
    pred_log = pointwise_head(all_features)
```

其中：

- `common_head` 不能使用 case id 查表，只能使用几何参数、PSD、频率、模态频率、FRF 等可泛化输入。
- `scale_head` 不能使用 target 派生信息，只能使用节点坐标、mask、距离、mode shape 等输入。
- `residual_weight` 建议从 `0.1` 开始，避免结构过硬。

该结构的优势：

- 符合当前可视化揭示的中心区域物理规律。
- 减少 point-wise MLP 在中心区域的自由度。
- 有助于小样本、高峰值区域泛化。

风险：

- 如果某些 case 的中心节点确实存在不同局部模态参与，过强 factorization 会欠拟合。
- 需要保留 residual，并用验证集曲线图确认没有把真实差异抹平。

### 5.4 第四阶段：有限曲线族 / low-rank curve-family head

进一步观察后，中心区域并不一定只有一条公共频响曲线。更稳妥的假设是：

```text
同一 case 的中心区域频率-应力曲线由有限个走势族组成。
```

rank-1 的 `common_response × node_scale` 只是该假设的特例。更一般的形式是：

```text
log_stress_i(f) ≈ bias_i(case)
                + Σ_k w_i,k(case) * G_k(case, f)
                + residual_i(case, f)
```

其中：

- `G_k(case, f)`：第 `k` 个典型频率响应走势，主要由频率、PSD、模态频率、FRF 放大等 case-frequency 特征决定。
- `w_i,k(case)`：节点对不同曲线走势族的参与权重，主要由节点几何、中心区域位置、mode shape、modal gradient 等空间特征决定。
- `bias_i(case)`：节点整体幅值倍率，用于表达热点 patch 的稳定放大倍数。
- `residual_i(case, f)`：小残差项，处理模态交叉、局部异常和 Mises 非线性带来的非低秩部分。

该形式有更直接的理论支撑：

- 结构动力学的模态叠加认为动态应力响应可以由有限 normal mode stresses 叠加。
- POD/SVD 降阶方法把空间-频率响应矩阵分解为少量主模态和对应响应系数。
- 频响分析中的 low-rank approximation 说明频域响应矩阵可以存在有效低秩近似。
- DeepONet / low-rank neural operator 的 branch-trunk 结构，本质也是有限基函数和系数的乘积组合。

因此，中心区域更适合从 point-wise MLP 升级为低秩曲线族结构：

```text
frequency_head(global/modal/frequency features) -> G_1...G_K
node_mixing_head(node geometry/modal shape features) -> w_1...w_K
node_scale_head(node geometry/hotspot features) -> bias_i
residual_head(all features) -> small residual

pred_log = bias_i + Σ_k w_i,k * G_k + λ * residual
```

推荐初始配置：

```yaml
model:
  low_rank_curve_head: true
  center_curve_rank: 3
  center_residual_weight: 0.10
```

注意事项：

- 不应强行假设 rank-1；先用 SVD/PCA 诊断确认 rank-1/2/3/5 的解释率。
- `frequency_head` 不应使用 node id、case id 查表，只能使用可泛化的全局/频率/模态/FRF 特征。
- `node_mixing_head` 不应使用 target 派生信息，只能使用节点空间和模态形状特征。
- `residual_weight` 初始取小值，避免低秩结构被普通 point-wise residual 完全覆盖。
- top1/top5 的幅值校准仍需要 peak underprediction loss；low-rank 结构解决“曲线走势族和空间参与”，不自动解决“峰值倍率偏低”。

## 6. 实验计划

### 6.1 实验 0：固定基线

目的：确保后续对比稳定。

输出：

- `test_metrics.json`
- `center_region_curve_summary.csv`
- 8 case 中心区域曲线图
- `center_region_peak_scatter.png`

当前基线目录：

```text
node/outputs/node_mlp_v6_fullpart_disk_center/center_region_frequency_curves_test_8cases
```

### 6.2 实验 1：只加中心区域 TopK loss

配置：

```yaml
loss:
  disk_center_top5_loss_weight: 0.15
  disk_center_top1_loss_weight: 0.25
```

评估：

- 比较 `disk_center_stress_top1/top5_within25_ratio`
- 比较 `disk_center_stress_top1/top5_pred_target_ratio`
- 检查 `fullpart` 和 `earpiece_region` 是否明显下降
- 画中心区域曲线

预期：

- 中心 top1/top5 within25 有明显提升。
- 峰值 pred/target ratio 上升。
- 曲线形状分叉可能仍存在，但幅值系统性欠预测会缓解。

### 6.3 实验 2：中心 TopK loss + 高响应点权重

在实验 1 基础上增加：

```yaml
loss:
  center_region_point_weight: 1.5
  center_region_top_quantile: 0.95
```

适用情况：

- 实验 1 的中心 topK 提升不够。
- 或者 top1 仍然严重欠预测。

风险控制：

- 如果背景区域出现明显 false peak，降低 point weight。
- 同时观察 disk/fullpart 指标，避免局部优化过度。

### 6.4 实验 3：弱曲线一致性正则

适用情况：

- 中心峰值幅值已有提升。
- 但同 case 中心节点预测曲线走势仍然明显分叉。

配置：

```yaml
loss:
  center_curve_consistency_weight: 0.01
```

逐步尝试：

```text
0.01 -> 0.03 -> 0.05
```

停止条件：

- `log_curve_corr_mean` 不下降明显。
- `disk_center_top5_within25` 提升或保持。
- 中心节点间预测曲线形状离散下降。

### 6.5 实验 4：中心区域 factorized head

适用情况：

- 前三个实验仍无法解决中心曲线分叉。
- 或者中心峰值提升依赖过大 loss 权重，导致全局指标下降。

实现优先级：

1. 保留现有 trunk。
2. 增加 `common_head`、`scale_head`、`residual_head`。
3. 仅对 `disk_center_region` 使用 factorized output。
4. 其他区域沿用原 point-wise output。

对比项：

- 无 factorization
- hard factorization：`common + scale`
- soft factorization：`common + scale + 0.1 * residual`

优先采用 soft factorization。

### 6.6 实验 5：有限曲线族低秩诊断与 rank-K head

实验 5 分三步，对应当前分支 `feature/node-mlp-center-low-rank-curves` 的验证 1、2、3。

#### 验证 1：中心区域 target 曲线低秩诊断

目的：先验证“有限曲线走势”是否是数据事实，而不是直接改模型。

对每个 case 构造：

```text
Y[node, freq] = log1p(target)
```

然后对 `Y` 做去均值 SVD，报告：

- rank-1 / rank-2 / rank-3 / rank-5 / rank-10 解释率；
- 同 case 中心节点去均值曲线的 pairwise correlation；
- top1/top5 节点在第一主成分权重上的分布；
- 中心点型和外环型 case 的差异。

命令：

```bash
PYTHONPATH=node conda run -n ci2n python node/center_curve_low_rank_diagnostics.py \
  --config node/configs/node_mlp_v6_disk_center_hotspot_features.yaml \
  --split test \
  --num-cases 32 \
  --output-dir node/outputs/node_mlp_v6_disk_center_low_rank_diagnostics
```

判断标准：

- 如果 rank-3 或 rank-5 解释率明显高，说明 finite curve-family 假设成立。
- 如果 rank-1 很高，说明原 `common_response × node_scale` 已足够。
- 如果 rank-3/rank-5 仍低，说明中心区域曲线族不是主要瓶颈，应继续优先做 peak calibration 或数据分桶。

#### 验证 2：只开 rank-K center head

目的：验证结构归纳偏置本身是否有收益。

配置：

```text
node/configs/node_mlp_v6_disk_center_low_rank_curves.yaml
```

该配置：

- 继承 hotspot feature 配置；
- 开启 `low_rank_curve_head: true`；
- 使用 `center_curve_rank: 3`；
- 使用 `center_residual_weight: 0.10`；
- 从 `node/outputs/node_mlp_v6_disk_center_hotspot_features/best.pt` 部分初始化。

命令：

```bash
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=node conda run -n ci2n python node/train_node_mlp.py \
  --config node/configs/node_mlp_v6_disk_center_low_rank_curves.yaml
```

#### 验证 3：rank-K head + peak calibration + 弱曲线一致性

目的：同时验证结构低秩和峰值幅值校准。

配置：

```text
node/configs/node_mlp_v6_disk_center_low_rank_curves_calibrated.yaml
```

相对验证 2 增加：

```yaml
loss:
  disk_center_peak_loss_weight: 0.35
  disk_center_top5_loss_weight: 0.30
  disk_center_top1_loss_weight: 0.55
  disk_center_top1_under_weight: 0.90
  disk_center_top5_under_weight: 0.50
  disk_center_under_margin_log: 0.09531018
  center_curve_consistency_weight: 0.02
```

该实验默认和验证 2 并行对照，仍从 hotspot feature checkpoint 初始化，而不是等待验证 2 的 best checkpoint：

```yaml
training:
  init_checkpoint: node/outputs/node_mlp_v6_disk_center_hotspot_features/best.pt
```

命令：

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=node conda run -n ci2n python node/train_node_mlp.py \
  --config node/configs/node_mlp_v6_disk_center_low_rank_curves_calibrated.yaml
```

验收重点：

- `disk_center_stress_top1_within25_ratio`
- `disk_center_stress_top5_within25_ratio`
- `disk_center_stress_top1_pred_target_ratio`
- `disk_center_stress_top5_pred_target_ratio`
- 曲线诊断中的 `log_curve_corr` 和 `peak_frequency_delta_hz`

预期解释：

- 如果验证 2 提升曲线相关但 within25 不明显，说明结构有效但峰值倍率仍不足。
- 如果验证 3 明显提升 top1/top5 pred/target 和 within25，说明问题需要“低秩曲线族 + 幅值校准”同时处理。
- 如果验证 3 牺牲普通区域指标过多，应降低 `center_curve_consistency_weight` 或 `disk_center_top1_under_weight`。

## 7. 泛化能力控制

允许使用：

- 几何参数
- 当前频率
- PSD
- 模态频率
- mode shape
- 节点坐标和节点区域 mask
- 训练 loss 内部的 target topK mask

禁止使用：

- case id 查表
- node id 查表
- 测试集统计量
- target topK mask 作为模型输入
- 真实峰值频率或真实峰值幅值作为模型输入

评估原则：

- 所有权重和 checkpoint selection 只基于 train/val。
- test 只用于最终报告。
- 中心区域指标提升不能以耳片区域和全零件指标大幅下降为代价。
- 曲线图必须同时看 val 和 test，避免只对测试样例解释。

## 8. 推荐执行顺序

优先顺序：

1. 固定当前 baseline 指标和曲线图。
2. 实现 `disk_center_top1/top5` auxiliary loss。
3. 用区域联合 score 选择 checkpoint。
4. 重新跑 fullpart/disk/center/earpiece 区域指标。
5. 重新画中心区域曲线。
6. 如果中心曲线仍分叉，再加弱曲线一致性正则。
7. 如果仍不稳定，再实现 `common frequency response × node scale` 的中心区域 factorized head。

不建议一开始直接改成复杂结构。先用区域化 loss 解决中心峰值监督不足，再决定是否需要结构约束。

## 9. 预期结论

当前中心区域问题的核心不是模型完全不懂模态频率，而是：

```text
模型知道峰值大概在哪个频率，
但不知道中心区域峰值应该放大到什么量级，
也没有学到同一 case 中心节点应该共享主要曲线形状。
```

因此，优化方向应从两个层面推进：

- 损失层面：让中心 top1/top5 峰值真正参与训练目标。
- 结构层面：在必要时把中心区域拆成共同频率响应和节点幅值系数，降低无意义自由度。

这样做不是为了记住中心区域，而是给模型加入更符合数据和物理规律的归纳偏置。只要不引入 case/node 查表和 target 派生输入，合理的区域 loss 和软结构约束通常会提高中心区域泛化能力。
