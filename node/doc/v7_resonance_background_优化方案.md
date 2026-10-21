# V7 resonance-background 优化方案

更新日期：2026-05-22  
目标任务：耳片区域点级 `MISES_psd_density` 预测  
建议版本名：`node_mlp_v7_resonance_background`

## 1. 当前问题判断

从 `case89_mises_full_export` 的热点/背景频率曲线可以看到：

- 热点点的主峰位置已经命中，`242.2Hz` 附近峰位和峰形基本正确。
- 背景点也被模型判定在 `242.2Hz` 附近随全局 FRF 起峰，但预测幅值明显偏大。
- 背景点在 `800Hz`、`1450Hz`、`1800Hz` 附近有接近 0 的波谷，当前模型预测不会下到接近 0，而是保留一个平滑的非零底噪。

这说明当前主要错误不是频率峰位错误，而是：

```text
模型知道当前频率会共振，
但不知道某些背景节点在该模态/频率下其实是弱参与或近零参与。
```

换句话说，当前模型把全局 FRF 放大因子过度迁移到了背景节点上。

## 2. V6 实验结论

V6 在 V5 region-distance 模型基础上增加了：

```text
target_bucket_weights
low_target_overprediction_loss
background_false_peak_loss
participation_residual_loss
```

当前结果显示：

| 指标 | V5 region | V6 当前 best | 结论 |
|---|---:|---:|---|
| test log MAE | 0.28777 | 0.28642 | 小幅变好 |
| test within25 | 0.58340 | 0.58702 | 小幅变好 |
| test relative MAE | 0.58263 | 0.57672 | 小幅变好 |
| test MAE | 4.516e5 | 4.733e5 | 变差 |
| test peak relative error | 0.1680 | 0.1861 | 变差 |
| test top1 log MAE | 0.1349 | 0.1468 | 变差 |
| test top5 log MAE | 0.1744 | 0.1799 | 变差 |
| pred/target mean | 0.984 | 0.944 | 低估加重 |

结论：

```text
V6 证明“压背景误报”方向有效，
但低值/背景约束过粗，会把真实热点也一起压低。
```

因此 V7 不应该继续简单加大低值 loss，而应该把问题改成：

```text
同一 case-frequency 下，真热点必须高于背景假峰；
背景近零波谷要允许模型预测到接近 0；
全局共振幅值和节点参与度要解耦。
```

## 3. V7 总体思路

V7 建议保留 V5 region-distance 的 238 维特征和 checkpoint，改训练目标与模型头。

核心改动：

```text
1. resonance hard negative 挖掘
2. hotspot-vs-false-peak pairwise ranking loss
3. near-zero / valley gate
4. peak-relative / drop-from-peak 结构
5. 频率曲线诊断作为强制验收
```

优先级建议：

```text
V7a: hard negative + ranking loss，不改模型结构
V7b: zero gate，增加一个 near-zero 分类头
V7c: peak-relative/drop-from-peak，进一步解耦峰值尺度和节点参与度
```

## 4. V7a：resonance hard negative + ranking loss

### 4.1 目的

只惩罚真正危险的背景假峰，而不是惩罚所有低响应点。

危险假峰定义：

```text
同一 case-frequency 中：
target rank 低，说明真实不是热点；
pred rank 高，说明模型误报成热点；
且通常出现在模态共振或低频主峰附近。
```

### 4.2 Hard negative 挖掘

用当前 V5 region 或 V6 best 在 train/val split 上跑预测，按每个 `case-frequency` 选点。

正样本：

```text
positive = target top1 / top5 / top10 节点
```

负样本：

```text
negative = target rank 后 50% 或 target < low_threshold
           且 pred rank 前 5% / 前 10%
```

推荐先保存为 CSV：

```text
case_name,frequency_hz,pos_node_index,neg_node_index,
pos_target,pos_pred,neg_target,neg_pred,
target_rank_fraction,pred_rank_fraction,mode_proximity
```

建议输出目录：

```text
node/outputs/diagnostics/v7_resonance_hard_negatives/
```

### 4.3 Pairwise ranking loss

对同一个 `case-frequency` 内的正负点对：

```text
L_rank = mean(relu(margin + pred_log_neg - pred_log_pos))
```

含义：

```text
真实热点预测值必须高于背景假峰预测值至少 margin。
```

推荐初始参数：

```yaml
ranking_loss_weight: 0.05
ranking_margin_log: 0.69314718  # log(2)
ranking_positive: target_top5
ranking_negative: hard_false_peak
max_pairs_per_sample: 64
```

这样不会强迫所有背景点都非常低，只要求：

```text
背景假峰不能排在真热点前面。
```

这直接对应当前曲线里的问题：`242.2Hz` 主峰附近，背景预测跟着热点起峰，但背景不应该接近热点量级。

## 5. V7b：near-zero / valley gate

### 5.1 目的

解决背景曲线近零波谷预测不下去的问题。

当前单回归头容易学出非零底噪：

```text
真实 target 接近 0；
预测保持 0.1、1、10 或更高；
绝对误差可能不大，但相对误差极差。
```

### 5.2 模型输出

共享 MLP backbone 后增加两个头：

```text
nonzero_logit: 当前点当前频率是否非零响应
stress_log:    非零响应幅值
```

预测组合：

```text
p_nonzero = sigmoid(nonzero_logit)
pred_raw = p_nonzero * expm1(stress_log)
```

或 log 空间近似：

```text
pred_log = p_nonzero * stress_log
```

### 5.3 zero label

不要只用严格 0，建议用近零阈值：

```text
zero_label = target < zero_threshold
```

初始扫描：

```text
zero_threshold = 1
zero_threshold = 10
zero_threshold = target_positive_p99 * 1e-7
```

### 5.4 Loss

```text
L = L_reg
  + zero_bce_weight * BCE(nonzero_logit, target >= zero_threshold)
  + zero_over_weight * relu(pred_log - log1p(zero_cap))^2
```

推荐初始参数：

```yaml
zero_gate:
  enabled: true
  zero_threshold: 10.0
  bce_weight: 0.10
  overprediction_weight: 0.03
  zero_cap: 10.0
```

zero overprediction loss 只对以下点开启：

```text
target < zero_threshold
且 target rank 后 50%
且不属于真实 top10/top20
```

不要像 V6 那样对所有低值点强压。

## 6. V7c：peak-relative / drop-from-peak 结构

### 6.1 目的

解耦：

```text
当前 case-frequency 的全局响应峰值尺度
当前节点相对峰值的参与度/drop
```

当前直接回归：

```text
pred_log(node, f)
```

容易把全局 FRF 峰值扩散到背景点。

建议改成：

```text
pred_peak_log(case, f) = graph/sample 级峰值头
drop_from_peak(node, f) = 节点距离峰值差
pred_log(node, f) = pred_peak_log - softplus(drop_from_peak)
```

对当前曲线的含义：

```text
hotspot 点 drop 小；
background 点 drop 大；
242.2Hz 全局峰可以保留，但背景不会被抬到热点量级。
```

### 6.2 Loss

```text
L = L_node_reg
  + peak_loss_weight * SmoothL1(pred_peak_log, target_peak_log)
  + drop_rank_loss_weight * pairwise_drop_loss
```

推荐先作为 V7c，而不是第一步就做，因为它需要改模型输出和 batch 汇总逻辑。

## 7. 频率连续性约束

当前图里背景波谷不是孤立问题，而是频率曲线形态问题。

建议先作为诊断和后续增强，不放在 V7a 第一版主训练里。

可选 loss：

```text
L_delta = SmoothL1(
  pred_log(f + df) - pred_log(f),
  target_log(f + df) - target_log(f)
)
```

远离模态处加二阶平滑：

```text
L_smooth = |pred_log(f + df) - 2 * pred_log(f) + pred_log(f - df)|
```

权重策略：

```text
远离模态：smooth 权重大
靠近模态：smooth 权重小，允许真实峰
```

它适合解决：

```text
背景曲线非物理底噪；
远离模态处突然冒峰；
近零波谷不能下探。
```

## 8. 评估指标

V7 不应只看全局 `log_mae` 和 `within25`，必须增加背景误报指标。

建议新增：

### 8.1 false peak rate

每个 `case-frequency`：

```text
pred top5% 中 target 不在 top20% 的比例
```

或：

```text
pred top10 节点里 target rank 后 50% 的个数
```

### 8.2 hotspot-vs-background margin

```text
margin = median(pred_log(target_top5)) - median(pred_log(hard_negative))
```

希望 margin 越大越好。

### 8.3 near-zero overprediction

对 `target < zero_threshold` 的点：

```text
over_cap_rate = mean(pred > zero_cap)
zero_over_log_mae = mean(max(pred_log - log1p(zero_cap), 0))
```

### 8.4 曲线诊断

继续保留以下频率曲线图：

```text
hotspot1 / hotspot2
background1 / background2
```

并单独统计：

```text
background within25
background low-frequency peak pred/target
background valley pred floor
hotspot peak pred/target
```

## 9. 推荐实验矩阵

### V7a ranking-only

```yaml
init_checkpoint: node/outputs/node_mlp_v5_exp_region_distance_within25_short/best.pt
low_target_overprediction_weight: 0.0
background_false_peak_weight: 0.0
participation_residual_weight: 0.0
ranking_loss_weight: 0.05
ranking_margin_log: 0.69314718
sample_peak_loss_weight: 0.25
sample_top5_loss_weight: 0.20
sample_top1_loss_weight: 0.35
```

目的：验证精准 hard negative ranking 是否能降低背景假峰，同时不压热点。

### V7b zero-gate

```yaml
zero_gate:
  enabled: true
  zero_threshold: 10.0
  bce_weight: 0.10
  overprediction_weight: 0.03
  zero_cap: 10.0
ranking_loss_weight: 0.03
```

目的：解决背景近零波谷预测不下去。

### V7c peak-relative

```yaml
stress_peak_relative:
  enabled: true
  combine_prediction: true
  peak_loss_weight: 0.20
  drop_rank_loss_weight: 0.05
```

目的：解耦全局峰值和节点参与度。

## 10. 成功标准

V7 至少要满足：

```text
test within25 >= V5 region
test log_mae <= V5 region
test peak_relative_error 不劣于 V5 region 超过 3%
test top1/top5 log_mae 不劣于 V5 region 超过 3%
background false peak rate 明显下降
背景频率曲线在 242.2Hz 不再被整体抬高
背景近零波谷预测 floor 明显降低
```

如果只提升 `within25/log_mae`，但继续牺牲 `top1/top5/peak`，则不能视为成功。

## 11. 当前建议

优先实现 V7a：

```text
hard negative CSV 构建脚本
pairwise ranking loss
false peak diagnostics
```

原因：

```text
它不需要改变 238 维特征；
可以复用 V5 region checkpoint；
直接针对当前图里的低频共振背景误报；
比 V6 的低值全量压制更精准，预计更不容易伤热点。
```

V7b 和 V7c 分别作为后续结构升级，用来处理近零波谷和全局峰值/节点参与度解耦。
