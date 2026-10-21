# Fullpart Feature Buckets vs Fusion Low-Rank 对比

日期：2026-06-17

## 1. 实验概览


| 项目         | fusion_low_rank_80g (base)                       | continue32 (fast_eval)               | feature_buckets (singleprocess)                      |
| ---------- | ------------------------------------------------ | ------------------------------------ | ---------------------------------------------------- |
| 仓库         | psd                                              | psd                                  | psdlc                                                |
| 基础配置       | `node_mlp_v6_fullpart_fusion_low_rank_80g`       | 继承 base                              | `node_mlp_v6_fullpart_disk_center` + feature buckets |
| 核心改动       | low-rank curve head (rank8) + RBF + modal region | 从 base epoch 32 继续训练                 | 物理特征百分位分桶 (7组bucket特征)                               |
| 特征维度       | 278                                              | 278                                  | 278 + bucket特征                                       |
| 模型头        | rank8 low-rank curve                             | rank8 low-rank curve                 | 标准 MLP head                                          |
| 初始权重       | disk_center best + earpiece checkpoint           | base epoch 32 best                   | fullpart_disk_center best                            |
| 训练环境       | ci2n                                             | ci2n                                 | yolo                                                 |
| 并行方式       | DDP 多卡                                           | DDP 多卡                               | 单进程 (num_workers=0)                                  |
| Best epoch | 32                                               | 32                                   | 24                                                   |
| 训练耗时       | ~30 epoch                                        | ~3h (32→34 epoch, 2 epoch)           | ~2.5天 (25 epoch)                                     |
| 单 epoch 耗时 | ~15min train + ~4min val                         | ~15min train + ~4min val (fast_eval) | ~1.5h train + ~28min val + test (全量top-k评估)          |


## 2. Val 指标对比


| 指标                           | base (ep32) | continue32 (ep32) | feature_buckets (ep24) |
| ---------------------------- | ----------- | ----------------- | ---------------------- |
| **fullpart log MAE**         | 0.3433      | **0.2895**        | 0.3219                 |
| **fullpart within25**        | 46.89%      | **53.02%**        | 47.36%                 |
| **disk_center log MAE**      | 0.2339      | **0.2179**        | 0.3003                 |
| **disk_center within25**     | 61.14%      | **64.70%**        | 54.13%                 |
| **earpiece_region log MAE**  | 0.5863      | **0.4962**        | 0.3472                 |
| **earpiece_region within25** | 31.25%      | 37.04%            | **50.03%**             |
| **disk log MAE**             | 0.3104      | **0.2616**        | 0.3184                 |
| **disk within25**            | 49.02%      | **55.18%**        | 47.00%                 |
| fullpart pred/target         | 0.9490      | 0.9382            | **0.9197**             |
| disk_center pred/target      | 1.2659      | 1.2583            | **0.4252**             |


## 3. 分区 Val top-k 详情对比

> continue32 的 fast_eval 模式只计算了 `earpiece_stress` 的 top1/top5，未按 region 拆分 top-k。
> base 模型有完整的 per-region top-k val 指标，作为 continue32 的参考基线。

### 3.1 base (fusion_low_rank_80g, ep32) Val 分区 top-k

| 分组 | log MAE | within25 | pred/target |
|---|---|---|---|
| **fullpart** overall | 0.3433 | 46.89% | 0.9490 |
| fullpart top1 | 0.3428 | 48.03% | 0.8707 |
| fullpart top5 | 0.3991 | 42.55% | 0.8791 |
| fullpart top10 | 0.4501 | 39.09% | 0.9018 |
| **earpiece_region** overall | 0.5863 | 31.25% | 0.9227 |
| earpiece_region top1 | 0.3764 | 43.81% | 0.7222 |
| earpiece_region top5 | 0.3433 | 48.57% | 0.8235 |
| earpiece_region top10 | 0.3573 | 47.00% | 0.8416 |
| **disk** overall | 0.3104 | 49.02% | 1.0277 |
| disk top1 | 0.4529 | 38.00% | 0.9300 |
| disk top5 | 0.4931 | 34.83% | 0.9406 |
| disk top10 | 0.4878 | 36.63% | 0.9819 |
| **disk_center** overall | 0.2339 | 61.14% | 1.2659 |
| disk_center top1 | 0.4779 | 37.02% | 1.1863 |
| disk_center top5 | 0.4397 | 37.86% | 1.2409 |
| disk_center top10 | 0.4069 | 37.96% | 1.2561 |

### 3.2 continue32 (ep32) Val 可用 top-k

| 指标 | 值 |
|---|---|
| earpiece_stress top1 log MAE | 0.2864 |
| earpiece_stress top1 within25 | 56.78% |
| earpiece_stress top1 pred/target | 0.8732 |
| earpiece_stress top5 log MAE | 0.3299 |
| earpiece_stress top5 within25 | 50.29% |
| earpiece_stress top5 pred/target | 0.8870 |

> continue32 的 earpiece_stress top1 within25 (56.78%) 比 base (48.03%) 提升了 8.7 个百分点。

### 3.3 注意：continue32 与 base 的 earpiece_stress 口径差异

continue32 的 `earpiece_stress` 指标实际上等于 `fullpart_stress`（数据完全一致，log MAE=0.2895, within25=53.02%），说明其 fast_eval 模式下没有真正区分 earpiece-only 口径，`earpiece_stress` top-k 实际上也是全零件口径的 top-k。真正的耳片区域指标是 `earpiece_region_stress`（log MAE=0.4962, within25=37.04%），但该区域没有 top-k 拆分。

## 4. Test 指标对比

| 指标 | feature_buckets (ep24) |
|---|---|
| fullpart log MAE | 0.3121 |
| fullpart within25 | 48.05% |
| disk_center log MAE | 0.2856 |
| disk_center within25 | 56.22% |
| earpiece_region log MAE | 0.3479 |
| earpiece_region within25 | 49.54% |

> continue32 的 test 评估尚未运行，无法对比。

## 5. feature_buckets 分区 test top-k 详情

### 5.1 耳片区域 (earpiece_region)


| 分组      | log MAE | within25 | pred/target |
| ------- | ------- | -------- | ----------- |
| overall | 0.3479  | 49.54%   | 0.9874      |
| top1    | 0.2637  | 61.09%   | 0.8849      |
| top1-5  | 0.2087  | 69.99%   | 0.9808      |
| top5    | 0.2196  | 68.24%   | 0.9382      |
| top5-10 | 0.2238  | 66.38%   | 0.9820      |
| top10   | 0.2217  | 67.31%   | 0.9486      |


耳片区域高值点表现不错，top1 within25 达 61%，top5 within25 达 68%。

### 5.2 圆盘中心 (disk_center)


| 分组      | log MAE | within25 | pred/target |
| ------- | ------- | -------- | ----------- |
| overall | 0.2856  | 56.22%   | 0.4430      |
| top1    | 0.8892  | 17.49%   | 0.1763      |
| top1-5  | 0.5490  | 31.48%   | 1.1208      |
| top5    | 0.6163  | 28.71%   | 0.2490      |
| top5-10 | 0.4279  | 40.38%   | 1.4381      |
| top10   | 0.5220  | 34.55%   | 0.2945      |


圆盘中心高值点严重低估：top1 pred/target 只有 0.18，top5 只有 0.25。整体 pred/target=0.44 说明对中心区域系统性低估非常严重。

### 5.3 全圆盘区域 (disk)


| 分组      | log MAE | within25 | pred/target |
| ------- | ------- | -------- | ----------- |
| overall | 0.3071  | 47.84%   | 0.7358      |
| top1    | 0.4322  | 44.05%   | 0.4481      |
| top5    | 0.4126  | 42.88%   | 0.6085      |
| top10   | 0.4101  | 43.33%   | 0.6607      |


圆盘区域整体也明显低估，top1 pred/target=0.45。

## 6. 关键差异分析

### 5.1 low-rank curve head 的影响

continue32 继承了 base 的 rank8 low-rank curve head，而 feature_buckets 使用标准 MLP head（从 fullpart_disk_center 初始化）。这可能是 continue32 在 disk_center 上表现更好的关键原因——low-rank head 对圆盘中心这种具有强低秩特性的区域更有效。

### 5.2 physical feature buckets 的效果

feature_buckets 在 earpiece_region 上的表现（within25 50%）显著优于 base（31%）和 continue32（37%），说明物理特征百分位分桶对耳片区域有帮助。但在 disk_center 上反而更差，可能是标准 MLP head 无法有效利用这些特征。

### 5.3 disk_center 系统性偏差

feature_buckets 的 disk_center pred/target=0.43，意味着对中心区域预测值只有真实值的 43%。这比 base 的 1.27（过预测）和 continue32 的 1.26（过预测）方向完全相反，说明从 fullpart_disk_center 初始化的模型和从 fusion_low_rank 初始化的模型在中心区域的偏置方向不同。

### 5.4 训练效率

- feature_buckets 单进程 + 每 epoch 全量 top-k 评估 + test，导致每 epoch 约 2 小时，25 epoch 跑了 2.5 天
- continue32 的 fast_eval 模式 + DDP 多卡，每 epoch 约 20 分钟，效率高出 6 倍
- feature_buckets 的 data_wait 时间占比极高（train epoch 24: 1h12m data_wait vs 18min compute），说明单进程数据加载是主要瓶颈

## 7. 结论

1. **continue32 在 fullpart 和 disk_center 上全面优于 feature_buckets**，唯一弱点是 earpiece_region
2. **physical feature buckets 对耳片区域有一定帮助**（within25 从 31%→50%），但需要配合 low-rank head 才能同时兼顾圆盘中心
3. **low-rank curve head 是 disk_center 性能的关键**：有 low-rank head 的模型（base/continue32）disk_center log MAE 在 0.22-0.23，没有的（feature_buckets）在 0.30
4. **两个方向各有价值**：建议后续将 feature buckets + low-rank head 结合，同时优化训练效率（启用 DDP + fast_eval）
5. **feature_buckets 实验已停止**（epoch 25，2.5天），继续训练预期收益有限

