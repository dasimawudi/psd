# Remove Inner Disk 模型评估报告

> 更新时间：2026-06-26
> 模型：`node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32_remove_inner_disk` (Best Epoch 74)
> 状态：训练中（100 epoch 计划，early stopping patience=10），已完成 test 集全量 top 百分比评估

## 训练目的

本实验基于 continue32 基线模型（Best Epoch 32）的 `best.pt` 继续训练，旨在验证一个假设：**磁盘 inner hole 到 plate hole 外边缘之间的节点区域可能是噪声源，排除后能提升模型对高应力峰值点的拟合能力。**

具体动机来自 continue32 基线的两个观察：

1. **圆盘区域高响应点拟合困难**：continue32 的圆盘 top1% within25 仅 43.92%，甚至低于 top25%（51.74%），呈现"峰值不如中值"的异常梯度。怀疑 inner disk 区域的复杂几何边界引入了难以学习的应力分布模式，干扰了对真正高应力点的建模。

2. **disk_center 区域指标虽好但覆盖有限**：continue32 的 disk_center within25 达 64.70%，但该子区域仅占全零件面积的约 15%。排除该区域后，模型可将容量集中到其余 85% 节点上。

实验设计：从 continue32 的 best.pt 初始化，将 `exclude_inner_disk_to_plate_hole_outer_edge` 从 `false` 改为 `true`，对应停用 disk_center 相关 loss 权重，继续训练 100 epoch 观察指标变化。

### 与 continue32 基线的主要差异

| 配置项 | continue32 (基线) | remove_inner_disk (本模型) |
|---|---|---|
| `exclude_inner_disk_to_plate_hole_outer_edge` | `false` | `true` |
| 含义 | 保留磁盘 inner hole 到 plate hole 外边缘之间的节点 | 排除该区域节点 |

### 主要训练配置

| 配置项 | 值 |
|---|---|
| MLP hidden dims | `[256, 256, 128]` |
| low_rank_curve_head | `true`, rank=8, residual_weight=0.15 |
| activation | `silu` |
| dropout | `0.1` |
| layer norm | `true` |
| epochs | `100` |
| early stopping patience | `10` |
| batch_size | `4,000,000` points |
| sample_batch_size | `384` (训练) / `16` (评估) |
| lr | `6e-5` |
| weight_decay | `1e-4` |

### 特征工程

输入共 **278 维**，按特征组划分：

| 特征组 | 说明 |
|---|---|
| 基础几何 | 归一化坐标 (`x/y/z_norm`)、极坐标 (`r_norm`, `sin_theta`, `cos_theta`)、`dist_to_edge` |
| PSD 值 | `psd_value_at_frequency` + `log_psd_value_at_frequency`（当前频率点的响应值） |
| 耳片局部特征 | 耳片局部坐标 (`u/v/r`)、距耳孔/耳轴/耳根距离、耳片连接区域标识等 |
| 圆盘中心特征 | 中心区域径向分层 (3 环: inner/mid/outer + soft transition)、8 个 RBF 核、模态区域特征 |
| 板孔特征 | 距板孔中心/边缘距离、板孔壁邻近度、板孔角度周期特征 |
| 角度周期特征 | 耳片/板孔周期角度 (`sin/cos`)，捕获旋转对称性 |
| 应力区域距离 | 距各应力集中区域（耳孔、板孔、耳连接区等）的距离（归一化到 plate radius） |
| 节点区域 mask | `is_earpiece_region`, `is_disk_region` 等区域指示 |
| 振型特征 | 前 10 阶模态振型 (`U1/U2/U3/U_mag`) + 归一化变体 + 模态比 + FRF + 梯度 + baseline，含共振加权 (`resonance`, damping=0.02) |

### 损失函数

在 `log1p` 空间计算 MSE 为基础，叠加多层加权策略：

| 损失组件 | 配置 | 作用 |
|---|---|---|
| 逐点分位加权 | `weighting: target_quantile` | 按目标值在全样本中的分位自适应加权 |
| 高响应点加权 | `top1_weight=12.0`, `top5_weight=4.0` | 强化对峰值点的拟合 |
| 目标值分桶加权 | `[0,1)×1.35`, `[1,10)×1.2`, `[10,100)×1.1` | 低目标值区域适度提权 |
| 低目标过预测惩罚 | `weight=0.05`, target∈[0,100] | 抑制对低应力区域的高估 |
| 样本级峰值 loss | `peak=0.22`, `top5=0.18`, `top1=0.3` | 每帧内 top1%/top5%/峰值点额外监督 |
| 背景误峰抑制 | `weight=0.006`, margin=log(1.5) | 防止将低应力误判为峰值 |
| 中心区域点加权 | `center_region_point_weight=1.0` | 本模型已排除 inner disk，降至 1.0 |
| disk_center 峰值 loss | `top5=0.0`, `top1=0.0` | 本模型已排除 disk_center，置零 |

### 训练时 eval 配置

- `eval_topk_mode: selection`（训练中快速评估）
- `eval_every: 2`
- `test_on_best: false`
- `final_test_after_training: true`（训练结束后自动跑 full topk test）

## 评估口径

`top1% / top2% / top5% / top10% / top15% / top25%` 按百分比占比定义。

具体定义：对每个 case-frequency 帧，在全部节点（或区域内部节点）内，按真实 `MISES_psd_density` 从大到小排序，取 `rank / node_count <= 1% / 2% / 5% / 10% / 15% / 25%` 的累计高响应点，再汇总所有帧计算指标。

主要指标：
- `within25`：相对误差 `abs(pred - target) / abs(target) <= 25%` 的点占比。
- `相对 MAE`：逐点相对误差均值 `mean(abs(pred - target) / target)`（仅正目标）。
- `相对 log MAE`：`log1p(MISES_psd_density)` 空间里的 MAE，即 `mean(abs(log1p(pred) - log1p(target)))`。

## 数据集规模

数据按 case 维度 `train_ratio=0.8` 拆分，约 1000 个 case，随机划分为 train/val/test。每帧为非空的 case-frequency 样本。

| 划分 | 样本数 | 总点数 | 耳片区域点数 | 圆盘区域点数 |
|---:|---:|---:|---:|---:|
| 训练集 | 82,538 | 1,057,104,666 | ~2.4 亿 (估) | ~8.1 亿 (估) |
| 验证集 | 10,113 | 129,825,689 | 29,184,331 | 100,641,358 |
| 测试集 | 10,607 | 135,184,771 | 31,666,289 | 103,518,482 |

> 注：训练集不按区域单独统计，耳片/圆盘点数为按 val+test 区域占比（耳片 22.96%、圆盘 77.04%）估算。圆盘区域不含 disk_center（`exclude_inner_disk_to_plate_hole_outer_edge=true` 已排除）。

## Overall 结果

| 范围 | within25 | 相对 MAE | 相对 log MAE |
|---|---:|---:|---:|
| 全零件 (overall) | 54.19% | 81.66% | 0.2749 |
| 耳片区域 | 43.26% | 124.87% | 0.4121 |
| 圆盘区域 | 57.54% | 68.39% | 0.2329 |

全零件 overall 的 within25 达到 54.19%，相对 log MAE 为 0.2749。按区域拆分后，圆盘区域整体表现最好（within25 57.54%，相对 log MAE 0.2329），耳片区域整体较弱（within25 43.26%，相对 log MAE 0.4121），这与之前平坦目标区间的结论一致——耳片区域的极高响应点虽然拟合好，但中等响应点误差大，拉低了区域整体指标。

## 全零件 Top 百分比结果

| Top 百分比 | 点数 | within25 | 相对 MAE | 相对 log MAE |
|---|---:|---:|---:|---:|
| top1% | 1,346,987 | **70.80%** | 20.79% | 0.2046 |
| top2% | 2,698,709 | 66.51% | 23.65% | 0.2241 |
| top5% | 6,754,216 | 61.38% | 27.16% | 0.2523 |
| top10% | 13,513,764 | 56.00% | 30.84% | 0.2861 |
| top15% | 20,272,668 | 52.50% | 33.94% | 0.3117 |
| top25% | 33,792,248 | 48.13% | 40.10% | 0.3476 |

全零件从 top1% 到 top25% 的指标呈单调递减：top1% 的 within25 达到 70.80%，top5% 为 61.38%，到 top25% 降为 48.13%。高响应点的拟合质量明显优于中低响应点。

## 耳片区域 Top 百分比结果

| Top 百分比 | 点数 | within25 | 相对 MAE | 相对 log MAE |
|---|---:|---:|---:|---:|
| top1% | 311,562 | **73.14%** | 18.77% | 0.1999 |
| top2% | 628,170 | 72.68% | 19.33% | 0.1986 |
| top5% | 1,578,187 | 70.01% | 21.41% | 0.2065 |
| top10% | 3,161,871 | 65.76% | 24.39% | 0.2264 |
| top15% | 4,744,788 | 63.38% | 26.23% | 0.2393 |
| top25% | 7,912,555 | 59.73% | 28.96% | 0.2605 |

耳片区域的高响应 top 百分比表现是三个范围内最好的：top1% within25 达到 73.14%，top5% 仍有 70.01%。但从 top5% 扩展到 top25% 后，within25 从 70.01% 降到 59.73%，下降了 10 个百分点，说明耳片区域内峰值点拟合好但中高响应点仍有改善空间。

## 圆盘区域 Top 百分比结果

| Top 百分比 | 点数 | within25 | 相对 MAE | 相对 log MAE |
|---|---:|---:|---:|---:|
| top1% | 1,029,860 | 53.33% | 27.50% | 0.3205 |
| top2% | 2,065,209 | 50.52% | 29.40% | 0.3361 |
| top5% | 5,170,909 | 47.35% | 32.24% | 0.3533 |
| top10% | 10,347,149 | 45.43% | 35.48% | 0.3638 |
| top15% | 15,522,725 | 44.24% | 38.64% | 0.3704 |
| top25% | 25,875,640 | 43.86% | 44.42% | 0.3715 |

圆盘区域的 top 百分比梯度非常平缓：top1% within25 = 53.33%，top25% within25 = 43.86%，仅下降不到 10 个百分点。说明圆盘区域内即使是高响应点，拟合难度也较大。不过圆盘区域有 103M 点，整体稳定性好（overall within25 57.54% 高于全零件 overall 54.19%）。

## 三区域综合对比

| 区域 | overall within25 | top1% within25 | top5% within25 | top10% within25 | top25% within25 |
|---|---:|---:|---:|---:|---:|
| 全零件 | 54.19% | 70.80% | 61.38% | 56.00% | 48.13% |
| 耳片区域 | 43.26% | **73.14%** | **70.01%** | **65.76%** | **59.73%** |
| 圆盘区域 | **57.54%** | 53.33% | 47.35% | 45.43% | 43.86% |

| 区域 | overall 相对 log MAE | top1% 相对 log MAE | top5% 相对 log MAE | top25% 相对 log MAE |
|---|---:|---:|---:|---:|
| 全零件 | 0.2749 | 0.2046 | 0.2523 | 0.3476 |
| 耳片区域 | 0.4121 | **0.1999** | **0.2065** | **0.2605** |
| 圆盘区域 | **0.2329** | 0.3205 | 0.3533 | 0.3715 |

## 总结

1. **耳片区域的高响应点拟合最好**：top1% within25 = 73.14%，top5% within25 = 70.01%，是全模型最强的指标。但耳片区域 overall 只有 43.26%，说明耳片区域大量中低响应点误差大，拉低了区域整体水平。

2. **圆盘区域拟合最稳定但高响应点仍是短板**：从 top1% 到 top25% 的 within25 变化平缓（53.33% → 43.86%），但即使是 top1% 也只有 53.33%。与之前的 continue32 基线相比（需另外评估），`remove_inner_disk` 配置排除了 disk inner hole 到 plate hole 外边缘之间的节点，可能影响圆盘区域的点分布。

3. **全零件 overall within25 54.19%、top1% 70.80%、top5% 61.38%**，相比平坦区间的全量平坦区模型（63.30% / 86.34% / 87.15%），差距主要来源于本模型覆盖了所有目标值范围（不仅仅是 `[1e6, 1e7)` 的平坦区），因此整体相对误差更高。

4. 下一步建议：
   - 与 continue32 基线做量化对比，确认 `remove_inner_disk` 的有效性。
   - 耳片区域：top1%-top5% 已经较好，可考虑针对性提升 top10%-top25% 中高响应区域。
   - 圆盘区域：高响应点 within25 仍不足 55%，需要分析是否因排除 inner disk 区域导致特征缺失。

## 附录：模型信息

- **模型路径**: `node/outputs/node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32_remove_inner_disk/best.pt`
- **配置文件**: `node/configs/node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32_remove_inner_disk.yaml`
- **init checkpoint**: `node/outputs/node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32/best.pt`
- **评估脚本**: `node/eval_top_percent_metrics.py`
- **评估命令**:
  ```
  python node/eval_top_percent_metrics.py \
    --checkpoint node/outputs/node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32_remove_inner_disk/best.pt \
    --output-dir node/outputs/node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32_remove_inner_disk/top_percent_eval \
    --sample-batch-size 16 --num-workers 4
  ```
- **评估结果文件**:
  - `node/outputs/node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32_remove_inner_disk/top_percent_eval/test_top_percent_relative_metrics.json`
  - `node/outputs/node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32_remove_inner_disk/top_percent_eval/test_top_percent_relative_metrics.csv`
