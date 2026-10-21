# PSD 应力仿真 AI 代理模型

基于有限元数据训练的结构随机振动应力响应代理模型。输入结构几何、工况、模态等信息,逐节点预测单频 `MISES_psd_density`,并沿频率积分得到全场 RMises,用于快速筛查应力热点、指导高保真仿真资源分配。

## 当前版本(终版模型)

| 项 | 值 |
|---|---|
| 模型 | `node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32_remove_inner_disk_floor200` |
| 类型 | 节点级 MLP + 低秩频率曲线头 |
| 隐藏层 | `[256, 256, 128]`,SiLU,LayerNorm,dropout=0.1 |
| 低秩曲线头 | rank=8,残差权重=0.15 |
| 训练轮数 | 100 epoch(当前 best = epoch 100) |
| 学习率 / 权重衰减 | `6e-5` / `1e-4` |
| 训练点批量 | 4,000,000 points |
| 输入特征 | 278 维(几何、PSD/频率上下文、耳片/圆盘区域特征、模态与 FRF、区域标识) |
| 低响应阈值处理 | 目标 `MISES_psd_density < 200` 时,训练与评估统一按 `200` 参与 |

模型演进链(每一级在上一个 best.pt 上续训):

```
continue32(基线)
  └─ remove_inner_disk(去除内盘低响应背景点干扰)
       └─ remove_inner_disk_floor200(终版:<200 目标统一按 200 处理)
```

## 模型效果(独立测试集)

测试集按结构方案切分,共约 1.35 亿逐频点、100 个结构方案参与最终 RMises 评估。`within25` = 相对误差 ≤25% 的节点占比。

| 区域 | 维度 | 整体 within25 / 相对MAE | top1% within25 | top5% within25 | top25% within25 |
|---|---|---|---|---|---|
| 全零件 | 逐频 PSD | 85.38% / 17.11% | 78.15% | 74.36% | 71.25% |
| 全零件 | 最终 RMises | 90.10% / 12.39% | 95.67% | 94.75% | 89.47% |
| 耳片区域 | 逐频 PSD | 68.28% / 37.10% | 80.84% | 77.34% | 73.06% |
| 耳片区域 | 最终 RMises | 81.47% / 20.15% | 95.66% | 95.70% | 94.32% |
| 圆盘区域 | 逐频 PSD | 90.61% / 10.99% | 74.08% | 71.84% | 77.36% |
| 圆盘区域 | 最终 RMises | 92.75% / 10.00% | 94.38% | 92.12% | 85.25% |

要点:最终 RMises 维度整体 within25 全零件 90.10%、耳片 81.47%、圆盘 92.75%;耳片最终 RMises 的 top1%~top25% 热点 within25 均超过 94%,对工程高风险节点识别能力较强。

## 完整复现

以下文件均已入库,配合本机数据集即可复现终版模型:

- **代码**:`node/case7_node_mlp/`(训练/评估/特征/模型)
- **配置链**:
  - `node/configs/node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32.yaml`
  - `node/configs/node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32_remove_inner_disk.yaml`
  - `node/configs/node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32_remove_inner_disk_floor200.yaml`
- **checkpoint**:
  - `node/outputs/node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32/best.pt`
  - `node/outputs/node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32_remove_inner_disk/best.pt`
  - `node/outputs/node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32_remove_inner_disk_floor200/best.pt`
  - 终版 epoch 100 快照:`node/outputs/node_mlp_v6_fullpart_fusion_low_rank_80g_fast_eval_continue32_remove_inner_disk_floor200/final_eval_epoch100_20260628/best_epoch100_snapshot.pt`
- **scaler 缓存**(`node/cache/`,已强制入库):
  - `node_mlp_v6_fullpart_fusion_low_rank_80g_scalers.pt`
  - `node_mlp_v6_fullpart_fusion_low_rank_80g_remove_inner_disk_floor200_scalers.pt`

> 数据(`case7/` 等)不在 git 中,需放在本机对应路径;scaler 缓存缺省时也可由 `fit_scalers.py` 从数据重新生成。

## 文档

- 终版技术报告:`node/doc/toBdoc/（终）应力仿真 AI 代理技术报告——20260702.md`
- floor200 模型评估报告:`node/doc/remove_inner_disk_floor200模型报告_20260628.md`
- 建模示意图:`node/doc/modeling_diagrams_20260629/`
- 历史结果汇总:`node/模型训练结果汇总.md`

## 开启新实验

以 main 为基线创建新分支:

```bash
git checkout main
git checkout -b experiment/<新实验名>
```

修改 `node/configs/` 下的 yaml(继承自 floor200 配置链),或新建实验配置;训练入口见 `node/case7_node_mlp/train.py`。修改 `node/` 下训练/评估/特征/配置后,需同步更新 `node/README.md`。
