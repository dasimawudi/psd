# train7 V7 Resonance-Background Experiments

目标仍然是耳片区域点级 `MISES_psd_density` 预测。所有 V7 配置都继承 V5 region-distance 的耳片特征和 split，并默认从当前最好模型初始化：

```text
outputs/node_mlp_v5_exp_region_distance_within25_50ep/best.pt
```

## 实验入口

在仓库根目录执行，使用 `ci2n` 环境：

```bash
cd /data-ssd/libo/psd
PYTHONPATH=node conda run -n ci2n python -m train7.train --config node/train7/configs/v7a_ranking_only.yaml
PYTHONPATH=node conda run -n ci2n python -m train7.train --config node/train7/configs/v7b_zero_gate.yaml
PYTHONPATH=node conda run -n ci2n python -m train7.train --config node/train7/configs/v7c_peak_relative.yaml
```

## Hard Negative CSV

先用当前最好 V5 50ep checkpoint 挖掘背景假峰：

```bash
cd /data-ssd/libo/psd
PYTHONPATH=node conda run -n ci2n python -m train7.build_hard_negatives \
  --checkpoint node/outputs/node_mlp_v5_exp_region_distance_within25_50ep/best.pt \
  --split train \
  --output-dir node/outputs/diagnostics/v7_resonance_hard_negatives
```

输出：

```text
outputs/diagnostics/v7_resonance_hard_negatives/train_hard_negatives.csv
```

当前训练里的 ranking loss 是在线按 batch 挖掘 hard false peaks；CSV 主要用于诊断、抽样复核和固定 hard-negative 记录。

## 评估

```bash
cd /data-ssd/libo/psd
PYTHONPATH=node conda run -n ci2n python -m train7.evaluate \
  --checkpoint node/outputs/train7_v7a_ranking_only/best.pt \
  --split test \
  --output-dir node/outputs/train7_v7a_ranking_only/eval_test
```

V7 额外写出这些指标：

```text
earpiece_stress_false_peak_top5_rate
earpiece_stress_false_peak_top10_rate
earpiece_stress_hotspot_false_peak_margin_log
earpiece_stress_near_zero_over_cap_rate
earpiece_stress_near_zero_over_log_mae
```

## 实验说明

- `v7a_ranking_only.yaml`：不改模型结构，只加 hotspot-vs-false-peak pairwise ranking loss。
- `v7b_zero_gate.yaml`：增加 near-zero gate 头，用于压低背景近零波谷的非零底噪。
- `v7c_peak_relative.yaml`：增加 peak-relative/drop 头，解耦 case-frequency 全局峰值和节点参与度。

所有输出目录都在 `outputs/train7_*` 下。
