# Organize best region model branches and disk-center baseline

## Summary

- Add disk-center node scope support for node MLP datasets.
- Add YAML config inheritance support via `base_config` / `inherits`.
- Add the disk-center baseline config and three disk-center ablation configs.
- Add diagnostics for target-quantile within25 bands and random frequency-curve anomalies.
- Update `node/模型训练结果汇总.md` with the current best models for three regions:
  - earpiece: `node_mlp_v5_exp_region_distance_within25_50ep`
  - disk center: `node_mlp_v6_disk_center_baseline`
  - fullpart / full disk: `node_mlp_v6_fullpart_disk_center`

## Best region model summary

| Region | Best branch | Best model | Checkpoint |
| --- | --- | --- | --- |
| Earpiece | `best/earpiece-region` | `node_mlp_v5_exp_region_distance_within25_50ep` | `node/outputs/node_mlp_v5_exp_region_distance_within25_50ep/best.pt` |
| Disk center | `best/disk-center-region` | `node_mlp_v6_disk_center_baseline` | `node/outputs/node_mlp_v6_disk_center_baseline/best.pt` |
| Fullpart / full disk | `best/fullpart-disk-region` | `node_mlp_v6_fullpart_disk_center` | `node/outputs/node_mlp_v6_fullpart_disk_center/best.pt` |

## Key metrics

### Earpiece best: `node_mlp_v5_exp_region_distance_within25_50ep`

- val log MAE: `0.2830`
- test log MAE: `0.2831`
- test miss25: `0.4095`
- test top1 log: `0.1347`
- test top5 log: `0.1723`

### Disk center best: `node_mlp_v6_disk_center_baseline`

- test log MAE: `0.2807`
- test miss25: `0.4114`
- test top1 log: `0.7272`
- test top5 log: `0.5422`
- test peak relative error: `0.3549`

### Fullpart / full disk best: `node_mlp_v6_fullpart_disk_center`

- fullpart test log MAE: `0.3710`
- fullpart test miss25: `0.5863`
- full disk `disk_stress` test log MAE: `0.3731`
- full disk `disk_stress` test miss25: `0.5964`
- full disk `disk_stress` pred/target ratio: `0.6717`

## Verification

- `python -m py_compile node/analyze_random_frequency_curve_anomalies.py node/case7_node_mlp/target_quantile_within25_diagnostics.py node/case7_node_mlp/data.py node/case7_node_mlp/runtime.py`
- `git diff --check`

## Notes

Training outputs and checkpoints are intentionally not committed. They remain under `node/outputs/...` locally.
