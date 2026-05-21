# Stress MLP V1

This folder contains an isolated conditional-node-MLP baseline for the
`stress_only_v1` per-frequency stress task.

The trainer reuses the existing `stress_only_v1` data loading, scalers, loss,
checkpointing, and evaluation metrics. Only the model builder is replaced.
The default config drops mesh edges after loading because the MLP does not use
message passing and the smoothness loss is disabled.

Run from the repository root:

```bash
python stress_mlp_v1/train_stress_mlp.py --config stress_mlp_v1/configs/stress_cond_mlp_weighted_modes4_case1000.yaml
```
