# Stress-only V1

This folder is an isolated first-version branch for hotspot-focused stress prediction.

## Scope

- Does not modify the existing `case7_gnn` package.
- Removes the RTA branch from field prediction.
- Uses a single stress target:
  - `MISES_psd_density` for per-frequency samples.
  - `RMises_native` or `RMises` for final-response samples.
- First-version node augmentation only uses high-reliability features from `x,y,z,bc_mask`, `edges.csv`, and explicit `global.json` fields.
- The validation selection metric is hotspot-region error, not full-field average error.

## Model

`FieldGNN` now predicts:

- one output `[stress]` when `stress_two_stage.enabled: false`
- two outputs `[hotspot_logit, stress]` when `stress_two_stage.enabled: true`

The shared graph encoder is still conditioned by global geometry, PSD, modal frequencies, current frequency, and current-frequency-to-mode relations.

## First-version Features

Node features added in `augment_high_reliability_features`:

- `x / plate_radius`
- `y / plate_radius`
- `z / plate_thickness`
- `r / plate_radius`
- `(plate_radius - r) / plate_radius`
- `sin(theta)`, `cos(theta)`
- earpiece width ratios
- distance to nearest earpiece hole edge divided by `earpiece_HoleRadius`
- distance to nearest earpiece hole edge divided by `plate_radius`
- near-ear-hole soft mask
- `r / mass_couple_radius`
- signed center-coupling distance
- near-center-coupling soft mask

No root fillet, local earpiece hard partition, local thickness, free-surface distance, or center-hole-group distance is used in V1.

## Optimization Target

The loss is hotspot-focused:

- hotspot classification for the top response region
- strong hotspot regression
- very low background regression weight
- top-k stress regression
- peak stress consistency
- optional smoothness only outside hotspots

With `stress_two_stage.threshold_peak_ratio`, training hotspots can be defined
by each graph's own peak stress, for example `0.05` means nodes with
`stress >= 5% * peak_stress` are treated as hotspot nodes. When both
`threshold_quantile` and `threshold_peak_ratio` are set, `threshold_combine`
chooses whether the stricter (`max`) or more inclusive (`min`) threshold is used.

The reported `stress_hotspot_within25_ratio` uses the fixed
`stress_hotspot_metric` definition, independent of training-label experiments:
`stress >= min(q0.999, 0.05 * peak_stress)`.

Evaluation also reports the presentation metric:

```text
stress_hotspot_within25_ratio =
  fraction of true-hotspot nodes where abs(pred_stress - true_stress) / true_stress <= 0.25
```

The default checkpoint selection metric is the reverse:

```text
stress_hotspot_miss25_rate = 1 - stress_hotspot_within25_ratio
```

So lower validation `stress_hotspot_miss25_rate` means a larger share of true-hotspot stress predictions fall within the 25% relative-error band.

Run:

```powershell
python stress_only_v1/train_stress_only.py --config stress_only_v1/configs/stress_hotspot_case7new.yaml
```
