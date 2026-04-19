# Case7 GNN Starter

This project provides a minimal graph neural network pipeline for the `case7`
dataset in this workspace.

Implemented tasks:

1. Frequency regression
   Input:
   - node features from `nodes.csv`
   - edge features from `edges.csv`
   - shape features from `global.json -> params_list`
   Output:
   - `freq_top3`

2. Node field regression
   Input:
   - node features from `nodes.csv`
   - edge features from `edges.csv`
   - shape features from `params_list`
   - PSD working condition from `psd_points`
   Output:
   - per-node `RTA`
   - per-node `RMises`
   - The field model now uses a dual-head design with an extra RMises-specific
     refinement branch to better capture localized stress hotspots.
   - The encoder now also supports an optional case-conditioning path that
     compresses global geometry/load priors into a latent vector and uses it to
     modulate message passing and decoder features.
   - RMises can also run in an optional two-stage mode: classify hotspot nodes
     first, then regress only the hotspot intensity above a configured stress
     threshold.

## Important notes

- Only `case0` to `case6` are complete in the current dataset. `case7` is
  incomplete and is skipped automatically.
- The training pipeline now supports large datasets by loading cases on demand
  instead of keeping every graph in memory at once.
- Dataset split can be configured either with explicit case lists or with
  automatic ratio-based splitting. The default configs use ratio splitting.
- The current `psd_points` are identical in all valid cases. The code supports
  PSD as an input feature, but the current dataset cannot teach the model how
  different PSD conditions affect the response. To truly learn PSD influence,
  you need cases with varying PSD settings.
- `edges.csv` stores each mesh edge once. The loader converts the graph to an
  undirected graph by adding reverse edges. Reverse edge attributes use
  `[-dx, -dy, -dz, dist]`.
- `RMises` contains some small negative values in the raw data. By default, the
  field task clamps negative `RMises` to zero before applying the label
  transform.
- The field task can apply an RMises hotspot-weighted loss so high-stress
  regions contribute more strongly during training.
- The field task also supports optional physics-aware node feature augmentation
  and an edge-smoothness prior in the loss. A ready-to-run example is provided
  in `configs/field_physics.yaml`.

## Environment

Python:

- Python 3.11 is recommended.

Install PyTorch first according to your CPU/CUDA environment, then install the
rest:

```powershell
pip install -r requirements.txt
```

Recommended PyTorch versions:

- PyTorch 2.2 or newer

## Project layout

```text
train_case7.py
predict_case7.py
configs/
  frequency.yaml
  field.yaml
case7_gnn/
  data.py
  runtime.py
  scalers.py
  models.py
  trainer.py
  train.py
  predict.py
```

## Run

Frequency task:

```powershell
python train_case7.py --config configs/frequency.yaml
```

Field task:

```powershell
python train_case7.py --config configs/field.yaml
```

Field task with physics-aware features and loss:

```powershell
python train_case7.py --config configs/field_physics.yaml
```

For large datasets, the default configs automatically discover complete cases
under `dataset.root` and split them with:

- `train_ratio`
- `val_ratio`
- `test_ratio`
- `split_seed`

Optional dataset knobs:

- `include_cases`: restrict to a known subset before splitting
- `exclude_cases`: drop known bad cases before splitting
- `max_cases`: cap the discovered dataset size for quick experiments
- `scaler_fit_case_limit`: fit feature scalers on only the first N training
  cases when a full scaler pass is too slow
- `cache_dir`: store preprocessed per-case `.pt` files to avoid rereading large
  CSV files every epoch

Optional model knobs:

- `model.conditioning.enabled`: turn on case-level latent conditioning
- `model.conditioning.case_dim`: latent width used to encode global priors

Optional physics-aware feature knobs:

- `features.augment_node_physics`: append derived geometry priors to node inputs
- `features.boundary_band_ratio`: normalized boundary band used for the
  near-boundary indicator
- `features.earpiece_band_ratio`: normalized earpiece band used for the
  near-earpiece indicator

Optional physics-aware field loss knobs:

- `field_loss.physics_rta_smoothness_weight`: edge-based RTA smoothness prior
- `field_loss.physics_rmises_smoothness_weight`: edge-based RMises smoothness prior
- `field_loss.physics_distance_power`: distance normalization power for the
  smoothness term
- `field_loss.physics_exclude_boundary_edges`: skip edges touching constrained
  nodes when applying the smoothness prior
- `field_loss.physics_hotspot_exempt_quantile`: exempt the top-stress nodes
  from smoothness regularization to avoid flattening true hotspots

Optional RMises two-stage knobs:

- `rmises_two_stage.enabled`: enable hotspot classification + hotspot-strength regression
- `rmises_two_stage.threshold`: hotspot threshold in raw RMises units
- `rmises_two_stage.prob_threshold`: probability threshold used at inference time
- `rmises_two_stage.classification_weight`: classification loss weight
- `rmises_two_stage.regression_weight`: hotspot regression loss weight
- `rmises_two_stage.positive_class_weight`: positive-class reweighting for hotspot BCE

Inference after training:

```powershell
python predict_case7.py --checkpoint outputs/frequency/best.pt --case-dir case7/case6 --output-dir outputs/predict_frequency_case6
python predict_case7.py --checkpoint outputs/field/best.pt --case-dir case7/case6 --output-dir outputs/predict_field_case6
```

## Outputs

Each run writes to the configured `save_dir`:

- `best.pt`: best checkpoint
- `metrics.json`: best validation/test metrics
- `resolved_config.yaml`: config snapshot
- `history.csv`: epoch-level training and validation history
- `train.log`: console log persisted to file

Inference writes:

- `frequency_prediction.json` for the frequency task
- `field_prediction.csv` and `field_prediction_summary.json` for the field task
- in two-stage RMises mode, `field_prediction.csv` also includes
  `pred_hotspot_prob` and `pred_hotspot_label`

## Default assumption

The first task predicts `freq_top3` instead of the full variable-length
`frequencies` list. This keeps the output dimension fixed and matches the
recommended modal prior in the dataset description.
