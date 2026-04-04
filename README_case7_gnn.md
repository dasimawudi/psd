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

## Default assumption

The first task predicts `freq_top3` instead of the full variable-length
`frequencies` list. This keeps the output dimension fixed and matches the
recommended modal prior in the dataset description.
