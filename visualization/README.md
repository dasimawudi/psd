# Visualization Utilities

This folder contains visualization and dataset-distribution analysis helpers.
Run commands from the project root so relative dataset/config paths resolve
correctly.

Common commands:

```powershell
python visualization/visualize_nodes.py --input case7/case0/nodes.csv --value-column RMises --output-dir outputs/visualizations/node_viz_case0_rmises
python visualization/analyze_rmises_distribution.py --config configs/field.yaml
python visualization/analyze_node_target_distribution.py --config configs/field.yaml
python visualization/analyze_node_target_distribution_exact.py --config configs/field.yaml
python visualization/analyze_case7new_single_frequency_visuals.py
```

Generated visualization artifacts should go under `outputs/visualizations/`.
