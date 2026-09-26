# Experiments

Track experiments by commit, dataset version, split protocol, configuration, checkpoint and test results.

| Experiment | Spatial | Frequency | Temporal | Evaluation | Test AUC | Test F1 |
|---|---|---|---|---|---:|---:|
| Baseline | ✓ |  |  | in-domain |  |  |
| + Frequency | ✓ | ✓ |  | in-domain |  |  |
| + Temporal | ✓ | ✓ | ✓ | in-domain |  |  |
| Robustness | ✓ | ✓ | ✓ | JPEG / blur / resize |  |  |
| Cross-dataset | ✓ | ✓ | ✓ | held-out dataset |  |  |

Populate metric cells only after running the corresponding experiment.
