# Benchmark checklist

Before publishing metrics:

1. Split by source video / identity, not random frames.
2. Train on declared datasets.
3. Tune the operating threshold only on validation data.
4. Keep the final test set untouched.
5. Report accuracy, ROC-AUC, precision, recall and F1.
6. Add cross-dataset evaluation.
7. Report per-manipulation performance when labels are available.
8. Run compression variants.
9. Save the exact commit, configuration, dataset version and checkpoint hash.
10. Never present results from synthetic or convenience datasets as benchmark accuracy.
