# Reproducibility, provenance and release checklist

## Provenance

Dataset: `nyris/products10k-v1` (Hugging Face community-hosted mirror, accessed 2026-11-24). Original test retained. The data's own license and upstream usage conditions should be consulted; **do not commit product images**.

### Source audit

- 11 provisional classes, 734 records: train=535, val=133, test=66.
- Exact image duplicate audit detected 10 duplicate groups including conflicting classes `8227` and `8631`.
- Conservative repair excluded both ambiguous classes and two remaining duplicates, retaining 586 records.
- Cleaned partition: 427 train / 105 val / 54 original test from **9 classes**.
- Exact saved-JPEG decoded-pixel hash duplicate groups: 0.
- `BLOCKS_TRAINING: false`, `official_test_isolation_violated: false`.
- Risks not resolved: perceptually near-duplicate images, capture-session dependence, SKU-label identity errors, selected category representativeness, low test size.

### Model setup and experiment trace

The original run scripts on the group's computer must be copied **unchanged** to `src/`. No code was recreated from the logged numerical results. For the documented experiments:

```cmd
.venv\Scripts\python.exe check_gpu.py
.venv\Scripts\python.exe verify_pilot.py

.venv\Scripts\python.exe train_pilot.py --data local_pilot_clean --policy basic --epochs 8 --batch-size 16 --seed 2026 --out ..\results\group09
.venv\Scripts\python.exe train_pilot.py --data local_pilot_clean --policy generic --epochs 8 --batch-size 16 --seed 2026 --out ..\results\group09
.venv\Scripts\python.exe train_pilot.py --data local_pilot_clean --policy targeted --epochs 8 --batch-size 16 --seed 2026 --out ..\results\group09
```

Repeat each command with `--seed 2027` and `--seed 2028`. Check that all nine configs share an identical cleaned manifest, fixed model settings, and source distribution. If the source code differs from the above protocol, the actual source code and run configs take precedence.

- Architecture: EfficientNet-B0 (pretrained ImageNet weights); cross-entropy classification.
- Model selection: validation macro-F1. Final reported validation uses selected checkpoint and the same 105 val images.
- Conditions: clean, Gaussian blur, darkening, rectangular occlusion. The exact parameters are implemented by `train_pilot.py`.
- Same validation samples are reused across degradation conditions; do not interpret the reported n=105 per condition as independent samples.
- Numerical summaries are mean/sample standard deviation across **three training runs**, not confidence intervals.
- The official 54 test images were sealed in all nine run configuration records.

## Suggested repository content

```text
src/check_gpu.py
src/train_pilot.py
src/verify_pilot.py
src/plot_validation.py
src/extract_selected_skus.py
src/fix_pilot_duplicates.py
dataset_metadata/selected_skus_pilot.csv
dataset_metadata/manifest.csv                # ONLY sanitized, portable relative paths
dataset_metadata/audit.json
dataset_metadata/CLEANING_DECISIONS.csv
results/<basic|generic|targeted>/seed_<2026|2027|2028>/
    config.json
    history.json
    RUN_LOG.md
    val_metrics.json
    val_predictions.csv
results/summary_all_severities.csv
figures/severe_degradation_comparison.png
figures/blur_severity_curve.png
paper/main.tex
paper/references.bib
```

### Pre-publish checks

1. Check `manifest.csv`, log files and configs for personal directory paths and private tokens. Sanitize directory paths before pushing.
2. Do not publish original images, weights, caches, absolute filesystem layout, or downloaded papers without redistribution rights.
3. Ensure model source matches original validation metrics rather than regenerated or revised code.
4. Keep `best.pt` locally; optionally release via a separately licensed, documented model release later.
5. Verify plots and tables come from recorded metrics, and label results **validation-only**.
6. Before the final paper, complete manual label inspection and near-duplicate checks, freeze protocol, and evaluate the official test set once.
