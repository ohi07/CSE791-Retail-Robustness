# CSE791 — Retail SKU Classification Robustness

**Group 09 · Computer Vision · BRAC University**

**Research topic:** Evaluating Targeted Data Augmentation for Robust Retail Product Classification Under Image Degradation.

This repository documents an MSc CSE791 empirical pilot comparing three training-time augmentation policies for fine-grained retail SKU recognition. The fixed classifier is **ImageNet-pretrained EfficientNet-B0**. Current results are **validation-only and preliminary**, not final held-out test findings.

## Research questions

1. How does classification accuracy vary with blur, low illumination, and partial occlusion at increasing severities?
2. Does degradation-specific augmentation improve robustness over minimal and conventional augmentation?
3. Is there a trade-off in clean-image performance?

## Verified dataset and evaluation

- Source: [Nyris-hosted Products-10K variant](https://huggingface.co/datasets/nyris/products10k-v1). This is a community-hosted variant, not a guarantee of exact parity with the original competition release.
- Initial extraction: **11 selected SKU labels, 734 images** (535 train, 133 validation, 66 original test).
- Exact decoded-pixel audit detected **10 duplicate groups**, including cross-class ambiguities; classes **8227** and **8631** were excluded conservatively.
- Cleaned pilot: **9 SKUs, 586 images** — **427 train / 105 validation / 54 test**.
- Verified zero remaining exact-pixel duplicate groups and no original-test split contamination. Near-duplicates and identity consistency have **not** been comprehensively audited.
- Nine SKU IDs: `2885, 2929, 2937, 7378, 7488, 7592, 7964, 8424, 8627`.
- Selected samples span **cosmetics, necklaces, and handbags**. They are **not a packaged-grocery-only benchmark**.
- Official test data (6 images per class) remains unused in the reported nine runs.

## Experimental protocol

- Backbone: ImageNet-pretrained EfficientNet-B0.
- Policies: `basic`, `generic`, `targeted`.
- Runs: three seeds (`2026`, `2027`, `2028`) for each policy; **8 epochs** per run; batch size **16**; learning rate **0.0002**.
- GPU: NVIDIA RTX 4090, PyTorch CUDA runtime 12.8.
- Model selection uses **validation macro-F1**. Corruption testing is on the same 105 held-out validation images.
- Corruption types: Gaussian blur, low brightness, rectangular occlusion; three severity levels each.
- Reported means and **sample standard deviations** use the three independently trained seeds; these are *not* 95% confidence intervals.

## Preliminary validation findings

Accuracy (%) averaged across three seeds:

| Condition | Basic | Generic | Targeted |
| --- | ---: | ---: | ---: |
| Clean | 96.19 ± 0.95 | **96.83 ± 0.55** | 95.87 ± 0.55 |
| Severe blur | 65.08 ± 4.50 | 71.75 ± 4.89 | **83.81 ± 3.30** |
| Severe low light | 89.52 ± 0.95 | 93.02 ± 1.10 | **95.56 ± 1.98** |
| Severe occlusion | 79.37 ± 2.75 | **83.81 ± 2.52** | 83.49 ± 4.29 |

Targeted augmentation improves severe-blur validation accuracy by **12.06 percentage points** over Generic on average; it does *not* outperform Generic on severe occlusion, and its clean validation accuracy is **0.95 percentage points lower**. This is an exploratory small-sample study; do not generalize beyond these data without additional controls.

Raw summary: [results/summary_all_severities.csv](results/summary_all_severities.csv).

## Project status (24 November 2026)

- [x] Literature review and question definition (draft)
- [x] Dataset access, metadata audit and visual SKU screening
- [x] Split-preserving exact-duplicate repair and audit
- [x] CUDA-ready EfficientNet-B0 pipeline
- [x] 3 policies × 3 training seeds; validation corruption analysis
- [ ] Independent label and near-duplicate verification
- [ ] Confirm degradation severities against real mobile-camera conditions
- [ ] Freeze experiment protocol and perform single final test evaluation
- [ ] Expand to a coherent, larger SKU sample and/or real-world smartphone images
- [ ] Final ACM manuscript and release review

## Reproducing locally

The actual Python scripts and individual run records reside in the group's local project and should be committed under `src/` and `results/<policy>/seed_<seed>/` without altering their contents. Do not substitute a rewritten training implementation for the one used to generate these measurements.

Dataset images, `.venv`, model checkpoints, raw PDFs, and temporary caches **must not be committed**. See [the ignore rules](.gitignore) and [reproducibility notes](docs/REPRODUCIBILITY.md).

This research dataset is supplied by a third party; check its original non-commercial/research terms before reuse or redistribution. **No dataset images are distributed here.**

## Scope and limitations

This is a **fixed small-subset retail SKU pilot**, not an FMCG or full Products-10K benchmark. Test-set performance and robust statistical superiority have not been established. The three random-seed training outcomes share one fixed validation split; there is substantial uncertainty due to sample size and class selection.

## Attribution

CSE791 Research Methodology and Seminar, **Group 09 — Computer Vision**. Research records were reviewed and compiled for coursework; human verification and final scholarly editing remain necessary.
