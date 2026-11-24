# CSE791 Group 09 — ACM Midterm Research Proposal

This directory contains the editable LaTeX manuscript and the compiled midterm proposal for **CSE791 Research Methodology and Seminar, Group 09 (Computer Vision), BRAC University**.

**Study:** Evaluating Targeted Data Augmentation for Robust Retail Product Classification Under Image Degradation.

## Files

| File | Purpose |
| --- | --- |
| `main.tex` | Main ACM `acmart` / `sigconf` LaTeX source |
| `references.bib` | Bibliography database; verify publisher metadata and peer-reviewed status |
| `main.bbl` | Generated bibliography for reproducible compilation |
| `main.pdf` | Compiled midterm proposal |
| `blur_severity_curve.png` | Three-seed validation blur robustness figure used in the proposal |
| `severe_degradation_comparison.png` | Supplementary validation figure |
| `Literature_Matrix_and_Search_Log_Appendix.pdf` | Literature matrix and search-log appendix |
| `results_table.tex` | Standalone table for reuse; avoid including it twice if the table already appears inside `main.tex` |

## Build in Overleaf

1. Upload the contents of this directory to a new Overleaf project.
2. Set `main.tex` as the main/root document and select **pdfLaTeX**.
3. Compile the document with the `references.bib` bibliography. `main.bbl` is supplied as a fallback for limited environments.
4. Check the compiled page count, appendix placement, citations and author details against the course requirements.

## What the paper reports

- **Dataset:** Nyris-hosted Products-10K subset, conservatively cleaned to **9 retail SKU classes and 586 images** (427 train, 105 validation, 54 original test).
- **Model:** ImageNet-pretrained **EfficientNet-B0**.
- **Training protocols:** Basic, Generic, and Targeted augmentation; **3 seeds each (2026, 2027, 2028)**.
- **Evaluation:** Clean validation images and controlled blur, low-light, and occlusion at three severity levels.
- **Preliminary finding:** Targeted augmentation improves *severe-blur* validation accuracy by approximately **12.06 percentage points** over Generic on average; improvements are not universal across degradation types.
- **Important distinction:** The reported scores are from **validation images only**. The reserved original test split has **not** been evaluated, and no statistical-significance claim is made.

The selected products include cosmetics, necklaces, and handbags. This pilot is **not** a packaged-grocery-only study. Exact decoded-pixel duplicates were removed; perceptual near-duplicates and label consistency still require independent review.

Detailed methodology, source scripts, dataset audit and per-run metrics are in the repository's `src/`, `dataset_metadata/`, `docs/`, and `results/` directories.

## Submission review checklist

- [ ] Validate each bibliography record against the publisher or proceedings and confirm the minimum **20 peer-reviewed references**. Do not count a preprint as peer-reviewed solely because it appears in the bibliography.
- [ ] Review the actual contributions and student details for both group members.
- [ ] Follow the course's academic-integrity / AI-use disclosure rules.
- [ ] Confirm the ACM page limit and that the literature matrix/search log are included as instructed.
- [ ] Ensure the manuscript and slides consistently label the results **preliminary validation results**, not final test results.
- [ ] Confirm the repository excludes downloaded dataset images, local virtual environments and large model weights.

## Research artifacts

See the [repository README](../README.md) for the study overview and [reproducibility notes](../docs/REPRODUCIBILITY.md) for recorded commands and limitations.
