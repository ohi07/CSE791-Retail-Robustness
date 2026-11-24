# Group 09 — Updated ACM Midterm Manuscript (24 November 2026)

## Contents

- `main.tex`: Editable 2-column ACM `acmart`/`sigconf` manuscript.
- `references.bib`: 22 cited bibliography records (21 conference/journal items and one marked arXiv preprint).
- `main.bbl`: Generated bibliography file, included for compiling where BibTeX is unavailable.
- `main.pdf`: Compiled nine-page PDF (main text begins on page 1; references begin on page 6; appendix occupies pages 7–9).
- `blur_severity_curve.png`: Validation plot from the nine recorded runs.
- `severe_degradation_comparison.png`: Additional results figure (optional; the manuscript currently uses blur curve).
- `Literature_Matrix_and_Search_Log_Appendix.pdf`: Three-page literature matrix/search log included in `main.pdf`.

## Original dataset / model provenance

The updated manuscript is based on the Group 09 cleaned data audit and recorded nine training runs (3 augmentation policies × 3 seeds) uploaded on 24 November 2026. Measured scores refer **only to 105 validation images**, not the 54 reserved original test images. The pilot covers nine mixed-category SKU classes, not packaged groceries alone. Research limitations and the occlusion-seed issue are discussed in the manuscript.

## Overleaf

1. Create an Overleaf project, upload all files in this folder (or use the Overleaf ZIP).
2. Set `main.tex` as the root file; choose pdfLaTeX.
3. Recompile, with the bibliography `references.bib`. `main.bbl` is included as a fallback in environments without a BibTeX executable.
4. Verify the literature appendix's sourcing and your institution's specific author/AI-disclosure requirements before submission.

## Copy to GitHub

This folder is designed to be merged into your repository's `paper/` subdirectory:

```
CSE791-Retail-Robustness/
  paper/
    main.tex
    references.bib
    blur_severity_curve.png
    Literature_Matrix_and_Search_Log_Appendix.pdf
    main.bbl
    main.pdf
```

Your existing `paper/results_table.tex` can remain; `main.tex` already contains the principal pilot results table, so do not `\\input` the separate table as well without removing the embedded one.

## Required final checks

- At least 20 peer-reviewed entries must remain in the *cited* reference list; the Products-10K arXiv preprint is explicitly identified as a preprint and is **not counted** toward that minimum.
- Publisher pages and page numbers for newer articles should be independently checked. Some records in the supplied bibliography are missing proceedings page ranges or publisher fields; BibTeX may warn, although the compiled PDF is readable.
- Confirm the authors' actual contributions and any classroom academic-integrity/AI disclosure wording.
- Do not claim final held-out test accuracy or statistical significance based on three training seeds.
- Do not put dataset images, large model checkpoints or `.venv` in the GitHub repository.
