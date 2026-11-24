# Research Progress — Group 09

**Recorded date:** 2026-11-24

| Milestone | Evidence | Status |
| --- | --- | --- |
| Research questions / preliminary literature review | CSE791 working proposal | Drafted |
| Dataset access | Hugging Face streaming from PC | Verified |
| SKU count audit | 141,931 train + 55,376 test rows; 9,691 class IDs | Verified |
| Visual preview | 30 SKU contact-sheet review | Verified |
| Initial extraction | 11 classes, 734 images | Verified |
| Exact-duplicate diagnosis | 10 duplicate groups; ambiguous 8227/8631 | Verified |
| Conservative cleaned dataset | 9 classes; 427/105/54 train/val/test | Verified |
| EfficientNet-B0 training | 3 policies × 3 seeds | Completed |
| Corruption severity evaluation | 105 validation images; blur/low light/occlusion | Completed |
| Official test evaluation | No test metrics in 9 run configs | Not done |
| Independent near-duplicate and visual label audit | Risk recorded | Not done |
| Final journal-style manuscript and research gap verification | Pending scholarly review | Not done |

### Observations

Across three training seeds, targeted augmentation has the largest mean advantage over generic augmentation under severe blur (+12.063 percentage points), but does not show superiority on severe occlusion (−0.317 pp) and incurs a slight decrease in clean validation accuracy (−0.952 pp). Small sample size and a fixed partition limit broad claims.

**Course:** CSE791 · **Group:** 09 · **Scope:** retail SKU classification pilot (not packaged-grocery-only).
