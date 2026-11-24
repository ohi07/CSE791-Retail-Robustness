
from collections import Counter
from datasets import load_dataset
from pathlib import Path
import csv

REPO = "nyris/products10k-v1"
OUT = Path("audit")
OUT.mkdir(exist_ok=True)

counts = {}

for split in ["train", "test"]:
    print(f"\nAuditing {split} metadata...", flush=True)

    # Only read SKU identifiers, not image contents.
    ds = load_dataset(
        REPO,
        split=split,
        streaming=True,
        columns=["sku"]
    )

    counter = Counter()
    for row in ds:
        counter[str(row["sku"])] += 1

    counts[split] = counter
    print(f"Images: {sum(counter.values())}")
    print(f"Unique SKUs: {len(counter)}")

all_skus = sorted(
    set(counts["train"]) | set(counts["test"])
)

with (OUT / "sku_counts.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.writer(f)
    writer.writerow(["sku", "train_count", "test_count"])

    for sku in all_skus:
        writer.writerow([
            sku,
            counts["train"][sku],
            counts["test"][sku]
        ])

# Initial eligibility thresholds, not final selection.
eligible = [
    sku for sku in all_skus
    if counts["train"][sku] >= 20
    and counts["test"][sku] >= 5
]

eligible.sort(
    key=lambda sku: (
        -min(counts["train"][sku], 60),
        -counts["test"][sku],
        sku
    )
)

with (OUT / "candidate_skus.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.writer(f)
    writer.writerow(["sku", "train_count", "test_count"])

    for sku in eligible:
        writer.writerow([
            sku,
            counts["train"][sku],
            counts["test"][sku]
        ])

print(f"\nEligible classes: {len(eligible)}")
print("\nTop 20 candidate SKUs:")

for sku in eligible[:20]:
    print(
        f"SKU {sku}: "
        f"train={counts['train'][sku]}, "
        f"test={counts['test'][sku]}"
    )

print("\nAudit finished. Results saved in:", OUT.resolve())
