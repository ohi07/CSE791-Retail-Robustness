r"""Build visual-contact sheets for Products-10K candidate SKU selection.

Usage (Windows CMD from D:/CSE791/Products10K):
    .venv\Scripts\python.exe preview_products10k.py --limit 30 --per-sku 3

Dataset source: https://huggingface.co/datasets/nyris/products10k-v1
Uses only TRAIN images for candidate selection; does not examine test images.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps
from datasets import load_dataset

REPO = "nyris/products10k-v1"


def parse_args():
    p = argparse.ArgumentParser(description="Preview training images from candidate Products-10K SKUs")
    p.add_argument("--shortlist", type=Path, default=Path("shortlist_for_review.csv"))
    p.add_argument("--out", type=Path, default=Path("review_previews"))
    p.add_argument("--limit", type=int, default=30, help="Number of candidate SKUs to preview")
    p.add_argument("--per-sku", type=int, default=3, help="Training images per SKU")
    return p.parse_args()


def load_shortlist(path: Path, limit: int) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    needed = {"sku", "train_count", "test_count"}
    if not rows or not needed.issubset(rows[0]):
        raise ValueError(f"Shortlist needs columns {sorted(needed)}")
    candidates = []
    seen = set()
    for row in rows:
        sku = str(row["sku"]).strip()
        if not sku.isdecimal() or sku in seen:
            continue
        candidates.append(row)
        seen.add(sku)
        if len(candidates) == limit:
            break
    return candidates


def make_contact_sheets(candidates: list[dict], paths: dict[str, list[Path]], folder: Path,
                        per_page: int = 12, per_sku: int = 3) -> list[Path]:
    # A contact sheet is a review-only visual, NOT a training-data split or evaluation.
    tile_w, tile_h = 378, 249
    img_box_w, img_box_h = 112, 177
    cols = 3
    sheets = []
    for page_start in range(0, len(candidates), per_page):
        page_rows = candidates[page_start:page_start + per_page]
        rows = math.ceil(len(page_rows) / cols)
        board = Image.new("RGB", (cols * tile_w, rows * tile_h), "#f8fafc")
        dr = ImageDraw.Draw(board)
        for i, info in enumerate(page_rows):
            ox = (i % cols) * tile_w
            oy = (i // cols) * tile_h
            dr.rectangle((ox + 4, oy + 4, ox + tile_w - 4, oy + tile_h - 4),
                         fill="white", outline="#cbd5e1", width=2)
            sku = str(info["sku"])
            dr.text((ox + 15, oy + 15), f"SKU {sku}   train={info['train_count']}   test={info['test_count']}",
                    fill="#0f172a")
            files = paths.get(sku, [])
            if not files:
                dr.text((ox + 16, oy + 111), "NO IMAGE RETURNED", fill="#a52b2b")
            for j, path in enumerate(files[:per_sku]):
                # restrict preview display to 3 pictures per SKU to avoid overflowing the tile
                if j >= 3:
                    break
                with Image.open(path) as raw:
                    img = ImageOps.contain(raw.convert("RGB"), (img_box_w, img_box_h))
                x = ox + 15 + j * 118 + (img_box_w - img.width) // 2
                y = oy + 43 + (img_box_h - img.height) // 2
                board.paste(img, (x, y))
            dr.text((ox + 15, oy + 229), "Training split only", fill="#475569")
        out_file = folder / f"contact_sheet_{page_start // per_page + 1:02d}.jpg"
        board.save(out_file, quality=90)
        sheets.append(out_file)
    return sheets


def main():
    args = parse_args()
    if args.limit < 1 or not 1 <= args.per_sku <= 3:
        raise SystemExit("Use --limit >= 1 and --per-sku between 1 and 3")
    candidates = load_shortlist(args.shortlist, args.limit)
    if not candidates:
        raise SystemExit("No valid SKUs in shortlist")
    targets = {str(row["sku"]) for row in candidates}
    args.out.mkdir(parents=True, exist_ok=True)
    image_dir = args.out / "train_samples"
    image_dir.mkdir(exist_ok=True)
    collected = defaultdict(list)
    records = []
    print(f"Source: {REPO} | split=train | candidates={len(targets)}", flush=True)
    print("Loading with Parquet SKU predicate filtering (network transfer depends on shard layout).", flush=True)
    try:
        ds = load_dataset(
            REPO,
            split="train",
            streaming=True,
            filters=[("sku", "in", sorted(targets))],
        )
    except (TypeError, ValueError) as e:
        raise SystemExit(
            "The installed datasets library did not accept Parquet predicate filtering. "
            "Run: .venv\\Scripts\\python.exe -m pip install -U datasets pyarrow\n"
            f"Original error: {e}"
        ) from e
    seen = 0
    for sample in ds:
        seen += 1
        sku = str(sample["sku"])
        if sku not in targets or len(collected[sku]) >= args.per_sku:
            continue
        original = str(sample.get("img_file", ""))
        fpath = image_dir / f"sku_{sku}_train_{len(collected[sku]) + 1:02d}.jpg"
        try:
            sample["image"].convert("RGB").save(fpath, quality=90)
        except Exception as e:
            print(f"SKIP damaged/unreadable image {original}: {e}")
            continue
        collected[sku].append(fpath)
        records.append((sku, original, str(fpath)))
        print(f"  SKU {sku}: {len(collected[sku])}/{args.per_sku} train previews", flush=True)
        if all(len(collected[k]) >= args.per_sku for k in targets):
            break
        if seen % 1000 == 0:
            print(f"Processed {seen} filtered rows", flush=True)

    index = args.out / "preview_index.csv"
    with index.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sku", "source_image_path", "local_preview_path"])
        w.writerows(records)
    pages = make_contact_sheets(candidates, collected, args.out, per_sku=args.per_sku)
    print("\nPreview summary:")
    for info in candidates:
        sku = str(info["sku"])
        print(f"SKU {sku}: {len(collected[sku])}/{args.per_sku} images")
    print("\nContact sheet files:")
    for p in pages:
        print(" -", p.resolve())
    if any(len(collected[k]) == 0 for k in targets):
        print("WARNING: Some SKUs had no matches. Do not assume these SKUs are absent; "
              "check filter support / dataset revision and the upload's actual IDs.")
    print("\nNext: inspect sheets, choose 10-15 visually related SKUs and mark shortlist_for_review.csv.")
    print("Do not use TEST images to pick the most promising product classes.")


if __name__ == "__main__":
    main()
