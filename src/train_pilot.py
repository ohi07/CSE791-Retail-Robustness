#!/usr/bin/env python3
"""CSE791 Group 09: auditable EfficientNet-B0 SKU classification pilot.

The original test split is NEVER loaded unless --final-test is explicitly supplied.
Dataset is expected in the exact local_pilot_clean layout created by the repair tool.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageFilter
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

LEVELS = {
    "blur": {1: 1.0, 2: 2.3, 3: 4.0},
    "low_light": {1: 0.70, 2: 0.45, 3: 0.25},
    "occlusion": {1: 0.08, 2: 0.16, 3: 0.25},
}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def corrupt(img: Image.Image, kind: str, level: int, seed: int) -> Image.Image:
    img = img.convert("RGB").copy()
    if kind == "clean":
        return img
    severity = LEVELS[kind][level]
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=severity))
    if kind == "low_light":
        return ImageEnhance.Brightness(img).enhance(severity)
    if kind == "occlusion":
        rng = random.Random(seed)
        w, h = img.size
        side = max(1, int((severity * w * h) ** 0.5))
        side = min(side, w, h)
        x = rng.randint(0, w - side)
        y = rng.randint(0, h - side)
        ImageDraw.Draw(img).rectangle((x, y, x + side - 1, y + side - 1), fill=(128, 128, 128))
        return img
    raise ValueError(f"Unknown corruption: {kind}")


class TargetedAugmentation:
    def __call__(self, image):
        # Only applied to TRAIN images; train/test severity schedules are documented.
        if random.random() < 0.7:
            kind = random.choice(["blur", "low_light", "occlusion"])
            level = random.choice([1, 2, 3])
            image = corrupt(image, kind, level, random.randrange(1_000_000_000))
        return image


def build_transform(policy: str, training: bool):
    ops = [transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC)]
    if training:
        # A small, shared geometric augmentation keeps a fair baseline.
        ops += [transforms.RandomRotation(degrees=5)]
        if policy == "generic":
            ops += [transforms.RandAugment(num_ops=2, magnitude=7)]
        elif policy == "targeted":
            ops += [TargetedAugmentation()]
        elif policy != "basic":
            raise ValueError(policy)
    ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(ops)


def read_and_verify(dataset_dir: Path):
    audit_path = dataset_dir / "audit.json"
    manifest_path = dataset_dir / "manifest.csv"
    if not audit_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"Expected audit.json and manifest.csv in {dataset_dir}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("BLOCKS_TRAINING") is not False:
        raise RuntimeError("Dataset audit blocks training. STOP and re-check cleaning.")
    if audit.get("pixel_identical_duplicate_groups") != 0:
        raise RuntimeError("Exact-pixel duplicate audit must equal zero.")
    if audit.get("official_test_isolation_violated") is not False:
        raise RuntimeError("Official source test isolation was not verified.")

    with manifest_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("Empty manifest")
    required = {"path", "split", "label", "official_split", "sha256"}
    if not required.issubset(rows[0]):
        raise RuntimeError(f"Missing manifest columns: {required - set(rows[0])}")

    # Repair manifest paths are absolute on user's Windows PC. Also support relocation:
    # resolve missing stored absolute paths using images/{split}/{label}/{filename}.
    for r in rows:
        p = Path(r["path"])
        if not p.is_file():
            p = dataset_dir / "images" / r["split"] / r["label"] / Path(r["path"].replace("\\", "/")).name
        if not p.is_file():
            raise RuntimeError(f"Image missing: {r['path']}")
        r["resolved_path"] = str(p)
        if (r["split"] == "test") != (r["official_split"] == "test"):
            raise RuntimeError(f"Source test isolation violation: {p}")
    if len(set(r["sha256"] for r in rows)) != len(rows):
        raise RuntimeError("Duplicate image hash detected in manifest")
    split_rows = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    claimed = audit.get("split_counts", {})
    for s, subset in split_rows.items():
        if not subset or len(subset) != claimed.get(s):
            raise RuntimeError(f"Split {s}: manifest={len(subset)} audit={claimed.get(s)}")
    labels = sorted(set(r["label"] for r in rows))
    for s in split_rows:
        if set(r["label"] for r in split_rows[s]) != set(labels):
            raise RuntimeError(f"Class missing in split {s}")
    return audit, split_rows, labels


class ImageRows(Dataset):
    def __init__(self, rows, label_to_idx, policy="basic", training=False, kind="clean", level=0, seed=2026):
        self.rows = rows
        self.mapping = label_to_idx
        self.transform = build_transform(policy, training)
        self.kind = kind
        self.level = level
        self.seed = seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        rec = self.rows[idx]
        with Image.open(rec["resolved_path"]) as im:
            im = im.convert("RGB")
        if self.kind != "clean":
            # Match post-resize test severity across all source image dimensions.
            im = transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC)(im)
            # Stable per image, not batch iteration order. Do not alter between models.
            sample_seed = int(hashlib.sha256((str(self.seed) + "|" + rec["source_image"]).encode()).hexdigest()[:8], 16)
            im = corrupt(im, self.kind, self.level, sample_seed)
        return self.transform(im), self.mapping[rec["label"]], rec["source_image"]


def compute_metrics(y_true, y_pred, n_classes):
    n = len(y_true)
    acc = sum(a == b for a, b in zip(y_true, y_pred)) / n if n else 0.0
    f1s = []
    for label in range(n_classes):
        tp = sum(a == label and b == label for a, b in zip(y_true, y_pred))
        fp = sum(a != label and b == label for a, b in zip(y_true, y_pred))
        fn = sum(a == label and b != label for a, b in zip(y_true, y_pred))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return {"accuracy": round(acc, 6), "macro_f1": round(sum(f1s) / n_classes, 6), "n": n}


def run_epoch(model, loader, device, n_classes, optimizer=None, scaler=None, amp=False):
    train = optimizer is not None
    model.train(train)
    actual, predicted = [], []
    cumulative_loss = 0.0
    lossfn = nn.CrossEntropyLoss()
    for batch, (x, y, _) in enumerate(loader, 1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(x)
                loss = lossfn(logits, y)
            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        cumulative_loss += float(loss.detach().cpu()) * len(y)
        actual.extend(y.cpu().tolist())
        predicted.extend(logits.argmax(1).detach().cpu().tolist())
    result = compute_metrics(actual, predicted, n_classes)
    result["loss"] = round(cumulative_loss / max(1, len(actual)), 6)
    return result


@torch.no_grad()
def assess(model, loader, device, labels, amp=False):
    model.eval()
    actual, predicted, items = [], [], []
    for x, y, paths in loader:
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x.to(device, non_blocking=True))
        preds = logits.argmax(1).cpu().tolist()
        yy = y.tolist()
        actual.extend(yy)
        predicted.extend(preds)
        items.extend([{"source_image": p, "actual_sku": labels[a], "predicted_sku": labels[b]} for p, a, b in zip(paths, yy, preds)])
    return compute_metrics(actual, predicted, len(labels)), items


def main():
    p = argparse.ArgumentParser(description="CSE791 fixed EfficientNet-B0 pilot")
    p.add_argument("--data", type=Path, default=Path("local_pilot_clean"))
    p.add_argument("--out", type=Path, default=Path("results/baseline"))
    p.add_argument("--policy", choices=["basic", "generic", "targeted"], default="basic")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--no-pretrained", action="store_true", help="Smoke tests only, not comparable research baseline")
    p.add_argument("--final-test", action="store_true", help="Only after experimental protocol is FROZEN; evaluates official test")
    p.add_argument("--allow-cpu", action="store_true", help="For a software smoke test only")
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        p.error("Positive epochs and batch-size required")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.allow_cpu:
        raise SystemExit("ERROR: CUDA not detected. Run check_gpu.py and install the official CUDA PyTorch build; use --allow-cpu only for testing.")
    audit, splits, labels = read_and_verify(args.data)
    label_to_idx = {label: i for i, label in enumerate(labels)}
    amp = device.type == "cuda"
    if amp:
        torch.backends.cudnn.benchmark = True

    run_dir = args.out / args.policy / f"seed_{args.seed}"
    if (run_dir / "best.pt").exists():
        raise SystemExit(f"Existing trained checkpoint: {run_dir / 'best.pt'}. Choose a fresh --out to avoid overwriting.")
    run_dir.mkdir(parents=True, exist_ok=True)
    weights = None if args.no_pretrained else models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(labels))
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda") if amp else None
    def loader(rows, train=False, kind="clean", level=0):
        ds = ImageRows(rows, label_to_idx, policy=args.policy if train else "basic", training=train,
                       kind=kind, level=level, seed=args.seed)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=train, num_workers=args.workers,
                          pin_memory=amp)
    train_loader = loader(splits["train"], train=True)
    val_loader = loader(splits["val"])
    cfg = {"utc": datetime.now(timezone.utc).isoformat(), "policy": args.policy,
           "epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed, "lr": args.lr,
           "dataset": str(args.data), "audit_split_counts": audit["split_counts"],
           "classes": labels, "pretrained": not args.no_pretrained, "device": str(device),
           "device_name": torch.cuda.get_device_name(0) if amp else "CPU",
           "torch": torch.__version__, "torchvision": __import__("torchvision").__version__,
           "python": platform.python_version(), "final_test_requested": args.final_test}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    best = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, device, len(labels), optimizer, scaler, amp)
        va = run_epoch(model, val_loader, device, len(labels), amp=amp)
        record = {"epoch": epoch, "train": tr, "val": va}
        history.append(record)
        print(f"Epoch {epoch}/{args.epochs} train_acc={tr['accuracy']:.3f} val_acc={va['accuracy']:.3f} val_macro_f1={va['macro_f1']:.3f}", flush=True)
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if va["macro_f1"] > best:
            best = va["macro_f1"]
            torch.save({"state_dict": model.state_dict(), "classes": labels,
                        "policy": args.policy, "seed": args.seed, "best_epoch": epoch}, run_dir / "best.pt")

    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    results, predictions = [], []
    eval_split = "test" if args.final_test else "val"
    for kind in ["clean", *LEVELS]:
        levels = [0] if kind == "clean" else [1, 2, 3]
        for level in levels:
            metrics, items = assess(model, loader(splits[eval_split], kind=kind, level=level), device, labels, amp)
            results.append({"split": eval_split, "condition": kind, "severity": level, **metrics})
            predictions.extend({"split": eval_split, "condition": kind, "severity": level, **item} for item in items)
            print(f"{eval_split}: {kind} severity={level} accuracy={metrics['accuracy']:.3f} macro_f1={metrics['macro_f1']:.3f}", flush=True)
    (run_dir / f"{eval_split}_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (run_dir / f"{eval_split}_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "condition", "severity", "source_image", "actual_sku", "predicted_sku"])
        writer.writeheader()
        writer.writerows(predictions)
    with (run_dir / "RUN_LOG.md").open("w", encoding="utf-8") as f:
        f.write(f"# CSE791 Group 9 training log\n\nDate UTC: {cfg['utc']}\nPolicy: {args.policy}\nSeed: {args.seed}\nDataset: {args.data}\nSplit evaluated: {eval_split}\nBest validation macro-F1: {best}\nNo test used for checkpoint selection.\n")
    print(f"DONE. Results in {run_dir.resolve()}")
    if not args.final_test:
        print("Official TEST split was NOT evaluated. Keep it sealed until the final comparison.")


if __name__ == "__main__":
    main()
