#!/usr/bin/env python3
"""Extract a selected Nyris Products-10K pilot without loading the entire 20GB dataset.

Creates local images, train/val/test splits, leakage reports and a manifest
compatible with CSE791 Group09 run_experiment.py.

Only images from the TRAIN source split may enter the TRAIN or VAL partitions.
All original TEST images remain TEST. Images are not used to select SKUs here.

Requires: pip install datasets pillow
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from datasets import load_dataset

REPO = "nyris/products10k-v1"


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", type=Path, default=Path("selected_skus_pilot.csv"))
    p.add_argument("--output", type=Path, default=Path("local_pilot"))
    p.add_argument("--val-fraction", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--train-cap", type=int, default=0, help="0 means all available TRAIN images per SKU")
    p.add_argument("--test-cap", type=int, default=0, help="0 means all available TEST images per SKU")
    p.add_argument("--revision", type=str, help="Optional Hugging Face dataset git commit to pin")
    return p.parse_args()


def read_selection(filename):
    with filename.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or not {'sku'}.issubset(rows[0]):
        raise ValueError("selection CSV must contain a sku column")
    skus = list(dict.fromkeys(str(r['sku']).strip() for r in rows if str(r.get('sku','')).strip()))
    if not skus or not all(x.isdecimal() for x in skus):
        raise ValueError("SKUs must be numeric and nonempty")
    return skus


def pixel_hash(image):
    # Exact decoded-pixel hash, independent of JPEG compression and EXIF metadata.
    x = image.convert('RGB')
    payload = (str(x.size) + ':RGB').encode() + x.tobytes()
    return hashlib.sha256(payload).hexdigest()


def save_split(source_split, skus, root, cap, revision, log):
    # Current nyris parquet rows include sku, image, img_file.
    kwargs = {'split':source_split, 'streaming':True, 'filters':[('sku','in',skus)]}
    if revision: kwargs['revision']=revision
    ds = load_dataset(REPO, **kwargs)
    saved = Counter()
    found = []
    seen = 0
    print(f"\nFetching {source_split} split for {len(skus)} selected SKUs...",flush=True)
    for rec in ds:
        seen += 1
        sku = str(rec['sku'])
        if sku not in skus or (cap and saved[sku]>=cap):
            continue
        try:
            img = rec['image'].convert('RGB')
            h = pixel_hash(img)
            ordinal = saved[sku]+1
            dest = root / 'images' / source_split / sku / f"{source_split}_{sku}_{ordinal:04d}.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            img.save(dest, 'JPEG', quality=95)
            found.append({
                'path': str(dest.resolve()), 'label': sku,
                'split': source_split, 'official_split': source_split,
                'sha256':h, 'source_image':str(rec.get('img_file','')),
            })
            saved[sku] += 1
            if sum(saved.values())%40 == 0:
                print(f"{source_split}: {sum(saved.values())} images saved",flush=True)
        except Exception as e:
            msg=f"WARNING: {source_split} SKU {sku}, source {rec.get('img_file')}: {e}"
            print(msg,flush=True);log.append(msg)
        if cap and all(saved.get(s,0)>=cap for s in skus):
            break
    for sku in skus:
        print(f"  {source_split} SKU {sku}: {saved[sku]} images",flush=True)
        if not saved[sku]: log.append(f"Missing {source_split} images for SKU {sku}")
    return found


def validation_split(rows, skus, root, val_fraction, seed):
    # Group identical decoded pixels before splitting to prevent exact train/val leakage.
    for sku in skus:
        groups=defaultdict(list)
        for row in rows:
            if row['official_split']=='train' and row['label']==sku:
                groups[row['sha256']].append(row)
        keys=sorted(groups)
        random.Random(f"{seed}:{sku}").shuffle(keys)
        if len(keys)<2:raise RuntimeError(f"SKU {sku} has fewer than 2 distinct training images")
        desired=max(1,round(sum(map(len,groups.values()))*val_fraction))
        chosen=[]
        n=0
        for key in keys[:-1]:  # leave at least one hash group in training
            if n>=desired:break
            chosen.append(key)
            n+=len(groups[key])
        for key in chosen:
            for r in groups[key]:
                old=Path(r['path'])
                target=root/'images'/'val'/sku/old.name
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.move(str(old),str(target))
                r['path']=str(target.resolve())
                r['split']='val'


def duplicate_audit(rows):
    byhash=defaultdict(list)
    for r in rows:byhash[r['sha256']].append(r)
    issues=[]
    blocking=False
    for h,items in byhash.items():
        if len(items)<2:continue
        labels=set(r['label'] for r in items)
        splits=set(r['split'] for r in items)
        if len(labels)>1:
            blocking=True
            category='SAME_IMAGE_DIFFERENT_SKU'
        elif len(splits)>1:
            blocking=True
            category='CROSS_SPLIT_PIXEL_DUPLICATE'
        else:
            category='WITHIN_SPLIT_DUPLICATE'
        issues.append({'type':category,'pixel_sha256':h,'skus':';'.join(sorted(labels)),
                       'splits':';'.join(sorted(splits)),'paths':' | '.join(x['path'] for x in items)})
    return issues,blocking


def write_csv(path, rows, fieldnames):
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames,extrasaction='ignore')
        w.writeheader();w.writerows(rows)


def main():
    a=args_parser()
    if not 0<a.val_fraction<0.5:
        raise SystemExit('--val-fraction must be in (0, 0.5)')
    if a.train_cap<0 or a.test_cap<0:
        raise SystemExit('caps cannot be negative')
    skus=read_selection(a.selection)
    out=a.output.resolve()
    if (out/'manifest.csv').exists():
        raise SystemExit(f"{out/'manifest.csv'} already exists. Move/rename output folder before starting a fresh extraction.")
    out.mkdir(parents=True,exist_ok=True)
    log=[]
    train=save_split('train',skus,out,a.train_cap,a.revision,log)
    test=save_split('test',skus,out,a.test_cap,a.revision,log)
    rows=train+test
    if not rows:raise SystemExit('No images were downloaded; check the repository or network access.')
    validation_split(rows,skus,out,a.val_fraction,a.seed)
    duplicate_issues,blocking=duplicate_audit(rows)
    write_csv(out/'duplicate_audit.csv',duplicate_issues,['type','pixel_sha256','skus','splits','paths'])
    write_csv(out/'manifest.csv',rows,['path','label','split','sha256','official_split','source_image'])
    count=Counter(r['split'] for r in rows)
    bysku={sku:dict(Counter(r['split'] for r in rows if r['label']==sku)) for sku in skus}
    insufficient=any(any(not bysku[sku].get(k) for k in ['train','val','test']) for sku in skus)
    summary={
        'date_utc':datetime.now(timezone.utc).isoformat(),'source':REPO,'revision':a.revision,
        'selection_file':str(a.selection.resolve()),'skus':skus,'source_train_rows':len(train),
        'source_test_rows':len(test),'split_counts':dict(count),'class_split_counts':bysku,
        'validation_fraction':a.val_fraction,'seed':a.seed,
        'pixel_identical_duplicate_groups':len(duplicate_issues),
        'BLOCKS_TRAINING':bool(blocking or insufficient),
        'warnings':log,
        'method':'Original source test kept test-only. Validation is class-stratified from source train. Exact decoded-pixel hashes are grouped before train/val split. No model training has been performed.',
        'limitations':['Selection is intentionally nonrepresentative; only 11 pilot SKUs.',
                       'Near duplicates and capture-session leakage not detected by exact pixel hashes.',
                       'Original test size per SKU is small.',
                       'The selected groups are cosmetics, necklaces and handbags, not an FMCG-only sample.',
                       'No augmented images appear in test.'],
    }
    (out/'audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (out/'PROGRESS_LOG.txt').write_text('\n'.join([
        'CSE791 GROUP 09 — DATASET AUDIT AND EXTRACTION',
        f'UTC: {summary["date_utc"]}',f'Source: {REPO}',f'SKUs: {", ".join(skus)}',
        f'Train: {count["train"]} | Val: {count["val"]} | Test: {count["test"]}',
        f'Blockers: {summary["BLOCKS_TRAINING"]}',
        'Actions: training has NOT been run; test data has NOT been used for hyperparameter selection.',
        'Warnings:',*log,
    ]),encoding='utf-8')
    print('\nEXTRACTION SUMMARY',flush=True)
    print(json.dumps({'splits':dict(count),'classes':len(skus),'blocking_issues':blocking,
        'missing_splits':insufficient,'manifest':str(out/'manifest.csv'),
        'audit':str(out/'audit.json')},indent=2),flush=True)
    if blocking or insufficient:
        raise SystemExit('ATTENTION: Audit problems detected. Do NOT TRAIN before reviewing duplicate_audit.csv and audit.json.')
    print('DATASET READY FOR BASELINE TRAINING (after visual/label integrity check).',flush=True)

if __name__=='__main__': main()
