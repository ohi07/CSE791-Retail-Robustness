#!/usr/bin/env python3
"""Group 09 / CSE791: safe repair of extracted Products-10K pilot.

Never modifies the source folder. Produces a new audited, reproducible copy.

Scientific policy (conservative):
  - Exclude SKU 8227 and 8631, whose identical images have different labels.
  - When train/test share pixels, retain the original official TEST image.
  - Deduplicate within a split or across training/validation.
  - Never promote original test images into training or validation.
  - Verify every copied JPEG, recompute decoded-pixel hashes, and re-audit.

Run from D:\\CSE791\\Products10K:
  .venv\\Scripts\\python.exe fix_pilot_duplicates.py --source local_pilot
  .venv\\Scripts\\python.exe fix_pilot_duplicates.py --source local_pilot --apply

Requires Pillow (already installed).
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from PIL import Image

MANIFEST_FIELDS=['path','label','split','sha256','official_split','source_image']
REVIEW_FIELDS=['decision','reason','label','split','official_split','source_image','original_path','original_pixel_sha256']
EXCLUDED={'8227','8631'}


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, fieldnames, rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fieldnames,extrasaction='ignore')
        writer.writeheader();writer.writerows(rows)


def get_rel_path(raw, label, split):
    """Convert old absolute Windows path to safe relative images/split/SKU/file."""
    parts=PureWindowsPath(raw).parts
    image_indexes=[i for i,p in enumerate(parts) if p.lower()=='images']
    if len(image_indexes)!=1:
        raise ValueError(f'Expected one images/ directory in path: {raw}')
    relative=parts[image_indexes[0]:]
    if len(relative)!=4 or relative[1]!=split or relative[2]!=label:
        raise ValueError(f'Unexpected image path structure: {raw}')
    if relative[3] in {'.','..'} or not relative[3].lower().endswith(('.jpg','.jpeg','.png')):
        raise ValueError(f'Unsafe image filename: {raw}')
    return Path(*relative)


def pixel_hash(path):
    # Same pre-encoding hash convention as original extraction, now on copied JPEGs.
    with Image.open(path) as im:
        x=im.convert('RGB')
        payload=(str(x.size)+':RGB').encode()+x.tobytes()
    return hashlib.sha256(payload).hexdigest()


def group_issues(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['sha256']].append(r)
    result=[]
    for digest,items in groups.items():
        if len(items)<=1:continue
        labels={r['label'] for r in items};splits={r['split'] for r in items}
        issue_type=('CONFLICTING_LABELS' if len(labels)>1 else
                    'CROSS_SPLIT_DUPLICATE' if len(splits)>1 else
                    'WITHIN_SPLIT_DUPLICATE')
        result.append({'type':issue_type, 'pixel_sha256':digest,'labels':';'.join(sorted(labels)),
                       'splits':';'.join(sorted(splits)),
                       'paths':' | '.join(r['path'] for r in items)})
    return result


def build_plan(rows):
    if not rows:raise ValueError('Empty manifest')
    for i,r in enumerate(rows):
        if not all(f in r for f in MANIFEST_FIELDS):raise ValueError(f'Missing manifest columns at row {i}')
        if r['split'] not in {'train','val','test'}:
            raise ValueError(f'Unexpected split at row {i}: {r["split"]}')
        if (r['split']=='test')!=(r['official_split']=='test'):
            raise ValueError(f'Official train/test separation violated at row {i}')
        if r['official_split'] not in {'train','test'}:
            raise ValueError(f'Unexpected official split at row {i}')
        get_rel_path(r['path'],r['label'],r['split'])

    decisions=[];retained=[]
    remaining=[r for r in rows if r['label'] not in EXCLUDED]
    excluded=[r for r in rows if r['label'] in EXCLUDED]
    for r in excluded:
        decisions.append({**r,'decision':'EXCLUDE',
          'reason':'Hold class: exact identical images were labeled as both SKU 8227 and SKU 8631'})
    by_hash=defaultdict(list)
    for r in remaining:by_hash[r['sha256']].append(r)
    for sha, group in by_hash.items():
        labels={r['label'] for r in group}
        if len(labels)>1:
            raise ValueError(f'Unresolved conflicting labels in retained group: {labels}, hash {sha}')
        # Priority: preserve original TEST image, then TRAIN, then VAL;
        # this only handles duplicate content, not optimization of test performance.
        priority={'test':0,'train':1,'val':2}
        ranked=sorted(group,key=lambda r:(priority[r['split']],r['source_image'],r['path']))
        retained.append(ranked[0])
        for r in ranked[1:]:
            why=('Original source TEST retained; duplicate from development set excluded' if ranked[0]['split']=='test' else
                 'Repeated decoded pixels; retained one image in ' + ranked[0]['split'])
            decisions.append({**r,'decision':'EXCLUDE','reason':why})
    retained=sorted(retained,key=lambda r:(r['label'],r['split'],r['path']))
    if len(rows)!=len(retained)+len(decisions):raise AssertionError('Plan does not account for every input row')
    if group_issues(retained):raise AssertionError('Unresolved same-hash groups in manifest plan')
    counts=Counter(r['split'] for r in retained)
    skus=sorted({r['label'] for r in retained},key=int)
    for sku in skus:
        for split in ('train','val','test'):
            if not any(r['label']==sku and r['split']==split for r in retained):
                raise ValueError(f'{sku} missing {split} after cleaning')
    return retained,decisions,counts,skus


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path('local_pilot'))
    ap.add_argument('--output',type=Path,default=Path('local_pilot_clean'))
    ap.add_argument('--apply',action='store_true',help='Actually copy the cleaned dataset (default: preview only)')
    a=ap.parse_args();src=a.source.resolve();dst=a.output.resolve()
    if src==dst or src in dst.parents or dst in src.parents:
        raise SystemExit('Source and output must be separate sibling folders')
    src_manifest=src/'manifest.csv';src_audit=src/'audit.json'
    if not src_manifest.is_file():raise SystemExit(f'Missing {src_manifest}')
    if not src_audit.is_file():raise SystemExit(f'Missing {src_audit}')
    oldaudit=json.loads(src_audit.read_text(encoding='utf-8'))
    rows=read_csv(src_manifest)
    keep,discard,counts,skus=build_plan(rows)
    reasons=Counter(d['reason'] for d in discard)
    summary={'original_rows':len(rows),'retained_rows':len(keep),'excluded_rows':len(discard),
             'retained_skus':skus,'excluded_skus':sorted(EXCLUDED,key=int),
             'split_counts':dict(counts),'excluded_reasons':dict(reasons),
             'original_BLOCKS_TRAINING':oldaudit.get('BLOCKS_TRAINING'),
             'test_images_kept':sum(r['split']=='test' for r in keep)}
    print('\nDATASET REPAIR PREVIEW (original images remain untouched)')
    print(json.dumps(summary,indent=2))
    if not a.apply:
        print('\nDRY RUN ONLY: no files changed.')
        print('To create the cleaned dataset, run again with --apply.')
        return

    if dst.exists() and any(dst.iterdir()):
        raise SystemExit(f'Output directory already exists and is not empty: {dst}. No files were changed.')
    dst.mkdir(parents=True,exist_ok=True)
    # Copy manifest/provenance, not raw images outside the retained set.
    for filename in ('manifest.csv','audit.json','duplicate_audit.csv','PROGRESS_LOG.txt'):
        f=src/filename
        if f.is_file():shutil.copy2(f,dst/('source_'+filename))

    new_rows=[]
    print(f'\nCopying and validating {len(keep)} retained images...',flush=True)
    for n, row in enumerate(keep,1):
        rel=get_rel_path(row['path'],row['label'],row['split'])
        source_path=src/rel
        output_path=dst/rel
        if not source_path.is_file():
            raise SystemExit(f'MISSING IMAGE: {source_path}\nFix source folder; do not train with partial output.')
        output_path.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source_path,output_path)
        # Recompute from the actual saved JPEG, not merely trusting the old manifest.
        hash_value=pixel_hash(output_path)
        new_rows.append({**row,'path':str(output_path.resolve()),'sha256':hash_value})
        if n%100==0:print(f'Validated {n}/{len(keep)} images',flush=True)

    # Recomputed image-pixel hashes: final integrity check.
    physical_issues=group_issues(new_rows)
    actual_counts=Counter(r['split'] for r in new_rows)
    actual_labels=sorted({r['label'] for r in new_rows},key=int)
    bad_orig=any((r['split']=='test')!=(r['official_split']=='test') for r in new_rows)
    missing=any(not any(r['label']==lab and r['split']==split for r in new_rows)
                for lab in actual_labels for split in ('train','val','test'))
    blocked=bool(physical_issues or bad_orig or missing)

    write_csv(dst/'manifest.csv',MANIFEST_FIELDS,new_rows)
    write_csv(dst/'duplicate_audit.csv',['type','pixel_sha256','labels','splits','paths'],physical_issues)
    write_csv(dst/'CLEANING_DECISIONS.csv',REVIEW_FIELDS,[
        {'decision':x['decision'],'reason':x['reason'],'label':x['label'],'split':x['split'],
         'official_split':x['official_split'],'source_image':x['source_image'],
         'original_path':x['path'],'original_pixel_sha256':x['sha256']} for x in discard
    ])
    by_sku={sku:dict(Counter(r['split'] for r in new_rows if r['label']==sku)) for sku in actual_labels}
    result={
        'date_utc':datetime.now(timezone.utc).isoformat(),
        'source':oldaudit.get('source','nyris/products10k-v1'),
        'original_dataset':str(src),'source_audit_preserved':str(dst/'source_audit.json'),
        'cleaning_policy':'Conservative: exclude both labels 8227,8631; keep original test in any same-hash conflict; drop other exact duplicates',
        'excluded_skus':sorted(EXCLUDED,key=int),'skus':actual_labels,
        'split_counts':dict(actual_counts),'class_split_counts':by_sku,
        'source_rows':len(rows),'retained_rows':len(new_rows),'excluded_rows':len(discard),
        'actual_saved_jpeg_pixel_hashes_recomputed':True,
        'pixel_identical_duplicate_groups':len(physical_issues),
        'missing_splits':missing,'official_test_isolation_violated':bad_orig,
        'BLOCKS_TRAINING':blocked,
        'limitations':['Exact decoded-pixel duplicates only; near duplicates and capture-session leakage were not audited.',
                       'Only nine pilot SKU classes after conservative label-conflict exclusion.',
                       'Mixed product categories, not packaged groceries; change study claim accordingly.',
                       'Only six test images per SKU; report uncertainty and avoid broad generalization.',
                       'SKU labels and visual identity still require independent human review.',
                       'Training and test evaluation have not yet occurred.'],
        'training_completed':False
    }
    (dst/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (dst/'PROGRESS_LOG.txt').write_text('\n'.join([
      'CSE791 GROUP 09 — CONSERVATIVE DATASET CLEANING',
      f'Date UTC: {result["date_utc"]}',
      f'Original: {src}; Cleaned: {dst}',
      'Original source dataset and files were not modified.',
      f'Excluded ambiguous SKU labels: {", ".join(sorted(EXCLUDED,key=int))}',
      f'Split counts: train={actual_counts["train"]}, val={actual_counts["val"]}, test={actual_counts["test"]}',
      f'Images excluded: {len(discard)}; saved JPEG duplicate groups after re-hashing: {len(physical_issues)}',
      f'BLOCKS_TRAINING: {blocked}',
      'No pretrained or trained model results are claimed; independent near-duplicate and label audits still advised.'
    ])+'\n',encoding='utf-8')
    print('\nFINAL CLEAN AUDIT',json.dumps({
        'classes':len(actual_labels),'split_counts':dict(actual_counts),
        'pixel_duplicate_groups':len(physical_issues),'BLOCKS_TRAINING':blocked,
        'manifest':str(dst/'manifest.csv'),'audit':str(dst/'audit.json')
    },indent=2),flush=True)
    if blocked:
        raise SystemExit('BLOCKED: physical image audit still found a problem. Do NOT train. Upload output audit.json and duplicate_audit.csv.')
    print('\nSUCCESS: A separate clean dataset copy was prepared. Original local_pilot is unchanged.')
    print('Next: human-check representative labels and near-duplicates, then run baseline using local_pilot_clean\\manifest.csv.')

if __name__=='__main__':
    try:main()
    except (ValueError,AssertionError,OSError) as e:
        print(f'ERROR: {e}',file=sys.stderr);sys.exit(2)
