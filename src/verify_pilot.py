"""Lightweight manifest/split check, no torch needed."""
from pathlib import Path
from collections import Counter
from train_pilot import read_and_verify

root = Path('local_pilot_clean')
audit, splits, labels = read_and_verify(root)
print('Data directory:', root.resolve())
print('Classes:', ', '.join(labels))
print('Split counts:', {s: len(v) for s, v in splits.items()})
print('Train class counts:', dict(Counter(r['label'] for r in splits['train'])))
print('Val class counts:', dict(Counter(r['label'] for r in splits['val'])))
print('Test class counts:', dict(Counter(r['label'] for r in splits['test'])))
print('Exact duplicates reported:', audit['pixel_identical_duplicate_groups'])
print('BLOCKS_TRAINING:', audit['BLOCKS_TRAINING'])
print('PASS: verified audit, paths, split counts, class coverage, original test isolation.')
