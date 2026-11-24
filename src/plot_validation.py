"""Visualize results AFTER an actual training run; never creates invented metrics."""
import argparse
import csv
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--run', type=Path, required=True, help='e.g., ..\\results\\group09\\basic\\seed_2026')
a = p.parse_args()
path = a.run / 'val_metrics.json'
if not path.exists():
    raise SystemExit(f'Missing real validation metrics: {path}. Train first.')
rows = json.loads(path.read_text(encoding='utf-8'))
with (a.run / 'validation_summary.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['split','condition','severity','accuracy','macro_f1','n'])
    w.writeheader()
    w.writerows(rows)
try:
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit('CSV generated. To create PNG plots: .venv\\Scripts\\python.exe -m pip install matplotlib')
for metric in ('accuracy', 'macro_f1'):
    fig, ax = plt.subplots(figsize=(7.5,4.5))
    base = next(r[metric] for r in rows if r['condition'] == 'clean')
    for kind in ('blur','low_light','occlusion'):
        vals = [r for r in rows if r['condition'] == kind]
        vals.sort(key=lambda v:v['severity'])
        ax.plot([0] + [v['severity'] for v in vals], [base] + [v[metric] for v in vals], '-o', label=kind.replace('_',' '))
    ax.set(ylim=(0,1), xlabel='Synthetic degradation severity (0=clean)', ylabel=metric.replace('_',' ').title(),
           title='Products-10K pilot validation only (not official test)')
    ax.set_xticks([0,1,2,3])
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(a.run / f'validation_{metric}_severity.png', dpi=180)
    plt.close(fig)
print('Created validation_summary.csv and metric PNGs in:', a.run.resolve())
