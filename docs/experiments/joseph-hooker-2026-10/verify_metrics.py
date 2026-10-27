"""Recalculate published aggregates; does not validate source labels or predictions."""
from pathlib import Path
import json
import numpy as np

data = json.loads((Path(__file__).parent / 'metrics.json').read_text(encoding='utf-8'))
pages = data['pages']
assert len(pages) == 12 and len({page['page_id'] for page in pages}) == 12
chars = np.array([page['reference_characters'] for page in pages], dtype=np.int64)
base = np.array([page['baseline_character_errors'] for page in pages], dtype=np.int64)
candidate = np.array([page['candidate_character_errors'] for page in pages], dtype=np.int64)
assert sum(p['source_lines'] for p in pages) == 221
assert sum(p['accepted_lines'] for p in pages) == 72
assert int(chars.sum()) == 1668 and int(base.sum()) == 928 and int(candidate.sum()) == 657
rng = np.random.default_rng(20261028)
indices = rng.integers(0, 12, size=(20000, 12))
denominators = chars[indices].sum(axis=1)
assert (denominators > 0).all()
differences = (base - candidate)[indices].sum(axis=1) / denominators
lo, hi = np.percentile(differences, [2.5, 97.5])
assert abs(float(lo) - data['bootstrap_cer_reduction_ci'][0]) < 1e-12
assert abs(float(hi) - data['bootstrap_cer_reduction_ci'][1]) < 1e-12
print(f'CER: {base.sum() / chars.sum():.2%} -> {candidate.sum() / chars.sum():.2%}')
print(f'CER reduction: {(base.sum() - candidate.sum()) / chars.sum() * 100:.2f} percentage points')
print(f'Paired 95% cluster bootstrap interval: [{lo * 100:.2f}, {hi * 100:.2f}] percentage points')
print('Recomputed from public aggregates; underlying references and predictions are not included.')
