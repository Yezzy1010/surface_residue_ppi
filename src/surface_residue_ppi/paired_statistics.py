"""Paired uncertainty for chain metrics, resampling linked complexes together."""
from __future__ import annotations

import numpy as np


def component_ids(records):
    """Connect complexes transitively if either chain cluster is shared."""
    parents = list(range(len(records)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    for i, row in enumerate(records):
        for cluster in row['cluster_pair']:
            if cluster in seen:
                parents[find(i)] = find(seen[cluster])
            else:
                seen[cluster] = i
    ids = {}
    return np.array([ids.setdefault(find(i), len(ids)) for i in range(len(records))], dtype=np.int64)


def paired_cluster_bootstrap(left, right, components, replicates=2000, seed=20260927, confidence=.95):
    """Estimate right-minus-left chain macro delta; cluster sizes retain weight.

    The same component draw is used for both methods. Undefined chains must
    agree exactly and remain excluded. Components are resampled, not residues.
    """
    left, right = np.asarray(left, float), np.asarray(right, float)
    components = np.asarray(components)
    if left.ndim != 1 or left.shape != right.shape or components.shape != left.shape:
        raise ValueError('Paired chain arrays must have equal one-dimensional shapes')
    valid = np.isfinite(left)
    if not np.array_equal(valid, np.isfinite(right)) or not valid.any():
        raise ValueError('Methods have different/empty defined chain sets')
    if replicates <= 0 or not 0 < confidence < 1:
        raise ValueError('Invalid bootstrap configuration')
    _, inverse = np.unique(components, return_inverse=True)
    n = int(inverse.max()) + 1
    count = np.bincount(inverse[valid], minlength=n)
    sums = np.bincount(inverse[valid], weights=(right-left)[valid], minlength=n)
    rng = np.random.default_rng(seed)
    values = []
    undefined_draws = 0
    for _ in range(replicates):
        draw = rng.integers(n, size=n)
        denominator = count[draw].sum()
        if denominator:
            values.append(float(sums[draw].sum() / denominator))
        else:
            undefined_draws += 1
    if not values:
        raise ValueError('No defined bootstrap draws')
    alpha = (1-confidence)/2
    bounds = np.quantile(values, [alpha, 1-alpha])
    return dict(delta=float(np.mean((right-left)[valid])), low=float(bounds[0]), high=float(bounds[1]),
                confidence=confidence, components=n, defined_chains=int(valid.sum()),
                undefined_chains=int((~valid).sum()), replicates=replicates,
                undefined_draws=undefined_draws, seed=seed,
                method='paired component percentile bootstrap of chain-weighted means')
