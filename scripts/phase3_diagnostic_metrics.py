"""Recompute diagnostic metrics on common and surface-only validation masks."""
import numpy as np
from phase2_training_metrics import binary_metrics, macro_metrics


def metric_arrays(archive, surface_only=False):
    labels, scores, offsets, surface = (archive[k] for k in ('labels', 'logits', 'offsets', 'surface_mask'))
    if (labels.ndim != 1 or scores.shape != labels.shape or surface.shape != labels.shape
            or surface.dtype != np.bool_ or offsets.ndim != 1 or len(offsets) < 2
            or offsets[0] != 0 or offsets[-1] != len(labels) or np.any(np.diff(offsets) <= 0)
            or not np.isfinite(scores).all() or np.any((labels != 0) & (labels != 1))):
        raise ValueError('Invalid diagnostic prediction arrays')
    mask = surface if surface_only else np.ones(len(labels), dtype=np.bool_)
    rows, valid_rows = [], []
    for a, b in zip(offsets[:-1], offsets[1:]):
        row = binary_metrics(labels[a:b][mask[a:b]], scores[a:b][mask[a:b]])
        rows.append(row)
        if mask[a:b].any():
            valid_rows.append(row)
    macro = macro_metrics(valid_rows, binary_metrics(labels[mask], scores[mask]))
    macro.update(surface_only=surface_only, excluded_empty_chains=len(rows) - len(valid_rows))
    return {key: np.array([row[key] for row in rows]) for key in ('average_precision', 'auroc')}, macro


def saved_metrics(archive):
    return metric_arrays(archive)[1], metric_arrays(archive, True)[1]
