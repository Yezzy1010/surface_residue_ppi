"""Metric aggregation for the Phase 2 residue-level task."""
from __future__ import annotations

import math

import numpy as np


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    labels = labels.astype(np.int64, copy=False)
    scores = scores.astype(np.float64, copy=False)
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    ap = float("nan")
    if positives:
        order = np.argsort(-scores, kind="mergesort")
        sorted_labels = labels[order]
        sorted_scores = scores[order]
        cumulative_tp = np.cumsum(sorted_labels)
        ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), len(sorted_scores) - 1]
        tp = cumulative_tp[ends]
        fp = (ends + 1) - tp
        prev_recall = np.r_[0.0, tp[:-1] / positives]
        recall = tp / positives
        precision = tp / (tp + fp)
        ap = float(np.sum((recall - prev_recall) * precision))
    auc = float("nan")
    if positives and negatives:
        ranks = np.empty(len(scores), dtype=np.float64)
        asc = np.argsort(scores, kind="mergesort")
        sorted_asc = scores[asc]
        start = 0
        while start < len(scores):
            end = start + 1
            while end < len(scores) and sorted_asc[end] == sorted_asc[start]:
                end += 1
            ranks[asc[start:end]] = (start + 1 + end) / 2.0
            start = end
        auc = float((ranks[labels == 1].sum() - positives * (positives + 1) / 2.0) /
                    (positives * negatives))
    return dict(average_precision=ap, auroc=auc, positives=positives, negatives=negatives)


def macro_metrics(instance_metrics: list[dict], pooled: dict) -> dict:
    """Aggregate the primary metrics over predefined protein instances."""
    if not instance_metrics:
        raise ValueError("No validation instances available")
    valid_ap = [row["average_precision"] for row in instance_metrics
                if math.isfinite(row["average_precision"])]
    valid_auc = [row["auroc"] for row in instance_metrics
                 if math.isfinite(row["auroc"])]
    if not valid_ap:
        raise ValueError("No validation instance has positive labels for AP")
    if not valid_auc:
        raise ValueError("No validation instance has both classes for AUROC")
    return dict(average_precision=float(np.mean(valid_ap)),
                auroc=float(np.mean(valid_auc)),
                positives=pooled["positives"], negatives=pooled["negatives"],
                instances=len(instance_metrics), ap_instances=len(valid_ap),
                auroc_instances=len(valid_auc),
                ap_undefined_instances=len(instance_metrics) - len(valid_ap),
                auroc_undefined_instances=len(instance_metrics) - len(valid_auc),
                pooled_average_precision=pooled["average_precision"],
                pooled_auroc=pooled["auroc"])
