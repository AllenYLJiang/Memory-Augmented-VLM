"""Sorted weighted metrics, exact overlap reconstruction, source-level bootstrap."""
from __future__ import annotations

from collections import defaultdict
import numpy as np
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_auc_score

SCORE_DECIMALS = 12


def metrics(y, scores, threshold=.5, weights=None):
    y, scores = np.asarray(y, dtype=int), np.round(np.asarray(scores, dtype=float), SCORE_DECIMALS)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    if not len(y) or len(y) != len(scores) or not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Metrics require all planned finite scores and explicit binary labels")
    if w.shape != y.shape or not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError("Invalid metric weights")
    pred = scores >= threshold
    tp, tn = float(w[(y == 1) & pred].sum()), float(w[(y == 0) & ~pred].sum())
    fp, fn = float(w[(y == 0) & pred].sum()), float(w[(y == 1) & ~pred].sum())
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    precision, rec, _ = precision_recall_curve(y, scores, sample_weight=w) if tp + fn else ([], [], [])
    return {"weight": float(w.sum()), "AP_step": float(average_precision_score(y, scores, sample_weight=w)) if tp + fn else None,
            "PR_AUC_trapezoid": float(auc(rec, precision)) if tp + fn else None,
            "ROC_AUC": float(roc_auc_score(y, scores, sample_weight=w)) if len(set(y)) == 2 else None,
            "accuracy": (tp + tn) / float(w.sum()), "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
            "recall": recall, "specificity": specificity, "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn, "threshold": float(threshold)}


def choose_threshold(y, s):
    if len(set(y)) != 2:
        raise ValueError("Both classes required for threshold calibration")
    # At most a few hundred pilot windows; no frame-scale threshold scan.
    s = np.round(np.asarray(s, dtype=float), SCORE_DECIMALS)
    thresholds = sorted(set(map(float, s)) | {float(np.nextafter(max(s), np.inf))})
    return max(thresholds, key=lambda t: (metrics(y, s, t)["balanced_accuracy"], t))


def blocks(n_frames, windows, scores, intervals):
    """Lossless run-length encoding of mean-overlap frame scores, not interpolation."""
    if len(windows) != len(scores):
        raise ValueError("Every dense window must have a score")
    change = defaultdict(lambda: [0, 0.])
    boundaries = {0, n_frames}
    for r, s in zip(windows, scores):
        a, b = int(r["start_frame"]), int(r["end_frame_exclusive"])
        if not 0 <= a < b <= n_frames or not np.isfinite(s):
            raise ValueError("Invalid span/score")
        change[a][0] += 1
        change[a][1] += float(s)
        change[b][0] -= 1
        change[b][1] -= float(s)
        boundaries.update((a, b))
    for a, b in intervals:
        if not 0 <= a < b <= n_frames:
            raise ValueError("Annotation outside exact frame count; do not silently clip")
        boundaries.update((a, b))
    cuts, count, total, result = sorted(boundaries), 0, 0., []
    for a, b in zip(cuts, cuts[1:]):
        count += change[a][0]
        total += change[a][1]
        if count <= 0:
            raise ValueError(f"Uncovered frames [{a}, {b})")
        label = int(any(x <= a and b <= y for x, y in intervals))
        result.append((label, round(total / count, SCORE_DECIMALS), b - a))
    return result


def paired_bootstrap(y, a, b, groups, weights=None, repeats=500, seed=919):
    y, a, b = np.asarray(y), np.round(np.asarray(a), SCORE_DECIMALS), np.round(np.asarray(b), SCORE_DECIMALS)
    groups = np.asarray(groups)
    w = np.ones(len(y)) if weights is None else np.asarray(weights)
    unique, inv = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(repeats):
        multiplicity = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))
        rw = w * multiplicity[inv]
        keep = rw > 0
        if len(np.unique(y[keep])) != 2:
            continue
        deltas.append(float(average_precision_score(y[keep], b[keep], sample_weight=rw[keep]) -
                            average_precision_score(y[keep], a[keep], sample_weight=rw[keep])))
    return {"unit": "source_group", "source_groups": len(unique), "replicates_requested": repeats,
            "replicates_valid": len(deltas), "AP_delta_B_minus_A_CI95": np.quantile(deltas, [.025, .975]).tolist() if deltas else None}
