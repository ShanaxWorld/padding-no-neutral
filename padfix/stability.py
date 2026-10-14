"""Per-image stability metrics (paper, Sect. 3.4), copied from the experiment notebooks.

A prediction is a dict with numpy arrays: boxes (N,4, xyxy), scores (N,), labels (N,).
"""
import numpy as np


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def greedy_match(p, q, thr, iou_min=0.5):
    """Greedy, score-ordered, same-class matching of p's and q's detections with score >= thr."""
    ia = np.where(p["scores"] >= thr)[0]; ib = np.where(q["scores"] >= thr)[0]
    ia = ia[np.argsort(-p["scores"][ia])]
    M = iou_matrix(p["boxes"][ia], q["boxes"][ib]); used = set(); pairs = []
    for r, i in enumerate(ia):
        best, bc = -1.0, None
        for c, j in enumerate(ib):
            if c not in used and p["labels"][i] == q["labels"][j] and M[r, c] > best:
                best, bc = M[r, c], c
        if bc is not None and best >= iou_min:
            used.add(bc); pairs.append((i, ib[bc], best))
    return pairs, ia, ib


def compare(ref, new, score_thr=0.5, moved_iou=0.9):
    """ref: batch-1 prediction, new: batched prediction of the same image.

    Returns whether the image changed ("flip": any confident detection moved, lost or new),
    the per-detection counts, and the largest score change among matched detections (score >= 0.05).
    """
    pairs, ia, ib = greedy_match(ref, new, score_thr)
    stable = sum(iou >= moved_iou for _, _, iou in pairs)
    moved = len(pairs) - stable
    lost, gained = len(ia) - len(pairs), len(ib) - len(pairs)
    allp, _, _ = greedy_match(ref, new, 0.05)
    dscore = max([abs(ref["scores"][i] - new["scores"][j]) for i, j, _ in allp], default=0.0)
    return {"flip": (moved + lost + gained) > 0, "stable": stable, "moved": moved, "lost": lost,
            "gained": gained, "max_score_delta": dscore}


def change_rate(refs, news, **kw):
    """Share of images whose confident detections change (paper: "images changed")."""
    flips = [compare(r, n, **kw)["flip"] for r, n in zip(refs, news)]
    return float(np.mean(flips)) if flips else 0.0
