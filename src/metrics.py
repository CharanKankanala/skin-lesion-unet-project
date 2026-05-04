"""
Evaluation metrics for binary segmentation.

All metrics expect:
    preds   - raw model logits, shape (N, 1, H, W) or (N, H, W)
    targets - binary ground-truth masks (0 or 1), same shape

Logits are passed through a sigmoid and thresholded at 0.5 before computing
the metric, so the same `preds` tensor that was fed to the loss can be
reused here.
"""

import torch

EPS = 1e-6


def _binarize(preds, threshold=0.5):
    preds = torch.sigmoid(preds)
    return (preds > threshold).float()


def dice_score(preds, targets, threshold=0.5):
    """Sørensen–Dice coefficient: 2|A∩B| / (|A|+|B|)."""
    preds = _binarize(preds, threshold).view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()
    return ((2.0 * intersection) / (preds.sum() + targets.sum() + EPS)).item()


def iou_score(preds, targets, threshold=0.5):
    """Intersection over Union (Jaccard): |A∩B| / |A∪B|."""
    preds = _binarize(preds, threshold).view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return (intersection / (union + EPS)).item()


def precision_score(preds, targets, threshold=0.5):
    """Precision: TP / (TP + FP). High precision => few false-positive pixels."""
    preds = _binarize(preds, threshold).view(-1)
    targets = targets.view(-1)

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    return (tp / (tp + fp + EPS)).item()


def recall_score(preds, targets, threshold=0.5):
    """Recall (sensitivity): TP / (TP + FN). High recall => few missed lesion pixels."""
    preds = _binarize(preds, threshold).view(-1)
    targets = targets.view(-1)

    tp = (preds * targets).sum()
    fn = ((1 - preds) * targets).sum()
    return (tp / (tp + fn + EPS)).item()
