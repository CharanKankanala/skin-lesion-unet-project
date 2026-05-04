import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)

        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (
            preds.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        return self.bce(preds, targets)


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss, useful under foreground-background imbalance."""
    def __init__(self, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.dice = DiceLoss(smooth=smooth)
        self.bce = BCELoss()

    def forward(self, preds, targets):
        return self.dice(preds, targets) + self.bce(preds, targets)


def get_loss(name):
    """Factory for selecting a loss function by name."""
    name = name.lower()
    if name == "dice":
        return DiceLoss()
    if name == "bce":
        return BCELoss()
    if name in ("combined", "dice_bce", "dicebce"):
        return DiceBCELoss()
    raise ValueError(f"Unknown loss '{name}'. Choose from: dice, bce, combined.")
