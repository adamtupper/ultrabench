import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """
    Combined Dice + BCE loss for binary segmentation.

    Drop-in replacement for CrossEntropyLoss: accepts [B, 2, H, W] logits
    and [B, H, W] long targets (0=background, 1=foreground).

    The foreground channel (index 1) is used for both BCE and Dice.
    """

    def __init__(
        self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1.0
    ):
        """
        Args:
            bce_weight: Weight for the BCE component.
            dice_weight: Weight for the Dice component.
            smooth: Additive smoothing to avoid division by zero.
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, 2, H, W] logits
            target: [B, H, W] long (0 or 1)

        Returns:
            Scalar loss.
        """
        fg_logits = pred[:, 1]  # [B, H, W] foreground logits
        target_f = target.float()  # [B, H, W]

        bce = F.binary_cross_entropy_with_logits(fg_logits, target_f)

        prob = torch.sigmoid(fg_logits)
        intersection = (prob * target_f).sum(dim=(-2, -1))
        dice = 1 - (2 * intersection + self.smooth) / (
            prob.sum(dim=(-2, -1)) + target_f.sum(dim=(-2, -1)) + self.smooth
        )
        dice = dice.mean()

        return self.bce_weight * bce + self.dice_weight * dice


class DiceCELoss(nn.Module):
    """
    Combined Dice + CrossEntropy loss for binary and multi-class segmentation.

    Accepts [B, num_classes, H, W] logits and [B, H, W] long targets.

    CE is applied over all classes. Soft Dice is computed per foreground class
    (1..num_classes-1) using softmax probabilities and averaged across classes.
    For binary (num_classes=2) this is numerically equivalent to DiceBCELoss.
    """

    def __init__(
        self, ce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1.0
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   [B, num_classes, H, W] logits
            target: [B, H, W] long (class indices 0..num_classes-1)

        Returns:
            Scalar loss.
        """
        ce = F.cross_entropy(pred, target)

        probs = torch.softmax(pred, dim=1)  # [B, C, H, W]
        num_classes = pred.shape[1]
        dice_per_class = []
        for c in range(1, num_classes):
            p = probs[:, c]  # [B, H, W]
            t = (target == c).float()  # [B, H, W]
            intersection = (p * t).sum(dim=(-2, -1))
            dice_c = 1 - (2 * intersection + self.smooth) / (
                p.sum(dim=(-2, -1)) + t.sum(dim=(-2, -1)) + self.smooth
            )
            dice_per_class.append(dice_c.mean())
        dice = torch.stack(dice_per_class).mean()

        return self.ce_weight * ce + self.dice_weight * dice
