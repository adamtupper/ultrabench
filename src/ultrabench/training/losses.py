import torch
import torch.nn as nn


class DiceCELoss(nn.Module):
    """Combined Dice + CrossEntropy loss using MONAI's DiceCELoss backend.

    Accepts [B, num_classes, H, W] logits and [B, H, W] long targets.

    Works for binary (num_classes=2) and multi-class (num_classes>=3).
    MONAI is imported inside __init__ so this module can be imported without
    MONAI installed; an ImportError is raised only on instantiation.
    """

    def __init__(
        self,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1.0,
    ):
        """Initialise DiceCELoss.

        Args:
            ce_weight: Scalar weight applied to the cross-entropy component.
            dice_weight: Scalar weight applied to the Dice component.
            smooth: Additive smoothing applied to both the Dice numerator and
                denominator. Defaults to 1.0.
        """
        super().__init__()
        from monai.losses import DiceCELoss as _MonaiDiceCELoss

        self._loss = _MonaiDiceCELoss(
            softmax=True,
            to_onehot_y=True,
            include_background=False,
            smooth_nr=smooth,
            smooth_dr=smooth,
            lambda_dice=dice_weight,
            lambda_ce=ce_weight,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute combined Dice + CE loss.

        Args:
            pred: [B, num_classes, H, W] logits.
            target: [B, H, W] long (class indices 0..num_classes-1).

        Returns:
            Scalar loss tensor.
        """
        return self._loss(pred, target.unsqueeze(1))
