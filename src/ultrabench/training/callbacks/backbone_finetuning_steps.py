"""Step-based backbone finetuning callback."""

import logging
from typing import Any, Callable

import lightning.pytorch as pl
from lightning.pytorch.callbacks import BaseFinetuning
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from torch.nn import Module
from torch.optim import Optimizer
from typing_extensions import override

log = logging.getLogger(__name__)


def multiplicative(_step: int) -> float:
    """Default multiplicative factor for backbone LR scheduling."""
    return 1.001


class BackboneFinetuningSteps(BaseFinetuning):
    """Finetune a backbone model based on a user-defined LR schedule.

    Step-based equivalent of
    :class:`lightning.pytorch.callbacks.BackboneFinetuning`. Unfreezes the
    backbone at a specific global step instead of an epoch boundary, then
    applies per-step LR scaling via ``lambda_func`` on every subsequent
    training step. ``finetune_function`` is a no-op; all LR scaling occurs
    in ``on_train_batch_start``.

    Args:
        unfreeze_backbone_at_step: Global step at which the backbone will be
            unfrozen.
        lambda_func: Multiplicative scheduling function applied each step to
            increase the backbone learning rate.
        backbone_initial_ratio_lr: Used to scale down the backbone LR compared
            to the rest of the model.
        backbone_initial_lr: Optional initial learning rate for the backbone.
            By default, ``current_learning_rate * backbone_initial_ratio_lr``
            is used.
        should_align: Whether to align backbone LR with the current LR when the
            backbone LR reaches it.
        initial_denom_lr: When unfreezing the backbone, the initial LR will be
            ``current_learning_rate / initial_denom_lr`` (only used when
            ``backbone_initial_lr`` and ``backbone_initial_ratio_lr`` are not
            set).
        train_bn: Whether to make Batch Normalization trainable.
        verbose: Display current learning rate for model and backbone.
        rounding: Precision for displaying learning rate.
    """

    def __init__(
        self,
        unfreeze_backbone_at_step: int = 1000,
        lambda_func: Callable = multiplicative,
        backbone_initial_ratio_lr: float = 10e-2,
        backbone_initial_lr: float | None = None,
        should_align: bool = True,
        initial_denom_lr: float = 10.0,
        train_bn: bool = True,
        verbose: bool = False,
        rounding: int = 12,
    ) -> None:
        super().__init__()
        self.unfreeze_step = unfreeze_backbone_at_step
        self.lambda_func = lambda_func
        self.backbone_initial_ratio_lr = backbone_initial_ratio_lr
        self.backbone_initial_lr = backbone_initial_lr
        self.should_align = should_align
        self.initial_denom_lr = initial_denom_lr
        self.train_bn = train_bn
        self.verbose = verbose
        self.rounding = rounding
        self.previous_backbone_lr: float | None = None

    @override
    def state_dict(self) -> dict[str, Any]:
        return {
            "internal_optimizer_metadata": self._internal_optimizer_metadata,
            "previous_backbone_lr": self.previous_backbone_lr,
        }

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.previous_backbone_lr = state_dict["previous_backbone_lr"]
        super().load_state_dict(state_dict)

    @override
    def on_fit_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Validate backbone attribute; restore gradient state on resume.

        Raises:
            MisconfigurationException: If LightningModule has no nn.Module
                ``backbone`` attribute.
        """
        if hasattr(pl_module, "backbone") and isinstance(pl_module.backbone, Module):
            super().on_fit_start(trainer, pl_module)
            if self.previous_backbone_lr is not None:
                # Step-based deviation: backbone may have been
                # unfrozen mid-epoch before checkpoint.
                # freeze_before_training() in setup() re-froze it,
                # so re-enable gradients now past the unfreeze step.
                self.make_trainable(pl_module.backbone)
            return
        raise MisconfigurationException(
            "The LightningModule should have a nn.Module `backbone` attribute"
        )

    @override
    def freeze_before_training(self, pl_module: "pl.LightningModule") -> None:
        self.freeze(pl_module.backbone)

    @override
    def finetune_function(
        self,
        pl_module: "pl.LightningModule",
        epoch: int,
        optimizer: Optimizer,
    ) -> None:
        """No-op. LR scaling is applied per-step in on_train_batch_start."""
        return

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Unfreeze the backbone at the target global step, then scale LR.

        Args:
            trainer: The Lightning Trainer.
            pl_module: The LightningModule being trained.
            batch: The current batch.
            batch_idx: The index of the current batch.
        """
        optimizer = trainer.optimizers[0]

        if self.previous_backbone_lr is None:
            if trainer.global_step < self.unfreeze_step:
                return

            num_param_groups = len(optimizer.param_groups)
            current_lr = optimizer.param_groups[0]["lr"]
            initial_backbone_lr = (
                self.backbone_initial_lr
                if self.backbone_initial_lr is not None
                else current_lr * self.backbone_initial_ratio_lr
            )
            self.previous_backbone_lr = initial_backbone_lr
            self.unfreeze_and_add_param_group(
                pl_module.backbone,
                optimizer,
                initial_backbone_lr,
                train_bn=self.train_bn,
                initial_denom_lr=self.initial_denom_lr,
            )
            # Must call _store() here. BaseFinetuning.on_train_epoch_start()
            # already ran for this epoch and captured num_param_groups before
            # unfreeze. At the next epoch start it will see
            # num_param_groups == len(current) and be a no-op — the backbone
            # group would never be recorded without this.
            self._store(pl_module, 0, num_param_groups, optimizer.param_groups)
            if self.verbose:
                log.info(
                    "Current lr: %s, Backbone lr: %s",
                    round(current_lr, self.rounding),
                    round(initial_backbone_lr, self.rounding),
                )
        else:
            current_lr = optimizer.param_groups[0]["lr"]
            next_backbone_lr = (
                self.lambda_func(trainer.global_step) * self.previous_backbone_lr
            )
            next_backbone_lr = (
                current_lr
                if (self.should_align and next_backbone_lr > current_lr)
                else next_backbone_lr
            )
            optimizer.param_groups[-1]["lr"] = next_backbone_lr
            self.previous_backbone_lr = next_backbone_lr
            if self.verbose:
                log.info(
                    "Current lr: %s, Backbone lr: %s",
                    round(current_lr, self.rounding),
                    round(next_backbone_lr, self.rounding),
                )
