"""Integration tests for BackboneFinetuningSteps against a real Trainer.

The unit tests in test_backbone_finetuning_steps.py drive the callback with
mock trainers, which cannot reproduce how Lightning manages module train/eval
modes during fit (its evaluation loop captures and restores submodule modes
directly rather than calling LightningModule.train()). These tests run a short
real training loop instead.
"""

from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.utils.data import DataLoader, TensorDataset

from ultrabench2.training.callbacks import BackboneFinetuningSteps

MAX_STEPS = 16
UNFREEZE_STEP = 8
VAL_CHECK_INTERVAL = 6
BATCH_SIZE = 8


class _TinyModule(pl.LightningModule):
    """Minimal stand-in for UltraBenchModule with a backbone and a head."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(4, 4),
            nn.BatchNorm1d(4),
            nn.Dropout(0.5),
            nn.Linear(4, 4),
        )
        self.head = nn.Linear(4, 2)
        self.rows: list[dict[str, Any]] = []

    @property
    def bn(self) -> nn.BatchNorm1d:
        """Return the backbone's BatchNorm layer."""
        layer = self.backbone[1]
        assert isinstance(layer, nn.BatchNorm1d)
        return layer

    @property
    def bn_running_mean(self) -> torch.Tensor:
        """Return the BatchNorm layer's running mean."""
        running_mean = self.bn.running_mean
        assert running_mean is not None
        return running_mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run backbone then head.

        Args:
            x: Input tensor.

        Returns:
            Logits.
        """
        return self.head(self.backbone(x))

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Record module modes, then compute the loss.

        Args:
            batch: Tuple of (inputs, targets).
            batch_idx: Index of the current batch.

        Returns:
            Scalar loss tensor.
        """
        x, y = batch
        self.rows.append(
            {
                "step": self.trainer.global_step,
                "backbone_training": self.backbone.training,
                "bn_training": self.bn.training,
                "bn_running_mean": self.bn_running_mean.clone(),
                "num_groups": len(self.trainer.optimizers[0].param_groups),
            }
        )
        return nn.functional.cross_entropy(self(x), y)

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        """Compute the validation loss.

        Args:
            batch: Tuple of (inputs, targets).
            batch_idx: Index of the current batch.
        """
        x, y = batch
        self.log("val_loss", nn.functional.cross_entropy(self(x), y))

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Build a head-only optimizer with a per-step cosine schedule.

        Mirrors UltraBenchModule.configure_optimizers.

        Returns:
            Dict with the optimizer and its LR scheduler config.
        """
        head_params = [
            p for n, p in self.named_parameters() if not n.startswith("backbone.")
        ]
        optimizer = torch.optim.AdamW(head_params, lr=1e-3, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_STEPS
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def _loader() -> DataLoader:
    """Return a small deterministic dataloader."""
    inputs = torch.randn(64, 4)
    targets = torch.randint(0, 2, (64,))
    return DataLoader(TensorDataset(inputs, targets), batch_size=BATCH_SIZE)


def _run(train_bn: bool, unfreeze_step: int = UNFREEZE_STEP) -> _TinyModule:
    """Train a _TinyModule for a few steps with the callback attached.

    Args:
        train_bn: Value of the callback's train_bn flag.
        unfreeze_step: Step at which the backbone unfreezes. Set beyond
            max_steps to reproduce the head-only protocol.

    Returns:
        The trained module, holding its per-step records.
    """
    pl.seed_everything(42, workers=True)
    module = _TinyModule()
    trainer = pl.Trainer(
        max_epochs=-1,
        max_steps=MAX_STEPS,
        check_val_every_n_epoch=None,
        val_check_interval=VAL_CHECK_INTERVAL,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[
            BackboneFinetuningSteps(
                unfreeze_backbone_at_step=unfreeze_step,
                backbone_warmup_steps=4,
                backbone_num_layer_groups=2,
                train_bn=train_bn,
            )
        ],
    )
    trainer.fit(module, _loader(), _loader())
    return module


class TestBackboneModeDuringFit:
    """The backbone's train/eval mode tracks its frozen state during fit."""

    def test_frozen_backbone_stays_in_eval_mode(self):
        """The backbone is in eval mode at every step before unfreezing."""
        module = _run(train_bn=False)
        frozen = [r for r in module.rows if r["step"] < UNFREEZE_STEP]
        assert frozen, "no steps recorded before the unfreeze step"
        # A validation run happens mid-way; its mode restore must not leak.
        assert max(r["step"] for r in frozen) >= VAL_CHECK_INTERVAL
        assert all(not r["backbone_training"] for r in frozen)

    def test_backbone_trains_after_unfreeze(self):
        """The backbone is in train mode at every step after unfreezing."""
        module = _run(train_bn=False)
        unfrozen = [r for r in module.rows if r["step"] >= UNFREEZE_STEP]
        assert unfrozen, "no steps recorded after the unfreeze step"
        assert max(r["step"] for r in unfrozen) > VAL_CHECK_INTERVAL * 2
        assert all(r["backbone_training"] for r in unfrozen)
        assert unfrozen[-1]["num_groups"] == 1 + 2

    def test_bn_stats_frozen_for_whole_run_when_train_bn_false(self):
        """BN never leaves eval mode, so its running stats never move."""
        module = _run(train_bn=False)
        assert all(not r["bn_training"] for r in module.rows)
        for row in module.rows:
            assert torch.equal(row["bn_running_mean"], torch.zeros(4))
        assert torch.equal(module.bn_running_mean, torch.zeros(4))

    def test_bn_stats_track_data_when_train_bn_true(self):
        """train_bn=True keeps BN in train mode, so its stats update."""
        module = _run(train_bn=True)
        assert all(r["bn_training"] for r in module.rows)
        assert not torch.equal(module.bn_running_mean, torch.zeros(4))
        # The rest of the backbone still follows the frozen state.
        frozen = [r for r in module.rows if r["step"] < UNFREEZE_STEP]
        assert all(not r["backbone_training"] for r in frozen)

    def test_head_only_protocol_never_leaves_eval_mode(self):
        """With no unfreeze step, the backbone stays frozen and in eval mode.

        Mirrors head_only mode, where unfreeze_backbone_at_step resolves to
        max_steps + 1.
        """
        module = _run(train_bn=False, unfreeze_step=MAX_STEPS + 1)
        assert all(not r["backbone_training"] for r in module.rows)
        assert all(r["num_groups"] == 1 for r in module.rows)
        assert all(not p.requires_grad for p in module.backbone.parameters())
