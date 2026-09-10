"""Unit tests for the BackboneFinetuningSteps callback."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from lightning.pytorch.utilities.exceptions import MisconfigurationException

from ultrabench.training.callbacks import BackboneFinetuningSteps

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_module_with_backbone() -> MagicMock:
    """Return a minimal LightningModule mock with a backbone attribute.

    The module's named_parameters() returns backbone params only. Use
    _make_module_and_optimizer() when _store() will be called (which needs ALL
    optimizer params to be present in named_parameters).
    """
    module = MagicMock()
    module.backbone = nn.Sequential(
        nn.Linear(4, 4),
        nn.Linear(4, 2),
    )
    module.named_parameters.return_value = list(module.backbone.named_parameters())
    return module


def _make_module_and_optimizer(
    lr: float = 1e-3,
) -> tuple[MagicMock, torch.optim.SGD]:
    """Return a (module, optimizer) pair suitable for tests that call _store().

    _store() maps every optimizer param to a name via named_parameters(). Both
    the head params (initial optimizer group) and backbone params must be
    present in named_parameters, or _store() will raise a KeyError.
    """
    backbone = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    head = nn.Linear(4, 2)

    module = MagicMock()
    module.backbone = backbone
    # All params (head + backbone) must be reachable via named_parameters
    all_named = list(head.named_parameters()) + [
        (f"backbone.{n}", p) for n, p in backbone.named_parameters()
    ]
    module.named_parameters.return_value = all_named

    optimizer = torch.optim.SGD(head.parameters(), lr=lr)
    return module, optimizer


def _make_bn_module_and_optimizer(
    lr: float = 1e-3,
) -> tuple[MagicMock, torch.optim.SGD]:
    """Return a (module, optimizer) pair whose backbone contains BatchNorm.

    Mirrors _make_module_and_optimizer, but the backbone also holds a
    BatchNorm and a Dropout layer so train_bn and train/eval mode handling
    can be exercised.
    """
    backbone = nn.Sequential(
        nn.Linear(4, 4),
        nn.BatchNorm1d(4),
        nn.Dropout(0.5),
        nn.Linear(4, 2),
    )
    head = nn.Linear(4, 2)

    module = MagicMock()
    module.backbone = backbone
    module.named_parameters.return_value = list(head.named_parameters()) + [
        (f"backbone.{n}", p) for n, p in backbone.named_parameters()
    ]

    optimizer = torch.optim.SGD(head.parameters(), lr=lr)
    return module, optimizer


def _bn_layer(backbone: nn.Module) -> nn.BatchNorm1d:
    """Return the single BatchNorm layer of a helper backbone."""
    return next(m for m in backbone.modules() if isinstance(m, nn.BatchNorm1d))


def _make_optimizer(lr: float = 1e-3) -> torch.optim.SGD:
    """Return an SGD optimizer over a single dummy parameter group."""
    param = nn.Parameter(torch.zeros(1))
    return torch.optim.SGD([param], lr=lr)


def _make_trainer(global_step: int = 0) -> MagicMock:
    """Return a minimal Trainer mock."""
    trainer = MagicMock()
    trainer.global_step = global_step
    return trainer


def _backbone_groups(optimizer: torch.optim.Optimizer, start: int = 1) -> list[dict]:
    """Return the optimizer's param groups from `start` onward."""
    return optimizer.param_groups[start:]


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------


class TestInit:
    """Tests for BackboneFinetuningSteps.__init__."""

    def test_default_params(self):
        """Default constructor parameters are stored correctly."""
        cb = BackboneFinetuningSteps()
        assert cb.unfreeze_step == 1000
        assert cb.backbone_max_lr_ratio == pytest.approx(0.1)
        assert cb.backbone_warmup_steps == 500
        assert cb.backbone_layer_decay == pytest.approx(0.75)
        assert cb.backbone_num_layer_groups == 4
        assert cb.train_bn is False
        assert cb.verbose is False
        assert cb.rounding == 12
        assert cb._unfrozen is False

    def test_custom_params(self):
        """Custom constructor parameters are stored correctly."""
        cb = BackboneFinetuningSteps(
            unfreeze_backbone_at_step=500,
            backbone_max_lr_ratio=0.01,
            backbone_warmup_steps=100,
            backbone_layer_decay=0.9,
            backbone_num_layer_groups=2,
            train_bn=True,
            verbose=True,
            rounding=6,
        )
        assert cb.unfreeze_step == 500
        assert cb.backbone_max_lr_ratio == pytest.approx(0.01)
        assert cb.backbone_warmup_steps == 100
        assert cb.backbone_layer_decay == pytest.approx(0.9)
        assert cb.backbone_num_layer_groups == 2
        assert cb.train_bn is True
        assert cb.verbose is True
        assert cb.rounding == 6


# ---------------------------------------------------------------------------
# TestFreezeBeforeTraining
# ---------------------------------------------------------------------------


class TestFreezeBeforeTraining:
    """Tests for BackboneFinetuningSteps.freeze_before_training."""

    def test_freezes_backbone(self):
        """All backbone parameters have requires_grad=False after freeze."""
        cb = BackboneFinetuningSteps()
        module = _make_module_with_backbone()
        cb.freeze_before_training(module)
        for p in module.backbone.parameters():
            assert not p.requires_grad

    def test_non_backbone_params_untouched(self):
        """Parameters not in the backbone remain trainable after freeze."""
        cb = BackboneFinetuningSteps()
        head_param = nn.Parameter(torch.zeros(4))

        module = MagicMock()
        module.backbone = nn.Linear(4, 2)
        # The head param is not part of backbone
        for p in module.backbone.parameters():
            pass  # force backbone params to exist

        cb.freeze_before_training(module)
        # head_param was never touched
        assert head_param.requires_grad

    def test_backbone_in_eval_mode_after_freeze(self):
        """The frozen backbone is switched to eval mode."""
        cb = BackboneFinetuningSteps()
        module, _ = _make_bn_module_and_optimizer()
        cb.freeze_before_training(module)
        assert not module.backbone.training

    def test_bn_fully_frozen_when_train_bn_false(self):
        """train_bn=False freezes BN params, stats and mode."""
        cb = BackboneFinetuningSteps(train_bn=False)
        module, _ = _make_bn_module_and_optimizer()
        cb.freeze_before_training(module)

        bn = _bn_layer(module.backbone)
        assert not bn.training
        assert not bn.track_running_stats
        assert all(not p.requires_grad for p in bn.parameters())

    def test_bn_left_in_train_mode_when_train_bn_true(self):
        """train_bn=True keeps BN layers training while the rest is frozen."""
        cb = BackboneFinetuningSteps(train_bn=True)
        module, _ = _make_bn_module_and_optimizer()
        cb.freeze_before_training(module)

        bn = _bn_layer(module.backbone)
        assert not module.backbone.training  # backbone itself is frozen
        assert bn.training
        assert bn.track_running_stats
        assert all(p.requires_grad for p in bn.parameters())


# ---------------------------------------------------------------------------
# TestOnFitStart
# ---------------------------------------------------------------------------


class TestOnFitStart:
    """Tests for BackboneFinetuningSteps.on_fit_start."""

    def test_raises_when_no_backbone(self):
        """MisconfigurationException raised when module has no backbone."""
        cb = BackboneFinetuningSteps()
        module = MagicMock(spec=[])  # no attributes
        trainer = _make_trainer()
        with pytest.raises(MisconfigurationException):
            cb.on_fit_start(trainer, module)

    def test_raises_when_backbone_not_nn_module(self):
        """MisconfigurationException raised when backbone is not nn.Module."""
        cb = BackboneFinetuningSteps()
        module = MagicMock()
        module.backbone = "not_a_module"
        trainer = _make_trainer()
        with pytest.raises(MisconfigurationException):
            cb.on_fit_start(trainer, module)

    def test_no_exception_with_valid_backbone(self):
        """No exception raised when backbone is a valid nn.Module."""
        cb = BackboneFinetuningSteps()
        module = _make_module_with_backbone()
        trainer = _make_trainer()
        optimizer = _make_optimizer()
        trainer.optimizers = [optimizer]
        # Patch super().on_fit_start so we don't need full Trainer state
        with patch.object(BackboneFinetuningSteps.__bases__[0], "on_fit_start"):
            cb.on_fit_start(trainer, module)

    def test_rebuilds_param_groups_on_resume(self):
        """Backbone param groups are rebuilt when resuming past unfreeze."""
        cb = BackboneFinetuningSteps(backbone_num_layer_groups=2)
        cb._unfrozen = True  # simulate resumed-from-checkpoint state

        module, optimizer = _make_module_and_optimizer()
        # Freeze backbone first (as freeze_before_training would have done)
        for p in module.backbone.parameters():
            p.requires_grad_(False)

        trainer = _make_trainer()
        trainer.optimizers = [optimizer]
        with patch.object(BackboneFinetuningSteps.__bases__[0], "on_fit_start"):
            cb.on_fit_start(trainer, module)

        for p in module.backbone.parameters():
            assert p.requires_grad
        assert len(optimizer.param_groups) == 1 + 2  # head + 2 layer groups


# ---------------------------------------------------------------------------
# TestFinetuneFunctionLrScaling
# ---------------------------------------------------------------------------


class TestFinetuneFunctionLrScaling:
    """Tests for BackboneFinetuningSteps.finetune_function."""

    def test_no_op(self):
        """finetune_function is always a no-op; LR scaling happens per-step."""
        cb = BackboneFinetuningSteps()
        optimizer = _make_optimizer(lr=1e-3)
        optimizer.add_param_group(
            {
                "params": [nn.Parameter(torch.zeros(1))],
                "lr": 5e-4,
            }
        )
        module = MagicMock()
        cb.finetune_function(module, epoch=0, optimizer=optimizer)
        assert optimizer.param_groups[-1]["lr"] == pytest.approx(5e-4)


# ---------------------------------------------------------------------------
# TestOnTrainBatchStart
# ---------------------------------------------------------------------------


class TestOnTrainBatchStart:
    """Tests for BackboneFinetuningSteps.on_train_batch_start."""

    def _run_batch_start(
        self,
        cb: BackboneFinetuningSteps,
        global_step: int,
        module: MagicMock,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Call on_train_batch_start with a mock trainer."""
        trainer = _make_trainer(global_step=global_step)
        trainer.optimizers = [optimizer]
        cb.on_train_batch_start(trainer, module, batch={}, batch_idx=0)

    def test_no_op_before_unfreeze_step(self):
        """No param group is added before the unfreeze step."""
        cb = BackboneFinetuningSteps(unfreeze_backbone_at_step=100)
        module, optimizer = _make_module_and_optimizer()
        self._run_batch_start(cb, global_step=50, module=module, optimizer=optimizer)
        assert len(optimizer.param_groups) == 1
        assert cb._unfrozen is False

    def test_unfreezes_at_unfreeze_step(self):
        """Backbone is unfrozen and new param groups added at target step."""
        cb = BackboneFinetuningSteps(
            unfreeze_backbone_at_step=100, backbone_num_layer_groups=4
        )
        module, optimizer = _make_module_and_optimizer(lr=1e-3)
        for p in module.backbone.parameters():
            p.requires_grad_(False)
        self._run_batch_start(cb, global_step=100, module=module, optimizer=optimizer)
        # backbone (2 Linear layers = 4 tensors) split into 4 groups
        assert len(optimizer.param_groups) == 1 + 4
        for p in module.backbone.parameters():
            assert p.requires_grad

    def test_unfreezes_only_once(self):
        """on_train_batch_start twice at step >= threshold adds groups once."""
        cb = BackboneFinetuningSteps(
            unfreeze_backbone_at_step=100, backbone_num_layer_groups=2
        )
        module, optimizer = _make_module_and_optimizer()
        for p in module.backbone.parameters():
            p.requires_grad_(False)
        self._run_batch_start(cb, global_step=100, module=module, optimizer=optimizer)
        self._run_batch_start(cb, global_step=101, module=module, optimizer=optimizer)
        assert len(optimizer.param_groups) == 1 + 2

    def test_backbone_groups_capped_below_head_lr(self):
        """Backbone group LRs never exceed head_lr * backbone_max_lr_ratio."""
        ratio = 0.1
        cb = BackboneFinetuningSteps(
            unfreeze_backbone_at_step=0,
            backbone_max_lr_ratio=ratio,
            backbone_warmup_steps=0,
            backbone_layer_decay=1.0,
            backbone_num_layer_groups=2,
        )
        head_lr = 1e-2
        module, optimizer = _make_module_and_optimizer(lr=head_lr)
        self._run_batch_start(cb, global_step=0, module=module, optimizer=optimizer)
        for group in _backbone_groups(optimizer):
            assert group["lr"] == pytest.approx(head_lr * ratio)

    def test_head_group_untouched(self):
        """Group 0 (head) LR is never modified by the callback."""
        cb = BackboneFinetuningSteps(
            unfreeze_backbone_at_step=0, backbone_warmup_steps=0
        )
        head_lr = 1e-2
        module, optimizer = _make_module_and_optimizer(lr=head_lr)
        self._run_batch_start(cb, global_step=0, module=module, optimizer=optimizer)
        self._run_batch_start(cb, global_step=1, module=module, optimizer=optimizer)
        assert optimizer.param_groups[0]["lr"] == pytest.approx(head_lr)

    def test_linear_warmup_ramps_from_zero(self):
        """Backbone LR ramps linearly from 0 to target over warmup_steps."""
        ratio = 0.1
        warmup_steps = 10
        cb = BackboneFinetuningSteps(
            unfreeze_backbone_at_step=0,
            backbone_max_lr_ratio=ratio,
            backbone_warmup_steps=warmup_steps,
            backbone_layer_decay=1.0,
            backbone_num_layer_groups=1,
        )
        head_lr = 1e-2
        module, optimizer = _make_module_and_optimizer(lr=head_lr)

        self._run_batch_start(cb, global_step=0, module=module, optimizer=optimizer)
        assert optimizer.param_groups[-1]["lr"] == pytest.approx(0.0)

        self._run_batch_start(cb, global_step=5, module=module, optimizer=optimizer)
        assert optimizer.param_groups[-1]["lr"] == pytest.approx(head_lr * ratio * 0.5)

        self._run_batch_start(
            cb, global_step=warmup_steps, module=module, optimizer=optimizer
        )
        assert optimizer.param_groups[-1]["lr"] == pytest.approx(head_lr * ratio)

        # Warmup factor is clamped at 1.0 past warmup_steps.
        self._run_batch_start(
            cb, global_step=warmup_steps + 100, module=module, optimizer=optimizer
        )
        assert optimizer.param_groups[-1]["lr"] == pytest.approx(head_lr * ratio)

    def test_per_layer_decay_scales_shallow_groups_lower(self):
        """Shallower layer groups get a smaller LR than deeper ones."""
        decay = 0.5
        cb = BackboneFinetuningSteps(
            unfreeze_backbone_at_step=0,
            backbone_max_lr_ratio=1.0,
            backbone_warmup_steps=0,
            backbone_layer_decay=decay,
            backbone_num_layer_groups=4,
        )
        head_lr = 1.0
        module, optimizer = _make_module_and_optimizer(lr=head_lr)
        self._run_batch_start(cb, global_step=0, module=module, optimizer=optimizer)

        backbone_lrs = [group["lr"] for group in _backbone_groups(optimizer)]
        assert backbone_lrs == pytest.approx([decay**3, decay**2, decay**1, decay**0])

    def test_layer_decay_one_disables_per_layer_scaling(self):
        """backbone_layer_decay=1.0 gives every group the same LR."""
        cb = BackboneFinetuningSteps(
            unfreeze_backbone_at_step=0,
            backbone_max_lr_ratio=0.2,
            backbone_warmup_steps=0,
            backbone_layer_decay=1.0,
            backbone_num_layer_groups=4,
        )
        head_lr = 1e-2
        module, optimizer = _make_module_and_optimizer(lr=head_lr)
        self._run_batch_start(cb, global_step=0, module=module, optimizer=optimizer)

        backbone_lrs = [group["lr"] for group in _backbone_groups(optimizer)]
        assert backbone_lrs == pytest.approx([head_lr * 0.2] * 4)

    def test_num_layer_groups_capped_by_param_count(self):
        """Requesting more groups than params yields one group per param."""
        cb = BackboneFinetuningSteps(
            unfreeze_backbone_at_step=0, backbone_num_layer_groups=100
        )
        module, optimizer = _make_module_and_optimizer()
        self._run_batch_start(cb, global_step=0, module=module, optimizer=optimizer)
        # backbone has 4 param tensors (2 Linear layers)
        assert len(optimizer.param_groups) == 1 + 4

    def test_backbone_lr_tracks_decaying_head_lr(self):
        """Backbone LR follows the head's LR if it changes after unfreeze."""
        ratio = 0.1
        cb = BackboneFinetuningSteps(
            unfreeze_backbone_at_step=0,
            backbone_max_lr_ratio=ratio,
            backbone_warmup_steps=0,
            backbone_layer_decay=1.0,
            backbone_num_layer_groups=1,
        )
        module, optimizer = _make_module_and_optimizer(lr=1e-2)
        self._run_batch_start(cb, global_step=0, module=module, optimizer=optimizer)
        assert optimizer.param_groups[-1]["lr"] == pytest.approx(1e-2 * ratio)

        # Simulate the head's LR scheduler decaying the head LR.
        optimizer.param_groups[0]["lr"] = 1e-3
        self._run_batch_start(cb, global_step=1, module=module, optimizer=optimizer)
        assert optimizer.param_groups[-1]["lr"] == pytest.approx(1e-3 * ratio)

    def test_store_records_backbone_groups(self):
        """_internal_optimizer_metadata is populated with all param groups."""
        cb = BackboneFinetuningSteps(
            unfreeze_backbone_at_step=0, backbone_num_layer_groups=2
        )
        module, optimizer = _make_module_and_optimizer()
        for p in module.backbone.parameters():
            p.requires_grad_(False)
        self._run_batch_start(cb, global_step=0, module=module, optimizer=optimizer)
        assert 0 in cb._internal_optimizer_metadata
        assert len(cb._internal_optimizer_metadata[0]) == 1 + 2


# ---------------------------------------------------------------------------
# TestBackboneTrainEvalMode
# ---------------------------------------------------------------------------


class TestBackboneTrainEvalMode:
    """Tests that the callback owns the backbone's train/eval mode."""

    def _run_batch_start(
        self,
        cb: BackboneFinetuningSteps,
        global_step: int,
        module: MagicMock,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Call on_train_batch_start with a mock trainer."""
        trainer = _make_trainer(global_step=global_step)
        trainer.optimizers = [optimizer]
        cb.on_train_batch_start(trainer, module, batch={}, batch_idx=0)

    def test_backbone_kept_in_eval_before_unfreeze(self):
        """The backbone stays in eval mode while frozen."""
        cb = BackboneFinetuningSteps(unfreeze_backbone_at_step=100)
        module, optimizer = _make_bn_module_and_optimizer()
        cb.freeze_before_training(module)

        self._run_batch_start(cb, global_step=50, module=module, optimizer=optimizer)
        assert not module.backbone.training

    def test_mode_reasserted_after_external_change(self):
        """Eval mode is restored after something else flips it back on.

        Lightning's evaluation loop captures and restores submodule modes
        directly, so the callback must re-assert the mode every batch.
        """
        cb = BackboneFinetuningSteps(unfreeze_backbone_at_step=100)
        module, optimizer = _make_bn_module_and_optimizer()
        cb.freeze_before_training(module)

        module.backbone.train()  # simulate _ModuleMode.restore
        self._run_batch_start(cb, global_step=51, module=module, optimizer=optimizer)
        assert not module.backbone.training

    def test_backbone_in_train_mode_after_unfreeze(self):
        """The backbone switches to train mode once unfrozen."""
        cb = BackboneFinetuningSteps(
            unfreeze_backbone_at_step=100, backbone_num_layer_groups=2
        )
        module, optimizer = _make_bn_module_and_optimizer()
        cb.freeze_before_training(module)

        self._run_batch_start(cb, global_step=100, module=module, optimizer=optimizer)
        assert module.backbone.training

    def test_bn_stays_frozen_after_unfreeze_when_train_bn_false(self):
        """train_bn=False keeps BN frozen even after the backbone unfreezes."""
        cb = BackboneFinetuningSteps(
            unfreeze_backbone_at_step=100,
            backbone_num_layer_groups=2,
            train_bn=False,
        )
        module, optimizer = _make_bn_module_and_optimizer()
        cb.freeze_before_training(module)

        self._run_batch_start(cb, global_step=100, module=module, optimizer=optimizer)

        bn = _bn_layer(module.backbone)
        assert module.backbone.training
        assert not bn.training
        assert not bn.track_running_stats
        assert all(not p.requires_grad for p in bn.parameters())

        grouped = {id(p) for group in optimizer.param_groups for p in group["params"]}
        assert all(id(p) not in grouped for p in bn.parameters())

    def test_bn_trained_after_unfreeze_when_train_bn_true(self):
        """train_bn=True trains BN alongside the rest of the backbone."""
        cb = BackboneFinetuningSteps(
            unfreeze_backbone_at_step=100,
            backbone_num_layer_groups=2,
            train_bn=True,
        )
        module, optimizer = _make_bn_module_and_optimizer()
        cb.freeze_before_training(module)

        self._run_batch_start(cb, global_step=100, module=module, optimizer=optimizer)

        bn = _bn_layer(module.backbone)
        assert bn.training
        assert bn.track_running_stats

        grouped = {id(p) for group in optimizer.param_groups for p in group["params"]}
        assert all(id(p) in grouped for p in bn.parameters())


# ---------------------------------------------------------------------------
# TestStateDict
# ---------------------------------------------------------------------------


class TestStateDict:
    """Tests for BackboneFinetuningSteps.state_dict and load_state_dict."""

    def test_state_dict_has_unfrozen_flag(self):
        """unfrozen is False in state_dict before unfreezing."""
        cb = BackboneFinetuningSteps()
        sd = cb.state_dict()
        assert sd["unfrozen"] is False

    def test_state_dict_reflects_unfrozen_state(self):
        """unfrozen is True in state_dict after unfreezing."""
        cb = BackboneFinetuningSteps()
        cb._unfrozen = True
        sd = cb.state_dict()
        assert sd["unfrozen"] is True

    def test_state_dict_includes_optimizer_metadata(self):
        """internal_optimizer_metadata is included in state_dict."""
        cb = BackboneFinetuningSteps()
        cb._internal_optimizer_metadata = {0: [{"lr": 1e-3}]}
        sd = cb.state_dict()
        assert sd["internal_optimizer_metadata"] == {0: [{"lr": 1e-3}]}

    def test_unfrozen_flag_round_trips(self):
        """unfrozen flag survives a state_dict round-trip."""
        cb1 = BackboneFinetuningSteps()
        cb1._unfrozen = True
        cb1._internal_optimizer_metadata = {}

        cb2 = BackboneFinetuningSteps()
        cb2.load_state_dict(cb1.state_dict())
        assert cb2._unfrozen is True

    def test_load_state_dict_sets_restarting(self):
        """_restarting is set True after load_state_dict."""
        cb = BackboneFinetuningSteps()
        cb.load_state_dict(
            {
                "unfrozen": False,
                "internal_optimizer_metadata": {},
            }
        )
        assert cb._restarting is True


# ---------------------------------------------------------------------------
# TestBackboneProperty
# ---------------------------------------------------------------------------


class TestBackboneProperty:
    """Tests for UltraBenchModule.backbone attribute."""

    def _make_ultrabench_module(self):
        """Return a minimal UltraBenchModule with a real backbone and head."""
        from ultrabench.training.lightning import UltraBenchModule

        backbone = nn.Linear(4, 4)
        head = nn.Linear(4, 2)

        return UltraBenchModule(
            backbone=backbone,
            head=head,
            criterion=nn.CrossEntropyLoss(),
            optimizer_cfg=MagicMock(),
            scheduler_cfg=MagicMock(),
            task_type="classification",
            output_dir="/tmp",
            num_classes=2,
        ), backbone

    def test_backbone_returns_encoder(self):
        """UltraBenchModule.backbone is the backbone passed at construction."""
        module, backbone = self._make_ultrabench_module()
        assert module.backbone is backbone

    def test_backbone_is_nn_module(self):
        """UltraBenchModule.backbone is an nn.Module instance."""
        module, _ = self._make_ultrabench_module()
        assert isinstance(module.backbone, nn.Module)


# ---------------------------------------------------------------------------
# TestConfigureOptimizersExcludesBackbone
# ---------------------------------------------------------------------------


class TestConfigureOptimizersExcludesBackbone:
    """Regression tests for the configure_optimizers backbone-isolation fix."""

    def test_backbone_params_excluded_from_initial_optimizer(self):
        """configure_optimizers builds the optimizer over head params only."""
        from omegaconf import OmegaConf

        from ultrabench.training.lightning import UltraBenchModule

        backbone = nn.Linear(4, 4)
        head = nn.Linear(4, 2)

        optimizer_cfg = OmegaConf.create({"_target_": "torch.optim.AdamW", "lr": 1e-3})
        scheduler_cfg = OmegaConf.create(
            {
                "_target_": "torch.optim.lr_scheduler.ConstantLR",
                "factor": 1.0,
            }
        )

        module = UltraBenchModule(
            backbone=backbone,
            head=head,
            criterion=nn.CrossEntropyLoss(),
            optimizer_cfg=optimizer_cfg,
            scheduler_cfg=scheduler_cfg,
            task_type="classification",
            output_dir="/tmp",
            num_classes=2,
        )

        result = module.configure_optimizers()
        optimizer = result["optimizer"]

        assert len(optimizer.param_groups) == 1
        optimized_params = {id(p) for p in optimizer.param_groups[0]["params"]}
        backbone_param_ids = {id(p) for p in backbone.parameters()}
        head_param_ids = {id(p) for p in head.parameters()}

        assert optimized_params == head_param_ids
        assert optimized_params.isdisjoint(backbone_param_ids)
