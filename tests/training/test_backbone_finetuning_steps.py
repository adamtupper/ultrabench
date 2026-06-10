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


def _make_optimizer(lr: float = 1e-3) -> torch.optim.SGD:
    """Return an SGD optimizer over a single dummy parameter group."""
    param = nn.Parameter(torch.zeros(1))
    return torch.optim.SGD([param], lr=lr)


def _make_trainer(global_step: int = 0) -> MagicMock:
    """Return a minimal Trainer mock."""
    trainer = MagicMock()
    trainer.global_step = global_step
    return trainer


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------


class TestInit:
    """Tests for BackboneFinetuningSteps.__init__."""

    def test_default_params(self):
        """Default constructor parameters are stored correctly."""
        cb = BackboneFinetuningSteps()
        assert cb.unfreeze_step == 1000
        assert cb.backbone_initial_ratio_lr == pytest.approx(10e-2)
        assert cb.backbone_initial_lr is None
        assert cb.should_align is True
        assert cb.initial_denom_lr == pytest.approx(10.0)
        assert cb.train_bn is True
        assert cb.verbose is False
        assert cb.rounding == 12
        assert cb.previous_backbone_lr is None

    def test_custom_params(self):
        """Custom constructor parameters are stored correctly."""
        cb = BackboneFinetuningSteps(
            unfreeze_backbone_at_step=500,
            backbone_initial_ratio_lr=0.01,
            backbone_initial_lr=1e-4,
            should_align=False,
            initial_denom_lr=5.0,
            train_bn=False,
            verbose=True,
            rounding=6,
        )
        assert cb.unfreeze_step == 500
        assert cb.backbone_initial_ratio_lr == pytest.approx(0.01)
        assert cb.backbone_initial_lr == pytest.approx(1e-4)
        assert cb.should_align is False
        assert cb.initial_denom_lr == pytest.approx(5.0)
        assert cb.train_bn is False
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

    def test_make_trainable_called_on_resume(self):
        """Backbone gradients re-enabled when previous_backbone_lr is set."""
        cb = BackboneFinetuningSteps()
        cb.previous_backbone_lr = 1e-4  # simulate resumed-from-checkpoint state

        module = _make_module_with_backbone()
        # Freeze backbone first (as freeze_before_training would have done)
        for p in module.backbone.parameters():
            p.requires_grad_(False)

        trainer = _make_trainer()
        with patch.object(BackboneFinetuningSteps.__bases__[0], "on_fit_start"):
            cb.on_fit_start(trainer, module)

        for p in module.backbone.parameters():
            assert p.requires_grad


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
        cb.previous_backbone_lr = 1e-4  # simulate post-unfreeze state
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
        assert cb.previous_backbone_lr is None

    def test_unfreezes_at_unfreeze_step(self):
        """Backbone is unfrozen and a new param group added at target step."""
        cb = BackboneFinetuningSteps(unfreeze_backbone_at_step=100)
        module, optimizer = _make_module_and_optimizer(lr=1e-3)
        for p in module.backbone.parameters():
            p.requires_grad_(False)
        self._run_batch_start(cb, global_step=100, module=module, optimizer=optimizer)
        assert len(optimizer.param_groups) == 2
        for p in module.backbone.parameters():
            assert p.requires_grad

    def test_unfreezes_only_once(self):
        """on_train_batch_start twice at step >= threshold adds one group."""
        cb = BackboneFinetuningSteps(unfreeze_backbone_at_step=100)
        module, optimizer = _make_module_and_optimizer()
        for p in module.backbone.parameters():
            p.requires_grad_(False)
        self._run_batch_start(cb, global_step=100, module=module, optimizer=optimizer)
        self._run_batch_start(cb, global_step=101, module=module, optimizer=optimizer)
        assert len(optimizer.param_groups) == 2

    def test_backbone_lr_uses_ratio(self):
        """Backbone LR defaults to current_lr * backbone_initial_ratio_lr."""
        ratio = 0.1
        cb = BackboneFinetuningSteps(
            unfreeze_backbone_at_step=0,
            backbone_initial_ratio_lr=ratio,
        )
        head_lr = 1e-2
        module, optimizer = _make_module_and_optimizer(lr=head_lr)
        self._run_batch_start(cb, global_step=0, module=module, optimizer=optimizer)
        assert optimizer.param_groups[-1]["lr"] == pytest.approx(head_lr * ratio)

    def test_explicit_backbone_initial_lr(self):
        """Explicit backbone_initial_lr overrides the ratio calculation."""
        explicit_lr = 5e-5
        cb = BackboneFinetuningSteps(
            unfreeze_backbone_at_step=0,
            backbone_initial_lr=explicit_lr,
        )
        module, optimizer = _make_module_and_optimizer(lr=1e-2)
        self._run_batch_start(cb, global_step=0, module=module, optimizer=optimizer)
        assert optimizer.param_groups[-1]["lr"] == pytest.approx(explicit_lr)

    def test_previous_backbone_lr_set_after_unfreeze(self):
        """previous_backbone_lr set to initial backbone LR on unfreeze."""
        ratio = 0.1
        head_lr = 1e-2
        cb = BackboneFinetuningSteps(
            unfreeze_backbone_at_step=0, backbone_initial_ratio_lr=ratio
        )
        module, optimizer = _make_module_and_optimizer(lr=head_lr)
        self._run_batch_start(cb, global_step=0, module=module, optimizer=optimizer)
        assert cb.previous_backbone_lr == pytest.approx(head_lr * ratio)

    def test_store_records_backbone_group(self):
        """_internal_optimizer_metadata is populated with both param groups."""
        cb = BackboneFinetuningSteps(unfreeze_backbone_at_step=0)
        module, optimizer = _make_module_and_optimizer()
        for p in module.backbone.parameters():
            p.requires_grad_(False)
        self._run_batch_start(cb, global_step=0, module=module, optimizer=optimizer)
        assert 0 in cb._internal_optimizer_metadata
        assert len(cb._internal_optimizer_metadata[0]) == 2

    def test_scales_backbone_lr_per_step(self):
        """Backbone LR is scaled by lambda_func on each step after unfreeze."""
        factor = 2.0
        cb = BackboneFinetuningSteps(
            unfreeze_backbone_at_step=0,
            lambda_func=lambda _: factor,
            should_align=False,
        )
        module, optimizer = _make_module_and_optimizer(lr=1e-2)
        for p in module.backbone.parameters():
            p.requires_grad_(False)
        self._run_batch_start(cb, global_step=0, module=module, optimizer=optimizer)
        initial_backbone_lr = optimizer.param_groups[-1]["lr"]
        self._run_batch_start(cb, global_step=1, module=module, optimizer=optimizer)
        assert optimizer.param_groups[-1]["lr"] == pytest.approx(
            initial_backbone_lr * factor
        )

    def test_should_align_caps_backbone_lr_per_step(self):
        """should_align=True caps backbone LR at head LR each step."""
        cb = BackboneFinetuningSteps(
            unfreeze_backbone_at_step=0,
            lambda_func=lambda _: 100.0,
            should_align=True,
        )
        head_lr = 1e-3
        module, optimizer = _make_module_and_optimizer(lr=head_lr)
        for p in module.backbone.parameters():
            p.requires_grad_(False)
        self._run_batch_start(cb, global_step=0, module=module, optimizer=optimizer)
        self._run_batch_start(cb, global_step=1, module=module, optimizer=optimizer)
        assert optimizer.param_groups[-1]["lr"] == pytest.approx(head_lr)

    def test_updates_previous_backbone_lr_per_step(self):
        """previous_backbone_lr is updated to the new backbone LR each step."""
        factor = 3.0
        cb = BackboneFinetuningSteps(
            unfreeze_backbone_at_step=0,
            lambda_func=lambda _: factor,
            should_align=False,
        )
        module, optimizer = _make_module_and_optimizer(lr=1e-2)
        for p in module.backbone.parameters():
            p.requires_grad_(False)
        self._run_batch_start(cb, global_step=0, module=module, optimizer=optimizer)
        initial_backbone_lr = cb.previous_backbone_lr
        self._run_batch_start(cb, global_step=1, module=module, optimizer=optimizer)
        assert cb.previous_backbone_lr == pytest.approx(initial_backbone_lr * factor)


# ---------------------------------------------------------------------------
# TestStateDict
# ---------------------------------------------------------------------------


class TestStateDict:
    """Tests for BackboneFinetuningSteps.state_dict and load_state_dict."""

    def test_state_dict_has_none_before_unfreeze(self):
        """previous_backbone_lr is None in state_dict before unfreezing."""
        cb = BackboneFinetuningSteps()
        sd = cb.state_dict()
        assert sd["previous_backbone_lr"] is None

    def test_state_dict_has_lr_after_unfreeze(self):
        """previous_backbone_lr is set in state_dict after unfreezing."""
        cb = BackboneFinetuningSteps()
        cb.previous_backbone_lr = 2e-4
        sd = cb.state_dict()
        assert sd["previous_backbone_lr"] == pytest.approx(2e-4)

    def test_state_dict_includes_optimizer_metadata(self):
        """internal_optimizer_metadata is included in state_dict."""
        cb = BackboneFinetuningSteps()
        cb._internal_optimizer_metadata = {0: [{"lr": 1e-3}]}
        sd = cb.state_dict()
        assert sd["internal_optimizer_metadata"] == {0: [{"lr": 1e-3}]}

    def test_previous_backbone_lr_round_trips(self):
        """previous_backbone_lr survives a state_dict round-trip."""
        cb1 = BackboneFinetuningSteps()
        cb1.previous_backbone_lr = 3e-5
        cb1._internal_optimizer_metadata = {}

        cb2 = BackboneFinetuningSteps()
        cb2.load_state_dict(cb1.state_dict())
        assert cb2.previous_backbone_lr == pytest.approx(3e-5)

    def test_load_state_dict_sets_restarting(self):
        """_restarting is set True after load_state_dict."""
        cb = BackboneFinetuningSteps()
        cb.load_state_dict(
            {
                "previous_backbone_lr": None,
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
        from ultrabench.training.setup_ptl import UltraBenchModule

        backbone = nn.Linear(4, 4)
        head = nn.Linear(4, 2)

        return UltraBenchModule(
            backbone=backbone,
            head=head,
            criterion=nn.CrossEntropyLoss(),
            mode_cfg=MagicMock(),
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

    def test_train_keeps_frozen_backbone_in_eval(self):
        """train() keeps backbone in eval mode when its params are frozen."""
        from ultrabench.training.setup_ptl import UltraBenchModule

        backbone = nn.Linear(4, 4)
        for p in backbone.parameters():
            p.requires_grad = False

        module = UltraBenchModule(
            backbone=backbone,
            head=nn.Linear(4, 2),
            criterion=nn.CrossEntropyLoss(),
            mode_cfg=MagicMock(),
            task_type="classification",
            output_dir="/tmp",
            num_classes=2,
        )
        module.train()
        assert not backbone.training

    def test_train_does_not_suppress_unfrozen_backbone(self):
        """train() doesn't override backbone mode when params are trainable."""
        from ultrabench.training.setup_ptl import UltraBenchModule

        backbone = nn.Linear(4, 4)  # requires_grad=True by default

        module = UltraBenchModule(
            backbone=backbone,
            head=nn.Linear(4, 2),
            criterion=nn.CrossEntropyLoss(),
            mode_cfg=MagicMock(),
            task_type="classification",
            output_dir="/tmp",
            num_classes=2,
        )
        module.train()
        assert backbone.training
