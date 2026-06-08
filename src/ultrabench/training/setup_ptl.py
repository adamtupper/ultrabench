import csv
import logging
import os
from typing import Any, Dict, List, Optional

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_class, instantiate
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchmetrics.segmentation import DiceScore, MeanIoU

from ultrabench.core.rle import mask_to_rle

from .data.transforms import get_transforms
from .data.utils import get_splits
from .methods import DownstreamWrapper, FullFinetuningWrapper

logger = logging.getLogger(__name__)


class UltraBenchDataModule(pl.LightningDataModule):
    """
    Wraps dataset loading and DataLoader construction for all splits.
    """

    def __init__(self, cfg: Any, mode_cfg: Any, data_dir: str, task_type: str):
        super().__init__()
        self.cfg = cfg
        self.mode_cfg = mode_cfg
        self.data_dir = data_dir
        self.task_type = task_type
        self.batch_size = cfg.optimization.batch_size
        self.num_workers = cfg.system.data_loader_workers
        self.train_transform = get_transforms(task_type, "train", cfg, mode_cfg)
        self.test_transform = get_transforms(task_type, "test", cfg, mode_cfg)

    def setup(self, stage: Optional[str] = None) -> None:
        cfg = self.cfg
        self.train_dataset, self.val_dataset, self.test_dataset = get_splits(
            path=cfg.task.dataset,
            subset=cfg.task.subset,
            input_key=cfg.task.input,
            target_key=cfg.task.target,
            train_transform=self.train_transform,
            test_transform=self.test_transform,
            train_size=cfg.data.train_val_split,
            stratify_by=cfg.task.stratify_by,
            group_by=cfg.task.group_by,
            filter_by=cfg.task.filter_by,
        )
        logger.info(
            "Dataset sizes — Train: %d, Val: %d, Test: %d",
            len(self.train_dataset),
            len(self.val_dataset),
            len(self.test_dataset),
        )

    def _make_loader(self, dataset: Any, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(self.num_workers > 0),
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_dataset, shuffle=False)


class UltraBenchModule(pl.LightningModule):
    """
    LightningModule for ProbeWrapper and SegmentationWrapper (frozen encoder) tasks.
    Handles training, validation, and test steps with torchmetrics and CSV prediction saving.
    """

    def __init__(
        self,
        model: DownstreamWrapper,
        criterion: nn.Module,
        mode_cfg: Any,
        task_type: str,
        output_dir: str,
    ):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.mode_cfg = mode_cfg
        self.task_type = task_type
        self.output_dir = output_dir
        self.is_seg = task_type == "segmentation"
        num_classes = model.num_classes

        # Metrics declared as nn.Module attributes so PTL moves them to the right device
        # and resets them automatically between epochs.
        if self.is_seg:
            self.train_metric = DiceScore(num_classes=num_classes, input_format="mixed")
            self.val_metric = DiceScore(num_classes=num_classes, input_format="mixed")
            self.test_dice = DiceScore(num_classes=num_classes, input_format="mixed")
            self.test_positive_dice = DiceScore(
                num_classes=num_classes, include_background=False, input_format="mixed"
            )
            self.test_iou = MeanIoU(num_classes=num_classes, input_format="mixed")
        else:
            self.train_metric = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes
            )
            self.val_metric = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes
            )
            self.test_accuracy = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes
            )
            self.test_bal_accuracy = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes, average="macro"
            )
            self.test_f1_macro = torchmetrics.F1Score(
                task="multiclass", num_classes=num_classes, average="macro"
            )
            self.test_f1_weighted = torchmetrics.F1Score(
                task="multiclass", num_classes=num_classes, average="weighted"
            )
            self.test_mcc = torchmetrics.MatthewsCorrCoef(
                task="multiclass", num_classes=num_classes
            )

        self._test_rows: List[Dict] = []
        self.test_results: Dict[str, float] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    # TRAINING
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        images = batch["image"]
        targets = batch["target"]
        if self.is_seg:
            targets = targets.squeeze(-1).long()

        outputs = self(images)
        loss = self.criterion(outputs, targets)

        if self.is_seg:
            self.train_metric.update(outputs.detach(), targets)
            self.log(
                "train_dice",
                self.train_metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )
        else:
            self.train_metric.update(outputs.detach().argmax(dim=1), targets)
            self.log(
                "train_acc",
                self.train_metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # VALIDATION
    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        images = batch["image"]
        targets = batch["target"]
        if self.is_seg:
            targets = targets.squeeze(-1).long()

        outputs = self(images)
        loss = self.criterion(outputs, targets)

        if self.is_seg:
            self.val_metric.update(outputs, targets)
            self.log(
                "val_dice", self.val_metric, on_step=False, on_epoch=True, prog_bar=True
            )
        else:
            self.val_metric.update(outputs.argmax(dim=1), targets)
            self.log(
                "val_acc", self.val_metric, on_step=False, on_epoch=True, prog_bar=True
            )

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    # TEST
    def on_test_epoch_start(self) -> None:
        self._test_rows = []

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        images = batch["image"]
        targets = batch["target"]
        keys = batch["filename"]

        if self.is_seg:
            targets = targets.squeeze(-1).long()

        outputs = self(images)

        if self.is_seg:
            self.test_dice.update(outputs, targets)
            self.test_positive_dice.update(outputs, targets)
            self.test_iou.update(outputs, targets)
            preds = outputs.argmax(dim=1).cpu().numpy()
            for key, pred_mask in zip(keys, preds):
                for class_id in range(1, self.model.num_classes):
                    binary = (pred_mask == class_id).astype(np.uint8)
                    self._test_rows.append(
                        {
                            "image_id": key,
                            "class_id": class_id,
                            "encoded_pixels": mask_to_rle(binary),
                        }
                    )
        else:
            preds = outputs.argmax(dim=1)
            self.test_accuracy.update(preds, targets)
            self.test_bal_accuracy.update(preds, targets)
            self.test_f1_macro.update(preds, targets)
            self.test_f1_weighted.update(preds, targets)
            self.test_mcc.update(preds, targets)
            for key, label in zip(keys, preds.cpu().tolist()):
                self._test_rows.append({"image_id": key, "label": label})

    def on_test_epoch_end(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        csv_path = os.path.join(self.output_dir, "predictions.csv")

        if self.is_seg:
            fieldnames = ["image_id", "class_id", "encoded_pixels"]
            metrics = {
                "dice": round(self.test_dice.compute().item(), 4),  # ty:ignore[missing-argument]
                "positive_dice": round(self.test_positive_dice.compute().item(), 4),  # ty:ignore[missing-argument]
                "iou": round(self.test_iou.compute().item(), 4),  # ty:ignore[missing-argument]
            }
        else:
            fieldnames = ["image_id", "label"]
            metrics = {
                "accuracy": round(self.test_accuracy.compute().item(), 3),  # ty:ignore[missing-argument]
                "balanced_accuracy": round(self.test_bal_accuracy.compute().item(), 3),  # ty:ignore[missing-argument]
                "f1_macro": round(self.test_f1_macro.compute().item(), 3),  # ty:ignore[missing-argument]
                "f1_weighted": round(self.test_f1_weighted.compute().item(), 3),  # ty:ignore[missing-argument]
                "mcc": round(self.test_mcc.compute().item(), 3),  # ty:ignore[missing-argument]
            }

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._test_rows)

        self.test_results = metrics
        logger.info("Test metrics: %s", metrics)

    # OPTIMIZER
    def configure_optimizers(self):
        return instantiate(
            self.mode_cfg.optimizer,
            params=self.model.get_trainable_parameters(),
            _recursive_=False,
        )


class FullFinetuningModule(UltraBenchModule):
    """
    LightningModule for FullFinetuningWrapper.

    Phase 1 (epoch < probe_warmup_epochs): encoder frozen, head-only optimizer.
    Phase 2 (epoch >= probe_warmup_epochs): encoder unfrozen, two-group optimizer
        with separate LRs, optional cosine scheduler.
    """

    def __init__(
        self,
        model: DownstreamWrapper,
        criterion: nn.Module,
        mode_cfg: Any,
        task_type: str,
        output_dir: str,
    ):
        super().__init__(model, criterion, mode_cfg, task_type, output_dir)
        self.probe_warmup_epochs = int(getattr(mode_cfg, "probe_warmup_epochs", 0))
        self._phase2_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == self.probe_warmup_epochs:
            logger.info("Phase 2: unfreezing encoder and rebuilding optimizer")
            self.model.unfreeze_encoder()
            new_opt, self._phase2_scheduler = self._build_phase2_optimizer()
            self.trainer.optimizers = [new_opt]
            num_trainable = sum(
                p.numel() for p in self.model.get_trainable_parameters()
            )
            logger.info("Continue training %d parameters...", num_trainable)

        phase = 1 if self.current_epoch < self.probe_warmup_epochs else 2
        self.log("training_phase", float(phase), on_step=False, on_epoch=True)

    def on_train_epoch_end(self) -> None:
        if self._phase2_scheduler is not None:
            self._phase2_scheduler.step()

    def _build_phase2_optimizer(
        self,
    ) -> tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.LRScheduler]]:
        optimizer_cfg = self.mode_cfg.optimizer
        head_lr = float(optimizer_cfg.get("lr"))
        encoder_lr = head_lr * float(
            self.mode_cfg.optimizer_phase2.get("encoder_lr_scale")
        )
        weight_decay = float(optimizer_cfg.get("weight_decay"))

        optimizer = get_class(optimizer_cfg.get("_target_"))(
            params=[
                {"params": self.model.encoder.parameters(), "lr": encoder_lr},
                {"params": self.model.head.parameters(), "lr": head_lr},  # ty:ignore[unresolved-attribute]
            ],
            weight_decay=weight_decay,
        )
        logger.info(
            "Built phase 2 optimizer: encoder_lr=%.2e head_lr=%.2e weight_decay=%.2e",
            encoder_lr,
            head_lr,
            weight_decay,
        )

        phase2_warmup_epochs = int(getattr(self.mode_cfg, "phase2_warmup_epochs", 0))
        phase2_epochs = max(self.trainer.max_epochs - self.probe_warmup_epochs, 1)  # ty:ignore[unsupported-operator]
        warmup_cfg = getattr(self.mode_cfg, "warmup_scheduler", None)
        main_cfg = getattr(self.mode_cfg, "scheduler", None)

        schedulers, milestones = [], []

        if warmup_cfg is not None and phase2_warmup_epochs > 0:
            schedulers.append(
                instantiate(
                    warmup_cfg,
                    optimizer=optimizer,
                    total_iters=phase2_warmup_epochs,
                    _recursive_=False,
                )
            )
            milestones.append(phase2_warmup_epochs)
            logger.info(
                "Phase 2 warmup: scheduler=%s total_iters=%d",
                warmup_cfg.get("_target_"),
                phase2_warmup_epochs,
            )

        if main_cfg is not None:
            t_max = max(phase2_epochs - phase2_warmup_epochs, 1)
            schedulers.append(
                instantiate(
                    main_cfg,
                    optimizer=optimizer,
                    T_max=t_max,
                    _recursive_=False,
                )
            )
            logger.info(
                "Phase 2 main: scheduler=%s T_max=%d",
                main_cfg.get("_target_"),
                t_max,
            )

        if len(schedulers) > 1:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers,
                milestones=milestones,
            )
        elif schedulers:
            scheduler = schedulers[0]
        else:
            scheduler = None

        return optimizer, scheduler


def create_wrapper(
    encoder: nn.Module,
    encoder_transforms: Optional[Any],
    encoder_metadata: Dict[str, Any],
    mode_cfg: Any,
    num_classes: int,
    task_type: str,
    data_cfg: Optional[Any] = None,
) -> DownstreamWrapper:
    """Create appropriate downstream wrapper (unchanged from setup.py)."""
    if not hasattr(mode_cfg, "wrapper") or mode_cfg.wrapper is None:
        raise ValueError(
            "mode_cfg.wrapper is required. Configure mode_cfg.wrapper._target_ "
            "in the selected Hydra mode group."
        )
    extra_kwargs = {}
    if task_type == "segmentation" and data_cfg is not None:
        extra_kwargs["image_size"] = data_cfg.image_size

    return instantiate(
        mode_cfg.wrapper,
        encoder=encoder,
        encoder_transforms=encoder_transforms,
        encoder_metadata=encoder_metadata,
        num_classes=num_classes,
        task_type=task_type,
        _recursive_=False,
        **extra_kwargs,
    )


class DownstreamPipeline:
    """
    Main pipeline for running downstream evaluation with PyTorch Lightning.
    """

    def __init__(self, data_dir: str = "./ultrabench_data"):
        self.data_dir = data_dir

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "Using device: %s (CUDA_VISIBLE_DEVICES=%s)",
            self.device,
            os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
        )

        # Map amp dtype → Lightning precision string
        if self.device == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
            self.amp_dtype = torch.bfloat16
            self.precision = "bf16-mixed"
        elif self.device == "cuda":
            self.amp_dtype = torch.float16
            self.precision = "16-mixed"
        else:
            self.amp_dtype = torch.float32
            self.precision = "32-true"

        logger.info(
            "AMP dtype: %s  (Lightning precision=%s)", self.amp_dtype, self.precision
        )

    def load_encoder(self, model_cfg: Any):
        """Load pretrained encoder from checkpoint"""
        bundle = instantiate(
            model_cfg.loader,
            model_cfg=model_cfg,
            device=self.device,
            _recursive_=False,
        )
        logger.info(
            "Loaded model via Hydra loader target baseline=%s variant=%s "
            "checkpoint=%s feature_dim=%s",
            bundle.metadata.get("baseline"),
            bundle.metadata.get("variant"),
            bundle.metadata.get("checkpoint_path"),
            bundle.metadata.get("feature_dim"),
        )
        return bundle.encoder, bundle.transforms, bundle.metadata

    def run_downstream(
        self,
        cfg: Optional[Any] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run complete pipeline on a single dataset."""
        pl.seed_everything(cfg.optimization.seed, workers=True)  # ty:ignore[unresolved-attribute]

        dataset_name = HydraConfig.get().runtime.choices.task  # ty:ignore[unresolved-attribute]
        task_type = cfg.task.type  # ty:ignore[unresolved-attribute]
        mode_cfg = cfg.cls_mode if task_type == "classification" else cfg.seg_mode  # ty:ignore[unresolved-attribute]
        mode_name = mode_cfg.name
        baseline = getattr(cfg.model, "baseline", None)  # ty:ignore[unresolved-attribute]
        num_classes = cfg.task.num_classes  # ty:ignore[unresolved-attribute]

        logger.info("Loading dataset: %s", cfg.task.dataset)  # ty:ignore[unresolved-attribute]
        logger.info("Task type: %s", task_type)

        encoder, encoder_transforms, encoder_metadata = self.load_encoder(cfg.model)  # ty:ignore[unresolved-attribute]

        output_type = encoder_metadata.get("output_type", cfg.model.output_type)  # ty:ignore[unresolved-attribute]
        if task_type == "segmentation" and output_type == "cls":
            raise ValueError(
                "Encoder output_type='cls' produces a single vector per image and cannot "
                "be used for segmentation. Use a ViT encoder with output_type='patch_tokens'."
            )

        model = create_wrapper(
            encoder,
            encoder_transforms,
            encoder_metadata,
            mode_cfg,
            num_classes,
            task_type,
            data_cfg=cfg.data,  # ty:ignore[unresolved-attribute]
        )
        criterion = instantiate(mode_cfg.criterion, _recursive_=False).to(self.device)

        logger.info(
            "Initialized optimizer=%s criterion=%s for mode=%s",
            mode_cfg.optimizer.get("_target_"),
            mode_cfg.criterion.get("_target_"),
            mode_name,
        )

        module_cls = (
            FullFinetuningModule
            if isinstance(model, FullFinetuningWrapper)
            else UltraBenchModule
        )
        module = module_cls(model, criterion, mode_cfg, task_type, output_dir)  # ty:ignore[invalid-argument-type]

        datamodule = UltraBenchDataModule(cfg, mode_cfg, self.data_dir, task_type)

        callbacks = [
            LearningRateMonitor(logging_interval="epoch"),
            ModelCheckpoint(
                dirpath=output_dir,
                filename=f"{mode_name}_best",
                monitor="val_loss",
                save_top_k=1,
                save_last=True,
                mode="min",
            ),
        ]
        if cfg.optimization.patience is not None:  # ty:ignore[unresolved-attribute]
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=cfg.optimization.patience,  # ty:ignore[unresolved-attribute]
                    mode="min",
                )  # ty:ignore[invalid-argument-type]
            )

        num_trainable = sum(p.numel() for p in model.get_trainable_parameters())
        logger.info(
            "Starting training %d parameters for %d epochs with patience=%s",
            num_trainable,
            cfg.optimization.num_epochs,  # ty:ignore[unresolved-attribute]
            cfg.optimization.patience,  # ty:ignore[unresolved-attribute]
        )

        trainer = pl.Trainer(
            max_epochs=cfg.optimization.num_epochs,  # ty:ignore[unresolved-attribute]
            precision=self.precision,  # ty:ignore[invalid-argument-type]
            accumulate_grad_batches=cfg.optimization.accum_grad,  # ty:ignore[unresolved-attribute]
            gradient_clip_val=getattr(mode_cfg, "max_grad_norm", None),
            gradient_clip_algorithm="norm",
            callbacks=callbacks,  # ty:ignore[invalid-argument-type]
            logger=TensorBoardLogger(
                save_dir=output_dir,  # ty:ignore[invalid-argument-type]
                name="tensorboard",
                version="",
            ),
            accelerator="auto",
            devices=1,
            enable_progress_bar=cfg.system.enable_progress_bar,  # ty:ignore[unresolved-attribute]
            log_every_n_steps=cfg.system.log_every_n_steps,  # ty:ignore[unresolved-attribute]
        )

        trainer.fit(module, datamodule=datamodule)
        trainer.test(module, datamodule=datamodule, ckpt_path="best")

        return {
            "dataset_name": dataset_name,
            "task_type": task_type,
            "mode": mode_name,
            "baseline": baseline,
            "encoder_loaded": encoder is not None,
            "encoder_metadata": encoder_metadata,
            "num_train": len(datamodule.train_dataset),
            "num_val": len(datamodule.val_dataset),
            "num_test": len(datamodule.test_dataset),
            "test_metrics": module.test_results,
        }
