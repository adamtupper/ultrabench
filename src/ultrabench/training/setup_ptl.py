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
from hydra.utils import instantiate
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchmetrics.segmentation import DiceScore, MeanIoU

from ultrabench.core.rle import mask_to_rle
from ultrabench.models.backbone import Backbone
from ultrabench.training.callbacks import BackboneFinetuningSteps
from ultrabench.training.heads.cls_heads import get_cls_head
from ultrabench.training.heads.seg_heads import get_seg_head

from .data.transforms import get_train_test_transforms
from .data.utils import get_splits

logger = logging.getLogger(__name__)


class UltraBenchDataModule(pl.LightningDataModule):
    """Wraps dataset loading and DataLoader construction for all splits."""

    def __init__(
        self,
        path: str,
        subset: str | None,
        input_key: str,
        target_key: str,
        image_size: int,
        train_val_split: float,
        augmentation_strategy: str,
        stratify_by: str | None,
        group_by: str | None,
        filter_by: str | None,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.path = path
        self.subset = subset
        self.input_key = input_key
        self.target_key = target_key
        self.train_val_split = train_val_split
        self.stratify_by = stratify_by
        self.group_by = group_by
        self.filter_by = filter_by
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_transform, self.test_transform = get_train_test_transforms(
            image_size, augmentation_strategy=augmentation_strategy
        )

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset, self.val_dataset, self.test_dataset = get_splits(
            path=self.path,
            subset=self.subset,
            input_key=self.input_key,
            target_key=self.target_key,
            train_transform=self.train_transform,
            test_transform=self.test_transform,
            train_size=self.train_val_split,
            stratify_by=self.stratify_by,
            group_by=self.group_by,
            filter_by=self.filter_by,
        )
        logger.info(
            "Dataset sizes — Train: %d, Val: %d, Test: %d",
            len(self.train_dataset),
            len(self.val_dataset),
            len(self.test_dataset),
        )

    def _make_loader(
        self, dataset: Any, shuffle: bool, drop_last: bool = False
    ) -> DataLoader:
        """Construct a data loader.

        Args:
            dataset: Dataset (split) to load from.
            shuffle: Whether to shuffle the data (should be True for training).
            drop_last: Whether to drop the last incomplete batch (should be
                True for training).
        Returns:
            A DataLoader for the given dataset.
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(self.num_workers > 0),
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_loader(self.train_dataset, shuffle=True, drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._make_loader(self.test_dataset, shuffle=False)


class UltraBenchModule(pl.LightningModule):
    """LightningModule wrapping a Backbone and a task head.

    Supports classification and segmentation. The backbone may be frozen
    (probe) or unfrozen (full finetuning via BackboneFinetuningSteps
    callback). When the backbone is frozen, its eval mode is preserved
    across Lightning's train() calls via the train() override.

    Attributes:
        backbone: The pretrained encoder. Exposed as a direct attribute so
            BackboneFinetuningSteps can locate it.
        head: Task-specific head (classification or segmentation decoder).
        use_intermediate: If True, backbone is called with
            return_intermediate=True and intermediate outputs are passed to
            the head (setr_mla / segformer decoders only).
    """

    def __init__(
        self,
        backbone: Backbone,
        head: nn.Module,
        criterion: nn.Module,
        optimizer_cfg: Any,
        scheduler_cfg: Any,
        task_type: str,
        output_dir: str,
        num_classes: int,
        use_intermediate: bool = False,
    ):
        """Initialise UltraBenchModule.

        Args:
            backbone: Pretrained encoder implementing the Backbone interface.
            head: Task head — classification or segmentation decoder.
            criterion: Loss function.
            optimizer_cfg: Hydra config dict for the optimizer.
            scheduler_cfg: Hydra config dict for the learning rate scheduler.
            task_type: "classification" or "segmentation".
            output_dir: Directory for CSV predictions and checkpoints.
            num_classes: Number of output classes.
            use_intermediate: Pass backbone intermediate outputs to head
                instead of final features. True for setr_mla / segformer.
        """
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.criterion = criterion
        # self.mode_cfg = mode_cfg
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg
        self.task_type = task_type
        self.output_dir = output_dir
        self.num_classes = num_classes
        self.use_intermediate = use_intermediate
        self.is_seg = task_type == "segmentation"

        if self.is_seg:
            self.train_metric = DiceScore(num_classes=num_classes, input_format="mixed")
            self.val_metric = DiceScore(num_classes=num_classes, input_format="mixed")
            self.test_dice = DiceScore(num_classes=num_classes, input_format="mixed")
            self.test_positive_dice = DiceScore(
                num_classes=num_classes,
                include_background=False,
                input_format="mixed",
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
        """Run backbone → head forward pass.

        Args:
            x: Pre-processed input tensor [B, C, H, W].

        Returns:
            Logits [B, num_classes] for classification or
            [B, num_classes, H, W] for segmentation.
        """
        out = self.backbone(x)

        if self.use_intermediate:
            # setr_mla / segformer: pass list of intermediate tensors to head
            return self.head(out.intermediate_features)

        features = out.features  # [B, N, D] or [B, C, H, W]

        if self.is_seg:
            return self.head(features)

        # Classification: global average pool then head
        if features.dim() == 3:
            features = features.mean(dim=1)  # patch_tokens → [B, D]
        elif features.dim() == 4:
            features = features.mean(dim=(2, 3))  # spatial/multiscale → [B, C]
        return self.head(features)

    def train(self, mode: bool = True) -> "UltraBenchModule":
        """Keep the backbone in eval mode when all its parameters are frozen.

        Args:
            mode: Training mode flag passed through to all sub-modules.

        Returns:
            Self.
        """
        super().train(mode)
        if mode and not any(p.requires_grad for p in self.backbone.parameters()):
            self.backbone.eval()
        return self

    # TRAINING
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Compute loss and update train metric.

        Args:
            batch: Dict with "image" and "target" keys.
            batch_idx: Index of the current batch.

        Returns:
            Scalar loss tensor.
        """
        images = batch["image"]
        targets = batch["target"]
        if self.is_seg:
            targets = targets.squeeze(-1).long()

        outputs = self(images)
        loss = self.criterion(outputs, targets)

        if self.is_seg:
            self.train_metric(outputs.detach(), targets)
            self.log("train_dice", self.train_metric, prog_bar=True)
        else:
            self.train_metric(outputs.detach().argmax(dim=1), targets)
            self.log("train_acc", self.train_metric, prog_bar=True)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    # VALIDATION
    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        """Compute validation loss and metric.

        Args:
            batch: Dict with "image" and "target" keys.
            batch_idx: Index of the current batch.
        """
        images = batch["image"]
        targets = batch["target"]
        if self.is_seg:
            targets = targets.squeeze(-1).long()

        outputs = self(images)
        loss = self.criterion(outputs, targets)

        if self.is_seg:
            self.val_metric(outputs, targets)
            self.log("val_dice", self.val_metric, prog_bar=True)
        else:
            self.val_metric(outputs.argmax(dim=1), targets)
            self.log("val_acc", self.val_metric, prog_bar=True)

        self.log("val_loss", loss, prog_bar=True)

    # TEST
    def on_test_epoch_start(self) -> None:
        self._test_rows = []

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        """Accumulate test predictions and metrics.

        Args:
            batch: Dict with "image", "target", and "filename" keys.
            batch_idx: Index of the current batch.
        """
        images = batch["image"]
        targets = batch["target"]
        keys = batch["filename"]

        if self.is_seg:
            targets = targets.squeeze(-1).long()

        outputs = self(images)

        if self.is_seg:
            self.test_dice(outputs, targets)
            self.test_positive_dice(outputs, targets)
            self.test_iou(outputs, targets)
            preds = outputs.argmax(dim=1).cpu().numpy()
            for key, pred_mask in zip(keys, preds):
                for class_id in range(1, self.num_classes):
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
            self.test_accuracy(preds, targets)
            self.test_bal_accuracy(preds, targets)
            self.test_f1_macro(preds, targets)
            self.test_f1_weighted(preds, targets)
            self.test_mcc(preds, targets)
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

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers.

        Returns:
            Dict with the configured optimizer(s) an LR scheduler(s).
        """
        optimizer = instantiate(self.optimizer_cfg, params=self.parameters())
        scheduler = instantiate(self.scheduler_cfg, optimizer=optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # Updates every batch
                "frequency": 1,  # Calls scheduler.step() every 1 step
            },
        }


class DownstreamPipeline:
    """Main pipeline for downstream evaluation with PyTorch Lightning."""

    def __init__(self, data_dir: str = "./ultrabench_data"):
        self.data_dir = data_dir

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            "Using device: %s (CUDA_VISIBLE_DEVICES=%s)",
            self.device,
            os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
        )

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

    def load_backbone(self, model_cfg: Any) -> Backbone:
        """Instantiate and move backbone to the configured device.

        Args:
            model_cfg: Hydra model config with _target_ pointing to a
                Backbone subclass.

        Returns:
            Backbone instance on self.device.
        """
        backbone: Backbone = instantiate(model_cfg, _recursive_=False)
        backbone = backbone.to(self.device)
        logger.info(
            "Loaded backbone: %s  feature_dim=%d  output_type=%s",
            type(backbone).__name__,
            backbone.feature_dim,
            backbone.output_type,
        )
        return backbone

    def _build_head(
        self,
        backbone: Backbone,
        head_cfg: Any,
        num_classes: int,
        task_type: str,
        image_size: int | None = None,
    ) -> tuple[nn.Module, bool]:
        """Build task head from backbone attributes and mode config.

        Args:
            backbone: Loaded backbone (provides feature_dim, output_type,
                block_indices, scale_dims).
            head_cfg: Hydra config for the task head (e.g., linear classifier).
            num_classes: Number of output classes.
            task_type: "classification" or "segmentation".
            image_size: Spatial output resolution (required for segmentation).

        Returns:
            Tuple of (head, use_intermediate). use_intermediate is True only
            for setr_mla and segformer decoders.
        """
        if task_type == "classification":
            probe_type = getattr(head_cfg, "probe_type", "linear")
            head = get_cls_head(probe_type, backbone.feature_dim, num_classes)
            return head, False

        decoder_type = getattr(head_cfg, "decoder_type", "upsampler")
        use_intermediate = decoder_type in ("setr_mla", "segformer")
        if image_size is None:
            raise ValueError(
                "image_size is required for segmentation heads. "
                "Set data.image_size in the task config."
            )
        block_indices = (
            getattr(head_cfg, "block_indices", None) or backbone.block_indices
        )
        head = get_seg_head(
            decoder_type=decoder_type,
            output_type=backbone.output_type,
            feature_dim=backbone.feature_dim,
            scale_dims=getattr(backbone, "intermediate_feature_dims", None),
            num_classes=num_classes,
            image_size=image_size,
            block_indices=block_indices,
            use_residual=getattr(head_cfg, "upsampler_use_residual", True),
        )
        return head, use_intermediate

    def run_downstream(
        self,
        cfg: Optional[Any] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run complete downstream evaluation pipeline on a single dataset.

        Args:
            cfg: Hydra config object.
            output_dir: Directory for outputs (checkpoints, predictions CSV).

        Returns:
            Dict with dataset info and test metrics.
        """
        if output_dir is None:
            raise ValueError("output_dir is required")

        pl.seed_everything(cfg.optimization.seed, workers=True)  # ty:ignore[unresolved-attribute]

        dataset_name = HydraConfig.get().runtime.choices.task  # ty:ignore[unresolved-attribute]
        task_type = cfg.task.type  # ty:ignore[unresolved-attribute]
        mode_name = "{}_{}".format(
            HydraConfig.get().runtime.choices.head,  # ty:ignore[unresolved-attribute]
            HydraConfig.get().runtime.choices.mode,  # ty:ignore[unresolved-attribute]
        )
        num_classes = cfg.task.num_classes  # ty:ignore[unresolved-attribute]

        logger.info("Loading dataset: %s", cfg.task.dataset)  # ty:ignore[unresolved-attribute]
        logger.info("Task type: %s", task_type)

        backbone = instantiate(cfg.model)  # ty:ignore[unresolved-attribute]

        if task_type == "segmentation" and backbone.output_type == "cls":
            raise ValueError(
                "Backbone output_type='cls' produces a single vector per image "
                "and cannot be used for segmentation. Use a ViT backbone with "
                "output_type='patch_tokens'."
            )

        image_size = cfg.data.image_size  # ty:ignore[unresolved-attribute]
        head, use_intermediate = self._build_head(
            backbone,
            cfg.head,  # ty:ignore[unresolved-attribute]
            num_classes,
            task_type,
            image_size=image_size,
        )
        criterion = instantiate(cfg.task.criterion, _recursive_=False).to(self.device)  # ty:ignore[unresolved-attribute]

        logger.info(
            "Initialized optimizer=%s criterion=%s for mode=%s",
            cfg.mode.optimizer.get("_target_"),  # ty:ignore[unresolved-attribute]
            cfg.task.criterion.get("_target_"),  # ty:ignore[unresolved-attribute]
            mode_name,
        )

        module = UltraBenchModule(
            backbone=backbone,
            head=head,
            criterion=criterion,
            optimizer_cfg=cfg.mode.optimizer,  # ty:ignore[unresolved-attribute]
            scheduler_cfg=cfg.mode.scheduler,  # ty:ignore[unresolved-attribute]
            task_type=task_type,
            output_dir=output_dir,
            num_classes=num_classes,
            use_intermediate=use_intermediate,
        )

        datamodule = UltraBenchDataModule(
            path=cfg.task.dataset,  # ty:ignore[unresolved-attribute]
            subset=cfg.task.subset,  # ty:ignore[unresolved-attribute]
            input_key=cfg.task.input,  # ty:ignore[unresolved-attribute]
            target_key=cfg.task.target,  # ty:ignore[unresolved-attribute]
            image_size=image_size,
            augmentation_strategy=cfg.data.augmentation_strategy,  # ty:ignore[unresolved-attribute]
            train_val_split=cfg.data.train_val_split,  # ty:ignore[unresolved-attribute]
            stratify_by=cfg.task.stratify_by,  # ty:ignore[unresolved-attribute]
            group_by=cfg.task.group_by,  # ty:ignore[unresolved-attribute]
            filter_by=cfg.task.filter_by,  # ty:ignore[unresolved-attribute]
            batch_size=cfg.optimization.batch_size,  # ty:ignore[unresolved-attribute]
            num_workers=cfg.system.data_loader_workers,  # ty:ignore[unresolved-attribute]
        )

        unfreeze_backbone_at_step = cfg.optimization.unfreeze_backbone_at_step  # ty:ignore[unresolved-attribute]
        if unfreeze_backbone_at_step < 0:
            unfreeze_backbone_at_step = cfg.optimization.max_steps + 1  # ty:ignore[unresolved-attribute]

        num_trainable = sum(p.numel() for p in head.parameters())
        logger.info(
            "Starting training %d head parameters for %d steps with patience=%s",
            num_trainable,
            cfg.optimization.max_steps,  # ty:ignore[unresolved-attribute]
            cfg.optimization.patience,  # ty:ignore[unresolved-attribute]
        )

        callbacks: list = [
            LearningRateMonitor(logging_interval="step"),
            ModelCheckpoint(
                dirpath=output_dir,
                filename=f"{mode_name}_best",
                monitor="val_loss",
                save_top_k=1,
                save_last=True,
                mode="min",
            ),
            BackboneFinetuningSteps(
                unfreeze_backbone_at_step=unfreeze_backbone_at_step,
                backbone_initial_ratio_lr=cfg.optimization.backbone_initial_ratio_lr,  # ty:ignore[unresolved-attribute]
            ),
        ]
        if cfg.optimization.patience is not None:  # ty:ignore[unresolved-attribute]
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=cfg.optimization.patience,  # ty:ignore[unresolved-attribute]
                    mode="min",
                )
            )

        val_check_interval = (
            cfg.optimization.val_check_interval * cfg.optimization.accum_grad  # ty:ignore[unresolved-attribute]
        )

        trainer = pl.Trainer(
            max_epochs=-1,
            max_steps=cfg.optimization.max_steps,  # ty:ignore[unresolved-attribute]
            log_every_n_steps=cfg.system.log_every_n_steps,  # ty:ignore[unresolved-attribute]
            check_val_every_n_epoch=None,
            val_check_interval=val_check_interval,
            precision=self.precision,  # ty:ignore[invalid-argument-type]
            accumulate_grad_batches=cfg.optimization.accum_grad,  # ty:ignore[unresolved-attribute]
            gradient_clip_val=cfg.optimization.max_grad_norm,  # ty:ignore[unresolved-attribute]
            gradient_clip_algorithm="norm",
            callbacks=callbacks,
            logger=TensorBoardLogger(
                save_dir=output_dir,
                name="tensorboard",
                version="",
            ),
            accelerator="auto",
            devices=1,
            enable_progress_bar=cfg.system.enable_progress_bar,  # ty:ignore[unresolved-attribute]
        )

        trainer.fit(module, datamodule=datamodule)
        trainer.test(module, datamodule=datamodule, ckpt_path="best")

        return {
            "dataset_name": dataset_name,
            "task_type": task_type,
            "mode": mode_name,
            "backbone_type": type(backbone).__name__,
            "backbone_feature_dim": backbone.feature_dim,
            "num_train": len(datamodule.train_dataset),
            "num_val": len(datamodule.val_dataset),
            "num_test": len(datamodule.test_dataset),
            "test_metrics": module.test_results,
        }
