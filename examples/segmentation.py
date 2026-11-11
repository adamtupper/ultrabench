"""Train a segmentation decoder on a frozen backbone for an UltraBench task.

Example:
    uv run --group examples examples/segmentation.py \\
        --task tn3k_segmentation \\
        --backbone \\
        ultrabench.models.dinov3.dinov3_backbone.DINOv3ViTBackbone \\
        --backbone-kwargs '{"model_variant": "dinov3_vitb16",
            "repo_dir": "dinov3-repo",
            "checkpoint_path": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"}'

The script trains the decoder on the `train` split, keeps the decoder with
the best foreground Dice on a held-out part of it, writes the run-length
encoded test masks to `<output-dir>/predictions.csv` and scores them with
the evaluation layer.
"""

import argparse
import copy

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import (
    TaskDataset,
    add_common_arguments,
    build_backbone,
    load_splits,
    score,
    set_seed,
)
from torch.utils.data import DataLoader

from ultrabench.core.rle import mask_to_rle
from ultrabench.evaluate.tasks import get_task
from ultrabench.models.backbone import Backbone
from ultrabench.models.heads import DECODER_COMPATIBILITY, get_seg_head

DESCRIPTION = (
    "Train a segmentation decoder on a frozen backbone for an UltraBench task."
)
"""The summary shown by --help."""

INTERMEDIATE_DECODERS = ("setr_mla", "segformer")
"""Decoders that read the backbone's intermediate features."""


class DiceCELoss(nn.Module):
    """The mean of the cross-entropy and soft Dice losses.

    Args:
        num_classes: The number of classes, including the background.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            logits: The predicted logits, of shape [B, C, H, W].
            target: The class index of each pixel, of shape [B, H, W].

        Returns:
            The scalar loss.
        """
        probabilities = logits.softmax(dim=1)
        one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        intersection = (probabilities * one_hot).sum(dim=(2, 3))
        total = probabilities.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))
        dice = (2 * intersection + 1) / (total + 1)

        return 0.5 * F.cross_entropy(logits, target) + 0.5 * (1 - dice.mean())


def segment(
    backbone: Backbone, head: nn.Module, images: torch.Tensor, decoder: str
) -> torch.Tensor:
    """Predict the segmentation logits of a batch of images.

    Args:
        backbone: The frozen backbone.
        head: The segmentation decoder.
        images: A uint8 batch of shape [B, 3, H, W].
        decoder: The decoder type.

    Returns:
        The logits, of shape [B, num_classes, H, W].
    """
    with torch.no_grad():
        output = backbone(images)
    if decoder in INTERMEDIATE_DECODERS:
        return head(output.intermediate_features)

    return head(output.features)


@torch.no_grad()
def foreground_dice(
    backbone: Backbone,
    head: nn.Module,
    loader: DataLoader,
    decoder: str,
    num_classes: int,
    device: torch.device,
) -> float:
    """Compute the Dice score of the foreground classes over a split.

    Args:
        backbone: The frozen backbone.
        head: The segmentation decoder.
        loader: The loader of the split.
        decoder: The decoder type.
        num_classes: The number of classes, including the background.
        device: The device to run on.

    Returns:
        The Dice score, pooled over the split and averaged over classes.
    """
    head.eval()
    intersection = torch.zeros(num_classes - 1)
    total = torch.zeros(num_classes - 1)
    for batch in loader:
        logits = segment(backbone, head, batch["image"].to(device), decoder)
        prediction = logits.argmax(dim=1).cpu()
        for k in range(1, num_classes):
            predicted, actual = prediction == k, batch["target"] == k
            intersection[k - 1] += (predicted & actual).sum()
            total[k - 1] += predicted.sum() + actual.sum()

    return (2 * intersection / total.clamp(min=1)).mean().item()


def main(args: argparse.Namespace) -> None:
    """Train, select, predict and score.

    Args:
        args: The parsed command-line arguments.

    Raises:
        ValueError: If the task is not a segmentation task.
    """
    set_seed(args.seed)
    task = get_task(args.task)
    if task.task_type != "segmentation":
        raise ValueError(f"{args.task} is not a segmentation task.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train, val, test = load_splits(task, args.path, args.val_size, args.seed)
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    train_loader = DataLoader(TaskDataset(train, task), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(TaskDataset(val, task), **loader_kwargs)
    test_loader = DataLoader(TaskDataset(test, task), **loader_kwargs)

    backbone = build_backbone(
        args.backbone,
        args.backbone_kwargs,
        output_intermediate_features=args.decoder in INTERMEDIATE_DECODERS,
    ).to(device)
    head = get_seg_head(
        decoder_type=args.decoder,
        output_type=backbone.output_type,
        feature_dim=getattr(backbone, "seg_feature_dim", backbone.feature_dim),
        num_classes=task.num_classes,
        image_size=task.image_size,
        block_indices=getattr(backbone, "block_indices", None),
        scale_dims=backbone.intermediate_feature_dims,
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    criterion = DiceCELoss(task.num_classes)

    best_dice, best_head, step = -1.0, copy.deepcopy(head), 0
    for epoch in range(args.epochs):
        head.train()
        for batch in train_loader:
            logits = segment(backbone, head, batch["image"].to(device), args.decoder)
            loss = criterion(logits, batch["target"].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step == args.max_steps:
                break

        val_dice = foreground_dice(
            backbone, head, val_loader, args.decoder, task.num_classes, device
        )
        print(f"epoch {epoch}: loss {loss.item():.4f}, val Dice {val_dice:.4f}")
        if val_dice > best_dice:
            best_dice, best_head = val_dice, copy.deepcopy(head)
        if step == args.max_steps:
            break

    best_head.eval()
    rows = []
    with torch.no_grad():
        for batch in test_loader:
            logits = segment(
                backbone, best_head, batch["image"].to(device), args.decoder
            )
            masks = logits.argmax(dim=1).cpu().numpy()
            rows.extend(
                {
                    "image_id": image_id,
                    "class_id": k,
                    "encoded_pixels": mask_to_rle((mask == k).astype("uint8")),
                }
                for image_id, mask in zip(batch["image_id"], masks)
                for k in range(1, task.num_classes)
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.csv"
    pd.DataFrame(rows).to_csv(predictions_path, index=False)
    print(f"Wrote {predictions_path}")
    score(args.task, predictions_path, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    add_common_arguments(parser)
    parser.add_argument(
        "--decoder", choices=sorted(DECODER_COMPATIBILITY), default="upsampler"
    )
    main(parser.parse_args())
