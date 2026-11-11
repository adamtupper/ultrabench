"""Train a linear probe on a frozen backbone for a classification task.

Example:
    uv run --group examples examples/classification.py \\
        --task tn3k_classification \\
        --backbone \\
        ultrabench.models.dinov3.dinov3_backbone.DINOv3ViTBackbone \\
        --backbone-kwargs '{"model_variant": "dinov3_vitb16",
            "repo_dir": "dinov3-repo",
            "checkpoint_path": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"}'

The script trains the head on the `train` split, keeps the head with the
best macro F1 on a held-out part of it, writes the test predictions to
`<output-dir>/predictions.csv` and scores them with the evaluation layer.
"""

import argparse
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from common import (
    TaskDataset,
    add_common_arguments,
    build_backbone,
    load_splits,
    score,
    set_seed,
)
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from ultrabench.evaluate.tasks import get_task
from ultrabench.models.backbone import Backbone
from ultrabench.models.heads import get_cls_head

DESCRIPTION = "Train a linear probe on a frozen backbone for a classification task."
"""The summary shown by --help."""


@torch.no_grad()
def predict(
    backbone: Backbone, head: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Predict the class probabilities of every example in a split.

    Args:
        backbone: The frozen backbone.
        head: The classification head.
        loader: The loader of the split.
        device: The device to run on.

    Returns:
        The image IDs, the predicted probabilities and the targets.
    """
    head.eval()
    image_ids, probabilities, targets = [], [], []
    for batch in loader:
        features = backbone(batch["image"].to(device), pool=True).features
        probabilities.append(head(features).softmax(dim=1).cpu())
        targets.append(batch["target"])
        image_ids.extend(batch["image_id"])

    return image_ids, torch.cat(probabilities).numpy(), torch.cat(targets).numpy()


def main(args: argparse.Namespace) -> None:
    """Train, select, predict and score.

    Args:
        args: The parsed command-line arguments.

    Raises:
        ValueError: If the task is not a classification task.
    """
    set_seed(args.seed)
    task = get_task(args.task)
    if task.task_type != "classification":
        raise ValueError(f"{args.task} is not a classification task.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train, val, test = load_splits(task, args.path, args.val_size, args.seed)
    loader_kwargs = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    train_loader = DataLoader(TaskDataset(train, task), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(TaskDataset(val, task), **loader_kwargs)
    test_loader = DataLoader(TaskDataset(test, task), **loader_kwargs)

    backbone = build_backbone(args.backbone, args.backbone_kwargs).to(device)
    head = get_cls_head("linear", backbone.feature_dim, task.num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_f1, best_head, step = -1.0, copy.deepcopy(head), 0
    for epoch in range(args.epochs):
        head.train()
        for batch in train_loader:
            with torch.no_grad():
                features = backbone(batch["image"].to(device), pool=True).features
            loss = criterion(head(features), batch["target"].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step == args.max_steps:
                break

        _, probabilities, targets = predict(backbone, head, val_loader, device)
        val_f1 = f1_score(targets, probabilities.argmax(axis=1), average="macro")
        print(f"epoch {epoch}: loss {loss.item():.4f}, val macro F1 {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, best_head = val_f1, copy.deepcopy(head)
        if step == args.max_steps:
            break

    image_ids, probabilities, _ = predict(backbone, best_head, test_loader, device)
    predictions = pd.DataFrame({"image_id": image_ids})
    predictions["label"] = probabilities.argmax(axis=1)
    for k in range(task.num_classes):
        predictions[f"prob_{k}"] = probabilities[:, k]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    print(f"Wrote {predictions_path}")
    score(args.task, predictions_path, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    add_common_arguments(parser)
    main(parser.parse_args())
