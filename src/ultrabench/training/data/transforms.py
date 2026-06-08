"""
Data augmentation and optional preprocessing transforms using Albumentations.

Users can customize via:
1. cfg.data.preprocessing_fn - Optional preprocessing function name (from user_preprocessing.py)
2. Modify get_classification_transforms() or get_segmentation_transforms() for augmentations
"""

import logging
from typing import Any, Callable, Optional

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from usaugment.albumentations import DepthAttenuation, GaussianShadow, HazeArtifact

from data.user_preprocessing import get_preprocessing_fn  # ty:ignore[unresolved-import]

logger = logging.getLogger(__name__)


class PreprocessingTransform:
    """
    Wrapper to apply preprocessing function within Albumentations pipeline.
    Preprocessing is applied before augmentations.
    """

    def __init__(self, preprocessing_fn: Callable, cfg: Optional[Any] = None):
        self.preprocessing_fn = preprocessing_fn
        self.cfg = cfg

    def __call__(self, image, **kwargs):
        """Apply preprocessing to image."""
        return self.preprocessing_fn(image, self.cfg)


def get_mmotu_ub_transforms(img_size: int = 224) -> list:
    """
    MMOTU-specific augmentations: randomly applies one of rotate, translate,
    random crop, or gaussian noise per sample.
    """
    return [
        A.OneOf(
            [
                A.Rotate(limit=30, border_mode=0, value=0, p=1.0),  # ty:ignore[unknown-argument]
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0,
                    rotate_limit=0,
                    interpolation=1,
                    border_mode=0,
                    value=0,
                    p=1.0,
                ),  # ty:ignore[unknown-argument]
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.6, 1.0), p=1.0),
                A.GaussNoise(
                    var_limit=0.0225,
                    mean=0.0,
                    per_channel=False,
                    noise_scale_factor=1,
                    p=1.0,
                ),  # ty:ignore[unknown-argument]
            ],
            p=1.0,
        ),
    ]


def get_general_ub_transforms(img_size: int = 224) -> list:
    """
    General augmentations from ultrabench paper: randomly applies one of rotate, translate,
    random crop, or gaussian noise per sample.
    """
    return [
        A.OneOf(
            [
                A.Rotate(limit=30, border_mode=0, value=0, p=1.0),  # ty:ignore[unknown-argument]
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0,
                    rotate_limit=0,
                    interpolation=1,
                    border_mode=0,
                    value=0,
                    p=1.0,
                ),  # ty:ignore[unknown-argument]
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.6, 1.0), p=1.0),
                A.GaussNoise(
                    var_limit=0.0225,
                    mean=0.0,
                    per_channel=False,
                    noise_scale_factor=1,
                    p=1.0,
                ),  # ty:ignore[unknown-argument]
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=1.0
                ),
            ],
            p=1.0,
        ),
    ]


def get_usaugment_transforms() -> list:
    """Ultrasound-specific augmentations from usaugment (train split only)."""

    return [
        A.HorizontalFlip(p=0.5),
        DepthAttenuation(p=0.5),
        GaussianShadow(p=0.5),
        HazeArtifact(p=0.5),
        # SpeckleReduction(p=0.5),
    ]


def get_transforms(
    task_type: str, split: str, cfg: Any = None, mode_cfg: Any = None
) -> A.Compose:
    """
    Get data transforms/augmentations for a given task and split.

    Pipeline order:
    1. Optional preprocessing (from cfg.data.preprocessing_fn) - deterministic
    2. Augmentations (task-specific) - stochastic transformations
    3. Normalization + ToTensor

    Args:
        task_type: 'classification' or 'segmentation'
        split: 'train', 'validation', or 'test'
        cfg: Optional config object containing augmentation parameters

    Returns:
        Albumentations Compose object with transforms
    """

    # Image size
    img_size = cfg.data.image_size

    # Build transform list.
    transform_list = []

    # Step 1: Optional user preprocessing.
    preprocessing_name = getattr(cfg.data, "preprocessing_fn", None)

    if preprocessing_name:
        try:
            preprocessing_fn = get_preprocessing_fn(preprocessing_name)
            logger.info(
                "Using preprocessing_fn=%s in transform pipeline", preprocessing_name
            )
            transform_list.append(
                A.Lambda(
                    image=PreprocessingTransform(preprocessing_fn, cfg),
                    name=preprocessing_name,
                )
            )
        except (ImportError, ValueError) as e:
            logger.warning(
                "Unable to load preprocessing_fn=%s (%s). Continuing without preprocessing.",
                preprocessing_name,
                e,
            )

    # Step 2: Domain augmentations.
    domain_augmentation = (
        getattr(mode_cfg, "augmentation", None) if mode_cfg is not None else None
    )

    if task_type == "classification":
        transform_list.extend(
            get_classification_transforms(
                split, img_size, augmentation=domain_augmentation
            )
        )  # ty:ignore[invalid-argument-type]
    elif task_type == "segmentation":
        transform_list.extend(
            get_segmentation_transforms(
                split, img_size, augmentation=domain_augmentation
            )
        )  # ty:ignore[invalid-argument-type]
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    # Return composed transforms
    return A.Compose(transform_list, additional_targets={"scan_mask": "mask"})


def get_classification_transforms(
    split: str, img_size: int = 224, augmentation: str = None
) -> list:  # ty:ignore[invalid-parameter-default]
    """
    Get transforms for classification tasks.

    Customize augmentations here for your classification needs.

    Args:
        split: 'train', 'validation', or 'test'
        img_size: Target image size
        augmentation: Optional augmentation package to use (e.g. 'usaugment')

    Returns:
        List of Albumentations transforms (not Compose, will be composed with preprocessing)
    """

    if split == "train":
        # Training augmentations - add more as needed
        transforms = [
            A.Resize(img_size, img_size),
        ]
        if augmentation == "usaugment":
            transforms += get_usaugment_transforms()
        elif augmentation == "general":
            transforms += get_general_ub_transforms(img_size=img_size)
        elif augmentation == "mmotu":
            transforms += get_mmotu_ub_transforms(img_size=img_size)
        transforms += [ToTensorV2()]  # ty:ignore[unsupported-operator]
        return transforms

    else:
        # Validation/test - no augmentation, only preprocessing
        return [A.Resize(img_size, img_size), ToTensorV2()]


def get_segmentation_transforms(
    split: str, img_size: int = 224, augmentation: str = None
) -> list:  # ty:ignore[invalid-parameter-default]
    """
    Get transforms for segmentation tasks.

    Customize augmentations here for your segmentation needs.

    Args:
        split: 'train', 'validation', or 'test'
        img_size: Target image size
        augmentation: Optional augmentation package to use (e.g. 'usaugment')

    Returns:
        List of Albumentations transforms (not Compose, will be composed with preprocessing)
    """

    if split == "train":
        # Training augmentations - add more as needed
        transforms = [
            A.Resize(img_size, img_size),
        ]
        if augmentation == "usaugment":
            transforms += get_usaugment_transforms()
        elif augmentation == "general":
            transforms += get_general_ub_transforms(img_size=img_size)
        elif augmentation == "mmotu":
            transforms += get_mmotu_ub_transforms(img_size=img_size)
        transforms += [ToTensorV2()]  # ty:ignore[unsupported-operator]
        return transforms

    else:
        # Validation/test - no augmentation, only preprocessing
        return [A.Resize(img_size, img_size), ToTensorV2()]


def albumentations_transforms(
    examples: dict[str, list], transform: A.Compose
) -> dict[str, list]:
    """Apply Albumentations transforms to a batch of examples.

    Checks if target is a mask (segmentation) or not (classification)
    and applies transforms accordingly.

    Args:
        examples: Batch of examples from HuggingFace dataset.
        transform: Albumentations Compose object with transforms to
            apply.

    Returns:
        Transformed examples.
    """
    target_is_mask = isinstance(examples["target"][0], Image.Image)

    images = [np.array(image.convert("RGB")) for image in examples["image"]]
    scan_masks = [np.array(mask) for mask in examples["scan_mask"]]

    inputs = []
    if target_is_mask:
        targets = [np.array(target) for target in examples["target"]]
        for image, mask, scan_mask in zip(images, targets, scan_masks):
            inputs.append(transform(image=image, mask=mask, scan_mask=scan_mask))
    else:
        for image, scan_mask in zip(images, scan_masks):
            inputs.append(transform(image=image, scan_mask=scan_mask))

    examples["image"] = [input["image"] for input in inputs]
    examples["scan_mask"] = [input["scan_mask"] for input in inputs]

    if target_is_mask:
        examples["target"] = [input["mask"] for input in inputs]

    return examples


# =============================================================================
# CUSTOMIZATION GUIDE
# =============================================================================
#
# To customize augmentations for your use case:
#
# 1. Modify the get_classification_transforms() or get_segmentation_transforms()
#    functions above to add/remove/modify augmentations
#
# 2. Available Albumentations augmentations include:
#    - Geometric: HorizontalFlip, VerticalFlip, Rotate, ShiftScaleRotate, etc.
#    - Color: RandomBrightnessContrast, HueSaturationValue, ColorJitter, etc.
#    - Noise: GaussNoise, ISONoise, MultiplicativeNoise, etc.
#    - Blur: GaussianBlur, MotionBlur, MedianBlur, etc.
#    - Advanced: ElasticTransform, GridDistortion, OpticalDistortion, etc.
#
# 3. For ultrasound-specific augmentations, consider:
#    - Speckle noise simulation
#    - Acoustic shadowing effects
#    - Custom scan region masking
#
# 4. Example custom augmentation function:
#
#    def get_custom_transforms(split: str) -> A.Compose:
#        if split == 'train':
#            return A.Compose([
#                A.Resize(256, 256),
#                A.RandomCrop(224, 224),
#                A.OneOf([
#                    A.ElasticTransform(p=1),
#                    A.GridDistortion(p=1),
#                    A.OpticalDistortion(p=1),
#                ], p=0.3),
#                A.Normalize(),
#                ToTensorV2()
#            ])
#        ...
#
# 5. Then modify get_transforms() to use your custom function
#
# =============================================================================
