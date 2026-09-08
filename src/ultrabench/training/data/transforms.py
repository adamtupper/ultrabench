"""Data augmentation and optional preprocessing transforms using
Albumentations.

Users can customize via adding custom augmentation strategies and/or
preprocessing functions. Geometric transforms must not be applied to
test/validation sets to ensure consistent evaluation.
"""

import logging
from typing import Any, Callable, Optional

import albumentations as A
import cv2
import numpy as np
from albumentations.augmentations.geometric import functional as fgeometric
from albumentations.pytorch import ToTensorV2
from PIL import Image
from usaugment.albumentations import DepthAttenuation, GaussianShadow, HazeArtifact

from ultrabench.training.data.user_preprocessing import get_preprocessing_fn

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


class AdaptiveLongestMaxSize(A.LongestMaxSize):
    """LongestMaxSize that adapts the image kernel to the resize direction.

    cv2.INTER_AREA is the right kernel for shrinking, but above 1x OpenCV
    collapses it towards nearest-neighbour sampling, so enlargements use
    cv2.INTER_CUBIC instead. This mirrors the rule that
    ultrabench.process.utils.resize_and_pad applies when it builds the
    test.256 splits offline, so a training image and its published
    test.256 counterpart are resized identically.

    Masks are untouched: the inherited apply_to_mask uses
    mask_interpolation, which stays cv2.INTER_NEAREST in both directions.

    Albumentations gained an equivalent area_for_downscale parameter in
    2.0.7; this class can be dropped if the pin is ever raised.
    """

    def apply(self, img: np.ndarray, scale: float, **params: Any) -> np.ndarray:
        """Resize an image, choosing the kernel from the resize direction.

        Args:
            img: The image to resize, of shape (H, W) or (H, W, C).
            scale: The factor by which both edges are scaled.
            **params: The remaining transform parameters, unused.

        Returns:
            The resized image.
        """
        interpolation = self.interpolation
        if interpolation == cv2.INTER_AREA and scale >= 1:
            interpolation = cv2.INTER_CUBIC
        height, width = img.shape[:2]
        new_shape = (max(1, round(height * scale)), max(1, round(width * scale)))

        return fgeometric.resize(img, new_shape, interpolation=interpolation)


def get_mmotu_ub_transforms(img_size: int = 224) -> list:
    """TrivialAugment-style strategy specific to the MMOTU dataset tasks.

    Includes a subset of augmentations found to be effective for MMOTU tasks in
    Revisiting Data Augmentation for Ultrasound Images.

    Returns:
        List of Albumentations transforms.
    """
    return [
        A.OneOf(
            [
                A.Rotate(limit=30, border_mode=0, fill=0, p=1.0),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0,
                    rotate_limit=0,
                    interpolation=1,
                    border_mode=0,
                    fill=0,
                    p=1.0,
                ),
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.6, 1.0), p=1.0),
                A.GaussNoise(
                    (0, 0.0225),
                    mean_range=(0.0, 0.0),
                    per_channel=False,
                    noise_scale_factor=1,
                    p=1.0,
                ),
            ],
            p=1.0,
        ),
    ]


def get_general_ub_transforms(img_size: int = 224) -> list:
    """TrivialAugment-style strategy with domain-agnostic augmentations.

    These augmentations that were found to be effective across a wide variety
    of ultrasound tasks, as found in Revisiting Data Augmentation for
    Ultrasound Images.

    Returns:
        List of Albumentations transforms.
    """
    return [
        A.OneOf(
            [
                A.NoOp(p=1.0),
                A.Rotate(limit=30, border_mode=0, fill=0, p=1.0),
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0,
                    rotate_limit=0,
                    interpolation=1,
                    border_mode=0,
                    fill=0,
                    p=1.0,
                ),
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.6, 1.0), p=1.0),
                A.GaussNoise(
                    std_range=(0.0, 0.0225),
                    mean_range=(0.0, 0.0),
                    per_channel=False,
                    noise_scale_factor=1,
                    p=1.0,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=1.0
                ),
            ],
            p=1.0,
        ),
    ]


def get_usaugment_transforms() -> list:
    """Ultrasound-specific augmentations from USAugment.

    Returns:
        List of Albumentations transforms.
    """

    return [
        A.HorizontalFlip(p=0.5),
        DepthAttenuation(p=0.5),
        GaussianShadow(p=0.5),
        HazeArtifact(p=0.5),
        # SpeckleReduction(p=0.5),  # TODO: Should this be reenabled?
    ]


def get_train_test_transforms(
    image_size: int, augmentation_strategy: str | None = None
) -> tuple[A.Compose, A.Compose]:
    """Get default train and test transforms.

    The transforms should return uint8 image tensors. The model is responsible
    for resizing and normalizing the images internally in forward() to account
    for differences in native input resolution and normalization schemes.

    Args:
        image_size: Target spatial size in pixels for images and masks.
        augmentation_strategy: Optional string to specify which augmentation
            strategy to use during training. One of 'general', 'mmotu',
            'usaugment', or None. Default is None (no augmentation).

    Returns:
        Tuple of (train_transforms, test_transforms) as Albumentations Compose
        objects.
    """
    match augmentation_strategy:
        case "general":
            train_augmentations = get_general_ub_transforms(img_size=image_size)
        case "mmotu":
            train_augmentations = get_mmotu_ub_transforms(img_size=image_size)
        case "usaugment":
            train_augmentations = get_usaugment_transforms()
        case None:
            train_augmentations = []
        case _:
            raise ValueError(
                f"Unknown augmentation strategy: {augmentation_strategy}. "
            )

    transform_train = A.Compose(
        [
            AdaptiveLongestMaxSize(
                max_size=image_size,
                interpolation=cv2.INTER_AREA,
                mask_interpolation=cv2.INTER_NEAREST,
            ),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
                fill=0,
            ),
            *train_augmentations,
            ToTensorV2(),
        ],
        additional_targets={"scan_mask": "mask"},
    )

    transform_test = A.Compose(
        [
            AdaptiveLongestMaxSize(
                max_size=image_size,
                interpolation=cv2.INTER_AREA,
                mask_interpolation=cv2.INTER_NEAREST,
            ),
            A.PadIfNeeded(
                min_height=image_size,
                min_width=image_size,
                border_mode=0,
                fill=0,
            ),
            ToTensorV2(),
        ],
        additional_targets={"scan_mask": "mask"},
    )

    return transform_train, transform_test


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
# 1. Modify the get_classification_transforms() or
#    get_segmentation_transforms() functions above to add/remove/modify
#    augmentations
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
