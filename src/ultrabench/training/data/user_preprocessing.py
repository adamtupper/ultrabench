"""
User-defined preprocessing functions for UltraBench2o

Preprocessing is applied BEFORE augmentations and handles:
- Normalization strategies
- Contrast enhancement
- Channel conversions
- Intensity clipping/scaling
- Any deterministic transformations applied on-the-fly during training/eval

Users can define custom preprocessing functions here.
Each function should accept (image, cfg) and return preprocessed image.
"""

from typing import Any, Optional

import numpy as np

from ultrabench.training.data.preprocessing_helpers import (
    extract_ultrasound_region_more,
    how_much_color,
    process_pipeline,
)


def custom_preprocessing(image: np.ndarray, cfg: Optional[Any] = None) -> np.ndarray:
    """
    Template for custom preprocessing

    Args:
        image: Input image as numpy array [H, W, C] or [H, W], typically uint8
            [0, 255]
        cfg: Optional Hydra config for accessing parameters

    Returns:
        Preprocessed image as numpy array

    Example usage:
        # 1. In this file, implement your preprocessing:
        def custom_preprocessing(image, cfg=None):
            # Handle grayscale conversion
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # Apply bilateral filter for smoothing
            image = cv2.bilateralFilter(image, 9, 75, 75)

            # Custom intensity normalization
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)

            return image

        # 2. In config or CLI:
        python main_launch.py data.preprocessing_fn=custom_preprocessing

    Notes:
    - This function is called BEFORE augmentations
    - Applied during DataLoader.__getitem__() for on-the-fly processing
    - Should return image in same format (numpy array)
    """

    color_threshold = (
        float(getattr(cfg, "color_threshold", 0.15)) if cfg is not None else 0.15
    )

    dataset_name = None
    if cfg is not None:
        dataset_name = getattr(cfg, "dataset_name", None)
        if dataset_name is None and hasattr(cfg, "data"):
            dataset_name = getattr(cfg.data, "dataset_name", None)

    image_array = np.asarray(image)
    # if image_array.ndim == 3 and image_array.shape[2] == 1:
    #     image_array = image_array[:, :, 0]

    # if image_array.dtype != np.uint8:
    #     if np.max(image_array) <= 1.0:
    #         image_array = (image_array * 255.0).astype(np.uint8)
    #     else:
    #         image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    summed = np.sum(image_array)
    if summed == 0 or np.isnan(summed):
        # if image_array.ndim == 2:
        #     stacked = np.stack([image_array] * 3, axis=-1)
        #     return stacked.astype(np.float32) / 255.0
        # if image_array.ndim == 3 and image_array.shape[2] >= 3:
        #     return (image_array[:, :, :3].astype(np.float32) / 255.0)
        return image_array.astype(np.float32)

    operations = [
        "remove_colored_regions",
        "convert_to_grayscale",
        "autocrop",
        "intensity_rescale",
        "normalize_0to1",
    ]

    if image_array.ndim == 2:
        operations.remove("remove_colored_regions")
    else:
        if image_array.shape[2] < 3:
            operations.remove("remove_colored_regions")

    if image_array.ndim == 3 and image_array.shape[2] >= 3:
        color = how_much_color(image_array[:, :, :3], return_largest=True)
        if color > color_threshold:
            operations = [op for op in operations if op != "remove_colored_regions"]

    # if image_array.ndim == 3 and image_array.shape[2] > 3:
    #     image_array = image_array[:, :, :3]
    # print("break5: ", image_array.shape)
    processed = process_pipeline(
        image_array,
        operations=operations,
        glimpse=False,
    )

    region_mask = extract_ultrasound_region_more(processed, dataset_name)

    operations_with_stack = operations + ["triple_stack"]
    processed = process_pipeline(
        image_array,
        operations=operations_with_stack,
        within_mask=region_mask,
        glimpse=False,
    )

    return processed.astype(np.float32)


# Registry of available preprocessing functions
PREPROCESSING_FUNCTIONS = {
    "no_preprocessing": lambda x: x,
    "custom_preprocessing": custom_preprocessing,
}


def get_preprocessing_fn(name: str):
    """
    Get preprocessing function by name

    Args:
        name: Name of preprocessing function

    Returns:
        Preprocessing function

    Raises:
        ValueError: If preprocessing function not found
    """
    if name not in PREPROCESSING_FUNCTIONS:
        available = ", ".join(PREPROCESSING_FUNCTIONS.keys())
        raise ValueError(
            f"Preprocessing function '{name}' not found. Available options: {available}"
        )
    return PREPROCESSING_FUNCTIONS[name]


def list_preprocessing_functions():
    """List all available preprocessing functions"""
    print("\nAvailable preprocessing functions:")
    print("=" * 50)
    for name, fn in PREPROCESSING_FUNCTIONS.items():
        doc = fn.__doc__.split("\n")[1].strip() if fn.__doc__ else "No description"
        print(f"  {name:30s} - {doc}")
    print("=" * 50)
