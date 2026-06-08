import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import equalize_adapthist, rescale_intensity

############################## IMAGE PROCESSING ##############################


def detect_calipers(img):
    plus = np.load(
        "/raid/aradhachandran/thyroid/code/foundation_model/utils/thePlus.npy"
    )

    # Convert both images to grayscale if they have multiple channels
    img = np.squeeze(img)
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    if plus.ndim == 3:
        plus_gray = cv2.cvtColor(plus, cv2.COLOR_BGR2GRAY)
    else:
        plus_gray = plus

    # Ensure both are float32 and 2D
    img_gray = img_gray.astype("float32")
    plus_gray = plus_gray.astype("float32")

    # Make sure both are 2D arrays
    if img_gray.ndim > 2:
        img_gray = np.squeeze(img_gray)
    if plus_gray.ndim > 2:
        plus_gray = np.squeeze(plus_gray)

    res = cv2.matchTemplate(img_gray, plus_gray, cv2.TM_CCOEFF_NORMED)
    threshold = 0.4
    flag = False
    flag_counts = 0
    if res.max() > threshold:
        flag = True
        flag_counts += 1
    return flag, flag_counts


def how_much_color(img, color_threshold=15, return_largest=False):
    # Ensure 3-channel image
    if len(img.shape) == 2:
        return img  # already grayscale

    # Convert to float and separate channels
    img_float = img.astype(np.float32)
    b, g, r = cv2.split(img_float)

    # Compute deviation from grayscale
    mean_channel = (r + g + b) / 3.0
    color_diff = np.sqrt(
        ((r - mean_channel) ** 2 + (g - mean_channel) ** 2 + (b - mean_channel) ** 2)
    )

    # Normalize and threshold for "colored" pixels
    color_mask = (color_diff > color_threshold).astype(np.uint8)

    if return_largest:
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            color_mask, connectivity=8
        )

        if num_labels > 1:  # If there are connected components (excluding background)
            # Get areas of all components (excluding background which is label 0)
            component_areas = stats[1:, cv2.CC_STAT_AREA]
            largest_area = np.max(component_areas)
        else:
            largest_area = 0

        # Calculate fraction of largest component relative to frame size
        colored_fraction = largest_area / color_mask.size

    else:
        # Compute fraction of colored area
        colored_fraction = np.sum(color_mask) / color_mask.size

    return colored_fraction


def how_much_color_in_mask(img, color_threshold=15, return_largest=False, in_mask=None):
    """
    Calculate the fraction of colored pixels in an image.

    Args:
        img: Input image (grayscale or color)
        color_threshold: Threshold for color deviation to consider a pixel "colored"
        return_largest: If True, return fraction of largest connected component instead of all colored pixels
        in_mask: Binary mask of same shape as img. If provided, calculation is done only within masked region

    Returns:
        float: Fraction of colored pixels (or largest component) in the image (or masked region)
    """
    # Ensure 3-channel image
    if len(img.shape) == 2:
        return 0.0  # grayscale image has no color

    # Ensure in_mask is binary and same shape
    if in_mask.shape[:2] != img.shape[:2]:  # ty:ignore[unresolved-attribute]
        print(
            f"Mask shape {in_mask.shape[:2]} doesn't match image shape {img.shape[:2]}"  # ty:ignore[unresolved-attribute]
        )
        # Crop img from the center to match mask
        h_img, w_img = img.shape[:2]
        h_mask, w_mask = in_mask.shape[:2]  # ty:ignore[unresolved-attribute]
        start_h = (h_img - h_mask) // 2
        start_w = (w_img - w_mask) // 2
        img = img[start_h : start_h + h_mask, start_w : start_w + w_mask]

    # Convert to float and separate channels
    img_float = img.astype(np.float32)
    b, g, r = cv2.split(img_float)

    # Compute deviation from grayscale
    mean_channel = (r + g + b) / 3.0
    color_diff = np.sqrt(
        ((r - mean_channel) ** 2 + (g - mean_channel) ** 2 + (b - mean_channel) ** 2)
    )

    # Normalize and threshold for "colored" pixels
    color_mask = (color_diff > color_threshold).astype(np.uint8)

    # Apply input mask if provided
    if in_mask is not None:
        # Convert to binary if needed
        binary_mask = (in_mask > 0).astype(np.uint8)

        # Apply mask: only consider pixels where in_mask is True
        color_mask = color_mask * binary_mask
        total_pixels = np.sum(binary_mask)  # Total pixels in masked region

        if total_pixels == 0:
            return 0.0  # No pixels in mask
    else:
        total_pixels = color_mask.size  # Total pixels in image

    if return_largest:
        # Find connected components within the (possibly masked) color_mask
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            color_mask, connectivity=8
        )

        if num_labels > 1:  # If there are connected components (excluding background)
            # Get areas of all components (excluding background which is label 0)
            component_areas = stats[1:, cv2.CC_STAT_AREA]
            largest_area = np.max(component_areas)
        else:
            largest_area = 0

        # Calculate fraction of largest component relative to total pixels (image or masked region)
        colored_fraction = largest_area / total_pixels

    else:
        # Compute fraction of colored area relative to total pixels (image or masked region)
        colored_fraction = np.sum(color_mask) / total_pixels

    return colored_fraction


def remove_colored_regions(
    img, color_threshold=15, area_threshold=0.05, inpaint_radius=3
):
    """
    Detects colored regions in an image and inpaints them if they cover a small portion.
    If the image is grayscale or visually grayscale, returns it as-is.

    Args:
        img (np.ndarray): Input image (grayscale or RGB/BGR).
        color_threshold (float): Threshold for color deviation to consider a pixel "colored".
        area_threshold (float): Fraction of image area above which we skip inpainting (e.g., >1% colored).
        inpaint_radius (int): Radius for inpainting.

    Returns:
        np.ndarray: Processed image with colored regions removed/inpainted if appropriate.
    """

    # Ensure 3-channel image
    if len(img.shape) == 2:
        return img  # already grayscale

    # Convert to float and separate channels
    img_float = img.astype(np.float32)
    b, g, r = cv2.split(img_float)

    # Compute deviation from grayscale
    mean_channel = (r + g + b) / 3.0
    color_diff = np.sqrt(
        ((r - mean_channel) ** 2 + (g - mean_channel) ** 2 + (b - mean_channel) ** 2)
    )

    # Normalize and threshold for "colored" pixels
    color_mask = (color_diff > color_threshold).astype(np.uint8)

    # Compute fraction of colored area
    colored_fraction = np.sum(color_mask) / color_mask.size

    # If almost all pixels are "colored", treat as normal color image
    if colored_fraction > area_threshold:
        # Inpaint only small colored artifacts; if too much color, likely a real color image
        print(
            f"Image has {colored_fraction * 100:.2f}% colored area — skipping inpainting."
        )
        # plt.figure()
        # plt.imshow(img)
        # plt.show()
        return img

    # Otherwise, inpaint those small colored spots
    color_mask = cv2.medianBlur(color_mask, 3)
    inpainted = cv2.inpaint(img, color_mask, inpaint_radius, cv2.INPAINT_TELEA)

    return inpainted


def convert_to_grayscale(image_arr):
    if image_arr.ndim == 3 and image_arr.shape[2] == 3:
        gray = cv2.cvtColor(image_arr, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_arr  # already grayscale

    return np.squeeze(gray)


def autocrop(image_arr, threshold=0, ret_coords=False, mask=None):
    """
    Crops any edges below or equal to threshold. Returns cropped image.
    """
    # print(image_arr.shape)
    assert len(image_arr.shape) == 2
    if mask is not None:
        assert len(mask.squeeze().shape) == 2
    # print(flatImage.shape)
    rows = np.where(np.max(image_arr, 0) > threshold)[0]
    # print(rows.shape, rows.size)
    if rows.size:
        cols = np.where(np.max(image_arr, 1) > threshold)[0]
        image_arr = image_arr[cols[0] : cols[-1] + 1, rows[0] : rows[-1] + 1]
        if mask is not None:
            mask = mask[cols[0] : cols[-1] + 1, rows[0] : rows[-1] + 1]
        # mask = mask[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image_arr = image_arr[:1, :1]
        if mask is not None:
            mask = mask[:1, :1]

    if ret_coords:
        if rows.size != 0:
            # print(cols[0], cols[-1] + 1, rows[0], rows[-1] + 1)
            return cols[0], cols[-1] + 1, rows[0], rows[-1] + 1
        else:
            return 0, image_arr.shape[1], 0, image_arr.shape[0]
    else:
        if mask is not None:
            return image_arr, mask
        else:
            return image_arr


def intensity_rescale(image_arr, within_mask=None):
    """
    Rescales image intensities to the 2nd and 98th percentiles within the mask if provided.
    """
    if within_mask is None:
        within_mask = np.ones_like(image_arr)
    within_mask = within_mask.astype(bool)

    # print(image_arr.shape, within_mask.shape)
    p2, p98 = np.percentile(image_arr[within_mask], (2, 98))
    ret_img = rescale_intensity(image_arr, in_range=(p2, p98))

    return ret_img


def normalize_0to1(image_arr, within_mask=None):
    """
    Normalize image to [0, 1] range within the mask if provided.
    """
    if within_mask is None:
        within_mask = np.ones_like(image_arr)
    within_mask = within_mask.astype(bool)

    max_value, min_value = (
        np.max(image_arr[within_mask]),
        np.min(image_arr[within_mask]),
    )
    ret_img = (image_arr - min_value) / (max_value - min_value)

    return ret_img


def triple_stack(image_arr):
    """
    Convert a single-channel grayscale image to a 3-channel image by stacking.
    """
    if image_arr.ndim == 2:
        stacked = np.stack([image_arr] * 3, axis=-1)
    elif image_arr.ndim == 3 and image_arr.shape[2] == 1:
        stacked = np.concatenate([image_arr] * 3, axis=-1)
    else:
        stacked = image_arr

    return stacked


def crop_to_usregion(image_arr, within_mask=None, expand_crop=10):
    """
    Crop image to the bounding box with additional padding of the ultrasound region defined by within_mask.
    """
    if within_mask is None:
        return image_arr

    rows = np.where(np.max(within_mask, 0) > 0)[0]
    if rows.size:
        cols = np.where(np.max(within_mask, 1) > 0)[0]

        # Add boundary checks to prevent negative indices and out-of-bounds access
        h, w = image_arr.shape[:2]
        row_start = max(0, cols[0] - expand_crop)
        row_end = min(h, cols[-1] + 1 + expand_crop)
        col_start = max(0, rows[0] - expand_crop)
        col_end = min(w, rows[-1] + 1 + expand_crop)

        cropped = image_arr[row_start:row_end, col_start:col_end]

        # cropped = image_arr[
        #     cols[0] - expand_crop: cols[-1] + 1 + expand_crop,
        #     rows[0] - expand_crop: rows[-1] + 1 + expand_crop]
    else:
        cropped = image_arr[:1, :1]

    return cropped


def process_pipeline(
    image_arr, operations=[], within_mask=None, glimpse=False, mask=None
):
    """
    Process image through a series of operations specified in the operations list.
    """
    if glimpse:
        plt.figure(figsize=(3, 3))
        plt.title("Original")
        plt.imshow(image_arr, cmap="gray")
        plt.axis("off")
        plt.show()

    for operation in operations:
        func = globals().get(operation)
        if callable(func):
            if (
                operation in ["intensity_rescale", "normalize_0to1", "crop_to_usregion"]
                and within_mask is not None
            ):
                image_arr = func(image_arr, within_mask=within_mask)
            else:
                if operation in ["autocrop"] and mask is not None:
                    image_arr, mask = func(image_arr, mask=mask)
                else:
                    image_arr = func(image_arr)

            if glimpse:
                plt.figure(figsize=(2, 3))
                plt.title(operation)
                plt.imshow(image_arr, cmap="gray")
                plt.axis("off")
                plt.show()
        else:
            raise ValueError(f"Operation {operation} not recognized.")

    if mask is not None:
        assert image_arr.shape[0] == mask.shape[0]
        assert image_arr.shape[1] == mask.shape[1]
        return image_arr, mask
    else:
        return image_arr


##################################################################################
############################## US REGION EXTRACTION ##############################
##################################################################################


def extract_ultrasound_region(image_arr, dataset_name):  # THIS KINDA WORKS
    # the mask is just a circle that touches the center of each side
    if dataset_name == "prosus_tcia":
        h, w = image_arr.shape[:2]
        center = (w // 2, h // 2)
        radius = min(center[0], center[1], w - center[0], h - center[1])
        Y, X = np.ogrid[:h, :w]
        mask = (X - center[0]) ** 2 + (Y - center[1]) ** 2 <= radius**2
        return mask.astype(np.uint8) * 255

    if len(image_arr.shape) == 3:
        gray = image_arr[:, :, 0]  # cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_arr

    # Handle both 0-1 and 0-255 ranges
    if gray.max() <= 1.0:
        threshold_value = 15 / 255.0
    else:
        threshold_value = 15

    # Convert to uint8 for OpenCV operations
    if gray.dtype != np.uint8:
        gray_uint8 = (gray * 255).astype(np.uint8)
    else:
        gray_uint8 = gray

    # Ensure the array is contiguous in memory
    gray_uint8 = np.ascontiguousarray(gray_uint8)

    # Threshold
    _, binary = cv2.threshold(
        gray_uint8, int(threshold_value * 255), 255, cv2.THRESH_BINARY
    )

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Create masks with proper data type and layout
    largest = max(contours, key=cv2.contourArea)  # ty:ignore[no-matching-overload]
    hull = cv2.convexHull(largest)

    # Create masks as contiguous uint8 arrays
    mask1 = np.zeros(gray_uint8.shape, dtype=np.uint8)
    mask1 = np.ascontiguousarray(mask1)
    cv2.drawContours(mask1, [hull], -1, 255, thickness=cv2.FILLED)

    mask2 = np.zeros(gray_uint8.shape, dtype=np.uint8)
    mask2 = np.ascontiguousarray(mask2)
    cv2.drawContours(mask2, [largest], -1, 255, thickness=cv2.FILLED)

    # Continue with morphological operations
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask2 = cv2.GaussianBlur(mask2, (5, 5), 0)
    _, mask2 = cv2.threshold(mask2, 127, 255, cv2.THRESH_BINARY)

    region_mask = np.logical_or(mask1, mask2).astype(np.uint8) * 255

    return region_mask


def extract_ultrasound_region_more(image_arr, dataset_name):
    # Dataset-specific circular mask
    if dataset_name == "prosus_tcia":
        h, w = image_arr.shape[:2]
        center = (w // 2, h // 2)
        radius = min(center[0], center[1], w - center[0], h - center[1])
        Y, X = np.ogrid[:h, :w]
        mask = (X - center[0]) ** 2 + (Y - center[1]) ** 2 <= radius**2
        return mask.astype(np.uint8) * 255

    # Convert to grayscale
    gray = image_arr[..., 0] if image_arr.ndim == 3 else image_arr

    # Normalize dynamic range
    if gray.max() <= 1.0:
        gray_uint8 = (gray * 255).astype(np.uint8)
    else:
        gray_uint8 = gray.astype(np.uint8)

    gray_uint8 = np.ascontiguousarray(gray_uint8)

    # Threshold
    _, binary = cv2.threshold(gray_uint8, 15, 255, cv2.THRESH_BINARY)

    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Sort by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)  # ty:ignore[no-matching-overload]
    areas = [cv2.contourArea(c) for c in contours]

    # Consider multiple large contours if sizes are similar
    if areas[0] != 0:
        if len(areas) > 1 and (areas[1] / areas[0]) > 0.5:
            main_contours = contours[:2]
        else:
            main_contours = [contours[0]]
    else:
        mask = np.ones_like(image_arr)
        mask = mask.astype(bool)
        return mask
    # if len(areas) > 1 and (areas[1] / areas[0]) > 0.5:
    #     main_contours = contours[:2]
    # else:
    #     main_contours = [contours[0]]

    # Fill selected contours
    mask = np.zeros(gray_uint8.shape, dtype=np.uint8)
    cv2.fillPoly(mask, main_contours, 255)

    # Merge potential gap between them
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))

    # Optional: convex hull smoothing
    contours_combined, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours_combined:
        all_pts = np.vstack(contours_combined)
        hull = cv2.convexHull(all_pts)
        mask = np.zeros_like(mask)
        cv2.fillPoly(mask, [hull], 255)

    # Final smoothing
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # make it 0s and 1s
    mask = mask.astype(bool)

    return mask


def extract_ultrasound_region_improved(image_arr, full_frame_thresh=0.95):
    # --- 1. Grayscale & normalization ---
    if image_arr.ndim == 3:
        gray = image_arr[..., 0]
    else:
        gray = image_arr
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # ty:ignore[no-matching-overload]

    # --- 2. Contrast enhancement ---
    gray_eq = equalize_adapthist(gray, clip_limit=0.02)
    gray_eq = (gray_eq * 255).astype(np.uint8)

    # --- 3. Add texture info ---
    lap = cv2.Laplacian(gray_eq, cv2.CV_32F, ksize=3)
    texture = cv2.normalize(np.abs(lap), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)  # ty:ignore[no-matching-overload]
    combined = cv2.addWeighted(gray_eq, 0.7, texture, 0.3, 0)

    # --- 4. Adaptive threshold ---
    _, binary = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # --- 5. Morphological cleanup ---
    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    # --- 6. Largest contour + convex hull ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)  # ty:ignore[no-matching-overload]
        hull = cv2.convexHull(largest)
        mask = np.zeros_like(mask)
        cv2.fillPoly(mask, [hull], 255)

    # --- 7. Full-frame detection ---
    filled_fraction = np.sum(mask > 0) / mask.size
    if filled_fraction > full_frame_thresh:
        mask[:] = 255

    # --- 8. Smooth final mask ---
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    return mask
