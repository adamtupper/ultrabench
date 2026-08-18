# TN3K Dataset

TN3K ([GitHub](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation), [Hugging Face](https://huggingface.co/datasets/haifan-gong/TN3K)) is a dataset for thyroid nodule classification and segmentation. It consists of 3,493 thyroid ultrasound images from 2,421 patients that are divided into 2,879 training and 614 test images. Each image is accompanied by a binary segmentation mask indicating tumorous and non-tumorous regions, and a binary label indicating the pathology (benign or malignant). This version of the dataset was created for UltraBench 2 to support the evaluation of ultrasound models for thyroid nodule classification and segmentation.

> [!IMPORTANT]
> When using this dataset, please ensure that you cite the original work:
>
> Gong, H., Chen, J., Chen, G., Li, H., Li, G., & Chen, F. (2023). Thyroid region prior guided attention for ultrasound segmentation of thyroid nodules. Computers in Biology and Medicine, 155, 106389. https://doi.org/10.1016/j.compbiomed.2022.106389


## Splits

The dataset is split according to the original `train` and `test` splits, with the `train` split subsampled to a maximum of 1,000 images:

- `train`: 1,000 of the original 2,879 images, in their original sizes.
- `test`: 614 images in their original sizes.
- `test.256`: The same images and masks as the `test` split, but resized such that the longest edge measures 256 px and zero-padded (if needed) so that the final image measured 256 x 256 px.

Images are removed from the largest classes first, which makes the class distribution more uniform and keeps the rarer malignant nodules. The original training split holds 1,905 benign and 974 malignant nodules; the capped split holds 500 of each. The removed images are discarded rather than moved to the `test` split, because the images lack patient identifiers and moving them could introduce patient overlap between the splits. The `test` split is therefore unchanged from the original.

There is no patient overlap between the train and test sets. However, since the images lack patient identifiers, any further splits (e.g., training into train/validation) cannot guarantee that there is no overlap.

## Revisions

The dataset is versioned with Git tags on the Hugging Face Hub. Pin a revision to keep an experiment reproducible:

- `0.2`: The `train` split is subsampled to 1,000 images (500 benign and 500 malignant). The `test` split is unchanged.
- `0.1`: The original 2,879 training images (1,905 benign and 974 malignant).

## Features

All splits have the following features:

| Feature       | Feature Type      | Description                                                           |
| ------------- | ----------------- | --------------------------------------------------------------------- |
| `filename`    | `Value("string")` | The original filenames for the image (without the file extension).    |
| `image`       | `Image`           | The 3-channel ultrasound images.                                      |
| `nodule_mask` | `Image`           | The single-channel, binary nodule masks (background = 0, nodule = 1). |
| `scan_mask`   | `Image`           | Single-channel, binary ultrasound scan mask (all ones; the scan fills the entire image for linear-probe acquisitions). |
| `label`       | `ClassLabel`      | The pathology of the nodule(s) (benign = 0, malignant =  1).          |

## Tasks

The TN3K dataset supports three tasks:

- Thyroid nodule segmentation
- Thyroid nodule classification

## Usage

The easiest way to use this dataset is with [Hugging Face Datasets library](https://huggingface.co/docs/datasets/index):

```python
from datasets import load_dataset

train_data = datasets.load_dataset("UltraBench/TN3K", revision="0.2", split="train")
test_data = datasets.load_dataset("UltraBench/TN3K", revision="0.2", split="test.256")
```