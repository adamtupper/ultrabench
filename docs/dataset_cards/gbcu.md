# Gallbladder Cancer Ultrasound (GBCU) Dataset

> [!CAUTION]
> This dataset must not be made public as it is licensed under a Research Usage Agreement. See [here](https://gbc-iitd.github.io/data/gbcu) for more details.

The GBCU dataset contains a total of 1,255 annotated abdominal ultrasound images (consisting of 432 normal, 558 benign, and 265 malignant images) collected from 218 patients (71 normal, 100 benign, and 47 malignant). This version of the dataset was created for UltraBench 2 to support the evaluation of ultrasound models for gallbladder tumour classification.

> [!IMPORTANT]
> When using this dataset, please ensure that you cite the original source:
>
> Basu, S., Gupta, M., Rana, P., Gupta, P., & Arora, C. (2022). Surpassing the Human Accuracy: Detecting Gallbladder Cancer from USG Images with Curriculum Learning. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 20854–20864. https://doi.org/10.1109/CVPR52688.2022.02022

## Splits

The dataset is split according to the original `train` and `test` splits, with the `train` split subsampled to a maximum of 1,000 images:

- `train`: 1,000 of the original 1,133 training images, in their original sizes.
- `test`: The original 122 test images in their original sizes.
- `test.256`: The same images and masks as the `test` split, but resized such that the longest edge measures 256 px and zero-padded (if needed) so that the final image measures 256 x 256 px.

Images are removed from the largest classes first, which makes the class distribution more uniform and keeps the rarer malignant cases. The original training split holds 401 normal, 509 benign and 223 malignant images; the capped split holds 388 normal, 389 benign and all 223 malignant images. The removed images are discarded rather than moved to the `test` split, because the images lack patient identifiers and moving them could introduce patient overlap between the splits. The `test` split is therefore unchanged from the original.

## Revisions

The dataset is versioned with Git tags on the Hugging Face Hub. Pin a revision to keep an experiment reproducible:

- `0.2`: The `train` split is subsampled to 1,000 images (388 normal, 389 benign and 223 malignant). The `test` split is unchanged.
- `0.1`: The original 1,133 training images (401 normal, 509 benign and 223 malignant).

## Features

All splits have the following features:

| Feature      | Feature Type      | Description                                                                            |
| ------------ | ----------------- | -------------------------------------------------------------------------------------- |
| `filename`   | `Value("string")` | Original image filenames (without the file extension).                                 |
| `label`      | `ClassLabel`      | Tumour malignancy (normal = 0, benign = 1, malignant = 2).                             |
| `image`      | `Image`           | 3-channel ultrasound images.                                                           |
| `scan_mask`  | `Image`           | Single-channel, binary ultrasound scan masks generated using morphological operations. |

## Tasks

The GBCU dataset supports one task:

- Multi-class tumour classification

## Usage

The easiest way to use this dataset is with [Hugging Face Datasets library](https://huggingface.co/docs/datasets/index):

```python
from datasets import load_dataset

train_data = datasets.load_dataset("UltraBench/GBCU", revision="0.2", split="train")
test_data = datasets.load_dataset("UltraBench/GBCU", revision="0.2", split="test.256")
```
