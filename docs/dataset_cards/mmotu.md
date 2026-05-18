# Multi-Modality Ovarian Tumor 2D Ultrasound (MMOTU-2D) Dataset

The Multi-Modality Ovarian Tumor 2D Ultrasound (MMOTU-2D) dataset consists of 1469 ultrasound images each accompanied by semantic segmentation masks that identify the tumor in the image. In addition, each image is labeled according to the presence of each type of tumour (*chocolate cyst*, *serous cystadenoma*, *teratoma*, *theca cell tumour*, *simple cyst*, *normal ovary*, *mucinous cystadenoma*, and *high grade serous cystadenocarcinoma*). This version of the dataset was created for UltraBench 2 to support the evaluation of ultrasound models for ovarian tumor classification and segmentation.

> [!IMPORTANT]
> When using this dataset, please ensure that you cite the original work:
>
> Lyu, S., Zhao, Q., Bai, W., Cai, L., Cheng, G., Cui, G., Yang, M., Chen, L., & Zhou, H. (2026). Unsupervised cross-domain semantic segmentation on multi-modality ovarian tumor ultrasound data. Pattern Recognition, 171, 112311. https://doi.org/10.1016/j.patcog.2025.112311

## Splits

The dataset is split according to the original `train` and `test` splits with 1000 images (171 patients) and 469 images (76 patients), respectively:

- `train`: 1000 images from 171 patients in their original sizes.
- `test`: 469 images from 76 patients in their original sizes.
- `test.256`: The same images and masks as the `test` split, but resized such that the longest edge measures 256 px and zero-padded (if needed) so that the final image measured 256 x 256 px.

## Features

All splits have the following features:

| Feature      | Feature Type      | Description                                                                                |
| ------------ | ----------------- | ------------------------------------------------------------------------------------------ |
| `filename`   | `Value("string")` | Original image filenames (without the file extension).                                     |
| `image`      | `Image`           | 3-channel ultrasound images.                                                               |
| `scan_mask`  | `Image`           | The single-channel, binary ultrasound scan masks generated using morphological operations. |
| `tumor_mask` | `Image`           | Single-channel, binary tumor masks.                                                        |
| `tumor_type` | `ClassLabel`      | Tumor type (chocolate cyst = 0, serous cystadenoma = 1, teratoma = 2, theca cell tumour = 3, simple cyst = 4, normal ovary = 5, mucinous cystadenoma = 6, high grade serous cystadenocarcinoma = 7) |

## Tasks

The MMOTU-2D dataset supports two tasks:

- Tumor segmentation
- Tumor classification

## Usage

The easiest way to use this dataset is with [Hugging Face Datasets library](https://huggingface.co/docs/datasets/index):

```python
from datasets import load_dataset

train_data = datasets.load_dataset("UltraBench/MMOTU-2D", split="train")
test_data = datasets.load_dataset("UltraBench/MMOTU-2D", split="test.256")
```