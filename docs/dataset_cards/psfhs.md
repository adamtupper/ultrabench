# Pubic Symphysis and Fetal Head Segmentation (PSFHS) Dataset

The PSFHS dataset is a dataset for fetal head and pubic symphysis segmentation with 1,358 images from 1,124 patients. Each image is accompanied by pixel-level segmentation masks for the fetal head and pubic symphysis, supporting a three-class image segmentation task (_background_, _pubic symphysis_, and _fetal head_). This version of the dataset was created for UltraBench 2 to support the evaluation of ultrasound models for pubic symphysis and fetal head segmentation.

> [!IMPORTANT]
> When using this dataset, please ensure that you cite the original works:
>
> Chen, G., Bai, J., Ou, Z. et al. PSFHS: Intrapartum ultrasound image dataset for AI-based segmentation of pubic symphysis and fetal head. Sci Data 11, 436 (2024). https://doi-org.acces.bibl.ulaval.ca/10.1038/s41597-024-03266-4
>
> Bai, J. (2024). PSFHS [Data set]. In PSFHS: Intrapartum ultrasound image dataset for AI-based segmentation of pubic symphysis and fetal head (Version V1). 26th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI), Vancouver/Canada. Zenodo. https://doi.org/10.5281/zenodo.10969427

## Splits

The original dataset has been divided into `train` and `test` splits using a stratified 80:20 split:

- `train`: 80% of the images in their original 256 x 256 px resolution.
- `test`: The remaining 20% of images in their original 256 x 256 px resolution.
- `test.256`: The remaining 20% of images in their original 256 x 256 px resolution.

Due to the lack of patient identifiers, there may be some patient overlap between the `train` and `test` splits.

## Features

All splits have the following features:

| Feature      | Feature Type      | Description                                                                                                |
| ------------ | ----------------- | ---------------------------------------------------------------------------------------------------------- |
| `filename`   | `Value("string")` | Original image filenames (without the file extension).                                                     |
| `image`      | `Image`           | 3-channel ultrasound images.                                                                               |
| `scan_mask`  | `Image`           | Single-channel, binary ultrasound scan masks generated using morphological operations.                     |
| `psfh_mask`  | `Image`           | Single-channel, multi-class PS-FH segmentation mask (background = 0, pubic symphysis = 1, fetal head = 2). |

## Tasks

The PSFHS dataset supports one task:

- Multi-class PS-FH segmentation

## Usage

The easiest way to use this dataset is with [Hugging Face Datasets library](https://huggingface.co/docs/datasets/index):

```python
from datasets import load_dataset

train_data = datasets.load_dataset("UltraBench/PSFHS", split="train")
test_data = datasets.load_dataset("UltraBench/PSFHS", split="test")
```
