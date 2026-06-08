# BUS-BRA Dataset

The BUS-BRA dataset ([Zenodo](https://zenodo.org/records/8231412)) consists 1,875 breast ultrasound images from 1,064 patients. Each image is accompanied by biopsy-proven tumour labels (722 benign and 342 malignant cases), BI-RADS assessment categories (2 to 5) and segmentation masks of tumorous and non-tumorous (background) regions. This version of the dataset was created for UltraBench 2 to support the evaluation of ultrasound models for beast cancer screening tasks.

> [!IMPORTANT]
> When using this dataset, please ensure that you cite the original work:
>
> Gómez-Flores W, Gregorio-Calas MJ, Coelho de Albuquerque Pereira W. BUS-BRA: A breast ultrasound dataset for assessing computer-aided diagnosis systems. Med Phys. 2024;51:3110–3123. https://doi.org/10.1002/mp.16812

## Splits

The original dataset has been divided by patient into `train` and `test` splits using a 80:20 split:

- `train`: 80% of the patients and their images in their original sizes.
- `test`: The remaining 20% of patients and their images in their original sizes.
- `test.256`: The same images and masks as the `test` split, but resized such that the longest edge measures 256 px and zero-padded (if needed) so that the final image measured 256 x 256 px.

## Features

All splits have the following features:

| Feature           | Feature Type      | Description                                                           |
| ----------------- | ----------------- | --------------------------------------------------------------------- |
| `filename`        | `Value("string")` | Original filenames for the image (without the file extension).        |
| `patient`         | `Value("string")` | Patient IDs.                                                          |
| `side`            | `Value("string")` | The breast that was imaged (left or right)                       |
| `histology`       | `Value("string")` | The histopathology of the breast lesion                          |
| `histology_label` | `ClassLabel`      | The eight most common histopathologies and "other" (fibroadenoma = 0, invasive ductal carcinoma = 1, cyst = 2, fibrocystic changes = 3, invasive lobular carcinoma = 4, intraductal papilloma = 5, sclerosing adenosis = 6, hyperplasia = 7, other = 8) |
| `pathology`       | `ClassLabel`      | Pathology of the mass (benign = 0, malignant = 1)                     |
| `birads`          | `Value("int32)`   | BI-RADS scores.                                                       |
| `device`          | `Value("string")` | The device used.                                                      |
| `image`           | `Image`           | 3-channel ultrasound images.                                          |
| `tumor_mask`      | `Image`           | Single-channel, binary tumor masks (background = 0, tumor = 1).      |
| `scan_mask`       | `Image`           | Single-channel, binary ultrasound scan mask (all ones; the scan fills the entire image for linear-probe acquisitions). |

## Tasks

The BUS-BRA dataset supports three tasks:

- Breast tumor segmentation
- Breast tumor pathology classification
- Breast tumor histology classification

## Usage

The easiest way to use this dataset is with [Hugging Face Datasets library](https://huggingface.co/docs/datasets/index):

```python
from datasets import load_dataset

train_data = datasets.load_dataset("UltraBench/BUS-BRA", split="train")
test_data = datasets.load_dataset("UltraBench/BUS-BRA", split="test.256")
```