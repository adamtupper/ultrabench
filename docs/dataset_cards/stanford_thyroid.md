# Stanford Thyroid Dataset

> [!CAUTION]
> This dataset must not be made public as it is licensed under a Research Usage Agreement. See [here](https://stanfordaimi.azurewebsites.net/datasets/a72f2b02-7b53-4c5d-963c-d7253220bfd5) for more details.

The Stanford Thyroid Ultrasound Cine-clip dataset consists of 192 thyroid nodule ultrasound cine-clips (videos) collected from 167 patients for a total of 17,412 images. The images in each sequence are associated with pixel-level nodule segmentation masks, patient demographics, lesion size and location, TI-RADS descriptors, and histopathological diagnoses. We use the nodule masks for a thyroid nodule segmentation task. This version of the dataset was created for UltraBench 2 to support the evaluation of ultrasound models for thyroid nodule segmentation.

> [!IMPORTANT]
> When using this dataset, please ensure that you cite the original source:
>
> Stanford AIMI Center. Thyroid Ultrasound Cine-clip Dataset, October 2021. URL https://stanfordaimi.azurewebsites.net/datasets/a72f2b02-7b53-4c5d-963c-d7253220bfd5.

## Splits

The patients are split between `train` and `test` splits using an 80:20 split:

- `train`: 80% of patients with the images in their original resolution of 1054 × 802 px.
- `test`: The remaining 20% of patients with the images in their original resolution of 1054 × 802 px.
- `test.256`: The same images and masks as the `test` split, but resized such that the longest edge measures 256 px and zero-padded (if needed) so that the final image measures 256 x 256 px.

## Features

All splits have the following features:

| Feature                 | Feature Type       | Description                                                                            |
|-------------------------|--------------------|----------------------------------------------------------------------------------------|
| `filename`              | `Value("string")`  | Original image filenames (without the file extension).                                 |
| `patient`               | `Value("string")`  | The patient ID.                                                                        |
| `label`                 | `ClassLabel`       | Pathology (normal = 0, pneumoia = 1, covid = 2).                                       |
| `image`                 | `Image`            | 3-channel ultrasound images.                                                           |
| `scan_mask`             | `Image`            | Single-channel, binary ultrasound scan masks generated using morphological operations. |
| `tumor_mask`            | `Image`            | Single-channel, binary nodule/tumor segmentation mask.                                 |
| `pathology`             | `Value("string")   | The pathology of the lesion (benign or malignant).                                     |
| `age`                   | `Value("int32")`   | The age of the patient.                                                                |
| `sex`                   | `Value("string")`  | The sex of the patient.                                                                |
| `location`              | `Value("string")`  | The location of the lesion.                                                            |
| `lesion_size_x`         | `Value("float32")` | The size of the lesion in the x-direction.                                             |
| `lesion_size_y`         | `Value("float32")` | The size of the lesion in the y-direction.                                             |
| `lesion_size_z`         | `Value("float32")` | The size of the lesion in the z-direction.                                             |
| `tirads_level`          | `Value("int32")`   | The TIRADS level of the lesion.                                                        |
| `tirads_composition`    | `Value("string")`  | TIRADS composition descriptor.                                                         |
| `tirads_echogenicity`   | `Value("string")`  | TIRADS echogenicity descriptor.                                                        |
| `tirads_shape`          | `Value("string")`  | TIRADS shape descriptor.                                                               |
| `tirads_margin`         | `Value("string")`  | TIRADS margin descriptor.                                                              |
| `tirads_echogenicfoci`  | `Value("string")`  | TIRADS echogenic foci descriptor.                                                      |
| `histopath_diagnosis`   | `Value("string")`  | Histopathological diagnosis (when available).                                          |

## Tasks

The Stanford Thyroid dataset supports one task:

- Thyroid nodule segmentation

## Usage

The easiest way to use this dataset is with [Hugging Face Datasets library](https://huggingface.co/docs/datasets/index):

```python
from datasets import load_dataset

train_data = datasets.load_dataset("UltraBench/StanfordThyroid", split="train")
test_data = datasets.load_dataset("UltraBench/StanfordThyroid", split="test")
```