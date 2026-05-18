# POCUS Dataset

The Point-of-care Ultrasound (POCUS) dataset is a collection of convex and linear probe lung ultrasound images and videos from different sources that was created for the diagnosis of COVID-19. We use the convex probe videos images from the GitHub repository to extract 2,726 examples. Each image is labeled by the pathology (*normal*, *pneumonia*, *covid*) yielding a three-class classification problem. This version of the dataset was created for UltraBench 2 to support the evaluation of ultrasound models for COVID-19 classification.

> [!IMPORTANT]
> When using this dataset, please ensure that you cite the original source:
>
> Jannis Born, Nina Wiedemann, Manuel Cossio, Charlotte Buhre, Gabriel Brändle, Konstantin Leidermann, Julie Goulet, Avinash Aujayeb, Michael Moor, Bastian Rieck, and Karsten Borgwardt. Accelerating Detection of Lung Pathologies with Explainable Ultrasound Image Analysis. Applied Sciences, 11(2):672, January 2021. ISSN 2076-3417. doi: 10.3390/app11020672.

## Splits

We use the 142 convex probe videos and 29 convex probe images from the GitHub repository and follow the procedure described in their original paper to process them, sampling the videos at a rate of 3 Hz, up to a maximum of 30 frames, and grouping the frames by video to prevent data leakage between the train and test splits. In total, we extract 2,726 examples:

- `train`: 80% of patients with the images in their original resolution.
- `test`: The remaining 20% of patients with the images in their original resolution.
- `test.256`: The same images and masks as the `test` split, but resized such that the longest edge measures 256 px and zero-padded (if needed) so that the final image measures 256 x 256 px.

## Features

All splits have the following features:

| Feature      | Feature Type      | Description                                                                            |
| ------------ | ----------------- | -------------------------------------------------------------------------------------- |
| `filename`   | `Value("string")` | For still images: the image filename stem. For video frames: `{video_stem}_{frame_index}`. Always unique per example. |
| `source`     | `Value("string")` | Original source filename (with extension). All frames extracted from the same video share this value. Useful for grouping when splitting training data into train/validation sets. |
| `label`      | `ClassLabel`      | Pathology (normal = 0, pneumonia = 1, covid = 2).                                      |
| `image`      | `Image`           | 3-channel ultrasound images.                                                           |
| `scan_mask`  | `Image`           | Single-channel, binary ultrasound scan masks generated using morphological operations. |

## Tasks

The POCUS dataset supports one task:

- Lung disease classification

## Usage

The easiest way to use this dataset is with [Hugging Face Datasets library](https://huggingface.co/docs/datasets/index):

```python
from datasets import load_dataset

train_data = datasets.load_dataset("UltraBench/POCUS", split="train")
test_data = datasets.load_dataset("UltraBench/POCUS", split="test")
```