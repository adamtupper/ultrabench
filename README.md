# UltraBench

UltraBench is a standardized benchmark of 21 convex ultrasound image classification and semantic segmentation tasks drawn from 12 publicly available datasets covering 10 anatomical regions. This repository contains the datasets, the backbones and task heads of the evaluated models, and the evaluation layer used to score machine learning models on them, plus example training scripts.

## Installation

Install the datasets and the evaluation layer:

```bash
pip install ultrabench
```

Add the `models` extra to also install the backbones and task heads in `ultrabench.models`, which need PyTorch:

```bash
pip install "ultrabench[models]"
```

To develop UltraBench, install it from source. `uv sync` installs the `dev` and `examples` dependency groups:

```bash
uv sync
source .venv/bin/activate
nbstripout --install
pre-commit install
```

## Getting Started

Load any dataset in the collection with one function:

```python
import ultrabench

train_data = ultrabench.load_dataset(name="AUL", split="train", revision="0.4")
```

Most datasets are published preprocessed and download ready to use. A few cannot be republished in that form, so `load_dataset` preprocesses them on your machine the first time and caches the result. Two of those cannot be republished at all, so you download the raw data yourself and pass its directory:

```python
train_data = ultrabench.load_dataset(name="GBCU", path="path/to/raw/data")
```

See [the dataset documentation](docs/dataset_format.md) for the three kinds of dataset, where the cache lives, and how to pin a revision.

Train with any framework you like, and write your test predictions to a CSV for the evaluation layer. The [`examples/`](examples/) directory holds two short PyTorch scripts, one for classification and one for segmentation, that train a head on a frozen backbone and score the result.

## Leaderboard

The following table ranks the evaluated ultrasound foundation models by their average score across the four task categories. Each score is the mean over the tasks in that category, measured with macro F1 for classification and positive Dice for segmentation, under the full fine-tuning paradigm. The best score in each category is bold.

Read the scores of a model pretrained on data that overlaps a category's tasks with care. See [the evaluation protocol](docs/evaluation_protocol.md) for the task categories, the metrics, and how the scores are aggregated.

| Model | Pathological (classification) | Non-pathological (classification) | Pathological (segmentation) | Anatomical (segmentation) |
|-------|-------------------------------|-----------------------------------|-----------------------------|---------------------------|
| URFM-L | **0.645** | **0.757** | 0.782 | 0.822 |
| URFM-B | 0.620 | 0.744 | 0.770 | 0.821 |
| OpenUS | 0.606 | 0.735 | 0.763 | 0.822 |
| DINOv3 ViT-L | 0.583 | 0.710 | 0.774 | 0.815 |
| DINOv3 ViT-B | 0.574 | 0.716 | 0.768 | 0.811 |
| USF-MAE | 0.600 | 0.735 | 0.717 | 0.813 |
| Deblurring ConvMAE | 0.600 | 0.683 | 0.759 | 0.811 |
| DINOv3 ConvNeXt-L | 0.552 | 0.670 | **0.794** | 0.822 |
| DINOv3 ConvNeXt-B | 0.551 | 0.677 | 0.780 | 0.820 |
| Deblurring MAE | 0.589 | 0.679 | 0.747 | 0.810 |
| UltraFedFM | 0.617 | 0.684 | 0.715 | 0.807 |
| USFM | 0.575 | 0.678 | 0.745 | 0.813 |
| SAMUS | 0.564 | 0.644 | 0.776 | **0.826** |
| UltraSAM | 0.584 | 0.679 | 0.724 | 0.809 |
| EchoCare | 0.549 | 0.671 | 0.712 | 0.801 |
| TinyUSFM | 0.540 | 0.641 | 0.690 | 0.801 |

## Submissions

Add a model to the leaderboard by opening a pull request against `main`. A maintainer reviews every submission before merging it.

Produce the scores with the evaluation layer, following [the evaluation protocol](docs/evaluation_protocol.md):

1. Train the model on every task of the benchmark under the full fine-tuning paradigm, with five seeds per task. A category score covers all of the tasks in that category, so a missing task leaves the category unscored.
2. Write the test split predictions of each run to a prediction CSV, and lay the CSVs out as one directory per task. See [the evaluation documentation](docs/evaluation.md) for the format.
3. Score the submission with `python -m ultrabench.evaluate.protocol predictions=<your directory>`. It writes the per-task and per-category scores with their error bars, and prints the leaderboard row to paste into the table.

Your pull request adds one row to the [leaderboard](#leaderboard) table:

- Report the four category scores to three decimal places.
- Place the row in the position its average of the four scores earns it.
- Link the model name to the public repository for the model, so that readers reach the weights and the code that produced the scores.
- Move the bold in any column your model now leads, so that each column still marks its best score.

Describe how you trained the model in the pull request, and state which benchmark datasets, if any, appear in its pretraining data. A reviewer checks the row against the linked repository and asks for the per-task scores where a number needs verifying.

## Current Datasets & Tasks

The following table contains the list of supported datasets and tasks. Click on the article and data links to visit the original publications.

The **Access** column says how each dataset reaches you. *Processed* datasets are published ready to use. *Raw* datasets may be redistributed unmodified but not preprocessed, so `load_dataset` downloads the original files and preprocesses them on your machine. *Local* datasets may not be redistributed at all, so you download them from the original source and pass the directory as `path`.

| Dataset | Access | Region | Classification Tasks | Segmentation Tasks |
|---------|--------|--------|----------------------|--------------------|
| Annotated Ultrasound Liver ([article](https://doi.org/10.1093/bib/bbac569), [data](https://zenodo.org/records/7272660)) | Processed | Liver | Liver mass classification | Liver segmentation, Liver mass segmentation |
| BUS-BRA ([article](https://doi.org/10.1002/mp.16812), [data](https://zenodo.org/records/8231412)) | Processed | Breast | Histology classification, Pathology classification | Breast lesion segmentation |
| CAMUS ([article](https://ieeexplore.ieee.org/document/8649738/), [data](https://www.creatis.insa-lyon.fr/Challenge/camus/index.html)) | Processed | Heart | Image quality classification | Cardiac region segmentation |
| COVID-BLUES ([article](https://doi.org/10.1109/TUFFC.2025.3535879), [data](https://github.com/NinaWie/COVID-BLUES)) | Raw | Lungs | COVID-19 classification | |
| Fetal Planes ([article](https://www.nature.com/articles/s41598-020-67076-5), [data](https://zenodo.org/records/3904280)) | Processed | Fetus | Plane classification, Brain plane classification | |
| Gallbladder Cancer Ultrasound ([article](https://ieeexplore.ieee.org/document/9879895), [data](https://gbc-iitd.github.io/data/gbcu)) | Local | Gallbladder | Gallbladder tumour classification | |
| MASLD ([article](https://link.springer.com/article/10.1007/s11548-018-1843-2), [data](https://zenodo.org/records/1009146)) | Processed | Liver | Steatosis classification | |
| MMOTU ([article](http://arxiv.org/abs/2207.06799), [data](https://github.com/cv516Buaa/MMOTU_DS2Net)) | Raw | Ovaries | Ovarian tumour classification | Ovarian tumour segmentation |
| Open Kidney Ultrasound ([article](https://link.springer.com/chapter/10.1007/978-3-031-44521-7_15), [data](https://rsingla.ca/kidneyUS/)) | Local | Kidney | | Kidney capsule segmentation, Kidney region segmentation |
| PSFHS ([article](https://www.nature.com/articles/s41597-024-03266-4), [data](https://zenodo.org/records/10969427)) | Processed | Fetus | | Fetal head and pubic symphysis segmentation |
| STMUS ([data](https://data.mendeley.com/datasets/3jykz7wz8d)) | Processed | Muscle | | Muscle segmentation |
| TN3K ([article](https://doi.org/10.1016/j.compbiomed.2022.106389), [data](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation)) | Processed | Thyroid | Thyroid nodule classification | Thyroid nodule segmentation |
