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

The [leaderboard](docs/index.md) ranks the evaluated ultrasound foundation models by the average of their four category scores, with a standard error for each score. It is built from [`docs/_data/leaderboard.csv`](docs/_data/leaderboard.csv). See [the evaluation protocol](docs/evaluation_protocol.md) for the task categories, the metrics, and how the scores are aggregated.

To add a model to the leaderboard, open a pull request that adds its row to the CSV. See [Submissions](docs/submissions.md) for how to produce the scores and fill in the row.

## Documentation

The documentation in [`docs/`](docs/) is a Jekyll site. Install Ruby and Bundler, then serve the site locally from `docs/`:

```bash
cd docs
bundle install
bundle exec jekyll serve --livereload
```

Open <http://localhost:4000> to view it.

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
