---
title: Getting Started
nav_order: 2
---

# Getting Started

## Installation

Install the datasets and the evaluation layer:

```bash
pip install ultrabench
```

Add the `models` extra to also install the backbones and task heads in `ultrabench.models`, which need PyTorch:

```bash
pip install "ultrabench[models]"
```

To develop UltraBench, see [Contributing]({% link contributing.md %}).

## Load a dataset

Load any dataset in the collection with one function:

```python
import ultrabench

train_data = ultrabench.load_dataset(name="AUL", split="train", revision="0.4")
```

Most datasets are published preprocessed and download ready to use. A few cannot be republished in that form, so `load_dataset` preprocesses them on your machine the first time and caches the result. Two of those cannot be republished at all, so you download the raw data yourself and pass its directory as `path`. See [Datasets]({% link dataset_format.md %}) for the three kinds of dataset, where the cache lives, and how to pin a revision.

## Train and evaluate

Train with any framework you like, and write your test predictions to a CSV for the evaluation layer. The [`examples/`](https://github.com/adamtupper/ultrabench/tree/main/examples) directory holds two short PyTorch scripts, one for classification and one for segmentation, that train a head on a frozen backbone and score the result.

- [Evaluation]({% link evaluation.md %}) describes the prediction and ground-truth CSV formats and how to score them.
- [The evaluation protocol]({% link evaluation_protocol.md %}) describes the task categories, the metrics, and how the scores are aggregated.
- [Submissions]({% link submissions.md %}) describes how to add your model to the [leaderboard]({% link index.md %}#leaderboard).
