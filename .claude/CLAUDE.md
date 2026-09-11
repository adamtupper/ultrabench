# Project: UltraBench 2 - A Standardized Benchmark for Evaluating Ultrasound Foundation Models
UltraBench 2 a standardized evaluation framework for assessing how well ultrasound foundation models (pretrained vision encoders, like ViT or CNN backbones) generalize to downstream segmentation and classification tasks. It provides:

1. **A dataset layer.** Standardized versions (e.g., pre-processing, splitting, interface, etc.) of common classification and segmentation datasets.
2. **A training layer.** PyTorch framework for downstream training using common techniques (e.g., linear probing, lightweight segmentation decoder, full fine-tuning, etc.).
3. **An evaluation layer.** Standardized, framework-independent evaluation layer with a CSV interface (à la Kaggle competitions) for calculating the performance metrics from CSV files of predictions and ground-truth annotations.

## Stack
- Python 3.10+
- PyTorch 2.10+
- uv package manager
- Hydra
- pytest

## Trusted Sources
- [Python Documentation](https://docs.python.org/3/)
- [PyTorch Documentation](https://pytorch.org/docs/stable/)
- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)
- [Torchvision Documentation](https://pytorch.org/vision/stable/)
- [Hydra Documentation](https://hydra.cc/docs/intro/)

## Style Guidelines
- Python: `.claude/rules/python.md`
- Markdown: `.claude/markdown.md`

## Architecture

* **Tech Stack**: Python 3.10+, uv, PyTorch and Hydra for the training layer, pytest for testing, Typer for the dataset layer CLI, and Hydra for the evaluation and training layer CLIs.
* **Project Structure**:
    * `data/`: Contains the metadata files for each split of each dataset. These are tracked to ensure that the examples included in training and test subsets do not change unintentionally between versions.
    * `design_specs/`: Design specifications for the different layers of the framework.
    * `docs/`: The end user documentation. Written using Markdown.
    * `examples/`: Contains examples of how to use the datasets with different machine learning frameworks.
    *  `notebooks/`: Jupyter notebooks
    * `src/`: The source code for the ultrabench2 package.
    * `tests/`: Contains all unit and integration tests.
    * `pyproject.toml`: Package configuration, including project dependencies.
