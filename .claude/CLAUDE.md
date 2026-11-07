# Project: UltraBench - A Standardized Benchmark for Evaluating Ultrasound Foundation Models
UltraBench a standardized evaluation framework for assessing how well ultrasound foundation models (pretrained vision encoders, like ViT or CNN backbones) generalize to downstream segmentation and classification tasks. It provides:

1. **A dataset layer.** Standardized versions (e.g., pre-processing, splitting, interface, etc.) of common classification and segmentation datasets.
2. **A model layer.** PyTorch backbones for the evaluated foundation models and the task heads (linear classifier, segmentation decoders) trained on top of them, in `ultrabench.models`. Installed with the `models` extra. The package ships no training framework; `examples/` holds standalone training scripts instead.
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

* **Tech Stack**: Python 3.10+, uv, PyTorch for the model layer (`models` extra), pytest for testing, and Hydra for the evaluation layer CLI. Dependencies of the example scripts live in the `examples` dependency group, which uv locks together with the package's dependencies.
* **Project Structure**:
    * `data/`: Contains the metadata files for each split of each dataset. These are tracked to ensure that the examples included in training and test subsets do not change unintentionally between versions.
    * `design_specs/`: Design specifications for the different layers of the framework.
    * `docs/`: The end user documentation. Written using Markdown.
    * `examples/`: Standalone training scripts showing how to train on the benchmark tasks and score the results. Not part of the package.
    * `notebooks/`: Jupyter notebooks
    * `src/`: The source code for the ultrabench package.
    * `tests/`: Contains all unit and integration tests.
    * `pyproject.toml`: Package configuration, including project dependencies.
