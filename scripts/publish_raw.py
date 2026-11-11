"""Publish a dataset's original files to the Hugging Face Hub.

Some datasets may be redistributed unmodified but not in a preprocessed
form. Publishing their original files lets `ultrabench.load_dataset`
preprocess them on the user's machine without the user having to track
down the raw data themselves.

Usage:
    uv run python scripts/publish_raw.py DATASET RAW_DIR [--no-private]
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi
from processing.utils import verify_revision_unpublished

from ultrabench.datasets.builders.utils import DATASET_REVISION


@dataclass(frozen=True)
class RawRepo:
    """Where a dataset's original files come from and go to.

    Args:
        repo_id: The Hugging Face Hub repository to publish to.
        marker: A file that must be present, relative to the raw data
            directory, identifying it as the right dataset.
        allow_patterns: The files to upload. Everything else in the
            directory, such as the source code the datasets ship
            alongside their images, is left out.
    """

    repo_id: str
    marker: str
    allow_patterns: list[str]


RAW_DATASETS: dict[str, RawRepo] = {
    "covid-blues": RawRepo(
        repo_id="UltraBench/COVID-BLUES-raw",
        marker="covid_blues/clinical_variables.csv",
        allow_patterns=[
            "covid_blues/clinical_variables.csv",
            "covid_blues/lus_videos/*.mp4",
        ],
    ),
    "mmotu": RawRepo(
        repo_id="UltraBench/MMOTU-raw",
        marker="OTU_2d/train.txt",
        allow_patterns=["OTU_2d/**"],
    ),
}
"""The datasets whose original files may be republished."""


def publish_raw(dataset: str, raw_dir: str, private: bool = True) -> None:
    """Publish a dataset's original files to the Hugging Face Hub.

    Nothing is modified: the layout on the Hub is the layout
    the dataset's builder expects, so the loaders can pass a downloaded
    snapshot straight to the processing code.

    Args:
        dataset: The dataset to publish.
        raw_dir: The directory holding the raw data.
        private: Whether to create the repository as private.

    Raises:
        ValueError: If the dataset does not republish its raw data, or
            if the current revision is already tagged.
        FileNotFoundError: If the raw data directory does not hold the
            expected layout.
    """
    if dataset not in RAW_DATASETS:
        raise ValueError(
            f"{dataset!r} does not republish its raw data. "
            f"Valid datasets: {', '.join(sorted(RAW_DATASETS))}."
        )

    repo = RAW_DATASETS[dataset]
    raw_path = Path(raw_dir).expanduser()
    if not (raw_path / repo.marker).is_file():
        raise FileNotFoundError(
            f"{raw_path} does not look like the raw {dataset} data: "
            f"expected to find {repo.marker}."
        )

    verify_revision_unpublished(repo.repo_id)

    api = HfApi()
    api.create_repo(repo.repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=str(raw_path),
        repo_id=repo.repo_id,
        repo_type="dataset",
        allow_patterns=repo.allow_patterns,
        commit_message=f"Upload the raw data for revision {DATASET_REVISION}.",
    )

    print(
        f"Published the raw {dataset} data to {repo.repo_id}. Tag the "
        f"revision so that published results stay reproducible:\n"
        f"  hf repos tag create {repo.repo_id} {DATASET_REVISION} "
        f"--type dataset -m 'Upload the raw data.'"
    )


def main() -> None:
    """Parse the command-line arguments and publish the raw data."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("dataset", choices=sorted(RAW_DATASETS))
    parser.add_argument("raw_dir", help="The path to the raw data.")
    parser.add_argument(
        "--private",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to create the repository as private.",
    )
    args = parser.parse_args()
    publish_raw(args.dataset, args.raw_dir, private=args.private)


if __name__ == "__main__":
    main()
