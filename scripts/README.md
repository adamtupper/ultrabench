# Maintainer Scripts

These scripts download, process and publish the UltraBench datasets. They are for maintainers only: users load every dataset with `ultrabench.load_dataset`, which downloads the published datasets and preprocesses the raw and local ones on first use.

The scripts are not part of the installed package, and some of them need the `dev` dependency group (nibabel, SimpleITK, openpyxl, requests, zenodo-get). Run them from the repository root:

```bash
uv sync --group dev
```

| Script | Purpose |
|--------|---------|
| `download_dataset.py` | Download a raw dataset that is publicly downloadable. |
| `process_dataset.py` | Process a raw dataset into the UltraBench format, and optionally publish it. |
| `publish_raw.py` | Publish the original files of a dataset that may only be redistributed unmodified. |
| `generate_split_manifests.py` | Regenerate the tracked split manifests from a processed dataset. |

The per-dataset code lives in `downloading/` and `processing/`. The four datasets the package builds itself (GBCU, MMOTU, Open Kidney and COVID-BLUES) are processed by the builders in `src/ultrabench/datasets/builders/`, so the script and `load_dataset` always produce the same dataset. `processing/local.py` wraps those builders to save and publish them.

## Downloading

```bash
uv run python scripts/download_dataset.py <dataset> [<download_dir>]
```

| Dataset | Source |
|---------|--------|
| `aul` | Zenodo |
| `busbra` | Zenodo |
| `camus` | CREATIS |
| `covid-blues` | GitHub |
| `fetal-planes` | Zenodo |
| `masld` | Zenodo |
| `psfhs` | Zenodo |
| `stmus` | Mendeley Data |

The others (GBCU, MMOTU, Open Kidney, TN3K) must be requested from their authors and downloaded by hand. See the dataset pages in `docs/datasets/`.

## Processing

```bash
uv run python scripts/process_dataset.py <dataset> <raw_dir> <output_dir> [--size 256] [--publish]
```

- `<raw_dir>`: the downloaded raw data. For MASLD, this is the `.mat` file.
- `<output_dir>`: an existing directory. The dataset is written to `<output_dir>/UltraBench/<Dataset>/`, and the script refuses to overwrite an existing one.
- `--size`: the height and width of the resized `test.{size}` images (default `256`).
- `--publish`: push the dataset and its ground-truth CSVs to the Hugging Face Hub. This needs write access to the UltraBench organisation (`hf auth login`).

The supported datasets are `aul`, `busbra`, `camus`, `covid-blues`, `fetal-planes`, `gbcu`, `masld`, `mmotu`, `open-kidney`, `psfhs`, `stmus` and `tn3k`.

## Publishing Raw Data

COVID-BLUES and MMOTU may be redistributed unmodified but not in a preprocessed form. Their original files are published instead, and `load_dataset` preprocesses them locally:

```bash
uv run python scripts/publish_raw.py <dataset> <raw_dir> [--no-private]
```

Only the files the builders read are uploaded, in the layout they expect.

## Releasing a Revision

Datasets are versioned with Git tags on the Hub. The revision being produced is `DATASET_REVISION` in `src/ultrabench/datasets/builders/utils.py`.

1. Bump `DATASET_REVISION` and record the change in the revision table in `docs/dataset_format.md`.
2. If the revision re-splits a dataset, regenerate its manifests:

   ```bash
   uv run python scripts/generate_split_manifests.py <processed_dir>
   ```

3. Process and publish each dataset with `--publish`, and publish the raw datasets with `publish_raw.py`. Both refuse to publish a revision that is already tagged.
4. Tag each repository:

   ```bash
   hf repos tag create UltraBench/<Dataset> <revision> --type dataset -m "<description>"
   ```
