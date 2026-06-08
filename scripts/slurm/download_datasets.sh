#!/bin/bash

# Download each dataset to PROJECT using the HF CLI.
#
# Usage: bash scripts/slurm/download_datasets.sh
#
# Expects the HF_TOKEN environment variable to be set to a valid Hugging
# Face access token with read access to the UltraBench org.

module load arrow/21.0.0 python/3.12
virtualenv --no-download ~/venvs/hf
source ~/venvs/hf/bin/activate
pip install --no-index huggingface_hub datasets

hf auth login --token $HF_TOKEN

cd $project/ultrabench2

python scripts/slurm/download_dataset.py --name UltraBench/AUL --cache_dir $project/data/hf/datasets
python scripts/slurm/download_dataset.py --name UltraBench/BUS-BRA --cache_dir $project/data/hf/datasets
python scripts/slurm/download_dataset.py --name UltraBench/Butterfly --cache_dir $project/data/hf/datasets
python scripts/slurm/download_dataset.py --name UltraBench/CAMUS --cache_dir $project/data/hf/datasets
python scripts/slurm/download_dataset.py --name UltraBench/COVID-BLUES --cache_dir $project/data/hf/datasets
python scripts/slurm/download_dataset.py --name UltraBench/FetalPlanes --config fetal --cache_dir $project/data/hf/datasets
python scripts/slurm/download_dataset.py --name UltraBench/FetalPlanes --config fetal_brain --cache_dir $project/data/hf/datasets
python scripts/slurm/download_dataset.py --name UltraBench/GBCU --cache_dir $project/data/hf/datasets
python scripts/slurm/download_dataset.py --name UltraBench/MASLD --cache_dir $project/data/hf/datasets
python scripts/slurm/download_dataset.py --name UltraBench/MMOTU-2D --cache_dir $project/data/hf/datasets
python scripts/slurm/download_dataset.py --name UltraBench/OpenKidney --cache_dir $project/data/hf/datasets
python scripts/slurm/download_dataset.py --name UltraBench/POCUS --cache_dir $project/data/hf/datasets
python scripts/slurm/download_dataset.py --name UltraBench/PSFHS --cache_dir $project/data/hf/datasets
python scripts/slurm/download_dataset.py --name UltraBench/StanfordThyroid --cache_dir $project/data/hf/datasets
python scripts/slurm/download_dataset.py --name UltraBench/STMUS --cache_dir $project/data/hf/datasets
python scripts/slurm/download_dataset.py --name UltraBench/TN3K --cache_dir $project/data/hf/datasets

deactivate
rm -r ~/venvs/hf
module purge python/3.12
