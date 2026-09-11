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
pip install --no-index -e .

hf auth login --token $HF_TOKEN

cd $project/ultrabench2

# Datasets published ready to use.
for spec in "AUL" "BUS-BRA" "CAMUS:classification" "CAMUS:segmentation" \
            "FetalPlanes:fetal" "FetalPlanes:fetal_brain" "MASLD" "PSFHS" \
            "STMUS" "TN3K"; do
  name=${spec%%:*}
  config=${spec#*:}
  args="--name UltraBench/$name --revision 0.4 --cache_dir $project/data/hf/datasets"
  [ "$config" = "$name" ] || args="$args --config $config"
  python scripts/slurm/download_dataset.py $args
done

# Datasets preprocessed on this machine. The compute nodes run with
# HF_HUB_OFFLINE=1, so the cache has to be warmed here, and the two local
# datasets need their raw data staged under $project/data/raw.
export ULTRABENCH_CACHE=$project/data/ultrabench_cache

python -c "import ultrabench2; ultrabench2.load_dataset(name='MMOTU', revision='0.4')"
python -c "import ultrabench2; ultrabench2.load_dataset(name='COVID-BLUES', revision='0.4')"
python -c "import ultrabench2; ultrabench2.load_dataset(name='GBCU', path='$project/data/raw/GBCU-Shared')"
python -c "import ultrabench2; ultrabench2.load_dataset(name='OpenKidney', path='$project/data/raw/kidneyUS')"

deactivate
rm -r ~/venvs/hf
module purge python/3.12
