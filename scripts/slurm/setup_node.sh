#!/bin/bash

# Setup Slurm compute node
#
# Usage: source scripts/slurm/setup_node.sh

set -e

# Purge modules
module purge

# Load modules
module load gcc arrow/21.0.0 git-lfs python/3.12 cuda/12.6 cudnn/9.10.0.56 rust opencv/4.11.0

# Set environment variables
export TMPDIR=$SLURM_TMPDIR/tmp
export HF_HUB_OFFLINE=1
export HF_HOME=$project/data/hf
export TQDM_DISABLE=1
export NO_ALBUMENTATIONS_UPDATE=1

# Copy code to compute node
rsync -a $project/ultrabench2 $SLURM_TMPDIR --exclude-from=$project/ultrabench2/.gitignore

# Create virtual environment
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Install package and dependencies
cd $SLURM_TMPDIR/ultrabench2
pip install --no-index --upgrade pip
pip install --no-index --find-links=$project/wheels -e . --group training

# Change to working directory
cd $SLURM_TMPDIR
