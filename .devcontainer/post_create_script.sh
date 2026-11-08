#!/bin/bash

# Configure git
git config --global core.editor "nano"
git config --global sequence.editor "nano"

# Install non-Python dependencies
sudo apt-get update && sudo apt-get install -y unrar nvtop

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the package
uv sync --all-groups

# Setup pre-commit hooks
source .venv/bin/activate
nbstripout --install
pre-commit install

# Install dependencies for docs
cd docs && bundle install
