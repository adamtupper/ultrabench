#!/bin/bash

# Install non-Python dependencies
sudo apt-get update

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the package
uv sync

# Setup pre-commit hooks
source .venv/bin/activate
nbstripout --install
pre-commit install