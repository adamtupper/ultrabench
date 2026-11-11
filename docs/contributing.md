---
title: Contributing
nav_order: 6
---

# Contributing

Please read the [code of conduct](https://github.com/adamtupper/ultrabench/blob/main/CODE_OF_CONDUCT.md) before you contribute. Open an issue to discuss a change before you open a pull request.

## Install from source

`uv sync` installs the `dev` and `examples` dependency groups:

```bash
uv sync
source .venv/bin/activate
nbstripout --install
pre-commit install
```

## Build the documentation

The documentation in `docs/` is a Jekyll site. Install Ruby and Bundler, then serve the site locally from `docs/`:

```bash
cd docs
bundle install
bundle exec jekyll serve --livereload
```

Open <http://localhost:4000> to view it.

## Maintainer scripts

Downloading raw data, processing it, and publishing it to the Hugging Face Hub are maintainer tasks. See [`scripts/README.md`](https://github.com/adamtupper/ultrabench/blob/main/scripts/README.md).
