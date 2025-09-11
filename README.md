# Simulation-Based Inference: A Practical Guide

This repository contains code to regenerate the figures and numerical results of the
practical guide on simulation-based inference . Each figure directory under `paper/`
contains the simulator configuration, inference scripts / notebooks, and post-processing
needed to produce the corresponding figure panel(s).

## Repository Structure

- `paper/fig*/` – Figure-specific assets (notebooks, helper modules, generated `fig/`
  and `svg/` outputs).
- `tasks.py` – Invoke tasks for converting exported SVG figures to PDF/PNG via Inkscape.

## Installation

We recommend using `uv` for installing the dependencies. After installing `uv`, run

```bash
uv venv -p python3.11
source .venv/bin/activate
uv sync
```

## Reproducing Figures

1. Open the target `paper/figX_*/` notebook (e.g. `01_create_figure.ipynb`).
2. Execute cells to (a) run / load simulations, (b) run SBI methods, (c) assemble plots
   into `fig/`.

## Large Files / Pre‑simulated Data (Git LFS)

Some figures rely on pre‑simulated artifacts tracked with Git LFS (e.g. DDM,
gravitational wave); after cloning run `git lfs install && git lfs pull` to fetch
binaries (otherwise you only have pointer text files). Only add new large files if
essential for reproducibility and track them first with `git lfs track` before
committing.

## Contributing

Small improvements (tests, reproducibility, docs) are welcome via pull request.
