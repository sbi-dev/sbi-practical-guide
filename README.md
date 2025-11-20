
# Simulation-Based Inference: A Practical Guide

This repository contains the code to reproduce the figures and numerical results of the paper:

> **Deistler, Boelts et al. (2025)**, *Simulation-Based Inference: A Practical Guide*, arXiv: [2508.12939](https://arxiv.org/abs/2508.12939).

The implementation is based on the `sbi` toolbox and is organised so that each figure
(and its associated experiments) can be reproduced from a dedicated directory under
`paper/`.

---

## Overview

This repository is intended for researchers and practitioners who want to

- reproduce the figures and numerical experiments from the paper,
- inspect concrete SBI implementations for different applications,
- use the examples as templates for their own simulators and inference problems.

At a high level, you will find:

- figure-specific workflows for all main-text and appendix figures,
- complete SBI pipelines for the introductory example and the three applications
  discussed in the paper (astrophysics, psychophysics, neuroscience),
- post-processing scripts and notebooks to generate the final plots in the manuscript.

---

## Repository structure

- `paper/fig*/`  
  Figure-specific assets:
  - simulator definitions and configuration,
  - inference scripts and Jupyter notebooks,
  - post-processing code,
  - generated `fig/` and `svg/` outputs.

- `examples/`
  - notebooks containing full SBI workflows for the examples presented in the paper.

- `tasks.py`  
  Helper tasks, for example converting exported SVG figures to PDF/PNG via Inkscape.

---

## Installation

We recommend using [`uv`](https://github.com/astral-sh/uv) to create a reproducible
environment based on `pyproject.toml` and `uv.lock`.

```bash
uv venv -p python3.11
source .venv/bin/activate
uv sync
```

If you prefer a different environment manager, you can use `pyproject.toml` as a
reference for the required dependencies, but we only test with `uv`.

---

## Large files and pre-simulated data (Git LFS)

Several figures rely on pre-simulated artifacts (for instance for diffusion decision
modelling or gravitational-wave inference). These are tracked with [Git
LFS](https://git-lfs.com/).

After cloning the repository, run:

- `git lfs install`
- `git lfs pull`

Without this step, large binary files will be replaced by small pointer text files and
the corresponding notebooks or scripts will not run as intended.

---

## Questions, problems, and contributions

If something does not work as expected (installation, missing data, notebook errors,
figure mismatches), please open an issue on the GitHub issue tracker with:

- a short description of the problem,
- the commands you ran and the relevant environment details (OS, Python version),
- the figure directory and notebook you were using,
- the full error message or traceback.

Small improvements to the code, documentation, and reproducibility (for example
clarifying comments, robustness fixes, additional tests) are welcome via pull requests.

---

## Citing

If you use benefit from the paper or this repository, or adapt its code in your own work,
please cite:

```
@misc{DeistlerBoelts_simulationbased_2025,
  title = {Simulation-{{Based Inference}}: {{A Practical Guide}}},
  author = {Deistler, Michael and Boelts, Jan and Steinbach, Peter and Moss, Guy 
    and Moreau, Thomas and Gloeckler, Manuel and Rodrigues, Pedro L. C. and Linhart, Julia 
    and Lappalainen, Janne K. and Miller, Benjamin Kurt and Gon{\c c}alves, Pedro J. 
    and Lueckmann, Jan-Matthis and Schr{\"o}der, Cornelius and Macke, Jakob H.},
  year = 2025,
  doi = {10.48550/arXiv.2508.12939},
  archiveprefix = {arXiv}
}
```

---

## License

This project is released under the [MIT license](./LICENSE).
