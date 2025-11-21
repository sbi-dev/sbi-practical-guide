## Installation

Some dependencies are rather hard to install for this workflow. The `requirements.txt` was tested with python `3.11`. Python `3.12` works sometimes but not in a reproducible manner. For best results, install a separate environment to run this notebook.

1. create a `venv` by `python -m venv py311 --upgrade-deps` (alternatively `uv venv py311-uv --python 3.11 --extra-index-url https://download.pytorch.org/whl/cpu`)
2. setup that `venv` by `source py311/bin/activate`
3. (optional) install `uv` for faster installations
4. either do `uv pip install -r ./requirements.txt` or plain `python -m pip install -r ./requirements.txt`

If you like to train cpu-only, you have to install torch without CUDA support. This is best beformed between step 2 and 4 in the recipe above by running:
```shell
uv pip install --extra-index-url https://download.pytorch.org/whl/cpu torch
```
or without `uv`:
```shell
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu torch
```


The generated simulations and saved diagnostic results used in the notebook
`4_1_grav_waves.ipynb` (~2.5GB) are available upon request via a new issue.