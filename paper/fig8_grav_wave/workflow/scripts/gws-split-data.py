# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path

import hdf5plugin
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from gws_utils import get_git_describe

# meta data for identifying the simulation
GIT_DESCRIPTION = get_git_describe()

# %%
# Open the.pt file
thetas = torch.load("thetas_0.pt")
xs = torch.load("xs_0.pt")
# Print the shape of what is contained
print("theta", thetas.shape, thetas.dtype)
print("x", xs.shape, xs.dtype)

# %%
fig, ax = plt.subplots(1, 2)

for d in range(thetas.shape[-1]):
    ax[d].hist(thetas[..., d], bins=20)
    ax[d].set_xlabel(f"theta {d}")
    ax[d].set_ylabel(f"frequency")


# %%
fig, ax = plt.subplots(3, 2, figsize=(12, 5), sharex=True)

for s in range(3):
    for d in range(thetas.shape[-1]):
        ax[s, d].plot(xs[s, d, ...])
        ax[s, d].set_xlabel(f"t")
        if s == 0:
            ax[s, d].set_title(f"channel {d}")
        if d == 0:
            ax[s, d].set_ylabel(f"sample {s}")

# %%
# resolve root of input files
parent0 = Path("thetas_0.pt").resolve().parent
assert parent0.exists()

# %%
# search for all files in the dataset
thetas_locs = [Path(parent0 / f"thetas_{it:01.0f}.pt") for it in range(5)]
thetas_exist = [it.exists() for it in thetas_locs]
assert all(
    thetas_exist
), f"some theta files are missing from the dataset\n{thetas_locs}\n{thetas_exist}"

# %%
# loading and merging thetas
thetas_all = [torch.load(thepath) for thepath in thetas_locs]
thetas = torch.concat(thetas_all).float()
print(f"loaded thetas from {len(thetas_locs)} files, gives tensor of {thetas.shape}")

# %%
xs_locs = [Path(parent0 / f"xs_{it:01.0f}.pt") for it in range(5)]
xs_exist = [it.exists() for it in xs_locs]
assert all(
    xs_exist
), f"some x files are missing from the dataset\n{xs_locs}\n{xs_exist}"
assert len(xs_locs) == len(
    thetas_locs
), f"number of theta files and x files do not match: {len(thetas_locs)} != {len(xs_locs)}"

# %%
# loading and merging thetas
xs_all = [torch.load(thepath) for thepath in xs_locs]
xs_ = torch.concat(xs_all).float()
print(f"loaded xs from {len(xs_locs)} files, gives tensor of {xs_.shape}")

# %%
norm_style = "ligo" #or uniform
print(f"normalising x {norm_style} style")
xmin_, xmax_ = torch.min(xs_).item(), torch.max(xs_).item()

normloc = 0.
normscale = 1.
if "uniform" in norm_style:
    normloc = xmin_
    normscale = xmax_ - xmin_
else:
    #taken from https://journals.aps.org/prd/abstract/10.1103/PhysRevD.100.063015
    wsize = 256 #sliding window size
    means = xs_.mean(dim=-1)
    stds  = xs_.std(dim=-1)
    
    normloc   = torch.median(means, dim=0).values.unsqueeze(0).unsqueeze(-1)
    normscale = torch.median(stds , dim=0).values.unsqueeze(0).unsqueeze(-1)

xs = (xs_ - normloc)/normscale

xmin, xmax = torch.min(xs).item(), torch.max(xs).item()
print(f"normalised x from min/max {xmin_, xmax_} to {xmin, xmax}")

# %%
# plain train_test_split
theta_train, theta_test, x_train, x_test = train_test_split(
    thetas, xs, test_size=500, random_state=42
)
# NOTE: we could stratify by entries in theta too, this would be more precise;
#       given 50k samples and a testset of 500 being more precise might not be
#       worth the risk
print(
    f"split dataset into {theta_train.shape[0]} samples for training and {theta_test.shape[0]} samples for testing"
)

# %%
train_out = Path("gws-train.h5")
with h5py.File(train_out, "w") as trainf:
    tds = trainf.create_dataset(
        "thetas", data=theta_train, chunks=True, **hdf5plugin.Zfp(reversible=True)
    )
    xds = trainf.create_dataset(
        "xs", data=x_train, chunks=True, **hdf5plugin.Zfp(accuracy=0.0001)
    )
    xds.attrs["original_loc"] = normloc
    xds.attrs["original_scale"] =  normscale
    xds.attrs["was_normalized"] = True
    xds.attrs["norm_style"] = norm_style
    xds.attrs["git-describe"] = GIT_DESCRIPTION
    tds.attrs["git-describe"] = GIT_DESCRIPTION
    tds.attrs["was_normalized"] = False

print(
    f"wrote gws-train.h5",
    (theta_train.nbytes + x_train.nbytes) / (1024**2),
    "->",
    train_out.stat().st_size / (1024**2),
)

# %%
test_out = Path("gws-test.h5")
with h5py.File(test_out, "w") as testf:
    tds = testf.create_dataset(
        "thetas", data=theta_test, chunks=True, **hdf5plugin.Zfp(reversible=True)
    )
    xds = testf.create_dataset(
        "xs", data=x_test, chunks=True, **hdf5plugin.Zfp(accuracy=0.0001)
    )
    
    xds.attrs["original_loc"] = normloc
    xds.attrs["original_scale"] =  normscale
    xds.attrs["was_normalized"] = True
    xds.attrs["norm_style"] = norm_style
    xds.attrs["git-describe"] = GIT_DESCRIPTION
    tds.attrs["was_normalized"] = False
    tds.attrs["git-describe"] = GIT_DESCRIPTION



print(
    f"wrote gws-test.h5",
    (theta_test.nbytes + x_test.nbytes) / (1024**2),
    "->",
    test_out.stat().st_size / (1024**2),
)
