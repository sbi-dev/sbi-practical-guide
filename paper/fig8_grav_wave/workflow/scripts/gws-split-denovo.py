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

import h5py as h5
import hdf5plugin
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from loguru import logger
import sys

# %%
np.random.seed(42)
torch.random.manual_seed(41)

logger.remove() #remove default logger to sys.stdout
logger.add(sys.stdout,
           format="[{time:HH:mm:ss.S}] {file}:{line:03.0f} <lvl>{message}</lvl>",
           level="INFO",
           colorize=True)

# %%
# Open the.pt file
with h5.File("denovo-gws-01.h5", "r") as in5:
    thetas = torch.Tensor(np.asarray(in5["thetas"])).to(torch.float32)
    xs = torch.Tensor(np.asarray(in5["xs"])).to(torch.float32)
# Print the shape of what is contained
logger.info("theta {} {}", thetas.shape, thetas.dtype)
logger.info("x {} {}", xs.shape, xs.dtype)

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
parent0 = Path("denovo-gws-01.h5").resolve().parent
assert parent0.exists() and parent0.is_dir()

# %%
# search for all files in the dataset
gws_locs = list(parent0.glob("./denovo-gws-??.h5"))
logger.info("found {} files with relevant simulation data", len(gws_locs))

# %%
# loading and merging thetas
gws_all = [h5.File(thepath, "r") for thepath in gws_locs]
thetas_all = [torch.Tensor(np.asarray(h5file["thetas"])).to(torch.float32) for h5file in gws_all]
thetas = torch.concat(thetas_all).float()
logger.info(f"loaded thetas from {len(gws_locs)} files, gives tensor of {thetas.shape, thetas.dtype}")

# %%
# loading and merging xs
xs_all = [torch.Tensor(np.asarray(h5file["xs"])).to(torch.float32) for h5file in gws_all]
xs_ = torch.concat(xs_all).float()
logger.info(f"loaded xs from {len(gws_locs)} files, gives tensor of {xs_.shape, xs_.dtype}")
assert xs_.shape[0] == thetas.shape[0], f"inconsistent number of samples in theta {thetas.shape[0]} and xs {xs_.shape[0]}"


# %%
norm_style = "uniform" #or uniform
logger.info(f"normalising x {norm_style} style")
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
logger.info(f"normalised x from min/max {xmin_, xmax_} to {xmin, xmax}")

# %%
# plain train_test_split
theta_train, theta_test, x_train, x_test = train_test_split(
    thetas, xs, test_size=500, random_state=42
)
# NOTE: we could stratify by entries in theta too, this would be more precise;
#       given 50k samples and a testset of 500 being more precise might not be
#       worth the risk
logger.info(
    f"split dataset into {theta_train.shape[0]} samples for training and {theta_test.shape[0]} samples for testing"
)

# %%
train_out = Path("gws-train.h5")
with h5.File(train_out, "w") as trainf:
    tds = trainf.create_dataset(
        "thetas", data=theta_train, chunks=True, **hdf5plugin.Bitshuffle(nelems=0, cname="zstd")
    )
    xds = trainf.create_dataset(
        "xs", data=x_train, chunks=(320, 2, 8192), **hdf5plugin.Bitshuffle(nelems=0, cname="zstd", clevel=10)
    )
    
    xds.attrs["original_loc"] = normloc
    xds.attrs["original_scale"] = normscale
    xds.attrs["was_normalized"] = True
    xds.attrs["norm_style"] = norm_style
    tds.attrs["was_normalized"] = False

logger.info(
    f"wrote gws-train.h5",
    (theta_train.nbytes + x_train.nbytes) / (1024**2),
    "->",
    train_out.stat().st_size / (1024**2),
)

# %%
test_out = Path("gws-test.h5")
with h5.File(test_out, "w") as testf:
    tds = testf.create_dataset(
        "thetas", data=theta_test, chunks=True,  **hdf5plugin.Bitshuffle(nelems=0, cname="zstd")
    )
    xds = testf.create_dataset(
        "xs", data=x_test, chunks=(160,2,8192), **hdf5plugin.Bitshuffle(nelems=0, cname="zstd", clevel=10)
    )
    
    xds.attrs["original_loc"] = normloc
    xds.attrs["original_scale"] =  normscale
    xds.attrs["was_normalized"] = True
    xds.attrs["norm_style"] = norm_style
    tds.attrs["was_normalized"] = False
    

logger.info(
    f"wrote gws-test.h5",
    (theta_test.nbytes + x_test.nbytes) / (1024**2),
    "->",
    test_out.stat().st_size / (1024**2),
)
