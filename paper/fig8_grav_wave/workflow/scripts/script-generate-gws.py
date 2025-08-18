# code created by Manuel Glöckler
import os
import sys
import time
from functools import partial

import h5py
import hdf5plugin
import numpy as np
import pycbc
import torch
from sbi import utils
from torch import multiprocessing
from pathlib import Path
import click
from external.simulator import GravitationalWaveBenchmarkSimulator as gws
from gws_utils import get_git_describe

if "NUM_SIMS" not in os.environ.keys():
    NUM_SIMS = 10_000
else:
    NUM_SIMS = int(os.environ["NUM_SIMS"])



ROOTFOLDER = Path(__file__).resolve().parent

config_path = ROOTFOLDER / "external/config_file_pycbcmaster.ini"
pycbc_semver = pycbc.__version__.split(".")
if len(pycbc_semver) == 3 and pycbc_semver[0] == "2" and pycbc_semver[1] == "4":
    config_path = ROOTFOLDER / "external/config_file.ini"
print(f"using config file {config_path} for GWS simulator")

with open(config_path, "r") as cfile:
    config_content_str = cfile.read()
with open("requirements.txt", "r") as rfile:
    requirements_txt_str = rfile.read()
with open(ROOTFOLDER / "external/simulator.py", "r") as sfile:
    simulator_py_str = sfile.read()

default_simulator = gws(str(config_path))
# meta data for identifying the simulation
GIT_DESCRIPTION = get_git_describe()

def run_sim(theta, simulator=default_simulator):
    # np.random.seed(seed_)
    # torch.manual_seed(seed_)
    masses = torch.Tensor([theta[0], theta[1] * theta[0]])
    xs = simulator(masses.to("cpu"))
    return masses, xs

@click.command()
@click.option('-I', '--batch-id', default=1, type=int, help='batch index.')
@click.option('-n', '--num_sims', default=NUM_SIMS, type=int, help='number of simulations to perform')
@click.option('-j', '--num_workers', default=1, type=int, help='number of worker processes to use in parallel')
@click.option('-s', '--simulator_config_file', default=config_path, type=click.Path(), help='path to simulator config file')
def main(batch_id, num_sims, num_workers, simulator_config_file):
    """ will generate {NUM_SIMS} samples from simulator and store the output in gws-<BATCH-INDEX>.h5 """

    idx = int(batch_id)  # Read in the first argument

    #override with openmp settings
    if "OMP_NUM_THREADS" in os.environ.keys():
        openmp_ncores = int(os.environ["OMP_NUM_THREADS"])
        click.echo(f"overriding nworkers {num_workers} -> {openmp_ncores}")
        num_workers = openmp_ncores

    #reproducibility
    np.random.seed(idx)
    torch.manual_seed(idx)

    # generating mass1 and ratio r=m2/m1
    # this way we ensure that mass2 is always less than mass1
    # as mass2 = r*mass1
    prior = utils.BoxUniform(
        low=torch.tensor([40.0, 0.25]),
        high=torch.tensor([80.0, 0.99]),
    )

    #sample from the prior
    thetas = prior.sample((num_sims,)).view(-1, 2)

    print(
        f"simulating batch {idx} of {num_sims} simulated samples on {num_workers} detected cores"
    )
    #preparing the simulator
    simulator_ = gws(str(simulator_config_file))
    run_sim_ = partial(run_sim, simulator=simulator_)

    #simulate gravitational waves
    start = time.time()
    with multiprocessing.Pool(num_workers) as p:
        results = p.map(run_sim_, thetas)
    #split results
    masses, xs = zip(*results)
    #concat signals into one tensor
    xs = torch.concat(xs)
    end1 = time.time()

    dur_sec = end1 - start
    print(
        f"{num_sims} simulated samples produced in {dur_sec:.2f} s ({dur_sec/float(num_sims):02.4f} sample/s)"
    )
    print("thetas:", thetas.shape, thetas.dtype)
    print("xs:", xs.shape, xs.dtype)

    outfile = f"denovo-gws-{idx:02.0f}.h5"
    with h5py.File(outfile, "w") as out5:
        out5_xs = out5.create_dataset(
            "xs",
            data=xs,
            chunks=True,
            **hdf5plugin.Bitshuffle(nelems=0, cname="zstd", clevel=10),
        )
        out5_thetas = out5.create_dataset(
            "thetas",
            data=thetas,
            chunks=True,
            **hdf5plugin.Bitshuffle(nelems=0, cname="zstd", clevel=10),
        )
        out5_masses = out5.create_dataset(
            "masses",
            data=np.asarray(masses),
            chunks=True,
            **hdf5plugin.Bitshuffle(nelems=0, cname="zstd", clevel=10),
        )

        out5_xs.attrs["config_file"] = config_content_str
        out5_xs.attrs["requirements_txt"] = requirements_txt_str
        out5_xs.attrs["simulator_py"] = simulator_py_str
        out5_xs.attrs["pycbc_version"] = pycbc.__version__
        out5_xs.attrs["seed"] = idx
        out5_xs.attrs["git-describe"] = GIT_DESCRIPTION
        out5_thetas.attrs["seed"] = idx
        out5_masses.attrs["seed"] = idx

    end2 = time.time()
    dur_io_sec = end2 - end1
    print(
        f"{num_sims} simulated samples written to {outfile} in {dur_io_sec:.2f} s ({dur_io_sec/float(num_sims):02.4f} sample/s)"
    )
    return 0

# Run the simulator on multiple cores
# Note: This is neither a well written, nor a flexible script.
#       This is just there to run some simulations to demo sbi.
if __name__ == "__main__":
    rv = main()
    sys.exit(rv)
