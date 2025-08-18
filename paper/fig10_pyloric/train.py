from copy import deepcopy
import numpy as np
import os

import matplotlib.pyplot as plt
import pickle
import pandas as pd
import torch

from sbi.inference import NPE
from sbi.neural_nets import posterior_nn

from pyloric import simulate, create_prior, summary_stats

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def train(seed):
    _ = torch.manual_seed(seed)
    logging.info(f"Running with seed {seed}")

    print("Load data.")
    with open("results/all_circuit_parameters.pkl", "rb") as handle:
        theta = pickle.load(handle)
    with open("results/all_simulation_outputs.pkl", "rb") as handle:
        x = pickle.load(handle)
    x_o = np.load("results/xo_11deg.npy", allow_pickle=True)

    logging.info("Clean data.")
    clean_x = []
    for i in range(18):
        x_ = x.to_numpy()[:, i]
        nan_vals = np.isnan(x_)
        if i == 0:
            x_[x_ > 3000] = 3000
            x_[nan_vals] = 3000

        if i in [1, 2, 3]:
            x_[x_ > 500] = 500
            x_[nan_vals] = 500

        if i in [4, 5, 6]:
            x_[x_ > 1] = 1
            x_[nan_vals] = 1

        if i in [7, 8]:
            x_[nan_vals] = 1.0

        if i in [9, 10, 11, 12]:
            x_[x_ < -1000] = -1000
        if i in [9, 10, 11, 12]:
            x_[x_ > 1000] = 1000
            x_[nan_vals] = 1000

        if i in [13, 14]:
            x_[x_ > 1] = 1
            x_[nan_vals] = 1
        if i in [13, 14]:
            x_[x_ < -1] = -1

        if i in [15, 16, 17]:
            x_[x_ > 10] = 10
            x_[nan_vals] = 100            
        clean_x.append(x_)

    clean_x = torch.as_tensor(np.asarray(clean_x).T, dtype=torch.float32)
    clean_theta = torch.as_tensor(theta.to_numpy(), dtype=torch.float32)

    logging.info("Build neural net and trainer.")
    density_estimator = posterior_nn(
        "nsf",
        num_transforms=10,
        hidden_features=100,
    )
    inference = NPE(density_estimator=density_estimator, device="gpu")
    inference = inference.append_simulations(clean_theta, clean_x, data_device="cpu")

    logging.info("Training.")
    density_estimator = inference.train(
        max_num_epochs=400, training_batch_size=4096, stop_after_epochs=30
    )

    # logging.info("Saving.")
    # with open(f"results/inference_seed{seed}.pkl", "wb") as handle:
    #     pickle.dump(inference, handle)

    with open(f"results/density_estimator_seed{seed}.pkl", "wb") as handle:
        pickle.dump(density_estimator.to("cpu"), handle)

    posterior = inference.build_posterior()
    # with open(f"results/posterior_seed{seed}.pkl", "wb") as handle:
    #     pickle.dump(posterior, handle)

    logging.info("Sampling.")
    samples = posterior.sample((1_000,), x=torch.as_tensor(x_o, dtype=torch.float32).to("cuda")).to("cpu")

    logging.info("Generating posterior predictives.")
    p1 = create_prior()
    pars = p1.sample((1,))
    column_names = pars.columns
    parameter_set_pd = pd.DataFrame(np.asarray(samples), columns=column_names)

    logging.info("Simulating.")
    all_stats = []
    for i in range(100):
        simulation_output = simulate(parameter_set_pd.loc[i])
        summary_statistics = summary_stats(simulation_output)
        all_stats.append(summary_statistics)
    summary_statistics = pd.concat(all_stats)

    fraction = summary_statistics.dropna().shape[0] / summary_statistics.shape[0]
    logging.info(f"Fraction of no NaN: {fraction}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(args.seed)