import argparse
import pickle

import torch
from ddm_utils import (
    parallel_simulator,
    simulate_ddm_collapsing,
)
from sbi.inference import MNLE
from sbi.utils import BoxUniform
from sbi.utils.diagnostics_utils import get_posterior_samples_on_batch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun_simulations", action="store_true")
    parser.add_argument("--rerun_training", action="store_true")
    parser.add_argument("--rerun_diagnostics", action="store_true", default=False)
    args = parser.parse_args()

    # settings
    num_simulations = 200000
    num_posterior_samples = 1000
    num_sbc_samples = 200
    num_workers = 10
    num_trials = 100
    seed = 42

    mcmc_parameters = {
        "num_chains": 100,
        "thin": 2,
        "warmup_steps": 1000,
        "init_strategy": "sir",
    }

    prior = BoxUniform(
        low=torch.tensor([-2.5, 0.25, -0.25, 0.05, -1.0]),
        high=torch.tensor([2.5, 1.0, 0.25, 0.95, -0.1]),
    )

    simulator = parallel_simulator(
        num_workers=num_workers, show_progress=True, seed=seed
    )(simulate_ddm_collapsing)

    # Run simulations
    if args.rerun_simulations:
        theta = prior.sample((num_simulations,))
        x = simulator(theta)

        with open("data/sbi_ddm_collapsing_prior_theta_x.pkl", "wb") as f:
            pickle.dump((prior, theta, x), f)
    else:
        with open("data/sbi_ddm_collapsing_prior_theta_x.pkl", "rb") as f:
            prior, theta, x = pickle.load(f)

    # Run training
    if args.rerun_training:
        inferer = MNLE()
        density_estimator = inferer.append_simulations(theta, x).train()
        torch.save(density_estimator, "data/ddm_collapsing_estimator.pt")
    else:
        inferer = MNLE()
        density_estimator = torch.load("data/ddm_collapsing_estimator.pt")

    # Run diagnostics
    if args.rerun_diagnostics:
        posterior = inferer.build_posterior(
            prior=prior,
            density_estimator=density_estimator,
            mcmc_parameters=mcmc_parameters,
        )

        # Run SBC on single trial xs
        thetas = prior.sample((num_sbc_samples,))
        xs = []
        for i in range(num_sbc_samples):
            x = simulator(thetas[i].repeat(num_trials, 1))
            xs.append(x)
        xs = torch.cat(xs, dim=0)

        posterior_samples = get_posterior_samples_on_batch(
            xs,
            posterior,
            (num_posterior_samples,),
            num_workers,
            show_progress_bar=False,
            use_batched_sampling=False,
        )

        diagnostic_samples = {
            "theta": thetas,
            "x": xs,
            "posterior_samples": posterior_samples,
        }
        with open(
            f"data/ddm_collapsing_calibration_samples_{num_trials}.pt",
            "wb",
        ) as f:
            pickle.dump(diagnostic_samples, f)
