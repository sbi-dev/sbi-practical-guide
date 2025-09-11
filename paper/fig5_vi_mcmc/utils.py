import json
import os
import random
import time
from functools import partial
from typing import List, Tuple, Type, Union

import numpy as np
import pandas as pd
import torch
from sbi.inference import SNLE, SNPE, SNRE
from sbi.inference.posteriors.posterior_parameters import PosteriorParameters
from sbi.simulators.linear_gaussian import (
    diagonal_linear_gaussian,
    true_posterior_linear_gaussian_mvn_prior,
)
from sbi.utils.metrics import c2st

RESULTS_PATH = "./results.csv"


def get_results():
    """Load existing results or initialize empty results DataFrame.

    Ensures a consistent schema including the 'dimension' column.
    """
    expected_cols = [
        "method",
        "dimension",
        "sampling_method",
        "sampling_params",
        "times",
        "c2sts",
    ]
    if not os.path.exists(RESULTS_PATH):
        df = pd.DataFrame(columns=expected_cols)
        df.to_csv(RESULTS_PATH, index=False)
    else:
        df = pd.read_csv(RESULTS_PATH)
        # Backfill missing columns if older file exists
        for col in expected_cols:
            if col not in df.columns:
                df[col] = pd.NA
        # Reorder
        df = df[expected_cols]
    return df


def save_results(
    method: str,
    d: int,
    sampling_method: str,
    sampling_params: dict,
    times,
    c2sts,
):
    """Append benchmark results."""

    if len(times) != len(c2sts):
        raise ValueError(f"Length mismatch: times({len(times)}) != c2sts({len(c2sts)})")

    n = len(times)
    # Serialize params (dict) to stable JSON string for CSV storage
    params_json = json.dumps(sampling_params) if sampling_params is not None else None
    df = get_results()
    df_append = pd.DataFrame(
        {
            "method": [method] * n,
            "dimension": [d] * n,
            "sampling_method": [sampling_method] * n,
            "sampling_params": [params_json] * n,
            "times": list(times),
            "c2sts": list(c2sts),
        }
    )
    df = pd.concat([df, df_append], ignore_index=True)
    df.to_csv(RESULTS_PATH, index=False)


def query(method=None, d=None, sampling_method=None):
    df = get_results()
    if method is not None:
        df = df[df.method == method]
    if d is not None:
        df = df[df.dimension == d]
    if sampling_method is not None:
        df = df[df.sampling_method == sampling_method]

    return df


def d_dim_gaussian_linear_task(d: int):
    # Simple gaussian linear task for a given dimension d
    prior_mean = torch.tensor([0.0] * d)
    prior_std = torch.tensor([1.0] * d)

    prior = torch.distributions.Independent(
        torch.distributions.Normal(loc=prior_mean, scale=prior_std), 1
    )

    std = 0.1
    simulator = partial(diagonal_linear_gaussian, std=std)

    def true_posterior(x_o):
        return true_posterior_linear_gaussian_mvn_prior(
            x_o,
            likelihood_shift=0.0 * torch.zeros_like(prior_mean),
            likelihood_cov=torch.diag_embed(std**2 * torch.ones_like(prior_mean)),
            prior_mean=prior_mean,
            prior_cov=torch.diag_embed(prior_std**2),
        )

    return prior, simulator, true_posterior


def train_inference(
    method: Union[Type[SNPE], Type[SNLE], Type[SNRE]],
    d: int,
    num_simulations: int = 1000,
    seed: int = 0,
    device: str = "cpu",
):
    # Set seed for reproducibility
    set_seed(seed)

    prior, simulator, true_posterior = d_dim_gaussian_linear_task(d)
    thetas = prior.sample((num_simulations,))
    xs = simulator(thetas)

    method_inf = method(prior=prior, device=device, show_progress_bars=False)
    _ = method_inf.append_simulations(thetas, xs).train()

    return method_inf, true_posterior


def benchmark_sample_from_inference(
    method_inf: Union[SNPE, SNLE, SNRE],
    num_samples: int,
    x_o: torch.Tensor,
    sample_with: str,
    posterior_parameters: PosteriorParameters,
    seeds: List[int] = [1, 2, 3],
) -> Tuple[List[torch.Tensor], List[float]]:
    samples = []
    times = []
    for seed in seeds:
        set_seed(seed)

        if sample_with == "mcmc":
            posterior = method_inf.build_posterior(
                sample_with=sample_with,
                posterior_parameters=posterior_parameters,
            )
        elif sample_with == "direct":
            posterior = method_inf.build_posterior()
        elif sample_with == "vi":
            posterior = method_inf.build_posterior(
                sample_with=sample_with, posterior_parameters=posterior_parameters
            )
        else:
            raise ValueError(f"Unknown sampling method: {sample_with}")

        posterior.set_default_x(x_o)
        start_time = time.time()
        if sample_with == "vi":
            posterior.train(show_progress_bar=False, quality_control=False)
        sample = posterior.sample((num_samples,), show_progress_bars=False)
        end_time = time.time()

        samples.append(sample)
        times.append(end_time - start_time)

    return samples, times


def eval_samples(
    samples: List[torch.Tensor],
    true_posterior: torch.distributions.Distribution,
    seed: int = 42,
):
    # Evaluate samples using the true posterior
    set_seed(seed)
    c2st_values = []
    reference_samples = true_posterior.sample((len(samples[0]),))
    for sample in samples:
        c2st_values.append(float(c2st(sample, reference_samples)))
    return c2st_values


def set_seed(seed: int):
    # Set seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    return seed
