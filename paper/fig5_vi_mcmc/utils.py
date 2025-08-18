import torch
import numpy as np
import random
import os

from typing import Optional, Union, Type, List, Dict, Tuple

from sbi.simulators.linear_gaussian import (
    diagonal_linear_gaussian,
    true_posterior_linear_gaussian_mvn_prior,
)
from sbi.inference import SNPE, SNLE, SNRE
from sbi.utils.metrics import c2st

import time

from functools import partial
import pandas as pd

RESULTS_PATH = "./results.csv"

import os 

def get_results():
    if not os.path.exists(RESULTS_PATH):
        df = pd.DataFrame(columns=['method', "sampling_method", "sampling_params", "times", "c2sts" ])
        df.to_csv(RESULTS_PATH, index=False)
    else:
        df = pd.read_csv(RESULTS_PATH)
        
    return df 

def save_results(method:str, d:int, sampling_method:str, sampling_params:dict, times, c2sts):
    
    df = get_results()
    df_append = pd.DataFrame({'method': method, "dimension": d, "sampling_method": sampling_method, "sampling_params": sampling_params, "times": times, "c2sts": c2sts})
    
    df = pd.concat([df, df_append], ignore_index=True)
    df.to_csv(RESULTS_PATH, index=False)
    
    
def query(method = None, d = None, sampling_method = None):
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

    return method_inf,true_posterior


def benchmark_sample_from_inference(
    method_inf: Union[SNPE, SNLE, SNRE],
    num_samples: int,
    x_o: torch.Tensor,
    sample_with: str,
    mcmc_parameters: Optional[Dict] = None,
    mcmc_method = "slice_np",
    vi_parameters: Optional[Dict] = None,
    seeds: List[int] = [1, 2, 3],
) -> Tuple[List[torch.Tensor], List[float]]:

    samples = []
    times = []
    for seed in seeds:
        set_seed(seed)

        if sample_with == "mcmc":
            posterior = method_inf.build_posterior(
                sample_with=sample_with, mcmc_parameters=mcmc_parameters, mcmc_method=mcmc_method
            )
        elif sample_with == "direct":
            posterior = method_inf.build_posterior()
        elif sample_with == "vi":
            posterior = method_inf.build_posterior(
                sample_with=sample_with, vi_parameters=vi_parameters
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


def eval_samples(samples: List[torch.Tensor], true_posterior: torch.distributions.Distribution, seed:int = 42):
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
