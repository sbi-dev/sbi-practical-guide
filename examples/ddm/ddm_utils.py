import inspect
from typing import Callable, Union

import numpy as np
import sdeint
import torch
from joblib import Parallel, delayed
from tqdm import tqdm


# Decorator for parallel simulation
def parallel_simulator(num_workers=4, show_progress=True, seed: int = 42) -> Callable:
    def decorator(func):
        if "theta" not in inspect.signature(func).parameters:
            raise ValueError("The decorated function must have a 'theta' argument.")

        def wrapper(theta):
            # Convert theta to a list of tuples
            theta_list = theta.tolist()

            np.random.seed(seed)
            seeds = np.random.randint(0, 2**32 - 1, len(theta_list))

            def func_seeded(theta_i, seed):
                np.random.seed(seed)
                return func(theta_i)

            results = list(
                tqdm(
                    Parallel(n_jobs=num_workers, return_as="generator")(
                        delayed(func_seeded)(theta_i, seed)
                        for theta_i, seed in zip(theta_list, seeds)
                    ),
                    disable=not show_progress,
                    total=len(theta_list),
                )
            )

            # Convert results to PyTorch tensors
            results = torch.tensor(results, dtype=torch.float32)

            return results

        return wrapper

    return decorator


def integrate_ddm(v: float, x0: float, t_steps: np.ndarray) -> np.ndarray:
    # Define the drift and diffusion functions
    def f(x, t):
        return v

    def g(x, t):
        return 1

    x = sdeint.itoint(f, g, x0, t_steps)

    return x.flatten()


def find_rt_and_choice(y: np.ndarray, boundaries: np.ndarray, dt: float) -> tuple:
    # Find the first time step where the absolute value of X exceeds the boundary
    boundary_exceeded = np.abs(y) >= boundaries
    try:
        first_exceedance = np.where(boundary_exceeded)[0][0]
    except IndexError:
        first_exceedance = -1

    # Calculate the reaction time, including non-decision time
    rt = first_exceedance * dt

    # Determine the choice based on the sign of X at the first exceedance
    choice = 1 if y[first_exceedance] > 0 else 0

    return rt, choice


def simulate_ddm(
    theta: Union[list, np.ndarray, torch.Tensor], dt: float = 0.01, t_max: float = 5.0
) -> tuple:
    """
    Simulates the DDM.

    Args:
        theta: A tuple of parameters (v, a, w, T).
        dt: Time step for the simulation.
        T_max: Maximum simulation time.

    Returns:
        rt: Reaction time.
        choice: Choice (0 or 1).
    """

    v, a, w, ndt = theta

    # Simulate the DDM until it hits a boundary or reaches T_max
    t_steps = np.arange(0, t_max, dt)
    # Set the initial condition as offset from the starting point
    x0 = w

    x = integrate_ddm(v, x0, t_steps)

    rt, choice = find_rt_and_choice(x.flatten(), a * np.ones_like(t_steps), dt)

    return rt + ndt, choice


def simulate_ddm_collapsing(
    theta: Union[list, np.ndarray, torch.Tensor], dt: float = 0.01, t_max: float = 5.0
) -> tuple:
    """
    Simulates the DDM with linearly collapsing boundaries.

    Args:
        theta: A tuple of parameters (v, a, w, T, gamma).
        dt: Time step for the simulation.
        T_max: Maximum simulation time.

    Returns:
        rt: Reaction time.
        choice: Choice (0 or 1).
    """

    v, a, w, ndt, gamma = theta

    # Simulate the DDM until it hits a boundary or reaches T_max
    t_steps = np.arange(0, t_max, dt)
    x0 = w

    x = integrate_ddm(v, x0, t_steps)

    # Define the time-varying boundaries
    def boundaries(t):
        return a - gamma * t

    rt, choice = find_rt_and_choice(x.flatten(), boundaries(t_steps), dt)

    return rt + ndt, choice


def encode_zero_choice_as_negative_rts(rts_and_choices: torch.Tensor) -> torch.Tensor:
    """Returns the reaction times with zero choices encoded as negative values."""
    signed_rts = rts_and_choices[:, :1].clone()
    mask = rts_and_choices[:, 1] == 0
    signed_rts[mask, 0] *= -1
    return signed_rts
