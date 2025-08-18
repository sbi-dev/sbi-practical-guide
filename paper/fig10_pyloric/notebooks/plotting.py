import collections
import copy
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast
from warnings import warn

import matplotlib as mpl
import numpy as np
import six
import torch
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure, FigureBase
from matplotlib.patches import Rectangle
from scipy.stats import binom, gaussian_kde, iqr
from torch import Tensor

from sbi.analysis.conditional_density import eval_conditional_density
from sbi.utils.analysis_utils import pp_vals


def _sbc_rank_plot(
    ranks: Union[Tensor, np.ndarray, List[Tensor], List[np.ndarray]],
    num_posterior_samples: int,
    num_bins: Optional[int] = None,
    plot_type: str = "cdf",
    parameter_labels: Optional[List[str]] = None,
    ranks_labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    num_repeats: int = 50,
    line_alpha: float = 0.8,
    show_uniform_region: bool = True,
    uniform_region_alpha: float = 0.3,
    xlim_offset_factor: float = 0.1,
    num_cols: int = 4,
    params_in_subplots: bool = False,
    show_ylabel: bool = False,
    sharey: bool = False,
    fig: Optional[FigureBase] = None,
    legend_kwargs: Optional[Dict] = None,
    ax=None,  # no type hint to avoid hassle with pyright. Should be `array(Axes).`
    figsize: Optional[tuple] = None,
) -> Tuple[Figure, Axes]:
    """Plot simulation-based calibration ranks as empirical CDFs or histograms.

    Args:
        ranks: Tensor of ranks to be plotted shape (num_sbc_runs, num_parameters), or
            list of Tensors when comparing several sets of ranks, e.g., set of ranks
            obtained from different methods.
        num_bins: number of bins used for binning the ranks, default is
            num_sbc_runs / 20.
        plot_type: type of SBC plot, histograms ("hist") or empirical cdfs ("cdf").
        parameter_labels: list of labels for each parameter dimension.
        ranks_labels: list of labels for each set of ranks.
        colors: list of colors for each parameter dimension, or each set of ranks.
        num_repeats: number of repeats for each empirical CDF step (resolution).
        line_alpha: alpha for cdf lines or histograms.
        show_uniform_region: whether to plot the region showing the cdfs expected under
            uniformity.
        uniform_region_alpha: alpha for region showing the cdfs expected under
            uniformity.
        xlim_offset_factor: factor for empty space left and right of the histogram.
        num_cols: number of subplot columns, e.g., when plotting ranks of many
            parameters.
        params_in_subplots: whether to show each parameter in a separate subplot, or
            all in one.
        show_ylabel: whether to show ylabels and ticks.
        sharey: whether to share the y-labels, ticks, and limits across subplots.
        fig: figure object to plot in.
        ax: axis object, must contain as many sublpots as parameters or len(ranks).
        figsize: dimensions of figure object, default (8, 5) or (len(ranks) * 4, 5).

    Returns:
        fig, ax: figure and axis objects.

    """

    if isinstance(ranks, (Tensor, np.ndarray)):
        ranks_list = [ranks]
    else:
        assert isinstance(ranks, List)
        ranks_list = ranks
    for idx, rank in enumerate(ranks_list):
        assert isinstance(rank, (Tensor, np.ndarray))
        if isinstance(rank, Tensor):
            ranks_list[idx]: np.ndarray = rank.numpy()  # type: ignore

    plot_types = ["hist", "cdf"]
    assert plot_type in plot_types, (
        "plot type {plot_type} not implemented, use one in {plot_types}."
    )

    if legend_kwargs is None:
        legend_kwargs = dict(loc="best", handlelength=0.8)

    num_sbc_runs, num_parameters = ranks_list[0].shape
    num_ranks = len(ranks_list)

    # For multiple methods, and for the hist plots plot each param in a separate subplot
    if num_ranks > 1 or plot_type == "hist":
        params_in_subplots = True

    for ranki in ranks_list:
        assert ranki.shape == ranks_list[0].shape, (
            "all ranks in list must have the same shape."
        )

    num_rows = int(np.ceil(num_parameters / num_cols))
    if figsize is None:
        figsize = (num_parameters * 4, num_rows * 5) if params_in_subplots else (8, 5)

    if parameter_labels is None:
        parameter_labels = [f"dim {i + 1}" for i in range(num_parameters)]
    if ranks_labels is None:
        ranks_labels = [f"rank set {i + 1}" for i in range(num_ranks)]
    if num_bins is None:
        # Recommendation from Talts et al.
        num_bins = num_sbc_runs // 20

    # Plot one row subplot for each parameter, different "methods" on top of each other.
    if params_in_subplots:
        if fig is None or ax is None:
            fig, ax = plt.subplots(
                num_rows,
                min(num_parameters, num_cols),
                figsize=figsize,
                sharey=sharey,
            )
            ax = np.atleast_1d(ax)  # type: ignore
        else:
            assert ax.size >= num_parameters, (
                "There must be at least as many subplots as parameters."
            )
            num_rows = ax.shape[0] if ax.ndim > 1 else 1
        assert ax is not None

        col_idx, row_idx = 0, 0
        for ii, ranki in enumerate(ranks_list):
            for jj in range(num_parameters):
                col_idx = jj if num_rows == 1 else jj % num_cols
                row_idx = jj // num_cols
                plt.sca(ax[col_idx] if num_rows == 1 else ax[row_idx, col_idx])

                if plot_type == "cdf":
                    _plot_ranks_as_cdf(
                        ranki[:, jj],  # type: ignore
                        num_bins,
                        num_repeats,
                        ranks_label=ranks_labels[ii],
                        color=f"C{ii}" if colors is None else colors[ii],
                        xlabel=f"posterior ranks {parameter_labels[jj]}",
                        # Show legend and ylabel only in first subplot.
                        show_ylabel=jj == 0,
                        alpha=line_alpha,
                    )
                    if ii == 0 and show_uniform_region:
                        _plot_cdf_region_expected_under_uniformity(
                            num_sbc_runs,
                            num_bins,
                            num_repeats,
                            alpha=uniform_region_alpha,
                        )
                elif plot_type == "hist":
                    _plot_ranks_as_hist(
                        ranki[:, jj],  # type: ignore
                        num_bins,
                        num_posterior_samples,
                        ranks_label=ranks_labels[ii],
                        color="firebrick" if colors is None else colors[ii],
                        xlabel=f"posterior rank {parameter_labels[jj]}",
                        # Show legend and ylabel only in first subplot.
                        show_ylabel=show_ylabel,
                        alpha=line_alpha,
                        xlim_offset_factor=xlim_offset_factor,
                    )
                    # Plot expected uniform band.
                    _plot_hist_region_expected_under_uniformity(
                        num_sbc_runs,
                        num_bins,
                        num_posterior_samples,
                        alpha=uniform_region_alpha,
                    )
                else:
                    raise ValueError(
                        f"plot_type {plot_type} not defined, use one in {plot_types}"
                    )
                # Remove empty subplots.
        col_idx += 1
        while num_rows > 1 and col_idx < num_cols:
            ax[row_idx, col_idx].axis("off")
            col_idx += 1

    # When there is only one set of ranks show all params in a single subplot.
    else:
        if fig is None or ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        plt.sca(ax)
        ranki = ranks_list[0]
        for jj in range(num_parameters):
            _plot_ranks_as_cdf(
                ranki[:, jj],  # type: ignore
                num_bins,
                num_repeats,
                ranks_label=parameter_labels[jj],
                color=f"C{jj}" if colors is None else colors[jj],
                xlabel="posterior rank",
                # Plot ylabel and legend at last.
                show_ylabel=jj == (num_parameters - 1),
                alpha=line_alpha,
            )
        if show_uniform_region:
            _plot_cdf_region_expected_under_uniformity(
                num_sbc_runs,
                num_bins,
                num_repeats,
                alpha=uniform_region_alpha,
            )

    return fig, ax  # pyright: ignore[reportReturnType]


def _plot_ranks_as_cdf(
    ranks: np.ndarray,
    num_bins: int,
    num_repeats: int,
    ranks_label: Optional[str] = None,
    xlabel: Optional[str] = None,
    color: Optional[str] = None,
    alpha: float = 0.8,
    show_ylabel: bool = True,
    num_ticks: int = 3,
) -> None:
    """Plot ranks as empirical CDFs on the current axis.

    Args:
        ranks: SBC ranks in shape (num_sbc_runs, )
        num_bins: number of bins for the histogram, recommendation is num_sbc_runs / 20.
        num_repeats: number of repeats of each CDF step, i.e., resolution of the eCDF.
        ranks_label: label for the ranks, e.g., when comparing ranks of different
            methods.
        xlabel: label for the current parameter
        color: line color for the cdf.
        alpha: line transparency.
        show_ylabel: whether to show y-label "counts".
        show_legend: whether to show the legend, e.g., when comparing multiple ranks.
        num_ticks: number of ticks on the x-axis.
        legend_kwargs: kwargs for the legend.

    """
    # Generate histogram of ranks.
    hist, *_ = np.histogram(ranks, bins=num_bins, density=False)
    # Construct empirical CDF.
    histcs = hist.cumsum()
    # Plot cdf and repeat each stair step
    plt.plot(
        np.linspace(0, num_bins, num_repeats * num_bins),
        np.repeat(histcs / histcs.max(), num_repeats),
        label=ranks_label,
        color=color,
        alpha=alpha,
    )

    if show_ylabel:
        plt.yticks(np.linspace(0, 1, 3))
        plt.ylabel("empirical CDF")
    else:
        # Plot ticks only
        plt.yticks(np.linspace(0, 1, 3), [])

    plt.ylim(0, 1)
    plt.xlim(0, num_bins)
    plt.xticks(np.linspace(0, num_bins, num_ticks))
    plt.xlabel("posterior rank" if xlabel is None else xlabel)



def _plot_cdf_region_expected_under_uniformity(
    num_sbc_runs: int,
    num_bins: int,
    num_repeats: int,
    alpha: float = 0.2,
    color: str = "gray",
) -> None:
    """Plot region of empirical cdfs expected under uniformity on the current axis."""

    # Construct uniform histogram.
    uni_bins = binom(num_sbc_runs, p=1 / num_bins).ppf(0.5) * np.ones(num_bins)
    uni_bins_cdf = uni_bins.cumsum() / uni_bins.sum()
    # Decrease value one in last entry by epsilon to find valid
    # confidence intervals.
    uni_bins_cdf[-1] -= 1e-9

    lower = [binom(num_sbc_runs, p=p).ppf(0.005) for p in uni_bins_cdf]
    upper = [binom(num_sbc_runs, p=p).ppf(0.995) for p in uni_bins_cdf]

    # Plot grey area with expected ECDF.
    plt.fill_between(
        x=np.linspace(0, num_bins, num_repeats * num_bins),
        y1=np.repeat(lower / np.max(lower), num_repeats),
        y2=np.repeat(upper / np.max(upper), num_repeats),  # pyright: ignore[reportArgumentType]
        color=color,
        alpha=alpha,
        label="expected under uniformity",
        edgecolor="none",
    )
