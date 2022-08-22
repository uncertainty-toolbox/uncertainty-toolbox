"""
Visualizations for predictive uncertainties and metrics.
"""
from typing import Union, Tuple, List, Any, NoReturn
import pathlib

import numpy as np
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)
from shapely.geometry import Polygon, LineString
from shapely.ops import polygonize, unary_union

from uncertainty_toolbox.metrics_calibration import (
    get_proportion_lists,
    get_proportion_lists_vectorized,
    adversarial_group_calibration,
)


def plot_xy(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    x: np.ndarray,
    n_subset: Union[int, None] = None,
    ylims: Union[Tuple[float, float], None] = None,
    xlims: Union[Tuple[float, float], None] = None,
    num_stds_confidence_bound: int = 2,
    leg_loc: Union[int, str] = 3,
    ax: Union[matplotlib.axes.Axes, None] = None,
) -> matplotlib.axes.Axes:
    """Plot one-dimensional inputs with associated predicted values, predictive
    uncertainties, and true values.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        x: 1D array of input values for the held out dataset.
        n_subset: Number of points to plot after filtering.
        ylims: a tuple of y axis plotting bounds, given as (lower, upper).
        xlims: a tuple of x axis plotting bounds, given as (lower, upper).
        num_stds_confidence_bound: width of confidence band, in terms of number of
            standard deviations.
        leg_loc: location of legend as a str or legend code int.
        ax: matplotlib.axes.Axes object.

    Returns:
        matplotlib.axes.Axes object with plot added.
    """
    # Create ax if it doesn't exist
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Order points in order of increasing x
    order = np.argsort(x)
    y_pred, y_std, y_true, x = (
        y_pred[order],
        y_std[order],
        y_true[order],
        x[order],
    )

    # Optionally select a subset
    if n_subset is not None:
        [y_pred, y_std, y_true, x] = filter_subset([y_pred, y_std, y_true, x], n_subset)

    intervals = num_stds_confidence_bound * y_std

    h1 = ax.plot(x, y_true, ".", mec="#ff7f0e", mfc="None")
    h2 = ax.plot(x, y_pred, "-", c="#1f77b4", linewidth=2)
    h3 = ax.fill_between(
        x,
        y_pred - intervals,
        y_pred + intervals,
        color="lightsteelblue",
        alpha=0.4,
    )
    ax.legend(
        [h1[0], h2[0], h3],
        ["Observations", "Predictions", "$95\%$ Interval"],
        loc=leg_loc,
    )

    # Format plot
    if ylims is not None:
        ax.set_ylim(ylims)

    if xlims is not None:
        ax.set_xlim(xlims)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("Confidence Band")
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable="box")

    return ax


def plot_intervals(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    n_subset: Union[int, None] = None,
    ylims: Union[Tuple[float, float], None] = None,
    num_stds_confidence_bound: int = 2,
    ax: Union[matplotlib.axes.Axes, None] = None,
) -> matplotlib.axes.Axes:
    """Plot predictions and predictive intervals versus true values.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        n_subset: Number of points to plot after filtering.
        ylims: a tuple of y axis plotting bounds, given as (lower, upper).
        num_stds_confidence_bound: width of intervals, in terms of number of standard
            deviations.
        ax: matplotlib.axes.Axes object.

    Returns:
        matplotlib.axes.Axes object with plot added.
    """
    # Create ax if it doesn't exist
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Optionally select a subset
    if n_subset is not None:
        [y_pred, y_std, y_true] = filter_subset([y_pred, y_std, y_true], n_subset)

    # Compute intervals
    intervals = num_stds_confidence_bound * y_std

    # Plot
    ax.errorbar(
        y_true,
        y_pred,
        intervals,
        fmt="o",
        ls="none",
        linewidth=1.5,
        c="#1f77b4",
        alpha=0.5,
    )
    h1 = ax.plot(y_true, y_pred, "o", c="#1f77b4")

    # Determine lims
    if ylims is None:
        intervals_lower_upper = [y_pred - intervals, y_pred + intervals]
        lims_ext = [
            int(np.floor(np.min(intervals_lower_upper[0]))),
            int(np.ceil(np.max(intervals_lower_upper[1]))),
        ]
    else:
        lims_ext = ylims

    # plot 45-degree line
    h2 = ax.plot(lims_ext, lims_ext, "--", linewidth=1.5, c="#ff7f0e")

    # Legend
    ax.legend([h1[0], h2[0]], ["Predictions", "$f(x) = x$"], loc=4)

    # Format plot
    ax.set_xlim(lims_ext)
    ax.set_ylim(lims_ext)
    ax.set_xlabel("Observed Values")
    ax.set_ylabel("Predicted Values and Intervals")
    ax.set_title("Prediction Intervals")
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable="box")

    return ax


def plot_intervals_ordered(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    n_subset: Union[int, None] = None,
    ylims: Union[Tuple[float, float], None] = None,
    num_stds_confidence_bound: int = 2,
    ax: Union[matplotlib.axes.Axes, None] = None,
) -> matplotlib.axes.Axes:
    """Plot predictions and predictive intervals versus true values, with points ordered
    by true value along x-axis.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        n_subset: Number of points to plot after filtering.
        ylims: a tuple of y axis plotting bounds, given as (lower, upper).
        num_stds_confidence_bound: width of intervals, in terms of number of standard
            deviations.
        ax: matplotlib.axes.Axes object.

    Returns:
        matplotlib.axes.Axes object with plot added.
    """
    # Create ax if it doesn't exist
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Optionally select a subset
    if n_subset is not None:
        [y_pred, y_std, y_true] = filter_subset([y_pred, y_std, y_true], n_subset)

    order = np.argsort(y_true.flatten())
    y_pred, y_std, y_true = y_pred[order], y_std[order], y_true[order]
    xs = np.arange(len(order))
    intervals = num_stds_confidence_bound * y_std

    # Plot
    ax.errorbar(
        xs,
        y_pred,
        intervals,
        fmt="o",
        ls="none",
        linewidth=1.5,
        c="#1f77b4",
        alpha=0.5,
    )
    h1 = ax.plot(xs, y_pred, "o", c="#1f77b4")
    h2 = ax.plot(xs, y_true, "--", linewidth=2.0, c="#ff7f0e")

    # Legend
    ax.legend([h1[0], h2[0]], ["Predicted Values", "Observed Values"], loc=4)

    # Determine lims
    if ylims is None:
        intervals_lower_upper = [y_pred - intervals, y_pred + intervals]
        lims_ext = [
            int(np.floor(np.min(intervals_lower_upper[0]))),
            int(np.ceil(np.max(intervals_lower_upper[1]))),
        ]
    else:
        lims_ext = ylims

    # Format plot
    ax.set_ylim(lims_ext)
    ax.set_xlabel("Index (Ordered by Observed Value)")
    ax.set_ylabel("Predicted Values and Intervals")
    ax.set_title("Ordered Prediction Intervals")
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable="box")

    return ax


def plot_calibration(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    n_subset: Union[int, None] = None,
    curve_label: Union[str, None] = None,
    show: bool = False,
    vectorized: bool = True,
    exp_props: Union[np.ndarray, None] = None,
    obs_props: Union[np.ndarray, None] = None,
    ax: Union[matplotlib.axes.Axes, None] = None,
) -> matplotlib.axes.Axes:
    """Plot the observed proportion vs prediction proportion of outputs falling into a
    range of intervals, and display miscalibration area.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        n_subset: Number of points to plot after filtering.
        curve_label: legend label str for calibration curve.
        vectorized: plot using get_proportion_lists_vectorized.
        exp_props: plot using the given expected proportions.
        obs_props: plot using the given observed proportions.
        ax: matplotlib.axes.Axes object.

    Returns:
        matplotlib.axes.Axes object with plot added.
    """
    # Create ax if it doesn't exist
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Optionally select a subset
    if n_subset is not None:
        [y_pred, y_std, y_true] = filter_subset([y_pred, y_std, y_true], n_subset)

    if (exp_props is None) or (obs_props is None):
        # Compute exp_proportions and obs_proportions
        if vectorized:
            (
                exp_proportions,
                obs_proportions,
            ) = get_proportion_lists_vectorized(y_pred, y_std, y_true)
        else:
            (exp_proportions, obs_proportions) = get_proportion_lists(
                y_pred, y_std, y_true
            )
    else:
        # If expected and observed proportions are given
        exp_proportions = np.array(exp_props).flatten()
        obs_proportions = np.array(obs_props).flatten()
        if exp_proportions.shape != obs_proportions.shape:
            raise RuntimeError("exp_props and obs_props shape mismatch")

    # Set label
    if curve_label is None:
        curve_label = "Predictor"

    # Plot
    ax.plot([0, 1], [0, 1], "--", label="Ideal", c="#ff7f0e")
    ax.plot(exp_proportions, obs_proportions, label=curve_label, c="#1f77b4")
    ax.fill_between(exp_proportions, exp_proportions, obs_proportions, alpha=0.2)

    # Format plot
    ax.set_xlabel("Predicted Proportion in Interval")
    ax.set_ylabel("Observed Proportion in Interval")
    ax.axis("square")

    buff = 0.01
    ax.set_xlim([0 - buff, 1 + buff])
    ax.set_ylim([0 - buff, 1 + buff])

    ax.set_title("Average Calibration")

    # Compute miscalibration area
    polygon_points = []
    for point in zip(exp_proportions, obs_proportions):
        polygon_points.append(point)
    for point in zip(reversed(exp_proportions), reversed(exp_proportions)):
        polygon_points.append(point)
    polygon_points.append((exp_proportions[0], obs_proportions[0]))
    polygon = Polygon(polygon_points)
    x, y = polygon.exterior.xy  # original data
    ls = LineString(np.c_[x, y])  # closed, non-simple
    lr = LineString(ls.coords[:] + ls.coords[0:1])
    mls = unary_union(lr)
    polygon_area_list = [poly.area for poly in polygonize(mls)]
    miscalibration_area = np.asarray(polygon_area_list).sum()

    # Annotate plot with the miscalibration area
    ax.text(
        x=0.95,
        y=0.05,
        s="Miscalibration area = %.2f" % miscalibration_area,
        verticalalignment="bottom",
        horizontalalignment="right",
        fontsize="small",
    )

    return ax


def plot_adversarial_group_calibration(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    n_subset: Union[int, None] = None,
    cali_type: str = "mean_abs",
    curve_label: Union[str, None] = None,
    group_size: Union[np.ndarray, None] = None,
    score_mean: Union[np.ndarray, None] = None,
    score_stderr: Union[np.ndarray, None] = None,
    ax: Union[matplotlib.axes.Axes, None] = None,
) -> matplotlib.axes.Axes:
    """Plot adversarial group calibration plots by varying group size from 0% to 100% of
    dataset size and recording the worst calibration occurred for each group size.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        n_subset: Number of points to plot after filtering.
        cali_type: Calibration type str.
        curve_label: legend label str for calibration curve.
        group_size: 1D array of group size ratios in [0, 1].
        score_mean: 1D array of metric means for group size ratios in group_size.
        score_stderr: 1D array of metric standard devations for group size ratios in group_size.
        ax: matplotlib.axes.Axes object.

    Returns:
        matplotlib.axes.Axes object with plot added.
    """
    # Create ax if it doesn't exist
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    # Optionally select a subset
    if n_subset is not None:
        [y_pred, y_std, y_true] = filter_subset([y_pred, y_std, y_true], n_subset)

    # Compute group_size, score_mean, score_stderr
    if (group_size is None) or (score_mean is None):
        # Compute adversarial group calibration
        adv_group_cali_namespace = adversarial_group_calibration(
            y_pred, y_std, y_true, cali_type=cali_type
        )
        group_size = adv_group_cali_namespace.group_size
        score_mean = adv_group_cali_namespace.score_mean
        score_stderr = adv_group_cali_namespace.score_stderr
    else:
        # If expected and observed proportions are give
        group_size = np.array(group_size).flatten()
        score_mean = np.array(score_mean).flatten()
        score_stderr = np.array(score_stderr).flatten()
        if (group_size.shape != score_mean.shape) or (
            group_size.shape != score_stderr.shape
        ):
            raise RuntimeError(
                "Input arrays for adversarial group calibration shape mismatch"
            )

    # Set label
    if curve_label is None:
        curve_label = "Predictor"

    # Plot
    ax.plot(group_size, score_mean, "-o", label=curve_label, c="#1f77b4")
    ax.fill_between(
        group_size,
        score_mean - score_stderr,
        score_mean + score_stderr,
        alpha=0.2,
    )

    # Format plot
    buff = 0.02
    ax.set_xlim([0 - buff, 1 + buff])
    ax.set_ylim([0 - buff, 0.5 + buff])
    ax.set_xlabel("Group size")
    ax.set_ylabel("Calibration Error of Worst Group")
    ax.set_title("Adversarial Group Calibration")

    return ax


def plot_sharpness(
    y_std: np.ndarray,
    n_subset: Union[int, None] = None,
    ax: Union[matplotlib.axes.Axes, None] = None,
) -> matplotlib.axes.Axes:
    """Plot sharpness of the predictive uncertainties.

    Args:
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        n_subset: Number of points to plot after filtering.
        ax: matplotlib.axes.Axes object.

    Returns:
        matplotlib.axes.Axes object with plot added.
    """
    # Create ax if it doesn't exist
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Optionally select a subset
    if n_subset is not None:
        [y_pred, y_std, y_true] = filter_subset([y_pred, y_std, y_true], n_subset)

    # Plot sharpness curve
    ax.hist(y_std, edgecolor="#1f77b4", color="#a5c8e1", density=True)

    # Format plot
    xlim = (y_std.min(), y_std.max())
    ax.set_xlim(xlim)
    ax.set_xlabel("Predicted Standard Deviation")
    ax.set_ylabel("Normalized Frequency")
    ax.set_title("Sharpness")
    ax.set_yticklabels([])
    ax.set_yticks([])

    # Calculate and report sharpness
    sharpness = np.sqrt(np.mean(y_std**2))
    ax.axvline(x=sharpness, label="sharpness", color="k", linewidth=2, ls="--")

    if sharpness < (xlim[0] + xlim[1]) / 2:
        text = "\n  Sharpness = %.2f" % sharpness
        h_align = "left"
    else:
        text = "\nSharpness = %.2f  " % sharpness
        h_align = "right"

    ax.text(
        x=sharpness,
        y=ax.get_ylim()[1],
        s=text,
        verticalalignment="top",
        horizontalalignment=h_align,
        fontsize="small",
    )

    return ax


def plot_residuals_vs_stds(
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: np.ndarray,
    n_subset: Union[int, None] = None,
    ax: Union[matplotlib.axes.Axes, None] = None,
) -> matplotlib.axes.Axes:
    """Plot absolute value of the prediction residuals versus standard deviations of the
    predictive uncertainties.

    Args:
        y_pred: 1D array of the predicted means for the held out dataset.
        y_std: 1D array of the predicted standard deviations for the held out dataset.
        y_true: 1D array of the true labels in the held out dataset.
        n_subset: Number of points to plot after filtering.
        ax: matplotlib.axes.Axes object.

    Returns:
        matplotlib.axes.Axes object with plot added.
    """
    # Create ax if it doesn't exist
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Optionally select a subset
    if n_subset is not None:
        [y_pred, y_std, y_true] = filter_subset([y_pred, y_std, y_true], n_subset)

    # Compute residuals
    residuals = y_true - y_pred

    # Put stds on same scale as residuals
    residuals_sum = np.sum(np.abs(residuals))
    y_std_scaled = (y_std / np.sum(y_std)) * residuals_sum

    # Plot residuals vs standard devs
    h1 = ax.plot(y_std_scaled, np.abs(residuals), "o", c="#1f77b4")

    # Plot 45-degree line
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    lims = [np.min([xlims[0], ylims[0]]), np.max([xlims[1], ylims[1]])]
    h2 = ax.plot(lims, lims, "--", c="#ff7f0e")

    # Legend
    ax.legend([h1[0], h2[0]], ["Predictions", "$f(x) = x$"], loc=4)

    # Format plot
    ax.set_xlabel("Standard Deviations (Scaled)")
    ax.set_ylabel("Residuals (Absolute Value)")
    ax.set_title("Residuals vs. Predictive Standard Deviations")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.axis("square")

    return ax


def filter_subset(input_list: List[List[Any]], n_subset: int) -> List[List[Any]]:
    """Keep only n_subset random indices from all lists given in input_list.

    Args:
        input_list: list of lists.
        n_subset: Number of points to plot after filtering.

    Returns:
        List of all input lists with sizes reduced to n_subset.
    """
    assert type(n_subset) is int
    n_total = len(input_list[0])
    idx = np.random.choice(range(n_total), n_subset, replace=False)
    idx = np.sort(idx)
    output_list = []
    for inp in input_list:
        outp = inp[idx]
        output_list.append(outp)
    return output_list


def set_style(style_str: str = "default") -> NoReturn:
    """Set the matplotlib plotting style.

    Args:
        style_str: string for style file.
    """
    if style_str == "default":
        plt.style.use((pathlib.Path(__file__).parent / "matplotlibrc").resolve())


def save_figure(
    file_name: str = "figure",
    ext_list: Union[list, str, None] = None,
    white_background: bool = True,
) -> NoReturn:
    """Save matplotlib figure for all extensions in ext_list.

    Args:
        file_name: name of saved image file.
        ext_list: list of strings (or single string) denoting file type.
        white_background: set background of image to white if True.
    """

    # Default ext_list
    if ext_list is None:
        ext_list = ["pdf", "png"]

    # If ext_list is a single str
    if isinstance(ext_list, str):
        ext_list = [ext_list]

    # Set facecolor and edgecolor
    (fc, ec) = ("w", "w") if white_background else ("none", "none")

    # Save each type in ext_list
    for ext in ext_list:
        save_str = file_name + "." + ext
        plt.savefig(save_str, bbox_inches="tight", facecolor=fc, edgecolor=ec)
        print(f"Saved figure {save_str}")


def update_rc(key_str: str, value: Any) -> NoReturn:
    """Update matplotlibrc parameters.

    Args:
        key_str: string for a matplotlibrc parameter.
        value: associated value to set the matplotlibrc parameter.
    """
    plt.rcParams.update({key_str: value})
