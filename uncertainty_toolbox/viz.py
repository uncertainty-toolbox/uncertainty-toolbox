"""
Visualizations for predictive uncertainties and metrics.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
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


def plot_intervals(y_pred, y_std, y_true, n_subset=None, ylims=None, show=False):
    """
    Plot predicted values (y_pred) and intervals (y_std) vs observed values (y_true).
    """
    if n_subset is not None:
        [y_pred, y_std, y_true] = filter_subset([y_pred, y_std, y_true], n_subset)

    intervals = 2 * y_std  # TODO set with argument

    # Plot
    fig = plt.figure()
    fig.set_size_inches(5.0, 5.0)
    _ = plt.errorbar(
        y_true,
        y_pred,
        intervals,
        fmt="o",
        ls="none",
        linewidth=2.0,
        c="#1f77b4",
        alpha=0.5,
    )
    plt.plot(y_true, y_pred, "o", c="#1f77b4")
    ax = plt.gca()

    # Determine lims
    if ylims is None:
        intervals_lower_upper = [y_pred - intervals, y_pred + intervals]
        lims_ext = [
            int(np.floor(np.min(intervals_lower_upper[0]))),
            int(np.ceil(np.max(intervals_lower_upper[1]))),
        ]
    else:
        lims_ext = ylims

    # plot 45-degree parity line
    _ = ax.plot(lims_ext, lims_ext, "--", linewidth=1.5, c="#ff7f0e")

    # Format
    _ = ax.set_xlim(lims_ext)
    _ = ax.set_ylim(lims_ext)
    _ = ax.set_xlabel("Observed Values")
    _ = ax.set_ylabel("Predicted Values and Intervals")
    _ = ax.set_aspect("equal", "box")

    plt.title("Prediction Intervals")

    if show:
        plt.show()


def plot_intervals_ordered(
    y_pred, y_std, y_true, n_subset=None, ylims=None, show=False
):
    """
    Plot predicted values (y_pred) and intervals (y_std) vs observed values (y_true).
    """
    if n_subset is not None:
        [y_pred, y_std, y_true] = filter_subset([y_pred, y_std, y_true], n_subset)

    order = np.argsort(y_true.flatten())
    y_pred, y_std, y_true = y_pred[order], y_std[order], y_true[order]
    xs = np.arange(len(order))

    intervals = 2 * y_std  # TODO set with argument

    # Plot
    fig = plt.figure()
    fig.set_size_inches(5.0, 5.0)
    _ = plt.errorbar(
        xs,
        y_pred,
        intervals,
        fmt="o",
        ls="none",
        linewidth=1.5,
        c="#1f77b4",
        alpha=0.5,
    )
    h1 = plt.plot(xs, y_pred, "o", c="#1f77b4")
    h2 = plt.plot(xs, y_true, "--", linewidth=2.0, c="#ff7f0e")
    ax = plt.gca()

    # Legend
    plt.legend([h1[0], h2[0]], ["Predicted Values", "Observed Values"], loc=4)

    # Determine lims
    if ylims is None:
        intervals_lower_upper = [y_pred - intervals, y_pred + intervals]
        lims_ext = [
            int(np.floor(np.min(intervals_lower_upper[0]))),
            int(np.ceil(np.max(intervals_lower_upper[1]))),
        ]
    else:
        lims_ext = ylims

    # Format
    _ = ax.set_ylim(lims_ext)
    # _ = ax.set_xlabel('Observed Values Order')
    _ = ax.set_xlabel("Index (Ordered by Observed Value)")
    _ = ax.set_ylabel("Predicted Values and Intervals")
    _ = ax.set_aspect("auto", "box")

    plt.title("Ordered Prediction Intervals")

    if show:
        plt.show()


def plot_xy(
    y_pred, y_std, y_true, x, n_subset=None, ylims=None, xlims=None, show=False
):
    """Plot 1D input (x) and predicted/true (y_pred/y_true) values."""
    if n_subset is not None:
        [y_pred, y_std, y_true, x] = filter_subset([y_pred, y_std, y_true, x], n_subset)

    intervals = 2 * y_std  # TODO set with argument

    fig = plt.figure()
    fig.set_size_inches(5.0, 5.0)
    h1 = plt.plot(x, y_true, ".", mec="#ff7f0e", mfc="None")
    h2 = plt.plot(x, y_pred, "-", c="#1f77b4", linewidth=2)
    h3 = plt.fill_between(
        x,
        y_pred - intervals,
        y_pred + intervals,
        color="lightsteelblue",
        alpha=0.4,
    )
    plt.legend(
        [h1[0], h2[0], h3],
        ["Observations", "Predictions", "95\% Interval"],
        loc=3,
    )

    if ylims is not None:
        plt.ylim(ylims)

    if xlims is not None:
        plt.xlim(xlims)

    plt.xlabel("$x$")
    plt.ylabel("$y$")

    plt.title("Confidence Band")

    if show:
        plt.show()


def plot_parity(
    y_pred, y_true, n_subset=None, lims=None, axlabels=None, hexbins=False, show=False
):
    """
    Make parity plot using predicted values (y_pred) and observed values (y_true).
    """
    if n_subset is not None:
        [y_pred, y_true] = filter_subset([y_pred, y_true], n_subset)

    # Set lims
    if lims is None:
        print("Lims is None. Setting lims now:")
        min_max_true = (y_true.min(), y_true.max())
        min_max_pred = (y_pred.min(), y_pred.max())
        lims = (
            np.min((min_max_true[0], min_max_pred[0])),
            np.max((min_max_true[1], min_max_pred[1])),
        )
        lims_diff = lims[1] - lims[0]
        lims_ext = (lims[0] - 0.1 * lims_diff, lims[1] + 0.1 * lims_diff)

        print("min_max_true: {}".format(min_max_true))
        print("min_max_pred: {}".format(min_max_pred))
        print("lims: {}".format(lims))
        print("lims_ext: {}".format(lims_ext))

    # Set axlabels
    if axlabels is None:
        axlabels = ("Observed Values", "Predicted Values")

    # Set residuals
    residuals = y_pred - y_true

    # Plotting
    if hexbins:
        grid = sns.jointplot(
            y_true, y_pred, kind="hex", bins="log", gridsize=25, extent=lims * 2
        )
    else:
        grid = sns.jointplot(
            y_true,
            y_pred,
            kind="scatter",
            space=0,
            # marginal_kws=dict(kde=True, shade=True))
            marginal_kws=dict(kde=True),
        )

    ax = grid.ax_joint
    _ = ax.set_xlim(lims_ext)
    _ = ax.set_ylim(lims_ext)
    _ = ax.plot(lims_ext, lims_ext, "--")
    _ = ax.set_xlabel(axlabels[0])
    _ = ax.set_ylabel(axlabels[1])

    plt.title("Prediction Metrics")

    # Calculate the error metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mdae = median_absolute_error(y_true, y_pred)
    marpd = np.abs(2 * residuals / (np.abs(y_pred) + np.abs(y_true))).mean() * 100
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1]

    # Report
    fontsize = 12
    text = (
        "  MDAE = %.2f\n" % mdae
        + "  MAE = %.2f\n" % mae
        + "  RMSE = %.2f\n" % rmse
        + "  MARPD = %i%%\n" % marpd
        + "  R2 = %.2f\n" % r2
        + "  PPMCC = %i%%\n" % corr
    )
    # print('\nPredictive accuracy metrics:')
    # print('MDAE = %.2f' % mdae)
    # print('MAE = %.2f' % mae)
    # print('RMSE = %.2f' % rmse)
    # print('MARPD = %.2f' % marpd)
    # print('R2 = %.2f' % r2)
    # print('PPMCC = %.2f' % corr)
    _ = ax.text(
        x=lims[0],
        y=lims[1],
        s=text,
        horizontalalignment="left",
        verticalalignment="top",
        fontsize=fontsize,
    )
    fig = plt.gcf()
    fig.set_size_inches(5.0, 5.0)

    if show:
        plt.show()


def plot_calibration(
    y_pred,
    y_std,
    y_true,
    n_subset=None,
    curve_label=None,
    show=False,
    vectorized=True,
    exp_props=None,
    obs_props=None,
):
    """
    Make calibration plot using predicted mean values (y_pred), predicted std
    values (y_std), and observed values (y_true).
    """
    if n_subset is not None:
        [y_pred, y_std, y_true] = filter_subset([y_pred, y_std, y_true], n_subset)

    if (exp_props is None) or (obs_props is None):
        # Compute exp_proportions and obs_proportions
        if vectorized:
            (exp_proportions, obs_proportions) = get_proportion_lists_vectorized(
                y_pred, y_std, y_true
            )
        else:
            (exp_proportions, obs_proportions) = get_proportion_lists(
                y_pred, y_std, y_true
            )
    else:
        # If expected and observed proportions are give
        exp_proportions = np.array(exp_props).flatten()
        obs_proportions = np.array(obs_props).flatten()
        if exp_proportions.shape != obs_proportions.shape:
            raise RuntimeError("exp_props and obs_props shape mismatch")

    # Set figure defaults
    fontsize = 12

    # Set label
    if curve_label is None:
        curve_label = "Predictor"
    # Plot
    plt.figure()
    plt.plot([0, 1], [0, 1], "--", label="Ideal", c="#ff7f0e")
    plt.plot(exp_proportions, obs_proportions, label=curve_label, c="#1f77b4")
    plt.fill_between(exp_proportions, exp_proportions, obs_proportions, alpha=0.2)
    plt.xlabel("Predicted proportion in interval")
    plt.ylabel("Observed proportion in interval")
    plt.axis("square")
    buff = 0.01
    plt.xlim([0 - buff, 1 + buff])
    plt.ylim([0 - buff, 1 + buff])

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
    plt.text(
        x=0.95,
        y=0.05,
        s="Miscalibration area = %.2f" % miscalibration_area,
        verticalalignment="bottom",
        horizontalalignment="right",
        fontsize=fontsize,
    )

    fig = plt.gcf()
    fig.set_size_inches(5.0, 5.0)

    plt.title("Average Calibration")

    if show:
        plt.show()


def plot_adversarial_group_calibration(
    y_pred,
    y_std,
    y_true,
    n_subset=None,
    cali_type="mean_abs",
    curve_label=None,
    show=False,
    group_size=None,
    score_mean=None,
    score_stderr=None,
):
    """
    Plot adversarial group calibration plots by spanning group size between 0% to 100% of
    dataset size and recording the worst calibration occurred for each group size.
    """
    if n_subset is not None:
        [y_pred, y_std, y_true] = filter_subset([y_pred, y_std, y_true], n_subset)

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
    plt.figure()
    plt.plot(group_size, score_mean, "-o", label=curve_label, c="#1f77b4")
    plt.fill_between(
        group_size, score_mean - score_stderr, score_mean + score_stderr, alpha=0.2
    )
    plt.xlabel("Group size")
    plt.ylabel("Calibration error of worst group")
    plt.axis("square")
    buff = 0.02
    plt.xlim([0 - buff, 1 + buff])
    plt.ylim([0 - buff, 0.5 + buff])

    fig = plt.gcf()
    fig.set_size_inches(7.0, 5.0)

    plt.title("Adversarial Group Calibration")

    if show:
        plt.show()


def plot_sharpness(y_std, n_subset=None):
    """
    Make sharpness plot using predicted std values (y_std).
    """
    if n_subset is not None:
        [y_std] = filter_subset([y_std], n_subset)

    # Plot sharpness curve
    figsize = (5, 5)
    fontsize = 12
    xlim = (y_std.min(), y_std.max())
    fig_sharp = plt.figure(figsize=figsize)
    ax_sharp = sns.distplot(y_std, kde=False, norm_hist=True)
    ax_sharp.set_xlim(xlim)
    ax_sharp.set_xlabel("Predicted standard deviation")
    ax_sharp.set_ylabel("Normalized frequency")
    ax_sharp.set_yticklabels([])
    ax_sharp.set_yticks([])

    # Calculate and report sharpness
    sharpness = np.sqrt(np.mean(y_std ** 2))
    _ = ax_sharp.axvline(x=sharpness, label="sharpness")
    if sharpness < (xlim[0] + xlim[1]) / 2:
        text = "\n  Sharpness = %.2f" % sharpness
        h_align = "left"
    else:
        text = "\nSharpness = %.2f " % sharpness
        h_align = "right"
    _ = ax_sharp.text(
        x=sharpness,
        y=ax_sharp.get_ylim()[1],
        s=text,
        verticalalignment="top",
        horizontalalignment=h_align,
        fontsize=fontsize,
    )


def plot_residuals_vs_stds(residuals, stds):
    # Put stds on same scale as residuals
    res_sum = np.sum(np.abs(residuals))
    stds_scaled = (stds / np.sum(stds)) * res_sum
    # Plot
    plt.figure()
    plt.plot(stds_scaled, np.abs(residuals), "x")
    lims = [
        np.min([plt.xlim()[0], plt.ylim()[0]]),
        np.max([plt.xlim()[1], plt.ylim()[1]]),
    ]
    plt.plot(lims, lims, "--", label="Ideal")
    plt.xlabel("Standard deviations (scaled)")
    plt.ylabel("Residuals (absolute value)")
    plt.axis("square")
    plt.xlim(lims)
    plt.ylim(lims)


def filter_subset(input_list, n_subset):
    """Keep only n_subset random indices from everything in input_list."""
    assert type(n_subset) is int
    n_total = len(input_list[0])
    idx = np.random.choice(range(n_total), n_subset, replace=False)
    idx = np.sort(idx)
    output_list = []
    for inp in input_list:
        outp = inp[idx]
        output_list.append(outp)
    return output_list


if __name__ == "__main__":
    import data
    import metrics_calibration

    y_pred, y_std, y_true, x_true = data.synthetic_sine_heteroscedastic(100)
    print(
        metrics_calibration.adversarial_group_calibration(
            y_pred, y_std, y_true, "root_mean_sq"
        )
    )
    plot_calibration(y_pred, y_std, y_true, show=True)
    plot_adversarial_group_calibration(y_pred, y_std, y_true, show=True)
    plot_sharpness(y_std)
    plt.show()
