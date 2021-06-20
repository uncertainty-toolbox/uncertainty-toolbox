"""
Tests for visualizations.
"""

import pytest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from uncertainty_toolbox.viz import (
    plot_xy,
    plot_intervals,
    plot_intervals_ordered,
    plot_calibration,
    plot_adversarial_group_calibration,
    plot_sharpness,
    plot_residuals_vs_stds,
)


@pytest.fixture
def get_test_set():
    y_pred = np.array([1, 2, 3])
    y_std = np.array([0.1, 0.5, 1])
    y_true = np.array([1.5, 3, 2])
    x = np.array([4, 5, 6.5])
    return y_pred, y_std, y_true, x


@pytest.fixture
def get_fig_ax():
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    return fig, ax


def test_plot_xy_returns(get_test_set, get_fig_ax):
    """Test if plot_xy returns correct type."""
    y_pred, y_std, y_true, x = get_test_set
    fig, ax = get_fig_ax
    ax = plot_xy(y_pred, y_std, y_true, x, ax=ax)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_intervals_returns(get_test_set, get_fig_ax):
    """Test if plot_intervals returns correct type."""
    y_pred, y_std, y_true, _ = get_test_set
    fig, ax = get_fig_ax
    ax = plot_intervals(y_pred, y_std, y_true, ax=ax)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_intervals_ordered_returns(get_test_set, get_fig_ax):
    """Test if plot_intervals_ordered returns correct type."""
    y_pred, y_std, y_true, _ = get_test_set
    fig, ax = get_fig_ax
    ax = plot_intervals_ordered(y_pred, y_std, y_true, ax=ax)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_calibration_returns(get_test_set, get_fig_ax):
    """Test if plot_calibration returns correct type."""
    y_pred, y_std, y_true, _ = get_test_set
    fig, ax = get_fig_ax
    ax = plot_calibration(y_pred, y_std, y_true, ax=ax)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_adversarial_group_calibration_returns(get_test_set, get_fig_ax):
    """Test if plot_adversarial_group_calibration returns correct type."""
    y_pred, y_std, y_true, _ = get_test_set
    fig, ax = get_fig_ax
    ax = plot_adversarial_group_calibration(y_pred, y_std, y_true, ax=ax)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_sharpness_returns(get_test_set, get_fig_ax):
    """Test if plot_adversarial_group_calibration returns correct type."""
    _, y_std, _, _ = get_test_set
    fig, ax = get_fig_ax
    ax = plot_sharpness(y_std, ax=ax)
    assert isinstance(ax, matplotlib.axes.Axes)


def test_plot_residuals_vs_stds_returns(get_test_set, get_fig_ax):
    """Test if plot_residuals_vs_stds returns correct type."""
    y_pred, y_std, y_true, _ = get_test_set
    fig, ax = get_fig_ax
    ax = plot_residuals_vs_stds(y_pred, y_std, y_true, ax=ax)
    assert isinstance(ax, matplotlib.axes.Axes)
