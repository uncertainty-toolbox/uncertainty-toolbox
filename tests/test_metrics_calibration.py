"""
Tests for calibration metrics.
"""
import numpy as np
import pytest

from uncertainty_toolbox.metrics_calibration import (
    sharpness,
    root_mean_squared_calibration_error,
    mean_absolute_calibration_error,
    expected_normalized_calibration_error,
    adversarial_group_calibration,
    miscalibration_area,
    get_proportion_lists_vectorized,
    get_proportion_lists,
    get_proportion_in_interval,
    get_prediction_interval,
    get_proportion_under_quantile,
    get_quantile,
)


@pytest.fixture
def supply_test_set():
    y_pred = np.array([1, 2, 3])
    y_std = np.array([0.1, 0.5, 1])
    y_true = np.array([1.5, 3, 2])
    return y_pred, y_std, y_true


def test_sharpness_on_test_set(supply_test_set):
    """Test sharpness on the test set for some dummy values."""
    _, test_std, _ = supply_test_set
    assert np.abs(sharpness(test_std) - 0.648074069840786) < 1e-6


def test_root_mean_squared_calibration_error_on_test_set(supply_test_set):
    """Test root mean squared calibration error on some dummy values."""
    test_rmsce_nonvectorized_interval = root_mean_squared_calibration_error(
        *supply_test_set,
        num_bins=100,
        vectorized=False,
        recal_model=None,
        prop_type="interval"
    )
    test_rmsce_vectorized_interval = root_mean_squared_calibration_error(
        *supply_test_set,
        num_bins=100,
        vectorized=True,
        recal_model=None,
        prop_type="interval"
    )
    assert (
        np.abs(test_rmsce_nonvectorized_interval - test_rmsce_vectorized_interval)
        < 1e-6
    )
    assert np.abs(test_rmsce_vectorized_interval - 0.4165757476562379) < 1e-6

    test_rmsce_nonvectorized_quantile = root_mean_squared_calibration_error(
        *supply_test_set,
        num_bins=100,
        vectorized=False,
        recal_model=None,
        prop_type="quantile"
    )
    test_rmsce_vectorized_quantile = root_mean_squared_calibration_error(
        *supply_test_set,
        num_bins=100,
        vectorized=True,
        recal_model=None,
        prop_type="quantile"
    )
    assert (
        np.abs(test_rmsce_nonvectorized_quantile - test_rmsce_vectorized_quantile)
        < 1e-6
    )
    assert np.abs(test_rmsce_vectorized_quantile - 0.30362567774902066) < 1e-6


def test_mean_absolute_calibration_error_on_test_set(supply_test_set):
    """Test mean absolute calibration error on some dummy values."""
    test_mace_nonvectorized_interval = mean_absolute_calibration_error(
        *supply_test_set,
        num_bins=100,
        vectorized=False,
        recal_model=None,
        prop_type="interval"
    )
    test_mace_vectorized_interval = mean_absolute_calibration_error(
        *supply_test_set,
        num_bins=100,
        vectorized=True,
        recal_model=None,
        prop_type="interval"
    )
    assert (
        np.abs(test_mace_nonvectorized_interval - test_mace_vectorized_interval) < 1e-6
    )
    assert np.abs(test_mace_vectorized_interval - 0.3733333333333335) < 1e-6

    test_mace_nonvectorized_quantile = mean_absolute_calibration_error(
        *supply_test_set,
        num_bins=100,
        vectorized=False,
        recal_model=None,
        prop_type="quantile"
    )
    test_mace_vectorized_quantile = mean_absolute_calibration_error(
        *supply_test_set,
        num_bins=100,
        vectorized=True,
        recal_model=None,
        prop_type="quantile"
    )
    assert (
        np.abs(test_mace_nonvectorized_quantile - test_mace_vectorized_quantile) < 1e-6
    )
    assert np.abs(test_mace_vectorized_quantile - 0.23757575757575758) < 1e-6


def test_adversarial_group_calibration_on_test_set(supply_test_set):
    """Test adversarial group calibration on test set for some dummy values."""
    test_out_interval = adversarial_group_calibration(
        *supply_test_set,
        cali_type="mean_abs",
        prop_type="interval",
        num_bins=100,
        num_group_bins=10,
        draw_with_replacement=False,
        num_trials=10,
        num_group_draws=10,
        verbose=False
    )

    assert np.max(np.abs(test_out_interval.group_size - np.linspace(0, 1, 10))) < 1e-6
    assert np.all(test_out_interval.score_mean < 0.5)
    assert np.abs(test_out_interval.score_mean[-1] - 0.3733333333333335) < 1e-6
    assert np.min(test_out_interval.score_stderr) >= 0

    test_out_quantile = adversarial_group_calibration(
        *supply_test_set,
        cali_type="mean_abs",
        prop_type="quantile",
        num_bins=100,
        num_group_bins=10,
        draw_with_replacement=False,
        num_trials=10,
        num_group_draws=10,
        verbose=False
    )

    assert np.max(np.abs(test_out_quantile.group_size - np.linspace(0, 1, 10))) < 1e-6
    assert np.all(test_out_quantile.score_mean < 0.5)
    assert np.abs(test_out_quantile.score_mean[-1] - 0.2375757575757576) < 1e-6
    assert np.min(test_out_quantile.score_stderr) >= 0


def test_miscalibration_area_on_test_set(supply_test_set):
    """Test miscalibration area on some dummy values."""
    test_miscal_area_nonvectorized_interval = miscalibration_area(
        *supply_test_set,
        num_bins=100,
        vectorized=False,
        recal_model=None,
        prop_type="interval"
    )
    test_miscal_area_vectorized_interval = miscalibration_area(
        *supply_test_set,
        num_bins=100,
        vectorized=True,
        recal_model=None,
        prop_type="interval"
    )
    assert (
        np.abs(
            test_miscal_area_nonvectorized_interval
            - test_miscal_area_vectorized_interval
        )
        < 1e-6
    )
    assert np.abs(test_miscal_area_vectorized_interval - 0.37710437710437716) < 1e-6

    test_miscal_area_nonvectorized_quantile = miscalibration_area(
        *supply_test_set,
        num_bins=100,
        vectorized=False,
        recal_model=None,
        prop_type="quantile"
    )
    test_miscal_area_vectorized_quantile = miscalibration_area(
        *supply_test_set,
        num_bins=100,
        vectorized=True,
        recal_model=None,
        prop_type="quantile"
    )
    assert (
        np.abs(
            test_miscal_area_nonvectorized_quantile
            - test_miscal_area_vectorized_quantile
        )
        < 1e-6
    )
    assert np.abs(test_miscal_area_vectorized_quantile - 0.23916245791245785) < 1e-6


def test_vectorization_for_proportion_list_on_test_set(supply_test_set):
    """Test vectorization in get_proportion_lists on the test set for some dummy values."""
    (
        test_exp_props_nonvec_interval,
        test_obs_props_nonvec_interval,
    ) = get_proportion_lists(
        *supply_test_set, num_bins=100, recal_model=None, prop_type="interval"
    )

    (
        test_exp_props_vec_interval,
        test_obs_props_vec_interval,
    ) = get_proportion_lists_vectorized(
        *supply_test_set, num_bins=100, recal_model=None, prop_type="interval"
    )
    assert (
        np.max(np.abs(test_exp_props_nonvec_interval - test_exp_props_vec_interval))
        < 1e-6
    )
    assert (
        np.max(np.abs(test_obs_props_nonvec_interval - test_obs_props_vec_interval))
        < 1e-6
    )

    (
        test_exp_props_nonvec_quantile,
        test_obs_props_nonvec_quantile,
    ) = get_proportion_lists(
        *supply_test_set, num_bins=100, recal_model=None, prop_type="quantile"
    )

    (
        test_exp_props_vec_quantile,
        test_obs_props_vec_quantile,
    ) = get_proportion_lists_vectorized(
        *supply_test_set, num_bins=100, recal_model=None, prop_type="quantile"
    )
    assert (
        np.max(np.abs(test_exp_props_nonvec_quantile - test_exp_props_vec_quantile))
        < 1e-6
    )
    assert (
        np.max(np.abs(test_obs_props_nonvec_quantile - test_obs_props_vec_quantile))
        < 1e-6
    )


def test_get_proportion_lists_vectorized_on_test_set(supply_test_set):
    """Test get_proportion_lists_vectorized on the test set for some dummy values."""
    (
        test_exp_props_interval,
        test_obs_props_interval,
    ) = get_proportion_lists_vectorized(
        *supply_test_set, num_bins=100, recal_model=None, prop_type="interval"
    )
    assert test_exp_props_interval.shape == test_obs_props_interval.shape
    assert (
        np.max(np.abs(np.unique(test_exp_props_interval) - np.linspace(0, 1, 100)))
        < 1e-6
    )
    assert (
        np.max(
            np.abs(
                np.sort(np.unique(test_obs_props_interval))
                - np.array([0.0, 0.33333333, 0.66666667, 1.0])
            )
        )
        < 1e-6
    )

    (
        test_exp_props_quantile,
        test_obs_props_quantile,
    ) = get_proportion_lists_vectorized(
        *supply_test_set, num_bins=100, recal_model=None, prop_type="quantile"
    )
    assert test_exp_props_quantile.shape == test_obs_props_quantile.shape
    assert (
        np.max(np.abs(np.unique(test_exp_props_quantile) - np.linspace(0, 1, 100)))
        < 1e-6
    )
    assert (
        np.max(
            np.abs(
                np.sort(np.unique(test_obs_props_quantile))
                - np.array([0.0, 0.33333333, 0.66666667, 1.0])
            )
        )
        < 1e-6
    )


def test_get_proportion_lists_on_test_set(supply_test_set):
    """Test get_proportion_lists on the test set for some dummy values."""
    test_exp_props_interval, test_obs_props_interval = get_proportion_lists(
        *supply_test_set, num_bins=100, recal_model=None, prop_type="interval"
    )
    assert len(test_exp_props_interval) == len(test_obs_props_interval)
    assert (
        np.max(np.abs(np.unique(test_exp_props_interval) - np.linspace(0, 1, 100)))
        < 1e-6
    )
    assert (
        np.max(
            np.abs(
                np.sort(np.unique(test_obs_props_interval))
                - np.array([0.0, 0.33333333, 0.66666667, 1.0])
            )
        )
        < 1e-6
    )

    test_exp_props_quantile, test_obs_props_quantile = get_proportion_lists(
        *supply_test_set, num_bins=100, recal_model=None, prop_type="quantile"
    )
    assert len(test_exp_props_quantile) == len(test_obs_props_quantile)
    assert (
        np.max(np.abs(np.unique(test_exp_props_quantile) - np.linspace(0, 1, 100)))
        < 1e-6
    )
    assert (
        np.max(
            np.abs(
                np.sort(np.unique(test_obs_props_quantile))
                - np.array([0.0, 0.33333333, 0.66666667, 1.0])
            )
        )
        < 1e-6
    )


def test_get_proportion_in_interval_on_test_set(supply_test_set):
    """Test get_proportion_in_interval on the test set for some dummy values."""
    test_quantile_value_list = [
        (0.0, 0.0),
        (0.25, 0.0),
        (0.5, 0.0),
        (0.75, 0.3333333333333333),
        (1.0, 1.0),
    ]
    for test_q, test_val in test_quantile_value_list:
        assert (
            np.abs(
                get_proportion_in_interval(*supply_test_set, quantile=test_q) - test_val
            )
            < 1e-6
        )


def test_get_proportion_under_quantile_on_test_set(supply_test_set):
    """Test get_proportion_in_interval on the test set for some dummy values."""
    test_quantile_value_list = [
        (0.0, 0.0),
        (0.25, 0.6666666666666666),
        (0.5, 0.6666666666666666),
        (0.75, 0.6666666666666666),
        (1.0, 1.0),
    ]
    for test_q, test_val in test_quantile_value_list:
        assert (
            np.abs(
                get_proportion_under_quantile(*supply_test_set, quantile=test_q)
                - test_val
            )
            < 1e-6
        )


def test_get_prediction_interval_on_test_set(supply_test_set):
    """Test get_prediction_interval on the test set for some dummy values."""
    test_quantile_value_list = [
        (
            0.01,
            np.array([1.00125335, 2.00626673, 3.01253347]),
            np.array([0.99874665, 1.99373327, 2.98746653]),
        ),
        (
            0.25,
            np.array([1.03186394, 2.15931968, 3.31863936]),
            np.array([0.96813606, 1.84068032, 2.68136064]),
        ),
        (
            0.50,
            np.array([1.06744898, 2.33724488, 3.67448975]),
            np.array([0.93255102, 1.66275512, 2.32551025]),
        ),
        (
            0.75,
            np.array([1.11503494, 2.57517469, 4.15034938]),
            np.array([0.88496506, 1.42482531, 1.84965062]),
        ),
        (
            0.99,
            np.array([1.25758293, 3.28791465, 5.5758293]),
            np.array([0.74241707, 0.71208535, 0.4241707]),
        ),
    ]

    y_pred, y_std, y_true = supply_test_set

    with pytest.raises(Exception):
        bounds = get_prediction_interval(y_pred, y_std, quantile=0.0, recal_model=None)

    with pytest.raises(Exception):
        bounds = get_prediction_interval(y_pred, y_std, quantile=1.0, recal_model=None)

    for test_q, test_upper, test_lower in test_quantile_value_list:
        bounds = get_prediction_interval(
            y_pred, y_std, quantile=test_q, recal_model=None
        )
        upper_bound = bounds.upper
        lower_bound = bounds.lower
        assert np.max(np.abs(upper_bound - test_upper)) < 1e-6
        assert np.max(np.abs(upper_bound - test_upper)) < 1e-6


def test_get_quantile_on_test_set(supply_test_set):
    """Test get_prediction_interval on the test set for some dummy values."""
    test_quantile_value_list = [
        (0.01, np.array([0.76736521, 0.83682606, 0.67365213])),
        (
            0.25,
            np.array([0.93255102, 1.66275512, 2.32551025]),
        ),
        (
            0.50,
            np.array([1.0, 2.0, 3.0]),
        ),
        (
            0.75,
            np.array([1.06744898, 2.33724488, 3.67448975]),
        ),
        (
            0.99,
            np.array([1.23263479, 3.16317394, 5.32634787]),
        ),
    ]

    y_pred, y_std, y_true = supply_test_set

    with pytest.raises(Exception):
        bound = get_quantile(y_pred, y_std, quantile=0.0, recal_model=None)

    with pytest.raises(Exception):
        bound = get_quantile(y_pred, y_std, quantile=1.0, recal_model=None)

    for test_q, test_bound in test_quantile_value_list:
        bound = get_quantile(y_pred, y_std, quantile=test_q, recal_model=None)

        assert np.max(np.abs(bound - test_bound)) < 1e-6

@pytest.fixture
def perfect_prediction():
    """
    Returns y_pred, y_std, y_true in a scenario where the model predictions
    are exactly correct and the same predicted std. 
    We expect the RMSE=0 in every bin, so ENCE should be > 0 if std != 0 
    but let's check with a variety of patterns.
    """
    y_pred = np.array([0.0, 5.0, 10.0])
    y_true = np.array([0.0, 5.0, 10.0])
    y_std = np.array([1.0, 1.0, 1.0])   # Nonzero but identical std
    return y_pred, y_std, y_true


@pytest.fixture
def good_calibration():
    """
    Returns y_pred, y_std, y_true in a scenario where the predicted
    standard deviation aligns perfectly with the actual RMSE across bins.
    
    If we artificially make bin1's RMSE=0.5 and predicted std=0.5,
    bin2's RMSE=1.0 and predicted std=1.0, we expect ENCE near zero.
    """
    y_pred = np.array([0.0, 0.0, 2.0, 2.0])
    y_true = np.array([0.5, -0.5, 3.0, 1.0])  # so errors are 0.5, 0.5, 1.0, 1.0
    # predicted std matches the actual errors:
    y_std = np.array([0.5, 0.5, 1.0, 1.0])
    return y_pred, y_std, y_true


@pytest.fixture
def overconfident_prediction():
    """
    Returns y_pred, y_std, y_true in a scenario where the standard deviations
    are underestimates (overconfident). We expect a nonzero ENCE.
    """
    y_pred = np.array([1.0, 2.0, 3.0])
    y_true = np.array([1.5, 3.0, 2.0])  # errors are 0.5, 1.0, 1.0
    y_std = np.array([0.1, 0.1, 0.1])   # too small => mismatch
    return y_pred, y_std, y_true


def test_ence_perfect_prediction(perfect_prediction):
    """
    Even though predictions are perfect (error=0), if predicted std is not zero,
    each bin's difference is:
       rmv - rmse = rmv - 0 => diff / rmv = 1.0
    so we expect an ENCE around 1.0 if there's only one bin. If more bins,
    the result might vary. We'll check it doesn't crash and yields a sensible value.
    """
    y_pred, y_std, y_true = perfect_prediction
    ence = expected_normalized_calibration_error(y_pred, y_std, y_true, num_bins=1)
    # If bin=1 => rmv=1.0, rmse=0 => diff=1 => ence=1
    assert abs(ence - 1.0) < 1e-6, f"Expected ENCE ~ 1.0, got {ence}"


def test_ence_good_calibration(good_calibration):
    """
    Here the predicted std equals the actual RMSE bin by bin.
    We expect the ENCE to be near zero because there's near-perfect calibration.
    """
    y_pred, y_std, y_true = good_calibration
    # Two distinct std levels => let's do 2 bins
    ence = expected_normalized_calibration_error(y_pred, y_std, y_true, num_bins=2)
    # Ideally, if bin1: RMSE=0.5, RMV=0.5 => diff=0 => 0 error
    #        bin2: RMSE=1.0, RMV=1.0 => diff=0 => 0 error
    # ENCE => 0
    assert abs(ence) < 1e-7, f"Expected near zero ENCE, got {ence}"


def test_ence_overconfident(overconfident_prediction):
    """
    Overconfident scenario => we expect ENCE > 0
    """
    y_pred, y_std, y_true = overconfident_prediction
    ence = expected_normalized_calibration_error(y_pred, y_std, y_true, num_bins=3)
    # Since rmse is around ~0.83 (avg), but predicted std is 0.1, there's a big mismatch
    # We just check that it's definitely > 0
    assert ence > 0.01, f"Expected a noticeably positive ENCE, got {ence}"


def test_ence_raises_for_mismatched_shapes():
    """
    If shapes don't match, should raise ValueError
    """
    y_pred = np.array([1.0, 2.0])
    y_std = np.array([0.1])
    y_true = np.array([1.5, 3.0])
    with pytest.raises(ValueError):
        _ = expected_normalized_calibration_error(y_pred, y_std, y_true)


def test_ence_raises_for_nonpositive_std():
    """
    If any std <= 0, should raise ValueError
    """
    y_pred = np.array([1.0, 2.0, 3.0])
    y_true = np.array([1.5, 2.5, 3.5])
    # Negative std
    y_std = np.array([0.1, -0.1, 0.0])
    with pytest.raises(ValueError):
        _ = expected_normalized_calibration_error(y_pred, y_std, y_true)


def test_ence_raises_for_too_many_bins():
    """
    If num_bins > number_of_points, should raise ValueError
    """
    y_pred = np.array([1.0, 2.0, 3.0])
    y_true = np.array([1.1, 2.5, 3.5])
    y_std = np.array([0.5, 0.5, 0.5])
    with pytest.raises(ValueError):
        _ = expected_normalized_calibration_error(y_pred, y_std, y_true, num_bins=5)
