"""Tests for target capping and metrics (``cmapss.metrics``).

Expected values are worked out by hand in each test.
"""

import numpy as np
import pytest

from cmapss.metrics import cap_label, capped_target, score


def test_capped_target_clips_only_values_above_cap():
    out = capped_target([5, 99, 100, 101, 250], 100)
    np.testing.assert_array_equal(out, [5, 99, 100, 100, 100])


def test_uncapped_target_is_unchanged():
    y = [3.0, 150.0, 400.0]
    np.testing.assert_array_equal(capped_target(y, None), y)


def test_cap_label():
    assert cap_label(None) == "uncapped"
    assert cap_label(175) == "175"


def test_score_hand_calculation():
    y = np.array([10.0, 20.0, 40.0, 130.0])
    p = np.array([14.0, 17.0, 40.0, 110.0])
    # errors p-y = [4, -3, 0, -20]
    s = score(y, p)
    assert s["raw_MAE"] == pytest.approx((4 + 3 + 0 + 20) / 4)
    assert s["raw_RMSE"] == pytest.approx(np.sqrt((16 + 9 + 0 + 400) / 4))
    assert s["raw_bias"] == pytest.approx((4 - 3 + 0 - 20) / 4)
    # R2 = 1 - SS_res / SS_tot, with mean(y)=50 -> SS_tot = 1600+900+100+6400
    assert s["raw_R2"] == pytest.approx(1 - 425 / 9000)
    assert s["n"] == 4


def test_near_failure_metrics_use_actual_rul_at_most_30():
    y = np.array([5.0, 30.0, 31.0, 100.0])
    p = np.array([15.0, 25.0, 0.0, 100.0])
    s = score(y, p)
    assert s["near_n"] == 2  # 5 and 30 count; 31 does not
    assert s["near_MAE"] == pytest.approx((10 + 5) / 2)
    assert s["near_bias"] == pytest.approx((10 - 5) / 2)


def test_near_metrics_are_nan_when_no_engine_is_near_failure():
    s = score([50.0, 60.0], [55.0, 70.0])
    assert s["near_n"] == 0
    assert np.isnan(s["near_MAE"]) and np.isnan(s["near_bias"])


def test_common100_clips_both_prediction_and_truth():
    # Truth 300 vs prediction 120 is a 180 raw error but 0 after clipping.
    s = score([300.0, 50.0], [120.0, 50.0])
    assert s["raw_RMSE"] == pytest.approx(np.sqrt(180**2 / 2))
    assert s["common100_RMSE"] == pytest.approx(0.0)


def test_perfect_prediction_gives_zero_error_and_r2_of_one():
    y = np.array([10.0, 50.0, 90.0])
    s = score(y, y)
    assert s["raw_RMSE"] == 0 and s["raw_MAE"] == 0 and s["raw_bias"] == 0
    assert s["raw_R2"] == pytest.approx(1.0)


def test_positive_bias_means_overestimated_life():
    assert score([10.0, 20.0], [30.0, 40.0])["raw_bias"] > 0


def test_r2_is_nan_for_single_point_or_constant_truth():
    assert np.isnan(score([10.0], [12.0])["raw_R2"])
    assert np.isnan(score([10.0, 10.0], [12.0, 9.0])["raw_R2"])


def test_shape_mismatch_and_nonfinite_predictions_are_rejected():
    with pytest.raises(ValueError, match="shape mismatch"):
        score([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="non-finite"):
        score([1.0, 2.0], [1.0, np.nan])
