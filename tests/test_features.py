"""Tests for leakage-safe feature engineering (``cmapss.features``).

The central property is *causality*: the features at cycle ``t`` of an engine
must not change if anything after cycle ``t`` (or any other engine) changes.
"""

import numpy as np
import pandas as pd
import pytest

from cmapss.features import (
    add_regimes,
    add_temporal_features,
    build_features,
    temporal_columns,
)

SENSORS = ["sensor_2", "sensor_3"]


def _feature_cols(sensors, trends):
    return temporal_columns(sensors, add_trends=trends)


@pytest.mark.parametrize("trends", [False, True])
def test_features_do_not_depend_on_future_rows(train, trends):
    """Truncating an engine must leave earlier features unchanged."""
    full = add_temporal_features(train, SENSORS, 5, trends)
    cutoff = 25
    trunc_in = train[train.cycle <= cutoff]
    trunc = add_temporal_features(trunc_in, SENSORS, 5, trends)
    cols = _feature_cols(SENSORS, trends)
    a = full[full.cycle <= cutoff].sort_values(["engine_id", "cycle"])[cols]
    b = trunc.sort_values(["engine_id", "cycle"])[cols]
    np.testing.assert_allclose(a.to_numpy(), b.to_numpy(), rtol=0, atol=1e-12)


@pytest.mark.parametrize("trends", [False, True])
def test_changing_future_sensor_values_does_not_change_past_features(train, trends):
    base = add_temporal_features(train, SENSORS, 5, trends)
    tampered_in = train.copy()
    # Corrupt every reading after cycle 30 in every engine.
    tampered_in.loc[tampered_in.cycle > 30, SENSORS] += 1e6
    tampered = add_temporal_features(tampered_in, SENSORS, 5, trends)
    cols = _feature_cols(SENSORS, trends)
    past = base.cycle <= 30
    np.testing.assert_allclose(
        base.loc[past, cols].to_numpy(), tampered.loc[past, cols].to_numpy()
    )


def test_engines_are_independent(train):
    base = add_temporal_features(train, SENSORS, 5)
    tampered_in = train.copy()
    tampered_in.loc[tampered_in.engine_id == 2, SENSORS] *= 7  # change engine 2 only
    tampered = add_temporal_features(tampered_in, SENSORS, 5)
    cols = _feature_cols(SENSORS, False)
    e1 = base.engine_id == 1
    np.testing.assert_allclose(
        base.loc[e1, cols].to_numpy(), tampered.loc[e1, cols].to_numpy()
    )


def test_rolling_mean_std_and_delta_match_hand_calculation():
    df = pd.DataFrame(
        {
            "engine_id": [1] * 4 + [2] * 2,
            "cycle": [1, 2, 3, 4, 1, 2],
            "sensor_2": [10.0, 12.0, 11.0, 15.0, 100.0, 104.0],
        }
    )
    out = add_temporal_features(df, ["sensor_2"], window=3)
    e1 = out[out.engine_id == 1]
    # Window of 3: means over [10], [10,12], [10,12,11], [12,11,15].
    np.testing.assert_allclose(
        e1.sensor_2_rolling_mean, [10, 11, 11, (12 + 11 + 15) / 3]
    )
    # First cycle: std undefined -> 0, delta -> 0. Then sample std (ddof=1).
    assert e1.sensor_2_rolling_std.iloc[0] == 0
    assert e1.sensor_2_rolling_std.iloc[1] == pytest.approx(np.std([10, 12], ddof=1))
    np.testing.assert_allclose(e1.sensor_2_delta, [0, 2, -1, 4])
    # Engine 2 starts fresh and does not see engine 1's values.
    e2 = out[out.engine_id == 2]
    np.testing.assert_allclose(e2.sensor_2_rolling_mean, [100, 102])
    assert e2.sensor_2_delta.iloc[0] == 0


def test_change_from_start_and_slope_hand_calculation():
    df = pd.DataFrame(
        {
            "engine_id": [1] * 4,
            "cycle": [1, 2, 3, 4],
            "sensor_2": [10.0, 14.0, 18.0, 30.0],
        }
    )
    out = add_temporal_features(df, ["sensor_2"], window=3, add_trends=True)
    np.testing.assert_allclose(out.sensor_2_change_from_start, [0, 4, 8, 20])
    # slope = (x - x.shift(2)) / min(index, 2): NaN->0, NaN->0, (18-10)/2, (30-14)/2
    np.testing.assert_allclose(out.sensor_2_slope, [0, 0, 4, 8])


def test_output_is_finite_and_does_not_modify_input(train):
    before = train.copy()
    out = add_temporal_features(train, SENSORS, 5, True)
    pd.testing.assert_frame_equal(train, before)
    assert np.isfinite(out[_feature_cols(SENSORS, True)].to_numpy()).all()


def test_build_features_drops_sensors_constant_on_fit_only(train):
    fit = train[train.engine_id <= 7].copy()
    other = train[train.engine_id > 7].copy()
    # A sensor that varies only in the *evaluation* data must still be dropped.
    fit["sensor_5"] = 3.0
    other["sensor_5"] = np.linspace(0, 1, len(other))
    _, _, features = build_features(fit, other, "Improved RF")
    assert "sensor_1" not in features  # constant everywhere
    assert "sensor_5" not in features  # constant on fit data
    assert "sensor_2" in features


def test_build_features_rejects_unknown_variant(train):
    with pytest.raises(ValueError, match="unknown variant"):
        build_features(train, train, "RF + magic")


def test_build_features_columns_exist_and_are_finite(train):
    fit = train[train.engine_id <= 7]
    other = train[train.engine_id > 7]
    for variant in ["Improved RF", "RF + trends", "RF + regimes"]:
        f, o, features = build_features(fit, other, variant)
        assert set(features) <= set(f.columns) and set(features) <= set(o.columns)
        assert np.isfinite(f[features].to_numpy()).all()
        assert np.isfinite(o[features].to_numpy()).all()


def test_regime_statistics_come_from_fit_data_only(train):
    fit = train[train.engine_id <= 7].copy()
    other_a = train[train.engine_id > 7].copy()
    other_b = other_a.copy()
    other_b["sensor_2"] += 1000  # wildly different evaluation data
    f_a, _ = add_regimes(fit, other_a, ["sensor_2"])
    f_b, _ = add_regimes(fit, other_b, ["sensor_2"])
    # Fit-side regime features must be identical regardless of the other frame.
    np.testing.assert_allclose(f_a.sensor_2_regime_z, f_b.sensor_2_regime_z)
    np.testing.assert_array_equal(f_a.operating_regime, f_b.operating_regime)


def test_regime_z_scores_are_centred_within_each_regime(train):
    from conftest import make_raw
    from cmapss.data import prepare_frame

    multi = prepare_frame(make_raw(10, seed=4, n_regimes=3), "train")
    f, _ = add_regimes(multi, multi, ["sensor_2"], n_regimes=3)
    means = f.groupby("operating_regime").sensor_2_regime_z.mean()
    np.testing.assert_allclose(means.to_numpy(), 0, atol=1e-9)
