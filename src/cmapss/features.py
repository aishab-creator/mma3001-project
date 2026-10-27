"""Leakage-safe feature engineering.

Two kinds of feature are built:

* **Temporal features** (:func:`add_temporal_features`): rolling mean and
  standard deviation, first difference and (optionally) trend features. Every
  one is computed *per engine* with a *trailing* window, so the value at cycle
  ``t`` depends only on cycles ``<= t`` of the same engine. No future sensor
  value can leak into a feature, and one engine cannot influence another.
* **Operating-regime features** (:func:`add_regimes`): operating settings are
  clustered into regimes and each sensor is standardised within its regime.
  The clustering and the regime statistics are learned on the *fit* partition
  only and then applied to the other partition.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .data import SENSOR_COLS, SETTING_COLS

#: Variant names understood by :func:`build_features`.
VARIANTS = ("Improved RF", "RF + trends", "RF + regimes")

#: Sensors whose standard deviation on the fit data is below this are dropped.
MIN_SENSOR_STD = 1e-8


def add_temporal_features(df, sensors, window=5, add_trends=False):
    """Append rolling and difference features for each sensor.

    Parameters
    ----------
    df : pandas.DataFrame
        Table with ``engine_id``, ``cycle`` and the ``sensors`` columns.
    sensors : list of str
        Sensor columns to expand.
    window : int, optional
        Trailing window length in cycles (default 5).
    add_trends : bool, optional
        If True, also add ``<sensor>_slope`` and ``<sensor>_change_from_start``.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` (sorted by engine and cycle) with new columns
        ``<sensor>_rolling_mean``, ``_rolling_std``, ``_delta`` and, if
        requested, ``_slope`` and ``_change_from_start``.

    Notes
    -----
    * ``rolling_std`` is undefined for the first cycle of an engine (one
      observation); it is filled with 0 there. ``delta`` is also 0 at the
      first cycle. Neither value is imputed from other engines.
    * ``slope`` is the change over the last ``window - 1`` cycles divided by
      the number of cycles actually available (at least 1), so early cycles
      use a shorter span instead of reading data from before the engine began.
    * ``change_from_start`` is the sensor minus that engine's first reading.
    * All windows look backwards only, which is what prevents leakage.
    """
    df = df.sort_values(["engine_id", "cycle"]).copy()
    group = df.groupby("engine_id", sort=False)
    new_cols = {}
    for s in sensors:
        # min_periods=1 lets the window shrink at the start of an engine.
        new_cols[f"{s}_rolling_mean"] = group[s].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        new_cols[f"{s}_rolling_std"] = group[s].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        ).fillna(0)
        new_cols[f"{s}_delta"] = group[s].diff().fillna(0)
        if add_trends:
            new_cols[f"{s}_slope"] = group[s].transform(
                lambda x: (x - x.shift(window - 1))
                / np.minimum(np.arange(len(x)), window - 1).clip(min=1)
            ).fillna(0)
            new_cols[f"{s}_change_from_start"] = df[s] - group[s].transform("first")
    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)


def temporal_columns(sensors, add_trends=False):
    """List the column names that :func:`add_temporal_features` creates.

    Parameters
    ----------
    sensors : list of str
        Sensor columns that were expanded.
    add_trends : bool, optional
        Whether trend features were requested.

    Returns
    -------
    list of str
        Names in sensor-major order, e.g. ``sensor_2_rolling_mean``.
    """
    suffixes = ["rolling_mean", "rolling_std", "delta"]
    if add_trends:
        suffixes += ["slope", "change_from_start"]
    return [f"{s}_{suffix}" for s in sensors for suffix in suffixes]


def add_regimes(fit_df, other_df, sensors, n_regimes=6, random_state=42):
    """Add operating-regime id and regime-standardised sensors.

    Settings are scaled and clustered with k-means **on** ``fit_df`` **only**;
    ``other_df`` is assigned to the nearest learned cluster. For each sensor,
    the per-regime mean and standard deviation are likewise taken from
    ``fit_df`` and applied to both frames. This removes the sensor swing
    caused by throttle/load changes (FD002, FD004 have six conditions).

    Parameters
    ----------
    fit_df, other_df : pandas.DataFrame
        Training-side and evaluation-side tables. ``other_df`` never
        influences the clusters or statistics.
    sensors : list of str
        Sensors to standardise.
    n_regimes : int, optional
        Number of k-means clusters (default 6, the number of C-MAPSS
        operating conditions in FD002/FD004).
    random_state : int, optional
        Seed for k-means.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        Copies of ``fit_df`` and ``other_df`` with ``operating_regime`` and
        ``<sensor>_regime_z`` columns.
    """
    scaler = StandardScaler().fit(fit_df[SETTING_COLS])
    fit_settings = scaler.transform(fit_df[SETTING_COLS])
    other_settings = scaler.transform(other_df[SETTING_COLS])
    k = min(n_regimes, len(fit_df))
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(
        fit_settings
    )
    fit = fit_df.copy()
    other = other_df.copy()
    fit["operating_regime"] = kmeans.predict(fit_settings)
    other["operating_regime"] = kmeans.predict(other_settings)
    for s in sensors:
        # Regime statistics come from the fit partition alone.
        stats = fit.groupby("operating_regime")[s].agg(["mean", "std"])
        overall_std = float(fit[s].std())
        if not np.isfinite(overall_std) or overall_std < MIN_SENSOR_STD:
            overall_std = 1.0  # avoid dividing by ~0 for a flat sensor
        means_fit = fit["operating_regime"].map(stats["mean"])
        means_other = other["operating_regime"].map(stats["mean"])
        # A regime with a single row has NaN std: fall back to the overall std.
        stds_fit = (
            fit["operating_regime"].map(stats["std"]).fillna(overall_std).clip(lower=MIN_SENSOR_STD)
        )
        stds_other = (
            other["operating_regime"].map(stats["std"]).fillna(overall_std).clip(lower=MIN_SENSOR_STD)
        )
        fit[f"{s}_regime_z"] = (fit[s] - means_fit) / stds_fit
        other[f"{s}_regime_z"] = (other[s] - means_other) / stds_other
    return fit, other


def build_features(fit_raw, other_raw, variant, window=5, random_state=42):
    """Build the feature table for one model variant.

    Parameters
    ----------
    fit_raw : pandas.DataFrame
        Validated training-side table (output of
        :func:`cmapss.data.prepare_frame`).
    other_raw : pandas.DataFrame
        Validated evaluation-side table (validation snapshots or test data).
    variant : {"Improved RF", "RF + trends", "RF + regimes"}
        ``"Improved RF"`` uses rolling mean/std and deltas;
        ``"RF + trends"`` adds slope and change-from-start;
        ``"RF + regimes"`` adds operating-regime features.
    window : int, optional
        Rolling window in cycles.
    random_state : int, optional
        Seed for the regime k-means.

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame, list of str)
        Feature tables for the fit and other partitions, and the ordered list
        of model feature column names.

    Raises
    ------
    ValueError
        If ``variant`` is not recognised.

    Notes
    -----
    Near-constant sensors (std <= 1e-8) are dropped. That decision uses
    ``fit_raw`` only, so the evaluation data cannot influence which inputs
    the model sees.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    sensors = [s for s in SENSOR_COLS if fit_raw[s].std() > MIN_SENSOR_STD]
    add_trends = variant == "RF + trends"
    fit = add_temporal_features(fit_raw, sensors, window, add_trends)
    other = add_temporal_features(other_raw, sensors, window, add_trends)
    features = (
        ["cycle"] + SETTING_COLS + sensors + temporal_columns(sensors, add_trends)
    )
    if variant == "RF + regimes":
        fit, other = add_regimes(fit, other, sensors, random_state=random_state)
        features += ["operating_regime"] + [f"{s}_regime_z" for s in sensors]
    return fit, other, features
