"""Tests for splits, snapshots and the model pipeline.

Includes a small end-to-end run (synthetic data -> features -> capped fit ->
metrics) to confirm the pieces work together and respect the cap.
"""

import numpy as np
import pytest

from cmapss.features import build_features
from cmapss.metrics import capped_target, score
from cmapss.models import make_model
from cmapss.validation import MIN_LIFE_FRACTION, engine_split, snapshots


def test_engine_split_is_disjoint_and_sized_by_engines(train):
    fit, val = engine_split(train, seed=42, test_size=0.2)
    assert set(fit.engine_id).isdisjoint(val.engine_id)
    assert val.engine_id.nunique() == 2  # 20% of 10 engines
    assert len(fit) + len(val) == len(train)


def test_engine_split_is_reproducible(train):
    a = engine_split(train, 42)[1].engine_id.unique()
    b = engine_split(train, 42)[1].engine_id.unique()
    np.testing.assert_array_equal(a, b)


def test_snapshots_truncate_each_engine_within_life_bounds(train):
    snap = snapshots(train, seed=42)
    full_life = train.groupby("engine_id").cycle.max()
    kept = snap.groupby("engine_id").cycle.max()
    assert set(kept.index) == set(full_life.index)  # every engine still present
    assert (kept <= full_life).all()
    assert (kept >= np.ceil(MIN_LIFE_FRACTION * full_life)).all()


def test_snapshots_are_identical_for_the_same_seed(train):
    a = snapshots(train, 62)
    b = snapshots(train, 62)
    assert a.equals(b)


def test_snapshot_keeps_true_rul_of_the_full_engine(train):
    snap = snapshots(train, 42)
    last = snap.groupby("engine_id").tail(1)
    full_life = train.groupby("engine_id").cycle.max()
    # True RUL at the cut-off = failure cycle - cut-off cycle (not 0).
    expected = full_life.loc[last.engine_id].to_numpy() - last.cycle.to_numpy()
    np.testing.assert_array_equal(last.RUL_raw.to_numpy(), expected)


def test_model_is_reproducible_with_same_seed():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = X[:, 0] * 10 + 50
    p1 = make_model(20, 1).fit(X, y).predict(X)
    p2 = make_model(20, 1).fit(X, y).predict(X)
    # The forest uses all CPU cores, so floating-point summation order can
    # differ by ~1e-16; the models are otherwise identical.
    np.testing.assert_allclose(p1, p2, rtol=1e-9)


def test_model_median_imputer_handles_missing_values():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = X[:, 0] * 10 + 50
    X_missing = X.copy()
    X_missing[::7, 1] = np.nan
    pred = make_model(20, 1).fit(X, y).predict(X_missing)
    assert np.isfinite(pred).all()


def test_pipeline_end_to_end_respects_training_cap(train):
    fit_raw, val_raw = engine_split(train, 42)
    partial = snapshots(val_raw, 42)
    fit, val, features = build_features(fit_raw, partial, "Improved RF")
    last = val.groupby("engine_id", as_index=False).tail(1)

    cap = 30
    y_capped = capped_target(fit.RUL_raw, cap)
    assert y_capped.max() == cap  # labels really were clipped
    model = make_model(30, 42).fit(fit[features], y_capped)
    pred = np.maximum(0, model.predict(last[features]))

    # A forest trained on targets <= cap can never predict above the cap.
    assert pred.max() <= cap + 1e-9
    metrics = score(last.RUL_raw, pred)
    assert np.isfinite(metrics["raw_RMSE"]) and metrics["n"] == len(last)


def test_pipeline_learns_signal_better_than_mean_predictor(train):
    """Sanity check: drifting synthetic sensors should beat predicting the mean."""
    fit_raw, val_raw = engine_split(train, 3)
    fit, val, features = build_features(fit_raw, val_raw, "Improved RF")
    model = make_model(60, 0).fit(fit[features], fit.RUL_raw)
    rmse_model = score(val.RUL_raw, model.predict(val[features]))["raw_RMSE"]
    rmse_mean = score(val.RUL_raw, np.full(len(val), fit.RUL_raw.mean()))["raw_RMSE"]
    assert rmse_model < rmse_mean
