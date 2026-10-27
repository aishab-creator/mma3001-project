"""Engine-separated splits and truncated validation snapshots.

Validation here mimics how the official test set is built: each held-out
engine is observed only up to a random point in its life, and a single
prediction is made at the last visible cycle. Keeping whole engines on one
side of the split stops rows from the same engine appearing in both training
and validation.
"""

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

#: Earliest point of an engine's life (fraction) at which a snapshot may end.
MIN_LIFE_FRACTION = 0.4


def engine_split(train, seed, test_size=0.2):
    """Split a training table into fit and validation engines.

    Parameters
    ----------
    train : pandas.DataFrame
        Table with an ``engine_id`` column.
    seed : int
        Random seed for the split.
    test_size : float, optional
        Fraction of *engines* (not rows) held out (default 0.2).

    Returns
    -------
    (pandas.DataFrame, pandas.DataFrame)
        ``(fit_raw, val_raw)``; no engine appears in both.

    Raises
    ------
    AssertionError
        If the two sides share an engine (should be impossible; checked
        explicitly because leakage here would invalidate every result).
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    fit_idx, val_idx = next(splitter.split(train, groups=train.engine_id))
    fit_raw, val_raw = train.iloc[fit_idx].copy(), train.iloc[val_idx].copy()
    assert set(fit_raw.engine_id).isdisjoint(set(val_raw.engine_id))
    return fit_raw, val_raw


def snapshots(val, seed):
    """Truncate each validation engine at a random observation point.

    Parameters
    ----------
    val : pandas.DataFrame
        Validation engines with ``engine_id``, ``cycle`` and ``RUL_raw``.
    seed : int
        Seed for the cut-off draw, so every training cap is validated on
        identical snapshots.

    Returns
    -------
    pandas.DataFrame
        Only rows with ``cycle <= cutoff`` for each engine, where the cutoff
        is drawn uniformly between 40% and 100% of that engine's lifetime.

    Notes
    -----
    Truncate **before** computing temporal features. ``RUL_raw`` is kept as
    computed from the full engine, so it still encodes the true failure time.
    """
    rng = np.random.default_rng(seed)
    cutoffs = {}
    for engine, g in val.groupby("engine_id"):
        end = int(g.cycle.max())
        low = max(1, int(np.ceil(MIN_LIFE_FRACTION * end)))
        cutoffs[engine] = int(rng.integers(low, end + 1))
    return val.loc[val.cycle <= val.engine_id.map(cutoffs)].copy()
