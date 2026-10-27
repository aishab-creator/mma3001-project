"""Training-target capping and evaluation metrics.

A training **cap** ``c`` replaces the label ``RUL`` with ``min(RUL, c)`` when
fitting the model. Evaluation, by contrast, is always done against the true
(uncapped) RUL, because the project objective is full-lifetime prediction.
Mixing the two (scoring against each cap's own capped target) makes small caps
look better only because their target range is smaller.
"""

import numpy as np
from sklearn.metrics import r2_score

#: Engines with true RUL at or below this many cycles count as "near failure".
NEAR_FAILURE_CYCLES = 30
#: Horizon (cycles) for the common-100 diagnostic.
COMMON_HORIZON = 100


def cap_label(cap):
    """Return a string label for a cap (``None`` means uncapped).

    Examples
    --------
    >>> cap_label(100), cap_label(None)
    ('100', 'uncapped')
    """
    return "uncapped" if cap is None else str(int(cap))


def capped_target(y, cap):
    """Clip training labels at ``cap`` cycles.

    Parameters
    ----------
    y : array-like
        True RUL labels (cycles).
    cap : int or None
        Cap in cycles, or ``None`` to leave the labels unchanged.

    Returns
    -------
    numpy.ndarray
        ``min(y, cap)`` as floats; values below the cap are untouched.
        This is *not* a rescaling to 0-1.
    """
    a = np.asarray(y, dtype=float)
    return a if cap is None else np.minimum(a, cap)


def score(y, p):
    """Compute all evaluation metrics for predictions against true RUL.

    Parameters
    ----------
    y : array-like
        True (uncapped) RUL, in cycles.
    p : array-like
        Predicted RUL, same length as ``y``.

    Returns
    -------
    dict
        ``raw_RMSE``, ``raw_MAE``, ``raw_R2`` (NaN if undefined),
        ``raw_bias`` (mean of ``p - y``; positive = overestimated life),
        ``common100_RMSE`` (RMSE after clipping both ``p`` and ``y`` to
        100 cycles; a diagnostic, never used for selection),
        ``near_MAE`` and ``near_bias`` (engines with ``y <= 30``; NaN if
        there are none), ``near_n`` and ``n`` (counts).

    Raises
    ------
    ValueError
        If the shapes differ or ``p`` contains NaN/inf.
    """
    y, p = np.asarray(y, float), np.asarray(p, float)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y {y.shape} vs p {p.shape}")
    if not np.isfinite(p).all():
        raise ValueError("predictions contain non-finite values")
    e = p - y  # signed error: positive means life was overestimated
    near = y <= NEAR_FAILURE_CYCLES
    ce = np.minimum(p, COMMON_HORIZON) - np.minimum(y, COMMON_HORIZON)
    return dict(
        raw_RMSE=float(np.sqrt(np.mean(e**2))),
        raw_MAE=float(np.mean(abs(e))),
        # R2 needs at least two points and some spread in the truth.
        raw_R2=float(r2_score(y, p)) if len(y) > 1 and np.var(y) > 0 else np.nan,
        raw_bias=float(e.mean()),
        common100_RMSE=float(np.sqrt(np.mean(ce**2))),
        near_MAE=float(np.mean(abs(e[near]))) if near.any() else np.nan,
        near_bias=float(e[near].mean()) if near.any() else np.nan,
        near_n=int(near.sum()),
        n=len(y),
    )
