"""Shared pytest fixtures: small synthetic C-MAPSS-like data.

The tests never need the real NASA files. The synthetic engines follow the
same 26-column layout, with sensors that drift with age (so models have
something to learn) and one constant sensor (to exercise sensor dropping).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make ``import cmapss`` work without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cmapss.data import COLUMNS, SENSOR_COLS  # noqa: E402


def make_raw(n_engines=10, seed=0, lifetimes=None, n_regimes=1):
    """Build a raw (unnamed-column) table in NASA format.

    Parameters
    ----------
    n_engines : int
        Number of engines.
    seed : int
        Random seed.
    lifetimes : list of int, optional
        Cycles per engine; random between 40 and 80 if omitted.
    n_regimes : int
        Number of distinct operating-setting combinations.
    """
    rng = np.random.default_rng(seed)
    if lifetimes is None:
        lifetimes = rng.integers(40, 81, size=n_engines)
    # Fixed setting combinations so k-means has real clusters to find.
    combos = rng.uniform(0, 1, size=(n_regimes, 3)) * [40, 0.8, 100]
    rows = []
    for eid, life in enumerate(lifetimes, start=1):
        for cyc in range(1, int(life) + 1):
            settings = combos[rng.integers(n_regimes)]
            sensors = []
            for j in range(21):
                if j == 0:
                    sensors.append(500.0)  # constant sensor: should be dropped
                else:
                    # Degradation grows towards failure (as in C-MAPSS): the
                    # drift depends on the fraction of life used, plus noise.
                    frac = cyc / life
                    sensors.append(100 + j + 3.0 * j * frac**2 + rng.normal(0, 0.5))
            rows.append([eid, cyc, *settings, *sensors])
    return pd.DataFrame(rows)


@pytest.fixture
def raw_train():
    """Raw 10-engine run-to-failure table (no column names).

    Uses six operating regimes, matching FD002/FD004, so the regime
    clustering has six genuine groups to find.
    """
    return make_raw(n_engines=10, seed=1, n_regimes=6)


@pytest.fixture
def train(raw_train):
    """Validated training table with ``RUL_raw`` (named columns)."""
    from cmapss.data import prepare_frame

    return prepare_frame(raw_train, "train")


@pytest.fixture
def sensors():
    """All sensor column names."""
    return list(SENSOR_COLS)


@pytest.fixture
def columns():
    """Full NASA column order."""
    return list(COLUMNS)
