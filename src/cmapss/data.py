"""Reading and validating NASA C-MAPSS files.

Each C-MAPSS ``train_FD00x.txt`` / ``test_FD00x.txt`` file is a whitespace
separated table with 26 columns: engine id, cycle, three operating settings
and 21 sensor channels. ``RUL_FD00x.txt`` holds one number per test engine:
the true remaining life at that engine's last observed test cycle.

Inputs that do not match this format are **rejected with a** ``ValueError``
(rather than silently repaired), so a bad file fails loudly before any model
is fitted.
"""

from pathlib import Path

import numpy as np
import pandas as pd

#: Names of the three operating-setting columns.
SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]
#: Names of the 21 sensor columns.
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
#: Full column order of a C-MAPSS train/test file.
COLUMNS = ["engine_id", "cycle"] + SETTING_COLS + SENSOR_COLS


def prepare_frame(df, kind, name="data"):
    """Validate a raw C-MAPSS table and add the training RUL label.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw table with 26 unnamed columns, in the NASA column order.
    kind : {"train", "test"}
        ``"train"`` frames are run-to-failure, so a ``RUL_raw`` column
        (``last_cycle_of_engine - cycle``) is added. ``"test"`` frames are
        truncated and get no label (their labels come from ``RUL_FD00x.txt``).
    name : str, optional
        Label used in error messages (usually the file path).

    Returns
    -------
    pandas.DataFrame
        Named columns, sorted by ``(engine_id, cycle)``; for training data,
        also a non-negative ``RUL_raw`` column that is 0 at each engine's
        final cycle.

    Raises
    ------
    ValueError
        If the column count is wrong, any value is NaN/inf, or an
        ``(engine_id, cycle)`` pair is duplicated.

    Notes
    -----
    Sorting guarantees that cycles increase monotonically within each engine,
    which the rolling features in :mod:`cmapss.features` rely on. Duplicated
    pairs are rejected (not dropped) because they would make that ordering
    ambiguous.
    """
    if df.shape[1] != len(COLUMNS):
        raise ValueError(
            f"{name}: expected {len(COLUMNS)} columns, found {df.shape[1]}"
        )
    df = df.copy()
    df.columns = COLUMNS
    # Reject, rather than repair, repeated engine/cycle rows.
    if df[["engine_id", "cycle"]].duplicated().any():
        raise ValueError(f"{name}: duplicated (engine_id, cycle) rows")
    # Every value must be a finite number (no NaN, no +/- inf).
    if not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError(f"{name}: non-finite values found")
    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    if kind == "train":
        # Run-to-failure data: failure happens at each engine's last cycle.
        last_cycle = df.groupby("engine_id").cycle.transform("max")
        df["RUL_raw"] = last_cycle - df.cycle
    return df


def load_data(dataset, kind, data_dir):
    """Read one C-MAPSS file from disk and validate it.

    Parameters
    ----------
    dataset : str
        Subset name, ``"FD001"`` to ``"FD004"``.
    kind : {"train", "test", "RUL"}
        Which file to read: ``train_<dataset>.txt``, ``test_<dataset>.txt`` or
        ``RUL_<dataset>.txt``.
    data_dir : str or pathlib.Path
        Directory containing the NASA text files (``data/raw`` in this repo).

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        For ``"train"``/``"test"``, the validated table from
        :func:`prepare_frame`. For ``"RUL"``, a 1-D float array with one true
        RUL per test engine, ordered by engine id.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is malformed (see :func:`prepare_frame`), or a ``RUL``
        file has more than one column or non-finite values.
    """
    path = Path(data_dir) / f"{kind}_{dataset}.txt"
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if kind == "RUL":
        if df.shape[1] != 1:
            raise ValueError(f"{path}: RUL file must have exactly one column")
        values = df.iloc[:, 0].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: non-finite RUL values")
        return values
    return prepare_frame(df, kind, name=str(path))
