"""Tests for input validation and RUL labelling (``cmapss.data``)."""

import numpy as np
import pandas as pd
import pytest

from cmapss.data import COLUMNS, load_data, prepare_frame


def test_rul_label_is_last_cycle_minus_cycle(raw_train):
    df = prepare_frame(raw_train, "train")
    # Final cycle of every engine has RUL 0, and RUL is never negative.
    last = df.groupby("engine_id").tail(1)
    assert (last.RUL_raw == 0).all()
    assert (df.RUL_raw >= 0).all()
    # Spot-check one engine by hand: RUL = max_cycle - cycle.
    e1 = df[df.engine_id == 1]
    assert (e1.RUL_raw.to_numpy() == (e1.cycle.max() - e1.cycle).to_numpy()).all()


def test_test_frame_gets_no_rul_label(raw_train):
    assert "RUL_raw" not in prepare_frame(raw_train, "test").columns


def test_rows_are_sorted_even_if_input_is_shuffled(raw_train):
    shuffled = raw_train.sample(frac=1, random_state=3)
    df = prepare_frame(shuffled, "train")
    keys = list(zip(df.engine_id, df.cycle))
    assert keys == sorted(keys)
    # Within each engine cycles strictly increase (monotonic).
    assert (df.groupby("engine_id").cycle.diff().dropna() > 0).all()


def test_wrong_column_count_is_rejected(raw_train):
    with pytest.raises(ValueError, match="expected 26 columns"):
        prepare_frame(raw_train.iloc[:, :25], "train")


def test_duplicate_engine_cycle_is_rejected(raw_train):
    dup = pd.concat([raw_train, raw_train.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicated"):
        prepare_frame(dup, "train")


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_values_are_rejected(raw_train, bad):
    broken = raw_train.copy()
    broken.iloc[5, 10] = bad
    with pytest.raises(ValueError, match="non-finite"):
        prepare_frame(broken, "train")


def test_input_is_not_modified(raw_train):
    before = raw_train.copy()
    prepare_frame(raw_train, "train")
    pd.testing.assert_frame_equal(raw_train, before)


def test_load_data_round_trip(tmp_path, raw_train):
    # Write NASA-style whitespace-separated files and read them back.
    raw_train.to_csv(tmp_path / "train_FD001.txt", sep=" ", header=False, index=False)
    pd.Series([12.0, 30.0, 5.0]).to_csv(
        tmp_path / "RUL_FD001.txt", header=False, index=False
    )
    df = load_data("FD001", "train", tmp_path)
    assert list(df.columns[: len(COLUMNS)]) == COLUMNS
    assert "RUL_raw" in df.columns
    rul = load_data("FD001", "RUL", tmp_path)
    assert rul.dtype == float and rul.tolist() == [12.0, 30.0, 5.0]


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data("FD001", "train", tmp_path)


def test_rul_file_with_two_columns_is_rejected(tmp_path):
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(
        tmp_path / "RUL_FD001.txt", sep=" ", header=False, index=False
    )
    with pytest.raises(ValueError, match="exactly one column"):
        load_data("FD001", "RUL", tmp_path)
