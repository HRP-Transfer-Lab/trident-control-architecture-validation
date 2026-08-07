import pandas as pd
import pytest

from trident_validation.splits import (
    SplitValidationError,
    assert_dataset_holdout_exact,
    assert_no_participant_overlap,
    dataset_holdout,
    participant_group_kfold,
    participant_train_test_split,
)
from trident_validation.synthetic import make_synthetic_window_table


def _participant_keys(frame: pd.DataFrame, indices: tuple[int, ...]) -> set[tuple[str, str]]:
    return set(
        frame.loc[list(indices), ["source_dataset", "participant_id"]]
        .astype(str)
        .agg(tuple, axis=1)
        .tolist()
    )


def test_participant_train_test_split_is_leakage_free_and_reproducible():
    frame = make_synthetic_window_table(seed=88)

    split_a = participant_train_test_split(frame, test_size=0.25, seed=20260807)
    split_b = participant_train_test_split(frame, test_size=0.25, seed=20260807)

    assert split_a == split_b
    assert_no_participant_overlap(frame, split_a)
    assert not _participant_keys(frame, split_a.train_indices).intersection(
        _participant_keys(frame, split_a.test_indices)
    )


def test_participant_group_kfold_keeps_all_windows_for_each_participant_together():
    frame = make_synthetic_window_table(seed=89)

    folds = list(participant_group_kfold(frame, n_splits=5, seed=11))

    assert len(folds) == 5
    assert folds == list(participant_group_kfold(frame, n_splits=5, seed=11))
    for split in folds:
        assert_no_participant_overlap(frame, split)


def test_dataset_holdout_is_exact():
    frame = make_synthetic_window_table(seed=90)
    holdout = sorted(frame["source_dataset"].unique())[1]

    split = dataset_holdout(frame, holdout_dataset=holdout)

    assert_dataset_holdout_exact(frame, split)
    assert set(frame.loc[list(split.test_indices), "source_dataset"].unique()) == {holdout}
    assert holdout not in set(frame.loc[list(split.train_indices), "source_dataset"].unique())


def test_dataset_holdout_rejects_missing_source():
    frame = make_synthetic_window_table(seed=91)

    with pytest.raises(SplitValidationError):
        dataset_holdout(frame, holdout_dataset="not_a_source")

