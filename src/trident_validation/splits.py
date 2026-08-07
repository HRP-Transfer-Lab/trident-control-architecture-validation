"""Leakage-safe split utilities."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterator, Sequence

import numpy as np
import pandas as pd


class SplitValidationError(ValueError):
    """Raised when a split would leak participants or sources."""


@dataclass(frozen=True)
class SplitDefinition:
    """Train/test row-index split with reproducibility metadata."""

    name: str
    seed: int | None
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    participant_columns: tuple[str, ...]
    fold_index: int | None = None
    n_splits: int | None = None
    holdout_dataset: str | None = None

    def to_manifest(self) -> dict[str, int | str | None]:
        """Return counts only; never include participant identifiers."""

        return {
            "name": self.name,
            "seed": self.seed,
            "fold_index": self.fold_index,
            "n_splits": self.n_splits,
            "holdout_dataset": self.holdout_dataset,
            "n_train_rows": len(self.train_indices),
            "n_test_rows": len(self.test_indices),
        }


def participant_train_test_split(
    frame: pd.DataFrame,
    *,
    test_size: float = 0.2,
    seed: int,
    participant_columns: Sequence[str] = ("source_dataset", "participant_id"),
) -> SplitDefinition:
    """Create a deterministic participant-grouped train/test split."""

    _require_split_columns(frame, participant_columns)
    if not 0 < test_size < 1:
        raise SplitValidationError("test_size must be between 0 and 1")

    groups = _unique_group_keys(frame, participant_columns)
    if len(groups) < 2:
        raise SplitValidationError("at least two participant groups are required")

    shuffled = _shuffled_groups(groups, seed)
    n_test = max(1, math.ceil(len(shuffled) * test_size))
    n_test = min(n_test, len(shuffled) - 1)
    test_groups = set(shuffled[:n_test])
    split = _split_from_test_groups(
        frame,
        name="participant_train_test_split",
        seed=seed,
        participant_columns=tuple(participant_columns),
        test_groups=test_groups,
    )
    assert_no_participant_overlap(frame, split)
    return split


def participant_group_kfold(
    frame: pd.DataFrame,
    *,
    n_splits: int,
    seed: int,
    participant_columns: Sequence[str] = ("source_dataset", "participant_id"),
) -> Iterator[SplitDefinition]:
    """Yield deterministic participant-grouped folds."""

    _require_split_columns(frame, participant_columns)
    groups = _unique_group_keys(frame, participant_columns)
    if n_splits < 2:
        raise SplitValidationError("n_splits must be at least 2")
    if n_splits > len(groups):
        raise SplitValidationError("n_splits cannot exceed participant groups")

    shuffled = _shuffled_groups(groups, seed)
    folds = np.array_split(shuffled, n_splits)
    for fold_index, fold_groups in enumerate(folds):
        fold_group_tuples = [tuple(group) for group in fold_groups.tolist()]
        split = _split_from_test_groups(
            frame,
            name="participant_group_kfold",
            seed=seed,
            participant_columns=tuple(participant_columns),
            test_groups=set(fold_group_tuples),
            fold_index=fold_index,
            n_splits=n_splits,
        )
        assert_no_participant_overlap(frame, split)
        yield split


def dataset_holdout(
    frame: pd.DataFrame,
    *,
    holdout_dataset: str,
    dataset_column: str = "source_dataset",
) -> SplitDefinition:
    """Hold out every row from one source dataset."""

    if dataset_column not in frame.columns:
        raise SplitValidationError(f"{dataset_column} column is required")
    test_mask = frame[dataset_column] == holdout_dataset
    if not test_mask.any():
        raise SplitValidationError(f"{holdout_dataset} is not present")
    if test_mask.all():
        raise SplitValidationError("dataset holdout would leave no training rows")
    split = SplitDefinition(
        name="dataset_holdout",
        seed=None,
        train_indices=tuple(frame.index[~test_mask].tolist()),
        test_indices=tuple(frame.index[test_mask].tolist()),
        participant_columns=("source_dataset", "participant_id"),
        holdout_dataset=holdout_dataset,
    )
    assert_dataset_holdout_exact(frame, split, dataset_column=dataset_column)
    return split


def assert_no_participant_overlap(frame: pd.DataFrame, split: SplitDefinition) -> None:
    """Fail if any participant group appears in both train and test."""

    train_groups = _group_key_set(frame.loc[list(split.train_indices)], split.participant_columns)
    test_groups = _group_key_set(frame.loc[list(split.test_indices)], split.participant_columns)
    overlap = train_groups.intersection(test_groups)
    if overlap:
        raise SplitValidationError("participant overlap between train and test")


def assert_dataset_holdout_exact(
    frame: pd.DataFrame,
    split: SplitDefinition,
    *,
    dataset_column: str = "source_dataset",
) -> None:
    """Fail if a dataset holdout leaves held-out rows in train or other rows in test."""

    if split.holdout_dataset is None:
        raise SplitValidationError("split has no holdout_dataset")
    train_values = set(frame.loc[list(split.train_indices), dataset_column].unique())
    test_values = set(frame.loc[list(split.test_indices), dataset_column].unique())
    if split.holdout_dataset in train_values:
        raise SplitValidationError("held-out dataset appears in training rows")
    if test_values != {split.holdout_dataset}:
        raise SplitValidationError("test rows are not exactly the held-out dataset")


def _split_from_test_groups(
    frame: pd.DataFrame,
    *,
    name: str,
    seed: int,
    participant_columns: tuple[str, ...],
    test_groups: set[tuple[str, ...]],
    fold_index: int | None = None,
    n_splits: int | None = None,
) -> SplitDefinition:
    groups = _group_keys_for_rows(frame, participant_columns)
    test_mask = groups.isin(test_groups)
    if not test_mask.any() or test_mask.all():
        raise SplitValidationError("split must contain train and test rows")
    return SplitDefinition(
        name=name,
        seed=seed,
        train_indices=tuple(frame.index[~test_mask].tolist()),
        test_indices=tuple(frame.index[test_mask].tolist()),
        participant_columns=participant_columns,
        fold_index=fold_index,
        n_splits=n_splits,
    )


def _require_split_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise SplitValidationError(f"missing split columns: {', '.join(missing)}")


def _unique_group_keys(frame: pd.DataFrame, columns: Sequence[str]) -> list[tuple[str, ...]]:
    return sorted(_group_key_set(frame, columns))


def _shuffled_groups(groups: list[tuple[str, ...]], seed: int) -> list[tuple[str, ...]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups))
    return [groups[int(index)] for index in order]


def _group_key_set(frame: pd.DataFrame, columns: Sequence[str]) -> set[tuple[str, ...]]:
    return set(_group_keys_for_rows(frame, columns).tolist())


def _group_keys_for_rows(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    return frame.loc[:, list(columns)].astype(str).agg(tuple, axis=1)
