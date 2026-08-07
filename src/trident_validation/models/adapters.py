"""Training-fitted feature adapters for tournament models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from trident_validation.schema import STRUCTURAL_FEATURES_BY_FLAG


class FeatureAdapterError(ValueError):
    """Raised when model-specific feature preparation is unsafe."""


@dataclass(frozen=True)
class FeatureMatrix:
    """Prepared model matrix plus observed-cell mask."""

    values: np.ndarray
    observed_mask: np.ndarray
    feature_columns: tuple[str, ...]


@dataclass
class TrainingFeatureAdapter:
    """Model-specific imputation and scaling fitted on training rows only.

    Canonical data remain untouched. Missing feature cells are imputed only in
    the returned model matrix, using statistics estimated in ``fit``.
    """

    feature_columns: Sequence[str]
    participant_columns: Sequence[str] = ("source_dataset", "participant_id")
    imputation_strategy: str = "training_feature_mean"
    scaling_strategy: str = "training_mean_std"
    min_observed_features_per_window: int = 1
    means_: pd.Series | None = field(default=None, init=False)
    scales_: pd.Series | None = field(default=None, init=False)
    metadata_: dict[str, Any] = field(default_factory=dict, init=False)
    fitted_: bool = field(default=False, init=False)

    def fit(self, frame: pd.DataFrame) -> "TrainingFeatureAdapter":
        """Fit imputation and scaling statistics from training rows only."""

        _require_columns(frame, self.feature_columns)
        _require_columns(frame, self.participant_columns)
        matrix = _numeric_frame(frame, self.feature_columns)
        observed_by_feature = matrix.notna().sum(axis=0)
        if (observed_by_feature == 0).any():
            missing = observed_by_feature[observed_by_feature == 0].index.tolist()
            raise FeatureAdapterError(
                "cannot fit adapter because feature(s) have no observed training values: "
                + ", ".join(missing)
            )

        means = matrix.mean(axis=0, skipna=True)
        scales = matrix.std(axis=0, skipna=True, ddof=0).replace(0.0, 1.0).fillna(1.0)
        structural_counts = _structural_missing_counts(frame, self.feature_columns)
        missing_counts = matrix.isna().sum(axis=0).astype(int).to_dict()
        technical_counts = {
            feature: int(missing_counts[feature] - structural_counts.get(feature, 0))
            for feature in self.feature_columns
        }
        participant_count = (
            frame.loc[:, list(self.participant_columns)].astype(str).drop_duplicates().shape[0]
        )

        self.means_ = means
        self.scales_ = scales
        self.metadata_ = {
            "adapter": "TrainingFeatureAdapter",
            "fitted_on_n_rows": int(frame.shape[0]),
            "fitted_on_n_participant_groups": int(participant_count),
            "feature_columns": list(self.feature_columns),
            "imputation_strategy": self.imputation_strategy,
            "scaling_strategy": self.scaling_strategy,
            "min_observed_features_per_window": int(self.min_observed_features_per_window),
            "training_feature_means": {key: float(value) for key, value in means.items()},
            "training_feature_scales": {key: float(value) for key, value in scales.items()},
            "training_observed_counts": {
                key: int(value) for key, value in observed_by_feature.items()
            },
            "training_missing_counts": missing_counts,
            "training_structural_missing_counts": structural_counts,
            "training_technical_missing_counts": technical_counts,
        }
        self.fitted_ = True
        return self

    def transform(self, frame: pd.DataFrame) -> FeatureMatrix:
        """Return an imputed standardised model matrix without mutating input."""

        self._require_fitted()
        _require_columns(frame, self.feature_columns)
        assert self.means_ is not None
        assert self.scales_ is not None
        matrix = _numeric_frame(frame, self.feature_columns)
        observed_mask = matrix.notna().to_numpy(dtype=bool)
        observed_per_row = observed_mask.sum(axis=1)
        if (observed_per_row < self.min_observed_features_per_window).any():
            raise FeatureAdapterError(
                "one or more rows have fewer observed features than required"
            )
        standardised = (matrix - self.means_) / self.scales_
        imputed = standardised.fillna(0.0).to_numpy(dtype=float)
        return FeatureMatrix(
            values=imputed,
            observed_mask=observed_mask,
            feature_columns=tuple(self.feature_columns),
        )

    def observed_standardised(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return standardised features while preserving NaNs."""

        self._require_fitted()
        _require_columns(frame, self.feature_columns)
        assert self.means_ is not None
        assert self.scales_ is not None
        matrix = _numeric_frame(frame, self.feature_columns)
        return (matrix - self.means_) / self.scales_

    def get_metadata(self) -> dict[str, Any]:
        """Return serialisable adapter metadata for manifests."""

        self._require_fitted()
        return dict(self.metadata_)

    def _require_fitted(self) -> None:
        if not self.fitted_:
            raise FeatureAdapterError("fit must be called before transform")


def _numeric_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    return frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise FeatureAdapterError(f"missing feature columns: {', '.join(missing)}")


def _structural_missing_counts(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> dict[str, int]:
    feature_to_flag = {
        feature: flag
        for flag, features in STRUCTURAL_FEATURES_BY_FLAG.items()
        for feature in features
    }
    counts: dict[str, int] = {}
    for feature in feature_columns:
        flag = feature_to_flag.get(feature)
        if flag is None or flag not in frame.columns:
            counts[feature] = 0
            continue
        structural_mask = ~frame[flag].astype(bool)
        if feature in frame.columns:
            counts[feature] = int(frame.loc[structural_mask, feature].isna().sum())
        else:
            counts[feature] = int(structural_mask.sum())
    return counts

