"""M0: deterministic general-performance baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from trident_validation.metrics import mean_squared_error_observed
from trident_validation.models.base import BaseValidationModel, ModelScore


class M0ModelError(ValueError):
    """Raised when M0 cannot be fitted or scored."""


@dataclass
class M0GeneralPerformanceModel(BaseValidationModel):
    """One-factor general-performance reconstruction baseline.

    Missing feature cells are left missing in source data. For the internal SVD
    projection only, missing standardised cells are set to zero, equivalent to
    the training mean. Holdout reconstruction error is computed only over
    observed cells.
    """

    feature_columns: Sequence[str]
    model_id: str = "M0_general_performance"
    positive_direction_features: Sequence[str] = (
        "accuracy",
        "mean_response_speed",
        "throughput_proxy",
        "vigilance_engagement",
        "inhibitory_stability",
        "reciprocal_rt",
    )
    negative_direction_features: Sequence[str] = (
        "median_rt_ms",
        "rt_cv",
        "lapse_rate",
        "false_start_rate",
    )
    means_: pd.Series | None = field(default=None, init=False)
    scales_: pd.Series | None = field(default=None, init=False)
    component_: np.ndarray | None = field(default=None, init=False)
    explained_variance_ratio_: float | None = field(default=None, init=False)
    fitted_: bool = field(default=False, init=False)

    def fit(self, frame: pd.DataFrame, y: Any = None, groups: Any = None) -> "M0GeneralPerformanceModel":
        _require_columns(frame, self.feature_columns)
        matrix = _numeric_matrix(frame, self.feature_columns)
        observed_columns = matrix.columns[matrix.notna().any(axis=0)].tolist()
        if not observed_columns:
            raise M0ModelError("at least one observed feature is required")
        if observed_columns != list(self.feature_columns):
            matrix = matrix.loc[:, observed_columns]
            self.feature_columns = tuple(observed_columns)

        means = matrix.mean(axis=0, skipna=True)
        scales = matrix.std(axis=0, skipna=True, ddof=0).replace(0.0, 1.0).fillna(1.0)
        z = ((matrix - means) / scales).fillna(0.0).to_numpy(dtype=float)
        if z.shape[0] < 2:
            raise M0ModelError("at least two training rows are required")

        _, singular_values, vt = np.linalg.svd(z, full_matrices=False)
        component = vt[0].astype(float)
        component = _orient_component(
            component,
            self.feature_columns,
            self.positive_direction_features,
            self.negative_direction_features,
        )
        total_variance = float(np.sum(singular_values**2))
        explained = float(singular_values[0] ** 2 / total_variance) if total_variance > 0 else 0.0

        self.means_ = means
        self.scales_ = scales
        self.component_ = component
        self.explained_variance_ratio_ = explained
        self.fitted_ = True
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        _require_columns(frame, self.feature_columns)
        z = self._standardise(frame).fillna(0.0).to_numpy(dtype=float)
        component = self.component_
        assert component is not None
        score = z @ component
        return pd.DataFrame(
            {f"{self.model_id}_score": score},
            index=frame.index,
        )

    def score_holdout(self, frame: pd.DataFrame, y: Any = None) -> ModelScore:
        self._require_fitted()
        z_frame = self._standardise(frame)
        z = z_frame.to_numpy(dtype=float)
        z_filled = np.nan_to_num(z, nan=0.0)
        component = self.component_
        assert component is not None
        scores = z_filled @ component
        reconstruction = np.outer(scores, component)
        mse = mean_squared_error_observed(z, reconstruction)
        return ModelScore(
            model_id=self.model_id,
            metrics={
                "heldout_reconstruction_mse": mse,
                "heldout_reconstruction_rmse": float(np.sqrt(mse)),
            },
            metadata=self.get_model_metadata(),
        )

    def get_model_metadata(self) -> dict[str, Any]:
        self._require_fitted()
        assert self.component_ is not None
        return {
            "model_id": self.model_id,
            "family": "baseline",
            "feature_columns": list(self.feature_columns),
            "component_loadings": {
                feature: float(value)
                for feature, value in zip(self.feature_columns, self.component_, strict=True)
            },
            "explained_variance_ratio": float(self.explained_variance_ratio_ or 0.0),
            "labels": "neutral_numeric_projection",
        }

    def _standardise(self, frame: pd.DataFrame) -> pd.DataFrame:
        assert self.means_ is not None
        assert self.scales_ is not None
        matrix = _numeric_matrix(frame, self.feature_columns)
        return (matrix - self.means_) / self.scales_

    def _require_fitted(self) -> None:
        if not self.fitted_:
            raise M0ModelError("fit must be called before prediction or scoring")


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise M0ModelError(f"missing feature columns: {', '.join(missing)}")


def _numeric_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    return frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")


def _orient_component(
    component: np.ndarray,
    feature_columns: Sequence[str],
    positive_features: Sequence[str],
    negative_features: Sequence[str],
) -> np.ndarray:
    directional_sum = 0.0
    for index, feature in enumerate(feature_columns):
        if feature in positive_features:
            directional_sum += component[index]
        elif feature in negative_features:
            directional_sum -= component[index]
    if directional_sum < 0:
        return -component
    return component

