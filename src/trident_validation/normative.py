"""Leakage-safe normative trait-state preprocessing interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd


class NormativeModelError(ValueError):
    """Raised when the normative residualiser cannot be used safely."""


class ProspectiveLeakageError(NormativeModelError):
    """Raised when a personal baseline would use current or future sessions."""


class BaseNormativeModel(ABC):
    """Common interface for N0, N1 and later N2 normative models."""

    normative_id: str

    @abstractmethod
    def fit(self, frame: pd.DataFrame) -> "BaseNormativeModel":
        """Fit training-only normative parameters."""

    @abstractmethod
    def transform(self, frame: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        """Apply fitted parameters to held-out rows without refitting."""

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform a training table."""

        return self.fit(frame).transform(frame)

    @abstractmethod
    def get_metadata(self) -> dict[str, Any]:
        """Return serialisable non-sensitive metadata."""


@dataclass
class N0RawNormativeModel(BaseNormativeModel):
    """No-normalisation comparator for future normative model tournaments."""

    feature_columns: Sequence[str]
    normative_id: str = "N0_raw_no_normalisation"
    fitted_: bool = field(default=False, init=False)

    def fit(self, frame: pd.DataFrame) -> "N0RawNormativeModel":
        self._require_columns(frame, self.feature_columns)
        self.fitted_ = True
        return self

    def transform(self, frame: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
        if not self.fitted_:
            raise NormativeModelError("fit must be called before transform")
        self._require_columns(frame, self.feature_columns)
        result = frame.copy()
        for feature in self.feature_columns:
            result[f"{feature}_raw"] = result[feature]
            result[f"{feature}_expected"] = np.nan
            result[f"{feature}_population_z"] = result[feature]
            result[f"{feature}_deviation"] = result[feature]
            result[f"{feature}_uncertainty"] = np.nan
            result[f"{feature}_prior_session_count"] = 0
        return result

    def get_metadata(self) -> dict[str, Any]:
        return {
            "normative_id": self.normative_id,
            "model": "raw_no_normalisation",
            "feature_columns": list(self.feature_columns),
            "prospective_baseline_mode": "none",
        }

    @staticmethod
    def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise NormativeModelError(f"missing columns: {', '.join(missing)}")


@dataclass(frozen=True)
class FeatureNorms:
    """Training-only summary statistics for one feature."""

    global_location: float
    global_scale: float
    residual_scale: float
    context_locations: dict[tuple[str, ...], float]
    source_locations: dict[str, float]
    task_locations: dict[str, float]


@dataclass
class N1SimpleNormativeResidualizer(BaseNormativeModel):
    """Transparent Stage 1 residualiser.

    This first pass estimates source/task/session expectations from training
    data only. Personal baselines during transformation are sequential and use
    completed prior sessions only, never current or future sessions.
    """

    feature_columns: Sequence[str]
    normative_id: str = "N1_simple_median_residualiser"
    context_columns: Sequence[str] = ("source_dataset", "task_id", "practice_or_session_index")
    participant_columns: Sequence[str] = ("source_dataset", "participant_id")
    session_order_column: str = "practice_or_session_index"
    session_id_column: str = "session_id"
    min_prior_sessions: int = 1
    feature_norms_: dict[str, FeatureNorms] = field(default_factory=dict, init=False)
    training_participant_offsets_: dict[str, dict[tuple[str, ...], float]] = field(
        default_factory=dict,
        init=False,
    )
    fitted_: bool = field(default=False, init=False)

    def fit(self, frame: pd.DataFrame) -> "N1SimpleNormativeResidualizer":
        """Fit training-only population and context expectations."""

        self._require_columns(frame, self.feature_columns)
        self._require_columns(frame, self.context_columns)
        self._require_columns(frame, self.participant_columns)

        feature_norms: dict[str, FeatureNorms] = {}
        participant_offsets: dict[str, dict[tuple[str, ...], float]] = {}
        for feature in self.feature_columns:
            values = pd.to_numeric(frame[feature], errors="coerce")
            observed = frame.loc[values.notna()].copy()
            observed_values = values.loc[values.notna()].astype(float)
            if observed_values.empty:
                raise NormativeModelError(f"{feature} has no observed training values")

            global_location = float(observed_values.median())
            global_scale = _safe_scale(observed_values)

            context_locations = {
                _tuple_key(key): float(group[feature].median())
                for key, group in observed.groupby(list(self.context_columns), dropna=False, sort=True)
            }
            source_locations = {
                str(key): float(group[feature].median())
                for key, group in observed.groupby("source_dataset", dropna=False, sort=True)
            }
            task_locations = {
                str(key): float(group[feature].median())
                for key, group in observed.groupby("task_id", dropna=False, sort=True)
            }

            residuals = observed_values - observed.apply(
                lambda row: self._lookup_expected_from_maps(
                    row,
                    context_locations=context_locations,
                    source_locations=source_locations,
                    task_locations=task_locations,
                    global_location=global_location,
                ),
                axis=1,
            )
            residual_scale = _safe_scale(residuals)

            offsets = {}
            observed_with_residuals = observed.assign(_context_residual=residuals.to_numpy())
            for key, group in observed_with_residuals.groupby(
                list(self.participant_columns),
                dropna=False,
                sort=True,
            ):
                offsets[_tuple_key(key)] = float(group["_context_residual"].median())
            participant_offsets[feature] = offsets

            feature_norms[feature] = FeatureNorms(
                global_location=global_location,
                global_scale=global_scale,
                residual_scale=residual_scale,
                context_locations=context_locations,
                source_locations=source_locations,
                task_locations=task_locations,
            )

        self.feature_norms_ = feature_norms
        self.training_participant_offsets_ = participant_offsets
        self.fitted_ = True
        return self

    def transform(
        self,
        frame: pd.DataFrame,
        *,
        use_training_participant_offsets: bool = False,
        personal_baseline_mode: str = "strict_prospective",
    ) -> pd.DataFrame:
        """Append normative output columns to ``frame`` without refitting."""

        if personal_baseline_mode != "strict_prospective":
            raise ProspectiveLeakageError(
                "N1 supports only strict_prospective personal baselines"
            )
        if not self.fitted_:
            raise NormativeModelError("fit must be called before transform")
        self._require_columns(frame, self.feature_columns)
        self._require_columns(frame, self.context_columns)
        self._require_columns(frame, self.participant_columns)
        self._require_columns(frame, (self.session_id_column, self.session_order_column))

        result = frame.copy()
        sort_columns = [
            *self.participant_columns,
            self.session_order_column,
            self.session_id_column,
            "window_start_trial" if "window_start_trial" in frame.columns else self.session_id_column,
        ]
        ordered = result.sort_values(sort_columns, kind="mergesort")

        for feature, norms in self.feature_norms_.items():
            result[f"{feature}_raw"] = result[feature]
            result[f"{feature}_expected"] = np.nan
            result[f"{feature}_population_z"] = np.nan
            result[f"{feature}_deviation"] = np.nan
            result[f"{feature}_uncertainty"] = norms.residual_scale
            result[f"{feature}_prior_session_count"] = 0

            prior_residuals: dict[tuple[str, ...], list[float]] = {}
            completed_session_counts: dict[tuple[str, ...], int] = {}

            for _, session_rows in ordered.groupby(
                [*self.participant_columns, self.session_order_column, self.session_id_column],
                dropna=False,
                sort=False,
            ):
                first_row = session_rows.iloc[0]
                participant_key = _row_tuple(first_row, self.participant_columns)
                completed_prior = prior_residuals.get(participant_key, [])
                prior_count = completed_session_counts.get(participant_key, 0)
                personal_offset = self._personal_offset(
                    feature,
                    participant_key,
                    completed_prior,
                    prior_count,
                    use_training_participant_offsets=use_training_participant_offsets,
                )

                current_session_residuals: list[float] = []
                for index, row in session_rows.iterrows():
                    raw = pd.to_numeric(pd.Series([row[feature]]), errors="coerce").iloc[0]
                    expected_context = self._lookup_expected(row, norms)
                    expected = expected_context + personal_offset
                    result.at[index, f"{feature}_expected"] = expected
                    result.at[index, f"{feature}_prior_session_count"] = prior_count
                    if pd.notna(raw):
                        raw_float = float(raw)
                        result.at[index, f"{feature}_population_z"] = (
                            raw_float - norms.global_location
                        ) / norms.global_scale
                        result.at[index, f"{feature}_deviation"] = raw_float - expected
                        current_session_residuals.append(raw_float - expected_context)

                if current_session_residuals:
                    prior_residuals.setdefault(participant_key, []).extend(current_session_residuals)
                    completed_session_counts[participant_key] = prior_count + 1

        return result

    def transform_with_prior_observations(
        self,
        frame: pd.DataFrame,
        prior_frame: pd.DataFrame,
        *,
        use_training_participant_offsets: bool = False,
    ) -> pd.DataFrame:
        """Transform target rows using explicit prior observations only.

        This helper is intentionally strict. For every target participant,
        every supplied prior observation must precede the earliest target
        session in the batch.
        """

        if not self.fitted_:
            raise NormativeModelError("fit must be called before transform")
        self._assert_external_prior_is_strictly_past(frame, prior_frame)
        marker = "__trident_validation_is_target"
        order = "__trident_validation_target_order"
        if marker in frame.columns or marker in prior_frame.columns:
            raise NormativeModelError(f"{marker} is reserved")
        target = frame.copy()
        target[marker] = True
        target[order] = range(len(target))
        prior = prior_frame.copy()
        prior[marker] = False
        prior[order] = -1
        combined = pd.concat([prior, target], ignore_index=True, sort=False)
        transformed = self.transform(
            combined,
            use_training_participant_offsets=use_training_participant_offsets,
        )
        target_transformed = transformed.loc[transformed[marker]].copy()
        target_transformed = target_transformed.sort_values(order, kind="mergesort")
        target_transformed.index = frame.index
        return target_transformed.drop(columns=[marker, order])

    def get_metadata(self) -> dict[str, Any]:
        """Return non-sensitive fitted metadata."""

        if not self.fitted_:
            raise NormativeModelError("model is not fitted")
        return {
            "normative_id": self.normative_id,
            "model": "simple_training_median_context_residualizer",
            "feature_columns": list(self.feature_columns),
            "context_columns": list(self.context_columns),
            "participant_columns": list(self.participant_columns),
            "session_order_column": self.session_order_column,
            "session_id_column": self.session_id_column,
            "min_prior_sessions": self.min_prior_sessions,
            "prospective_baseline_mode": "strict_prospective",
        }

    def _lookup_expected(self, row: pd.Series, norms: FeatureNorms) -> float:
        return self._lookup_expected_from_maps(
            row,
            context_locations=norms.context_locations,
            source_locations=norms.source_locations,
            task_locations=norms.task_locations,
            global_location=norms.global_location,
        )

    def _lookup_expected_from_maps(
        self,
        row: pd.Series,
        *,
        context_locations: dict[tuple[str, ...], float],
        source_locations: dict[str, float],
        task_locations: dict[str, float],
        global_location: float,
    ) -> float:
        context_key = _row_tuple(row, self.context_columns)
        if context_key in context_locations:
            return context_locations[context_key]
        source_key = str(row["source_dataset"])
        if source_key in source_locations:
            return source_locations[source_key]
        task_key = str(row["task_id"])
        if task_key in task_locations:
            return task_locations[task_key]
        return global_location

    def _personal_offset(
        self,
        feature: str,
        participant_key: tuple[str, ...],
        completed_prior: list[float],
        prior_session_count: int,
        *,
        use_training_participant_offsets: bool,
    ) -> float:
        if prior_session_count >= self.min_prior_sessions and completed_prior:
            return float(np.median(completed_prior))
        if use_training_participant_offsets:
            return self.training_participant_offsets_.get(feature, {}).get(participant_key, 0.0)
        return 0.0

    @staticmethod
    def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise NormativeModelError(f"missing columns: {', '.join(missing)}")

    def _assert_external_prior_is_strictly_past(
        self,
        frame: pd.DataFrame,
        prior_frame: pd.DataFrame,
    ) -> None:
        self._require_columns(frame, (*self.participant_columns, self.session_order_column))
        self._require_columns(prior_frame, (*self.participant_columns, self.session_order_column))
        target = frame.loc[:, [*self.participant_columns, self.session_order_column]].copy()
        prior = prior_frame.loc[:, [*self.participant_columns, self.session_order_column]].copy()
        for column in self.participant_columns:
            target[column] = target[column].astype(str)
            prior[column] = prior[column].astype(str)
        target[self.session_order_column] = pd.to_numeric(
            target[self.session_order_column],
            errors="coerce",
        )
        prior[self.session_order_column] = pd.to_numeric(
            prior[self.session_order_column],
            errors="coerce",
        )
        target_orders = target.groupby(
            list(self.participant_columns),
            dropna=False,
            sort=True,
        )[self.session_order_column].min()
        if target_orders.isna().any() or prior[self.session_order_column].isna().any():
            raise ProspectiveLeakageError("session order must be numeric for prospective checks")
        for key, group in prior.groupby(list(self.participant_columns), dropna=False, sort=True):
            tuple_key = _tuple_key(key)
            if tuple_key not in target_orders.index:
                continue
            earliest_target = float(target_orders.loc[tuple_key])
            if (group[self.session_order_column] >= earliest_target).any():
                raise ProspectiveLeakageError(
                    "external prior observations include current or future sessions"
                )


def _safe_scale(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
    if clean.empty:
        return 1.0
    q75 = clean.quantile(0.75)
    q25 = clean.quantile(0.25)
    iqr_scale = float((q75 - q25) / 1.349) if q75 > q25 else 0.0
    std_scale = float(clean.std(ddof=0))
    scale = iqr_scale or std_scale or 1.0
    return max(scale, 1e-12)


def _tuple_key(value: Any) -> tuple[str, ...]:
    if isinstance(value, tuple):
        return tuple(str(item) for item in value)
    return (str(value),)


def _row_tuple(row: pd.Series, columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(row[column]) for column in columns)


NormativeResidualizer = N1SimpleNormativeResidualizer
