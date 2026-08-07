"""Common statistical interface for tournament candidates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence

import pandas as pd


@dataclass(frozen=True)
class ModelScore:
    """Holdout scoring output shared by model families."""

    model_id: str
    metrics: dict[str, float]
    metadata: dict[str, Any]
    primary_metric: str | None = None
    primary_value: float | None = None
    diagnostics: dict[str, float] | None = None
    secondary_metrics: dict[str, float] | None = None


@dataclass(frozen=True)
class FeatureRequirements:
    """Canonical feature-vector requirements declared before fitting."""

    feature_columns: tuple[str, ...]
    min_observed_features_per_window: int = 1
    allows_structural_missingness: bool = True


class BaseValidationModel(ABC):
    """Legacy development-diagnostic interface.

    Milestone 1 created this light interface for deterministic diagnostics.
    Formal tournament models should implement :class:`TournamentModel`.
    """

    model_id: str

    @abstractmethod
    def fit(self, frame: pd.DataFrame, y: Any = None, groups: Any = None) -> "BaseValidationModel":
        """Fit the model using training rows only."""

    @abstractmethod
    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Predict or project held-out rows without refitting."""

    def predict_proba(self, frame: pd.DataFrame) -> pd.DataFrame | None:
        """Return probabilities where meaningful."""

        return None

    @abstractmethod
    def score_holdout(self, frame: pd.DataFrame, y: Any = None) -> ModelScore:
        """Score held-out rows using the fitted model."""

    @abstractmethod
    def get_model_metadata(self) -> dict[str, Any]:
        """Return serialisable, label-neutral model metadata."""


class TournamentModel(ABC):
    """Formal model-scoring contract for M0-M5 tournament candidates.

    The universal primary comparison surface is held-out predictive
    log-density on untouched participant or dataset holdouts. Model-specific
    diagnostics such as BIC, AIC, reconstruction error, ARI or component entropy
    may be reported, but are not the universal primary score.
    """

    model_id: str
    random_state: int

    @abstractmethod
    def feature_requirements(self) -> FeatureRequirements:
        """Declare required canonical features before fitting."""

    @abstractmethod
    def fit(self, frame: pd.DataFrame, y: Any = None, groups: Any = None) -> "TournamentModel":
        """Fit every transformation and parameter using training rows only."""

    @abstractmethod
    def score_samples(self, frame: pd.DataFrame) -> pd.Series:
        """Return held-out log-density per valid canonical window."""

    @abstractmethod
    def predict_representation(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Project held-out rows into a neutral learned representation."""

    def predict_proba(self, frame: pd.DataFrame) -> pd.DataFrame | None:
        """Return class/state probabilities where a model defines them."""

        return None

    def score_holdout(self, frame: pd.DataFrame, y: Any = None) -> ModelScore:
        """Score a holdout using the universal predictive-density contract."""

        sample_scores = self.score_samples(frame)
        valid_scores = sample_scores.dropna()
        total = float(valid_scores.sum()) if not valid_scores.empty else float("nan")
        mean_per_window = float(valid_scores.mean()) if not valid_scores.empty else float("nan")
        metrics = {
            "heldout_log_likelihood_total": total,
            "heldout_log_density_mean_per_window": mean_per_window,
            "n_valid_windows": float(valid_scores.shape[0]),
        }
        return ModelScore(
            model_id=self.model_id,
            metrics=metrics,
            metadata=self.get_model_metadata(),
            primary_metric="heldout_log_density_mean_per_window",
            primary_value=mean_per_window,
        )

    @abstractmethod
    def get_model_metadata(self) -> dict[str, Any]:
        """Return serialisable, label-neutral model metadata."""
