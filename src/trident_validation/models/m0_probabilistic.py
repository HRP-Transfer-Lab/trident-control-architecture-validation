"""Formal M0 probabilistic one-factor baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from trident_validation.models.adapters import TrainingFeatureAdapter
from trident_validation.models.base import FeatureRequirements, ModelScore, TournamentModel


class M0ProbabilisticError(ValueError):
    """Raised when probabilistic M0 cannot be fitted or scored."""


@dataclass
class M0ProbabilisticPCAModel(TournamentModel):
    """Probabilistic PCA baseline for formal tournament scoring."""

    feature_columns: Sequence[str]
    random_state: int
    model_id: str = "M0_probabilistic_general_performance"
    latent_dimensions: int = 1
    min_observed_features_per_window: int = 1
    jitter: float = 1e-8
    adapter_: TrainingFeatureAdapter | None = field(default=None, init=False)
    loadings_: np.ndarray | None = field(default=None, init=False)
    noise_variance_: float | None = field(default=None, init=False)
    covariance_: np.ndarray | None = field(default=None, init=False)
    eigenvalues_: np.ndarray | None = field(default=None, init=False)
    fitted_: bool = field(default=False, init=False)

    def feature_requirements(self) -> FeatureRequirements:
        return FeatureRequirements(
            feature_columns=tuple(self.feature_columns),
            min_observed_features_per_window=self.min_observed_features_per_window,
            allows_structural_missingness=True,
        )

    def fit(self, frame: pd.DataFrame, y: Any = None, groups: Any = None) -> "M0ProbabilisticPCAModel":
        """Fit PPCA parameters from training participants only."""

        adapter = TrainingFeatureAdapter(
            feature_columns=self.feature_columns,
            min_observed_features_per_window=self.min_observed_features_per_window,
        ).fit(frame)
        prepared = adapter.transform(frame)
        x = prepared.values
        if x.shape[0] < 2:
            raise M0ProbabilisticError("at least two training rows are required")
        if x.shape[1] < 1:
            raise M0ProbabilisticError("at least one feature is required")
        if not 1 <= self.latent_dimensions <= x.shape[1]:
            raise M0ProbabilisticError("latent_dimensions must be between 1 and feature count")

        covariance = (x.T @ x) / x.shape[0]
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        if x.shape[1] == 1 or self.latent_dimensions >= x.shape[1]:
            noise_variance = max(float(eigenvalues[0]) * 0.05, self.jitter)
        else:
            trailing = eigenvalues[self.latent_dimensions :]
            noise_variance = float(np.mean(trailing)) if trailing.size else self.jitter
            noise_variance = max(noise_variance, self.jitter)

        loading_scales = np.sqrt(
            np.maximum(eigenvalues[: self.latent_dimensions] - noise_variance, self.jitter)
        )
        loadings = eigenvectors[:, : self.latent_dimensions] * loading_scales
        loadings = _orient_loadings(loadings, tuple(adapter.feature_columns))
        model_covariance = loadings @ loadings.T + noise_variance * np.eye(x.shape[1])
        model_covariance = model_covariance + self.jitter * np.eye(x.shape[1])

        self.adapter_ = adapter
        self.loadings_ = loadings
        self.noise_variance_ = noise_variance
        self.covariance_ = model_covariance
        self.eigenvalues_ = eigenvalues
        self.feature_columns = tuple(adapter.feature_columns)
        self.fitted_ = True
        return self

    def score_samples(self, frame: pd.DataFrame) -> pd.Series:
        """Return marginal Gaussian log-density per held-out window."""

        self._require_fitted()
        assert self.adapter_ is not None
        assert self.covariance_ is not None
        standardised = self.adapter_.observed_standardised(frame)
        values = standardised.to_numpy(dtype=float)
        observed_mask = standardised.notna().to_numpy(dtype=bool)
        scores = [
            _marginal_logpdf(row, mask, self.covariance_, jitter=self.jitter)
            for row, mask in zip(values, observed_mask, strict=True)
        ]
        return pd.Series(scores, index=frame.index, name=f"{self.model_id}_log_density")

    def predict_representation(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return the posterior one-factor mean for held-out windows."""

        self._require_fitted()
        assert self.adapter_ is not None
        assert self.loadings_ is not None
        assert self.noise_variance_ is not None
        standardised = self.adapter_.observed_standardised(frame)
        values = standardised.to_numpy(dtype=float)
        observed_mask = standardised.notna().to_numpy(dtype=bool)
        scores = [
            _posterior_factor_mean(row, mask, self.loadings_, self.noise_variance_)
            for row, mask in zip(values, observed_mask, strict=True)
        ]
        columns = [
            f"{self.model_id}_factor_{index + 1}"
            for index in range(self.latent_dimensions)
        ]
        return pd.DataFrame(
            scores,
            columns=columns,
            index=frame.index,
        )

    def score_holdout(self, frame: pd.DataFrame, y: Any = None) -> ModelScore:
        sample_scores = self.score_samples(frame)
        valid_scores = sample_scores.dropna()
        observed_count = self._observed_feature_count(frame)
        total = float(valid_scores.sum()) if not valid_scores.empty else float("nan")
        mean_per_window = float(valid_scores.mean()) if not valid_scores.empty else float("nan")
        mean_per_feature = (
            float(total / observed_count) if observed_count > 0 and np.isfinite(total) else float("nan")
        )
        diagnostics = self._diagnostics(frame)
        metrics = {
            "heldout_log_likelihood_total": total,
            "heldout_log_density_mean_per_window": mean_per_window,
            "heldout_log_density_mean_per_observed_feature": mean_per_feature,
            "n_valid_windows": float(valid_scores.shape[0]),
            "n_observed_feature_values": float(observed_count),
        }
        return ModelScore(
            model_id=self.model_id,
            metrics=metrics,
            metadata=self.get_model_metadata(),
            primary_metric="heldout_log_density_mean_per_window",
            primary_value=mean_per_window,
            diagnostics=diagnostics,
            secondary_metrics={
                "heldout_log_density_mean_per_observed_feature": mean_per_feature,
            },
        )

    def get_model_metadata(self) -> dict[str, Any]:
        self._require_fitted()
        assert self.adapter_ is not None
        assert self.loadings_ is not None
        assert self.noise_variance_ is not None
        assert self.eigenvalues_ is not None
        metadata = {
            "model_id": self.model_id,
            "family": "baseline",
            "model_form": "probabilistic_pca",
            "random_state": int(self.random_state),
            "randomness": "closed_form_deterministic_ppca_with_explicit_seed_recorded",
            "latent_dimensions": int(self.latent_dimensions),
            "feature_requirements": {
                "feature_columns": list(self.feature_columns),
                "min_observed_features_per_window": int(self.min_observed_features_per_window),
                "allows_structural_missingness": True,
            },
            "primary_metric": "heldout_log_density_mean_per_window",
            "diagnostic_metrics": [
                "heldout_log_density_mean_per_observed_feature",
                "heldout_reconstruction_mse",
                "ppca_noise_variance",
            ],
            "latent_labels": "neutral_numeric_factor",
            "noise_variance": float(self.noise_variance_),
            "eigenvalues": [float(value) for value in self.eigenvalues_],
            "imputation": self.adapter_.get_metadata(),
        }
        if self.latent_dimensions == 1:
            metadata["model_form"] = "one_factor_probabilistic_pca"
            metadata["loadings"] = {
                feature: float(value)
                for feature, value in zip(self.feature_columns, self.loadings_[:, 0], strict=True)
            }
        else:
            metadata["loadings"] = {
                f"factor_{factor_index + 1}": {
                    feature: float(value)
                    for feature, value in zip(
                        self.feature_columns,
                        self.loadings_[:, factor_index],
                        strict=True,
                    )
                }
                for factor_index in range(self.latent_dimensions)
            }
        return metadata

    def _observed_feature_count(self, frame: pd.DataFrame) -> int:
        assert self.adapter_ is not None
        return int(self.adapter_.observed_standardised(frame).notna().to_numpy(dtype=bool).sum())

    def _diagnostics(self, frame: pd.DataFrame) -> dict[str, float]:
        assert self.adapter_ is not None
        assert self.loadings_ is not None
        standardised = self.adapter_.observed_standardised(frame)
        values = standardised.to_numpy(dtype=float)
        observed_mask = standardised.notna().to_numpy(dtype=bool)
        factors = self.predict_representation(frame).to_numpy(dtype=float)
        reconstruction = factors @ self.loadings_.T
        residual = values - reconstruction
        if observed_mask.any():
            mse = float(np.mean((residual[observed_mask]) ** 2))
        else:
            mse = float("nan")
        return {
            "heldout_reconstruction_mse": mse,
            "ppca_noise_variance": float(self.noise_variance_ or float("nan")),
        }

    def _require_fitted(self) -> None:
        if not self.fitted_:
            raise M0ProbabilisticError("fit must be called before prediction or scoring")


def _marginal_logpdf(
    row: np.ndarray,
    observed_mask: np.ndarray,
    covariance: np.ndarray,
    *,
    jitter: float,
) -> float:
    if not observed_mask.any():
        return float("nan")
    observed_values = row[observed_mask]
    observed_covariance = covariance[np.ix_(observed_mask, observed_mask)]
    observed_covariance = observed_covariance + jitter * np.eye(observed_covariance.shape[0])
    sign, logdet = np.linalg.slogdet(observed_covariance)
    if sign <= 0:
        observed_covariance = observed_covariance + 100 * jitter * np.eye(observed_covariance.shape[0])
        sign, logdet = np.linalg.slogdet(observed_covariance)
        if sign <= 0:
            raise M0ProbabilisticError("non-positive covariance determinant")
    solved = np.linalg.solve(observed_covariance, observed_values)
    dimension = observed_values.shape[0]
    return float(
        -0.5
        * (
            dimension * np.log(2 * np.pi)
            + logdet
            + float(observed_values.T @ solved)
        )
    )


def _posterior_factor_mean(
    row: np.ndarray,
    observed_mask: np.ndarray,
    loadings: np.ndarray,
    noise_variance: float,
) -> list[float]:
    if not observed_mask.any():
        return [float("nan")] * loadings.shape[1]
    observed_values = row[observed_mask]
    observed_loadings = loadings[observed_mask]
    precision = np.eye(loadings.shape[1]) + (observed_loadings.T @ observed_loadings) / noise_variance
    numerator = (observed_loadings.T @ observed_values) / noise_variance
    posterior_mean = np.linalg.solve(precision, numerator)
    return [float(value) for value in posterior_mean]


def _orient_loadings(loadings: np.ndarray, feature_columns: tuple[str, ...]) -> np.ndarray:
    oriented = loadings.copy()
    for factor_index in range(oriented.shape[1]):
        oriented[:, factor_index] = _orient_loading(oriented[:, factor_index], feature_columns)
    return oriented


def _orient_loading(loadings: np.ndarray, feature_columns: tuple[str, ...]) -> np.ndarray:
    positive = {
        "accuracy",
        "mean_response_speed",
        "throughput_proxy",
        "vigilance_engagement",
        "inhibitory_stability",
        "reciprocal_rt",
    }
    negative = {"median_rt_ms", "rt_cv", "lapse_rate", "false_start_rate"}
    directional_sum = 0.0
    for index, feature in enumerate(feature_columns):
        if feature in positive:
            directional_sum += loadings[index]
        elif feature in negative:
            directional_sum -= loadings[index]
    if directional_sum < 0:
        return -loadings
    return loadings
