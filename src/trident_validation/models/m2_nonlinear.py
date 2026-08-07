"""M2: nonlinear vigilance/readiness baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from trident_validation.models._density import diagonal_gaussian_logpdf, logsumexp
from trident_validation.models.adapters import TrainingFeatureAdapter
from trident_validation.models.base import FeatureRequirements, ModelScore, TournamentModel
from trident_validation.models.m0_probabilistic import _orient_loadings


class M2NonlinearError(ValueError):
    """Raised when M2 cannot be fitted or scored."""


@dataclass
class M2NonlinearVigilanceModel(TournamentModel):
    """One-dimensional latent baseline with quadratic feature curves."""

    feature_columns: Sequence[str]
    random_state: int
    model_id: str = "M2_nonlinear_vigilance"
    min_observed_features_per_window: int = 1
    quadrature_points: int = 31
    jitter: float = 1e-8
    adapter_: TrainingFeatureAdapter | None = field(default=None, init=False)
    projection_: np.ndarray | None = field(default=None, init=False)
    coefficients_: np.ndarray | None = field(default=None, init=False)
    residual_variance_: np.ndarray | None = field(default=None, init=False)
    fitted_: bool = field(default=False, init=False)

    def feature_requirements(self) -> FeatureRequirements:
        return FeatureRequirements(
            feature_columns=tuple(self.feature_columns),
            min_observed_features_per_window=self.min_observed_features_per_window,
            allows_structural_missingness=True,
        )

    def fit(self, frame: pd.DataFrame, y: Any = None, groups: Any = None) -> "M2NonlinearVigilanceModel":
        adapter = TrainingFeatureAdapter(
            feature_columns=self.feature_columns,
            min_observed_features_per_window=self.min_observed_features_per_window,
        ).fit(frame)
        prepared = adapter.transform(frame)
        x = prepared.values
        if x.shape[0] < 3:
            raise M2NonlinearError("at least three training rows are required")
        covariance = (x.T @ x) / x.shape[0]
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        projection = eigenvectors[:, np.argsort(eigenvalues)[::-1][:1]]
        projection = _orient_loadings(projection, tuple(adapter.feature_columns))[:, 0]
        z = x @ projection
        z_scale = float(np.std(z, ddof=0)) or 1.0
        z = (z - float(np.mean(z))) / z_scale
        design = np.column_stack([np.ones_like(z), z, z**2 - 1.0])
        coefficients = np.linalg.pinv(design) @ x
        fitted = design @ coefficients
        residuals = x - fitted
        residual_variance = np.maximum(np.var(residuals, axis=0), self.jitter)

        self.adapter_ = adapter
        self.projection_ = projection
        self.coefficients_ = coefficients
        self.residual_variance_ = residual_variance
        self.feature_columns = tuple(adapter.feature_columns)
        self.fitted_ = True
        return self

    def score_samples(self, frame: pd.DataFrame) -> pd.Series:
        self._require_fitted()
        assert self.adapter_ is not None
        assert self.coefficients_ is not None
        assert self.residual_variance_ is not None
        standardised = self.adapter_.observed_standardised(frame)
        values = standardised.to_numpy(dtype=float)
        observed_mask = standardised.notna().to_numpy(dtype=bool)
        nodes, log_weights = self._standard_normal_quadrature()
        scores = [
            self._row_log_density(row, mask, nodes, log_weights)
            for row, mask in zip(values, observed_mask, strict=True)
        ]
        return pd.Series(scores, index=frame.index, name=f"{self.model_id}_log_density")

    def predict_representation(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        assert self.adapter_ is not None
        standardised = self.adapter_.observed_standardised(frame)
        values = standardised.to_numpy(dtype=float)
        observed_mask = standardised.notna().to_numpy(dtype=bool)
        nodes, log_weights = self._standard_normal_quadrature()
        posterior_means = [
            self._posterior_moments(row, mask, nodes, log_weights)
            for row, mask in zip(values, observed_mask, strict=True)
        ]
        return pd.DataFrame(
            posterior_means,
            columns=[
                f"{self.model_id}_latent_readiness",
                f"{self.model_id}_latent_readiness_squared",
            ],
            index=frame.index,
        )

    def score_holdout(self, frame: pd.DataFrame, y: Any = None) -> ModelScore:
        base = TournamentModel.score_holdout(self, frame, y=y)
        metadata = self.get_model_metadata()
        return ModelScore(
            model_id=base.model_id,
            metrics=base.metrics,
            metadata=metadata,
            primary_metric=base.primary_metric,
            primary_value=base.primary_value,
            diagnostics={"mean_residual_variance": float(np.mean(self.residual_variance_))},
        )

    def get_model_metadata(self) -> dict[str, Any]:
        self._require_fitted()
        assert self.adapter_ is not None
        assert self.coefficients_ is not None
        assert self.residual_variance_ is not None
        return {
            "model_id": self.model_id,
            "family": "nonlinear_baseline",
            "model_form": "quadratic_one_dimensional_latent_curve",
            "random_state": int(self.random_state),
            "randomness": "closed_form_deterministic_with_explicit_seed_recorded",
            "primary_metric": "heldout_log_density_mean_per_window",
            "latent_labels": "neutral_numeric_readiness_curve",
            "quadrature_points": int(self.quadrature_points),
            "feature_requirements": {
                "feature_columns": list(self.feature_columns),
                "min_observed_features_per_window": int(self.min_observed_features_per_window),
                "allows_structural_missingness": True,
            },
            "coefficients": {
                feature: {
                    "intercept": float(self.coefficients_[0, index]),
                    "linear": float(self.coefficients_[1, index]),
                    "quadratic": float(self.coefficients_[2, index]),
                    "residual_variance": float(self.residual_variance_[index]),
                }
                for index, feature in enumerate(self.feature_columns)
            },
            "imputation": self.adapter_.get_metadata(),
        }

    def _row_log_density(
        self,
        row: np.ndarray,
        observed_mask: np.ndarray,
        nodes: np.ndarray,
        log_weights: np.ndarray,
    ) -> float:
        if not observed_mask.any():
            return float("nan")
        assert self.coefficients_ is not None
        assert self.residual_variance_ is not None
        component_logs = []
        for node, log_weight in zip(nodes, log_weights, strict=True):
            basis = np.array([1.0, node, node**2 - 1.0])
            mean = basis @ self.coefficients_
            component_logs.append(
                log_weight
                + diagonal_gaussian_logpdf(row, mean, self.residual_variance_, observed_mask)
            )
        return logsumexp(np.array(component_logs, dtype=float))

    def _posterior_moments(
        self,
        row: np.ndarray,
        observed_mask: np.ndarray,
        nodes: np.ndarray,
        log_weights: np.ndarray,
    ) -> list[float]:
        if not observed_mask.any():
            return [float("nan"), float("nan")]
        log_density_by_node = np.array(
            [
                self._row_log_density(row, observed_mask, np.array([node]), np.array([log_weight]))
                for node, log_weight in zip(nodes, log_weights, strict=True)
            ],
            dtype=float,
        )
        normaliser = logsumexp(log_density_by_node)
        weights = np.exp(log_density_by_node - normaliser)
        return [
            float(np.sum(weights * nodes)),
            float(np.sum(weights * (nodes**2))),
        ]

    def _standard_normal_quadrature(self) -> tuple[np.ndarray, np.ndarray]:
        nodes, weights = np.polynomial.hermite.hermgauss(self.quadrature_points)
        standard_nodes = np.sqrt(2.0) * nodes
        standard_weights = weights / np.sqrt(np.pi)
        return standard_nodes.astype(float), np.log(standard_weights.astype(float))

    def _require_fitted(self) -> None:
        if not self.fitted_:
            raise M2NonlinearError("fit must be called before prediction or scoring")

