"""Static diagonal Gaussian mixture baselines for M3 and M4."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from trident_validation.models._density import logsumexp, mixture_logpdf
from trident_validation.models.adapters import TrainingFeatureAdapter
from trident_validation.models.base import FeatureRequirements, ModelScore, TournamentModel


class MixtureModelError(ValueError):
    """Raised when a static mixture model cannot be fitted or scored."""


@dataclass
class StaticGaussianMixtureModel(TournamentModel):
    """Diagonal Gaussian mixture with neutral component labels."""

    feature_columns: Sequence[str]
    random_state: int
    n_components: int
    model_id: str
    family: str = "static_mixture"
    min_observed_features_per_window: int = 1
    max_iter: int = 100
    tol: float = 1e-6
    variance_floor: float = 1e-6
    adapter_: TrainingFeatureAdapter | None = field(default=None, init=False)
    weights_: np.ndarray | None = field(default=None, init=False)
    means_: np.ndarray | None = field(default=None, init=False)
    variances_: np.ndarray | None = field(default=None, init=False)
    train_log_likelihood_: float | None = field(default=None, init=False)
    n_iter_: int | None = field(default=None, init=False)
    fitted_: bool = field(default=False, init=False)

    def feature_requirements(self) -> FeatureRequirements:
        return FeatureRequirements(
            feature_columns=tuple(self.feature_columns),
            min_observed_features_per_window=self.min_observed_features_per_window,
            allows_structural_missingness=True,
        )

    def fit(self, frame: pd.DataFrame, y: Any = None, groups: Any = None) -> "StaticGaussianMixtureModel":
        if self.n_components < 2:
            raise MixtureModelError("mixture models require at least two components")
        adapter = TrainingFeatureAdapter(
            feature_columns=self.feature_columns,
            min_observed_features_per_window=self.min_observed_features_per_window,
        ).fit(frame)
        prepared = adapter.transform(frame)
        x = prepared.values
        if x.shape[0] <= self.n_components:
            raise MixtureModelError("more training rows than components are required")
        weights, means, variances = _initialise_mixture(
            x,
            n_components=self.n_components,
            random_state=self.random_state,
            variance_floor=self.variance_floor,
        )
        previous_log_likelihood = -np.inf
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            responsibilities, log_likelihood = _expectation(
                x,
                weights,
                means,
                variances,
            )
            component_weight = responsibilities.sum(axis=0) + 1e-12
            weights = component_weight / component_weight.sum()
            means = (responsibilities.T @ x) / component_weight[:, None]
            for component_index in range(self.n_components):
                residual = x - means[component_index]
                variances[component_index] = (
                    responsibilities[:, component_index][:, None] * residual**2
                ).sum(axis=0) / component_weight[component_index]
            variances = np.maximum(variances, self.variance_floor)
            if abs(log_likelihood - previous_log_likelihood) < self.tol:
                break
            previous_log_likelihood = log_likelihood

        self.adapter_ = adapter
        self.weights_ = weights
        self.means_ = means
        self.variances_ = variances
        self.train_log_likelihood_ = float(previous_log_likelihood)
        self.n_iter_ = n_iter
        self.feature_columns = tuple(adapter.feature_columns)
        self.fitted_ = True
        return self

    def score_samples(self, frame: pd.DataFrame) -> pd.Series:
        self._require_fitted()
        assert self.adapter_ is not None
        assert self.weights_ is not None
        assert self.means_ is not None
        assert self.variances_ is not None
        standardised = self.adapter_.observed_standardised(frame)
        values = standardised.to_numpy(dtype=float)
        observed_mask = standardised.notna().to_numpy(dtype=bool)
        scores = [
            mixture_logpdf(row, self.weights_, self.means_, self.variances_, mask)
            for row, mask in zip(values, observed_mask, strict=True)
        ]
        return pd.Series(scores, index=frame.index, name=f"{self.model_id}_log_density")

    def predict_representation(self, frame: pd.DataFrame) -> pd.DataFrame:
        probabilities = self.predict_proba(frame)
        assert probabilities is not None
        return probabilities

    def predict_proba(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        assert self.adapter_ is not None
        assert self.weights_ is not None
        assert self.means_ is not None
        assert self.variances_ is not None
        standardised = self.adapter_.observed_standardised(frame)
        values = standardised.to_numpy(dtype=float)
        observed_mask = standardised.notna().to_numpy(dtype=bool)
        rows = [
            _posterior_probabilities(row, self.weights_, self.means_, self.variances_, mask)
            for row, mask in zip(values, observed_mask, strict=True)
        ]
        return pd.DataFrame(
            rows,
            columns=[f"{self.model_id}_component_{index}" for index in range(self.n_components)],
            index=frame.index,
        )

    def score_holdout(self, frame: pd.DataFrame, y: Any = None) -> ModelScore:
        base = TournamentModel.score_holdout(self, frame, y=y)
        diagnostics = self._diagnostics(frame)
        return ModelScore(
            model_id=base.model_id,
            metrics=base.metrics,
            metadata=self.get_model_metadata(),
            primary_metric=base.primary_metric,
            primary_value=base.primary_value,
            diagnostics=diagnostics,
            secondary_metrics={
                "mean_posterior_entropy": diagnostics["mean_posterior_entropy"],
            },
        )

    def get_model_metadata(self) -> dict[str, Any]:
        self._require_fitted()
        assert self.adapter_ is not None
        assert self.weights_ is not None
        assert self.means_ is not None
        assert self.variances_ is not None
        return {
            "model_id": self.model_id,
            "family": self.family,
            "model_form": "diagonal_gaussian_mixture",
            "random_state": int(self.random_state),
            "randomness": "deterministic_em_initialised_from_explicit_seed",
            "n_components": int(self.n_components),
            "component_labels": [f"component_{index}" for index in range(self.n_components)],
            "primary_metric": "heldout_log_density_mean_per_window",
            "diagnostic_metrics": [
                "bic",
                "aic",
                "mean_posterior_entropy",
                "smallest_component_weight",
            ],
            "feature_requirements": {
                "feature_columns": list(self.feature_columns),
                "min_observed_features_per_window": int(self.min_observed_features_per_window),
                "allows_structural_missingness": True,
            },
            "component_weights": [float(value) for value in self.weights_],
            "component_means": _component_table(self.means_, self.feature_columns),
            "component_variances": _component_table(self.variances_, self.feature_columns),
            "n_iter": int(self.n_iter_ or 0),
            "train_log_likelihood": float(self.train_log_likelihood_ or float("nan")),
            "imputation": self.adapter_.get_metadata(),
        }

    def _diagnostics(self, frame: pd.DataFrame) -> dict[str, float]:
        assert self.weights_ is not None
        probabilities = self.predict_proba(frame).to_numpy(dtype=float)
        entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)), axis=1)
        train_rows = float(self.adapter_.metadata_["fitted_on_n_rows"]) if self.adapter_ else float("nan")
        n_parameters = self._n_parameters()
        train_ll = float(self.train_log_likelihood_ or float("nan"))
        return {
            "mean_posterior_entropy": float(np.mean(entropy)),
            "smallest_component_weight": float(np.min(self.weights_)),
            "bic": float(n_parameters * np.log(train_rows) - 2.0 * train_ll),
            "aic": float(2.0 * n_parameters - 2.0 * train_ll),
        }

    def _n_parameters(self) -> int:
        feature_count = len(self.feature_columns)
        return (self.n_components - 1) + self.n_components * feature_count * 2

    def _require_fitted(self) -> None:
        if not self.fitted_:
            raise MixtureModelError("fit must be called before prediction or scoring")


@dataclass
class M3ThreeProfileMixtureModel(StaticGaussianMixtureModel):
    """Three-component neutral static mixture baseline."""

    feature_columns: Sequence[str]
    random_state: int
    n_components: int = 3
    model_id: str = "M3_three_profile_mixture"


@dataclass
class M4FourProfileMixtureModel(StaticGaussianMixtureModel):
    """Four-component neutral static mixture baseline."""

    feature_columns: Sequence[str]
    random_state: int
    n_components: int = 4
    model_id: str = "M4_four_pace_profile_mixture"


def _initialise_mixture(
    x: np.ndarray,
    *,
    n_components: int,
    random_state: int,
    variance_floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    centred = x - x.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    axis = centred @ vt[0]
    jitter = rng.normal(0.0, 1e-9, size=axis.shape[0])
    order = np.argsort(axis + jitter)
    groups = np.array_split(order, n_components)
    means = np.vstack([x[group].mean(axis=0) for group in groups])
    global_variance = np.maximum(np.var(x, axis=0), variance_floor)
    variances = np.vstack([
        np.maximum(np.var(x[group], axis=0), variance_floor)
        if len(group) > 1
        else global_variance
        for group in groups
    ])
    weights = np.array([len(group) / x.shape[0] for group in groups], dtype=float)
    return weights, means, variances


def _expectation(
    x: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
) -> tuple[np.ndarray, float]:
    component_logs = []
    observed = np.ones(x.shape[1], dtype=bool)
    for row in x:
        logs = np.array(
            [
                np.log(max(weight, 1e-300))
                + mixture_component_logpdf(row, mean, variance, observed)
                for weight, mean, variance in zip(weights, means, variances, strict=True)
            ],
            dtype=float,
        )
        normaliser = logsumexp(logs)
        component_logs.append(logs - normaliser)
    log_responsibilities = np.vstack(component_logs)
    responsibilities = np.exp(log_responsibilities)
    row_log_likelihoods = [
        logsumexp(
            np.array(
                [
                    np.log(max(weight, 1e-300))
                    + mixture_component_logpdf(row, mean, variance, observed)
                    for weight, mean, variance in zip(weights, means, variances, strict=True)
                ],
                dtype=float,
            )
        )
        for row in x
    ]
    return responsibilities, float(np.sum(row_log_likelihoods))


def _posterior_probabilities(
    row: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    observed_mask: np.ndarray,
) -> list[float]:
    logs = np.array(
        [
            np.log(max(weight, 1e-300))
            + mixture_component_logpdf(row, mean, variance, observed_mask)
            for weight, mean, variance in zip(weights, means, variances, strict=True)
        ],
        dtype=float,
    )
    normaliser = logsumexp(logs)
    return [float(value) for value in np.exp(logs - normaliser)]


def mixture_component_logpdf(
    row: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    observed_mask: np.ndarray,
) -> float:
    from trident_validation.models._density import diagonal_gaussian_logpdf

    return diagonal_gaussian_logpdf(row, mean, variance, observed_mask)


def _component_table(values: np.ndarray, feature_columns: Sequence[str]) -> dict[str, dict[str, float]]:
    return {
        f"component_{component_index}": {
            feature: float(value)
            for feature, value in zip(feature_columns, row, strict=True)
        }
        for component_index, row in enumerate(values)
    }

