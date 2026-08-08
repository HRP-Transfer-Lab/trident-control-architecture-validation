"""M2: nonlinear vigilance/readiness baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from trident_validation.models._density import LOG_2PI, diagonal_gaussian_logpdf, logsumexp
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
        scores = self._log_density_by_row(values, observed_mask, nodes, log_weights)
        return pd.Series(scores, index=frame.index, name=f"{self.model_id}_log_density")

    def predict_representation(self, frame: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        assert self.adapter_ is not None
        standardised = self.adapter_.observed_standardised(frame)
        values = standardised.to_numpy(dtype=float)
        observed_mask = standardised.notna().to_numpy(dtype=bool)
        nodes, log_weights = self._standard_normal_quadrature()
        posterior_means = self._posterior_moments_by_row(
            values,
            observed_mask,
            nodes,
            log_weights,
        )
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

    def _log_density_by_row(
        self,
        values: np.ndarray,
        observed_mask: np.ndarray,
        nodes: np.ndarray,
        log_weights: np.ndarray,
    ) -> np.ndarray:
        log_density_by_node = self._log_density_by_node(
            values,
            observed_mask,
            nodes,
            log_weights,
        )
        return _logsumexp_axis1(log_density_by_node)

    def _posterior_moments_by_row(
        self,
        values: np.ndarray,
        observed_mask: np.ndarray,
        nodes: np.ndarray,
        log_weights: np.ndarray,
    ) -> np.ndarray:
        log_density_by_node = self._log_density_by_node(
            values,
            observed_mask,
            nodes,
            log_weights,
        )
        normaliser = _logsumexp_axis1(log_density_by_node)
        weights = np.exp(log_density_by_node - normaliser[:, None])
        moments = np.column_stack(
            [
                weights @ nodes,
                weights @ (nodes**2),
            ]
        )
        moments[~np.isfinite(normaliser)] = np.nan
        return moments

    def _log_density_by_node(
        self,
        values: np.ndarray,
        observed_mask: np.ndarray,
        nodes: np.ndarray,
        log_weights: np.ndarray,
    ) -> np.ndarray:
        assert self.coefficients_ is not None
        assert self.residual_variance_ is not None
        basis = np.column_stack([np.ones_like(nodes), nodes, nodes**2 - 1.0])
        means_by_node = basis @ self.coefficients_
        output = np.full((values.shape[0], nodes.shape[0]), np.nan, dtype=float)
        for mask in np.unique(observed_mask, axis=0):
            row_selector = np.all(observed_mask == mask, axis=1)
            if not bool(mask.any()):
                continue
            subset = values[row_selector][:, mask]
            subset_means = means_by_node[:, mask]
            subset_variance = np.maximum(self.residual_variance_[mask], 1e-10)
            residual = subset[:, None, :] - subset_means[None, :, :]
            node_logs = -0.5 * np.sum(
                LOG_2PI
                + np.log(subset_variance)[None, None, :]
                + (residual**2) / subset_variance[None, None, :],
                axis=2,
            )
            output[row_selector] = log_weights[None, :] + node_logs
        return output

    def _standard_normal_quadrature(self) -> tuple[np.ndarray, np.ndarray]:
        nodes, weights = np.polynomial.hermite.hermgauss(self.quadrature_points)
        standard_nodes = np.sqrt(2.0) * nodes
        standard_weights = weights / np.sqrt(np.pi)
        return standard_nodes.astype(float), np.log(standard_weights.astype(float))

    def _require_fitted(self) -> None:
        if not self.fitted_:
            raise M2NonlinearError("fit must be called before prediction or scoring")


@dataclass
class M2ProjectionSearchVigilanceModel(M2NonlinearVigilanceModel):
    """Experimental M2 variant that searches for the latent readiness projection."""

    model_id: str = "M2_projection_search_vigilance"
    n_random_projection_candidates: int = 128
    projection_search_train_log_likelihood_: float | None = field(default=None, init=False)
    projection_search_candidates_evaluated_: int | None = field(default=None, init=False)

    def fit(
        self,
        frame: pd.DataFrame,
        y: Any = None,
        groups: Any = None,
    ) -> "M2ProjectionSearchVigilanceModel":
        adapter = TrainingFeatureAdapter(
            feature_columns=self.feature_columns,
            min_observed_features_per_window=self.min_observed_features_per_window,
        ).fit(frame)
        prepared = adapter.transform(frame)
        x = prepared.values
        if x.shape[0] < 3:
            raise M2NonlinearError("at least three training rows are required")

        candidates = _projection_search_candidates(
            x,
            feature_columns=tuple(adapter.feature_columns),
            random_state=self.random_state,
            n_random=self.n_random_projection_candidates,
        )
        observed_mask = np.ones(x.shape, dtype=bool)
        nodes, log_weights = self._standard_normal_quadrature()
        best: dict[str, Any] | None = None
        for projection in candidates:
            coefficients, residual_variance = _fit_quadratic_given_projection(
                x,
                projection,
                jitter=self.jitter,
            )
            score = float(
                np.sum(
                    _m2_log_density_by_row(
                        x,
                        observed_mask,
                        coefficients,
                        residual_variance,
                        nodes,
                        log_weights,
                    )
                )
            )
            if best is None or score > float(best["score"]):
                best = {
                    "score": score,
                    "projection": projection,
                    "coefficients": coefficients,
                    "residual_variance": residual_variance,
                }
        if best is None:
            raise M2NonlinearError("projection search produced no candidate")

        self.adapter_ = adapter
        self.projection_ = best["projection"]
        self.coefficients_ = best["coefficients"]
        self.residual_variance_ = best["residual_variance"]
        self.projection_search_train_log_likelihood_ = float(best["score"])
        self.projection_search_candidates_evaluated_ = int(candidates.shape[0])
        self.feature_columns = tuple(adapter.feature_columns)
        self.fitted_ = True
        return self

    def get_model_metadata(self) -> dict[str, Any]:
        metadata = super().get_model_metadata()
        metadata.update(
            {
                "model_form": "quadratic_one_dimensional_latent_curve_projection_search",
                "projection_search": {
                    "n_random_projection_candidates": int(self.n_random_projection_candidates),
                    "n_candidates_evaluated": int(
                        self.projection_search_candidates_evaluated_ or 0
                    ),
                    "train_log_likelihood": float(
                        self.projection_search_train_log_likelihood_ or float("nan")
                    ),
                },
            }
        )
        return metadata


@dataclass
class M2LatentCurveEMModel(M2NonlinearVigilanceModel):
    """Experimental M2 variant fitted by EM on the latent quadratic curve."""

    model_id: str = "M2_latent_curve_em"
    quadrature_points: int = 101
    max_em_iter: int = 80
    em_tol: float = 1e-5
    em_train_log_likelihood_: float | None = field(default=None, init=False)
    em_n_iter_: int | None = field(default=None, init=False)
    em_converged_: bool | None = field(default=None, init=False)
    em_final_likelihood_change_: float | None = field(default=None, init=False)
    em_final_parameter_change_: float | None = field(default=None, init=False)

    def fit(self, frame: pd.DataFrame, y: Any = None, groups: Any = None) -> "M2LatentCurveEMModel":
        adapter = TrainingFeatureAdapter(
            feature_columns=self.feature_columns,
            min_observed_features_per_window=self.min_observed_features_per_window,
        ).fit(frame)
        prepared = adapter.transform(frame)
        x = prepared.values
        observed_mask = prepared.observed_mask
        if x.shape[0] < 3:
            raise M2NonlinearError("at least three training rows are required")

        covariance = (x.T @ x) / x.shape[0]
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        projection = eigenvectors[:, np.argsort(eigenvalues)[::-1][:1]]
        projection = _orient_loadings(projection, tuple(adapter.feature_columns))[:, 0]
        coefficients, residual_variance = _fit_quadratic_given_projection(
            x,
            projection,
            jitter=self.jitter,
        )
        nodes, log_weights = self._standard_normal_quadrature()
        previous_e_step_ll: float | None = None
        last_e_step_ll = -np.inf
        final_parameter_change = float("nan")
        converged = False
        n_iter = 0
        for n_iter in range(1, self.max_em_iter + 1):
            responsibilities, e_step_ll = _m2_posterior_node_weights(
                x,
                observed_mask,
                coefficients,
                residual_variance,
                nodes,
                log_weights,
            )
            old_coefficients = coefficients.copy()
            old_residual_variance = residual_variance.copy()
            coefficients, residual_variance = _m2_weighted_curve_update(
                x,
                observed_mask,
                responsibilities,
                nodes,
                jitter=self.jitter,
            )
            final_parameter_change = _m2_parameter_change(
                old_coefficients,
                old_residual_variance,
                coefficients,
                residual_variance,
            )
            last_e_step_ll = e_step_ll
            if previous_e_step_ll is not None and abs(e_step_ll - previous_e_step_ll) < self.em_tol:
                converged = True
                break
            previous_e_step_ll = e_step_ll

        responsibilities, train_ll = _m2_posterior_node_weights(
            x,
            observed_mask,
            coefficients,
            residual_variance,
            nodes,
            log_weights,
        )
        del responsibilities
        final_likelihood_change = abs(train_ll - last_e_step_ll)
        if final_likelihood_change < self.em_tol:
            converged = True

        self.adapter_ = adapter
        self.projection_ = projection
        self.coefficients_ = coefficients
        self.residual_variance_ = residual_variance
        self.em_train_log_likelihood_ = float(train_ll)
        self.em_n_iter_ = int(n_iter)
        self.em_converged_ = bool(converged)
        self.em_final_likelihood_change_ = float(final_likelihood_change)
        self.em_final_parameter_change_ = float(final_parameter_change)
        self.feature_columns = tuple(adapter.feature_columns)
        self.fitted_ = True
        return self

    def get_model_metadata(self) -> dict[str, Any]:
        metadata = super().get_model_metadata()
        metadata.update(
            {
                "model_form": "quadratic_one_dimensional_latent_curve_em",
                "em": {
                    "max_iter": int(self.max_em_iter),
                    "tol": float(self.em_tol),
                    "n_iter": int(self.em_n_iter_ or 0),
                    "converged": bool(self.em_converged_),
                    "final_likelihood_change": float(
                        self.em_final_likelihood_change_ or float("nan")
                    ),
                    "final_parameter_change": float(
                        self.em_final_parameter_change_ or float("nan")
                    ),
                    "train_log_likelihood": float(
                        self.em_train_log_likelihood_ or float("nan")
                    ),
                },
            }
        )
        return metadata


@dataclass
class M2EMV1Model(M2LatentCurveEMModel):
    """Versioned M2_EM_v1 candidate for the static V2 tournament contract."""

    model_id: str = "M2_EM_v1"
    quadrature_points: int = 201
    max_em_iter: int = 120

    def get_model_metadata(self) -> dict[str, Any]:
        metadata = super().get_model_metadata()
        metadata.update(
            {
                "model_id": self.model_id,
                "versioned_candidate": "M2_EM_v1",
                "status": "candidate_for_static_tournament_v2",
                "claim_boundary": "new versioned candidate; not a retroactive replacement for frozen M2.6 M2_closed_form_v1",
                "predictive_stability_contract": {
                    "quadrature_points": int(self.quadrature_points),
                    "max_iter": int(self.max_em_iter),
                    "strict_optimizer_convergence_required_for_valid_score": False,
                    "convergence_metadata_required": True,
                },
            }
        )
        return metadata


def _logsumexp_axis1(values: np.ndarray) -> np.ndarray:
    maximum = np.max(values, axis=1)
    finite = np.isfinite(maximum)
    output = np.full(values.shape[0], np.nan, dtype=float)
    if finite.any():
        stable = values[finite] - maximum[finite, None]
        output[finite] = maximum[finite] + np.log(np.sum(np.exp(stable), axis=1))
    output[~finite] = maximum[~finite]
    return output


def _fit_quadratic_given_projection(
    x: np.ndarray,
    projection: np.ndarray,
    *,
    jitter: float,
) -> tuple[np.ndarray, np.ndarray]:
    z = x @ projection
    z_scale = float(np.std(z, ddof=0)) or 1.0
    z = (z - float(np.mean(z))) / z_scale
    design = np.column_stack([np.ones_like(z), z, z**2 - 1.0])
    coefficients = np.linalg.pinv(design) @ x
    fitted = design @ coefficients
    residuals = x - fitted
    residual_variance = np.maximum(np.var(residuals, axis=0), jitter)
    return coefficients, residual_variance


def _projection_search_candidates(
    x: np.ndarray,
    *,
    feature_columns: tuple[str, ...],
    random_state: int,
    n_random: int,
) -> np.ndarray:
    covariance = (x.T @ x) / x.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    pca_order = np.argsort(eigenvalues)[::-1]
    candidates = [eigenvectors[:, index] for index in pca_order]

    rng = np.random.default_rng(random_state)
    random_candidates = rng.normal(size=(max(0, int(n_random)), x.shape[1]))
    random_norms = np.linalg.norm(random_candidates, axis=1)
    for candidate, norm in zip(random_candidates, random_norms, strict=True):
        if norm > 0:
            candidates.append(candidate / norm)

    oriented = _orient_loadings(np.column_stack(candidates), feature_columns).T
    unique: list[np.ndarray] = []
    for candidate in oriented:
        norm = float(np.linalg.norm(candidate))
        if norm == 0:
            continue
        candidate = candidate / norm
        if not any(abs(float(np.dot(candidate, existing))) > 0.999999 for existing in unique):
            unique.append(candidate)
    return np.vstack(unique)


def _m2_log_density_by_row(
    values: np.ndarray,
    observed_mask: np.ndarray,
    coefficients: np.ndarray,
    residual_variance: np.ndarray,
    nodes: np.ndarray,
    log_weights: np.ndarray,
) -> np.ndarray:
    basis = np.column_stack([np.ones_like(nodes), nodes, nodes**2 - 1.0])
    means_by_node = basis @ coefficients
    output = np.full((values.shape[0], nodes.shape[0]), np.nan, dtype=float)
    for mask in np.unique(observed_mask, axis=0):
        row_selector = np.all(observed_mask == mask, axis=1)
        if not bool(mask.any()):
            continue
        subset = values[row_selector][:, mask]
        subset_means = means_by_node[:, mask]
        subset_variance = np.maximum(residual_variance[mask], 1e-10)
        residual = subset[:, None, :] - subset_means[None, :, :]
        node_logs = -0.5 * np.sum(
            LOG_2PI
            + np.log(subset_variance)[None, None, :]
            + (residual**2) / subset_variance[None, None, :],
            axis=2,
        )
        output[row_selector] = log_weights[None, :] + node_logs
    return _logsumexp_axis1(output)


def _m2_posterior_node_weights(
    values: np.ndarray,
    observed_mask: np.ndarray,
    coefficients: np.ndarray,
    residual_variance: np.ndarray,
    nodes: np.ndarray,
    log_weights: np.ndarray,
) -> tuple[np.ndarray, float]:
    component_logs = _m2_log_density_by_node_matrix(
        values,
        observed_mask,
        coefficients,
        residual_variance,
        nodes,
        log_weights,
    )
    normaliser = _logsumexp_axis1(component_logs)
    responsibilities = np.exp(component_logs - normaliser[:, None])
    responsibilities[~np.isfinite(normaliser)] = 0.0
    return responsibilities, float(np.nansum(normaliser))


def _m2_weighted_curve_update(
    values: np.ndarray,
    observed_mask: np.ndarray,
    responsibilities: np.ndarray,
    nodes: np.ndarray,
    *,
    jitter: float,
) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack([np.ones_like(nodes), nodes, nodes**2 - 1.0])
    coefficients = np.zeros((design.shape[1], values.shape[1]), dtype=float)
    residual_variance = np.full(values.shape[1], jitter, dtype=float)
    for feature_index in range(values.shape[1]):
        row_selector = observed_mask[:, feature_index]
        if not bool(row_selector.any()):
            continue
        weights_by_node = responsibilities[row_selector].sum(axis=0)
        xtwx = design.T @ (weights_by_node[:, None] * design)
        xtwy = design.T @ (responsibilities[row_selector].T @ values[row_selector, feature_index])
        beta = np.linalg.pinv(xtwx) @ xtwy
        coefficients[:, feature_index] = beta
        fitted_by_node = design @ beta
        residual = values[row_selector, feature_index][:, None] - fitted_by_node[None, :]
        weighted_sse = float(np.sum(responsibilities[row_selector] * residual**2))
        residual_variance[feature_index] = max(
            weighted_sse / float(row_selector.sum()),
            jitter,
        )
    return coefficients, residual_variance


def _m2_parameter_change(
    old_coefficients: np.ndarray,
    old_residual_variance: np.ndarray,
    new_coefficients: np.ndarray,
    new_residual_variance: np.ndarray,
) -> float:
    coefficient_change = float(np.max(np.abs(new_coefficients - old_coefficients)))
    residual_change = float(np.max(np.abs(new_residual_variance - old_residual_variance)))
    return max(coefficient_change, residual_change)


def _m2_log_density_by_node_matrix(
    values: np.ndarray,
    observed_mask: np.ndarray,
    coefficients: np.ndarray,
    residual_variance: np.ndarray,
    nodes: np.ndarray,
    log_weights: np.ndarray,
) -> np.ndarray:
    basis = np.column_stack([np.ones_like(nodes), nodes, nodes**2 - 1.0])
    means_by_node = basis @ coefficients
    output = np.full((values.shape[0], nodes.shape[0]), np.nan, dtype=float)
    for mask in np.unique(observed_mask, axis=0):
        row_selector = np.all(observed_mask == mask, axis=1)
        if not bool(mask.any()):
            continue
        subset = values[row_selector][:, mask]
        subset_means = means_by_node[:, mask]
        subset_variance = np.maximum(residual_variance[mask], 1e-10)
        residual = subset[:, None, :] - subset_means[None, :, :]
        node_logs = -0.5 * np.sum(
            LOG_2PI
            + np.log(subset_variance)[None, None, :]
            + (residual**2) / subset_variance[None, None, :],
            axis=2,
        )
        output[row_selector] = log_weights[None, :] + node_logs
    return output

