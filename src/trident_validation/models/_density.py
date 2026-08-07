"""Small numerical helpers for held-out density calculations."""

from __future__ import annotations

import numpy as np


LOG_2PI = float(np.log(2.0 * np.pi))


def logsumexp(values: np.ndarray) -> float:
    """Stable log(sum(exp(values))) for one-dimensional arrays."""

    if values.size == 0:
        return float("nan")
    maximum = float(np.max(values))
    if not np.isfinite(maximum):
        return maximum
    return float(maximum + np.log(np.sum(np.exp(values - maximum))))


def diagonal_gaussian_logpdf(
    values: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    observed_mask: np.ndarray,
) -> float:
    """Marginal diagonal Gaussian log density over observed cells."""

    if not observed_mask.any():
        return float("nan")
    observed_values = values[observed_mask]
    observed_mean = mean[observed_mask]
    observed_variance = np.maximum(variance[observed_mask], 1e-10)
    residual = observed_values - observed_mean
    return float(
        -0.5
        * np.sum(LOG_2PI + np.log(observed_variance) + (residual**2) / observed_variance)
    )


def mixture_logpdf(
    values: np.ndarray,
    weights: np.ndarray,
    means: np.ndarray,
    variances: np.ndarray,
    observed_mask: np.ndarray,
) -> float:
    """Marginal log density under a diagonal Gaussian mixture."""

    component_logs = np.array(
        [
            np.log(max(weight, 1e-300))
            + diagonal_gaussian_logpdf(values, mean, variance, observed_mask)
            for weight, mean, variance in zip(weights, means, variances, strict=True)
        ],
        dtype=float,
    )
    return logsumexp(component_logs)

