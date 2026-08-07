"""Common scoring helpers."""

from __future__ import annotations

import numpy as np


def mean_squared_error_observed(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error over observed cells only."""

    observed = ~np.isnan(y_true)
    if not observed.any():
        return float("nan")
    return float(np.mean((y_true[observed] - y_pred[observed]) ** 2))

