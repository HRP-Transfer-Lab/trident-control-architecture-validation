"""M1: two-dimensional continuous control manifold baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from trident_validation.models.m0_probabilistic import M0ProbabilisticPCAModel


@dataclass
class M1ContinuousManifoldModel(M0ProbabilisticPCAModel):
    """Two-factor PPCA model for continuous static heterogeneity."""

    feature_columns: Sequence[str]
    random_state: int
    model_id: str = "M1_continuous_control_manifold"
    latent_dimensions: int = 2

