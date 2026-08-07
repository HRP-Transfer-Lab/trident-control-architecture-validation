"""Static M0-M4 tournament helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from trident_validation.models.base import ModelScore, TournamentModel
from trident_validation.models.m0_probabilistic import M0ProbabilisticPCAModel
from trident_validation.models.m1_continuous import M1ContinuousManifoldModel
from trident_validation.models.m2_nonlinear import M2NonlinearVigilanceModel
from trident_validation.models.mixture import (
    M3ThreeProfileMixtureModel,
    M4FourProfileMixtureModel,
)
from trident_validation.splits import SplitDefinition


STATIC_MODEL_IDS = (
    "M0_probabilistic_general_performance",
    "M1_continuous_control_manifold",
    "M2_nonlinear_vigilance",
    "M3_three_profile_mixture",
    "M4_four_pace_profile_mixture",
)


@dataclass(frozen=True)
class TournamentResult:
    """Score for one model on one split."""

    model_id: str
    split_name: str
    fold_index: int | None
    primary_metric: str
    primary_value: float
    metrics: dict[str, float]
    diagnostics: dict[str, float]
    metadata: dict[str, object]


def build_static_model_suite(
    *,
    feature_columns: Sequence[str],
    random_state: int,
) -> list[TournamentModel]:
    """Build M0-M4 only, with explicit reproducible seeds."""

    return [
        M0ProbabilisticPCAModel(feature_columns=feature_columns, random_state=random_state),
        M1ContinuousManifoldModel(feature_columns=feature_columns, random_state=random_state + 1),
        M2NonlinearVigilanceModel(feature_columns=feature_columns, random_state=random_state + 2),
        M3ThreeProfileMixtureModel(feature_columns=feature_columns, random_state=random_state + 3),
        M4FourProfileMixtureModel(feature_columns=feature_columns, random_state=random_state + 4),
    ]


def score_static_models_on_split(
    frame: pd.DataFrame,
    split: SplitDefinition,
    *,
    feature_columns: Sequence[str],
    random_state: int,
) -> list[TournamentResult]:
    """Fit and score M0-M4 on one precomputed leakage-safe split."""

    train = frame.loc[list(split.train_indices)]
    test = frame.loc[list(split.test_indices)]
    results: list[TournamentResult] = []
    for model in build_static_model_suite(
        feature_columns=feature_columns,
        random_state=random_state,
    ):
        fitted = model.fit(train)
        score = fitted.score_holdout(test)
        results.append(_result_from_score(score, split))
    return results


def results_to_frame(results: Sequence[TournamentResult]) -> pd.DataFrame:
    """Convert tournament results to a compact tabular summary."""

    return pd.DataFrame(
        [
            {
                "model_id": result.model_id,
                "split_name": result.split_name,
                "fold_index": result.fold_index,
                "primary_metric": result.primary_metric,
                "primary_value": result.primary_value,
                **{f"metric_{key}": value for key, value in result.metrics.items()},
                **{f"diagnostic_{key}": value for key, value in result.diagnostics.items()},
            }
            for result in results
        ]
    )


def _result_from_score(score: ModelScore, split: SplitDefinition) -> TournamentResult:
    return TournamentResult(
        model_id=score.model_id,
        split_name=split.name,
        fold_index=split.fold_index,
        primary_metric=score.primary_metric or "heldout_log_density_mean_per_window",
        primary_value=float(score.primary_value),
        metrics=score.metrics,
        diagnostics=score.diagnostics or {},
        metadata=score.metadata,
    )

