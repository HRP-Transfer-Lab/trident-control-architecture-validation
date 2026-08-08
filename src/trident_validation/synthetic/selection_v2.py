"""Tiered selection rule for the static tournament V2 contract."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence

import numpy as np
import pandas as pd

from trident_validation.models.static_tournament_v2 import (
    STATIC_V2_MODEL_IDS,
    static_v2_tier,
)
from trident_validation.synthetic.recovery import PRIMARY_METRIC, assert_no_ground_truth_columns


@dataclass(frozen=True)
class TieredModelSelection:
    """V2 selection output with explicit same-tier ambiguity reporting."""

    selected_model_id: str
    numerical_best_model_id: str
    primary_metric: str
    selection_reason: str
    selected_tier: int
    numerical_best_tier: int
    same_tier_ambiguous: bool
    ambiguous_model_ids: tuple[str, ...]
    uncertainty_by_model: pd.DataFrame
    per_window_winner: str
    participant_weighted_winner: str
    per_observed_feature_winner: str
    normalisation_sensitive: bool


def select_preferred_model_v2(
    model_scores: pd.DataFrame,
    participant_scores: pd.DataFrame,
    *,
    practical_equivalence_margin: float = 0.01,
    paired_ci_z: float = 1.96,
    model_ids: Sequence[str] = STATIC_V2_MODEL_IDS,
) -> TieredModelSelection:
    """Select under the V2 tier contract.

    Lower tiers are preferred only when they are not meaningfully worse than the
    numerical best. Within a tier, practical ties are reported explicitly.
    """

    assert_no_ground_truth_columns(model_scores)
    if model_scores.empty:
        raise ValueError("model_scores must not be empty")
    scores = model_scores.copy()
    scores["structural_tier"] = scores["model_id"].map(static_v2_tier)
    valid = scores[np.isfinite(scores[PRIMARY_METRIC])].copy()
    if valid.empty:
        raise ValueError("no finite primary model scores are available")
    valid = valid.sort_values(
        [PRIMARY_METRIC, "structural_tier"],
        ascending=[False, True],
        kind="mergesort",
    )
    numerical_best = str(valid.iloc[0]["model_id"])
    uncertainty = paired_uncertainty_against_best_v2(
        participant_scores,
        best_model_id=numerical_best,
        model_ids=tuple(str(model_id) for model_id in model_ids),
        practical_equivalence_margin=practical_equivalence_margin,
        paired_ci_z=paired_ci_z,
    )
    uncertainty_by_model = uncertainty.set_index("model_id", drop=False)
    valid_ids = tuple(str(model_id) for model_id in valid["model_id"])
    not_worse_ids = tuple(
        model_id
        for model_id in valid_ids
        if model_id in uncertainty_by_model.index
        and not bool(uncertainty_by_model.loc[model_id, "meaningfully_worse_than_numerical_best"])
    )
    if not not_worse_ids:
        not_worse_ids = (numerical_best,)
    lowest_tier = min(static_v2_tier(model_id) for model_id in not_worse_ids)
    lowest_tier_ids = tuple(
        model_id for model_id in not_worse_ids if static_v2_tier(model_id) == lowest_tier
    )
    ranked_lowest = valid[valid["model_id"].isin(lowest_tier_ids)].copy()
    ranked_lowest = ranked_lowest.sort_values(
        [PRIMARY_METRIC, "model_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    selected_model_id = str(ranked_lowest.iloc[0]["model_id"])
    same_tier_ambiguous = len(lowest_tier_ids) > 1
    if same_tier_ambiguous:
        selection_reason = "same_tier_practical_ambiguity_reported"
    elif selected_model_id == numerical_best:
        selection_reason = "numerical_best"
    else:
        selection_reason = "lower_tier_not_meaningfully_worse_than_numerical_best"

    winners = {
        "per_window": _metric_winner_v2(scores, PRIMARY_METRIC),
        "participant_weighted": _metric_winner_v2(
            scores,
            "heldout_log_density_participant_weighted",
        ),
        "per_observed_feature": _metric_winner_v2(
            scores,
            "heldout_log_density_mean_per_observed_feature",
        ),
    }
    return TieredModelSelection(
        selected_model_id=selected_model_id,
        numerical_best_model_id=numerical_best,
        primary_metric=PRIMARY_METRIC,
        selection_reason=selection_reason,
        selected_tier=static_v2_tier(selected_model_id),
        numerical_best_tier=static_v2_tier(numerical_best),
        same_tier_ambiguous=same_tier_ambiguous,
        ambiguous_model_ids=tuple(sorted(lowest_tier_ids)) if same_tier_ambiguous else tuple(),
        uncertainty_by_model=uncertainty,
        per_window_winner=winners["per_window"],
        participant_weighted_winner=winners["participant_weighted"],
        per_observed_feature_winner=winners["per_observed_feature"],
        normalisation_sensitive=len(set(winners.values())) > 1,
    )


def paired_uncertainty_against_best_v2(
    participant_scores: pd.DataFrame,
    *,
    best_model_id: str,
    model_ids: Sequence[str],
    practical_equivalence_margin: float,
    paired_ci_z: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_id in model_ids:
        if model_id not in participant_scores.columns:
            continue
        diffs = participant_scores[best_model_id] - participant_scores[model_id]
        diffs = diffs.dropna()
        n_pairs = int(diffs.shape[0])
        mean_delta = float(diffs.mean()) if n_pairs else float("nan")
        sd_delta = float(diffs.std(ddof=1)) if n_pairs > 1 else float("nan")
        se_delta = sd_delta / sqrt(n_pairs) if n_pairs > 1 else float("nan")
        ci_low = mean_delta - paired_ci_z * se_delta if n_pairs > 1 else float("nan")
        ci_high = mean_delta + paired_ci_z * se_delta if n_pairs > 1 else float("nan")
        meaningfully_worse = (
            model_id != best_model_id
            and n_pairs > 1
            and mean_delta > practical_equivalence_margin
            and ci_low > 0
        )
        rows.append(
            {
                "model_id": str(model_id),
                "structural_tier": static_v2_tier(str(model_id)),
                "paired_n_participants": n_pairs,
                "paired_delta_best_minus_model": mean_delta,
                "paired_delta_se": se_delta,
                "paired_delta_ci_low": ci_low,
                "paired_delta_ci_high": ci_high,
                "practical_equivalence_margin": practical_equivalence_margin,
                "meaningfully_worse_than_numerical_best": bool(meaningfully_worse),
            }
        )
    return pd.DataFrame(rows)


def _metric_winner_v2(model_scores: pd.DataFrame, metric: str) -> str:
    ranked = model_scores.copy()
    ranked["structural_tier"] = ranked["model_id"].map(static_v2_tier)
    ranked = ranked[np.isfinite(ranked[metric])].sort_values(
        [metric, "structural_tier"],
        ascending=[False, True],
        kind="mergesort",
    )
    if ranked.empty:
        return "not_available"
    return str(ranked.iloc[0]["model_id"])
