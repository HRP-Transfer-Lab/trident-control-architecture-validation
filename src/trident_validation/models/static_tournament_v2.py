"""Static tournament V2 model-space contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from trident_validation.models.base import TournamentModel
from trident_validation.models.m0_probabilistic import M0ProbabilisticPCAModel
from trident_validation.models.m1_continuous import M1ContinuousManifoldModel
from trident_validation.models.m2_nonlinear import M2EMV1Model
from trident_validation.models.mixture import (
    M3ThreeProfileMixtureModel,
    M4FourProfileMixtureModel,
)


STATIC_TOURNAMENT_V2_ID = "static_tournament_v2"
STATIC_V2_MODEL_IDS = (
    "M0_probabilistic_general_performance",
    "M1_continuous_control_manifold",
    "M2_EM_v1",
    "M3_three_profile_mixture",
    "M4_four_pace_profile_mixture",
)
STATIC_V2_MODEL_TIERS = {
    "M0_probabilistic_general_performance": 0,
    "M1_continuous_control_manifold": 1,
    "M2_EM_v1": 1,
    "M3_three_profile_mixture": 2,
    "M4_four_pace_profile_mixture": 2,
}
STATIC_V2_MODEL_ROLES = {
    "M0_probabilistic_general_performance": "general_factor_baseline",
    "M1_continuous_control_manifold": "structured_continuous_linear_manifold",
    "M2_EM_v1": "structured_continuous_nonlinear_readiness",
    "M3_three_profile_mixture": "secondary_discrete_falsification_candidate",
    "M4_four_pace_profile_mixture": "secondary_discrete_falsification_candidate",
}


@dataclass(frozen=True)
class StaticTournamentV2Contract:
    """Serialisable description of the V2 static model-selection contract."""

    contract_id: str
    model_ids: tuple[str, ...]
    model_tiers: dict[str, int]
    model_roles: dict[str, str]
    same_tier_policy: str
    frozen_predecessor_note: str

    def to_dict(self) -> dict[str, object]:
        return {
            "contract_id": self.contract_id,
            "model_ids": list(self.model_ids),
            "model_tiers": dict(self.model_tiers),
            "model_roles": dict(self.model_roles),
            "same_tier_policy": self.same_tier_policy,
            "frozen_predecessor_note": self.frozen_predecessor_note,
        }


STATIC_TOURNAMENT_V2_CONTRACT = StaticTournamentV2Contract(
    contract_id=STATIC_TOURNAMENT_V2_ID,
    model_ids=STATIC_V2_MODEL_IDS,
    model_tiers=STATIC_V2_MODEL_TIERS,
    model_roles=STATIC_V2_MODEL_ROLES,
    same_tier_policy=(
        "select credible held-out winner; if same-tier models are practically tied, "
        "report ambiguity rather than applying arbitrary numeric-order simplicity"
    ),
    frozen_predecessor_note=(
        "M2_closed_form_v1 remains the frozen M2.6/M2.6b estimator. "
        "M2_EM_v1 is a new versioned candidate."
    ),
)


def build_static_model_suite_v2(
    *,
    feature_columns: Sequence[str],
    random_state: int,
) -> list[TournamentModel]:
    """Build the M0/M1/M2_EM_v1/M3/M4 static V2 suite."""

    return [
        M0ProbabilisticPCAModel(feature_columns=feature_columns, random_state=random_state),
        M1ContinuousManifoldModel(feature_columns=feature_columns, random_state=random_state + 1),
        M2EMV1Model(feature_columns=feature_columns, random_state=random_state + 2),
        M3ThreeProfileMixtureModel(feature_columns=feature_columns, random_state=random_state + 3),
        M4FourProfileMixtureModel(feature_columns=feature_columns, random_state=random_state + 4),
    ]


def static_v2_tier(model_id: str) -> int:
    """Return the V2 structural tier, placing unknown diagnostics after registered tiers."""

    return STATIC_V2_MODEL_TIERS.get(model_id, max(STATIC_V2_MODEL_TIERS.values()) + 1)
