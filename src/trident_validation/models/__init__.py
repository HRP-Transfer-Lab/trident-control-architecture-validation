"""Model interfaces and Milestone 1/1.5 baseline models."""

from .adapters import FeatureMatrix, TrainingFeatureAdapter
from .base import BaseValidationModel, FeatureRequirements, ModelScore, TournamentModel
from .m0_general import M0GeneralPerformanceModel
from .m0_probabilistic import M0ProbabilisticPCAModel
from .m1_continuous import M1ContinuousManifoldModel
from .m2_nonlinear import M2EMV1Model, M2NonlinearVigilanceModel
from .mixture import M3ThreeProfileMixtureModel, M4FourProfileMixtureModel
from .static_tournament import (
    STATIC_MODEL_IDS,
    TournamentResult,
    build_static_model_suite,
    results_to_frame,
    score_static_models_on_split,
)
from .static_tournament_v2 import (
    STATIC_TOURNAMENT_V2_CONTRACT,
    STATIC_V2_MODEL_IDS,
    build_static_model_suite_v2,
)

__all__ = [
    "BaseValidationModel",
    "FeatureMatrix",
    "FeatureRequirements",
    "ModelScore",
    "M0GeneralPerformanceModel",
    "M0ProbabilisticPCAModel",
    "M1ContinuousManifoldModel",
    "M2NonlinearVigilanceModel",
    "M2EMV1Model",
    "M3ThreeProfileMixtureModel",
    "M4FourProfileMixtureModel",
    "STATIC_MODEL_IDS",
    "STATIC_TOURNAMENT_V2_CONTRACT",
    "STATIC_V2_MODEL_IDS",
    "TournamentModel",
    "TournamentResult",
    "TrainingFeatureAdapter",
    "build_static_model_suite",
    "build_static_model_suite_v2",
    "results_to_frame",
    "score_static_models_on_split",
]
