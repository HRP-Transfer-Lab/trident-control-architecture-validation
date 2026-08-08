import pandas as pd

from trident_validation.models.static_tournament_v2 import (
    STATIC_TOURNAMENT_V2_CONTRACT,
    STATIC_V2_MODEL_IDS,
    build_static_model_suite_v2,
)
from trident_validation.synthetic.fixtures import CORE_SYNTHETIC_FEATURES
from trident_validation.synthetic.selection_v2 import select_preferred_model_v2


def test_static_v2_contract_declares_m2_em_candidate():
    contract = STATIC_TOURNAMENT_V2_CONTRACT.to_dict()

    assert contract["contract_id"] == "static_tournament_v2"
    assert contract["model_ids"] == list(STATIC_V2_MODEL_IDS)
    assert contract["model_tiers"]["M1_continuous_control_manifold"] == 1
    assert contract["model_tiers"]["M2_EM_v1"] == 1


def test_static_v2_suite_uses_fixed_m2_em_v1_settings():
    suite = build_static_model_suite_v2(
        feature_columns=CORE_SYNTHETIC_FEATURES,
        random_state=123,
    )
    by_id = {model.model_id: model for model in suite}

    assert tuple(by_id) == STATIC_V2_MODEL_IDS
    assert by_id["M2_EM_v1"].quadrature_points == 201
    assert by_id["M2_EM_v1"].max_em_iter == 120


def test_v2_selection_reports_same_tier_ambiguity():
    model_scores = pd.DataFrame(
        [
            {
                "model_id": "M0_probabilistic_general_performance",
                "heldout_log_density_mean_per_window": -1.00,
                "heldout_log_density_participant_weighted": -1.00,
                "heldout_log_density_mean_per_observed_feature": -0.20,
            },
            {
                "model_id": "M1_continuous_control_manifold",
                "heldout_log_density_mean_per_window": -0.50,
                "heldout_log_density_participant_weighted": -0.50,
                "heldout_log_density_mean_per_observed_feature": -0.10,
            },
            {
                "model_id": "M2_EM_v1",
                "heldout_log_density_mean_per_window": -0.49,
                "heldout_log_density_participant_weighted": -0.49,
                "heldout_log_density_mean_per_observed_feature": -0.098,
            },
            {
                "model_id": "M3_three_profile_mixture",
                "heldout_log_density_mean_per_window": -0.70,
                "heldout_log_density_participant_weighted": -0.70,
                "heldout_log_density_mean_per_observed_feature": -0.14,
            },
        ]
    )
    participant_scores = pd.DataFrame(
        {
            "M0_probabilistic_general_performance": [-1.00, -1.00, -1.00, -1.00],
            "M1_continuous_control_manifold": [-0.50, -0.50, -0.50, -0.50],
            "M2_EM_v1": [-0.49, -0.49, -0.49, -0.49],
            "M3_three_profile_mixture": [-0.70, -0.70, -0.70, -0.70],
        }
    )

    selection = select_preferred_model_v2(
        model_scores,
        participant_scores,
        practical_equivalence_margin=0.02,
    )

    assert selection.selected_model_id == "M2_EM_v1"
    assert selection.numerical_best_model_id == "M2_EM_v1"
    assert selection.same_tier_ambiguous
    assert selection.ambiguous_model_ids == (
        "M1_continuous_control_manifold",
        "M2_EM_v1",
    )
    assert selection.selection_reason == "same_tier_practical_ambiguity_reported"
