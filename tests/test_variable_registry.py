from pathlib import Path

import pytest

from trident_validation.config import ConfigValidationError, load_yaml_config
from trident_validation.mechanistic.variable_registry import (
    CAUSAL_FAMILY_IDS,
    PROGRAMME_IDS,
    REPRESENTATIONAL_LAYER_IDS,
    REQUIRED_VARIABLE_IDS,
    load_variable_registry,
    validate_variable_registry,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "config/hrp_stack_variable_registry_v2.yaml"


def test_hrp_stack_variable_registry_v2_validates_required_variables_and_families():
    registry = load_variable_registry(REGISTRY_PATH)

    assert registry.registry_id == "hrp_stack_variable_registry_v2"
    assert tuple(registry.variables) == REQUIRED_VARIABLE_IDS
    assert tuple(registry.programmes) == PROGRAMME_IDS
    assert tuple(registry.representational_layers) == REPRESENTATIONAL_LAYER_IDS
    assert tuple(registry.causal_families) == CAUSAL_FAMILY_IDS
    assert len(registry.identifiability_gates) >= 6
    assert len(registry.programme_transition_gates) >= 3


def test_registry_separates_capability_representational_and_strategic_programmes():
    registry = load_variable_registry(REGISTRY_PATH)

    assert tuple(registry.programmes["capability_state"]["variables"]) == ("K", "C_signal", "V")
    assert tuple(registry.programmes["strategic"]["variables"]) == (
        "A_evidence",
        "T_commit",
        "PC_calibration",
    )
    assert tuple(registry.programmes["representational"]["layer_candidates"]) == REPRESENTATIONAL_LAYER_IDS
    assert (
        registry.programmes["capability_state"]["core_question"]
        == "what resources or current conditions are available"
    )
    assert (
        registry.programmes["representational"]["core_question"]
        == "where structural capacity or information bottlenecks bind"
    )
    assert (
        registry.programmes["strategic"]["core_question"]
        == "how the person uses evidence and chooses action under changing context"
    )


def test_registry_blocks_single_task_residuals_as_representational_capacity():
    registry = load_variable_registry(REGISTRY_PATH)
    wm = registry.representational_layers["L_WM"]

    assert wm["status"] == "blocked_pending_independent_second_WM_indicator"
    assert wm["minimum_independent_indicators"] == 2
    assert wm["can_be_estimated_from_single_task_residual"] is False
    assert wm["bottleneck_tests_allowed_before_gate"] is False
    assert "ListSort_Unadj" in wm["current_hcp_eligible_indicators"]
    assert "WM capacity" in wm["forbidden_labels_before_gate"]


def test_registry_keeps_transfer_external_and_blocks_trident_claims():
    config = load_yaml_config(REGISTRY_PATH)
    registry = load_variable_registry(REGISTRY_PATH)

    assert config["registry"]["formal_claims_allowed"] is False
    assert config["registry"]["real_transfer_outcomes_allowed"] is False
    assert config["registry"]["trident_validation_claim_allowed"] is False
    assert config["registry"]["neural_criticality_claim_allowed"] is False
    assert config["registry"]["cusp_required"] is False

    transfer = registry.variables["Transfer_external"]
    assert transfer["measurement_status"] == "external_criterion"
    assert transfer["forbidden_as_latent_input"] is True
    assert "freeze_predictions_before_real_transfer_outcome_inspection" in transfer["negative_controls"]


def test_registry_keeps_pace_descriptive_until_supported():
    registry = load_variable_registry(REGISTRY_PATH)
    pace = registry.variables["P_pace"]

    assert pace["measurement_status"] == "descriptive_until_independently_supported"
    assert "do_not_force_four_profiles" in pace["negative_controls"]
    assert "do_not_treat_pace_as_natural_kind_without_identifiability" in pace["negative_controls"]


def test_registry_rejects_unknown_generated_variable():
    config = load_yaml_config(REGISTRY_PATH)
    config["causal_families"]["MECH0"]["generated_variables"].append("unknown_latent")

    with pytest.raises(ConfigValidationError, match="unknown ids"):
        validate_variable_registry(config)


def test_registry_rejects_programme_boundary_weakening():
    config = load_yaml_config(REGISTRY_PATH)
    config["programmes"]["capability_state"]["variables"].append("A_evidence")

    with pytest.raises(ConfigValidationError, match="capability_state programme"):
        validate_variable_registry(config)


def test_registry_rejects_layer_capacity_from_single_task_residual():
    config = load_yaml_config(REGISTRY_PATH)
    config["representational_layers"]["L_WM"]["can_be_estimated_from_single_task_residual"] = True

    with pytest.raises(ConfigValidationError, match="single-task residual"):
        validate_variable_registry(config)


def test_registry_rejects_premature_bottleneck_tests():
    config = load_yaml_config(REGISTRY_PATH)
    config["representational_layers"]["L_WM"]["bottleneck_tests_allowed_before_gate"] = True

    with pytest.raises(ConfigValidationError, match="bottleneck tests"):
        validate_variable_registry(config)
