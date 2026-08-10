from pathlib import Path

import pytest

from trident_validation.config import ConfigValidationError, load_yaml_config
from trident_validation.mechanistic.variable_registry import (
    CANDIDATE_LAYER_IDS,
    CANDIDATE_OPERATOR_IDS,
    CAUSAL_FAMILY_IDS,
    ORGANISATION_HYPOTHESIS_IDS,
    REQUIRED_VARIABLE_IDS,
    load_variable_registry,
    validate_variable_registry,
)


ROOT = Path(__file__).resolve().parents[1]
V2_REGISTRY_PATH = ROOT / "config/hrp_stack_variable_registry_v2.yaml"
V3_REGISTRY_PATH = ROOT / "config/hrp_stack_variable_registry_v3.yaml"


def test_hrp_stack_variable_registry_v2_remains_historical_and_loadable():
    registry = load_variable_registry(V2_REGISTRY_PATH)

    assert registry.registry_id == "hrp_stack_variable_registry_v2"
    assert registry.status == "frozen_for_mechanistic_synthetic_design"
    assert tuple(registry.variables) == REQUIRED_VARIABLE_IDS
    assert tuple(registry.causal_families) == CAUSAL_FAMILY_IDS
    assert len(registry.identifiability_gates) >= 6
    assert registry.systems == {}
    assert registry.candidate_layers == {}
    assert registry.candidate_operators == {}


def test_registry_keeps_transfer_external_and_blocks_trident_claims():
    config = load_yaml_config(V2_REGISTRY_PATH)
    registry = load_variable_registry(V2_REGISTRY_PATH)

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
    registry = load_variable_registry(V2_REGISTRY_PATH)
    pace = registry.variables["P_pace"]

    assert pace["measurement_status"] == "descriptive_until_independently_supported"
    assert "do_not_force_four_profiles" in pace["negative_controls"]
    assert "do_not_treat_pace_as_natural_kind_without_identifiability" in pace["negative_controls"]


def test_registry_rejects_unknown_generated_variable():
    config = load_yaml_config(V2_REGISTRY_PATH)
    config["causal_families"]["MECH0"]["generated_variables"].append("unknown_latent")

    with pytest.raises(ConfigValidationError, match="unknown ids"):
        validate_variable_registry(config)


def test_hrp_stack_variable_registry_v3_extends_v2_with_ontology_neutral_programmes():
    registry = load_variable_registry(V3_REGISTRY_PATH)

    assert registry.registry_id == "hrp_stack_variable_registry_v3"
    assert registry.status == "prospective_working_architecture_for_discriminative_testing"
    assert tuple(registry.variables) == REQUIRED_VARIABLE_IDS
    assert tuple(registry.representational_hypotheses) == ORGANISATION_HYPOTHESIS_IDS
    assert tuple(registry.candidate_layers) == CANDIDATE_LAYER_IDS
    assert tuple(registry.candidate_operators) == CANDIDATE_OPERATOR_IDS
    assert registry.systems["operating_capability_state"]["variables"]["C"]["registry_variable_id"] == "C_signal"
    assert registry.systems["strategic"]["variables"]["PC_calibration"]["not_a_parallel_strategy_ability"] is True


def test_v3_contains_all_three_organisational_hypotheses_without_preference():
    registry = load_variable_registry(V3_REGISTRY_PATH)

    assert set(registry.representational_hypotheses) == {
        "layer",
        "cross_cutting_operator",
        "hybrid",
    }
    assert all(
        hypothesis["measurement_status"] == "candidate"
        and hypothesis["formal_claims_allowed"] is False
        and hypothesis["preferred_by_registry"] is False
        for hypothesis in registry.representational_hypotheses.values()
    )


def test_v3_rejects_removing_operator_alternatives():
    config = load_yaml_config(V3_REGISTRY_PATH)
    base = load_variable_registry(V2_REGISTRY_PATH)
    del config["systems"]["representational"]["candidate_operators"]["OP_relational"]

    with pytest.raises(ConfigValidationError, match="candidate_operators missing"):
        validate_variable_registry(config, base_registry=base)


def test_v3_rejects_forcing_layer_ontology_as_confirmed():
    config = load_yaml_config(V3_REGISTRY_PATH)
    base = load_variable_registry(V2_REGISTRY_PATH)
    config["systems"]["representational"]["organisation_hypotheses"]["layer"]["preferred_by_registry"] = True

    with pytest.raises(ConfigValidationError, match="ontology as confirmed"):
        validate_variable_registry(config, base_registry=base)


def test_v3_blocks_single_task_specific_capacity():
    config = load_yaml_config(V3_REGISTRY_PATH)
    base = load_variable_registry(V2_REGISTRY_PATH)
    config["systems"]["representational"]["candidate_layers"]["L_WM"]["single_task_estimation_allowed"] = True

    with pytest.raises(ConfigValidationError, match="single-task specific capacity"):
        validate_variable_registry(config, base_registry=base)


def test_v3_blocks_premature_bottleneck_tests():
    config = load_yaml_config(V3_REGISTRY_PATH)
    base = load_variable_registry(V2_REGISTRY_PATH)
    config["gates"]["K_by_specific_or_C_by_specific_before_independent_candidate"]["status"] = "allowed"

    with pytest.raises(ConfigValidationError, match="must remain forbidden"):
        validate_variable_registry(config, base_registry=base)


def test_v3_blocks_strategy_from_generic_rt_accuracy():
    config = load_yaml_config(V3_REGISTRY_PATH)
    base = load_variable_registry(V2_REGISTRY_PATH)
    config["gates"]["strategy_from_generic_rt_accuracy"]["status"] = "allowed"

    with pytest.raises(ConfigValidationError, match="must remain forbidden"):
        validate_variable_registry(config, base_registry=base)


def test_v3_freezes_m6_2_as_ontology_neutral_shared_wm_candidate():
    registry = load_variable_registry(V3_REGISTRY_PATH)
    m6_2 = registry.m6_2_freeze

    assert m6_2["fitting_allowed_now"] is False
    assert m6_2["forbidden_indicator"] == "WM_Task_2bk_Acc"
    assert "WM_shared_specific_candidate" in m6_2["allowed_positive_label"]
    assert "L_WM" in m6_2["forbidden_positive_labels"]
    assert m6_2["primary_test"]["residuals_must_be_cross_fitted"] is True
    assert m6_2["primary_test"]["negative_control"] == "CardSort"
