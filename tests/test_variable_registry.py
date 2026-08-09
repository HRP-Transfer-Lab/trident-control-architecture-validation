from pathlib import Path

import pytest

from trident_validation.config import ConfigValidationError, load_yaml_config
from trident_validation.mechanistic.variable_registry import (
    CAUSAL_FAMILY_IDS,
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
    assert tuple(registry.causal_families) == CAUSAL_FAMILY_IDS
    assert len(registry.identifiability_gates) >= 6


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
