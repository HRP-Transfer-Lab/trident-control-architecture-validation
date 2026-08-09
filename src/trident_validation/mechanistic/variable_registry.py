"""Validation for the HRP/Trident Stack variable architecture registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from trident_validation.config import ConfigValidationError, load_yaml_config


REQUIRED_VARIABLE_IDS = (
    "K",
    "V",
    "C_signal",
    "A_evidence",
    "T_commit",
    "PC_calibration",
    "R_dynamic",
    "P_pace",
    "Y_behavior",
    "Transfer_external",
)
CAUSAL_FAMILY_IDS = (
    "MECH0",
    "MECH1",
    "MECH2",
    "MECH3",
    "MECH4",
    "MECH5",
    "MECH6",
    "MECH7",
)


@dataclass(frozen=True)
class VariableRegistry:
    """Validated variable architecture registry."""

    registry_id: str
    status: str
    variables: dict[str, dict[str, Any]]
    causal_families: dict[str, dict[str, Any]]
    identifiability_gates: tuple[dict[str, Any], ...]


def load_variable_registry(path: str | Path) -> VariableRegistry:
    """Load and validate the HRP/Trident Stack variable registry."""

    return validate_variable_registry(load_yaml_config(path))


def validate_variable_registry(config: dict[str, Any]) -> VariableRegistry:
    """Validate the M3 variable architecture contract."""

    registry = _required_mapping(config, "registry")
    variables = _required_mapping(config, "variables")
    families = _required_mapping(config, "causal_families")
    gates = config.get("identifiability_gates")
    if not isinstance(gates, list) or not gates:
        raise ConfigValidationError("identifiability_gates must be a non-empty list")

    if registry.get("id") != "hrp_stack_variable_registry_v2":
        raise ConfigValidationError("registry.id must be hrp_stack_variable_registry_v2")
    if registry.get("formal_claims_allowed") is not False:
        raise ConfigValidationError("registry must prohibit formal claims")
    if registry.get("real_transfer_outcomes_allowed") is not False:
        raise ConfigValidationError("registry must prohibit real transfer outcomes")
    if registry.get("trident_validation_claim_allowed") is not False:
        raise ConfigValidationError("registry must prohibit Trident validation claims")
    if registry.get("neural_criticality_claim_allowed") is not False:
        raise ConfigValidationError("registry must prohibit neural criticality claims")
    if registry.get("cusp_required") is not False:
        raise ConfigValidationError("registry must not require a cusp")

    missing_variables = set(REQUIRED_VARIABLE_IDS).difference(variables)
    if missing_variables:
        raise ConfigValidationError(
            "variables missing required ids: " + ", ".join(sorted(missing_variables))
        )
    missing_families = set(CAUSAL_FAMILY_IDS).difference(families)
    if missing_families:
        raise ConfigValidationError(
            "causal_families missing required ids: " + ", ".join(sorted(missing_families))
        )

    for variable_id in REQUIRED_VARIABLE_IDS:
        _validate_variable(variable_id, variables[variable_id])
    for family_id in CAUSAL_FAMILY_IDS:
        _validate_family(family_id, families[family_id], variables)
    for index, gate in enumerate(gates):
        _validate_gate(index, gate, families)

    transfer = variables["Transfer_external"]
    if transfer.get("measurement_status") != "external_criterion":
        raise ConfigValidationError("Transfer_external must be an external criterion")
    if transfer.get("forbidden_as_latent_input") is not True:
        raise ConfigValidationError("Transfer_external must be forbidden as a latent input")
    if transfer.get("can_define_transfer_outcome") is not True:
        raise ConfigValidationError("Transfer_external must define transfer outcomes")

    pace = variables["P_pace"]
    if pace.get("measurement_status") != "descriptive_until_independently_supported":
        raise ConfigValidationError("P_pace must remain descriptive until independently supported")

    return VariableRegistry(
        registry_id=str(registry["id"]),
        status=str(registry.get("status", "")),
        variables={str(key): dict(value) for key, value in variables.items()},
        causal_families={str(key): dict(value) for key, value in families.items()},
        identifiability_gates=tuple(dict(gate) for gate in gates),
    )


def _validate_variable(variable_id: str, record: Any) -> None:
    if not isinstance(record, dict):
        raise ConfigValidationError(f"{variable_id} must be a mapping")
    for field in ("label", "role", "timescale", "measurement_status", "observable_families"):
        if field not in record:
            raise ConfigValidationError(f"{variable_id} missing required field: {field}")
    observables = record["observable_families"]
    if not isinstance(observables, list) or not observables:
        raise ConfigValidationError(f"{variable_id}.observable_families must be a non-empty list")


def _validate_family(
    family_id: str,
    record: Any,
    variables: dict[str, Any],
) -> None:
    if not isinstance(record, dict):
        raise ConfigValidationError(f"{family_id} must be a mapping")
    for field in ("label", "synthetic_truth", "generated_variables", "identifiability_question"):
        if field not in record:
            raise ConfigValidationError(f"{family_id} missing required field: {field}")
    generated = record["generated_variables"]
    if not isinstance(generated, list) or not generated:
        raise ConfigValidationError(f"{family_id}.generated_variables must be a non-empty list")
    unknown = set(str(variable) for variable in generated).difference(variables)
    if unknown:
        raise ConfigValidationError(
            f"{family_id}.generated_variables contains unknown ids: "
            + ", ".join(sorted(unknown))
        )
    if "Y_behavior" not in generated:
        raise ConfigValidationError(f"{family_id} must generate Y_behavior")


def _validate_gate(
    index: int,
    record: Any,
    families: dict[str, Any],
) -> None:
    if not isinstance(record, dict):
        raise ConfigValidationError(f"identifiability_gates[{index}] must be a mapping")
    for field in ("id", "truth_families", "required_result"):
        if field not in record:
            raise ConfigValidationError(f"identifiability_gates[{index}] missing field: {field}")
    truth_families = record["truth_families"]
    if not isinstance(truth_families, list) or len(truth_families) < 2:
        raise ConfigValidationError(
            f"identifiability_gates[{index}].truth_families must contain at least two families"
        )
    unknown = set(str(family) for family in truth_families).difference(families)
    if unknown:
        raise ConfigValidationError(
            f"identifiability_gates[{index}] references unknown families: "
            + ", ".join(sorted(unknown))
        )


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict) or not value:
        raise ConfigValidationError(f"{key} must be a non-empty mapping")
    return value
