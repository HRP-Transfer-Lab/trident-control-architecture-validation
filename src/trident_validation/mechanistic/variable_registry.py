"""Validation for the HRP/Trident Stack variable architecture registries."""

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
SYSTEM_IDS = (
    "operating_capability_state",
    "strategic",
    "representational",
)
ORGANISATION_HYPOTHESIS_IDS = (
    "layer",
    "cross_cutting_operator",
    "hybrid",
)
CANDIDATE_LAYER_IDS = (
    "L_attention",
    "L_WM",
    "L_predictive",
    "L_reasoning",
)
CANDIDATE_OPERATOR_IDS = (
    "OP_binding",
    "OP_relational",
    "OP_predictive",
)

# Backwards-compatible aliases for older callers/tests.
PROGRAMME_IDS = SYSTEM_IDS
REPRESENTATIONAL_LAYER_IDS = CANDIDATE_LAYER_IDS


@dataclass(frozen=True)
class VariableRegistry:
    """Validated variable architecture registry."""

    registry_id: str
    status: str
    variables: dict[str, dict[str, Any]]
    causal_families: dict[str, dict[str, Any]]
    identifiability_gates: tuple[dict[str, Any], ...]
    systems: dict[str, dict[str, Any]]
    representational_hypotheses: dict[str, dict[str, Any]]
    candidate_layers: dict[str, dict[str, Any]]
    candidate_operators: dict[str, dict[str, Any]]
    gates: dict[str, dict[str, Any]]
    m6_2_freeze: dict[str, Any]

    @property
    def programmes(self) -> dict[str, dict[str, Any]]:
        """Compatibility alias for the V3 systems map."""

        return self.systems

    @property
    def representational_layers(self) -> dict[str, dict[str, Any]]:
        """Compatibility alias for V3 candidate layers."""

        return self.candidate_layers

    @property
    def programme_transition_gates(self) -> tuple[dict[str, Any], ...]:
        """Compatibility view over V3 gates."""

        return tuple({"id": key, **value} for key, value in self.gates.items())


def load_variable_registry(path: str | Path) -> VariableRegistry:
    """Load and validate an HRP/Trident Stack variable registry."""

    registry_path = Path(path)
    config = load_yaml_config(registry_path)
    registry = _required_mapping(config, "registry")
    if registry.get("id") == "hrp_stack_variable_registry_v3":
        base_path = registry_path.parent / str(registry.get("base_registry_path", ""))
        if not base_path.exists():
            base_path = registry_path.parents[0].parent / str(registry.get("base_registry_path", ""))
        base = load_variable_registry(base_path)
        return validate_variable_registry(config, base_registry=base)
    return validate_variable_registry(config)


def validate_variable_registry(
    config: dict[str, Any],
    *,
    base_registry: VariableRegistry | None = None,
) -> VariableRegistry:
    """Validate a V2 historical registry or the V3 prospective registry."""

    registry = _required_mapping(config, "registry")
    registry_id = registry.get("id")
    if registry_id == "hrp_stack_variable_registry_v2":
        return _validate_v2_registry(config)
    if registry_id == "hrp_stack_variable_registry_v3":
        if base_registry is None:
            raise ConfigValidationError("V3 registry validation requires the historical V2 base registry")
        return _validate_v3_registry(config, base_registry)
    raise ConfigValidationError("registry.id must be hrp_stack_variable_registry_v2 or hrp_stack_variable_registry_v3")


def _validate_v2_registry(config: dict[str, Any]) -> VariableRegistry:
    registry = _required_mapping(config, "registry")
    variables = _required_mapping(config, "variables")
    families = _required_mapping(config, "causal_families")
    gates = config.get("identifiability_gates")
    if not isinstance(gates, list) or not gates:
        raise ConfigValidationError("identifiability_gates must be a non-empty list")

    if registry.get("id") != "hrp_stack_variable_registry_v2":
        raise ConfigValidationError("registry.id must be hrp_stack_variable_registry_v2")
    _validate_common_registry_boundaries(registry)

    _validate_variables_and_families(variables, families, gates)
    _validate_special_historical_variables(variables)

    return VariableRegistry(
        registry_id=str(registry["id"]),
        status=str(registry.get("status", "")),
        variables={str(key): dict(value) for key, value in variables.items()},
        causal_families={str(key): dict(value) for key, value in families.items()},
        identifiability_gates=tuple(dict(gate) for gate in gates),
        systems={},
        representational_hypotheses={},
        candidate_layers={},
        candidate_operators={},
        gates={},
        m6_2_freeze={},
    )


def _validate_v3_registry(config: dict[str, Any], base_registry: VariableRegistry) -> VariableRegistry:
    registry = _required_mapping(config, "registry")
    if registry.get("status") != "prospective_working_architecture_for_discriminative_testing":
        raise ConfigValidationError("V3 status must be prospective_working_architecture_for_discriminative_testing")
    if registry.get("base_registry_id") != "hrp_stack_variable_registry_v2":
        raise ConfigValidationError("V3 must declare hrp_stack_variable_registry_v2 as its base")
    if registry.get("formulated_after_m6_1") is not True:
        raise ConfigValidationError("V3 must declare that it was formulated after M6.1")
    if registry.get("changes_frozen_m6_1_analysis") is not False:
        raise ConfigValidationError("V3 must not change the frozen M6.1 analysis")
    _validate_common_registry_boundaries(registry)

    systems = _required_mapping(config, "systems")
    missing_systems = set(SYSTEM_IDS).difference(systems)
    if missing_systems:
        raise ConfigValidationError("systems missing required ids: " + ", ".join(sorted(missing_systems)))
    _validate_operating_system(systems["operating_capability_state"])
    _validate_strategic_system(systems["strategic"])
    representational = _validate_representational_system(systems["representational"])

    m6_2 = _required_mapping(config, "m6_2_ontology_neutral_freeze")
    _validate_m6_2_freeze(m6_2)
    future = _required_mapping(config, "future_discriminative_representational_design")
    if future.get("status") != "planning_only" or future.get("run_now") is not False:
        raise ConfigValidationError("future representational design must be planning_only and not run now")
    c_control = _required_mapping(config, "future_C_control_design")
    if c_control.get("status") != "prospective_only" or c_control.get("run_now") is not False:
        raise ConfigValidationError("future C-control design must be prospective_only and not run now")
    gates = _required_mapping(config, "gates")
    _validate_v3_gates(gates)

    return VariableRegistry(
        registry_id=str(registry["id"]),
        status=str(registry.get("status", "")),
        variables=base_registry.variables,
        causal_families=base_registry.causal_families,
        identifiability_gates=base_registry.identifiability_gates,
        systems={str(key): dict(value) for key, value in systems.items()},
        representational_hypotheses=representational["organisation_hypotheses"],
        candidate_layers=representational["candidate_layers"],
        candidate_operators=representational["candidate_operators"],
        gates={str(key): dict(value) for key, value in gates.items()},
        m6_2_freeze=dict(m6_2),
    )


def _validate_common_registry_boundaries(registry: dict[str, Any]) -> None:
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


def _validate_variables_and_families(
    variables: dict[str, Any],
    families: dict[str, Any],
    gates: list[Any],
) -> None:
    missing_variables = set(REQUIRED_VARIABLE_IDS).difference(variables)
    if missing_variables:
        raise ConfigValidationError("variables missing required ids: " + ", ".join(sorted(missing_variables)))
    missing_families = set(CAUSAL_FAMILY_IDS).difference(families)
    if missing_families:
        raise ConfigValidationError("causal_families missing required ids: " + ", ".join(sorted(missing_families)))
    for variable_id in REQUIRED_VARIABLE_IDS:
        _validate_variable(variable_id, variables[variable_id])
    for family_id in CAUSAL_FAMILY_IDS:
        _validate_family(family_id, families[family_id], variables)
    for index, gate in enumerate(gates):
        _validate_identifiability_gate(index, gate, families)


def _validate_special_historical_variables(variables: dict[str, Any]) -> None:
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


def _validate_operating_system(record: Any) -> None:
    if not isinstance(record, dict):
        raise ConfigValidationError("operating_capability_state must be a mapping")
    variables = _required_mapping(record, "variables")
    if tuple(variables) != ("K", "C", "V"):
        raise ConfigValidationError("operating/capability-state system must contain K, C and V in order")
    if variables["C"].get("registry_variable_id") != "C_signal":
        raise ConfigValidationError("C must preserve C_signal as the backwards-compatible registry id")
    if "Flanker-derived C predicting independent CardSort" not in str(variables["C"].get("m6_1_boundary", "")):
        raise ConfigValidationError("C boundary must preserve the narrow M6.1 interpretation")
    if variables["V"].get("psychometric_stability_like_K_required") is not False:
        raise ConfigValidationError("V must not be required to behave psychometrically like K")
    for key, spec in variables.items():
        _validate_candidate_spec(f"operating_capability_state.{key}", spec)


def _validate_strategic_system(record: Any) -> None:
    if not isinstance(record, dict):
        raise ConfigValidationError("strategic system must be a mapping")
    if record.get("system_symbol") != "S":
        raise ConfigValidationError("strategic control must be represented as S")
    variables = _required_mapping(record, "variables")
    if tuple(variables) != ("A_evidence", "T_commit", "PC_calibration"):
        raise ConfigValidationError("strategic system must contain A_evidence, T_commit and PC_calibration")
    pc = variables["PC_calibration"]
    if pc.get("not_a_parallel_strategy_ability") is not True:
        raise ConfigValidationError("PC must not be represented as merely a third parallel strategy ability")
    if record.get("hcp_m6_1_status") != "untested":
        raise ConfigValidationError("S/A/T/PC must remain untested by HCP M6.1")
    if "do_not_infer_strategy_from_ordinary_accuracy_or_rt" not in record.get("forbidden_collapses", []):
        raise ConfigValidationError("strategy-from-RT/accuracy must remain blocked")
    for key, spec in variables.items():
        _validate_candidate_spec(f"strategic.{key}", spec)


def _validate_representational_system(record: Any) -> dict[str, dict[str, dict[str, Any]]]:
    if not isinstance(record, dict):
        raise ConfigValidationError("representational system must be a mapping")
    if record.get("no_ontology_preferred_by_registry_fiat") is not True:
        raise ConfigValidationError("representational registry must not prefer one ontology by fiat")
    hypotheses = _required_mapping(record, "organisation_hypotheses")
    missing_hypotheses = set(ORGANISATION_HYPOTHESIS_IDS).difference(hypotheses)
    if missing_hypotheses:
        raise ConfigValidationError(
            "organisation_hypotheses missing required alternatives: "
            + ", ".join(sorted(missing_hypotheses))
        )
    for hypothesis_id in ORGANISATION_HYPOTHESIS_IDS:
        hypothesis = hypotheses[hypothesis_id]
        _validate_candidate_spec(f"organisation_hypotheses.{hypothesis_id}", hypothesis)
        if hypothesis.get("preferred_by_registry") is not False:
            raise ConfigValidationError("forcing layer/operator/hybrid ontology as confirmed is forbidden")
    layers = _required_mapping(record, "candidate_layers")
    operators = _required_mapping(record, "candidate_operators")
    _validate_candidate_collection("candidate_layers", layers, CANDIDATE_LAYER_IDS)
    _validate_candidate_collection("candidate_operators", operators, CANDIDATE_OPERATOR_IDS)
    return {
        "organisation_hypotheses": {str(key): dict(value) for key, value in hypotheses.items()},
        "candidate_layers": {str(key): dict(value) for key, value in layers.items()},
        "candidate_operators": {str(key): dict(value) for key, value in operators.items()},
    }


def _validate_candidate_collection(
    label: str,
    records: dict[str, Any],
    required_ids: tuple[str, ...],
) -> None:
    missing = set(required_ids).difference(records)
    if missing:
        raise ConfigValidationError(f"{label} missing required ids: " + ", ".join(sorted(missing)))
    for candidate_id in required_ids:
        _validate_candidate_spec(f"{label}.{candidate_id}", records[candidate_id])
        if records[candidate_id].get("single_task_estimation_allowed") is not False:
            raise ConfigValidationError(f"{candidate_id} single-task specific capacity remains blocked")


def _validate_candidate_spec(label: str, record: Any) -> None:
    if not isinstance(record, dict):
        raise ConfigValidationError(f"{label} must be a mapping")
    if record.get("measurement_status") != "candidate":
        raise ConfigValidationError(f"{label}.measurement_status must be candidate")
    if record.get("formal_claims_allowed") is not False:
        raise ConfigValidationError(f"{label} must prohibit formal claims")


def _validate_m6_2_freeze(record: dict[str, Any]) -> None:
    if record.get("status") != "frozen_pending_valid_second_WM_indicator_support":
        raise ConfigValidationError("M6.2 must remain frozen pending valid second-indicator support")
    if record.get("forbidden_indicator") != "WM_Task_2bk_Acc":
        raise ConfigValidationError("WM_Task_2bk_Acc must remain forbidden for M6.2")
    if record.get("fitting_allowed_now") is not False:
        raise ConfigValidationError("M6.2 fitting must remain disabled")
    if "L_WM" not in record.get("forbidden_positive_labels", []):
        raise ConfigValidationError("M6.2 must not establish L_WM directly")
    if "WM_shared_specific_candidate" not in record.get("allowed_positive_label", []):
        raise ConfigValidationError("M6.2 must use the ontology-neutral WM_shared_specific_candidate label")
    primary = _required_mapping(record, "primary_test")
    if primary.get("residuals_must_be_cross_fitted") is not True:
        raise ConfigValidationError("M6.2 residuals must be cross-fitted")
    if primary.get("negative_control") != "CardSort":
        raise ConfigValidationError("CardSort must remain the M6.2 non-WM negative control")


def _validate_v3_gates(gates: dict[str, Any]) -> None:
    required = (
        "single_task_residual_to_specific_capacity",
        "two_same_domain_tasks_to_confirmed_layer_ontology",
        "domain_weight_difference_to_new_latent_factor",
        "K_by_specific_or_C_by_specific_before_independent_candidate",
        "strategy_from_generic_rt_accuracy",
        "layer_operator_hybrid_choice_after_target_outcome_inspection",
        "transfer_outcome_used_to_define_predictor",
    )
    missing = set(required).difference(gates)
    if missing:
        raise ConfigValidationError("gates missing required ids: " + ", ".join(sorted(missing)))
    for gate_id in required:
        if gates[gate_id].get("status") != "forbidden":
            raise ConfigValidationError(f"{gate_id} gate must remain forbidden")


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


def _validate_identifiability_gate(
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
