"""Validation for the staged public mechanism test ladder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from trident_validation.config import ConfigValidationError, load_yaml_config
from trident_validation.mechanistic.variable_registry import load_variable_registry
from trident_validation.provenance import hash_file, hash_mapping


STAGE_IDS = (
    "STAGE1_K_C_V",
    "STAGE2_PACE_EXPRESSION",
    "STAGE3_DYNAMIC_TRAJECTORIES",
    "STAGE4_CRITICALITY_INTERPRETATION",
)


@dataclass(frozen=True)
class PublicMechanismTestLadder:
    """Validated public mechanism test ladder."""

    registry_id: str
    status: str
    stages: tuple[dict[str, Any], ...]
    support_status_by_variable: dict[str, str]
    config_hash: str
    readiness_matrix_hash: str


def load_public_mechanism_test_ladder(
    path: str | Path = "config/public_mechanism_test_ladder_v1.yaml",
    *,
    repo_root: str | Path | None = None,
) -> PublicMechanismTestLadder:
    """Load and validate the staged public mechanism test ladder."""

    config_path = Path(path)
    root = Path(repo_root) if repo_root is not None else config_path.resolve().parents[1]
    config = load_yaml_config(config_path)
    return validate_public_mechanism_test_ladder(config, repo_root=root)


def validate_public_mechanism_test_ladder(
    config: dict[str, Any],
    *,
    repo_root: str | Path,
) -> PublicMechanismTestLadder:
    """Validate stage ordering, support gates and claims boundaries."""

    root = Path(repo_root)
    registry = _required_mapping(config, "registry")
    rules = _required_mapping(config, "global_rules")
    stages = config.get("stages")
    if not isinstance(stages, list) or len(stages) != len(STAGE_IDS):
        raise ConfigValidationError("public mechanism ladder must define exactly four stages")
    if registry.get("id") != "public_mechanism_test_ladder_v1":
        raise ConfigValidationError("registry.id must be public_mechanism_test_ladder_v1")
    if registry.get("status") != "pre_analysis_protocol":
        raise ConfigValidationError("registry.status must be pre_analysis_protocol")
    for field in (
        "formal_claims_allowed",
        "real_transfer_outcomes_allowed",
        "trident_validation_claim_allowed",
        "neural_criticality_claim_allowed",
        "cusp_required",
    ):
        if registry.get(field) is not False:
            raise ConfigValidationError(f"registry.{field} must be false")
    for field in (
        "participant_isolated_validation_required",
        "dataset_transport_required_when_feasible",
        "no_post_outcome_model_redefinition",
        "no_transfer_outcome_inspection",
        "no_forced_pace_profiles",
        "no_neural_criticality_claim",
        "criticality_language_requires_final_stage",
    ):
        if rules.get(field) is not True:
            raise ConfigValidationError(f"global_rules.{field} must be true")

    variable_registry = load_variable_registry(root / str(registry["variable_registry_path"]))
    readiness_path = root / str(registry["readiness_matrix_path"])
    readiness = pd.read_csv(readiness_path)
    support = dict(zip(readiness["variable_id"], readiness["support_status"], strict=True))
    for variable_id in variable_registry.variables:
        if variable_id not in support:
            raise ConfigValidationError(f"readiness matrix missing variable: {variable_id}")

    if tuple(stage["id"] for stage in stages) != STAGE_IDS:
        raise ConfigValidationError("public mechanism ladder stages are out of order")
    _validate_stage1(stages[0], support)
    _validate_stage2(stages[1], support)
    _validate_stage3(stages[2], support)
    _validate_stage4(stages[3], support)
    return PublicMechanismTestLadder(
        registry_id=str(registry["id"]),
        status=str(registry["status"]),
        stages=tuple(dict(stage) for stage in stages),
        support_status_by_variable={str(key): str(value) for key, value in support.items()},
        config_hash=hash_mapping(config),
        readiness_matrix_hash=hash_file(readiness_path),
    )


def _validate_stage1(stage: dict[str, Any], support: dict[str, str]) -> None:
    _require_stage(stage, "STAGE1_K_C_V", "next_real_data_analysis_candidate")
    primary = set(stage["variables"]["primary"])
    if primary != {"K", "C_signal", "V"}:
        raise ConfigValidationError("stage 1 primary variables must be K, C_signal and V")
    if support["K"] != "supported":
        raise ConfigValidationError("stage 1 requires supported K")
    if support["C_signal"] != "partial" or support["V"] != "partial":
        raise ConfigValidationError("stage 1 must label C_signal and V as partial support")
    _require_excluded(stage, {"P_pace", "R_dynamic", "Transfer_external", "PC_calibration", "A_evidence"})
    _require_item(stage, "candidate_comparisons", "K_plus_C_signal_by_V")


def _validate_stage2(stage: dict[str, Any], support: dict[str, str]) -> None:
    _require_stage(stage, "STAGE2_PACE_EXPRESSION", "gated_after_stage1_review")
    if support["P_pace"] != "descriptive_only":
        raise ConfigValidationError("stage 2 requires P_pace to remain descriptive-only")
    _require_item(stage, "candidate_comparisons", "continuous_K_C_V_plus_descriptive_PACE_expression")
    _require_item(stage, "proceed_if", "no_forced_four_profile_claim")


def _validate_stage3(stage: dict[str, Any], support: dict[str, str]) -> None:
    _require_stage(stage, "STAGE3_DYNAMIC_TRAJECTORIES", "gated_after_stage2_review")
    if support["R_dynamic"] != "partial_temporal_not_regime":
        raise ConfigValidationError("stage 3 requires R_dynamic to remain partial temporal support")
    _require_item(stage, "candidate_comparisons", "generic_hidden_state_dynamics")
    _require_item(stage, "candidate_comparisons", "adaptive_locked_scattered_regime_candidates")
    _require_item(stage, "proceed_if", "regime_labels_are_neutral")


def _validate_stage4(stage: dict[str, Any], support: dict[str, str]) -> None:
    _require_stage(stage, "STAGE4_CRITICALITY_INTERPRETATION", "gated_after_stage3_review")
    if support["R_dynamic"] != "partial_temporal_not_regime":
        raise ConfigValidationError("stage 4 cannot start from established dynamic-regime support")
    required = set(stage.get("required_signatures", ()))
    for signature in (
        "transition_sensitive_recovery",
        "reproducible_dwell_and_reentry_structure",
        "hysteresis_or_path_dependence_beyond_ar_nulls",
        "separation_from_generic_hmm_hsmm_and_ar_models",
    ):
        if signature not in required:
            raise ConfigValidationError(f"stage 4 missing required signature: {signature}")
    forbidden = set(stage.get("forbidden_interpretations", ()))
    for item in ("neural_criticality", "brain_critical_zone", "literal_cusp_without_registered_cusp_competitor"):
        if item not in forbidden:
            raise ConfigValidationError(f"stage 4 must forbid {item}")
    _require_item(stage, "proceed_if", "claims_are_limited_to_behavioural_dynamic_analogue")


def _require_stage(stage: dict[str, Any], expected_id: str, expected_status: str) -> None:
    if stage.get("id") != expected_id:
        raise ConfigValidationError(f"expected stage {expected_id}")
    if stage.get("status") != expected_status:
        raise ConfigValidationError(f"{expected_id}.status must be {expected_status}")
    variables = stage.get("variables")
    if not isinstance(variables, dict):
        raise ConfigValidationError(f"{expected_id}.variables must be a mapping")
    if "Transfer_external" not in variables.get("excluded", ()):
        raise ConfigValidationError(f"{expected_id} must exclude real transfer outcomes")


def _require_excluded(stage: dict[str, Any], expected: set[str]) -> None:
    excluded = set(stage["variables"].get("excluded", ()))
    missing = expected.difference(excluded)
    if missing:
        raise ConfigValidationError(
            f"{stage['id']} missing excluded variables: " + ", ".join(sorted(missing))
        )


def _require_item(stage: dict[str, Any], key: str, item: str) -> None:
    values = stage.get(key)
    if not isinstance(values, list) or item not in values:
        raise ConfigValidationError(f"{stage['id']} must include {item} in {key}")


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict) or not value:
        raise ConfigValidationError(f"{key} must be a non-empty mapping")
    return value
