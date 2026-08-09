"""Pre-outcome M2.8 mechanistic identifiability design validation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from trident_validation.config import ConfigValidationError, load_yaml_config
from trident_validation.mechanistic.variable_registry import (
    CAUSAL_FAMILY_IDS,
    REQUIRED_VARIABLE_IDS,
    VariableRegistry,
    load_variable_registry,
)
from trident_validation.provenance import hash_file, hash_mapping


FORBIDDEN_PRIMARY_TERMS = (
    "literal_cusp",
    "cusp",
    "bifurcation",
    "neural_criticality",
    "criticality",
)
REQUIRED_SEED_STREAMS = ("structural", "nuisance", "observation", "split", "model")


@dataclass(frozen=True)
class MechanisticIdentifiabilityPlan:
    """Validated pre-outcome M2.8 design."""

    registry_id: str
    status: str
    variable_registry_path: Path
    variable_registry_commit: str
    config_hash: str
    schedule_hash: str
    schedule: pd.DataFrame


def load_mechanistic_identifiability_plan(
    config_path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> MechanisticIdentifiabilityPlan:
    """Load, validate and materialise the deterministic M2.8 smoke schedule."""

    path = Path(config_path)
    root = Path(repo_root) if repo_root is not None else path.resolve().parents[1]
    config = load_yaml_config(path)
    variable_registry_path = root / str(config["registry"]["variable_registry_path"])
    variable_registry = load_variable_registry(variable_registry_path)
    return validate_mechanistic_identifiability_plan(
        config,
        variable_registry=variable_registry,
        variable_registry_path=variable_registry_path,
    )


def validate_mechanistic_identifiability_plan(
    config: dict[str, Any],
    *,
    variable_registry: VariableRegistry,
    variable_registry_path: str | Path,
) -> MechanisticIdentifiabilityPlan:
    """Validate M2.8 design constraints and return its deterministic schedule."""

    registry = _required_mapping(config, "registry")
    design = _required_mapping(config, "primary_design")
    smoke = _required_mapping(config, "smoke_design")
    gates = config.get("identifiability_gates")
    scoring_families = config.get("candidate_scoring_families")
    if not isinstance(gates, list) or not gates:
        raise ConfigValidationError("identifiability_gates must be a non-empty list")
    if not isinstance(scoring_families, list) or not scoring_families:
        raise ConfigValidationError("candidate_scoring_families must be a non-empty list")

    _validate_registry_header(registry)
    _validate_claim_boundaries(registry, design)
    _validate_primary_family_set(design)
    _validate_seed_streams(smoke)
    _validate_scoring_families(scoring_families, variable_registry)
    _validate_gates(gates, variable_registry)

    schedule = build_identifiability_schedule(config, variable_registry=variable_registry)
    schedule_hash = dataframe_hash(schedule)
    return MechanisticIdentifiabilityPlan(
        registry_id=str(registry["id"]),
        status=str(registry["status"]),
        variable_registry_path=Path(variable_registry_path),
        variable_registry_commit=str(registry["variable_registry_commit"]),
        config_hash=hash_mapping(config),
        schedule_hash=schedule_hash,
        schedule=schedule,
    )


def build_identifiability_schedule(
    config: dict[str, Any],
    *,
    variable_registry: VariableRegistry,
) -> pd.DataFrame:
    """Build a deterministic gate x truth-family x replicate schedule."""

    smoke = _required_mapping(config, "smoke_design")
    gates = config["identifiability_gates"]
    replicates = int(smoke["replicates_per_truth_family_per_gate"])
    master_seed = int(smoke["master_seed"])
    rows: list[dict[str, Any]] = []
    task_index = 0
    for gate_index, gate in enumerate(gates):
        gate_id = str(gate["id"])
        for family_index, family_id in enumerate(gate["truth_families"]):
            family = variable_registry.causal_families[str(family_id)]
            for replicate_index in range(replicates):
                unit_id = f"{gate_id}_{family_id}_replicate_{replicate_index:03d}"
                row = {
                    "task_index": task_index,
                    "unit_id": unit_id,
                    "gate_index": gate_index,
                    "gate_id": gate_id,
                    "truth_family_index": family_index,
                    "truth_family_id": str(family_id),
                    "truth_family_label": str(family["label"]),
                    "replicate_index": replicate_index,
                    "generated_variables": "|".join(str(item) for item in family["generated_variables"]),
                    "primary_variables": "|".join(str(item) for item in gate["primary_variables"]),
                    "participants": int(smoke["participants_per_replicate"]),
                    "sessions_per_participant": int(smoke["sessions_per_participant"]),
                    "blocks_per_session": int(smoke["blocks_per_session"]),
                    "participant_holdout_fraction": float(smoke["participant_holdout_fraction"]),
                    "master_seed": master_seed,
                    "structural_seed": child_seed(master_seed, unit_id, "structural"),
                    "nuisance_seed": child_seed(master_seed, unit_id, "nuisance"),
                    "observation_seed": child_seed(master_seed, unit_id, "observation"),
                    "split_seed": child_seed(master_seed, unit_id, "split"),
                    "model_seed": child_seed(master_seed, unit_id, "model"),
                    "transfer_external_use": str(config["primary_design"]["transfer_external_use"]),
                    "real_wrapper_transfer_outcomes_inspected": bool(
                        config["primary_design"]["real_wrapper_transfer_outcomes_inspected"]
                    ),
                    "truth_columns_stripped_before_scoring_required": bool(
                        config["primary_design"]["truth_columns_stripped_before_scoring_required"]
                    ),
                    "variable_registry_commit": str(config["registry"]["variable_registry_commit"]),
                }
                rows.append(row)
                task_index += 1
    return pd.DataFrame(rows)


def write_identifiability_plan_outputs(
    config_path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> MechanisticIdentifiabilityPlan:
    """Write the pre-outcome schedule and manifest registered in the config."""

    path = Path(config_path)
    root = Path(repo_root) if repo_root is not None else path.resolve().parents[1]
    config = load_yaml_config(path)
    plan = load_mechanistic_identifiability_plan(path, repo_root=root)
    outputs = _required_mapping(config, "outputs")
    schedule_csv = root / str(outputs["schedule_csv"])
    schedule_json = root / str(outputs["schedule_json"])
    manifest_json = root / str(outputs["manifest_json"])

    _atomic_write_csv(schedule_csv, plan.schedule)
    _atomic_write_json(schedule_json, plan.schedule.to_dict(orient="records"))
    manifest = {
        "study_id": plan.registry_id,
        "status": plan.status,
        "protocol_path": str(config["registry"]["protocol_path"]),
        "variable_registry_path": str(config["registry"]["variable_registry_path"]),
        "variable_registry_commit": plan.variable_registry_commit,
        "config_path": str(path.as_posix()),
        "config_hash": plan.config_hash,
        "schedule_hash": plan.schedule_hash,
        "n_units": int(plan.schedule.shape[0]),
        "n_gates": int(plan.schedule["gate_id"].nunique()),
        "truth_family_ids": sorted(plan.schedule["truth_family_id"].unique().tolist()),
        "formal_claims_allowed": False,
        "real_transfer_outcomes_allowed": False,
        "real_wrapper_transfer_outcomes_inspected": False,
        "literal_cusp_status": str(config["primary_design"]["literal_cusp_status"]),
        "transfer_external_use": str(config["primary_design"]["transfer_external_use"]),
        "output_hashes": {
            "schedule_csv": hash_file(schedule_csv),
            "schedule_json": hash_file(schedule_json),
        },
    }
    _atomic_write_json(manifest_json, manifest)
    return plan


def dataframe_hash(frame: pd.DataFrame) -> str:
    """Hash a schedule table deterministically."""

    csv = frame.to_csv(index=False, lineterminator="\n")
    return "sha256:" + hashlib.sha256(csv.encode("utf-8")).hexdigest()


def child_seed(*parts: object) -> int:
    """Derive a deterministic 32-bit child seed from stable schedule fields."""

    payload = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def _validate_registry_header(registry: dict[str, Any]) -> None:
    if registry.get("id") != "mechanistic_identifiability_v1":
        raise ConfigValidationError("registry.id must be mechanistic_identifiability_v1")
    if registry.get("status") != "pre_outcome_smoke_design":
        raise ConfigValidationError("registry.status must be pre_outcome_smoke_design")
    commit = str(registry.get("variable_registry_commit", ""))
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ConfigValidationError("registry.variable_registry_commit must be a lowercase SHA")


def _validate_claim_boundaries(registry: dict[str, Any], design: dict[str, Any]) -> None:
    for field in (
        "formal_claims_allowed",
        "real_transfer_outcomes_allowed",
        "trident_validation_claim_allowed",
        "neural_criticality_claim_allowed",
        "cusp_required",
    ):
        if registry.get(field) is not False:
            raise ConfigValidationError(f"registry.{field} must be false")
    if design.get("real_wrapper_transfer_outcomes_inspected") is not False:
        raise ConfigValidationError("real wrapper-transfer outcomes must remain uninspected")
    if design.get("participant_isolated_split_required") is not True:
        raise ConfigValidationError("participant-isolated splitting must be required")
    if design.get("truth_columns_stripped_before_scoring_required") is not True:
        raise ConfigValidationError("truth-column stripping must be required")
    if design.get("pace_status") != "descriptive_until_independently_supported":
        raise ConfigValidationError("PACE must remain descriptive until independently supported")
    literal_cusp_status = str(design.get("literal_cusp_status", ""))
    if literal_cusp_status != "later_registered_competitor_only":
        raise ConfigValidationError("literal cusp must be a later registered competitor only")
    _reject_forbidden_primary_terms(design)


def _validate_primary_family_set(design: dict[str, Any]) -> None:
    family_ids = design.get("primary_causal_family_ids")
    if tuple(family_ids or ()) != CAUSAL_FAMILY_IDS:
        raise ConfigValidationError("primary_causal_family_ids must match the M3 registry order")


def _validate_seed_streams(smoke: dict[str, Any]) -> None:
    streams = tuple(str(item) for item in smoke.get("seed_streams", ()))
    if streams != REQUIRED_SEED_STREAMS:
        raise ConfigValidationError("seed_streams must be structural/nuisance/observation/split/model")
    if int(smoke.get("replicates_per_truth_family_per_gate", 0)) < 1:
        raise ConfigValidationError("replicates_per_truth_family_per_gate must be positive")
    if int(smoke.get("participants_per_replicate", 0)) < 2:
        raise ConfigValidationError("participants_per_replicate must be at least 2")
    holdout = float(smoke.get("participant_holdout_fraction", -1.0))
    if not 0.0 < holdout < 1.0:
        raise ConfigValidationError("participant_holdout_fraction must be between 0 and 1")


def _validate_scoring_families(
    scoring_families: list[Any],
    variable_registry: VariableRegistry,
) -> None:
    for index, record in enumerate(scoring_families):
        if not isinstance(record, dict):
            raise ConfigValidationError(f"candidate_scoring_families[{index}] must be a mapping")
        family_id = str(record.get("id", ""))
        _reject_forbidden_primary_terms(record)
        tests_against = record.get("tests_against")
        if not isinstance(tests_against, list) or not tests_against:
            raise ConfigValidationError(f"{family_id} tests_against must be a non-empty list")
        unknown = set(str(item) for item in tests_against).difference(variable_registry.causal_families)
        if unknown:
            raise ConfigValidationError(
                f"{family_id} references unknown truth families: " + ", ".join(sorted(unknown))
            )


def _validate_gates(gates: list[Any], variable_registry: VariableRegistry) -> None:
    registry_gates = {
        str(gate["id"]): tuple(str(item) for item in gate["truth_families"])
        for gate in variable_registry.identifiability_gates
    }
    for index, gate in enumerate(gates):
        if not isinstance(gate, dict):
            raise ConfigValidationError(f"identifiability_gates[{index}] must be a mapping")
        gate_id = str(gate.get("id", ""))
        if gate_id not in registry_gates:
            raise ConfigValidationError(f"{gate_id} is not registered in the M3 variable registry")
        truth_families = tuple(str(item) for item in gate.get("truth_families", ()))
        if truth_families != registry_gates[gate_id]:
            raise ConfigValidationError(f"{gate_id} truth families must match the M3 registry")
        variables = gate.get("primary_variables")
        if not isinstance(variables, list) or not variables:
            raise ConfigValidationError(f"{gate_id}.primary_variables must be a non-empty list")
        unknown_variables = set(str(item) for item in variables).difference(REQUIRED_VARIABLE_IDS)
        if unknown_variables:
            raise ConfigValidationError(
                f"{gate_id} references unknown variables: " + ", ".join(sorted(unknown_variables))
            )
        if gate.get("decision_status") != "pre_outcome":
            raise ConfigValidationError(f"{gate_id}.decision_status must be pre_outcome")


def _reject_forbidden_primary_terms(record: dict[str, Any]) -> None:
    rendered = repr(record).lower()
    for term in FORBIDDEN_PRIMARY_TERMS:
        if term in rendered and "later_registered_competitor_only" not in rendered:
            raise ConfigValidationError(f"primary M2.8 design contains forbidden term: {term}")


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict) or not value:
        raise ConfigValidationError(f"{key} must be a non-empty mapping")
    return value


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for materialising the pre-outcome M2.8 design."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/mechanistic_identifiability_v1.yaml",
        help="Path to the M2.8 mechanistic identifiability config.",
    )
    args = parser.parse_args(argv)
    plan = write_identifiability_plan_outputs(args.config)
    print(
        json.dumps(
            {
                "study_id": plan.registry_id,
                "status": plan.status,
                "n_units": int(plan.schedule.shape[0]),
                "schedule_hash": plan.schedule_hash,
                "formal_claims_allowed": False,
                "real_transfer_outcomes_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
