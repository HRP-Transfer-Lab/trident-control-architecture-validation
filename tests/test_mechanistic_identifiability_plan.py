from pathlib import Path
import uuid

import pytest
import yaml

from trident_validation.config import ConfigValidationError, load_yaml_config
from trident_validation.mechanistic.identifiability_plan import (
    REQUIRED_SEED_STREAMS,
    build_identifiability_schedule,
    dataframe_hash,
    load_mechanistic_identifiability_plan,
    validate_mechanistic_identifiability_plan,
    write_identifiability_plan_outputs,
)
from trident_validation.mechanistic.variable_registry import CAUSAL_FAMILY_IDS, load_variable_registry


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "config/mechanistic_identifiability_v1.yaml"
REGISTRY_PATH = ROOT / "config/hrp_stack_variable_registry_v2.yaml"


def _scratch_dir() -> Path:
    path = ROOT / "reports/generated/test_mechanistic_identifiability_plan" / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_mechanistic_identifiability_plan_validates_and_builds_schedule():
    plan = load_mechanistic_identifiability_plan(PLAN_PATH, repo_root=ROOT)

    assert plan.registry_id == "mechanistic_identifiability_v1"
    assert plan.status == "pre_outcome_smoke_design"
    assert plan.variable_registry_commit == "4073c76357954a386b30d2cad792e4c6bb4eb393"
    assert plan.schedule.shape[0] == 24
    assert plan.schedule["task_index"].nunique() == 24
    assert sorted(plan.schedule["task_index"].tolist()) == list(range(24))
    assert plan.schedule["real_wrapper_transfer_outcomes_inspected"].eq(False).all()
    assert plan.schedule["truth_columns_stripped_before_scoring_required"].eq(True).all()
    assert plan.schedule["transfer_external_use"].eq("synthetic_heldout_diagnostic_only").all()
    assert dataframe_hash(plan.schedule) == plan.schedule_hash


def test_schedule_is_deterministic_and_uses_independent_seed_streams():
    config = load_yaml_config(PLAN_PATH)
    registry = load_variable_registry(REGISTRY_PATH)

    first = build_identifiability_schedule(config, variable_registry=registry)
    second = build_identifiability_schedule(config, variable_registry=registry)

    assert dataframe_hash(first) == dataframe_hash(second)
    for stream in REQUIRED_SEED_STREAMS:
        assert f"{stream}_seed" in first.columns
    seed_columns = [f"{stream}_seed" for stream in REQUIRED_SEED_STREAMS]
    assert first[seed_columns].nunique(axis=1).eq(len(seed_columns)).all()


def test_plan_consumes_frozen_registry_families_and_gates():
    plan = load_mechanistic_identifiability_plan(PLAN_PATH, repo_root=ROOT)
    config = load_yaml_config(PLAN_PATH)
    registry = load_variable_registry(REGISTRY_PATH)

    assert tuple(config["primary_design"]["primary_causal_family_ids"]) == CAUSAL_FAMILY_IDS
    assert set(plan.schedule["truth_family_id"]).issubset(registry.causal_families)
    registry_gate_ids = {gate["id"] for gate in registry.identifiability_gates}
    assert set(config_gate["id"] for config_gate in config["identifiability_gates"]) == registry_gate_ids


def test_real_transfer_outcome_inspection_is_rejected():
    config = load_yaml_config(PLAN_PATH)
    registry = load_variable_registry(REGISTRY_PATH)
    config["primary_design"]["real_wrapper_transfer_outcomes_inspected"] = True

    with pytest.raises(ConfigValidationError, match="uninspected"):
        validate_mechanistic_identifiability_plan(
            config,
            variable_registry=registry,
            variable_registry_path=REGISTRY_PATH,
        )


def test_primary_literal_cusp_candidate_is_rejected():
    config = load_yaml_config(PLAN_PATH)
    registry = load_variable_registry(REGISTRY_PATH)
    config["candidate_scoring_families"].append(
        {
            "id": "SCORE8_literal_cusp_bifurcation",
            "status": "placeholder_not_implemented",
            "tests_against": ["MECH7"],
        }
    )

    with pytest.raises(ConfigValidationError, match="forbidden term"):
        validate_mechanistic_identifiability_plan(
            config,
            variable_registry=registry,
            variable_registry_path=REGISTRY_PATH,
        )


def test_gate_truth_families_must_match_registry():
    config = load_yaml_config(PLAN_PATH)
    registry = load_variable_registry(REGISTRY_PATH)
    config["identifiability_gates"][0]["truth_families"] = ["MECH0", "MECH2"]

    with pytest.raises(ConfigValidationError, match="truth families must match"):
        validate_mechanistic_identifiability_plan(
            config,
            variable_registry=registry,
            variable_registry_path=REGISTRY_PATH,
        )


def test_plan_writer_materialises_schedule_and_manifest():
    scratch = _scratch_dir()
    config = load_yaml_config(PLAN_PATH)
    config["outputs"]["schedule_csv"] = str(scratch / "schedule.csv")
    config["outputs"]["schedule_json"] = str(scratch / "schedule.json")
    config["outputs"]["manifest_json"] = str(scratch / "manifest.json")
    config_path = scratch / "mechanistic_identifiability_v1.yaml"

    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    plan = write_identifiability_plan_outputs(config_path, repo_root=ROOT)

    assert (scratch / "schedule.csv").exists()
    assert (scratch / "schedule.json").exists()
    assert (scratch / "manifest.json").exists()
    assert plan.schedule_hash == dataframe_hash(plan.schedule)
