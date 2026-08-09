from pathlib import Path

import pytest

from trident_validation.config import ConfigValidationError, load_yaml_config
from trident_validation.mechanistic.public_ladder import (
    STAGE_IDS,
    load_public_mechanism_test_ladder,
    validate_public_mechanism_test_ladder,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/public_mechanism_test_ladder_v1.yaml"


def test_public_mechanism_test_ladder_validates_staged_order():
    ladder = load_public_mechanism_test_ladder(CONFIG_PATH, repo_root=ROOT)

    assert ladder.registry_id == "public_mechanism_test_ladder_v1"
    assert ladder.status == "pre_analysis_protocol"
    assert tuple(stage["id"] for stage in ladder.stages) == STAGE_IDS
    assert ladder.stages[0]["status"] == "next_real_data_analysis_candidate"
    assert ladder.stages[1]["status"] == "gated_after_stage1_review"
    assert ladder.stages[2]["status"] == "gated_after_stage2_review"
    assert ladder.stages[3]["status"] == "gated_after_stage3_review"


def test_stage1_is_k_c_signal_v_only_before_pace_or_dynamics():
    ladder = load_public_mechanism_test_ladder(CONFIG_PATH, repo_root=ROOT)
    stage1 = ladder.stages[0]

    assert set(stage1["variables"]["primary"]) == {"K", "C_signal", "V"}
    assert {"P_pace", "R_dynamic", "Transfer_external"}.issubset(
        set(stage1["variables"]["excluded"])
    )
    assert ladder.support_status_by_variable["K"] == "supported"
    assert ladder.support_status_by_variable["C_signal"] == "partial"
    assert ladder.support_status_by_variable["V"] == "partial"


def test_pace_dynamic_and_criticality_are_gated():
    ladder = load_public_mechanism_test_ladder(CONFIG_PATH, repo_root=ROOT)
    stage2, stage3, stage4 = ladder.stages[1:]

    assert stage2["status"] == "gated_after_stage1_review"
    assert "no_forced_four_profile_claim" in stage2["proceed_if"]
    assert stage3["status"] == "gated_after_stage2_review"
    assert "regime_labels_are_neutral" in stage3["proceed_if"]
    assert stage4["status"] == "gated_after_stage3_review"
    assert "neural_criticality" in stage4["forbidden_interpretations"]
    assert "claims_are_limited_to_behavioural_dynamic_analogue" in stage4["proceed_if"]


def test_ladder_rejects_starting_with_pace_in_stage1():
    config = load_yaml_config(CONFIG_PATH)
    config["stages"][0]["variables"]["primary"].append("P_pace")

    with pytest.raises(ConfigValidationError, match="stage 1 primary variables"):
        validate_public_mechanism_test_ladder(config, repo_root=ROOT)


def test_ladder_rejects_neural_criticality_claim_permission():
    config = load_yaml_config(CONFIG_PATH)
    config["registry"]["neural_criticality_claim_allowed"] = True

    with pytest.raises(ConfigValidationError, match="neural_criticality_claim_allowed"):
        validate_public_mechanism_test_ladder(config, repo_root=ROOT)
