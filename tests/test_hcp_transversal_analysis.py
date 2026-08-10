from pathlib import Path
import json

import pytest
import yaml

from trident_validation.config import ConfigValidationError, load_yaml_config
from trident_validation.mechanistic.hcp_transversal_analysis import (
    run_hcp_transversal_analysis_plan,
    validate_hcp_transversal_analysis_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/hcp_ya_transversal_analysis_v1.yaml"


def test_hcp_transversal_analysis_config_is_locked_and_claim_bounded():
    config = load_yaml_config(CONFIG_PATH)
    validate_hcp_transversal_analysis_config(config)

    assert config["analysis"]["model_fitting_enabled"] is False
    assert config["analysis"]["requires_later_analysis_freeze_commit"] is True
    assert config["analysis"]["trident_validation_claim_allowed"] is False
    assert config["analysis"]["predictive_calibration_claim_allowed"] is False
    assert config["inputs"]["family_structure_policy"]["ordinary_participant_folds_allowed"] is False
    assert config["validation"]["all_scaling_inside_training_folds"] is True
    assert config["validation"]["outcome_specific_K_exclusion_required"] is True


def test_hcp_transversal_analysis_rejects_enabled_model_fitting():
    config = load_yaml_config(CONFIG_PATH)
    config["analysis"]["model_fitting_enabled"] = True

    with pytest.raises(ConfigValidationError, match="model_fitting_enabled"):
        validate_hcp_transversal_analysis_config(config)


def test_hcp_transversal_analysis_rejects_outcome_overlap_with_c_or_v():
    config = load_yaml_config(CONFIG_PATH)
    config["outcome_domains"]["wm_list_sorting"]["columns"] = ["SCPT_SPEC"]

    with pytest.raises(ConfigValidationError, match="overlaps with C/V"):
        validate_hcp_transversal_analysis_config(config)


def test_hcp_transversal_analysis_rejects_global_or_overlap_k_columns():
    config = load_yaml_config(CONFIG_PATH)
    config["coordinate_sources"]["K"]["columns"].append("CogTotalComp_Unadj")

    with pytest.raises(ConfigValidationError, match="forbidden HCP overlap"):
        validate_hcp_transversal_analysis_config(config)


def test_hcp_transversal_analysis_plan_blocks_without_passed_preflight():
    work_dir = ROOT / "reports" / "generated" / "test_hcp_transversal_analysis_blocked"
    work_dir.mkdir(parents=True, exist_ok=True)
    config = load_yaml_config(CONFIG_PATH)
    config["analysis"]["current_preflight_summary_path"] = str(work_dir / "missing_preflight.json")
    config["outputs"]["plan_report_md"] = str(work_dir / "plan.md")
    config_path = work_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = run_hcp_transversal_analysis_plan(
        config_path,
        repo_root=ROOT,
        output_dir=work_dir / "out",
        write_outputs=True,
    )

    assert result.summary["status"] == "blocked_no_preflight_summary"
    assert result.summary["model_fitting_allowed_now"] is False
    assert (work_dir / "out" / "analysis_plan_summary.json").exists()
    assert (work_dir / "plan.md").exists()


def test_hcp_transversal_analysis_plan_ready_after_mock_passed_preflight():
    work_dir = ROOT / "reports" / "generated" / "test_hcp_transversal_analysis_ready"
    work_dir.mkdir(parents=True, exist_ok=True)
    preflight = {
        "analysis_id": "hcp_ya_transversal_v1",
        "status": "data_support_passed_ready_to_freeze_analysis",
        "support_passed": True,
    }
    preflight_path = work_dir / "preflight_summary.json"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    config = load_yaml_config(CONFIG_PATH)
    config["analysis"]["current_preflight_summary_path"] = str(preflight_path)
    config["outputs"]["plan_report_md"] = str(work_dir / "plan.md")
    config_path = work_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = run_hcp_transversal_analysis_plan(
        config_path,
        repo_root=ROOT,
        output_dir=work_dir / "out",
        write_outputs=False,
    )
    domain_plan = result.domain_plan.set_index("domain")

    assert result.summary["status"] == "ready_to_freeze_analysis_config"
    assert result.summary["model_fitting_allowed_now"] is False
    assert result.summary["requires_later_analysis_freeze_commit"] is True
    assert "ListSort_Unadj" not in domain_plan.loc["wm_list_sorting", "k_columns_after_exclusion"].split("|")
    assert "PMAT24_A_CR" not in domain_plan.loc["reasoning_pmat", "k_columns_after_exclusion"].split("|")
    assert "Flanker_Unadj" not in domain_plan.loc["wm_list_sorting", "k_columns_after_exclusion"].split("|")
    assert "CardSort_Unadj" not in domain_plan.loc["wm_list_sorting", "k_columns_after_exclusion"].split("|")
    assert bool(domain_plan.loc["reasoning_relational", "required_for_support"]) is False
    assert set(result.model_plan["model_id"]) == {
        "K",
        "K_plus_C_candidate",
        "K_plus_V",
        "K_plus_C_candidate_plus_V",
        "K_plus_C_candidate_plus_V_plus_C_by_V",
    }
    assert result.layer_specific_plan["outcome_reuse_allowed"].eq(False).all()
