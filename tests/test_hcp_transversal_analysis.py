from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from trident_validation.config import ConfigValidationError, load_yaml_config
from trident_validation.mechanistic.hcp_transversal_analysis import (
    _fold_constructs,
    make_participant_folds,
    run_hcp_transversal_analysis,
    run_hcp_transversal_analysis_plan,
    validate_hcp_transversal_analysis_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/hcp_ya_transversal_analysis_v1.yaml"


def test_hcp_transversal_analysis_config_is_frozen_and_claim_bounded():
    config = load_yaml_config(CONFIG_PATH)
    validate_hcp_transversal_analysis_config(config)

    assert config["analysis"]["model_fitting_enabled"] is True
    assert config["analysis"]["requires_pre_outcome_freeze_commit"] is True
    assert config["analysis"]["trident_validation_claim_allowed"] is False
    assert config["analysis"]["confirmed_g_claim_allowed"] is False
    assert config["inputs"]["cohort_mode"] == "hcp_100_unrelated"
    assert config["inputs"]["family_id_column"] is None
    assert config["inputs"]["family_structure_policy"]["ordinary_participant_folds_allowed"] is False
    assert config["validation"]["all_scaling_inside_training_folds"] is True
    assert config["validation"]["split_strategy"] == "participant_isolated_official_hcp_100_unrelated"
    assert config["validation"]["bootstrap_iterations"] == 1000
    assert config["m6_2_layer_specific_capacity_gate"]["status"] == "blocked_pending_independent_second_WM_indicator"
    assert config["m6_3_bottleneck_gate"]["k_by_w_tests_allowed"] is False


def test_hcp_transversal_analysis_rejects_disabled_model_fitting_after_freeze():
    config = load_yaml_config(CONFIG_PATH)
    config["analysis"]["model_fitting_enabled"] = False

    with pytest.raises(ConfigValidationError, match="model_fitting_enabled"):
        validate_hcp_transversal_analysis_config(config)


def test_hcp_transversal_analysis_rejects_outcome_overlap_with_constructs():
    config = load_yaml_config(CONFIG_PATH)
    config["outcome_domains"]["wm_list_sorting"]["columns"] = ["SCPT_SPEC"]

    with pytest.raises(ConfigValidationError, match="overlaps with C/V"):
        validate_hcp_transversal_analysis_config(config)


def test_hcp_transversal_analysis_rejects_global_or_overlap_k_columns():
    config = load_yaml_config(CONFIG_PATH)
    config["constructs"]["K_candidate"]["columns"].append("CogTotalComp_Unadj")

    with pytest.raises(ConfigValidationError, match="frozen source variables"):
        validate_hcp_transversal_analysis_config(config)


def test_hcp_transversal_analysis_rejects_changed_model_or_contrast_set():
    config = load_yaml_config(CONFIG_PATH)
    config["model_set"].append({"id": "extra", "predictors": []})

    with pytest.raises(ConfigValidationError, match="model_set"):
        validate_hcp_transversal_analysis_config(config)

    config = load_yaml_config(CONFIG_PATH)
    config["registered_contrasts"]["interaction_increment"]["status"] = "primary"
    with pytest.raises(ConfigValidationError, match="interaction increment"):
        validate_hcp_transversal_analysis_config(config)


def test_hcp_transversal_analysis_plan_ready_after_mock_passed_preflight():
    work_dir = ROOT / "reports" / "generated" / "test_hcp_transversal_analysis_plan"
    work_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = _write_mock_preflight(work_dir)
    config = _configured_for_work_dir(work_dir, preflight_path)
    config_path = work_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = run_hcp_transversal_analysis_plan(
        config_path,
        repo_root=ROOT,
        output_dir=work_dir / "out",
        write_outputs=True,
    )

    assert result.summary["status"] == "ready_to_run_frozen_analysis"
    assert result.summary["model_fitting_enabled"] is True
    assert result.summary["model_fitting_allowed_now"] is False
    assert "WM_Task_2bk_Acc" not in result.domain_plan.loc[
        result.domain_plan["domain"] == "wm_list_sorting", "k_columns_after_exclusion"
    ].iloc[0]
    assert "wm_nback" not in set(result.model_plan["domain"])
    assert "reasoning_relational" not in set(result.model_plan["domain"])
    assert (work_dir / "plan.md").exists()


def test_make_participant_folds_isolates_subjects():
    subjects = pd.Series([f"S{i:04d}" for i in range(100)])
    folds = make_participant_folds(subjects, n_folds=5, seed=20260822)

    assert [len(fold) for fold in folds] == [20, 20, 20, 20, 20]
    assert len(set().union(*(set(fold) for fold in folds))) == 100
    for index, fold in enumerate(folds):
        others = set().union(*(set(other) for j, other in enumerate(folds) if j != index))
        assert set(fold).isdisjoint(others)


def test_constructs_use_training_fold_standardisation_only():
    config = load_yaml_config(CONFIG_PATH)
    train = pd.DataFrame(
        {
            "ProcSpeed_Unadj": [0.0, 1.0, 2.0, 3.0],
            "PicSeq_Unadj": [0.0, 1.0, 2.0, 3.0],
            "ReadEng_Unadj": [0.0, 1.0, 2.0, 3.0],
            "PicVocab_Unadj": [0.0, 1.0, 2.0, 3.0],
            "Flanker_Unadj": [0.0, 1.0, 2.0, 3.0],
            "SCPT_SEN": [0.0, 1.0, 2.0, 3.0],
            "SCPT_SPEC": [0.0, 1.0, 2.0, 3.0],
        }
    )
    test = pd.DataFrame(
        {
            "ProcSpeed_Unadj": [1000.0],
            "PicSeq_Unadj": [1000.0],
            "ReadEng_Unadj": [1000.0],
            "PicVocab_Unadj": [1000.0],
            "Flanker_Unadj": [1000.0],
            "SCPT_SEN": [1000.0],
            "SCPT_SPEC": [1000.0],
        }
    )

    train_features, test_features = _fold_constructs(train, test, config)

    train_mean = np.array([0.0, 1.0, 2.0, 3.0]).mean()
    train_sd = np.array([0.0, 1.0, 2.0, 3.0]).std(ddof=0)
    assert np.isclose(train_features["K"].iloc[0], (0.0 - train_mean) / train_sd)
    assert np.isclose(test_features["K"].iloc[0], (1000.0 - train_mean) / train_sd)


def test_hcp_transversal_analysis_runs_on_mock_data_and_writes_participant_free_outputs():
    work_dir = ROOT / "reports" / "generated" / "test_hcp_transversal_analysis_run"
    work_dir.mkdir(parents=True, exist_ok=True)
    preflight_path = _write_mock_preflight(work_dir)
    data_path = work_dir / "mock_extract.csv"
    _mock_hcp_extract(n=100).to_csv(data_path, index=False)
    config = _configured_for_work_dir(work_dir, preflight_path)
    config["inputs"]["hcp_extract_path"] = str(data_path)
    config_path = work_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = run_hcp_transversal_analysis(
        config_path,
        repo_root=ROOT,
        output_dir=work_dir / "out",
        write_outputs=True,
    )

    assert result.summary["status"] == "m6_1_analysis_complete_exploratory_no_confirmatory_claim"
    assert result.summary["n_subjects"] == 100
    assert set(result.model_summary["model_id"]) == {
        "M0_intercept",
        "M1_K",
        "M2_K_plus_C",
        "M3_K_plus_V",
        "M4_K_plus_C_plus_V",
        "M5_K_plus_C_plus_V_plus_C_by_V",
    }
    assert set(result.contrast_summary["contrast"]) == {
        "K_increment",
        "C_increment_beyond_K",
        "V_increment_beyond_K",
        "joint_CV_increment_beyond_K",
        "interaction_increment",
    }
    assert result.diagnostic_summary["pmat_rt_used_in_reasoning_composite"] is False
    assert result.diagnostic_summary["wm_task_2bk_acc_used"] is False
    assert result.manifest["split"]["fold_sizes"] == [20, 20, 20, 20, 20]
    assert "train_participants" not in result.manifest["split"]
    assert "test_participants" not in result.manifest["split"]

    for path in [
        work_dir / "out" / "model_summary.csv",
        work_dir / "out" / "contrast_summary.csv",
        work_dir / "summary.json",
        work_dir / "manifest.json",
        work_dir / "report.md",
    ]:
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert "S0000" not in text
        assert "S0099" not in text


def _write_mock_preflight(work_dir: Path) -> Path:
    preflight = {
        "analysis_id": "hcp_ya_transversal_v1",
        "status": "data_support_passed_ready_to_freeze_analysis",
        "support_passed": True,
        "cohort_mode": "hcp_100_unrelated",
        "input_unique_subjects": 100,
    }
    preflight_path = work_dir / "preflight_summary.json"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    return preflight_path


def _configured_for_work_dir(work_dir: Path, preflight_path: Path) -> dict:
    config = load_yaml_config(CONFIG_PATH)
    config["analysis"]["current_preflight_summary_path"] = str(preflight_path)
    config["outputs"]["plan_report_md"] = str(work_dir / "plan.md")
    config["outputs"]["analysis_report_md"] = str(work_dir / "report.md")
    config["outputs"]["participant_free_summary_json"] = str(work_dir / "summary.json")
    config["outputs"]["manifest_json"] = str(work_dir / "manifest.json")
    return config


def _mock_hcp_extract(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(271828)
    base = rng.normal(size=n)
    control = rng.normal(size=n)
    vigilance = rng.normal(size=n)
    return pd.DataFrame(
        {
            "Subject": [f"S{i:04d}" for i in range(n)],
            "ProcSpeed_Unadj": base + rng.normal(scale=0.5, size=n),
            "PicSeq_Unadj": base + rng.normal(scale=0.5, size=n),
            "ReadEng_Unadj": base + rng.normal(scale=0.5, size=n),
            "PicVocab_Unadj": base + rng.normal(scale=0.5, size=n),
            "Flanker_Unadj": control + rng.normal(scale=0.5, size=n),
            "SCPT_SEN": vigilance + rng.normal(scale=0.5, size=n),
            "SCPT_SPEC": vigilance + rng.normal(scale=0.5, size=n),
            "CardSort_Unadj": 0.4 * base + 0.5 * control + rng.normal(scale=0.7, size=n),
            "ListSort_Unadj": 0.5 * base + 0.2 * control + 0.1 * vigilance + rng.normal(scale=0.8, size=n),
            "PMAT24_A_CR": 0.6 * base + 0.1 * vigilance + rng.normal(scale=0.8, size=n),
            "PMAT24_A_RTCR": rng.normal(size=n),
            "WM_Task_2bk_Acc": np.nan,
            "Relational_Task_Acc": np.nan,
        }
    )
