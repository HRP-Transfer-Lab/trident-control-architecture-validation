from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from trident_validation.config import ConfigValidationError, load_yaml_config
from trident_validation.mechanistic.hcp_transversal_sensitivity import (
    _fold_features,
    run_hcp_transversal_sensitivity,
    validate_hcp_transversal_sensitivity_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/hcp_ya_transversal_sensitivity_v1.yaml"


def test_hcp_sensitivity_config_is_claim_bounded_and_post_m6_1():
    config = load_yaml_config(CONFIG_PATH)
    validate_hcp_transversal_sensitivity_config(config)

    assert config["analysis"]["formulated_after_m6_1_outcome_inspection"] is True
    assert config["analysis"]["changes_frozen_m6_1_analysis"] is False
    assert config["analysis"]["confirmed_g_claim_allowed"] is False
    assert config["analysis"]["representational_operator_claim_allowed"] is False
    assert config["analysis"]["strategic_policy_claim_allowed"] is False
    assert config["inputs"]["participant_level_predictions_written"] is False
    assert config["inputs"]["participant_level_scores_written"] is False
    assert config["validation"]["split_strategy"] == "participant_isolated_official_hcp_100_unrelated"


def test_hcp_sensitivity_rejects_forbidden_k_overlap():
    config = load_yaml_config(CONFIG_PATH)
    config["k_sensitivity_variants"][0]["columns"].append("CardSort_Unadj")

    with pytest.raises(ConfigValidationError, match="forbidden columns"):
        validate_hcp_transversal_sensitivity_config(config)


def test_hcp_sensitivity_rejects_using_blocked_outcomes():
    config = load_yaml_config(CONFIG_PATH)
    config["outcome_domains"]["wm_list_sorting"]["columns"] = ["WM_Task_2bk_Acc"]

    with pytest.raises(ConfigValidationError, match="forbidden diagnostic/blocked"):
        validate_hcp_transversal_sensitivity_config(config)


def test_hcp_sensitivity_rejects_confirmatory_boundary():
    config = load_yaml_config(CONFIG_PATH)
    config["interpretation_boundaries"]["tests_binding_operator"] = True

    with pytest.raises(ConfigValidationError, match="tests_binding_operator"):
        validate_hcp_transversal_sensitivity_config(config)


def test_hcp_sensitivity_variant_scaling_uses_training_fold_only():
    config = load_yaml_config(CONFIG_PATH)
    variant = config["k_sensitivity_variants"][0]
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

    train_features, test_features = _fold_features(train, test, config, variant)

    train_mean = np.array([0.0, 1.0, 2.0, 3.0]).mean()
    train_sd = np.array([0.0, 1.0, 2.0, 3.0]).std(ddof=0)
    assert np.isclose(train_features["K"].iloc[0], (0.0 - train_mean) / train_sd)
    assert np.isclose(test_features["K"].iloc[0], (1000.0 - train_mean) / train_sd)


def test_hcp_sensitivity_runs_on_mock_data_and_writes_participant_free_outputs():
    work_dir = ROOT / "reports" / "generated" / "test_hcp_transversal_sensitivity"
    work_dir.mkdir(parents=True, exist_ok=True)
    data_path = work_dir / "mock_extract.csv"
    _mock_hcp_extract(n=100).to_csv(data_path, index=False)
    config = _configured_for_work_dir(work_dir, data_path)
    config_path = work_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = run_hcp_transversal_sensitivity(
        config_path,
        repo_root=ROOT,
        output_dir=work_dir / "out",
        write_outputs=True,
    )

    assert result.summary["status"] == "m6_1b_sensitivity_complete_exploratory_no_confirmatory_claim"
    assert result.summary["n_subjects"] == 100
    assert result.summary["wm_task_2bk_acc_used"] is False
    assert result.summary["relational_task_acc_used"] is False
    assert result.summary["pmat_rt_used"] is False
    assert set(result.k_robustness_summary["k_variant"]) == {
        "registered_full_k",
        "no_proc_speed",
        "no_picseq",
        "no_readeng",
        "no_picvocab",
        "language_crystallized_pair",
        "speed_sequence_pair",
    }
    assert set(result.demand_pattern_summary["contrast"]) == {
        "C_increment_beyond_registered_K",
        "V_increment_beyond_registered_K",
        "joint_CV_increment_beyond_registered_K",
    }
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


def _configured_for_work_dir(work_dir: Path, data_path: Path) -> dict:
    config = load_yaml_config(CONFIG_PATH)
    config["inputs"]["hcp_extract_path"] = str(data_path)
    config["outputs"]["report_md"] = str(work_dir / "report.md")
    config["outputs"]["participant_free_summary_json"] = str(work_dir / "summary.json")
    config["outputs"]["manifest_json"] = str(work_dir / "manifest.json")
    return config


def _mock_hcp_extract(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(314159)
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
            "ListSort_Unadj": 0.5 * base + 0.2 * control + rng.normal(scale=0.8, size=n),
            "PMAT24_A_CR": 0.6 * base + 0.1 * vigilance + rng.normal(scale=0.8, size=n),
            "PMAT24_A_RTCR": rng.normal(size=n),
            "WM_Task_2bk_Acc": np.nan,
            "Relational_Task_Acc": np.nan,
        }
    )
