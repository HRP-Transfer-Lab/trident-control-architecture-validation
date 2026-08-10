from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from trident_validation.config import ConfigValidationError, load_yaml_config
from trident_validation.mechanistic.hcp_transversal import (
    run_hcp_transversal_preflight,
    validate_hcp_transversal_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/hcp_ya_transversal_v1.yaml"


def test_hcp_transversal_config_is_claim_bounded_and_preflight_only():
    config = load_yaml_config(CONFIG_PATH)
    validate_hcp_transversal_config(config)

    assert config["analysis"]["formal_claims_allowed"] is False
    assert config["analysis"]["trident_validation_claim_allowed"] is False
    assert config["analysis"]["predictive_calibration_claim_allowed"] is False
    assert config["analysis"]["t_commit_claim_allowed"] is False
    assert config["decision_boundary"]["model_fitting_allowed_by_this_config"] is False
    assert config["inputs"]["cohort_mode"] == "hcp_100_unrelated"
    assert config["inputs"]["family_id_column"] is None
    assert config["inputs"]["participant_isolated_cv_allowed"] is True
    assert config["inputs"]["family_isolated_cv_required"] is False
    assert config["inputs"]["family_structure_policy"]["ordinary_participant_folds_allowed"] is False


def test_hcp_transversal_rejects_outcome_overlap_with_c_or_v():
    config = load_yaml_config(CONFIG_PATH)
    config["outcome_domains"]["reasoning_pmat"]["columns"] = ["SCPT_SEN"]

    with pytest.raises(ConfigValidationError, match="overlaps with C/V"):
        validate_hcp_transversal_config(config)


def test_hcp_transversal_rejects_flanker_or_card_sort_inside_k():
    config = load_yaml_config(CONFIG_PATH)
    config["predictor_sources"]["K"]["columns"].append("Flanker_Unadj")

    with pytest.raises(ConfigValidationError, match="forbidden HCP overlap"):
        validate_hcp_transversal_config(config)


def test_hcp_transversal_removes_outcome_columns_from_domain_k():
    config = load_yaml_config(CONFIG_PATH)
    df = _mock_hcp_extract(n=160)
    work_dir = ROOT / "reports" / "generated" / "test_hcp_transversal"
    work_dir.mkdir(parents=True, exist_ok=True)
    data_path = work_dir / "hcp_extract.csv"
    df.to_csv(data_path, index=False)
    config["inputs"]["hcp_extract_path"] = str(data_path)
    config["inputs"]["hcp_extract_sha256"] = None
    config["outputs"]["preflight_report_md"] = str(work_dir / "preflight.md")
    config_path = work_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = run_hcp_transversal_preflight(
        config_path,
        repo_root=ROOT,
        output_dir=work_dir / "out",
        write_outputs=True,
    )
    domain_support = result.domain_support.set_index("domain")

    assert result.summary["support_passed"] is True
    assert "ListSort_Unadj" not in domain_support.loc["wm_list_sorting", "k_columns_after_exclusion"].split("|")
    assert "PMAT24_A_CR" not in domain_support.loc["reasoning_pmat", "k_columns_after_exclusion"].split("|")
    assert "Flanker_Unadj" not in domain_support.loc["wm_list_sorting", "k_columns_after_exclusion"].split("|")
    assert "CardSort_Unadj" not in domain_support.loc["wm_list_sorting", "k_columns_after_exclusion"].split("|")
    assert bool(domain_support.loc["reasoning_relational", "required_for_support"]) is False
    assert bool(domain_support.loc["reasoning_relational", "support_passed"]) is False
    assert bool(domain_support.loc["wm_nback", "required_for_support"]) is False
    assert bool(domain_support.loc["wm_nback", "analysis_eligible"]) is False
    assert bool(domain_support.loc["wm_nback", "support_passed"]) is False
    assert result.split_support["family_isolated_cv_feasible"] is False
    assert result.split_support["unrelated_only_feasible"] is True
    assert result.split_support["participant_isolated_cv_allowed"] is True
    assert result.split_support["split_strategy"] == "participant_isolated_official_hcp_100_unrelated"
    assert result.split_support["ordinary_participant_folds_allowed"] is False
    assert "participant_id" not in result.column_support.columns
    assert (work_dir / "out" / "preflight_summary.json").exists()


def test_hcp_transversal_unrelated_only_allows_missing_family_id_for_valid_mock_extract():
    config = load_yaml_config(CONFIG_PATH)
    df = _mock_hcp_extract(n=100).drop(columns=["Family_ID"])
    work_dir = ROOT / "reports" / "generated" / "test_hcp_transversal_100_unrelated"
    work_dir.mkdir(parents=True, exist_ok=True)
    data_path = work_dir / "hcp_extract.csv"
    df.to_csv(data_path, index=False)
    config["inputs"]["hcp_extract_path"] = str(data_path)
    config["inputs"]["hcp_extract_sha256"] = None
    config["outputs"]["preflight_report_md"] = str(work_dir / "preflight.md")
    config_path = work_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = run_hcp_transversal_preflight(
        config_path,
        repo_root=ROOT,
        output_dir=work_dir / "out",
        write_outputs=True,
    )

    assert result.summary["support_passed"] is True
    assert result.summary["model_fitting_allowed"] is False
    assert result.summary["input_unique_subjects"] == 100
    assert result.split_support["n_unrelated_participants"] == 100
    assert result.split_support["family_id_available"] is False
    assert result.split_support["participant_isolated_cv_allowed"] is True


def test_hcp_transversal_family_mode_without_family_id_still_fails():
    config = load_yaml_config(CONFIG_PATH)
    config["inputs"]["cohort_mode"] = "family"
    config["inputs"]["family_id_column"] = "MissingFamily"
    config["inputs"]["unrelated_indicator_column"] = "MissingUnrelated"
    config["inputs"]["participant_isolated_cv_allowed"] = False
    config["inputs"]["family_isolated_cv_required"] = True
    df = _mock_hcp_extract(n=160)
    work_dir = ROOT / "reports" / "generated" / "test_hcp_transversal_no_family"
    work_dir.mkdir(parents=True, exist_ok=True)
    data_path = work_dir / "hcp_extract.csv"
    df.to_csv(data_path, index=False)
    config["inputs"]["hcp_extract_path"] = str(data_path)
    config["inputs"]["hcp_extract_sha256"] = None
    config_path = work_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = run_hcp_transversal_preflight(
        config_path,
        repo_root=ROOT,
        output_dir=work_dir / "out",
        write_outputs=False,
    )

    assert result.summary["support_passed"] is False
    assert result.split_support["family_isolated_cv_feasible"] is False
    assert result.split_support["unrelated_only_feasible"] is False
    assert result.split_support["participant_isolated_cv_allowed"] is False
    assert result.split_support["ordinary_participant_folds_allowed"] is False


def test_hcp_transversal_reports_missing_data_file_without_fitting():
    config = load_yaml_config(CONFIG_PATH)
    work_dir = ROOT / "reports" / "generated" / "test_hcp_transversal_missing"
    work_dir.mkdir(parents=True, exist_ok=True)
    config["inputs"]["hcp_extract_path"] = str(work_dir / "missing.csv")
    config["outputs"]["preflight_report_md"] = str(work_dir / "preflight.md")
    config_path = work_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = run_hcp_transversal_preflight(
        config_path,
        repo_root=ROOT,
        output_dir=work_dir / "out",
        write_outputs=True,
    )

    assert result.summary["status"] == "data_file_missing"
    assert result.summary["model_fitting_allowed"] is False
    assert result.column_support.empty
    assert (work_dir / "out" / "preflight_summary.json").exists()


def _mock_hcp_extract(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(31415)
    return pd.DataFrame(
        {
            "Subject": [f"S{i:04d}" for i in range(n)],
            "Family_ID": [f"F{i // 2:04d}" for i in range(n)],
            "Flanker_Unadj": rng.normal(size=n),
            "ProcSpeed_Unadj": rng.normal(size=n),
            "CardSort_Unadj": rng.normal(size=n),
            "PicSeq_Unadj": rng.normal(size=n),
            "ReadEng_Unadj": rng.normal(size=n),
            "PicVocab_Unadj": rng.normal(size=n),
            "SCPT_SEN": rng.normal(size=n),
            "SCPT_SPEC": rng.normal(size=n),
            "ListSort_Unadj": rng.normal(size=n),
            "PMAT24_A_CR": rng.normal(size=n),
            "PMAT24_A_RTCR": rng.normal(size=n),
        }
    )
