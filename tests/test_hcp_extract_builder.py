from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trident_validation.config import ConfigValidationError
from trident_validation.mechanistic.hcp_extract_builder import (
    _assert_participant_output_path_allowed,
    build_hcp_transversal_extract,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "config/hcp_ya_transversal_extract_schema_v1.yaml"


def test_hcp_extract_builder_writes_canonical_subset_and_summary():
    work_dir = ROOT / "reports" / "generated" / "test_hcp_extract_builder"
    work_dir.mkdir(parents=True, exist_ok=True)
    source = work_dir / "authorised_hcp_export.csv"
    output = work_dir / "canonical_extract.csv"
    summary_path = work_dir / "summary.json"
    data = _mock_authorised_hcp_export(n=120)
    data["Extra_Column_Not_Registered"] = np.arange(len(data))
    data.to_csv(source, index=False)

    summary = build_hcp_transversal_extract(
        source,
        schema_path=SCHEMA_PATH,
        repo_root=ROOT,
        output_path=output,
        summary_path=summary_path,
        force=True,
    )
    written = pd.read_csv(output)

    assert summary["model_fitting_allowed"] is False
    assert summary["participant_level_data_in_git_allowed"] is False
    assert summary["n_rows"] == 120
    assert "Extra_Column_Not_Registered" not in written.columns
    assert "Relational_Task_Acc" in written.columns
    assert written["Relational_Task_Acc"].isna().all()
    assert "WM_Task_2bk_Acc" in written.columns
    assert written["WM_Task_2bk_Acc"].isna().all()
    assert summary["missing_optional_columns_filled_empty"] == [
        "WM_Task_2bk_Acc",
        "Relational_Task_Acc",
    ]
    assert summary_path.exists()


def test_hcp_extract_builder_rejects_missing_required_columns():
    work_dir = ROOT / "reports" / "generated" / "test_hcp_extract_builder_missing"
    work_dir.mkdir(parents=True, exist_ok=True)
    source = work_dir / "authorised_hcp_export.csv"
    data = _mock_authorised_hcp_export(n=120).drop(columns=["Flanker_Unadj"])
    data.to_csv(source, index=False)

    with pytest.raises(ConfigValidationError, match="missing required"):
        build_hcp_transversal_extract(
            source,
            schema_path=SCHEMA_PATH,
            repo_root=ROOT,
            output_path=work_dir / "canonical_extract.csv",
            force=True,
        )


def test_hcp_extract_builder_merges_two_files_one_to_one_for_unrelated_mode():
    work_dir = ROOT / "reports" / "generated" / "test_hcp_extract_builder_two_file"
    work_dir.mkdir(parents=True, exist_ok=True)
    main = work_dir / "main.csv"
    vigilance = work_dir / "vigilance.csv"
    output = work_dir / "canonical_extract.csv"
    source = _mock_authorised_hcp_export(n=100).drop(columns=["Family_ID", "Unrelated"])
    source[
        [
            "Subject",
            "ProcSpeed_Unadj",
            "PicSeq_Unadj",
            "ReadEng_Unadj",
            "PicVocab_Unadj",
            "Flanker_Unadj",
            "CardSort_Unadj",
            "ListSort_Unadj",
            "PMAT24_A_CR",
            "PMAT24_A_RTCR",
        ]
    ].to_csv(main, index=False)
    source[["Subject", "SCPT_SEN", "SCPT_SPEC"]].to_csv(vigilance, index=False)

    summary = build_hcp_transversal_extract(
        [main, vigilance],
        schema_path=SCHEMA_PATH,
        repo_root=ROOT,
        output_path=output,
        cohort_mode="hcp_100_unrelated",
        hcp_100_unrelated_provenance_declared=True,
        force=True,
    )
    written = pd.read_csv(output)

    assert summary["cohort_mode"] == "hcp_100_unrelated"
    assert summary["merge_validation"]["status"] == "one_to_one_subject_merge_validated"
    assert summary["merge_validation"]["n_rows"] == 100
    assert summary["n_unique_subjects"] == 100
    assert "SCPT_SEN" in written.columns
    assert "Family_ID" in written.columns
    assert written["Family_ID"].isna().all()


def test_hcp_extract_builder_rejects_subject_set_mismatch_in_two_file_merge():
    work_dir = ROOT / "reports" / "generated" / "test_hcp_extract_builder_mismatch"
    work_dir.mkdir(parents=True, exist_ok=True)
    left = _mock_authorised_hcp_export(n=100).drop(columns=["Family_ID", "Unrelated"])
    right = left[["Subject", "SCPT_SEN", "SCPT_SPEC"]].copy()
    right.loc[0, "Subject"] = "S9999"
    main = work_dir / "main.csv"
    vigilance = work_dir / "vigilance.csv"
    left.drop(columns=["SCPT_SEN", "SCPT_SPEC"]).to_csv(main, index=False)
    right.to_csv(vigilance, index=False)

    with pytest.raises(ConfigValidationError, match="different Subject set"):
        build_hcp_transversal_extract(
            [main, vigilance],
            schema_path=SCHEMA_PATH,
            repo_root=ROOT,
            output_path=work_dir / "canonical_extract.csv",
            cohort_mode="hcp_100_unrelated",
            hcp_100_unrelated_provenance_declared=True,
            force=True,
        )


def test_hcp_extract_builder_rejects_duplicate_subjects_in_two_file_merge():
    work_dir = ROOT / "reports" / "generated" / "test_hcp_extract_builder_duplicates"
    work_dir.mkdir(parents=True, exist_ok=True)
    left = _mock_authorised_hcp_export(n=100).drop(columns=["Family_ID", "Unrelated"])
    right = left[["Subject", "SCPT_SEN", "SCPT_SPEC"]].copy()
    right.loc[1, "Subject"] = right.loc[0, "Subject"]
    main = work_dir / "main.csv"
    vigilance = work_dir / "vigilance.csv"
    left.drop(columns=["SCPT_SEN", "SCPT_SPEC"]).to_csv(main, index=False)
    right.to_csv(vigilance, index=False)

    with pytest.raises(ConfigValidationError, match="duplicate Subject"):
        build_hcp_transversal_extract(
            [main, vigilance],
            schema_path=SCHEMA_PATH,
            repo_root=ROOT,
            output_path=work_dir / "canonical_extract.csv",
            cohort_mode="hcp_100_unrelated",
            hcp_100_unrelated_provenance_declared=True,
            force=True,
        )


def test_hcp_extract_builder_rejects_family_mode_without_family_id():
    work_dir = ROOT / "reports" / "generated" / "test_hcp_extract_builder_family_missing"
    work_dir.mkdir(parents=True, exist_ok=True)
    source = work_dir / "authorised_hcp_export.csv"
    data = _mock_authorised_hcp_export(n=100).drop(columns=["Family_ID"])
    data.to_csv(source, index=False)

    with pytest.raises(ConfigValidationError, match="Family_ID"):
        build_hcp_transversal_extract(
            source,
            schema_path=SCHEMA_PATH,
            repo_root=ROOT,
            output_path=work_dir / "canonical_extract.csv",
            force=True,
        )


def test_participant_level_output_must_be_ignored_not_tracked():
    with pytest.raises(ConfigValidationError, match="tracked path"):
        _assert_participant_output_path_allowed(ROOT, ROOT / "config" / "hcp_ya_transversal_v1.yaml")
    with pytest.raises(ConfigValidationError, match="non-ignored repository path"):
        _assert_participant_output_path_allowed(ROOT, ROOT / "not_ignored_hcp_extract.csv")


def _mock_authorised_hcp_export(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(20260824)
    return pd.DataFrame(
        {
            "Subject": [f"S{i:04d}" for i in range(n)],
            "Family_ID": [f"F{i // 2:04d}" for i in range(n)],
            "Unrelated": [i % 2 == 0 for i in range(n)],
            "ProcSpeed_Unadj": rng.normal(size=n),
            "PicSeq_Unadj": rng.normal(size=n),
            "ReadEng_Unadj": rng.normal(size=n),
            "PicVocab_Unadj": rng.normal(size=n),
            "Flanker_Unadj": rng.normal(size=n),
            "SCPT_SEN": rng.normal(size=n),
            "SCPT_SPEC": rng.normal(size=n),
            "CardSort_Unadj": rng.normal(size=n),
            "ListSort_Unadj": rng.normal(size=n),
            "PMAT24_A_CR": rng.normal(size=n),
            "PMAT24_A_RTCR": rng.normal(size=n),
        }
    )
