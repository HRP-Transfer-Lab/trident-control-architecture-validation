from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trident_validation.config import ConfigValidationError
from trident_validation.mechanistic.hcp_extract_builder import build_hcp_transversal_extract


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
