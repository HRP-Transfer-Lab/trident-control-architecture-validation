from pathlib import Path

import pandas as pd
import pytest

from trident_validation.config import ConfigValidationError, load_yaml_config
from trident_validation.mechanistic.hcp_extract_schema import (
    load_hcp_extract_schema,
    validate_hcp_extract_schema,
    write_empty_extract_template,
)


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "config/hcp_ya_transversal_extract_schema_v1.yaml"


def test_hcp_extract_schema_is_claim_bounded_and_ordered():
    schema = load_hcp_extract_schema(SCHEMA_PATH, repo_root=ROOT)

    assert schema.schema_id == "hcp_ya_transversal_extract_schema_v1"
    assert schema.participant_level_data_in_git_allowed is False
    assert schema.model_fitting_allowed is False
    assert schema.columns[:2] == ("Subject", "Family_ID")
    assert "ListSort_Unadj" in schema.columns
    assert "PMAT24_A_CR" in schema.columns
    assert "NIH_Flanker_Unadj" not in schema.columns
    assert schema.source_to_canonical["Flanker_Unadj"] == "Flanker_Unadj"
    assert len(schema.columns) == len(set(schema.columns))


def test_hcp_extract_schema_rejects_participant_data_in_git_permission():
    config = load_yaml_config(SCHEMA_PATH)
    config["schema"]["participant_level_data_in_git_allowed"] = True

    with pytest.raises(ConfigValidationError, match="participant-level data"):
        validate_hcp_extract_schema(config, repo_root=ROOT)


def test_hcp_extract_template_writes_empty_header_only():
    config = load_yaml_config(SCHEMA_PATH)
    work_dir = ROOT / "reports" / "generated" / "test_hcp_extract_schema"
    work_dir.mkdir(parents=True, exist_ok=True)
    config["schema"]["target_path"] = str(work_dir / "hcp_template.csv")
    schema = validate_hcp_extract_schema(config, repo_root=ROOT)

    path = write_empty_extract_template(schema, force=True)
    written = pd.read_csv(path)

    assert tuple(written.columns) == schema.columns
    assert len(written) == 0


def test_hcp_extract_template_refuses_to_overwrite_participant_rows():
    config = load_yaml_config(SCHEMA_PATH)
    work_dir = ROOT / "reports" / "generated" / "test_hcp_extract_schema_refuse"
    work_dir.mkdir(parents=True, exist_ok=True)
    target = work_dir / "hcp_template.csv"
    config["schema"]["target_path"] = str(target)
    schema = validate_hcp_extract_schema(config, repo_root=ROOT)
    pd.DataFrame([{column: "value" for column in schema.columns}]).to_csv(target, index=False)

    with pytest.raises(ConfigValidationError, match="participant rows"):
        write_empty_extract_template(schema)
