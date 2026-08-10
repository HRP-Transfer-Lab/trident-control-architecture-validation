"""HCP-YA transversal extract schema utilities.

This module writes an empty local CSV template for authorised HCP-YA extracts.
It never writes participant rows.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from trident_validation.config import ConfigValidationError, load_yaml_config


@dataclass(frozen=True)
class HCPExtractSchema:
    """Validated HCP-YA extract schema."""

    schema_id: str
    target_path: Path
    columns: tuple[str, ...]
    participant_level_data_in_git_allowed: bool
    model_fitting_allowed: bool


def load_hcp_extract_schema(
    path: str | Path = "config/hcp_ya_transversal_extract_schema_v1.yaml",
    *,
    repo_root: str | Path | None = None,
) -> HCPExtractSchema:
    """Load and validate the HCP-YA extract schema."""

    schema_path = Path(path)
    root = Path(repo_root) if repo_root is not None else schema_path.resolve().parents[1]
    config = load_yaml_config(root / schema_path if not schema_path.is_absolute() else schema_path)
    return validate_hcp_extract_schema(config, repo_root=root)


def validate_hcp_extract_schema(config: dict[str, Any], *, repo_root: str | Path) -> HCPExtractSchema:
    """Validate schema and return canonical column order."""

    schema = _required_mapping(config, "schema")
    if schema.get("id") != "hcp_ya_transversal_extract_schema_v1":
        raise ConfigValidationError("schema.id must be hcp_ya_transversal_extract_schema_v1")
    if schema.get("status") != "local_extract_schema_no_participant_data":
        raise ConfigValidationError("schema.status must be local_extract_schema_no_participant_data")
    if schema.get("participant_level_data_in_git_allowed") is not False:
        raise ConfigValidationError("participant-level data in Git must be false")
    if schema.get("model_fitting_allowed") is not False:
        raise ConfigValidationError("schema must not authorise model fitting")

    columns: list[str] = []
    columns.extend(_mapping_keys(config["identity"]["required"]))
    columns.extend(_mapping_keys(config["identity"].get("optional", {})))
    for variable in ("K", "C_signal", "V"):
        columns.extend(_mapping_keys(config["predictor_candidates"][variable]["columns"]))
    for domain in ("attention_control", "wm_list_sorting", "wm_nback", "reasoning_pmat"):
        columns.extend(_mapping_keys(config["heldout_outcomes"][domain]["columns"]))

    deduped = tuple(dict.fromkeys(columns))
    if len(deduped) != len(set(deduped)):
        raise ConfigValidationError("schema column de-duplication failed")
    anti = _required_mapping(config, "anti_circularity")
    for field in (
        "outcome_column_must_not_define_same_domain_predictor",
        "list_sorting_excluded_from_K_when_wm_outcome",
        "pmat_excluded_from_K_when_reasoning_outcome",
        "c_signal_and_v_must_not_overlap_heldout_outcomes",
    ):
        if anti.get(field) is not True:
            raise ConfigValidationError(f"anti_circularity.{field} must be true")
    if anti.get("ordinary_participant_folds_allowed") is not False:
        raise ConfigValidationError("ordinary participant folds must be false")

    return HCPExtractSchema(
        schema_id=str(schema["id"]),
        target_path=_resolve_repo_path(Path(repo_root), str(schema["target_path"])),
        columns=deduped,
        participant_level_data_in_git_allowed=False,
        model_fitting_allowed=False,
    )


def write_empty_extract_template(schema: HCPExtractSchema, *, force: bool = False) -> Path:
    """Write an empty CSV header template and return its path."""

    path = schema.target_path
    if path.exists() and not force:
        existing = pd.read_csv(path, nrows=1)
        if len(existing) > 0:
            raise ConfigValidationError(
                f"{path} appears to contain participant rows; refusing to overwrite"
            )
        if tuple(existing.columns) != schema.columns:
            raise ConfigValidationError(
                f"{path} already exists with different columns; use --force only for an empty template rewrite"
            )
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=list(schema.columns)).to_csv(path, index=False)
    return path


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict) or not value:
        raise ConfigValidationError(f"{key} must be a non-empty mapping")
    return value


def _mapping_keys(value: dict[str, Any]) -> list[str]:
    if not isinstance(value, dict):
        raise ConfigValidationError("schema section must be a mapping")
    return [str(key) for key in value.keys()]


def _resolve_repo_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", default="config/hcp_ya_transversal_extract_schema_v1.yaml")
    parser.add_argument("--write-template", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    schema = load_hcp_extract_schema(args.schema)
    if not args.write_template:
        print("\n".join(schema.columns))
        return
    path = write_empty_extract_template(schema, force=args.force)
    print(path)


if __name__ == "__main__":
    main()
