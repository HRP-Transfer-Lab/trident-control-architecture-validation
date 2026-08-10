"""Build the local canonical HCP-YA transversal extract.

This utility consumes an authorised local HCP export and writes only the
registered canonical columns needed for the support preflight. It does not
download HCP data and does not fit models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from trident_validation.config import ConfigValidationError
from trident_validation.mechanistic.hcp_extract_schema import (
    HCPExtractSchema,
    canonicalize_hcp_extract_columns,
    load_hcp_extract_schema,
)
from trident_validation.provenance import get_git_commit, hash_file


def build_hcp_transversal_extract(
    source_path: str | Path,
    *,
    schema_path: str | Path = "config/hcp_ya_transversal_extract_schema_v1.yaml",
    repo_root: str | Path | None = None,
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build the canonical HCP extract and return a participant-free summary."""

    schema_file = Path(schema_path)
    root = Path(repo_root) if repo_root is not None else schema_file.resolve().parents[1]
    schema = load_hcp_extract_schema(schema_file, repo_root=root)
    source = _resolve_repo_path(root, source_path)
    if not source.exists():
        raise ConfigValidationError(f"HCP source export does not exist: {source}")
    destination = _resolve_repo_path(root, output_path) if output_path is not None else schema.target_path
    if destination.exists() and not force:
        existing = pd.read_csv(destination, nrows=1)
        if len(existing) > 0:
            raise ConfigValidationError(
                f"{destination} appears to contain participant rows; use --force only for an intentional local overwrite"
            )

    raw = _read_table(source)
    canonical = canonicalize_hcp_extract_columns(raw, schema)
    missing_required = [column for column in schema.required_columns if column not in canonical.columns]
    if missing_required:
        raise ConfigValidationError(
            "authorised HCP export is missing required canonical/source columns: "
            + ", ".join(missing_required)
        )

    output = _canonical_subset(canonical, schema)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    summary = {
        "builder_id": "hcp_ya_transversal_extract_builder_v1",
        "model_fitting_allowed": False,
        "participant_level_data_in_git_allowed": False,
        "source_path": str(source),
        "source_sha256": hash_file(source),
        "output_path": str(destination),
        "output_sha256": hash_file(destination),
        "n_rows": int(len(output)),
        "n_columns": int(len(output.columns)),
        "columns": list(output.columns),
        "required_columns": list(schema.required_columns),
        "optional_columns": list(schema.optional_columns),
        "missing_optional_columns_filled_empty": [
            column for column in schema.optional_columns if column not in canonical.columns
        ],
        "git_commit": get_git_commit(root),
    }
    if summary_path is not None:
        summary_destination = _resolve_repo_path(root, summary_path)
        summary_destination.parent.mkdir(parents=True, exist_ok=True)
        summary_destination.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return summary


def _canonical_subset(data: pd.DataFrame, schema: HCPExtractSchema) -> pd.DataFrame:
    output = pd.DataFrame(index=data.index)
    for column in schema.columns:
        if column in data.columns:
            output[column] = data[column]
        elif column in schema.optional_columns:
            output[column] = pd.NA
        else:
            raise ConfigValidationError(f"required HCP column missing after canonicalization: {column}")
    return output


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ConfigValidationError(f"unsupported HCP source extension: {path.suffix}")


def _resolve_repo_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Authorised local HCP CSV/TSV/Parquet export")
    parser.add_argument("--schema", default="config/hcp_ya_transversal_extract_schema_v1.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary", default="reports/generated/hcp_ya_transversal_v1/extract_build_summary.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    summary = build_hcp_transversal_extract(
        args.source,
        schema_path=args.schema,
        output_path=args.output,
        summary_path=args.summary,
        force=args.force,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
