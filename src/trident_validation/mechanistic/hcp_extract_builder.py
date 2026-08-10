"""Build the local canonical HCP-YA transversal extract.

This utility consumes an authorised local HCP export and writes only the
registered canonical columns needed for the support preflight. It does not
download HCP data and does not fit models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any
from collections.abc import Sequence

import pandas as pd

from trident_validation.config import ConfigValidationError
from trident_validation.mechanistic.hcp_extract_schema import (
    HCPExtractSchema,
    canonicalize_hcp_extract_columns,
    load_hcp_extract_schema,
)
from trident_validation.provenance import get_git_commit, hash_file


def build_hcp_transversal_extract(
    source_path: str | Path | Sequence[str | Path],
    *,
    schema_path: str | Path = "config/hcp_ya_transversal_extract_schema_v1.yaml",
    repo_root: str | Path | None = None,
    output_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    cohort_mode: str = "family",
    hcp_100_unrelated_provenance_declared: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Build the canonical HCP extract and return a participant-free summary."""

    schema_file = Path(schema_path)
    root = Path(repo_root) if repo_root is not None else schema_file.resolve().parents[1]
    schema = load_hcp_extract_schema(schema_file, repo_root=root)
    sources = [_resolve_repo_path(root, source) for source in _source_list(source_path)]
    if not sources:
        raise ConfigValidationError("at least one authorised HCP source export is required")
    for source in sources:
        if not source.exists():
            raise ConfigValidationError(f"HCP source export does not exist: {source}")
    if cohort_mode not in {"family", "hcp_100_unrelated"}:
        raise ConfigValidationError("cohort_mode must be family or hcp_100_unrelated")
    if cohort_mode == "hcp_100_unrelated" and not hcp_100_unrelated_provenance_declared:
        raise ConfigValidationError(
            "cohort_mode=hcp_100_unrelated requires declared official HCP 100 Unrelated Subjects provenance"
        )
    destination = _resolve_repo_path(root, output_path) if output_path is not None else schema.target_path
    if destination.exists() and not force:
        existing = pd.read_csv(destination, nrows=1)
        if len(existing) > 0:
            raise ConfigValidationError(
                f"{destination} appears to contain participant rows; use --force only for an intentional local overwrite"
            )

    canonical, source_summaries, merge_summary = _read_and_merge_sources(sources, schema)
    missing_required = [column for column in schema.required_columns if column not in canonical.columns]
    if cohort_mode == "family" and "Family_ID" not in canonical.columns:
        missing_required.append("Family_ID")
    if missing_required:
        raise ConfigValidationError(
            "authorised HCP export is missing required canonical/source columns: "
            + ", ".join(missing_required)
        )

    output = _canonical_subset(canonical, schema)
    if len(output) > 0:
        _assert_participant_output_path_allowed(root, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    summary = {
        "builder_id": "hcp_ya_transversal_extract_builder_v1",
        "model_fitting_allowed": False,
        "participant_level_data_in_git_allowed": False,
        "cohort_mode": cohort_mode,
        "official_hcp_100_unrelated_subjects_group_declared": bool(
            hcp_100_unrelated_provenance_declared
        ),
        "source_path": str(sources[0]) if len(sources) == 1 else None,
        "source_sha256": hash_file(sources[0]) if len(sources) == 1 else None,
        "source_paths": [str(source) for source in sources],
        "source_filenames": [source.name for source in sources],
        "source_summaries": source_summaries,
        "merge_validation": merge_summary,
        "output_path": str(destination),
        "output_sha256": hash_file(destination),
        "n_rows": int(len(output)),
        "n_unique_subjects": int(output["Subject"].nunique()) if "Subject" in output else 0,
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


def _source_list(source_path: str | Path | Sequence[str | Path]) -> list[str | Path]:
    if isinstance(source_path, (str, Path)):
        return [source_path]
    return list(source_path)


def _read_and_merge_sources(
    sources: list[Path],
    schema: HCPExtractSchema,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    subject_sets: list[set[str]] = []
    row_counts: list[int] = []
    duplicate_noncanonical_dropped: dict[str, list[str]] = {}

    for source in sources:
        frame = canonicalize_hcp_extract_columns(_read_table(source), schema)
        if "Subject" not in frame.columns:
            raise ConfigValidationError(f"{source.name} is missing Subject")
        if frame["Subject"].isna().any():
            raise ConfigValidationError(f"{source.name} contains missing Subject values")
        duplicate_count = int(frame["Subject"].duplicated(keep=False).sum())
        if duplicate_count:
            raise ConfigValidationError(f"{source.name} contains duplicate Subject values")
        subjects = {str(subject) for subject in frame["Subject"]}
        frames.append(frame)
        subject_sets.append(subjects)
        row_counts.append(int(len(frame)))
        summaries.append(
            {
                "filename": source.name,
                "sha256": hash_file(source),
                "n_rows": int(len(frame)),
                "n_unique_subjects": int(len(subjects)),
                "n_columns": int(len(frame.columns)),
                "columns": list(frame.columns),
                "subject_column_present": True,
                "subject_unique": True,
            }
        )

    first_count = row_counts[0]
    if any(count != first_count for count in row_counts):
        raise ConfigValidationError("HCP source export row counts differ unexpectedly")
    first_subjects = subject_sets[0]
    for source, subjects in zip(sources[1:], subject_sets[1:]):
        if subjects != first_subjects:
            raise ConfigValidationError(f"{source.name} has a different Subject set")

    merged = frames[0].copy()
    registered_columns = set(schema.columns)
    for source, frame in zip(sources[1:], frames[1:]):
        overlap = sorted(set(merged.columns).intersection(frame.columns).difference({"Subject"}))
        registered_overlap = sorted(set(overlap).intersection(registered_columns))
        if registered_overlap:
            raise ConfigValidationError(
                f"{source.name} repeats registered HCP columns: " + ", ".join(registered_overlap)
            )
        if overlap:
            duplicate_noncanonical_dropped[source.name] = overlap
            frame = frame.drop(columns=overlap)
        previous_rows = len(merged)
        merged = merged.merge(frame, on="Subject", how="inner", validate="one_to_one")
        if len(merged) != previous_rows:
            raise ConfigValidationError("HCP source merge changed row count")

    merge_summary = {
        "status": "one_to_one_subject_merge_validated",
        "n_sources": len(sources),
        "n_rows": int(len(merged)),
        "n_unique_subjects": int(merged["Subject"].nunique()),
        "subject_sets_identical": True,
        "source_row_counts_identical": True,
        "duplicates_present": False,
        "duplicate_noncanonical_columns_dropped_from_later_sources": duplicate_noncanonical_dropped,
    }
    return merged, summaries, merge_summary


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


def _assert_participant_output_path_allowed(root: Path, destination: Path) -> None:
    try:
        relative = destination.resolve().relative_to(root.resolve())
    except ValueError:
        return
    relative_posix = relative.as_posix()
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative_posix],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked.returncode == 0:
        raise ConfigValidationError(f"refusing to write participant-level data to tracked path: {relative_posix}")
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", relative_posix],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if ignored.returncode != 0:
        raise ConfigValidationError(
            f"refusing to write participant-level data to non-ignored repository path: {relative_posix}"
        )


def _resolve_repo_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        action="append",
        help="Authorised local HCP CSV/TSV/Parquet export; repeat for a validated Subject merge",
    )
    parser.add_argument("--schema", default="config/hcp_ya_transversal_extract_schema_v1.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary", default="reports/generated/hcp_ya_transversal_v1/extract_build_summary.json")
    parser.add_argument("--cohort-mode", default="family", choices=["family", "hcp_100_unrelated"])
    parser.add_argument(
        "--official-hcp-100-unrelated-provenance",
        action="store_true",
        help="Declare that the input was exported from the official HCP 100 Unrelated Subjects group",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    summary = build_hcp_transversal_extract(
        args.source,
        schema_path=args.schema,
        output_path=args.output,
        summary_path=args.summary,
        cohort_mode=args.cohort_mode,
        hcp_100_unrelated_provenance_declared=args.official_hcp_100_unrelated_provenance,
        force=args.force,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
