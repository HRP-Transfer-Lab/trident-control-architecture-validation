"""Canonical window-schema validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


class SchemaValidationError(ValueError):
    """Raised when a canonical window table violates the data contract."""


WINDOW_KEY_FIELDS = (
    "source_dataset",
    "participant_id",
    "session_id",
    "task_id",
    "block_id",
    "window_id",
)

REQUIRED_IDENTITY_FIELDS = (
    "source_dataset",
    "source_version",
    "participant_id",
    "session_id",
    "task_id",
    "block_id",
    "window_id",
    "window_start_trial",
    "window_end_trial",
    "n_trials_total",
    "n_trials_valid",
)

REQUIRED_PROVENANCE_FIELDS = (
    "source_file_or_table",
    "source_commit_or_release",
    "source_hash_if_available",
    "preprocessing_version",
    "feature_version",
)

AVAILABILITY_FLAGS = (
    "has_conflict_cost",
    "has_post_error",
    "has_vigilance",
    "has_switch_structure",
    "has_confidence",
    "has_change_point",
)

STRUCTURAL_FEATURES_BY_FLAG = {
    "has_conflict_cost": ("conflict_cost_rt", "conflict_cost_accuracy"),
    "has_post_error": ("post_error_adjustment", "error_burstiness", "recovery_slope"),
    "has_vigilance": (
        "vigilance_engagement",
        "inhibitory_stability",
        "reciprocal_rt",
        "slow_tail_response_speed",
        "lapse_rate",
        "false_start_rate",
        "vigilance_drift",
    ),
    "has_switch_structure": ("switch_rate",),
    "has_confidence": (),
    "has_change_point": (),
}


@dataclass(frozen=True)
class SchemaReport:
    """Summary of a successfully validated canonical window table."""

    n_rows: int
    n_participants: int
    n_sources: int
    n_tasks: int
    structural_missing_cells: int


def validate_window_schema(
    frame: pd.DataFrame,
    *,
    require_availability_flags: bool = True,
) -> SchemaReport:
    """Validate the canonical participant-session-task-window table.

    The function checks contract safety and never fills missing values.
    Structural absence is valid only when the corresponding availability flag is
    false and the feature cells remain missing.
    """

    if not isinstance(frame, pd.DataFrame):
        raise SchemaValidationError("window table must be a pandas DataFrame")
    if frame.empty:
        raise SchemaValidationError("window table must not be empty")

    _require_columns(frame, REQUIRED_IDENTITY_FIELDS)
    _require_columns(frame, REQUIRED_PROVENANCE_FIELDS)
    if require_availability_flags:
        _require_columns(frame, AVAILABILITY_FLAGS)

    duplicated = frame.duplicated(list(WINDOW_KEY_FIELDS), keep=False)
    if duplicated.any():
        raise SchemaValidationError("duplicate canonical window keys found")

    for field in REQUIRED_IDENTITY_FIELDS + REQUIRED_PROVENANCE_FIELDS:
        if frame[field].isna().any():
            raise SchemaValidationError(f"{field} must not contain missing values")

    for count_field in ("n_trials_total", "n_trials_valid"):
        values = pd.to_numeric(frame[count_field], errors="coerce")
        if values.isna().any():
            raise SchemaValidationError(f"{count_field} must be numeric")
        if (values <= 0).any():
            raise SchemaValidationError(f"{count_field} must be positive")

    if (frame["n_trials_valid"] > frame["n_trials_total"]).any():
        raise SchemaValidationError("n_trials_valid must be <= n_trials_total")
    if (frame["window_end_trial"] < frame["window_start_trial"]).any():
        raise SchemaValidationError("window_end_trial must be >= window_start_trial")

    structural_missing_cells = 0
    for flag, feature_columns in STRUCTURAL_FEATURES_BY_FLAG.items():
        if flag not in frame.columns:
            continue
        if frame[flag].isna().any():
            raise SchemaValidationError(f"{flag} must not contain missing values")
        false_mask = ~frame[flag].astype(bool)
        true_mask = frame[flag].astype(bool)
        present_features = [column for column in feature_columns if column in frame.columns]
        if true_mask.any() and feature_columns and not present_features:
            raise SchemaValidationError(f"{flag} is true but no matching feature columns exist")
        for column in present_features:
            structural_missing_cells += int(frame.loc[false_mask, column].isna().sum())
            if frame.loc[false_mask, column].notna().any():
                raise SchemaValidationError(
                    f"{column} has values where {flag} marks structural absence"
                )

    return SchemaReport(
        n_rows=len(frame),
        n_participants=frame[_participant_identity_columns(frame)].drop_duplicates().shape[0],
        n_sources=frame["source_dataset"].nunique(dropna=False),
        n_tasks=frame["task_id"].nunique(dropna=False),
        structural_missing_cells=structural_missing_cells,
    )


def require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    """Public wrapper for explicit column checks used by downstream modules."""

    _require_columns(frame, tuple(columns))


def _require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise SchemaValidationError(f"missing required columns: {', '.join(missing)}")


def _participant_identity_columns(frame: pd.DataFrame) -> list[str]:
    if "source_dataset" in frame.columns:
        return ["source_dataset", "participant_id"]
    return ["participant_id"]

