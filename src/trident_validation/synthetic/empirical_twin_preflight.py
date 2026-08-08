"""M2.7 empirical-twin nuisance preflight."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd

from trident_validation.config import load_yaml_config
from trident_validation.models.static_tournament_v2 import STATIC_TOURNAMENT_V2_CONTRACT
from trident_validation.provenance import get_git_commit, hash_file, hash_mapping
from trident_validation.schema import AVAILABILITY_FLAGS, validate_window_schema
from trident_validation.synthetic.fixtures import CORE_SYNTHETIC_FEATURES, make_synthetic_window_table
from trident_validation.synthetic.recovery import ground_truth_columns


PREFLIGHT_STUDY_ID = "empirical_twin_preflight_v1"
DEFAULT_CONFIG_PATH = Path("config/empirical_twin_preflight_v1.yaml")
DEFAULT_OUTPUT_DIR = Path("reports/generated/empirical_twin_preflight_v1")
TEMPLATE_COLUMNS = ("source_dataset", "task_id")


def run_empirical_twin_preflight(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path | None = None,
    input_tables: Sequence[str | Path] | None = None,
    allow_fallback_fixture: bool = True,
) -> dict[str, Any]:
    """Run nuisance-only preflight for M2.7 empirical-twin design."""

    start = time.perf_counter()
    config_path = Path(config_path)
    config = load_yaml_config(config_path)
    _validate_preflight_config(config)
    frame, input_mode, input_metadata = _load_or_make_preflight_frame(
        config,
        config_base_dir=config_path.parent,
        input_tables=input_tables,
        allow_fallback_fixture=allow_fallback_fixture,
    )
    frame = canonicalise_empirical_window_table(frame)
    report = validate_window_schema(frame)
    _assert_no_truth_inputs(frame)
    feature_columns = tuple(config["nuisance_features"]["core_features"])
    nuisance = estimate_empirical_nuisance(frame, feature_columns=feature_columns)
    target_dir = Path(output_dir or config["outputs"]["directory"])
    paths = _write_preflight_outputs(
        nuisance,
        output_dir=target_dir,
        schema_report=report,
        input_mode=input_mode,
        metadata={
            "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "input_mode": input_mode,
            "input_tables": input_metadata,
            "validation_repo_commit": _repo_commit_or_unknown(Path.cwd()),
            "config_path": str(config_path),
            "config_hash": hash_file(config_path),
            "config_content_hash": hash_mapping(config),
            "schema_preprocessing_version": "empirical-twin-preflight-v2-nuisance-decomposition",
            "n_rows": int(report.n_rows),
            "n_participants": int(report.n_participants),
            "n_sessions": int(
                frame.loc[:, ["source_dataset", "participant_id", "session_id"]]
                .drop_duplicates()
                .shape[0]
            ),
            "n_sources": int(report.n_sources),
            "n_tasks": int(report.n_tasks),
            "n_templates": int(nuisance["template_summary"].shape[0]),
            "formal_claims_allowed": False,
        },
    )
    summary = {
        "study_id": PREFLIGHT_STUDY_ID,
        "input_mode": input_mode,
        "static_tournament_contract": STATIC_TOURNAMENT_V2_CONTRACT.contract_id,
        "n_rows": int(report.n_rows),
        "n_participants": int(report.n_participants),
        "n_sources": int(report.n_sources),
        "n_tasks": int(report.n_tasks),
        "n_templates": int(nuisance["template_summary"].shape[0]),
        "formal_claims_allowed": False,
        "runtime_seconds": round(float(time.perf_counter() - start), 3),
    }
    summary_path = target_dir / "empirical_twin_preflight_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    paths["summary"] = summary_path
    return {
        **summary,
        "paths": {key: str(value) for key, value in paths.items()},
    }


def estimate_empirical_nuisance(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = CORE_SYNTHETIC_FEATURES,
) -> dict[str, pd.DataFrame]:
    """Estimate nuisance summaries without reading latent truth columns."""

    _assert_no_truth_inputs(frame)
    features = tuple(feature_columns)
    template_summary = _template_summary(frame, features)
    template_support = _template_support(frame, features)
    feature_summary = _feature_summary(frame, features)
    covariance_raw = _template_covariance(frame, features, level="raw")
    covariance_between = _template_covariance(frame, features, level="between_person")
    covariance_session = _template_covariance(frame, features, level="session")
    covariance_within = _template_covariance(frame, features, level="within_session")
    variance = _variance_decomposition(frame, features)
    temporal = _temporal_summary(frame, features)
    missingness = _missingness_summary(frame, features)
    return {
        "template_summary": template_summary,
        "template_support": template_support,
        "feature_summary": feature_summary,
        "covariance_raw": covariance_raw,
        "covariance_between_person": covariance_between,
        "covariance_session": covariance_session,
        "covariance_within_session": covariance_within,
        "variance_decomposition": variance,
        "temporal_summary": temporal,
        "missingness_summary": missingness,
    }


def canonicalise_empirical_window_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a canonical M2.7 empirical window table.

    Native canonical tables are returned unchanged. Flow Zone cognitive-window
    outputs are mapped into the M2.7 canonical schema for nuisance estimation.
    """

    if {"source_dataset", "task_id", "window_start_trial", "n_trials_valid"}.issubset(
        frame.columns
    ):
        return frame.copy()
    if {"dataset_id", "task_family", "window_id", "window_size", "window_index"}.issubset(
        frame.columns
    ):
        return _canonicalise_flowzone_cognitive_windows(frame)
    return frame.copy()


def _load_or_make_preflight_frame(
    config: dict[str, Any],
    *,
    config_base_dir: Path,
    input_tables: Sequence[str | Path] | None = None,
    allow_fallback_fixture: bool = True,
) -> tuple[pd.DataFrame, str, list[dict[str, Any]]]:
    input_paths = _configured_input_paths(
        config["inputs"].get("empirical_window_tables", []),
        config_base_dir=config_base_dir,
    )
    if input_tables:
        input_paths.extend(Path(path) for path in input_tables)
    if input_paths:
        frames = [_read_window_table(path) for path in input_paths]
        metadata = [_input_metadata(path) for path in input_paths]
        return pd.concat(frames, ignore_index=True), "empirical_window_tables", metadata
    if not allow_fallback_fixture:
        raise ValueError("no empirical window tables provided")
    fixture = config["inputs"]["fallback_smoke_fixture"]
    if not bool(fixture.get("enabled", False)):
        raise ValueError("no empirical_window_tables configured and fallback smoke fixture disabled")
    frame = make_synthetic_window_table(
            seed=int(fixture["seed"]),
            n_datasets=int(fixture["n_datasets"]),
            participants_per_dataset=int(fixture["participants_per_dataset"]),
            sessions_per_participant=int(fixture["sessions_per_participant"]),
            min_windows_per_session=int(fixture["min_windows_per_session"]),
            max_windows_per_session=int(fixture["max_windows_per_session"]),
    )
    return (
        frame,
        "fallback_smoke_fixture",
        [
            {
                "path": "synthetic.infrastructure_fixture",
                "checksum": "not_applicable",
                "source_repository": "not_applicable",
                "source_commit_or_release": "not_applicable",
            }
        ],
    )


def _configured_input_paths(raw_entries: Sequence[Any], *, config_base_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for entry in raw_entries:
        if isinstance(entry, dict):
            if "path" not in entry:
                raise ValueError("empirical_window_tables entries must include path")
            raw_path = Path(str(entry["path"]))
        else:
            raw_path = Path(str(entry))
        paths.append(raw_path if raw_path.is_absolute() else config_base_dir / raw_path)
    return paths


def _read_window_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"empirical window table not found: {path}")
    suffixes = "".join(path.suffixes[-2:]).lower()
    if path.suffix.lower() == ".csv" or suffixes == ".csv.gz":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"unsupported empirical table format: {path}")


def _canonicalise_flowzone_cognitive_windows(frame: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    output["source_dataset"] = frame["dataset_id"].astype(str)
    output["source_version"] = "flow-zone-cognitive-windows"
    output["participant_id"] = frame["participant_id"].astype(str)
    output["session_id"] = (
        frame["session_id"].astype(str)
        if "session_id" in frame.columns
        else "session_unknown"
    )
    output["task_id"] = frame["task_family"].astype(str)
    output["block_id"] = frame["block_raw"].astype(str) if "block_raw" in frame.columns else "block_unknown"
    output["window_id"] = frame["window_id"].astype(str)
    window_size = pd.to_numeric(frame["window_size"], errors="coerce").fillna(1).astype(int)
    window_index = pd.to_numeric(frame["window_index"], errors="coerce").fillna(0).astype(int)
    output["window_start_trial"] = (window_index * window_size + 1).astype(int)
    output["window_end_trial"] = (output["window_start_trial"] + window_size - 1).astype(int)
    output["n_trials_total"] = pd.to_numeric(
        frame.get("n_trials", window_size),
        errors="coerce",
    ).fillna(window_size).astype(int)
    output["n_trials_valid"] = pd.to_numeric(
        frame.get("n_valid_correct_rt", frame.get("n_valid_rt", output["n_trials_total"])),
        errors="coerce",
    ).fillna(output["n_trials_total"]).clip(lower=1).astype(int)
    output["source_file_or_table"] = "flow-zone-zone-validation:data/processed/cognitive_windows"
    output["source_commit_or_release"] = "external-flow-zone-github-database-derived"
    output["source_hash_if_available"] = "not_recorded_in_input_table"
    output["preprocessing_version"] = "m2.7-flowzone-cognitive-window-adapter-v1"
    output["feature_version"] = "canonical-window-v1"

    output["accuracy"] = pd.to_numeric(frame.get("accuracy"), errors="coerce")
    output["median_rt_ms"] = pd.to_numeric(frame.get("median_rt_ms"), errors="coerce")
    output["mean_response_speed"] = _flowzone_response_speed(frame)
    output["rt_cv"] = pd.to_numeric(frame.get("rt_cv"), errors="coerce")
    output["throughput_proxy"] = pd.to_numeric(frame.get("throughput_proxy"), errors="coerce")
    output["trial_count"] = output["n_trials_valid"]
    output["practice_or_session_index"] = pd.to_numeric(
        frame.get("window_index", 0),
        errors="coerce",
    ).fillna(0)
    output["time_on_task"] = output["practice_or_session_index"] * output["n_trials_total"]
    output["condition_mix"] = frame.get("control_cost_type", "mixed")
    output["congruency_mix"] = np.where(
        pd.to_numeric(frame.get("control_cost_supported", False), errors="coerce").fillna(0).astype(bool),
        "mixed",
        "not_applicable",
    )
    output["switch_rate"] = np.nan
    output["lure_rate"] = np.nan
    output["difficulty_level"] = 1
    output["soa_or_foreperiod"] = np.nan
    output["response_mapping"] = "source_defined"
    output["input_device"] = "unknown"
    output["timing_quality"] = "source_defined"
    output["browser_focus_flags"] = "unknown"

    has_conflict = _flowzone_bool(frame, "control_cost_supported")
    has_post_error = _flowzone_bool(frame, "pes_supported")
    has_vigilance = output["task_id"].str.contains("sart", case=False, na=False)
    output["has_conflict_cost"] = has_conflict
    output["has_post_error"] = has_post_error
    output["has_vigilance"] = has_vigilance
    output["has_switch_structure"] = False
    output["has_confidence"] = False
    output["has_change_point"] = False

    output["conflict_cost_rt"] = pd.to_numeric(frame.get("control_cost_rt_ms"), errors="coerce").where(has_conflict)
    output["conflict_cost_accuracy"] = pd.to_numeric(frame.get("control_cost_acc"), errors="coerce").where(has_conflict)
    output["post_error_adjustment"] = pd.to_numeric(frame.get("post_error_slowing_ms"), errors="coerce").where(has_post_error)
    output["error_burstiness"] = pd.to_numeric(frame.get("error_burstiness"), errors="coerce").where(has_post_error)
    output["recovery_slope"] = np.nan
    output["vigilance_engagement"] = (1.0 - pd.to_numeric(frame.get("nonresponse_rate"), errors="coerce")).where(has_vigilance)
    output["inhibitory_stability"] = output["accuracy"].where(has_vigilance)
    output["reciprocal_rt"] = output["mean_response_speed"].where(has_vigilance)
    output["slow_tail_response_speed"] = (output["mean_response_speed"] * (1.0 - pd.to_numeric(frame.get("slow_tail_rate"), errors="coerce"))).where(has_vigilance)
    output["lapse_rate"] = pd.to_numeric(frame.get("nonresponse_rate"), errors="coerce").where(has_vigilance)
    output["false_start_rate"] = pd.to_numeric(frame.get("fast_error_rate"), errors="coerce").where(has_vigilance)
    output["vigilance_drift"] = pd.to_numeric(frame.get("rt_drift"), errors="coerce").where(has_vigilance)

    for flag in AVAILABILITY_FLAGS:
        output[flag] = output[flag].fillna(False).astype(bool)
    return output


def _flowzone_response_speed(frame: pd.DataFrame) -> pd.Series:
    if "mean_response_speed" in frame.columns:
        return pd.to_numeric(frame["mean_response_speed"], errors="coerce")
    if "mean_rt_ms" in frame.columns:
        mean_rt = pd.to_numeric(frame["mean_rt_ms"], errors="coerce")
        return 1000.0 / mean_rt
    median_rt = pd.to_numeric(frame.get("median_rt_ms"), errors="coerce")
    return 1000.0 / median_rt


def _flowzone_bool(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    values = frame[column]
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def _template_summary(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(list(TEMPLATE_COLUMNS), dropna=False, sort=True):
        source, task = keys
        sessions = group.loc[
            :, ["source_dataset", "participant_id", "session_id"]
        ].drop_duplicates()
        participant_sessions = sessions.groupby(
            ["source_dataset", "participant_id"], sort=True
        ).size()
        row: dict[str, Any] = {
            "source_dataset": str(source),
            "task_id": str(task),
            "n_rows": int(group.shape[0]),
            "n_participants": int(
                group.loc[:, ["source_dataset", "participant_id"]].drop_duplicates().shape[0]
            ),
            "n_repeat_participants": int((participant_sessions > 1).sum()),
            "n_sessions": int(sessions.shape[0]),
            "n_windows": int(group.shape[0]),
            "mean_n_trials_valid": float(pd.to_numeric(group["n_trials_valid"]).mean()),
            "sd_n_trials_valid": float(pd.to_numeric(group["n_trials_valid"]).std(ddof=1)),
            "p10_n_trials_valid": float(
                pd.to_numeric(group["n_trials_valid"], errors="coerce").quantile(0.10)
            ),
            "p50_n_trials_valid": float(
                pd.to_numeric(group["n_trials_valid"], errors="coerce").quantile(0.50)
            ),
            "p90_n_trials_valid": float(
                pd.to_numeric(group["n_trials_valid"], errors="coerce").quantile(0.90)
            ),
        }
        for feature in features:
            values = pd.to_numeric(group[feature], errors="coerce")
            row[f"{feature}_mean"] = float(values.mean())
            row[f"{feature}_sd"] = float(values.std(ddof=1))
            row[f"{feature}_coverage"] = float(values.notna().mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _template_support(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(list(TEMPLATE_COLUMNS), dropna=False, sort=True):
        source, task = keys
        sessions = group.loc[
            :, ["source_dataset", "participant_id", "session_id"]
        ].drop_duplicates()
        participant_sessions = sessions.groupby(
            ["source_dataset", "participant_id"], sort=True
        ).size()
        session_windows = group.groupby(
            ["source_dataset", "participant_id", "session_id"], sort=True
        ).size()
        feature_coverages = [
            float(pd.to_numeric(group[feature], errors="coerce").notna().mean())
            for feature in features
        ]
        trial_counts = pd.to_numeric(group["n_trials_valid"], errors="coerce")
        row = {
            "source_dataset": str(source),
            "task_id": str(task),
            "n_rows": int(group.shape[0]),
            "n_participants": int(
                group.loc[:, ["source_dataset", "participant_id"]].drop_duplicates().shape[0]
            ),
            "n_sessions": int(sessions.shape[0]),
            "n_repeat_participants": int((participant_sessions > 1).sum()),
            "n_windows": int(group.shape[0]),
            "valid_trials_p10": float(trial_counts.quantile(0.10)),
            "valid_trials_p50": float(trial_counts.quantile(0.50)),
            "valid_trials_p90": float(trial_counts.quantile(0.90)),
            "feature_coverage_min": float(np.nanmin(feature_coverages)),
            "feature_coverage_mean": float(np.nanmean(feature_coverages)),
            "missingness_max": float(
                max(1.0 - coverage for coverage in feature_coverages)
                if feature_coverages
                else float("nan")
            ),
            "temporal_support": _support_label(int((session_windows >= 3).sum()), 1),
            "practice_support": _support_label(int((participant_sessions >= 2).sum()), 1),
            "variance_component_support": _variance_support_label(group),
            "between_person_covariance_support": _support_label(
                group.loc[:, ["source_dataset", "participant_id"]]
                .drop_duplicates()
                .shape[0],
                2,
            ),
            "session_covariance_support": _support_label(
                int((participant_sessions >= 2).sum()),
                1,
            ),
            "within_session_covariance_support": _support_label(
                int((session_windows >= 2).sum()),
                1,
            ),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _feature_summary(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in features:
        values = pd.to_numeric(frame[feature], errors="coerce")
        template_means = (
            frame.assign(_feature_value=values)
            .dropna(subset=["_feature_value"])
            .groupby(list(TEMPLATE_COLUMNS), sort=True)["_feature_value"]
            .mean()
        )
        rows.append(
            {
                "feature": feature,
                "n_observed": int(values.notna().sum()),
                "missing_rate": float(values.isna().mean()),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "p10": float(values.quantile(0.10)),
                "p50": float(values.quantile(0.50)),
                "p90": float(values.quantile(0.90)),
                "source_task_mean_variance": float(template_means.var(ddof=1)),
                "n_source_task_templates_observed": int(template_means.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def _template_covariance(
    frame: pd.DataFrame,
    features: tuple[str, ...],
    *,
    level: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(list(TEMPLATE_COLUMNS), dropna=False, sort=True):
        source, task = keys
        matrix, support_status, support_reason = _covariance_matrix_for_level(
            group,
            features,
            level=level,
        )
        for feature_a in features:
            for feature_b in features:
                pair = pd.DataFrame(
                    {
                        "feature_a": matrix[feature_a],
                        "feature_b": matrix[feature_b],
                    }
                ).dropna()
                if pair.shape[0] < 2:
                    covariance = float("nan")
                    correlation = float("nan")
                else:
                    covariance = float(pair.cov().iloc[0, 1])
                    correlation = (
                        float(pair.corr().iloc[0, 1])
                        if float(pair["feature_a"].std(ddof=1)) > 0
                        and float(pair["feature_b"].std(ddof=1)) > 0
                        else float("nan")
                    )
                rows.append(
                    {
                        "source_dataset": str(source),
                        "task_id": str(task),
                        "covariance_level": level,
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "n_units": int(pair.shape[0]),
                        "support_status": (
                            support_status
                            if pair.shape[0] >= 2
                            else "unsupported"
                        ),
                        "support_reason": (
                            support_reason
                            if pair.shape[0] >= 2
                            else "fewer_than_two_complete_units_for_feature_pair"
                        ),
                        "covariance": covariance,
                        "correlation": correlation,
                    }
                )
    return pd.DataFrame(rows)


def _variance_decomposition(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(list(TEMPLATE_COLUMNS), dropna=False, sort=True):
        source, task = keys
        for feature in features:
            rows.append(
                {
                    "source_dataset": str(source),
                    "task_id": str(task),
                    "feature": feature,
                    **_nested_variance_components(group, feature),
                }
            )
    return pd.DataFrame(rows)


def _temporal_summary(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(list(TEMPLATE_COLUMNS), dropna=False, sort=True):
        source, task = keys
        for feature in features:
            lag1 = _lag1_autocorrelation_summary(group, feature)
            practice = _practice_slope_summary(group, feature)
            fatigue = _fatigue_slope_summary(group, feature)
            rows.append(
                {
                    "source_dataset": str(source),
                    "task_id": str(task),
                    "feature": feature,
                    **lag1,
                    **practice,
                    **fatigue,
                }
            )
    return pd.DataFrame(rows)


def _lag1_autocorrelation_summary(frame: pd.DataFrame, feature: str) -> dict[str, Any]:
    correlations: list[float] = []
    lengths: list[int] = []
    grouped = frame.sort_values(
        ["source_dataset", "task_id", "participant_id", "session_id", "window_start_trial"]
    ).groupby(["source_dataset", "task_id", "participant_id", "session_id"], sort=True)
    for _, group in grouped:
        values = pd.to_numeric(group[feature], errors="coerce").dropna().to_numpy(dtype=float)
        lengths.append(int(values.shape[0]))
        if values.shape[0] < 3 or float(np.std(values[:-1])) == 0 or float(np.std(values[1:])) == 0:
            continue
        correlations.append(float(np.corrcoef(values[:-1], values[1:])[0, 1]))
    return {
        "lag1_usable_sequences": int(len(correlations)),
        "lag1_median_sequence_length": float(np.median(lengths)) if lengths else float("nan"),
        "lag1_mean_autocorrelation": float(np.mean(correlations)) if correlations else float("nan"),
        "lag1_median_autocorrelation": float(np.median(correlations)) if correlations else float("nan"),
        "lag1_sd_autocorrelation": (
            float(np.std(correlations, ddof=1)) if len(correlations) > 1 else float("nan")
        ),
        "lag1_support_status": "estimated" if correlations else "unsupported",
        "lag1_support_reason": (
            "within_participant_session_sequences"
            if correlations
            else "no_within_task_sequences_with_at_least_three_varying_windows"
        ),
    }


def _practice_slope_summary(frame: pd.DataFrame, feature: str) -> dict[str, Any]:
    slopes: list[float] = []
    session_means = (
        frame.assign(_feature_value=pd.to_numeric(frame[feature], errors="coerce"))
        .dropna(subset=["_feature_value"])
        .groupby(["source_dataset", "task_id", "participant_id", "session_id"], sort=True)
        .agg(
            feature_mean=("_feature_value", "mean"),
            session_order=("practice_or_session_index", "mean"),
        )
        .reset_index()
    )
    for _, group in session_means.groupby(
        ["source_dataset", "task_id", "participant_id"], sort=True
    ):
        x = pd.to_numeric(group["session_order"], errors="coerce")
        y = pd.to_numeric(group["feature_mean"], errors="coerce")
        valid = x.notna() & y.notna()
        if int(valid.sum()) < 2 or float(np.std(x[valid])) == 0:
            continue
        slope = np.polyfit(x[valid].to_numpy(dtype=float), y[valid].to_numpy(dtype=float), deg=1)[0]
        slopes.append(float(slope))
    return {
        "practice_usable_participants": int(len(slopes)),
        "practice_mean_slope": float(np.mean(slopes)) if slopes else float("nan"),
        "practice_median_slope": float(np.median(slopes)) if slopes else float("nan"),
        "practice_sd_slope": float(np.std(slopes, ddof=1)) if len(slopes) > 1 else float("nan"),
        "practice_support_status": "estimated" if slopes else "unsupported",
        "practice_support_reason": (
            "participants_with_repeated_sessions"
            if slopes
            else "no_participants_with_two_or_more_sessions_for_this_source_task"
        ),
    }


def _fatigue_slope_summary(frame: pd.DataFrame, feature: str) -> dict[str, Any]:
    predictor = "time_on_task" if "time_on_task" in frame.columns else "window_start_trial"
    slopes: list[float] = []
    grouped = frame.groupby(["source_dataset", "task_id", "participant_id", "session_id"], sort=True)
    for _, group in grouped:
        x = pd.to_numeric(group[predictor], errors="coerce")
        y = pd.to_numeric(group[feature], errors="coerce")
        valid = x.notna() & y.notna()
        if int(valid.sum()) < 2 or float(np.std(x[valid])) == 0:
            continue
        slope = np.polyfit(x[valid].to_numpy(dtype=float), y[valid].to_numpy(dtype=float), deg=1)[0]
        slopes.append(float(slope))
    return {
        "fatigue_usable_sessions": int(len(slopes)),
        "fatigue_predictor": predictor,
        "fatigue_mean_slope": float(np.mean(slopes)) if slopes else float("nan"),
        "fatigue_median_slope": float(np.median(slopes)) if slopes else float("nan"),
        "fatigue_sd_slope": float(np.std(slopes, ddof=1)) if len(slopes) > 1 else float("nan"),
        "fatigue_support_status": "estimated" if slopes else "unsupported",
        "fatigue_support_reason": (
            "sessions_with_two_or_more_windows"
            if slopes
            else "no_sessions_with_two_or_more_varying_windows_for_this_source_task"
        ),
    }


def _missingness_summary(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(list(TEMPLATE_COLUMNS), dropna=False, sort=True):
        source, task = keys
        for feature in features:
            values = pd.to_numeric(group[feature], errors="coerce")
            rows.append(
                {
                    "source_dataset": str(source),
                    "task_id": str(task),
                    "feature": feature,
                    "n_rows": int(group.shape[0]),
                    "n_observed": int(values.notna().sum()),
                    "n_missing": int(values.isna().sum()),
                    "missing_rate": float(values.isna().mean()),
                    "missingness_type": "incidental_or_technical",
                }
            )
    return pd.DataFrame(rows)


def _covariance_matrix_for_level(
    group: pd.DataFrame,
    features: tuple[str, ...],
    *,
    level: str,
) -> tuple[pd.DataFrame, str, str]:
    matrix = group.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    if level == "raw":
        return matrix, "estimated", "raw_window_covariance_descriptive"

    participant_keys = ["source_dataset", "participant_id"]
    session_keys = ["source_dataset", "participant_id", "session_id"]
    if level == "between_person":
        participant_means = matrix.join(group.loc[:, participant_keys]).groupby(
            participant_keys,
            sort=True,
        )[list(features)].mean()
        status = _support_label(participant_means.shape[0], 2)
        reason = (
            "participant_mean_covariance"
            if status == "estimated"
            else "fewer_than_two_participants"
        )
        return participant_means.reset_index(drop=True), status, reason

    session_means = matrix.join(group.loc[:, session_keys]).groupby(session_keys, sort=True)[
        list(features)
    ].mean()
    participant_sessions = session_means.reset_index().groupby(participant_keys, sort=True).size()
    if level == "session":
        participant_mean_of_sessions = session_means.groupby(level=participant_keys, sort=True).transform("mean")
        session_deviations = session_means - participant_mean_of_sessions
        status = _support_label(int((participant_sessions >= 2).sum()), 1)
        reason = (
            "session_deviations_from_participant_baseline"
            if status == "estimated"
            else "no_participants_with_repeated_sessions"
        )
        return session_deviations.reset_index(drop=True), status, reason

    if level == "within_session":
        session_lookup = group.loc[:, session_keys].join(matrix)
        session_mean_lookup = session_means.rename(
            columns={feature: f"{feature}_session_mean" for feature in features}
        )
        with_session_means = session_lookup.join(session_mean_lookup, on=session_keys)
        residuals = pd.DataFrame(index=group.index)
        for feature in features:
            residuals[feature] = (
                with_session_means[feature] - with_session_means[f"{feature}_session_mean"]
            )
        session_windows = group.groupby(session_keys, sort=True).size()
        status = _support_label(int((session_windows >= 2).sum()), 1)
        reason = (
            "window_deviations_from_session_baseline"
            if status == "estimated"
            else "no_sessions_with_repeated_windows"
        )
        return residuals.reset_index(drop=True), status, reason

    raise ValueError(f"unknown covariance level: {level}")


def _nested_variance_components(group: pd.DataFrame, feature: str) -> dict[str, Any]:
    participant_keys = ["source_dataset", "participant_id"]
    session_keys = ["source_dataset", "participant_id", "session_id"]
    values = pd.to_numeric(group[feature], errors="coerce")
    observed = group.loc[:, session_keys].copy()
    observed["_value"] = values
    observed = observed.dropna(subset=["_value"])
    n_observed = int(observed.shape[0])
    n_participants = int(observed.loc[:, participant_keys].drop_duplicates().shape[0])
    n_sessions = int(observed.loc[:, session_keys].drop_duplicates().shape[0])
    participant_session_counts = (
        observed.loc[:, session_keys]
        .drop_duplicates()
        .groupby(participant_keys, sort=True)
        .size()
    )
    session_window_counts = observed.groupby(session_keys, sort=True).size()

    total_variance = float(observed["_value"].var(ddof=1)) if n_observed > 1 else float("nan")
    participant_means = observed.groupby(participant_keys, sort=True)["_value"].mean()
    session_means = observed.groupby(session_keys, sort=True)["_value"].mean()

    between = (
        float(participant_means.var(ddof=1))
        if participant_means.shape[0] > 1
        else float("nan")
    )
    session_baselines = session_means.groupby(level=participant_keys, sort=True).transform("mean")
    session_deviations = session_means - session_baselines
    session_variance = (
        float(session_deviations.var(ddof=1))
        if int((participant_session_counts >= 2).sum()) > 0
        and session_deviations.dropna().shape[0] > 1
        else float("nan")
    )
    session_mean_lookup = session_means.rename("session_mean")
    with_session = observed.join(session_mean_lookup, on=session_keys)
    window_residuals = with_session["_value"] - with_session["session_mean"]
    window_variance = (
        float(window_residuals.var(ddof=1))
        if int((session_window_counts >= 2).sum()) > 0
        and window_residuals.dropna().shape[0] > 1
        else float("nan")
    )
    component_sum = np.nansum([between, session_variance, window_variance])
    return {
        "n_observed": n_observed,
        "n_participants": n_participants,
        "n_participants_with_repeated_sessions": int((participant_session_counts >= 2).sum()),
        "n_sessions": n_sessions,
        "n_sessions_with_repeated_windows": int((session_window_counts >= 2).sum()),
        "n_windows": n_observed,
        "total_variance": total_variance,
        "between_participant_variance": between,
        "session_within_participant_variance": session_variance,
        "window_within_session_variance": window_variance,
        "between_participant_fraction": _safe_fraction(between, component_sum),
        "session_within_participant_fraction": _safe_fraction(session_variance, component_sum),
        "window_within_session_fraction": _safe_fraction(window_variance, component_sum),
        "between_support_status": _support_label(participant_means.shape[0], 2),
        "session_support_status": _support_label(int((participant_session_counts >= 2).sum()), 1),
        "window_support_status": _support_label(int((session_window_counts >= 2).sum()), 1),
        "estimation_method": "nested_mean_deviation_variance_v1",
        "support_reason": _variance_support_reason(
            n_participants=n_participants,
            repeated_participants=int((participant_session_counts >= 2).sum()),
            repeated_window_sessions=int((session_window_counts >= 2).sum()),
        ),
    }


def _support_label(available_units: int, minimum_units: int) -> str:
    return "estimated" if available_units >= minimum_units else "unsupported"


def _variance_support_label(group: pd.DataFrame) -> str:
    participant_keys = ["source_dataset", "participant_id"]
    session_keys = ["source_dataset", "participant_id", "session_id"]
    n_participants = group.loc[:, participant_keys].drop_duplicates().shape[0]
    participant_sessions = (
        group.loc[:, session_keys].drop_duplicates().groupby(participant_keys, sort=True).size()
    )
    session_windows = group.groupby(session_keys, sort=True).size()
    if n_participants >= 2 and (participant_sessions >= 2).any() and (session_windows >= 2).any():
        return "full_nested_support"
    if n_participants >= 2:
        return "partial_between_person_support"
    return "unsupported"


def _variance_support_reason(
    *,
    n_participants: int,
    repeated_participants: int,
    repeated_window_sessions: int,
) -> str:
    reasons: list[str] = []
    if n_participants < 2:
        reasons.append("fewer_than_two_participants")
    if repeated_participants < 1:
        reasons.append("no_repeated_sessions_within_participant")
    if repeated_window_sessions < 1:
        reasons.append("no_repeated_windows_within_session")
    return ";".join(reasons) if reasons else "full_nested_support"


def _safe_fraction(numerator: float, denominator: float) -> float:
    if np.isnan(numerator) or np.isnan(denominator) or denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def _write_preflight_outputs(
    nuisance: dict[str, pd.DataFrame],
    *,
    output_dir: Path,
    schema_report: Any,
    input_mode: str,
    metadata: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in nuisance.items():
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    metadata_path = output_dir / "empirical_twin_preflight_provenance.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    paths["provenance"] = metadata_path
    report_path = output_dir / "empirical_twin_preflight_report.md"
    report_path.write_text(
        _preflight_markdown_report(
            nuisance,
            schema_report=schema_report,
            input_mode=input_mode,
            metadata=metadata,
        ),
        encoding="utf-8",
    )
    paths["report"] = report_path
    return paths


def _preflight_markdown_report(
    nuisance: dict[str, pd.DataFrame],
    *,
    schema_report: Any,
    input_mode: str,
    metadata: dict[str, Any],
) -> str:
    support = nuisance["template_support"]
    variance = nuisance["variance_decomposition"]
    temporal = nuisance["temporal_summary"]
    sparse_templates = support[
        (support["variance_component_support"] != "full_nested_support")
        | (support["feature_coverage_min"] < 0.8)
    ]
    between_rows = variance[variance["between_support_status"] == "estimated"]
    session_rows = variance[variance["session_support_status"] == "estimated"]
    window_rows = variance[variance["window_support_status"] == "estimated"]
    lag1_rows = temporal[temporal["lag1_support_status"] == "estimated"]
    practice_rows = temporal[temporal["practice_support_status"] == "estimated"]
    fatigue_rows = temporal[temporal["fatigue_support_status"] == "estimated"]
    return "\n".join(
        [
            "# M2.7 Empirical-Twin Preflight",
            "",
            f"input_mode: `{input_mode}`",
            f"static_contract: `{STATIC_TOURNAMENT_V2_CONTRACT.contract_id}`",
            "formal_claims_allowed: `false`",
            "model_recovery_performed: `false`",
            "",
            "## Schema",
            "",
            f"rows: {schema_report.n_rows}",
            f"participants: {schema_report.n_participants}",
            f"sessions: {metadata['n_sessions']}",
            f"sources: {schema_report.n_sources}",
            f"tasks: {schema_report.n_tasks}",
            f"source/task templates: {metadata['n_templates']}",
            "",
            "## Scientific Boundary",
            "",
            (
                "Empirical data are used for nuisance estimation only. "
                "No M0/M1/M2_EM/M3/M4 model recovery, APC inference, PACE inference "
                "or Trident-state inference is performed."
            ),
            "",
            "## Nuisance Audit",
            "",
            (
                "Adequate source/task templates are those with enough rows, repeated "
                "participants/sessions/windows and feature coverage to estimate the "
                "requested nuisance quantity. Sparse templates are exposed rather than "
                "pooled automatically."
            ),
            f"- templates with full nested variance support: {(support['variance_component_support'] == 'full_nested_support').sum()} / {support.shape[0]}",
            f"- sparse or low-coverage templates: {sparse_templates.shape[0]}",
            f"- between-person variance rows estimated: {between_rows.shape[0]}",
            f"- session-level variance rows estimated: {session_rows.shape[0]}",
            f"- within-session/window variance rows estimated: {window_rows.shape[0]}",
            f"- lag-1 temporal rows estimated: {lag1_rows.shape[0]}",
            f"- across-session practice rows estimated: {practice_rows.shape[0]}",
            f"- within-session fatigue rows estimated: {fatigue_rows.shape[0]}",
            "",
            "## Audit Questions",
            "",
            "1. Source/task support is reported in `template_support.csv`.",
            "2. Between-person versus within-person feature variance is reported in `variance_decomposition.csv`.",
            "3. Session-to-session variation is reported separately from stable participant differences.",
            "4. Window/block variation within session is reported separately from session variation.",
            "5. Temporal autocorrelation is estimated only within source x task x participant x session sequences.",
            "6. Across-session practice slopes are estimated only where participants have repeated sessions.",
            "7. Within-session fatigue slopes are estimated only where sessions have repeated windows.",
            "8. Between-person covariance is reported in `covariance_between_person.csv`.",
            "9. Within-person covariance is split into `covariance_session.csv` and `covariance_within_session.csv`.",
            "10. Missingness by source/task/feature is reported in `missingness_summary.csv`.",
            "11. Sparse templates are not discarded or pooled in this preflight.",
            "12. Source/task differences should be reviewed before any averaged human nuisance template is registered.",
            "13. Pooling or shrinkage, if needed, must be pre-registered before empirical-twin generation.",
            "",
            "## Outputs",
            "",
            *[f"- `{name}`: {frame.shape[0]} rows" for name, frame in nuisance.items()],
            "- `empirical_twin_preflight_provenance.json`: provenance and audit metadata",
            "",
            "No participant identifiers are written to this report.",
        ]
    )


def _input_metadata(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    source_repo, source_commit = _infer_source_repo_metadata(resolved)
    return {
        "path": str(resolved),
        "checksum": hash_file(resolved),
        "source_repository": source_repo,
        "source_commit_or_release": source_commit,
    }


def _infer_source_repo_metadata(path: Path) -> tuple[str, str]:
    for parent in [path.parent, *path.parents]:
        if parent.name == "flow-zone-zone-validation":
            commit = _repo_commit_or_unknown(parent)
            return "HRP-Transfer-Lab/flow-zone-zone-validation", commit
    return "unknown", "unknown"


def _repo_commit_or_unknown(repo_root: Path) -> str:
    try:
        return get_git_commit(repo_root)
    except Exception:
        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={repo_root.resolve().as_posix()}",
                "rev-parse",
                "HEAD",
            ],
            cwd=repo_root,
            check=False,
            text=True,
            capture_output=True,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"


def _assert_no_truth_inputs(frame: pd.DataFrame) -> None:
    truth_columns = ground_truth_columns(frame)
    if truth_columns:
        raise ValueError(
            "empirical nuisance input must not contain ground-truth columns: "
            + ", ".join(truth_columns)
        )


def _validate_preflight_config(config: dict[str, Any]) -> None:
    if config.get("study", {}).get("id") != PREFLIGHT_STUDY_ID:
        raise ValueError(f"study.id must be {PREFLIGHT_STUDY_ID}")
    contract = config.get("contract", {})
    if contract.get("static_tournament_contract") != STATIC_TOURNAMENT_V2_CONTRACT.contract_id:
        raise ValueError("preflight must use static_tournament_v2")
    if not bool(contract.get("nuisance_only_from_empirical_data")):
        raise ValueError("nuisance_only_from_empirical_data must be true")
    if bool(contract.get("formal_claims_allowed")):
        raise ValueError("preflight config must not allow formal claims")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run M2.7 empirical-twin nuisance preflight.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--input-table",
        action="append",
        default=None,
        help="Canonical empirical window table path; may be supplied more than once.",
    )
    parser.add_argument(
        "--no-fallback-fixture",
        action="store_true",
        help="Fail if no empirical input table is supplied.",
    )
    args = parser.parse_args(argv)
    result = run_empirical_twin_preflight(
        config_path=args.config,
        output_dir=args.output_dir,
        input_tables=args.input_table,
        allow_fallback_fixture=not args.no_fallback_fixture,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
