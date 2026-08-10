"""M2.7 empirical-twin informative pilot runner."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd

from trident_validation.config import load_yaml_config
from trident_validation.models.static_tournament_v2 import (
    STATIC_V2_MODEL_IDS,
    build_static_model_suite_v2,
)
from trident_validation.provenance import hash_file, hash_mapping
from trident_validation.splits import participant_train_test_split
from trident_validation.synthetic.empirical_twin_v1 import (
    EMPIRICAL_BACKGROUND_CONTRACT_ID,
    EMPIRICAL_BACKGROUND_CONTRACT_SHA,
    EMPIRICAL_TWIN_V1_ID,
    FEATURES,
    M2_7_EMPIRICAL_TWIN_WORLD_REGISTRY,
    STATIC_MODEL_SELECTION_CONTRACT_V2_SHA,
    _child_seed,
    _dataframe_hash,
    _directory_checksum,
    _n_participants,
    _participant_scores,
    _repo_commit_or_unknown,
    _run_id,
    _seed_schedule,
    generate_empirical_twin_dataset,
    load_background_spec,
)
from trident_validation.synthetic.recovery import (
    PRIMARY_METRIC,
    assert_no_ground_truth_columns,
    strip_ground_truth_columns,
)
from trident_validation.synthetic.selection_v2 import select_preferred_model_v2


PILOT_STUDY_ID = "empirical_twin_v1_pilot"
DEFAULT_CONFIG_PATH = Path("config/empirical_twin_v1_pilot.yaml")
PILOT_GENERATOR_SHA = "9699427176961a3064e9db8da23a3b287a441b1b"
BLAS_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}
EXPLORATORY_REPLICATES_PER_WORLD = 10
DISCRETE_MODEL_IDS = ("M3_three_profile_mixture", "M4_four_pace_profile_mixture")
CONTINUOUS_WORLD_IDS = ("ETW0", "ETW1", "ETW2")
FROZEN_DISCRETE_CONTRASTS = (
    ("ETW3", "M3_three_profile_mixture - M1_continuous_control_manifold"),
    ("ETW3", "M3_three_profile_mixture - M2_EM_v1"),
    ("ETW4", "M4_four_pace_profile_mixture - M3_three_profile_mixture"),
    ("ETW4", "M4_four_pace_profile_mixture - M1_continuous_control_manifold"),
    ("ETW4", "M4_four_pace_profile_mixture - M2_EM_v1"),
)


def run_empirical_twin_v1_pilot(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path | None = None,
    workers: int | None = None,
    prepare_only: bool = False,
    benchmark_only: bool = False,
) -> dict[str, Any]:
    _set_thread_environment()
    start = time.perf_counter()
    config_path = Path(config_path)
    config = load_yaml_config(config_path)
    _validate_pilot_config(config)
    target_dir = Path(output_dir or config["outputs"]["directory"])
    target_dir.mkdir(parents=True, exist_ok=True)
    schedule, schedule_hash = prepare_pilot_schedule(
        config,
        config_path=config_path,
        output_dir=target_dir,
    )
    if prepare_only:
        result = {
            "study_id": PILOT_STUDY_ID,
            "mode": "prepare_only",
            "n_units": int(schedule.shape[0]),
            "schedule_hash": schedule_hash,
            "formal_claims_allowed": False,
        }
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return result
    if benchmark_only:
        result = benchmark_representative_unit(
            config=config,
            config_path=config_path,
            output_dir=target_dir,
            schedule=schedule,
            schedule_hash=schedule_hash,
        )
        _atomic_write_json(target_dir / "pilot_runtime_benchmark.json", result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return result

    worker_count = int(workers or config["execution"]["process_workers"])
    checkpoint_dir = target_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        row
        for row in schedule.to_dict(orient="records")
        if not _checkpoint_valid(_checkpoint_path(checkpoint_dir, row), schedule_hash)
    ]
    _run_pending_units(
        pending,
        config_path=config_path,
        output_dir=target_dir,
        schedule_hash=schedule_hash,
        workers=worker_count,
        progress_interval=float(config["execution"]["progress_interval_seconds"]),
    )
    paths = aggregate_pilot_outputs(
        config=config,
        config_path=config_path,
        output_dir=target_dir,
        schedule=schedule,
        schedule_hash=schedule_hash,
        runtime_seconds=float(time.perf_counter() - start),
    )
    result = {
        "study_id": PILOT_STUDY_ID,
        "mode": "pilot",
        "n_units": int(schedule.shape[0]),
        "schedule_hash": schedule_hash,
        "formal_claims_allowed": False,
        "paths": {key: str(value) for key, value in paths.items()},
    }
    print(json.dumps({k: v for k, v in result.items() if k != "paths"}, indent=2), flush=True)
    return result


def prepare_pilot_schedule(
    config: dict[str, Any],
    *,
    config_path: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, str]:
    background = load_background_spec(
        config["inputs"]["acdc_preflight_dir"],
        config["inputs"]["paired_preflight_dir"],
    )
    schedule = _build_schedule(config, config_path=config_path, templates=background.template_summary)
    schedule_hash = _dataframe_hash(schedule)
    schedule_csv = Path(config["pilot_design"]["schedule_csv"])
    schedule_json = Path(config["pilot_design"]["schedule_json"])
    for existing_path in (schedule_csv, output_dir / "pilot_schedule.csv"):
        if existing_path.exists():
            existing = pd.read_csv(existing_path)
            existing_hash = _dataframe_hash(existing)
            if existing_hash != schedule_hash:
                if not _same_schedule_except_task_index(existing, schedule):
                    raise ValueError(
                        f"pilot schedule mismatch at {existing_path}: {existing_hash} != {schedule_hash}"
                    )
    _atomic_write_csv(schedule_csv, schedule)
    _atomic_write_json(schedule_json, schedule.to_dict(orient="records"))
    _atomic_write_csv(output_dir / "pilot_schedule.csv", schedule)
    _atomic_write_json(output_dir / "pilot_schedule.json", schedule.to_dict(orient="records"))
    manifest = _pilot_manifest(
        config,
        config_path=config_path,
        output_dir=output_dir,
        schedule=schedule,
        schedule_hash=schedule_hash,
    )
    _atomic_write_json(output_dir / "pilot_manifest.json", manifest)
    return schedule, schedule_hash


def benchmark_representative_unit(
    *,
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path,
    schedule: pd.DataFrame,
    schedule_hash: str,
) -> dict[str, Any]:
    row = schedule[schedule["world_id"] == "ETW2"].sort_values("replicate_index").iloc[0].to_dict()
    checkpoint_dir = output_dir / "benchmark_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result = _run_unit(
        row,
        config_path=str(config_path),
        checkpoint_dir=str(checkpoint_dir),
        schedule_hash=schedule_hash,
        benchmark=True,
    )["payload"]
    diagnostics = pd.DataFrame(result["fit_diagnostics"])
    generation = result["generation_summary"]
    unit_runtime = float(result["runtime"]["unit_runtime_seconds"])
    worker_count = float(config["execution"]["process_workers"])
    return {
        "study_id": PILOT_STUDY_ID,
        "benchmark_scope": "one_ETW2_unit",
        "model_winner_reported": False,
        "rows_per_twin": int(generation["n_rows"]),
        "participants_per_twin": int(generation["n_participants"]),
        "heldout_participants": int(result["split_audit"]["n_test_participants"]),
        "generation_runtime_seconds": float(result["runtime"]["generation_runtime_seconds"]),
        "complete_tournament_runtime_seconds": float(result["runtime"]["tournament_runtime_seconds"]),
        "runtime_per_twin_seconds": unit_runtime,
        "runtime_by_model_seconds": diagnostics.set_index("model_id")["runtime_seconds"].to_dict(),
        "m2_em_runtime_seconds": float(
            diagnostics.loc[diagnostics["model_id"] == "M2_EM_v1", "runtime_seconds"].iloc[0]
        ),
        "projected_runtime_50_units_seconds_serial": unit_runtime * 50.0,
        "projected_runtime_50_units_seconds_2_workers": unit_runtime * 50.0 / worker_count,
        "projected_disk_usage_mb": round(float(_directory_size(checkpoint_dir)) / 1_000_000.0 * 50.0, 3),
        "checkpoint_directory": str(output_dir / "checkpoints"),
    }


def aggregate_pilot_outputs(
    *,
    config: dict[str, Any],
    config_path: Path,
    output_dir: Path,
    schedule: pd.DataFrame,
    schedule_hash: str,
    runtime_seconds: float,
) -> dict[str, Path]:
    checkpoint_dir = output_dir / "checkpoints"
    payloads = [
        _read_checkpoint(_checkpoint_path(checkpoint_dir, row), schedule_hash)
        for row in schedule.to_dict(orient="records")
    ]
    complete = [payload for payload in payloads if payload and payload.get("status") == "complete"]
    failed = [payload for payload in payloads if payload and payload.get("status") == "failed"]
    model_scores = pd.DataFrame(_flatten(complete, "model_scores"))
    participant_scores = pd.DataFrame(_flatten(complete, "participant_scores"))
    selections = pd.DataFrame([payload["selection_summary"] for payload in complete])
    diagnostics = pd.DataFrame(_flatten(complete, "fit_diagnostics"))
    runtime = pd.DataFrame([payload["runtime"] for payload in complete + failed])
    background = pd.DataFrame(_flatten(complete, "generator_audit"))
    support = pd.DataFrame(_flatten(complete, "support_audit_summary"))
    contrasts = paired_contrasts(participant_scores)
    background_summary = background_realism_summary(background)
    runtime_summary = runtime_summary_table(runtime)
    world_selection_summary = world_by_model_selection_summary(selections)
    false_discrete_summary = false_discrete_pilot_summary(selections)
    etw0_diagnostics = etw0_m0_m1_diagnostics(participant_scores, diagnostics)
    etw2_diagnostics = etw2_m2_em_diagnostics(participant_scores, diagnostics)
    discrete_summary = discrete_contrast_summary(contrasts)
    failure_runtime = failure_runtime_summary(diagnostics, runtime)

    paths = {
        "model_scores": output_dir / "model_scores.csv",
        "participant_scores": output_dir / "participant_scores.csv",
        "selection_summary": output_dir / "selection_summary.csv",
        "paired_contrasts": output_dir / "paired_contrasts.csv",
        "fit_diagnostics": output_dir / "fit_diagnostics.csv",
        "runtime_summary": output_dir / "runtime_summary.csv",
        "background_realism_summary": output_dir / "background_realism_summary.csv",
        "support_summary": output_dir / "support_summary.csv",
        "world_model_selection_summary": output_dir / "world_model_selection_summary.csv",
        "false_discrete_pilot_summary": output_dir / "false_discrete_pilot_summary.csv",
        "etw0_m0_m1_diagnostics": output_dir / "etw0_m0_m1_diagnostics.csv",
        "etw2_m2_em_diagnostics": output_dir / "etw2_m2_em_diagnostics.csv",
        "discrete_contrasts_summary": output_dir / "discrete_contrasts_summary.csv",
        "failure_runtime_summary": output_dir / "failure_runtime_summary.csv",
    }
    for key, frame in {
        "model_scores": model_scores,
        "participant_scores": participant_scores,
        "selection_summary": selections,
        "paired_contrasts": contrasts,
        "fit_diagnostics": diagnostics,
        "runtime_summary": runtime_summary,
        "background_realism_summary": background_summary,
        "support_summary": support,
        "world_model_selection_summary": world_selection_summary,
        "false_discrete_pilot_summary": false_discrete_summary,
        "etw0_m0_m1_diagnostics": etw0_diagnostics,
        "etw2_m2_em_diagnostics": etw2_diagnostics,
        "discrete_contrasts_summary": discrete_summary,
        "failure_runtime_summary": failure_runtime,
    }.items():
        _atomic_write_csv(paths[key], frame)
    report_path = output_dir / "M2_7_EMPIRICAL_TWIN_PILOT_REPORT.md"
    _atomic_write_text(
        report_path,
        pilot_report(
            model_scores=model_scores,
            selections=selections,
            contrasts=contrasts,
            diagnostics=diagnostics,
            runtime_summary=runtime_summary,
            background_summary=background_summary,
            world_selection_summary=world_selection_summary,
            false_discrete_summary=false_discrete_summary,
            etw0_diagnostics=etw0_diagnostics,
            etw2_diagnostics=etw2_diagnostics,
            discrete_summary=discrete_summary,
            failure_runtime=failure_runtime,
        ),
    )
    manifest = _pilot_manifest(
        config,
        config_path=config_path,
        output_dir=output_dir,
        schedule=schedule,
        schedule_hash=schedule_hash,
    )
    manifest.update(
        {
            "n_complete_units": len(complete),
            "n_failed_units": len(failed),
            "runtime_seconds": runtime_seconds,
            "outputs": {key: str(path) for key, path in paths.items()},
            "output_checksums": {key: hash_file(path) for key, path in paths.items()},
            "report": str(report_path),
            "report_checksum": hash_file(report_path),
        }
    )
    manifest_path = output_dir / "pilot_manifest.json"
    _atomic_write_json(manifest_path, manifest)
    paths["manifest"] = manifest_path
    paths["report"] = report_path
    return paths


def paired_contrasts(participant_scores: pd.DataFrame) -> pd.DataFrame:
    if participant_scores.empty:
        return pd.DataFrame()
    planned = {
        "ETW0": [("M1_continuous_control_manifold", "M0_probabilistic_general_performance")],
        "ETW1": [
            ("M1_continuous_control_manifold", "M0_probabilistic_general_performance"),
            ("M1_continuous_control_manifold", "M2_EM_v1"),
        ],
        "ETW2": [("M2_EM_v1", "M1_continuous_control_manifold")],
        "ETW3": [
            ("M3_three_profile_mixture", "M1_continuous_control_manifold"),
            ("M3_three_profile_mixture", "M2_EM_v1"),
        ],
        "ETW4": [
            ("M4_four_pace_profile_mixture", "M3_three_profile_mixture"),
            ("M4_four_pace_profile_mixture", "M1_continuous_control_manifold"),
            ("M4_four_pace_profile_mixture", "M2_EM_v1"),
        ],
    }
    rows: list[dict[str, Any]] = []
    for (world, replicate), group in participant_scores.groupby(["world_id", "replicate_index"], sort=True):
        wide = group.pivot_table(index="participant_key", columns="model_id", values="heldout_log_density")
        for model_a, model_b in planned.get(str(world), []):
            if model_a not in wide or model_b not in wide:
                continue
            diff = (wide[model_a] - wide[model_b]).dropna()
            rows.append(
                {
                    "world_id": world,
                    "replicate_index": int(replicate),
                    "contrast": f"{model_a} - {model_b}",
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_participants": int(diff.shape[0]),
                    "mean_delta": float(diff.mean()) if not diff.empty else float("nan"),
                    "median_delta": float(diff.median()) if not diff.empty else float("nan"),
                    "sd_delta": float(diff.std(ddof=1)) if diff.shape[0] > 1 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def background_realism_summary(generator_audit: pd.DataFrame) -> pd.DataFrame:
    if generator_audit.empty:
        return pd.DataFrame()
    finite = generator_audit[
        np.isfinite(generator_audit["target"]) & np.isfinite(generator_audit["realised"])
    ].copy()
    if finite.empty:
        return pd.DataFrame()
    finite["parameter_family"] = finite["parameter"].map(_background_parameter_family)
    return (
        finite.groupby(["world_id", "parameter_family", "parameter", "feature"], dropna=False)
        .agg(
            n_rows=("absolute_delta", "size"),
            target_mean=("target", "mean"),
            realised_mean=("realised", "mean"),
            median_absolute_delta=("absolute_delta", "median"),
            p90_absolute_delta=("absolute_delta", lambda values: float(values.quantile(0.9))),
        )
        .reset_index()
    )


def world_by_model_selection_summary(selections: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "world_id",
        "model_id",
        "n_units",
        "numerical_best_count",
        "numerical_best_rate",
        "numerical_best_wilson95_low",
        "numerical_best_wilson95_high",
        "selected_count",
        "selected_rate",
        "selected_wilson95_low",
        "selected_wilson95_high",
        "same_tier_ambiguity_count",
        "same_tier_ambiguity_rate",
        "same_tier_ambiguity_wilson95_low",
        "same_tier_ambiguity_wilson95_high",
        "normalisation_sensitive_count",
        "normalisation_sensitive_rate",
        "normalisation_sensitive_wilson95_low",
        "normalisation_sensitive_wilson95_high",
        "exploratory_n_per_world",
        "rate_label",
    ]
    if selections.empty:
        return pd.DataFrame(columns=columns)
    complete = _complete_selections(selections)
    rows: list[dict[str, Any]] = []
    for world_id in M2_7_EMPIRICAL_TWIN_WORLD_REGISTRY:
        world = complete[complete["world_id"].astype(str) == world_id]
        denominator = int(world.shape[0])
        ambiguity_count = int(_boolean_sum(world.get("same_tier_ambiguous", pd.Series(dtype=bool))))
        normalisation_count = int(_boolean_sum(world.get("normalisation_sensitive", pd.Series(dtype=bool))))
        ambiguity_low, ambiguity_high = _wilson_interval(ambiguity_count, denominator)
        normalisation_low, normalisation_high = _wilson_interval(normalisation_count, denominator)
        for model_id in STATIC_V2_MODEL_IDS:
            numerical_count = int((world.get("numerical_best_model_id", pd.Series(dtype=str)).astype(str) == model_id).sum())
            selected_count = int((world.get("selected_model_id", pd.Series(dtype=str)).astype(str) == model_id).sum())
            numerical_low, numerical_high = _wilson_interval(numerical_count, denominator)
            selected_low, selected_high = _wilson_interval(selected_count, denominator)
            rows.append(
                {
                    "world_id": world_id,
                    "model_id": model_id,
                    "n_units": denominator,
                    "numerical_best_count": numerical_count,
                    "numerical_best_rate": _safe_rate(numerical_count, denominator),
                    "numerical_best_wilson95_low": numerical_low,
                    "numerical_best_wilson95_high": numerical_high,
                    "selected_count": selected_count,
                    "selected_rate": _safe_rate(selected_count, denominator),
                    "selected_wilson95_low": selected_low,
                    "selected_wilson95_high": selected_high,
                    "same_tier_ambiguity_count": ambiguity_count,
                    "same_tier_ambiguity_rate": _safe_rate(ambiguity_count, denominator),
                    "same_tier_ambiguity_wilson95_low": ambiguity_low,
                    "same_tier_ambiguity_wilson95_high": ambiguity_high,
                    "normalisation_sensitive_count": normalisation_count,
                    "normalisation_sensitive_rate": _safe_rate(normalisation_count, denominator),
                    "normalisation_sensitive_wilson95_low": normalisation_low,
                    "normalisation_sensitive_wilson95_high": normalisation_high,
                    "exploratory_n_per_world": EXPLORATORY_REPLICATES_PER_WORLD,
                    "rate_label": "exploratory_pilot_n10_imprecise",
                }
            )
    return pd.DataFrame(rows, columns=columns)


def false_discrete_pilot_summary(selections: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scope",
        "world_id",
        "selected_model_set",
        "numerator",
        "denominator",
        "rate",
        "wilson95_low",
        "wilson95_high",
        "diagnostic_label",
    ]
    if selections.empty:
        return pd.DataFrame(columns=columns)
    complete = _complete_selections(selections)
    rows: list[dict[str, Any]] = []
    for world_id in CONTINUOUS_WORLD_IDS:
        world = complete[complete["world_id"].astype(str) == world_id]
        rows.append(_false_discrete_row(world, scope="world", world_id=world_id))
    combined = complete[complete["world_id"].astype(str).isin(CONTINUOUS_WORLD_IDS)]
    rows.append(_false_discrete_row(combined, scope="combined_continuous_ETW0_ETW2", world_id="ETW0_ETW1_ETW2"))
    return pd.DataFrame(rows, columns=columns)


def etw0_m0_m1_diagnostics(participant_scores: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    diff = _paired_density_differences(
        participant_scores,
        world_id="ETW0",
        model_a="M1_continuous_control_manifold",
        model_b="M0_probabilistic_general_performance",
    )
    rows.append(
        _numeric_summary_row(
            diff["delta"] if "delta" in diff else pd.Series(dtype=float),
            metric="M1_minus_M0_participant_isolated_heldout_density",
            world_id="ETW0",
            model_id="M1_continuous_control_manifold",
            sample_unit="participant_pairs",
            interpretation_guard="do_not_treat_small_density_advantage_as_substantive_multidimensionality",
        )
    )
    m1 = _diagnostic_rows(diagnostics, world_id="ETW0", model_id="M1_continuous_control_manifold")
    for column in ("second_first_eigenvalue_ratio", "second_loading_fraction"):
        rows.append(
            _numeric_summary_row(
                pd.to_numeric(m1.get(column, pd.Series(dtype=float)), errors="coerce"),
                metric=f"M1_{column}",
                world_id="ETW0",
                model_id="M1_continuous_control_manifold",
                sample_unit="replicates",
                interpretation_guard="prospective_dimensionality_diagnostic_no_posthoc_threshold_change",
            )
        )
    return pd.DataFrame(rows)


def etw2_m2_em_diagnostics(participant_scores: pd.DataFrame, diagnostics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    diff = _paired_density_differences(
        participant_scores,
        world_id="ETW2",
        model_a="M2_EM_v1",
        model_b="M1_continuous_control_manifold",
    )
    rows.append(
        _numeric_summary_row(
            diff["delta"] if "delta" in diff else pd.Series(dtype=float),
            metric="M2_EM_v1_minus_M1_participant_isolated_heldout_density",
            world_id="ETW2",
            model_id="M2_EM_v1",
            sample_unit="participant_pairs",
            interpretation_guard="strict_convergence_not_newly_required_for_valid_score",
        )
    )
    em = _diagnostic_rows(diagnostics, world_id="ETW2", model_id="M2_EM_v1")
    for column in (
        "em_n_iter",
        "em_final_likelihood_change",
        "em_final_parameter_change",
        "residual_variance_mean",
        "residual_variance_min",
        "residual_variance_max",
        "runtime_seconds",
    ):
        rows.append(
            _numeric_summary_row(
                pd.to_numeric(em.get(column, pd.Series(dtype=float)), errors="coerce"),
                metric=column,
                world_id="ETW2",
                model_id="M2_EM_v1",
                sample_unit="replicates",
                interpretation_guard="strict_convergence_not_newly_required_for_valid_score",
            )
        )
    for column, metric in (
        ("em_converged", "em_converged_rate"),
        ("em_iteration_cap_flag", "em_iteration_cap_rate"),
    ):
        numerator = int(_boolean_sum(em.get(column, pd.Series(dtype=bool))))
        denominator = int(em.shape[0])
        low, high = _wilson_interval(numerator, denominator)
        rows.append(
            {
                "world_id": "ETW2",
                "model_id": "M2_EM_v1",
                "metric": metric,
                "sample_unit": "replicates",
                "n": denominator,
                "numerator": numerator,
                "denominator": denominator,
                "rate": _safe_rate(numerator, denominator),
                "wilson95_low": low,
                "wilson95_high": high,
                "mean": float("nan"),
                "median": float("nan"),
                "sd": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "interpretation_guard": "strict_convergence_not_newly_required_for_valid_score",
            }
        )
    return pd.DataFrame(rows)


def discrete_contrast_summary(contrasts: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "world_id",
        "contrast",
        "n_replicates",
        "n_participant_pairs_total",
        "mean_delta",
        "median_delta",
        "sd_replicate_mean_delta",
    ]
    if contrasts.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for world_id, contrast in FROZEN_DISCRETE_CONTRASTS:
        subset = contrasts[
            (contrasts["world_id"].astype(str) == world_id)
            & (contrasts["contrast"].astype(str) == contrast)
        ]
        values = pd.to_numeric(subset.get("mean_delta", pd.Series(dtype=float)), errors="coerce").dropna()
        rows.append(
            {
                "world_id": world_id,
                "contrast": contrast,
                "n_replicates": int(values.shape[0]),
                "n_participant_pairs_total": int(
                    pd.to_numeric(
                        subset.get("n_participants", pd.Series(dtype=float)),
                        errors="coerce",
                    ).sum()
                ),
                "mean_delta": float(values.mean()) if not values.empty else float("nan"),
                "median_delta": float(values.median()) if not values.empty else float("nan"),
                "sd_replicate_mean_delta": float(values.std(ddof=1)) if values.shape[0] > 1 else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def failure_runtime_summary(diagnostics: pd.DataFrame, runtime: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not diagnostics.empty:
        for (world_id, model_id), group in diagnostics.groupby(["world_id", "model_id"], dropna=False):
            denominator = int(group.shape[0])
            numerator = int((group.get("fit_status", pd.Series(dtype=str)).astype(str) != "success").sum())
            low, high = _wilson_interval(numerator, denominator)
            rows.append(
                {
                    "summary": "fit_failures_by_world_model",
                    "world_id": world_id,
                    "model_id": model_id,
                    "metric": "fit_failure_rate",
                    "numerator": numerator,
                    "denominator": denominator,
                    "rate": _safe_rate(numerator, denominator),
                    "wilson95_low": low,
                    "wilson95_high": high,
                    "n": denominator,
                    "mean": float("nan"),
                    "median": float("nan"),
                    "total": float("nan"),
                }
            )
            model_runtime = pd.to_numeric(group.get("runtime_seconds", pd.Series(dtype=float)), errors="coerce").dropna()
            rows.append(_runtime_summary_row(model_runtime, "runtime_by_world_model", world_id, model_id, "runtime_seconds"))
    if not runtime.empty:
        for world_id, group in runtime.groupby("world_id", dropna=False):
            denominator = int(group.shape[0])
            numerator = int((group.get("status", pd.Series(dtype=str)).astype(str) != "complete").sum())
            low, high = _wilson_interval(numerator, denominator)
            rows.append(
                {
                    "summary": "unit_failures_by_world",
                    "world_id": world_id,
                    "model_id": "ALL",
                    "metric": "unit_failure_rate",
                    "numerator": numerator,
                    "denominator": denominator,
                    "rate": _safe_rate(numerator, denominator),
                    "wilson95_low": low,
                    "wilson95_high": high,
                    "n": denominator,
                    "mean": float("nan"),
                    "median": float("nan"),
                    "total": float("nan"),
                }
            )
            values = pd.to_numeric(group.get("unit_runtime_seconds", pd.Series(dtype=float)), errors="coerce").dropna()
            rows.append(_runtime_summary_row(values, "runtime_by_world", world_id, "ALL", "unit_runtime_seconds"))
        values = pd.to_numeric(runtime.get("unit_runtime_seconds", pd.Series(dtype=float)), errors="coerce").dropna()
        rows.append(_runtime_summary_row(values, "runtime_total", "ALL", "ALL", "unit_runtime_seconds"))
    return pd.DataFrame(rows)


def _same_schedule_except_task_index(existing: pd.DataFrame, current: pd.DataFrame) -> bool:
    if list(existing.columns) != list(current.columns):
        return False
    if "task_index" not in existing.columns:
        return False
    existing_without_index = existing.drop(columns=["task_index"])
    current_without_index = current.drop(columns=["task_index"])
    return _dataframe_hash(existing_without_index) == _dataframe_hash(current_without_index)


def _background_parameter_family(parameter: object) -> str:
    name = str(parameter)
    if name == "source_task_shift":
        return "source_task_shift"
    if name.startswith("between_person"):
        return "between_person"
    if name.startswith("session_within_person"):
        return "session"
    if name.startswith("window_within_session"):
        return "window"
    if name.startswith("lag1"):
        return "lag1"
    if name.startswith("missingness"):
        return "missingness"
    if name.startswith("trial_count"):
        return "trial_count"
    if name.startswith("repeated_person_stability"):
        return "repeated_person_stability"
    return "other"


def _complete_selections(selections: pd.DataFrame) -> pd.DataFrame:
    if selections.empty:
        return selections.copy()
    if "selection_status" not in selections:
        return selections.copy()
    return selections[selections["selection_status"].astype(str) == "complete"].copy()


def _boolean_sum(values: pd.Series) -> int:
    if values.empty:
        return 0
    if values.dtype == bool:
        return int(values.sum())
    lowered = values.astype(str).str.lower()
    return int(lowered.isin({"true", "1", "yes"}).sum())


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _wilson_interval(numerator: int, denominator: int, *, z: float = 1.96) -> tuple[float, float]:
    if denominator <= 0:
        return float("nan"), float("nan")
    p = numerator / denominator
    denominator_adj = 1.0 + z**2 / denominator
    centre = (p + z**2 / (2.0 * denominator)) / denominator_adj
    half_width = (
        z
        * math.sqrt((p * (1.0 - p) + z**2 / (4.0 * denominator)) / denominator)
        / denominator_adj
    )
    return float(max(0.0, centre - half_width)), float(min(1.0, centre + half_width))


def _false_discrete_row(frame: pd.DataFrame, *, scope: str, world_id: str) -> dict[str, Any]:
    denominator = int(frame.shape[0])
    numerator = int(frame.get("selected_model_id", pd.Series(dtype=str)).astype(str).isin(DISCRETE_MODEL_IDS).sum())
    low, high = _wilson_interval(numerator, denominator)
    return {
        "scope": scope,
        "world_id": world_id,
        "selected_model_set": "M3_or_M4",
        "numerator": numerator,
        "denominator": denominator,
        "rate": _safe_rate(numerator, denominator),
        "wilson95_low": low,
        "wilson95_high": high,
        "diagnostic_label": "pilot_false_discrete_selection_diagnostic_not_confirmatory_error_rate",
    }


def _paired_density_differences(
    participant_scores: pd.DataFrame,
    *,
    world_id: str,
    model_a: str,
    model_b: str,
) -> pd.DataFrame:
    if participant_scores.empty:
        return pd.DataFrame(columns=["world_id", "replicate_index", "participant_key", "delta"])
    subset = participant_scores[participant_scores["world_id"].astype(str) == world_id]
    if subset.empty:
        return pd.DataFrame(columns=["world_id", "replicate_index", "participant_key", "delta"])
    wide = subset.pivot_table(
        index=["world_id", "replicate_index", "participant_key"],
        columns="model_id",
        values="heldout_log_density",
    )
    if model_a not in wide or model_b not in wide:
        return pd.DataFrame(columns=["world_id", "replicate_index", "participant_key", "delta"])
    output = wide[[model_a, model_b]].dropna().reset_index()
    output["delta"] = pd.to_numeric(output[model_a], errors="coerce") - pd.to_numeric(output[model_b], errors="coerce")
    return output.loc[:, ["world_id", "replicate_index", "participant_key", "delta"]]


def _diagnostic_rows(diagnostics: pd.DataFrame, *, world_id: str, model_id: str) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame()
    return diagnostics[
        (diagnostics["world_id"].astype(str) == world_id)
        & (diagnostics["model_id"].astype(str) == model_id)
    ].copy()


def _numeric_summary_row(
    values: pd.Series,
    *,
    metric: str,
    world_id: str,
    model_id: str,
    sample_unit: str,
    interpretation_guard: str,
) -> dict[str, Any]:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "world_id": world_id,
        "model_id": model_id,
        "metric": metric,
        "sample_unit": sample_unit,
        "n": int(finite.shape[0]),
        "numerator": np.nan,
        "denominator": np.nan,
        "rate": np.nan,
        "wilson95_low": np.nan,
        "wilson95_high": np.nan,
        "mean": float(finite.mean()) if not finite.empty else float("nan"),
        "median": float(finite.median()) if not finite.empty else float("nan"),
        "sd": float(finite.std(ddof=1)) if finite.shape[0] > 1 else float("nan"),
        "min": float(finite.min()) if not finite.empty else float("nan"),
        "max": float(finite.max()) if not finite.empty else float("nan"),
        "interpretation_guard": interpretation_guard,
    }


def _runtime_summary_row(
    values: pd.Series,
    summary: str,
    world_id: object,
    model_id: object,
    metric: str,
) -> dict[str, Any]:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "summary": summary,
        "world_id": world_id,
        "model_id": model_id,
        "metric": metric,
        "numerator": np.nan,
        "denominator": np.nan,
        "rate": np.nan,
        "wilson95_low": np.nan,
        "wilson95_high": np.nan,
        "n": int(finite.shape[0]),
        "mean": float(finite.mean()) if not finite.empty else float("nan"),
        "median": float(finite.median()) if not finite.empty else float("nan"),
        "total": float(finite.sum()) if not finite.empty else float("nan"),
    }


def _frame_section(title: str, frame: pd.DataFrame, note: str) -> list[str]:
    lines = [f"## {title}", "", note, ""]
    if frame.empty:
        lines += ["No rows available.", ""]
    else:
        lines += [frame.to_string(index=False), ""]
    return lines


def runtime_summary_table(runtime: pd.DataFrame) -> pd.DataFrame:
    if runtime.empty:
        return pd.DataFrame()
    total = {
        "unit_id": "ALL",
        "world_id": "ALL",
        "replicate_index": -1,
        "unit_runtime_seconds": float(runtime["unit_runtime_seconds"].sum()),
        "generation_runtime_seconds": float(runtime["generation_runtime_seconds"].sum()),
        "tournament_runtime_seconds": float(runtime["tournament_runtime_seconds"].sum()),
        "status": "summary",
    }
    return pd.concat([runtime, pd.DataFrame([total])], ignore_index=True)


def pilot_report(
    *,
    model_scores: pd.DataFrame,
    selections: pd.DataFrame,
    contrasts: pd.DataFrame,
    diagnostics: pd.DataFrame,
    runtime_summary: pd.DataFrame,
    background_summary: pd.DataFrame,
    world_selection_summary: pd.DataFrame,
    false_discrete_summary: pd.DataFrame,
    etw0_diagnostics: pd.DataFrame,
    etw2_diagnostics: pd.DataFrame,
    discrete_summary: pd.DataFrame,
    failure_runtime: pd.DataFrame,
) -> str:
    lines = [
        "# M2.7 Empirical-Twin Pilot Report",
        "",
        "**Status:** EXPLORATORY / INFORMATIVE PILOT",
        "",
        "n = 10 replicates/world.",
        "No confirmatory architecture-recovery claim.",
        "No confirmatory M2.7 recovery claim is made.",
        "No Trident-G/APC/PACE/neural-criticality/cusp validation claim is made.",
        "",
        "Pilot rates are exploratory and imprecise; Wilson intervals are descriptive.",
        "",
    ]
    lines += _frame_section(
        "World-By-Model Selection",
        world_selection_summary,
        "Numerical-best, selected-model, same-tier ambiguity and normalisation-sensitive counts/rates.",
    )
    lines += _frame_section(
        "False-Discrete Pilot Diagnostic",
        false_discrete_summary,
        "Selected model in {M3, M4} under continuous ETW0-ETW2 truth; not a confirmatory error rate.",
    )
    lines += _frame_section(
        "ETW0 M0/M1 Diagnostics",
        etw0_diagnostics,
        "A small M1 density advantage alone is not interpreted as substantive multidimensionality.",
    )
    lines += _frame_section(
        "ETW2 M2_EM_v1 Diagnostics",
        etw2_diagnostics,
        "Strict EM convergence is not newly required for a valid score under the frozen contract.",
    )
    lines += _frame_section(
        "ETW3/ETW4 Discrete Contrasts",
        discrete_summary,
        "Frozen discrete-world contrasts specified before outcome inspection.",
    )
    lines += _frame_section(
        "Failures And Runtime",
        failure_runtime,
        "Fit failures, unit failures, runtime/twin, runtime/model and total runtime.",
    )
    if not selections.empty:
        selected = selections.groupby(["world_id", "selected_model_id"]).size().reset_index(name="count")
        numerical = selections.groupby(["world_id", "numerical_best_model_id"]).size().reset_index(name="count")
        ambiguity = selections.groupby("world_id")["same_tier_ambiguous"].mean().reset_index(name="ambiguity_rate")
        lines += [
            "## Legacy Selection Detail",
            "",
            selected.to_string(index=False),
            "",
            "Numerical-best counts:",
            "",
            numerical.to_string(index=False),
            "",
            "Same-tier ambiguity rate:",
            "",
            ambiguity.to_string(index=False),
            "",
        ]
    if not model_scores.empty:
        summary = (
            model_scores.groupby(["world_id", "model_id"], dropna=False)[PRIMARY_METRIC]
            .agg(["mean", "median", "count"])
            .reset_index()
        )
        lines += ["## Held-Out Density", "", summary.to_string(index=False), ""]
    if not contrasts.empty:
        contrast_summary = (
            contrasts.groupby(["world_id", "contrast"], dropna=False)["mean_delta"]
            .agg(["mean", "median", "count"])
            .reset_index()
        )
        lines += ["## Legacy Frozen Contrasts Detail", "", contrast_summary.to_string(index=False), ""]
    if not diagnostics.empty:
        failure = diagnostics.groupby(["world_id", "model_id", "fit_status"]).size().reset_index(name="count")
        lines += ["## Legacy Fit Diagnostics Detail", "", failure.to_string(index=False), ""]
    if not runtime_summary.empty:
        lines += ["## Runtime Detail", "", runtime_summary.tail(8).to_string(index=False), ""]
    if not background_summary.empty:
        lines += ["## Background Realism", "", background_summary.head(80).to_string(index=False), ""]
    lines += [
        "## Boundary",
        "",
        "Pilot outcomes are allowed to be weak, ambiguous or unfavourable.",
        "No empirical-background parameter, ETW truth or model definition is changed here.",
        "",
    ]
    return "\n".join(lines)


def _run_pending_units(
    pending: list[dict[str, Any]],
    *,
    config_path: Path,
    output_dir: Path,
    schedule_hash: str,
    workers: int,
    progress_interval: float,
) -> None:
    if not pending:
        print("M2.7 pilot | no pending units; resume found all checkpoints complete", flush=True)
        return
    checkpoint_dir = output_dir / "checkpoints"
    total = len(pending)
    complete = 0
    failed = 0
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=_set_thread_environment) as pool:
        futures = {
            pool.submit(
                _run_unit,
                row,
                config_path=str(config_path),
                checkpoint_dir=str(checkpoint_dir),
                schedule_hash=schedule_hash,
                benchmark=False,
            ): row
            for row in pending
        }
        while futures:
            done, _ = wait(futures, timeout=progress_interval, return_when=FIRST_COMPLETED)
            if not done:
                _print_progress(total, complete, failed, started)
                continue
            for future in done:
                row = futures.pop(future)
                try:
                    status = future.result()["payload"]["status"]
                except Exception as exc:  # defensive worker failure path
                    _atomic_write_checkpoint(
                        _checkpoint_path(checkpoint_dir, row),
                        _failed_payload(row, schedule_hash=schedule_hash, error=repr(exc)),
                    )
                    status = "failed"
                if status == "complete":
                    complete += 1
                else:
                    failed += 1
            _print_progress(total, complete, failed, started)


def _run_unit(
    row: dict[str, Any],
    *,
    config_path: str,
    checkpoint_dir: str,
    schedule_hash: str,
    benchmark: bool,
) -> dict[str, Any]:
    _set_thread_environment()
    unit_start = time.perf_counter()
    config = load_yaml_config(config_path)
    background = load_background_spec(
        config["inputs"]["acdc_preflight_dir"],
        config["inputs"]["paired_preflight_dir"],
    )
    generator_config = dict(config["generator"])
    generator_config.update(
        {
            "template_records": json.loads(str(row["template_records_json"])),
            "capture_component_audit": True,
        }
    )
    generation_start = time.perf_counter()
    run = generate_empirical_twin_dataset(
        world_id=str(row["world_id"]),  # type: ignore[arg-type]
        replicate_index=int(row["replicate_index"]),
        background=background,
        config=generator_config,
    )
    generation_runtime = time.perf_counter() - generation_start
    model_frame = strip_ground_truth_columns(run.dataset)
    assert_no_ground_truth_columns(model_frame)
    split = participant_train_test_split(
        model_frame,
        test_size=float(config["tournament"]["test_fraction"]),
        seed=int(row["split_seed"]),
    )
    tournament_start = time.perf_counter()
    model_scores, participant_scores, fit_diagnostics = _fit_score_models_for_pilot(
        model_frame,
        train_indices=tuple(split.train_indices),
        test_indices=tuple(split.test_indices),
        random_state=int(row["model_seed"]),
        unit=row,
    )
    selection = _selection_summary(
        model_scores,
        participant_scores,
        unit=row,
        practical_equivalence_margin=float(config["tournament"]["practical_equivalence_margin"]),
        paired_ci_z=float(config["tournament"]["paired_ci_z"]),
    )
    status = "complete" if selection.get("selection_status") == "complete" else "failed"
    split_audit = {
        "unit_id": row["unit_id"],
        "world_id": row["world_id"],
        "replicate_index": int(row["replicate_index"]),
        "n_train_rows": len(split.train_indices),
        "n_test_rows": len(split.test_indices),
        "n_train_participants": _n_participants(model_frame.loc[list(split.train_indices)]),
        "n_test_participants": _n_participants(model_frame.loc[list(split.test_indices)]),
        "participant_isolated": True,
        "truth_leakage": "none_detected",
    }
    payload = {
        "status": status,
        "unit_id": row["unit_id"],
        "schedule_hash": schedule_hash,
        "unit": row,
        "generation_summary": run.generation_summary,
        "split_audit": split_audit,
        "model_scores": model_scores.to_dict(orient="records"),
        "participant_scores": participant_scores.to_dict(orient="records"),
        "selection_summary": selection,
        "fit_diagnostics": fit_diagnostics.to_dict(orient="records"),
        "generator_audit": run.generator_audit.assign(unit_id=row["unit_id"]).to_dict(orient="records"),
        "support_audit_summary": (
            run.support_audit.groupby(["quantity", "support_status"], dropna=False)
            .size()
            .reset_index(name="count")
            .assign(unit_id=row["unit_id"], world_id=row["world_id"], replicate_index=int(row["replicate_index"]))
            .to_dict(orient="records")
        ),
        "runtime": {
            "unit_id": row["unit_id"],
            "world_id": row["world_id"],
            "replicate_index": int(row["replicate_index"]),
            "unit_runtime_seconds": float(time.perf_counter() - unit_start),
            "generation_runtime_seconds": float(generation_runtime),
            "tournament_runtime_seconds": float(time.perf_counter() - tournament_start),
            "status": status,
        },
        "benchmark": bool(benchmark),
    }
    _atomic_write_checkpoint(_checkpoint_path(Path(checkpoint_dir), row), payload)
    return {"payload": payload}


def _fit_score_models_for_pilot(
    frame_without_truth: pd.DataFrame,
    *,
    train_indices: tuple[int, ...],
    test_indices: tuple[int, ...],
    random_state: int,
    unit: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = frame_without_truth.loc[list(train_indices)].copy()
    test = frame_without_truth.loc[list(test_indices)].copy()
    observed_counts = test.loc[:, list(FEATURES)].notna().sum(axis=1)
    total_observed = int(observed_counts.sum())
    model_rows: list[dict[str, Any]] = []
    participant_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for model in build_static_model_suite_v2(feature_columns=FEATURES, random_state=random_state):
        started = time.perf_counter()
        base = _unit_model_fields(unit, model.model_id)
        try:
            fitted = model.fit(train)
            sample_scores = fitted.score_samples(test).astype(float)
            holdout = fitted.score_holdout(test)
            valid = sample_scores.dropna()
            participant_means = _participant_scores(test, sample_scores)
            total = float(valid.sum()) if not valid.empty else float("nan")
            diagnostics = holdout.diagnostics or {}
            model_rows.append(
                {
                    **base,
                    "primary_metric": holdout.primary_metric or PRIMARY_METRIC,
                    "heldout_log_likelihood_total": total,
                    PRIMARY_METRIC: float(valid.mean()) if not valid.empty else float("nan"),
                    "heldout_log_density_participant_weighted": float(participant_means.mean()),
                    "heldout_log_density_mean_per_observed_feature": (
                        float(total / total_observed)
                        if total_observed and np.isfinite(total)
                        else float("nan")
                    ),
                    "n_valid_windows": int(valid.shape[0]),
                    "n_test_participants": int(participant_means.shape[0]),
                    "n_observed_feature_values": total_observed,
                    **{f"diagnostic_{key}": float(value) for key, value in diagnostics.items()},
                }
            )
            for participant_key, value in participant_means.items():
                participant_rows.append(
                    {
                        **base,
                        "participant_key": "|".join(str(part) for part in participant_key),
                        "heldout_log_density": float(value),
                    }
                )
            diagnostic_rows.append(
                {
                    **base,
                    "fit_status": "success",
                    "runtime_seconds": float(time.perf_counter() - started),
                    **_model_fit_diagnostics(fitted),
                }
            )
        except Exception as exc:
            model_rows.append({**base, "primary_metric": PRIMARY_METRIC, PRIMARY_METRIC: float("nan")})
            diagnostic_rows.append(
                {
                    **base,
                    "fit_status": "failed",
                    "runtime_seconds": float(time.perf_counter() - started),
                    "error": repr(exc),
                }
            )
    return pd.DataFrame(model_rows), pd.DataFrame(participant_rows), pd.DataFrame(diagnostic_rows)


def _selection_summary(
    model_scores: pd.DataFrame,
    participant_scores: pd.DataFrame,
    *,
    unit: dict[str, Any],
    practical_equivalence_margin: float,
    paired_ci_z: float,
) -> dict[str, Any]:
    base = {
        "unit_id": unit["unit_id"],
        "world_id": unit["world_id"],
        "replicate_index": int(unit["replicate_index"]),
    }
    try:
        wide = participant_scores.pivot_table(
            index="participant_key",
            columns="model_id",
            values="heldout_log_density",
        )
        selection = select_preferred_model_v2(
            model_scores,
            wide,
            practical_equivalence_margin=practical_equivalence_margin,
            paired_ci_z=paired_ci_z,
            model_ids=STATIC_V2_MODEL_IDS,
        )
        return {
            **base,
            "selection_status": "complete",
            "selected_model_id": selection.selected_model_id,
            "numerical_best_model_id": selection.numerical_best_model_id,
            "selection_reason": selection.selection_reason,
            "selected_tier": int(selection.selected_tier),
            "numerical_best_tier": int(selection.numerical_best_tier),
            "same_tier_ambiguous": bool(selection.same_tier_ambiguous),
            "ambiguous_model_ids": ";".join(selection.ambiguous_model_ids),
            "normalisation_sensitive": bool(selection.normalisation_sensitive),
        }
    except Exception as exc:
        return {**base, "selection_status": "failed", "selection_error": repr(exc)}


def _build_schedule(
    config: dict[str, Any],
    *,
    config_path: Path,
    templates: pd.DataFrame,
) -> pd.DataFrame:
    tasks = tuple(str(task) for task in config["pilot_design"]["matched_support_intersection"])
    worlds = tuple(str(world) for world in config["generator"]["worlds"])
    replicates = int(config["generator"]["replicates_per_world"])
    schedule_seed = int(config["pilot_design"]["schedule_seed"])
    master_seed = int(config["generator"]["seed"])
    by_task = _eligible_templates_by_task(templates, tasks)
    selected = _selected_templates_without_replacement(by_task, replicates=replicates, seed=schedule_seed)
    rows: list[dict[str, Any]] = []
    for replicate_index in range(replicates):
        pair = [selected[task].iloc[replicate_index] for task in tasks]
        template_json = json.dumps([_json_ready(row.to_dict()) for row in pair], sort_keys=True)
        template_ids = {
            f"{task.lower()}_template_source_dataset": str(selected[task].iloc[replicate_index]["source_dataset"])
            for task in tasks
        }
        template_ids.update(
            {
                f"{task.lower()}_template_id": str(selected[task].iloc[replicate_index]["source_dataset"])
                for task in tasks
            }
        )
        template_ids.update(
            {
                f"{task.lower()}_template_task_id": str(selected[task].iloc[replicate_index]["task_id"])
                for task in tasks
            }
        )
        for world_offset, world_id in enumerate(worlds):
            seed_schedule = _seed_schedule(master_seed, world_id, replicate_index)
            run_id = _run_id(EMPIRICAL_TWIN_V1_ID, world_id, replicate_index, master_seed)
            rows.append(
                {
                    "task_index": replicate_index * len(worlds) + world_offset,
                    "unit_id": f"{world_id}_replicate_{replicate_index:03d}",
                    "world_id": world_id,
                    "aligned_model_id": M2_7_EMPIRICAL_TWIN_WORLD_REGISTRY[world_id]["aligned_model_id"],
                    "replicate_index": replicate_index,
                    **template_ids,
                    "template_records_json": template_json,
                    **seed_schedule,
                    "split_seed": _child_seed(run_id, "split"),
                    "model_seed": _child_seed(run_id, "model"),
                    "generator_sha": config["contract"].get("generator_commit", PILOT_GENERATOR_SHA),
                    "empirical_background_contract_sha": config["contract"]["empirical_background_contract_commit"],
                    "static_model_contract_sha": config["contract"]["static_tournament_contract_commit"],
                    "pilot_config_hash": hash_file(config_path),
                }
            )
    return pd.DataFrame(rows)


def _eligible_templates_by_task(templates: pd.DataFrame, tasks: Sequence[str]) -> dict[str, pd.DataFrame]:
    frame = templates.copy()
    frame["n_rows"] = pd.to_numeric(frame["n_rows"], errors="coerce")
    frame["n_participants"] = pd.to_numeric(frame["n_participants"], errors="coerce")
    coverage_mask = np.ones(frame.shape[0], dtype=bool)
    for feature in FEATURES:
        coverage_mask &= pd.to_numeric(frame[f"{feature}_coverage"], errors="coerce").to_numpy() >= 0.95
    eligible = frame[
        frame["task_id"].astype(str).isin(tasks)
        & (frame["n_rows"] >= 100)
        & (frame["n_participants"] >= 50)
        & coverage_mask
    ].copy()
    output: dict[str, pd.DataFrame] = {}
    for task in tasks:
        task_rows = eligible[eligible["task_id"].astype(str) == str(task)].sort_values(
            ["source_dataset", "task_id"],
            kind="mergesort",
        )
        if task_rows.empty:
            raise ValueError(f"no contract-eligible templates for task {task}")
        output[str(task)] = task_rows.reset_index(drop=True)
    return output


def _selected_templates_without_replacement(
    by_task: dict[str, pd.DataFrame],
    *,
    replicates: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    selected: dict[str, pd.DataFrame] = {}
    for task, rows in by_task.items():
        if rows.shape[0] < replicates:
            raise ValueError(f"not enough eligible {task} templates for without-replacement schedule")
        order = rng.permutation(rows.shape[0])[:replicates]
        selected[task] = rows.iloc[order].reset_index(drop=True)
    return selected


def _pilot_manifest(
    config: dict[str, Any],
    *,
    config_path: Path,
    output_dir: Path,
    schedule: pd.DataFrame,
    schedule_hash: str,
) -> dict[str, Any]:
    stability = Path(config["inputs"]["paired_preflight_dir"]) / "paired_session_repeated_person_stability.csv"
    return {
        "study_id": PILOT_STUDY_ID,
        "status": "exploratory_informative_pilot",
        "formal_claims_allowed": False,
        "model_recovery_claim_allowed": False,
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "empirical_background_contract": EMPIRICAL_BACKGROUND_CONTRACT_ID,
        "empirical_background_contract_sha": EMPIRICAL_BACKGROUND_CONTRACT_SHA,
        "static_tournament_contract_sha": STATIC_MODEL_SELECTION_CONTRACT_V2_SHA,
        "generator_sha": config["contract"].get("generator_commit", PILOT_GENERATOR_SHA),
        "audit_correction_commit": config["contract"]["audit_correction_commit"],
        "validation_repo_commit": _repo_commit_or_unknown(Path.cwd()),
        "config_path": str(config_path),
        "config_hash": hash_file(config_path),
        "config_content_hash": hash_mapping(config),
        "schedule_hash": schedule_hash,
        "n_units": int(schedule.shape[0]),
        "schedule_seed": int(config["pilot_design"]["schedule_seed"]),
        "matched_support_intersection": list(config["pilot_design"]["matched_support_intersection"]),
        "acdc_aggregate_checksum": _directory_checksum(Path(config["inputs"]["acdc_preflight_dir"])),
        "paired_aggregate_checksum": _directory_checksum(Path(config["inputs"]["paired_preflight_dir"])),
        "paired_repeated_person_stability_checksum": hash_file(stability),
        "processed_paired_source_checksum": config["inputs"]["paired_processed_source_checksum"],
        "output_dir": str(output_dir),
    }


def _checkpoint_path(checkpoint_dir: Path, row: dict[str, Any] | pd.Series) -> Path:
    return checkpoint_dir / f"{row['unit_id']}.json"


def _checkpoint_valid(path: Path, schedule_hash: str) -> bool:
    payload = _read_checkpoint(path, schedule_hash)
    return bool(payload and payload.get("status") == "complete")


def _read_checkpoint(path: Path, schedule_hash: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    checksum = payload.get("unit_checksum")
    copy = dict(payload)
    copy.pop("unit_checksum", None)
    if payload.get("schedule_hash") != schedule_hash:
        return None
    if checksum != hash_mapping(copy):
        return None
    return payload


def _atomic_write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    clean = _json_ready(payload)
    copy = dict(clean)
    copy.pop("unit_checksum", None)
    clean["unit_checksum"] = hash_mapping(copy)
    _atomic_write_json(path, clean)


def _failed_payload(row: dict[str, Any], *, schedule_hash: str, error: str) -> dict[str, Any]:
    return {
        "status": "failed",
        "unit_id": row["unit_id"],
        "schedule_hash": schedule_hash,
        "unit": row,
        "error": error,
        "runtime": {
            "unit_id": row["unit_id"],
            "world_id": row["world_id"],
            "replicate_index": int(row["replicate_index"]),
            "unit_runtime_seconds": float("nan"),
            "generation_runtime_seconds": float("nan"),
            "tournament_runtime_seconds": float("nan"),
            "status": "failed",
        },
    }


def _unit_model_fields(unit: dict[str, Any], model_id: str) -> dict[str, Any]:
    return {
        "unit_id": unit["unit_id"],
        "world_id": unit["world_id"],
        "aligned_model_id": unit["aligned_model_id"],
        "replicate_index": int(unit["replicate_index"]),
        "model_id": model_id,
    }


def _model_fit_diagnostics(model: Any) -> dict[str, Any]:
    output: dict[str, Any] = {}
    if model.model_id == "M1_continuous_control_manifold":
        eigen = np.asarray(getattr(model, "eigenvalues_", []), dtype=float)
        loadings = np.asarray(getattr(model, "loadings_", []), dtype=float)
        output["second_first_eigenvalue_ratio"] = (
            float(eigen[1] / eigen[0]) if eigen.shape[0] > 1 and eigen[0] > 0 else float("nan")
        )
        norms = np.linalg.norm(loadings, axis=0) if loadings.ndim == 2 else np.array([])
        output["second_loading_fraction"] = (
            float(norms[1] / norms.sum()) if norms.shape[0] > 1 and norms.sum() > 0 else float("nan")
        )
    metadata = model.get_model_metadata()
    em = metadata.get("em") if isinstance(metadata, dict) else None
    if isinstance(em, dict):
        n_iter = int(em.get("n_iter", 0))
        max_iter = int(em.get("max_iter", 0))
        output.update(
            {
                "em_n_iter": n_iter,
                "em_converged": bool(em.get("converged", False)),
                "em_iteration_cap_flag": bool(max_iter and n_iter >= max_iter),
                "em_final_likelihood_change": float(em.get("final_likelihood_change", float("nan"))),
                "em_final_parameter_change": float(em.get("final_parameter_change", float("nan"))),
                "em_train_log_likelihood": float(em.get("train_log_likelihood", float("nan"))),
                "em_max_iter": max_iter,
            }
        )
    if hasattr(model, "residual_variance_"):
        residual = np.asarray(getattr(model, "residual_variance_"), dtype=float)
        output["residual_variance_mean"] = float(np.nanmean(residual))
        output["residual_variance_min"] = float(np.nanmin(residual))
        output["residual_variance_max"] = float(np.nanmax(residual))
    return output


def _print_progress(total: int, complete: int, failed: int, started: float) -> None:
    elapsed = time.perf_counter() - started
    done = complete + failed
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else float("nan")
    print(
        f"M2.7 pilot | {done}/{total} units finished | complete {complete} | "
        f"failed {failed} | elapsed {elapsed:.1f}s | ETA {eta:.1f}s",
        flush=True,
    )


def _set_thread_environment() -> None:
    for key, value in BLAS_THREAD_ENV.items():
        os.environ.setdefault(key, value)


def _validate_pilot_config(config: dict[str, Any]) -> None:
    if config.get("study", {}).get("id") != PILOT_STUDY_ID:
        raise ValueError(f"study.id must be {PILOT_STUDY_ID}")
    if bool(config.get("contract", {}).get("formal_claims_allowed")):
        raise ValueError("pilot config must not allow formal claims")
    if config.get("contract", {}).get("empirical_background_contract") != EMPIRICAL_BACKGROUND_CONTRACT_ID:
        raise ValueError("pilot must use EMPIRICAL_BACKGROUND_CONTRACT_V1")
    if config.get("contract", {}).get("empirical_background_contract_commit") != EMPIRICAL_BACKGROUND_CONTRACT_SHA:
        raise ValueError("pilot empirical-background contract SHA mismatch")
    if config.get("contract", {}).get("static_tournament_contract_commit") != STATIC_MODEL_SELECTION_CONTRACT_V2_SHA:
        raise ValueError("pilot static contract SHA mismatch")
    if not _is_disabled(config.get("contract", {}).get("session_order_context_trend")):
        raise ValueError("primary pilot requires session_order_context_trend: off")
    if not _is_disabled(config.get("generator", {}).get("session_order_context_trend")):
        raise ValueError("primary pilot requires session_order_context_trend: off")
    if not _is_disabled(config.get("generator", {}).get("paired_cross_task_session_covariance")):
        raise ValueError("primary pilot requires paired_cross_task_session_covariance: off")
    if tuple(config.get("tournament", {}).get("models", [])) != STATIC_V2_MODEL_IDS:
        raise ValueError("pilot model list must match static V2 model ids")
    if tuple(config.get("pilot_design", {}).get("matched_support_intersection", [])) != ("Stroop", "Flanker"):
        raise ValueError("primary pilot requires matched-support task intersection: Stroop, Flanker")
    if int(config.get("generator", {}).get("n_templates", 0)) != 2:
        raise ValueError("primary pilot requires n_templates: 2")
    if int(config.get("generator", {}).get("participants_per_template", 0)) != 24:
        raise ValueError("primary pilot requires participants_per_template: 24")
    if int(config.get("generator", {}).get("sessions_per_participant", 0)) != 2:
        raise ValueError("primary pilot requires sessions_per_participant: 2")
    if int(config.get("generator", {}).get("windows_per_session", 0)) != 16:
        raise ValueError("primary pilot requires windows_per_session: 16")
    if tuple(config.get("generator", {}).get("worlds", [])) != tuple(M2_7_EMPIRICAL_TWIN_WORLD_REGISTRY):
        raise ValueError("primary pilot worlds must be ETW0-ETW4 in registry order")


def _is_disabled(value: Any) -> bool:
    if isinstance(value, bool):
        return not value
    return str(value).strip().lower() in {"off", "false", "0", "no"}


def _flatten(payloads: Sequence[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        rows.extend(payload.get(key, []))
    return rows


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run M2.7 empirical-twin informative pilot.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--benchmark-only", action="store_true")
    args = parser.parse_args(argv)
    run_empirical_twin_v1_pilot(
        config_path=args.config,
        output_dir=args.output_dir,
        workers=args.workers,
        prepare_only=args.prepare_only,
        benchmark_only=args.benchmark_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
