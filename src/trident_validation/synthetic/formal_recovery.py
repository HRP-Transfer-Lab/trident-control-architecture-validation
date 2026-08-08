"""Formal synthetic W0-W4 model-recovery study runner."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd

from trident_validation.config import load_yaml_config
from trident_validation.models.static_tournament import STATIC_MODEL_IDS
from trident_validation.provenance import (
    get_git_commit,
    hash_file,
    hash_mapping,
    package_versions,
    write_run_manifest,
)
from trident_validation.splits import participant_train_test_split
from trident_validation.synthetic.fixtures import CORE_SYNTHETIC_FEATURES
from trident_validation.synthetic.recovery import (
    DISCRETE_MODEL_IDS,
    NULL_OR_CONTINUOUS_WORLD_IDS,
    PRIMARY_METRIC,
    _component_recovery_after_freeze,
    _model_complexity_rank,
    _split_audit,
    assert_no_ground_truth_columns,
    fit_score_models_without_truth,
    select_preferred_model,
    strip_ground_truth_columns,
)
from trident_validation.synthetic.worlds import (
    STATIC_SYNTHETIC_WORLD_IDS,
    WORLD_MODEL_ALIGNMENT,
    StaticWorldId,
    make_static_synthetic_world,
)


FORMAL_STUDY_ID = "formal_synthetic_recovery_v1"
FORMAL_CONFIG_PATH = Path("config/formal_synthetic_recovery_v1.yaml")
DEFAULT_OUTPUT_DIR = Path("reports/generated")
DEFAULT_MANIFEST_PATH = Path("manifests/formal_synthetic_recovery_v1.json")
DEFAULT_REPORT_PATH = Path("reports/formal_synthetic_recovery_v1.md")
DEFAULT_CHECKPOINT_DIR = Path("reports/generated/checkpoints/formal_synthetic_recovery_v1")
SUMMARY_OUTPUT_NAMES = {
    "synthetic_recovery_matrix": "synthetic_recovery_matrix.csv",
    "synthetic_correct_model_recovery": "synthetic_correct_model_recovery.csv",
    "synthetic_false_discrete_rate": "synthetic_false_discrete_rate.csv",
    "synthetic_model_confusions": "synthetic_model_confusions.csv",
    "synthetic_stress_recovery": "synthetic_stress_recovery.csv",
    "synthetic_participant_score_differences": "synthetic_participant_score_differences.csv",
    "synthetic_component_recovery": "synthetic_component_recovery.csv",
}


class FormalRecoveryError(RuntimeError):
    """Raised when the formal synthetic recovery protocol cannot run safely."""


@dataclass(frozen=True)
class FormalRecoveryTask:
    """One synthetic dataset in the formal recovery schedule."""

    task_index: int
    run_id: str
    analysis_set: str
    stress_family: str
    stress_level: str
    stress_value: float | str
    world_id: StaticWorldId
    replicate_index: int
    dataset_seed: int
    split_seed: int
    model_seed: int
    world_kwargs: dict[str, Any]

    def to_seed_record(self) -> dict[str, Any]:
        return {
            "task_index": self.task_index,
            "run_id": self.run_id,
            "analysis_set": self.analysis_set,
            "stress_family": self.stress_family,
            "stress_level": self.stress_level,
            "stress_value": self.stress_value,
            "world_id": self.world_id,
            "replicate_index": self.replicate_index,
            "dataset_seed": self.dataset_seed,
            "split_seed": self.split_seed,
            "model_seed": self.model_seed,
        }


@dataclass(frozen=True)
class FormalRecoveryResult:
    """All row sets emitted by one formal recovery task."""

    task_index: int
    run_summary: dict[str, Any]
    model_rows: list[dict[str, Any]]
    participant_difference_rows: list[dict[str, Any]]
    component_rows: list[dict[str, Any]]
    split_audit: dict[str, Any]


@dataclass(frozen=True)
class FormalRecoveryOutputs:
    """Aggregated formal study tables."""

    run_summary: pd.DataFrame
    model_audit: pd.DataFrame
    participant_score_differences: pd.DataFrame
    component_recovery: pd.DataFrame
    split_audit: pd.DataFrame
    recovery_matrix: pd.DataFrame
    correct_model_recovery: pd.DataFrame
    false_discrete_rate: pd.DataFrame
    model_confusions: pd.DataFrame
    stress_recovery: pd.DataFrame


def load_formal_config(path: str | Path = FORMAL_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the frozen formal synthetic recovery protocol."""

    config = load_yaml_config(path)
    _validate_formal_config(config)
    return config


def build_formal_seed_schedule(config: dict[str, Any]) -> list[FormalRecoveryTask]:
    """Build the deterministic task and seed schedule."""

    master_seed = int(config["seeds"]["master_seed"])
    worlds = tuple(config["worlds"])
    baseline = config["baseline"]
    stress = config["stress"]
    tasks: list[FormalRecoveryTask] = []

    baseline_kwargs = _baseline_world_kwargs(baseline)
    for world_id in worlds:
        for replicate_index in range(int(baseline["replicates_per_world"])):
            tasks.append(
                _make_task(
                    task_index=len(tasks),
                    master_seed=master_seed,
                    analysis_set="baseline",
                    stress_family="none",
                    stress_level="baseline",
                    stress_value="baseline",
                    world_id=world_id,
                    replicate_index=replicate_index,
                    world_kwargs=baseline_kwargs,
                )
            )

    for scenario in _formal_stress_scenarios(config):
        for world_id in worlds:
            for replicate_index in range(int(stress["replicates_per_world_per_level"])):
                tasks.append(
                    _make_task(
                        task_index=len(tasks),
                        master_seed=master_seed,
                        analysis_set="stress",
                        stress_family=scenario["stress_family"],
                        stress_level=scenario["stress_level"],
                        stress_value=scenario["stress_value"],
                        world_id=world_id,
                        replicate_index=replicate_index,
                        world_kwargs=scenario["world_kwargs"],
                    )
                )
    return tasks


def preflight_formal_recovery(
    config: dict[str, Any],
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    workers: int | None = None,
    benchmark_seconds_per_dataset: float | None = None,
) -> dict[str, Any]:
    """Return a dry-run summary without fitting models."""

    tasks = build_formal_seed_schedule(config)
    worker_count = _resolve_workers(workers)
    n_model_fits = len(tasks) * len(STATIC_MODEL_IDS)
    estimated_audit_rows = n_model_fits
    estimated_participant_diff_rows = _estimate_participant_difference_rows(tasks)
    estimated_disk_bytes = (
        estimated_audit_rows * 2200
        + estimated_participant_diff_rows * 360
        + len(tasks) * 800
    )
    estimated_runtime = "not_available_without_machine_specific_benchmark"
    if benchmark_seconds_per_dataset is not None:
        seconds = float(benchmark_seconds_per_dataset) * len(tasks)
        estimated_runtime = {
            "seconds": round(seconds, 3),
            "minutes": round(seconds / 60.0, 3),
            "basis": "benchmark_seconds_per_synthetic_dataset",
        }
    return {
        "study_id": FORMAL_STUDY_ID,
        "total_synthetic_datasets": len(tasks),
        "total_model_fits": n_model_fits,
        "estimated_replicate_audit_rows": estimated_audit_rows,
        "estimated_participant_score_difference_rows": estimated_participant_diff_rows,
        "estimated_disk_use_bytes": estimated_disk_bytes,
        "estimated_disk_use_mb": round(estimated_disk_bytes / (1024 * 1024), 2),
        "available_workers": worker_count,
        "blas_threads_per_worker": {
            name: os.environ.get(name, "1")
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "checkpoint_path": str(checkpoint_dir),
        "resume_supported": True,
        "progress_interval_seconds": "configurable",
        "estimated_runtime": estimated_runtime,
        "output_paths": _output_paths(output_dir, config),
        "seed_schedule_hash": hash_mapping({"seed_schedule": [task.to_seed_record() for task in tasks]}),
    }


def benchmark_formal_recovery(
    config: dict[str, Any],
    *,
    samples: int = 1,
    workers: int | None = None,
    executor: str = "process",
) -> dict[str, Any]:
    """Benchmark representative scheduled units without writing scientific outputs."""

    tasks = build_formal_seed_schedule(config)[: max(1, int(samples))]
    worker_count = min(_resolve_workers(workers), len(tasks))
    start = time.perf_counter()
    _run_tasks(
        tasks,
        config=config,
        workers=worker_count,
        executor=executor,
        checkpoint_dir=None,
        resume=False,
        progress_interval_seconds=0,
    )
    runtime_seconds = float(time.perf_counter() - start)
    seconds_per_dataset = runtime_seconds / len(tasks)
    preflight = preflight_formal_recovery(
        config,
        workers=worker_count,
        benchmark_seconds_per_dataset=seconds_per_dataset,
    )
    return {
        "study_id": FORMAL_STUDY_ID,
        "benchmark_synthetic_datasets": len(tasks),
        "benchmark_model_fits": len(tasks) * len(STATIC_MODEL_IDS),
        "workers": worker_count,
        "executor": executor,
        "runtime_seconds": round(runtime_seconds, 3),
        "seconds_per_synthetic_dataset": round(seconds_per_dataset, 3),
        "projected_total_runtime": preflight["estimated_runtime"],
        "note": "Benchmark outputs are not formal scientific results.",
    }


def run_formal_synthetic_recovery(
    *,
    config_path: str | Path = FORMAL_CONFIG_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    workers: int | None = None,
    executor: str = "process",
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    resume: bool = True,
    progress_interval_seconds: float = 20.0,
    require_clean: bool = True,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Run the formal synthetic recovery study and write outputs."""

    start = time.perf_counter()
    repo = Path(repo_root).resolve()
    config_file = Path(config_path)
    config = load_formal_config(config_file)
    if require_clean:
        ensure_clean_git_tree(repo)
    git_commit = get_git_commit(repo)
    git_clean_at_start = not get_git_dirty(repo)
    tasks = build_formal_seed_schedule(config)
    worker_count = _resolve_workers(workers)

    _limit_numeric_threads()
    results = _run_tasks(
        tasks,
        config=config,
        workers=worker_count,
        executor=executor,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        progress_interval_seconds=progress_interval_seconds,
    )
    outputs = aggregate_formal_results(results)
    paths = write_formal_outputs(
        outputs,
        config=config,
        output_dir=output_dir,
        report_path=report_path,
    )
    runtime_seconds = float(time.perf_counter() - start)
    seed_schedule = [task.to_seed_record() for task in tasks]
    seed_schedule_hash = hash_mapping({"seed_schedule": seed_schedule})
    output_hashes = {name: hash_file(path) for name, path in paths.items()}
    manifest = build_formal_manifest(
        config=config,
        config_path=config_file,
        git_commit=git_commit,
        git_clean_at_start=git_clean_at_start,
        seed_schedule=seed_schedule,
        seed_schedule_hash=seed_schedule_hash,
        output_paths=paths,
        output_hashes=output_hashes,
        runtime_seconds=runtime_seconds,
        workers=worker_count,
        executor=executor,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
    )
    write_run_manifest(manifest_path, manifest)
    return {
        "outputs": outputs,
        "paths": paths,
        "manifest_path": Path(manifest_path),
        "manifest": manifest,
        "runtime_seconds": runtime_seconds,
    }


def run_formal_checkpoint_shard(
    *,
    config_path: str | Path = FORMAL_CONFIG_PATH,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    workers: int | None = None,
    executor: str = "process",
    task_start: int = 0,
    task_count: int | None = None,
    progress_interval_seconds: float = 20.0,
    require_clean: bool = True,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Run a bounded task shard and write only per-task checkpoints."""

    repo = Path(repo_root).resolve()
    config = load_formal_config(config_path)
    if require_clean:
        ensure_clean_git_tree(repo)
    all_tasks = build_formal_seed_schedule(config)
    shard = _slice_tasks(all_tasks, task_start=task_start, task_count=task_count)
    worker_count = min(_resolve_workers(workers), max(1, len(shard)))
    _limit_numeric_threads()
    start = time.perf_counter()
    results = _run_tasks(
        shard,
        config=config,
        workers=worker_count,
        executor=executor,
        checkpoint_dir=checkpoint_dir,
        resume=True,
        progress_interval_seconds=progress_interval_seconds,
    )
    runtime_seconds = float(time.perf_counter() - start)
    return {
        "study_id": FORMAL_STUDY_ID,
        "mode": "checkpoint_shard",
        "checkpoint_dir": str(checkpoint_dir),
        "task_start": int(task_start),
        "task_count": len(shard),
        "completed_in_shard": len(results),
        "total_scheduled_tasks": len(all_tasks),
        "workers": worker_count,
        "executor": executor,
        "runtime_seconds": round(runtime_seconds, 3),
    }


def aggregate_formal_checkpoints(
    *,
    config_path: str | Path = FORMAL_CONFIG_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
    require_clean: bool = True,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Aggregate a complete checkpoint set into formal reports and manifest."""

    start = time.perf_counter()
    repo = Path(repo_root).resolve()
    config_file = Path(config_path)
    config = load_formal_config(config_file)
    if require_clean:
        ensure_clean_git_tree(repo)
    git_commit = get_git_commit(repo)
    git_clean_at_start = not get_git_dirty(repo)
    tasks = build_formal_seed_schedule(config)
    completed = _load_completed_checkpoints(tasks, Path(checkpoint_dir))
    missing = [task.run_id for task in tasks if task.run_id not in completed]
    if missing:
        raise FormalRecoveryError(
            f"cannot aggregate formal checkpoints; {len(missing)} scheduled task(s) are missing"
        )
    outputs = aggregate_formal_results([completed[task.run_id] for task in tasks])
    paths = write_formal_outputs(
        outputs,
        config=config,
        output_dir=output_dir,
        report_path=report_path,
    )
    runtime_seconds = float(time.perf_counter() - start)
    seed_schedule = [task.to_seed_record() for task in tasks]
    seed_schedule_hash = hash_mapping({"seed_schedule": seed_schedule})
    output_hashes = {name: hash_file(path) for name, path in paths.items()}
    manifest = build_formal_manifest(
        config=config,
        config_path=config_file,
        git_commit=git_commit,
        git_clean_at_start=git_clean_at_start,
        seed_schedule=seed_schedule,
        seed_schedule_hash=seed_schedule_hash,
        output_paths=paths,
        output_hashes=output_hashes,
        runtime_seconds=runtime_seconds,
        workers=0,
        executor="aggregate_checkpoints",
        checkpoint_dir=checkpoint_dir,
        resume=True,
    )
    write_run_manifest(manifest_path, manifest)
    return {
        "outputs": outputs,
        "paths": paths,
        "manifest_path": Path(manifest_path),
        "manifest": manifest,
        "runtime_seconds": runtime_seconds,
    }


def run_formal_task(task: FormalRecoveryTask, config: dict[str, Any]) -> FormalRecoveryResult:
    """Run one scheduled synthetic dataset through M0-M4 without truth leakage."""

    truth_frame = make_static_synthetic_world(
        task.world_id,
        seed=task.dataset_seed,
        **task.world_kwargs,
    )
    model_frame = strip_ground_truth_columns(truth_frame)
    assert_no_ground_truth_columns(model_frame)
    split = participant_train_test_split(
        model_frame,
        test_size=float(config["split"]["test_fraction"]),
        seed=task.split_seed,
    )
    frozen = fit_score_models_without_truth(
        model_frame,
        split,
        feature_columns=CORE_SYNTHETIC_FEATURES,
        random_state=task.model_seed,
    )
    selection = select_preferred_model(
        frozen.model_scores,
        frozen.participant_scores,
        practical_equivalence_margin=float(config["selection"]["practical_equivalence_margin"]),
        paired_ci_z=float(config["selection"]["paired_ci_z"]),
    )

    truth = truth_frame.loc[
        list(split.test_indices),
        ["synthetic_world_id", "synthetic_aligned_model_id", "synthetic_component_id"],
    ].copy()
    component_metrics = _component_recovery_after_freeze(frozen.probabilities_by_model, truth)
    aligned_model_id = WORLD_MODEL_ALIGNMENT[task.world_id]
    run_common = _run_common(task, selection, aligned_model_id, truth_frame)
    uncertainty = selection.uncertainty_by_model.set_index("model_id", drop=False)
    model_rows: list[dict[str, Any]] = []
    participant_difference_rows = _participant_difference_rows(
        task=task,
        participant_scores=frozen.participant_scores,
        numerical_best_model_id=selection.numerical_best_model_id,
    )
    component_rows: list[dict[str, Any]] = []

    for _, row in frozen.model_scores.iterrows():
        model_id = str(row["model_id"])
        model_row = {
            **run_common,
            "model_id": model_id,
            "model_complexity_rank": _model_complexity_rank(model_id),
            "is_aligned_model": model_id == aligned_model_id,
            "is_selected_model": model_id == selection.selected_model_id,
            "ground_truth_revealed_after_selection": True,
            **{
                key: row[key]
                for key in row.index
                if key not in {"model_id", "primary_metric"}
            },
        }
        if model_id in uncertainty.index:
            model_row.update(
                {
                    key: uncertainty.loc[model_id, key]
                    for key in uncertainty.columns
                    if key != "model_id"
                }
            )
        if model_id in component_metrics:
            model_row.update(component_metrics[model_id])
            component_row = {
                **run_common,
                "model_id": model_id,
                "n_model_components": _n_model_components(model_id),
                **component_metrics[model_id],
            }
            component_row["component_collapsed"] = (
                int(component_row["n_predicted_components"]) < int(component_row["n_model_components"])
            )
            component_rows.append(component_row)
        model_rows.append(model_row)

    split_audit = {
        "run_id": task.run_id,
        "task_index": task.task_index,
        "analysis_set": task.analysis_set,
        "stress_family": task.stress_family,
        "stress_level": task.stress_level,
        "true_world_id": task.world_id,
        **_split_audit(model_frame, split),
    }
    return FormalRecoveryResult(
        task_index=task.task_index,
        run_summary=run_common,
        model_rows=model_rows,
        participant_difference_rows=participant_difference_rows,
        component_rows=component_rows,
        split_audit=split_audit,
    )


def aggregate_formal_results(results: Sequence[FormalRecoveryResult]) -> FormalRecoveryOutputs:
    """Aggregate formal task results into all requested output tables."""

    ordered = sorted(results, key=lambda result: result.task_index)
    run_summary = pd.DataFrame([result.run_summary for result in ordered])
    model_audit = pd.DataFrame([row for result in ordered for row in result.model_rows])
    participant_score_differences = pd.DataFrame(
        [row for result in ordered for row in result.participant_difference_rows]
    )
    component_recovery = pd.DataFrame([row for result in ordered for row in result.component_rows])
    split_audit = pd.DataFrame([result.split_audit for result in ordered])
    return FormalRecoveryOutputs(
        run_summary=run_summary,
        model_audit=model_audit,
        participant_score_differences=participant_score_differences,
        component_recovery=component_recovery,
        split_audit=split_audit,
        recovery_matrix=formal_recovery_matrix(run_summary),
        correct_model_recovery=correct_model_recovery(run_summary),
        false_discrete_rate=formal_false_discrete_rate(run_summary),
        model_confusions=model_confusions(run_summary),
        stress_recovery=stress_recovery(run_summary),
    )


def formal_recovery_matrix(run_summary: pd.DataFrame) -> pd.DataFrame:
    """Return baseline 5 by 5 true-world by selected-model proportions."""

    baseline = run_summary[run_summary["analysis_set"] == "baseline"].copy()
    rows: list[dict[str, Any]] = []
    for world_id in STATIC_SYNTHETIC_WORLD_IDS:
        world_rows = baseline[baseline["true_world_id"] == world_id]
        n_runs = int(world_rows.shape[0])
        for model_id in STATIC_MODEL_IDS:
            n_selected = int((world_rows["selected_model_id"] == model_id).sum())
            rows.append(
                {
                    "true_world_id": world_id,
                    "aligned_model_id": WORLD_MODEL_ALIGNMENT[world_id],
                    "selected_model_id": model_id,
                    "n_runs": n_runs,
                    "n_selected": n_selected,
                    "selected_model_proportion": n_selected / n_runs if n_runs else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def correct_model_recovery(run_summary: pd.DataFrame) -> pd.DataFrame:
    """Return P(aligned model selected | true world) with Wilson intervals."""

    baseline = run_summary[run_summary["analysis_set"] == "baseline"].copy()
    rows: list[dict[str, Any]] = []
    for world_id in STATIC_SYNTHETIC_WORLD_IDS:
        world_rows = baseline[baseline["true_world_id"] == world_id]
        n_runs = int(world_rows.shape[0])
        n_correct = int(world_rows["correct_model_selected"].sum())
        estimate = n_correct / n_runs if n_runs else float("nan")
        ci_low, ci_high = binomial_wilson_interval(n_correct, n_runs)
        rows.append(
            {
                "true_world_id": world_id,
                "aligned_model_id": WORLD_MODEL_ALIGNMENT[world_id],
                "n_runs": n_runs,
                "n_aligned_model_selected": n_correct,
                "aligned_model_recovery_rate": estimate,
                "ci_method": "wilson_95",
                "ci_low": ci_low,
                "ci_high": ci_high,
                "decision_band": recovery_decision_band(estimate),
            }
        )
    return pd.DataFrame(rows)


def formal_false_discrete_rate(run_summary: pd.DataFrame) -> pd.DataFrame:
    """Return P(M3 or M4 selected | W0/W1/W2) for baseline and stress."""

    rows: list[dict[str, Any]] = []
    for group_keys, group in run_summary.groupby(
        ["analysis_set", "stress_family", "stress_level"],
        sort=True,
        dropna=False,
    ):
        analysis_set, stress_family, stress_level = group_keys
        null_rows = group[group["true_world_id"].isin(NULL_OR_CONTINUOUS_WORLD_IDS)]
        rows.append(
            _formal_false_discrete_row(
                null_rows,
                analysis_set=str(analysis_set),
                stress_family=str(stress_family),
                stress_level=str(stress_level),
                true_world_id="W0_W1_W2_combined",
            )
        )
        for world_id in NULL_OR_CONTINUOUS_WORLD_IDS:
            rows.append(
                _formal_false_discrete_row(
                    null_rows[null_rows["true_world_id"] == world_id],
                    analysis_set=str(analysis_set),
                    stress_family=str(stress_family),
                    stress_level=str(stress_level),
                    true_world_id=world_id,
                )
            )
    return pd.DataFrame(rows)


def model_confusions(run_summary: pd.DataFrame) -> pd.DataFrame:
    """Return selected-model counts/proportions for baseline and stress groups."""

    rows: list[dict[str, Any]] = []
    for group_keys, group in run_summary.groupby(
        ["analysis_set", "stress_family", "stress_level", "true_world_id"],
        sort=True,
        dropna=False,
    ):
        analysis_set, stress_family, stress_level, world_id = group_keys
        n_runs = int(group.shape[0])
        for model_id in STATIC_MODEL_IDS:
            n_selected = int((group["selected_model_id"] == model_id).sum())
            rows.append(
                {
                    "analysis_set": analysis_set,
                    "stress_family": stress_family,
                    "stress_level": stress_level,
                    "true_world_id": world_id,
                    "aligned_model_id": WORLD_MODEL_ALIGNMENT[str(world_id)],
                    "selected_model_id": model_id,
                    "n_runs": n_runs,
                    "n_selected": n_selected,
                    "selected_model_proportion": n_selected / n_runs if n_runs else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def stress_recovery(run_summary: pd.DataFrame) -> pd.DataFrame:
    """Summarise one-factor-at-a-time stress recovery."""

    stress = run_summary[run_summary["analysis_set"] == "stress"].copy()
    rows: list[dict[str, Any]] = []
    for group_keys, group in stress.groupby(
        ["stress_family", "stress_level", "stress_value", "true_world_id"],
        sort=True,
        dropna=False,
    ):
        stress_family, stress_level, stress_value, world_id = group_keys
        n_runs = int(group.shape[0])
        n_correct = int(group["correct_model_selected"].sum())
        correct_rate = n_correct / n_runs if n_runs else float("nan")
        ci_low, ci_high = binomial_wilson_interval(n_correct, n_runs)
        if world_id in NULL_OR_CONTINUOUS_WORLD_IDS:
            n_false_discrete = int(group["selected_model_id"].isin(DISCRETE_MODEL_IDS).sum())
            false_rate = n_false_discrete / n_runs if n_runs else float("nan")
        else:
            n_false_discrete = 0
            false_rate = float("nan")
        rows.append(
            {
                "stress_family": stress_family,
                "stress_level": stress_level,
                "stress_value": stress_value,
                "true_world_id": world_id,
                "aligned_model_id": WORLD_MODEL_ALIGNMENT[str(world_id)],
                "n_runs": n_runs,
                "n_aligned_model_selected": n_correct,
                "aligned_model_recovery_rate": correct_rate,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "decision_band": recovery_decision_band(correct_rate),
                "n_false_discrete_selected": n_false_discrete,
                "false_discrete_rate_for_null_world": false_rate,
                "normalisation_sensitive_rate": float(group["normalisation_sensitive"].mean()),
            }
        )
    return pd.DataFrame(rows)


def binomial_wilson_interval(
    successes: int,
    n: int,
    *,
    z: float = 1.96,
) -> tuple[float, float]:
    """Return a Wilson score confidence interval for a binomial proportion."""

    if n <= 0:
        return float("nan"), float("nan")
    p_hat = successes / n
    denominator = 1.0 + z**2 / n
    centre = (p_hat + z**2 / (2.0 * n)) / denominator
    margin = (
        z
        * np.sqrt((p_hat * (1.0 - p_hat) + z**2 / (4.0 * n)) / n)
        / denominator
    )
    return float(max(0.0, centre - margin)), float(min(1.0, centre + margin))


def recovery_decision_band(rate: float) -> str:
    """Return the pre-frozen V1 aligned-model recovery band."""

    if not np.isfinite(rate):
        return "not_available"
    if rate >= 0.70:
        return "strong_recovery"
    if rate >= 0.50:
        return "moderate_ambiguous_recovery"
    return "poor_recovery"


def false_discrete_decision_band(rate: float) -> str:
    """Return the pre-frozen V1 false discrete-structure band."""

    if not np.isfinite(rate):
        return "not_available"
    if rate <= 0.10:
        return "reassuring"
    if rate <= 0.20:
        return "caution"
    return "concerning"


def write_formal_outputs(
    outputs: FormalRecoveryOutputs,
    *,
    config: dict[str, Any],
    output_dir: str | Path,
    report_path: str | Path,
) -> dict[str, Path]:
    """Write compact CSVs, compressed audit artifact and human report."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        name: output_root / filename
        for name, filename in SUMMARY_OUTPUT_NAMES.items()
    }
    paths["replicate_audit"] = Path(config["outputs"]["compressed_replicate_audit"])
    paths["replicate_audit"].parent.mkdir(parents=True, exist_ok=True)

    outputs.recovery_matrix.to_csv(paths["synthetic_recovery_matrix"], index=False)
    outputs.correct_model_recovery.to_csv(paths["synthetic_correct_model_recovery"], index=False)
    outputs.false_discrete_rate.to_csv(paths["synthetic_false_discrete_rate"], index=False)
    outputs.model_confusions.to_csv(paths["synthetic_model_confusions"], index=False)
    outputs.stress_recovery.to_csv(paths["synthetic_stress_recovery"], index=False)
    outputs.participant_score_differences.to_csv(
        paths["synthetic_participant_score_differences"],
        index=False,
    )
    outputs.component_recovery.to_csv(paths["synthetic_component_recovery"], index=False)
    with gzip.open(paths["replicate_audit"], "wt", encoding="utf-8", newline="") as handle:
        outputs.model_audit.to_csv(handle, index=False)

    report = Path(report_path)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_formal_report(outputs, config=config), encoding="utf-8")
    paths["report"] = report
    return paths


def build_formal_manifest(
    *,
    config: dict[str, Any],
    config_path: str | Path,
    git_commit: str,
    git_clean_at_start: bool,
    seed_schedule: list[dict[str, Any]],
    seed_schedule_hash: str,
    output_paths: dict[str, Path],
    output_hashes: dict[str, str],
    runtime_seconds: float,
    workers: int,
    executor: str,
    checkpoint_dir: str | Path,
    resume: bool,
) -> dict[str, Any]:
    """Build the formal run manifest with seed schedule and output hashes."""

    return {
        "study_id": FORMAL_STUDY_ID,
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "git_commit": git_commit,
        "git_status_at_start": "clean" if git_clean_at_start else "dirty",
        "formal_config_path": str(config_path),
        "formal_config_hash": hash_file(config_path),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": Path(sys.executable).name,
        },
        "packages": package_versions(("numpy", "pandas", "PyYAML")),
        "master_seed": int(config["seeds"]["master_seed"]),
        "seed_schedule_hash": seed_schedule_hash,
        "seed_schedule": seed_schedule,
        "model_ids": list(STATIC_MODEL_IDS),
        "synthetic_world_versions": {
            world_id: f"{world_id}-synthetic-v1"
            for world_id in STATIC_SYNTHETIC_WORLD_IDS
        },
        "split_policy": {
            "policy": config["split"]["policy"],
            "test_fraction": float(config["split"]["test_fraction"]),
            "participant_isolated": True,
        },
        "replicate_counts": {
            "baseline_replicates_per_world": int(config["baseline"]["replicates_per_world"]),
            "stress_replicates_per_world_per_level": int(
                config["stress"]["replicates_per_world_per_level"]
            ),
        },
        "output_paths": {name: str(path) for name, path in sorted(output_paths.items())},
        "output_hashes": output_hashes,
        "replicate_audit": {
            "path": str(output_paths["replicate_audit"]),
            "sha256": output_hashes["replicate_audit"],
            "row_count": int(
                pd.read_csv(
                    output_paths["replicate_audit"],
                    compression="gzip",
                    low_memory=False,
                ).shape[0]
            ),
            "generation_command": (
                "python -m trident_validation.synthetic.formal_recovery "
                "--config config/formal_synthetic_recovery_v1.yaml"
            ),
        },
        "runtime_seconds": runtime_seconds,
        "workers": workers,
        "executor": executor,
        "checkpoint_dir": str(checkpoint_dir),
        "resume_enabled": bool(resume),
        "blas_threads_per_worker": {
            name: os.environ.get(name, "1")
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }


def render_formal_report(outputs: FormalRecoveryOutputs, *, config: dict[str, Any]) -> str:
    """Render the formal synthetic recovery report."""

    correct = outputs.correct_model_recovery
    false_rows = outputs.false_discrete_rate
    pooled = false_rows[
        (false_rows["analysis_set"] == "baseline")
        & (false_rows["true_world_id"] == "W0_W1_W2_combined")
    ]
    pooled_false_rate = (
        float(pooled.iloc[0]["false_discrete_rate"])
        if not pooled.empty
        else float("nan")
    )
    baseline_runs = outputs.run_summary[outputs.run_summary["analysis_set"] == "baseline"]
    software_rows = [
        ["Git tree clean at launch", "enforced by runner"],
        ["Participant-isolated splits", str(bool(outputs.split_audit["participant_isolated"].all()))],
        ["Ground truth reveal", "after frozen scoring and selection"],
        ["Baseline synthetic datasets", str(int(baseline_runs.shape[0]))],
        ["Primary metric", PRIMARY_METRIC],
        ["Selection rule", config["selection"]["tie_resolution"]],
    ]
    recovery_rows = [
        [
            row["true_world_id"],
            row["aligned_model_id"],
            str(int(row["n_runs"])),
            f"{float(row['aligned_model_recovery_rate']):.3f}",
            f"[{float(row['ci_low']):.3f}, {float(row['ci_high']):.3f}]",
            row["decision_band"],
        ]
        for _, row in correct.iterrows()
    ]
    false_report_rows = [
        [
            row["true_world_id"],
            str(int(row["n_runs"])),
            f"{float(row['false_discrete_rate']):.3f}",
            f"[{float(row['ci_low']):.3f}, {float(row['ci_high']):.3f}]",
            row["decision_band"],
        ]
        for _, row in false_rows[false_rows["analysis_set"] == "baseline"].iterrows()
    ]
    component_rows = _component_report_rows(outputs.component_recovery)
    stress_rows = _stress_report_rows(outputs.stress_recovery)
    normalisation_sensitive = float(baseline_runs["normalisation_sensitive"].mean())

    lines = [
        "# Formal Synthetic Model-Recovery Study V1",
        "",
        "## Scope",
        "",
        "This formal experiment uses only the existing synthetic W0-W4 generators and static M0-M4 tournament. It does not use real public datasets and does not implement M5, cPCA, CVQ, CVAE, HMMs, graph models or Attention Coach transport.",
        "",
        "The protocol was frozen in `config/formal_synthetic_recovery_v1.yaml` before running. The generating worlds, statistical model forms, selection rule, stress grid and decision bands are not changed in response to the results.",
        "",
        "## Software Validity",
        "",
        _markdown_table(["Check", "Result"], software_rows),
        "",
        "Software-valid execution means the experiment followed the frozen protocol. It does not imply that the candidate models recovered the generating architectures.",
        "",
        "## Model Recovery",
        "",
        _markdown_table(
            ["True world", "Aligned model", "N", "Recovery rate", "95% CI", "V1 band"],
            recovery_rows,
        ),
        "",
        "Decision bands are pre-frozen V1 engineering/interpretive bands: >=0.70 strong recovery, 0.50-0.69 moderate / ambiguous recovery, and <0.50 poor recovery.",
        "",
        "## Falsification",
        "",
        _markdown_table(
            ["Null/continuous world", "N", "False discrete rate", "95% CI", "V1 band"],
            false_report_rows,
        ),
        "",
        f"Pooled false discrete-structure rate band: {false_discrete_decision_band(pooled_false_rate)}.",
        "",
        "This estimates how often M3 or M4 is selected under W0/W1/W2, where the generating worlds contain no discrete profile structure.",
        "",
        "## W3/W4 Component Recovery",
        "",
        _markdown_table(
            [
                "True world",
                "Model",
                "N",
                "Mean ARI",
                "Mean entropy",
                "Mean size L1",
                "Collapse frequency",
            ],
            component_rows or [["not_available", "not_available", "0", "nan", "nan", "nan", "nan"]],
        ),
        "",
        "Component labels remain neutral. Matching is permutation-invariant and is used only after model selections and predictions are frozen.",
        "",
        "## Normalisation Sensitivity",
        "",
        f"Baseline winner depended on score normalisation in {normalisation_sensitive:.3f} of baseline replicates.",
        "",
        "## Stress Study",
        "",
        _markdown_table(
            ["Stress family", "Level", "Mean recovery", "Mean false discrete rate for W0-W2"],
            stress_rows,
        ),
        "",
        "Stress factors are varied one at a time around the baseline condition.",
        "",
        "## Machine-Readable Outputs",
        "",
        "- `reports/generated/synthetic_recovery_matrix.csv`",
        "- `reports/generated/synthetic_correct_model_recovery.csv`",
        "- `reports/generated/synthetic_false_discrete_rate.csv`",
        "- `reports/generated/synthetic_model_confusions.csv`",
        "- `reports/generated/synthetic_stress_recovery.csv`",
        "- `reports/generated/synthetic_participant_score_differences.csv`",
        "- `reports/generated/synthetic_component_recovery.csv`",
        "- `reports/generated/formal_synthetic_recovery_v1_replicates.csv.gz`",
        "- `manifests/formal_synthetic_recovery_v1.json`",
    ]
    return "\n".join(lines) + "\n"


def ensure_clean_git_tree(repo_root: str | Path = ".") -> None:
    """Refuse formal execution unless the Git working tree is clean."""

    repo = Path(repo_root).resolve()
    dirty = get_git_dirty(repo)
    if dirty:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={repo.as_posix()}", "status", "--porcelain"],
            cwd=repo,
            text=True,
            capture_output=True,
            check=False,
        )
        raise FormalRecoveryError(
            "formal synthetic recovery requires a clean Git tree before launch:\n"
            + result.stdout.strip()
        )


def get_git_dirty(repo_root: str | Path = ".") -> bool:
    """Return Git dirty status with command-local safe.directory handling."""

    repo = Path(repo_root).resolve()
    result = subprocess.run(
        ["git", "-c", f"safe.directory={repo.as_posix()}", "status", "--porcelain"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise FormalRecoveryError(result.stderr.strip() or "could not read Git status")
    return bool(result.stdout.strip())


def _run_tasks(
    tasks: Sequence[FormalRecoveryTask],
    *,
    config: dict[str, Any],
    workers: int,
    executor: str = "thread",
    checkpoint_dir: str | Path | None = None,
    resume: bool = False,
    progress_interval_seconds: float = 20.0,
) -> list[FormalRecoveryResult]:
    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    completed = _load_completed_checkpoints(tasks, checkpoint_root) if resume else {}
    results: list[FormalRecoveryResult] = list(completed.values())
    pending = [task for task in tasks if task.run_id not in completed]
    progress = _ProgressReporter(
        total=len(tasks),
        completed=len(results),
        started_at=time.perf_counter(),
        interval_seconds=float(progress_interval_seconds),
    )
    progress.emit(force=True)

    if not pending:
        return sorted(results, key=lambda result: result.task_index)

    if workers <= 1 or executor == "serial":
        for task in pending:
            result = run_formal_task(task, config)
            _write_completed_checkpoint(result, checkpoint_root)
            results.append(result)
            progress.mark_complete(task, result)
        return sorted(results, key=lambda result: result.task_index)

    pool_class = ProcessPoolExecutor if executor == "process" else ThreadPoolExecutor
    if executor not in {"process", "thread"}:
        raise FormalRecoveryError("executor must be one of: process, thread, serial")

    failures: list[tuple[FormalRecoveryTask, BaseException]] = []
    with pool_class(max_workers=workers) as pool:
        futures = {pool.submit(run_formal_task, task, config): task for task in pending}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
            except BaseException as exc:  # pragma: no cover - defensive path
                failures.append((task, exc))
                _write_failed_checkpoint(task, checkpoint_root, exc)
                progress.mark_failed(task)
                continue
            _write_completed_checkpoint(result, checkpoint_root)
            results.append(result)
            progress.mark_complete(task, result)
    if failures:
        failed = ", ".join(task.run_id for task, _ in failures[:5])
        raise FormalRecoveryError(
            f"{len(failures)} formal recovery task(s) failed; first failed run_id(s): {failed}"
        )
    return sorted(results, key=lambda result: result.task_index)


class _ProgressReporter:
    def __init__(
        self,
        *,
        total: int,
        completed: int,
        started_at: float,
        interval_seconds: float,
    ) -> None:
        self.total = total
        self.completed = completed
        self.failed = 0
        self.started_at = started_at
        self.interval_seconds = interval_seconds
        self._last_emit = 0.0

    def mark_complete(self, task: FormalRecoveryTask, result: FormalRecoveryResult) -> None:
        self.completed += 1
        self.emit(task=task, result=result)

    def mark_failed(self, task: FormalRecoveryTask) -> None:
        self.failed += 1
        self.emit(task=task, force=True)

    def emit(
        self,
        *,
        task: FormalRecoveryTask | None = None,
        result: FormalRecoveryResult | None = None,
        force: bool = False,
    ) -> None:
        if self.interval_seconds <= 0:
            return
        now = time.perf_counter()
        if not force and now - self._last_emit < self.interval_seconds:
            return
        self._last_emit = now
        elapsed = now - self.started_at
        rate = self.completed / elapsed if elapsed > 0 else 0.0
        remaining = max(0, self.total - self.completed)
        eta = remaining / rate if rate > 0 else float("nan")
        context = "initialising"
        if task is not None:
            context = f"{task.world_id} | {task.analysis_set} | {task.stress_level}"
        selected = ""
        if result is not None:
            selected = f" | selected {result.run_summary['selected_model_id']}"
        print(
            "M2.6 | "
            f"{context} | "
            f"{self.completed}/{self.total} datasets complete | "
            f"{self.completed * len(STATIC_MODEL_IDS)}/{self.total * len(STATIC_MODEL_IDS)} model fits complete | "
            f"elapsed {_format_duration(elapsed)} | "
            f"ETA {_format_duration(eta)} | "
            f"failures {self.failed}"
            f"{selected}",
            file=sys.stderr,
            flush=True,
        )


def _load_completed_checkpoints(
    tasks: Sequence[FormalRecoveryTask],
    checkpoint_root: Path | None,
) -> dict[str, FormalRecoveryResult]:
    if checkpoint_root is None:
        return {}
    completed: dict[str, FormalRecoveryResult] = {}
    for task in tasks:
        path = _checkpoint_path(checkpoint_root, task)
        if not path.exists():
            continue
        result = _read_completed_checkpoint(path, expected_run_id=task.run_id)
        completed[result.run_summary["run_id"]] = result
    return completed


def _write_completed_checkpoint(
    result: FormalRecoveryResult,
    checkpoint_root: Path | None,
) -> None:
    if checkpoint_root is None:
        return
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path_from_result(checkpoint_root, result)
    payload = {
        "status": "complete",
        "result": _result_to_jsonable(result),
    }
    payload["checksum"] = _checkpoint_checksum(payload)
    _atomic_json_write(path, payload)


def _write_failed_checkpoint(
    task: FormalRecoveryTask,
    checkpoint_root: Path | None,
    exc: BaseException,
) -> None:
    if checkpoint_root is None:
        return
    failed_root = checkpoint_root / "failed"
    failed_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": "failed",
        "task": task.to_seed_record(),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    payload["checksum"] = _checkpoint_checksum(payload)
    _atomic_json_write(failed_root / f"{task.task_index:05d}_{task.run_id}.json", payload)


def _read_completed_checkpoint(path: Path, *, expected_run_id: str) -> FormalRecoveryResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    checksum = str(payload.pop("checksum", ""))
    if checksum != _checkpoint_checksum(payload):
        raise FormalRecoveryError(f"checkpoint checksum mismatch: {path}")
    if payload.get("status") != "complete":
        raise FormalRecoveryError(f"checkpoint is not complete: {path}")
    result = _result_from_jsonable(payload["result"])
    if result.run_summary["run_id"] != expected_run_id:
        raise FormalRecoveryError(f"checkpoint run_id mismatch: {path}")
    return result


def _checkpoint_path(checkpoint_root: Path, task: FormalRecoveryTask) -> Path:
    return checkpoint_root / "complete" / f"{task.task_index:05d}_{task.run_id}.json"


def _checkpoint_path_from_result(checkpoint_root: Path, result: FormalRecoveryResult) -> Path:
    run_id = str(result.run_summary["run_id"])
    return checkpoint_root / "complete" / f"{result.task_index:05d}_{run_id}.json"


def _result_to_jsonable(result: FormalRecoveryResult) -> dict[str, Any]:
    return {
        "task_index": result.task_index,
        "run_summary": _jsonable(result.run_summary),
        "model_rows": _jsonable(result.model_rows),
        "participant_difference_rows": _jsonable(result.participant_difference_rows),
        "component_rows": _jsonable(result.component_rows),
        "split_audit": _jsonable(result.split_audit),
    }


def _result_from_jsonable(payload: dict[str, Any]) -> FormalRecoveryResult:
    return FormalRecoveryResult(
        task_index=int(payload["task_index"]),
        run_summary=dict(payload["run_summary"]),
        model_rows=[dict(row) for row in payload["model_rows"]],
        participant_difference_rows=[
            dict(row) for row in payload["participant_difference_rows"]
        ],
        component_rows=[dict(row) for row in payload["component_rows"]],
        split_audit=dict(payload["split_audit"]),
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _checkpoint_checksum(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    tmp.replace(path)


def _limit_numeric_threads() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")


def _format_duration(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _validate_formal_config(config: dict[str, Any]) -> None:
    if config.get("study", {}).get("id") != FORMAL_STUDY_ID:
        raise FormalRecoveryError("formal config study.id is invalid")
    worlds = tuple(config.get("worlds", ()))
    if worlds != STATIC_SYNTHETIC_WORLD_IDS:
        raise FormalRecoveryError("formal config worlds must be the frozen W0-W4 order")
    models = tuple(config.get("models", ()))
    if models != STATIC_MODEL_IDS:
        raise FormalRecoveryError("formal config models must be the frozen M0-M4 order")
    if config.get("split", {}).get("policy") != "participant_isolated":
        raise FormalRecoveryError("split.policy must be participant_isolated")
    if float(config["split"]["test_fraction"]) != 0.25:
        raise FormalRecoveryError("split.test_fraction must remain 0.25")
    if config["selection"]["primary_metric"] != PRIMARY_METRIC:
        raise FormalRecoveryError("selection.primary_metric must remain heldout log density per window")
    if not bool(config["stress"]["one_factor_at_a_time"]):
        raise FormalRecoveryError("stress.one_factor_at_a_time must be true")


def _baseline_world_kwargs(baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "n_datasets": int(baseline["n_datasets"]),
        "participants_per_dataset": int(baseline["participants_per_dataset"]),
        "sessions_per_participant": int(baseline["sessions_per_participant"]),
        "min_windows_per_session": int(baseline["min_windows_per_session"]),
        "max_windows_per_session": int(baseline["max_windows_per_session"]),
        "observation_noise_scale": float(baseline["observation_noise_scale"]),
        "source_shift_scale": float(baseline["source_shift_scale"]),
        "technical_missingness_rate": float(baseline["technical_missingness_rate"]),
        "latent_separation_scale": float(baseline["latent_separation_scale"]),
    }


def _formal_stress_scenarios(config: dict[str, Any]) -> list[dict[str, Any]]:
    baseline = _baseline_world_kwargs(config["baseline"])
    stress = config["stress"]
    scenarios: list[dict[str, Any]] = []
    for value in stress["participant_count_per_dataset"]:
        kwargs = {**baseline, "participants_per_dataset": int(value)}
        scenarios.append(_scenario("participant_count_per_dataset", str(value), int(value), kwargs))
    for value in stress["observation_noise_multiplier"]:
        kwargs = {**baseline, "observation_noise_scale": float(value)}
        scenarios.append(_scenario("observation_noise_multiplier", str(value), float(value), kwargs))
    for value in stress["missingness"]:
        kwargs = {**baseline, "technical_missingness_rate": float(value)}
        scenarios.append(_scenario("missingness", str(value), float(value), kwargs))
    for value in stress["source_dataset_shift_multiplier"]:
        kwargs = {**baseline, "source_shift_scale": float(value)}
        scenarios.append(_scenario("source_dataset_shift_multiplier", str(value), float(value), kwargs))
    for value in stress["latent_profile_separation_multiplier"]:
        kwargs = {**baseline, "latent_separation_scale": float(value)}
        scenarios.append(_scenario("latent_profile_separation_multiplier", str(value), float(value), kwargs))
    for level, window_config in stress["windows_per_session_condition"].items():
        kwargs = {
            **baseline,
            "min_windows_per_session": int(window_config["min_windows_per_session"]),
            "max_windows_per_session": int(window_config["max_windows_per_session"]),
        }
        scenarios.append(_scenario("windows_per_session_condition", str(level), str(level), kwargs))
    return scenarios


def _scenario(
    stress_family: str,
    stress_level: str,
    stress_value: float | str,
    world_kwargs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stress_family": stress_family,
        "stress_level": stress_level,
        "stress_value": stress_value,
        "world_kwargs": world_kwargs,
    }


def _make_task(
    *,
    task_index: int,
    master_seed: int,
    analysis_set: str,
    stress_family: str,
    stress_level: str,
    stress_value: float | str,
    world_id: str,
    replicate_index: int,
    world_kwargs: dict[str, Any],
) -> FormalRecoveryTask:
    run_id = _run_id(
        FORMAL_STUDY_ID,
        master_seed,
        analysis_set,
        stress_family,
        stress_level,
        world_id,
        replicate_index,
    )
    return FormalRecoveryTask(
        task_index=task_index,
        run_id=run_id,
        analysis_set=analysis_set,
        stress_family=stress_family,
        stress_level=stress_level,
        stress_value=stress_value,
        world_id=world_id,  # type: ignore[arg-type]
        replicate_index=replicate_index,
        dataset_seed=_child_seed(run_id, "dataset"),
        split_seed=_child_seed(run_id, "split"),
        model_seed=_child_seed(run_id, "model"),
        world_kwargs=dict(world_kwargs),
    )


def _run_common(
    task: FormalRecoveryTask,
    selection: Any,
    aligned_model_id: str,
    truth_frame: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "run_id": task.run_id,
        "task_index": task.task_index,
        "analysis_set": task.analysis_set,
        "stress_family": task.stress_family,
        "stress_level": task.stress_level,
        "stress_value": task.stress_value,
        "replicate_index": task.replicate_index,
        "dataset_seed": task.dataset_seed,
        "split_seed": task.split_seed,
        "model_seed": task.model_seed,
        "true_world_id": task.world_id,
        "aligned_model_id": aligned_model_id,
        "selected_model_id": selection.selected_model_id,
        "numerical_best_model_id": selection.numerical_best_model_id,
        "selection_reason": selection.selection_reason,
        "primary_metric": selection.primary_metric,
        "per_window_winner": selection.per_window_winner,
        "participant_weighted_winner": selection.participant_weighted_winner,
        "per_observed_feature_winner": selection.per_observed_feature_winner,
        "normalisation_sensitive": selection.normalisation_sensitive,
        "correct_model_selected": selection.selected_model_id == aligned_model_id,
        "selected_is_discrete": selection.selected_model_id in DISCRETE_MODEL_IDS,
        "n_sources": int(truth_frame["source_dataset"].nunique()),
        "participants_per_dataset": int(task.world_kwargs["participants_per_dataset"]),
        "sessions_per_participant": int(task.world_kwargs["sessions_per_participant"]),
        "min_windows_per_session": int(task.world_kwargs["min_windows_per_session"]),
        "max_windows_per_session": int(task.world_kwargs["max_windows_per_session"]),
        "observation_noise_scale": float(task.world_kwargs["observation_noise_scale"]),
        "source_shift_scale": float(task.world_kwargs["source_shift_scale"]),
        "technical_missingness_rate": float(task.world_kwargs["technical_missingness_rate"]),
        "latent_separation_scale": float(task.world_kwargs["latent_separation_scale"]),
    }


def _participant_difference_rows(
    *,
    task: FormalRecoveryTask,
    participant_scores: pd.DataFrame,
    numerical_best_model_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if numerical_best_model_id not in participant_scores:
        return rows
    best = participant_scores[numerical_best_model_id]
    for model_id in STATIC_MODEL_IDS:
        if model_id == numerical_best_model_id or model_id not in participant_scores:
            continue
        diff = (best - participant_scores[model_id]).dropna()
        n = int(diff.shape[0])
        mean = float(diff.mean()) if n else float("nan")
        sd = float(diff.std(ddof=1)) if n > 1 else float("nan")
        se = sd / np.sqrt(n) if n > 1 else float("nan")
        ci_low = mean - 1.96 * se if n > 1 else float("nan")
        ci_high = mean + 1.96 * se if n > 1 else float("nan")
        rows.append(
            {
                "run_id": task.run_id,
                "task_index": task.task_index,
                "analysis_set": task.analysis_set,
                "stress_family": task.stress_family,
                "stress_level": task.stress_level,
                "true_world_id": task.world_id,
                "numerical_best_model_id": numerical_best_model_id,
                "comparison_model_id": model_id,
                "participant_weighted_delta_best_minus_model": mean,
                "paired_n_participants": n,
                "paired_delta_sd": sd,
                "paired_delta_se": se,
                "paired_delta_ci_low": ci_low,
                "paired_delta_ci_high": ci_high,
            }
        )
    return rows


def _formal_false_discrete_row(
    frame: pd.DataFrame,
    *,
    analysis_set: str,
    stress_family: str,
    stress_level: str,
    true_world_id: str,
) -> dict[str, Any]:
    n_runs = int(frame.shape[0])
    n_discrete = int(frame["selected_model_id"].isin(DISCRETE_MODEL_IDS).sum())
    rate = n_discrete / n_runs if n_runs else float("nan")
    ci_low, ci_high = binomial_wilson_interval(n_discrete, n_runs)
    return {
        "analysis_set": analysis_set,
        "stress_family": stress_family,
        "stress_level": stress_level,
        "true_world_id": true_world_id,
        "n_runs": n_runs,
        "n_discrete_selected": n_discrete,
        "false_discrete_rate": rate,
        "ci_method": "wilson_95",
        "ci_low": ci_low,
        "ci_high": ci_high,
        "decision_band": false_discrete_decision_band(rate),
    }


def _component_report_rows(component_recovery: pd.DataFrame) -> list[list[str]]:
    if component_recovery.empty:
        return []
    rows: list[list[str]] = []
    baseline = component_recovery[component_recovery["analysis_set"] == "baseline"]
    for (world_id, model_id), group in baseline.groupby(["true_world_id", "model_id"], sort=True):
        rows.append(
            [
                str(world_id),
                str(model_id),
                str(int(group.shape[0])),
                f"{float(group['component_ari'].mean()):.3f}",
                f"{float(group['posterior_entropy_mean'].mean()):.3f}",
                f"{float(group['component_size_l1'].mean()):.3f}",
                f"{float(group['component_collapsed'].mean()):.3f}",
            ]
        )
    return rows


def _stress_report_rows(stress: pd.DataFrame) -> list[list[str]]:
    if stress.empty:
        return [["not_available", "not_available", "nan", "nan"]]
    rows: list[list[str]] = []
    for (family, level), group in stress.groupby(["stress_family", "stress_level"], sort=True):
        null_group = group[group["true_world_id"].isin(NULL_OR_CONTINUOUS_WORLD_IDS)]
        false_rate = (
            float(null_group["false_discrete_rate_for_null_world"].mean())
            if not null_group.empty
            else float("nan")
        )
        rows.append(
            [
                str(family),
                str(level),
                f"{float(group['aligned_model_recovery_rate'].mean()):.3f}",
                f"{false_rate:.3f}",
            ]
        )
    return rows


def _n_model_components(model_id: str) -> int:
    if model_id == "M3_three_profile_mixture":
        return 3
    if model_id == "M4_four_pace_profile_mixture":
        return 4
    return 0


def _estimate_participant_difference_rows(tasks: Sequence[FormalRecoveryTask]) -> int:
    total = 0
    for task in tasks:
        participant_count = int(task.world_kwargs["n_datasets"]) * int(
            task.world_kwargs["participants_per_dataset"]
        )
        total += int(np.ceil(participant_count * 0.25)) * (len(STATIC_MODEL_IDS) - 1)
    return total


def _output_paths(output_dir: str | Path, config: dict[str, Any]) -> dict[str, str]:
    output_root = Path(output_dir)
    paths = {
        name: str(output_root / filename)
        for name, filename in SUMMARY_OUTPUT_NAMES.items()
    }
    paths["replicate_audit"] = str(config["outputs"]["compressed_replicate_audit"])
    paths["manifest"] = str(config["outputs"]["manifest"])
    paths["report"] = str(config["outputs"]["report"])
    return paths


def _resolve_workers(workers: int | None) -> int:
    if workers is not None:
        return max(1, int(workers))
    available = os.cpu_count() or 1
    return max(1, available - 1)


def _slice_tasks(
    tasks: Sequence[FormalRecoveryTask],
    *,
    task_start: int = 0,
    task_count: int | None = None,
) -> list[FormalRecoveryTask]:
    start = max(0, int(task_start))
    if task_count is None:
        return list(tasks[start:])
    count = max(0, int(task_count))
    return list(tasks[start : start + count])


def _run_id(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _child_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(str(cell) for cell in row) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run formal synthetic recovery V1.")
    parser.add_argument("--config", default=str(FORMAL_CONFIG_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--report", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--executor", choices=("process", "thread", "serial"), default="process")
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--progress-interval-seconds", type=float, default=20.0)
    parser.add_argument("--benchmark-samples", type=int, default=0)
    parser.add_argument("--checkpoint-only", action="store_true")
    parser.add_argument("--aggregate-checkpoints", action="store_true")
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--task-count", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config = load_formal_config(args.config)
    _limit_numeric_threads()
    if args.benchmark_samples:
        payload = benchmark_formal_recovery(
            config,
            samples=args.benchmark_samples,
            workers=args.workers,
            executor=args.executor,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.checkpoint_only:
        payload = run_formal_checkpoint_shard(
            config_path=args.config,
            checkpoint_dir=args.checkpoint_dir,
            workers=args.workers,
            executor=args.executor,
            task_start=args.task_start,
            task_count=args.task_count,
            progress_interval_seconds=args.progress_interval_seconds,
            require_clean=True,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.aggregate_checkpoints:
        result = aggregate_formal_checkpoints(
            config_path=args.config,
            output_dir=args.output_dir,
            manifest_path=args.manifest,
            report_path=args.report,
            checkpoint_dir=args.checkpoint_dir,
            require_clean=True,
        )
        print(f"manifest: {result['manifest_path']}")
        for name, path in sorted(result["paths"].items()):
            print(f"{name}: {path}")
        print(f"runtime_seconds: {result['runtime_seconds']:.3f}")
        return 0
    if args.dry_run:
        payload = preflight_formal_recovery(
            config,
            output_dir=args.output_dir,
            checkpoint_dir=args.checkpoint_dir,
            workers=args.workers,
        )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    result = run_formal_synthetic_recovery(
        config_path=args.config,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        report_path=args.report,
        workers=args.workers,
        executor=args.executor,
        checkpoint_dir=args.checkpoint_dir,
        resume=not args.no_resume,
        progress_interval_seconds=args.progress_interval_seconds,
        require_clean=True,
    )
    print(f"manifest: {result['manifest_path']}")
    for name, path in sorted(result["paths"].items()):
        print(f"{name}: {path}")
    print(f"runtime_seconds: {result['runtime_seconds']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
