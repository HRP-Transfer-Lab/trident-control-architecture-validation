"""M2.6b exact-model recovery diagnostic for the static M0-M4 tournament."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from trident_validation.config import load_yaml_config
from trident_validation.models.m0_probabilistic import M0ProbabilisticPCAModel
from trident_validation.models.m1_continuous import M1ContinuousManifoldModel
from trident_validation.models.m2_nonlinear import (
    M2LatentCurveEMModel,
    M2NonlinearVigilanceModel,
    M2ProjectionSearchVigilanceModel,
)
from trident_validation.models.static_tournament import STATIC_MODEL_IDS
from trident_validation.splits import participant_train_test_split
from trident_validation.synthetic.fixtures import CORE_SYNTHETIC_FEATURES
from trident_validation.synthetic.formal_recovery import (
    _limit_numeric_threads,
    _slice_tasks,
    binomial_wilson_interval,
    recovery_decision_band,
)
from trident_validation.synthetic.recovery import (
    PRIMARY_METRIC,
    fit_score_models_without_truth,
    select_preferred_model,
    strip_ground_truth_columns,
    _observed_feature_counts,
    _participant_log_density_means,
)


ExactWorldId = Literal[
    "EW0_m0_exact",
    "EW1_m1_exact",
    "EW2_m2_exact",
    "EW3_m3_exact",
    "EW4_m4_exact",
]

EXACT_MODEL_STUDY_ID = "exact_model_recovery_v1"
EW2_EM_DIAGNOSTIC_STUDY_ID = "ew2_exact_with_m2_em_diagnostic_v1"
M2_EM_CONVERGENCE_STUDY_ID = "m2_em_convergence_diagnostic_v1"
EW0_M1_NESTING_STUDY_ID = "ew0_m1_nesting_diagnostic_v1"
EXACT_CONFIG_PATH = Path("config/exact_model_recovery_v1.yaml")
DEFAULT_OUTPUT_DIR = Path("reports/generated/exact_model_recovery_v1")
DEFAULT_EW2_EM_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "ew2_em_diagnostic"
DEFAULT_M2_EM_CONVERGENCE_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "m2_em_convergence"
DEFAULT_EW0_M1_NESTING_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "ew0_m1_nesting"
EM_M2_DIAGNOSTIC_MODEL_ID = "M2_latent_curve_em"
EXACT_MODEL_WORLD_IDS: tuple[ExactWorldId, ...] = (
    "EW0_m0_exact",
    "EW1_m1_exact",
    "EW2_m2_exact",
    "EW3_m3_exact",
    "EW4_m4_exact",
)
EXACT_WORLD_MODEL_ALIGNMENT: dict[str, str] = {
    "EW0_m0_exact": "M0_probabilistic_general_performance",
    "EW1_m1_exact": "M1_continuous_control_manifold",
    "EW2_m2_exact": "M2_nonlinear_vigilance",
    "EW3_m3_exact": "M3_three_profile_mixture",
    "EW4_m4_exact": "M4_four_pace_profile_mixture",
}


@dataclass(frozen=True)
class ExactRecoveryTask:
    task_index: int
    run_id: str
    exact_world_id: ExactWorldId
    replicate_index: int
    dataset_seed: int
    split_seed: int
    model_seed: int
    world_kwargs: dict[str, Any]


def load_exact_config(path: str | Path = EXACT_CONFIG_PATH) -> dict[str, Any]:
    config = load_yaml_config(path)
    _validate_exact_config(config)
    return config


def build_exact_seed_schedule(config: dict[str, Any]) -> list[ExactRecoveryTask]:
    master_seed = int(config["seeds"]["master_seed"])
    baseline = config["baseline"]
    world_kwargs = {
        "n_datasets": int(baseline["n_datasets"]),
        "participants_per_dataset": int(baseline["participants_per_dataset"]),
        "sessions_per_participant": int(baseline["sessions_per_participant"]),
        "min_windows_per_session": int(baseline["min_windows_per_session"]),
        "max_windows_per_session": int(baseline["max_windows_per_session"]),
        "technical_missingness_rate": float(baseline["technical_missingness_rate"]),
        "exact_signal_scale": float(baseline["exact_signal_scale"]),
    }
    tasks: list[ExactRecoveryTask] = []
    for replicate_index in range(int(baseline["replicates_per_world"])):
        for world_id in config["exact_worlds"]:
            run_id = _run_id(
                config["study"]["id"],
                master_seed,
                world_id,
                replicate_index,
            )
            tasks.append(
                ExactRecoveryTask(
                    task_index=len(tasks),
                    run_id=run_id,
                    exact_world_id=world_id,
                    replicate_index=replicate_index,
                    dataset_seed=_child_seed(run_id, "dataset"),
                    split_seed=_child_seed(run_id, "split"),
                    model_seed=_child_seed(run_id, "model"),
                    world_kwargs=world_kwargs,
                )
            )
    return tasks


def preflight_exact_model_recovery(config: dict[str, Any], *, workers: int | None = None) -> dict[str, Any]:
    tasks = build_exact_seed_schedule(config)
    worker_count = _resolve_workers(workers)
    return {
        "study_id": EXACT_MODEL_STUDY_ID,
        "total_exact_datasets": len(tasks),
        "total_model_fits": len(tasks) * len(STATIC_MODEL_IDS),
        "exact_worlds": list(config["exact_worlds"]),
        "models": list(STATIC_MODEL_IDS),
        "workers": worker_count,
        "stress_grid": "not_used",
        "checkpointing": "not_used_for_small_diagnostic",
        "output_dir": str(config["outputs"]["directory"]),
        "note": "M2.6b is diagnostic and does not amend the frozen M2.6 result.",
    }


def run_exact_model_recovery(
    *,
    config_path: str | Path = EXACT_CONFIG_PATH,
    output_dir: str | Path | None = None,
    workers: int | None = None,
    executor: str = "thread",
    task_start: int = 0,
    task_count: int | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    config = load_exact_config(config_path)
    tasks = _slice_tasks(build_exact_seed_schedule(config), task_start=task_start, task_count=task_count)
    worker_count = min(_resolve_workers(workers), max(1, len(tasks)))
    _limit_numeric_threads()
    results = _run_tasks(tasks, config=config, workers=worker_count, executor=executor)
    outputs = _aggregate_results(results)
    target_dir = Path(output_dir or config["outputs"]["directory"])
    paths = _write_outputs(outputs, output_dir=target_dir)
    return {
        "study_id": EXACT_MODEL_STUDY_ID,
        "task_count": len(tasks),
        "workers": worker_count,
        "executor": executor,
        "runtime_seconds": round(float(time.perf_counter() - start), 3),
        "paths": {key: str(value) for key, value in paths.items()},
        "summary": _summary_payload(outputs["correct_model_recovery"]),
    }


def run_m2_self_check(
    *,
    seed: int = 20260808,
    n_replicates: int = 12,
    n_datasets: int = 3,
    participants_per_dataset: int = 36,
    sessions_per_participant: int = 2,
    min_windows_per_session: int = 2,
    max_windows_per_session: int = 4,
    test_fraction: float = 0.25,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR / "m2_self_check",
) -> dict[str, Any]:
    """Compare oracle M2, fitted M2 and fitted M1 on M2-generated data."""

    rows: list[dict[str, Any]] = []
    for replicate_index in range(int(n_replicates)):
        run_id = _run_id("m2_self_check", seed, replicate_index)
        dataset_seed = _child_seed(run_id, "dataset")
        split_seed = _child_seed(run_id, "split")
        model_seed = _child_seed(run_id, "model")
        truth_frame, oracle = make_m2_self_check_world(
            seed=dataset_seed,
            n_datasets=n_datasets,
            participants_per_dataset=participants_per_dataset,
            sessions_per_participant=sessions_per_participant,
            min_windows_per_session=min_windows_per_session,
            max_windows_per_session=max_windows_per_session,
        )
        model_frame = strip_ground_truth_columns(truth_frame)
        split = participant_train_test_split(
            model_frame,
            test_size=test_fraction,
            seed=split_seed,
        )
        train = model_frame.loc[list(split.train_indices)]
        test = model_frame.loc[list(split.test_indices)]
        m1 = M1ContinuousManifoldModel(
            feature_columns=CORE_SYNTHETIC_FEATURES,
            random_state=model_seed + 1,
        ).fit(train)
        m2 = M2NonlinearVigilanceModel(
            feature_columns=CORE_SYNTHETIC_FEATURES,
            random_state=model_seed + 2,
        ).fit(train)
        m2_high_quadrature = M2NonlinearVigilanceModel(
            feature_columns=CORE_SYNTHETIC_FEATURES,
            random_state=model_seed + 2,
            quadrature_points=201,
        ).fit(train)
        m2_projection_search = M2ProjectionSearchVigilanceModel(
            feature_columns=CORE_SYNTHETIC_FEATURES,
            random_state=model_seed + 2,
            quadrature_points=201,
            n_random_projection_candidates=256,
        ).fit(train)
        m2_em = M2LatentCurveEMModel(
            feature_columns=CORE_SYNTHETIC_FEATURES,
            random_state=model_seed + 2,
            quadrature_points=201,
            max_em_iter=80,
            em_tol=1e-5,
        ).fit(train)
        m1_scores = m1.score_samples(test).astype(float)
        m2_scores = m2.score_samples(test).astype(float)
        m2_high_quadrature_scores = m2_high_quadrature.score_samples(test).astype(float)
        m2_projection_search_scores = m2_projection_search.score_samples(test).astype(float)
        m2_em_scores = m2_em.score_samples(test).astype(float)
        oracle_scores, oracle_linear_direction = _oracle_m2_scores(test, oracle, adapter=m2.adapter_)
        oracle_projection_scores = _fit_and_score_m2_fixed_projection(
            train,
            test,
            oracle_linear_direction,
            quadrature_points=201,
        )
        m2_projection_alignment = float(abs(np.dot(m2.projection_, oracle_linear_direction)))
        m2_projection_search_alignment = float(
            abs(np.dot(m2_projection_search.projection_, oracle_linear_direction))
        )
        rows.append(
            {
                "run_id": run_id,
                "replicate_index": replicate_index,
                "n_test_rows": int(test.shape[0]),
                "oracle_m2_mean_log_density": float(np.nanmean(oracle_scores)),
                "fitted_m2_mean_log_density": float(m2_scores.mean()),
                "fitted_m2_high_quadrature_mean_log_density": float(
                    m2_high_quadrature_scores.mean()
                ),
                "fitted_m2_projection_search_mean_log_density": float(
                    m2_projection_search_scores.mean()
                ),
                "fitted_m2_em_mean_log_density": float(m2_em_scores.mean()),
                "fitted_m2_oracle_projection_mean_log_density": float(
                    oracle_projection_scores.mean()
                ),
                "fitted_m1_mean_log_density": float(m1_scores.mean()),
                "oracle_minus_m1": float(np.nanmean(oracle_scores) - m1_scores.mean()),
                "fitted_m2_minus_m1": float(m2_scores.mean() - m1_scores.mean()),
                "fitted_m2_high_quadrature_minus_m1": float(
                    m2_high_quadrature_scores.mean() - m1_scores.mean()
                ),
                "fitted_m2_projection_search_minus_m1": float(
                    m2_projection_search_scores.mean() - m1_scores.mean()
                ),
                "fitted_m2_em_minus_m1": float(m2_em_scores.mean() - m1_scores.mean()),
                "fitted_m2_oracle_projection_minus_m1": float(
                    oracle_projection_scores.mean() - m1_scores.mean()
                ),
                "oracle_minus_fitted_m2": float(np.nanmean(oracle_scores) - m2_scores.mean()),
                "oracle_minus_fitted_m2_high_quadrature": float(
                    np.nanmean(oracle_scores) - m2_high_quadrature_scores.mean()
                ),
                "oracle_minus_fitted_m2_projection_search": float(
                    np.nanmean(oracle_scores) - m2_projection_search_scores.mean()
                ),
                "oracle_minus_fitted_m2_em": float(
                    np.nanmean(oracle_scores) - m2_em_scores.mean()
                ),
                "oracle_minus_fitted_m2_oracle_projection": float(
                    np.nanmean(oracle_scores) - oracle_projection_scores.mean()
                ),
                "m2_projection_alignment_abs": m2_projection_alignment,
                "m2_projection_search_alignment_abs": m2_projection_search_alignment,
                "m2_mean_residual_variance": float(np.mean(m2.residual_variance_)),
                "m2_high_quadrature_mean_residual_variance": float(
                    np.mean(m2_high_quadrature.residual_variance_)
                ),
                "m2_projection_search_mean_residual_variance": float(
                    np.mean(m2_projection_search.residual_variance_)
                ),
                "m2_em_mean_residual_variance": float(np.mean(m2_em.residual_variance_)),
                "m2_em_n_iter": int(m2_em.em_n_iter_ or 0),
                "m2_projection_search_candidates_evaluated": int(
                    m2_projection_search.projection_search_candidates_evaluated_ or 0
                ),
            }
        )
    frame = pd.DataFrame(rows)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    detail_path = output_root / "m2_self_check.csv"
    frame.to_csv(detail_path, index=False)
    summary = _m2_self_check_summary(frame)
    summary_path = output_root / "m2_self_check_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "study_id": "m2_self_check_v1",
        "n_replicates": int(n_replicates),
        "paths": {
            "detail": str(detail_path),
            "summary": str(summary_path),
        },
        "summary": summary,
    }


def run_ew2_em_diagnostic(
    *,
    config_path: str | Path = EXACT_CONFIG_PATH,
    config: dict[str, Any] | None = None,
    output_dir: str | Path = DEFAULT_EW2_EM_OUTPUT_DIR,
    n_replicates: int = 10,
    workers: int | None = None,
    executor: str = "thread",
    em_quadrature_points: int = 201,
    em_max_iter: int = 80,
) -> dict[str, Any]:
    """Run an EW2-only exact-model diagnostic with an added EM M2 candidate."""

    start = time.perf_counter()
    if config is None:
        config = load_exact_config(config_path)
    else:
        _validate_exact_config(config)
    ew2_tasks = [
        task
        for task in build_exact_seed_schedule(config)
        if task.exact_world_id == "EW2_m2_exact"
    ][: int(n_replicates)]
    if not ew2_tasks:
        raise ValueError("n_replicates must select at least one EW2 task")
    worker_count = min(_resolve_workers(workers), len(ew2_tasks))
    _limit_numeric_threads()
    results = _run_ew2_em_tasks(
        ew2_tasks,
        config=config,
        workers=worker_count,
        executor=executor,
        em_quadrature_points=em_quadrature_points,
        em_max_iter=em_max_iter,
    )
    outputs = _aggregate_ew2_em_results(results)
    target_dir = Path(output_dir)
    paths = _write_ew2_em_outputs(outputs, output_dir=target_dir)
    summary = _ew2_em_summary(outputs["run_summary"], outputs["model_scores"])
    summary_path = target_dir / "ew2_em_diagnostic_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    paths["summary"] = summary_path
    return {
        "study_id": EW2_EM_DIAGNOSTIC_STUDY_ID,
        "n_runs": len(ew2_tasks),
        "workers": worker_count,
        "executor": executor,
        "runtime_seconds": round(float(time.perf_counter() - start), 3),
        "paths": {key: str(value) for key, value in paths.items()},
        "summary": summary,
    }


def run_m2_em_convergence_diagnostic(
    *,
    config_path: str | Path = EXACT_CONFIG_PATH,
    config: dict[str, Any] | None = None,
    output_dir: str | Path = DEFAULT_M2_EM_CONVERGENCE_OUTPUT_DIR,
    n_replicates: int = 12,
    workers: int | None = None,
    executor: str = "thread",
    em_quadrature_points: int = 201,
    max_iters: Sequence[int] = (80, 120, 200),
) -> dict[str, Any]:
    """Check whether the M2_EM iteration cap is binding on exact EW2 datasets."""

    start = time.perf_counter()
    if config is None:
        config = load_exact_config(config_path)
    else:
        _validate_exact_config(config)
    ew2_tasks = [
        task
        for task in build_exact_seed_schedule(config)
        if task.exact_world_id == "EW2_m2_exact"
    ][: int(n_replicates)]
    if not ew2_tasks:
        raise ValueError("n_replicates must select at least one EW2 task")
    iter_grid = tuple(sorted({int(value) for value in max_iters}))
    if not iter_grid or any(value < 1 for value in iter_grid):
        raise ValueError("max_iters must contain positive integers")
    worker_count = min(_resolve_workers(workers), len(ew2_tasks))
    _limit_numeric_threads()
    rows = _run_m2_em_convergence_tasks(
        ew2_tasks,
        config=config,
        workers=worker_count,
        executor=executor,
        em_quadrature_points=em_quadrature_points,
        max_iters=iter_grid,
    )
    detail = pd.DataFrame(rows)
    summary_frame = _m2_em_convergence_summary_frame(detail)
    summary = _m2_em_convergence_summary_payload(summary_frame)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    detail_path = target_dir / "m2_em_convergence_detail.csv"
    summary_csv_path = target_dir / "m2_em_convergence_summary.csv"
    summary_json_path = target_dir / "m2_em_convergence_summary.json"
    detail.to_csv(detail_path, index=False)
    summary_frame.to_csv(summary_csv_path, index=False)
    summary_json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "study_id": M2_EM_CONVERGENCE_STUDY_ID,
        "n_runs": len(ew2_tasks),
        "max_iters": list(iter_grid),
        "workers": worker_count,
        "executor": executor,
        "runtime_seconds": round(float(time.perf_counter() - start), 3),
        "paths": {
            "detail": str(detail_path),
            "summary_csv": str(summary_csv_path),
            "summary": str(summary_json_path),
        },
        "summary": summary,
    }


def run_ew0_m1_nesting_diagnostic(
    *,
    config_path: str | Path = EXACT_CONFIG_PATH,
    config: dict[str, Any] | None = None,
    output_dir: str | Path = DEFAULT_EW0_M1_NESTING_OUTPUT_DIR,
    n_replicates: int = 12,
    bootstrap_samples: int = 16,
) -> dict[str, Any]:
    """Diagnose whether M1's second dimension is substantive under exact EW0 data."""

    start = time.perf_counter()
    if config is None:
        config = load_exact_config(config_path)
    else:
        _validate_exact_config(config)
    ew0_tasks = [
        task
        for task in build_exact_seed_schedule(config)
        if task.exact_world_id == "EW0_m0_exact"
    ][: int(n_replicates)]
    if not ew0_tasks:
        raise ValueError("n_replicates must select at least one EW0 task")
    _limit_numeric_threads()
    rows = [
        run_ew0_m1_nesting_task(
            task,
            config,
            bootstrap_samples=bootstrap_samples,
        )
        for task in ew0_tasks
    ]
    detail = pd.DataFrame(rows)
    summary = _ew0_m1_nesting_summary(detail)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    detail_path = target_dir / "ew0_m1_nesting_detail.csv"
    summary_path = target_dir / "ew0_m1_nesting_summary.json"
    detail.to_csv(detail_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "study_id": EW0_M1_NESTING_STUDY_ID,
        "n_runs": len(ew0_tasks),
        "bootstrap_samples": int(bootstrap_samples),
        "runtime_seconds": round(float(time.perf_counter() - start), 3),
        "paths": {
            "detail": str(detail_path),
            "summary": str(summary_path),
        },
        "summary": summary,
    }


def run_exact_task(task: ExactRecoveryTask, config: dict[str, Any]) -> dict[str, Any]:
    truth_frame = make_exact_model_world(
        task.exact_world_id,
        seed=task.dataset_seed,
        **task.world_kwargs,
    )
    model_frame = strip_ground_truth_columns(truth_frame)
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
    aligned_model_id = EXACT_WORLD_MODEL_ALIGNMENT[task.exact_world_id]
    common = {
        "run_id": task.run_id,
        "task_index": task.task_index,
        "exact_world_id": task.exact_world_id,
        "aligned_model_id": aligned_model_id,
        "replicate_index": task.replicate_index,
        "selected_model_id": selection.selected_model_id,
        "numerical_best_model_id": selection.numerical_best_model_id,
        "selection_reason": selection.selection_reason,
        "correct_model_selected": selection.selected_model_id == aligned_model_id,
        "per_window_winner": selection.per_window_winner,
        "participant_weighted_winner": selection.participant_weighted_winner,
        "per_observed_feature_winner": selection.per_observed_feature_winner,
        "normalisation_sensitive": selection.normalisation_sensitive,
        "n_sources": int(truth_frame["source_dataset"].nunique()),
        **task.world_kwargs,
    }
    model_rows = [
        {
            **common,
            "model_id": str(row["model_id"]),
            **{
                key: row[key]
                for key in row.index
                if key not in {"model_id", "primary_metric"}
            },
        }
        for _, row in frozen.model_scores.iterrows()
    ]
    participant_difference_rows = _participant_difference_rows(
        task=task,
        participant_scores=frozen.participant_scores,
        numerical_best_model_id=selection.numerical_best_model_id,
    )
    return {
        "run_summary": common,
        "model_rows": model_rows,
        "participant_difference_rows": participant_difference_rows,
    }


def run_ew2_em_task(
    task: ExactRecoveryTask,
    config: dict[str, Any],
    *,
    em_quadrature_points: int,
    em_max_iter: int,
) -> dict[str, Any]:
    if task.exact_world_id != "EW2_m2_exact":
        raise ValueError("EW2 EM diagnostic only accepts EW2_m2_exact tasks")
    truth_frame = make_exact_model_world(
        task.exact_world_id,
        seed=task.dataset_seed,
        **task.world_kwargs,
    )
    model_frame = strip_ground_truth_columns(truth_frame)
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
    model_scores, participant_scores = _append_em_m2_candidate(
        model_frame,
        split=split,
        model_scores=frozen.model_scores,
        participant_scores=frozen.participant_scores,
        random_state=task.model_seed + 2,
        em_quadrature_points=em_quadrature_points,
        em_max_iter=em_max_iter,
    )
    selection = select_preferred_model(
        model_scores,
        participant_scores,
        practical_equivalence_margin=float(config["selection"]["practical_equivalence_margin"]),
        paired_ci_z=float(config["selection"]["paired_ci_z"]),
    )
    score_by_model = model_scores.set_index("model_id")[PRIMARY_METRIC].astype(float)
    common = {
        "run_id": task.run_id,
        "task_index": task.task_index,
        "exact_world_id": task.exact_world_id,
        "aligned_model_id": EXACT_WORLD_MODEL_ALIGNMENT[task.exact_world_id],
        "diagnostic_model_id": EM_M2_DIAGNOSTIC_MODEL_ID,
        "replicate_index": task.replicate_index,
        "selected_model_id": selection.selected_model_id,
        "numerical_best_model_id": selection.numerical_best_model_id,
        "selection_reason": selection.selection_reason,
        "standard_m2_selected": selection.selected_model_id == "M2_nonlinear_vigilance",
        "em_m2_selected": selection.selected_model_id == EM_M2_DIAGNOSTIC_MODEL_ID,
        "em_m2_numerical_best": selection.numerical_best_model_id == EM_M2_DIAGNOSTIC_MODEL_ID,
        "em_minus_m1": float(
            score_by_model[EM_M2_DIAGNOSTIC_MODEL_ID]
            - score_by_model["M1_continuous_control_manifold"]
        ),
        "em_minus_standard_m2": float(
            score_by_model[EM_M2_DIAGNOSTIC_MODEL_ID]
            - score_by_model["M2_nonlinear_vigilance"]
        ),
        "per_window_winner": selection.per_window_winner,
        "participant_weighted_winner": selection.participant_weighted_winner,
        "per_observed_feature_winner": selection.per_observed_feature_winner,
        "normalisation_sensitive": selection.normalisation_sensitive,
        "n_sources": int(truth_frame["source_dataset"].nunique()),
        "em_quadrature_points": int(em_quadrature_points),
        "em_max_iter": int(em_max_iter),
        **task.world_kwargs,
    }
    model_rows = [
        {
            **common,
            "model_id": str(row["model_id"]),
            **{
                key: row[key]
                for key in row.index
                if key not in {"model_id", "primary_metric"}
            },
        }
        for _, row in model_scores.iterrows()
    ]
    participant_difference_rows = _participant_difference_rows_for_models(
        task=task,
        participant_scores=participant_scores,
        numerical_best_model_id=selection.numerical_best_model_id,
        model_ids=tuple(model_scores["model_id"].astype(str)),
    )
    return {
        "run_summary": common,
        "model_rows": model_rows,
        "participant_difference_rows": participant_difference_rows,
    }


def run_m2_em_convergence_task(
    task: ExactRecoveryTask,
    config: dict[str, Any],
    *,
    em_quadrature_points: int,
    max_iters: Sequence[int],
) -> list[dict[str, Any]]:
    if task.exact_world_id != "EW2_m2_exact":
        raise ValueError("M2_EM convergence diagnostic only accepts EW2_m2_exact tasks")
    truth_frame = make_exact_model_world(
        task.exact_world_id,
        seed=task.dataset_seed,
        **task.world_kwargs,
    )
    model_frame = strip_ground_truth_columns(truth_frame)
    split = participant_train_test_split(
        model_frame,
        test_size=float(config["split"]["test_fraction"]),
        seed=task.split_seed,
    )
    train = model_frame.loc[list(split.train_indices)].copy()
    test = model_frame.loc[list(split.test_indices)].copy()
    m1 = M1ContinuousManifoldModel(
        feature_columns=CORE_SYNTHETIC_FEATURES,
        random_state=task.model_seed + 1,
    ).fit(train)
    m1_scores = m1.score_samples(test).astype(float)
    m1_mean = float(m1_scores.dropna().mean())
    rows: list[dict[str, Any]] = []
    for max_iter in max_iters:
        model = M2LatentCurveEMModel(
            feature_columns=CORE_SYNTHETIC_FEATURES,
            random_state=task.model_seed + 2,
            quadrature_points=em_quadrature_points,
            max_em_iter=int(max_iter),
        ).fit(train)
        scores = model.score_samples(test).astype(float)
        em_mean = float(scores.dropna().mean())
        rows.append(
            {
                "run_id": task.run_id,
                "task_index": task.task_index,
                "exact_world_id": task.exact_world_id,
                "replicate_index": task.replicate_index,
                "max_iter": int(max_iter),
                "em_n_iter": int(model.em_n_iter_ or 0),
                "em_converged": bool(model.em_converged_),
                "em_hit_iteration_cap": int(model.em_n_iter_ or 0) >= int(max_iter),
                "em_train_log_likelihood": float(
                    model.em_train_log_likelihood_ or float("nan")
                ),
                "em_final_likelihood_change": float(
                    model.em_final_likelihood_change_ or float("nan")
                ),
                "em_final_parameter_change": float(
                    model.em_final_parameter_change_ or float("nan")
                ),
                "em_mean_residual_variance": float(np.mean(model.residual_variance_)),
                "m1_mean_log_density": m1_mean,
                "em_mean_log_density": em_mean,
                "em_minus_m1": float(em_mean - m1_mean),
                "em_beats_m1": em_mean > m1_mean,
                "em_quadrature_points": int(em_quadrature_points),
                **task.world_kwargs,
            }
        )
    return rows


def run_ew0_m1_nesting_task(
    task: ExactRecoveryTask,
    config: dict[str, Any],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    if task.exact_world_id != "EW0_m0_exact":
        raise ValueError("EW0/M1 nesting diagnostic only accepts EW0_m0_exact tasks")
    truth_frame = make_exact_model_world(
        task.exact_world_id,
        seed=task.dataset_seed,
        **task.world_kwargs,
    )
    model_frame = strip_ground_truth_columns(truth_frame)
    split = participant_train_test_split(
        model_frame,
        test_size=float(config["split"]["test_fraction"]),
        seed=task.split_seed,
    )
    train = model_frame.loc[list(split.train_indices)].copy()
    test = model_frame.loc[list(split.test_indices)].copy()
    m0 = M0ProbabilisticPCAModel(
        feature_columns=CORE_SYNTHETIC_FEATURES,
        random_state=task.model_seed,
    ).fit(train)
    m1 = M1ContinuousManifoldModel(
        feature_columns=CORE_SYNTHETIC_FEATURES,
        random_state=task.model_seed + 1,
    ).fit(train)
    m0_scores = m0.score_samples(test).astype(float)
    m1_scores = m1.score_samples(test).astype(float)
    m0_participant = _participant_log_density_means(test, m0_scores)
    m1_participant = _participant_log_density_means(test, m1_scores)
    paired_delta = (m1_participant - m0_participant).dropna()
    paired_n = int(paired_delta.shape[0])
    delta_mean = float(paired_delta.mean()) if paired_n else float("nan")
    delta_sd = float(paired_delta.std(ddof=1)) if paired_n > 1 else float("nan")
    delta_se = delta_sd / np.sqrt(paired_n) if paired_n > 1 else float("nan")
    second_stability = _m1_second_dimension_bootstrap_stability(
        train,
        base_model=m1,
        seed=_child_seed(task.run_id, "ew0_m1_bootstrap"),
        bootstrap_samples=bootstrap_samples,
    )
    magnitude = _m1_second_dimension_magnitude(m1)
    return {
        "run_id": task.run_id,
        "task_index": task.task_index,
        "exact_world_id": task.exact_world_id,
        "replicate_index": task.replicate_index,
        "m0_mean_log_density": float(m0_scores.dropna().mean()),
        "m1_mean_log_density": float(m1_scores.dropna().mean()),
        "m1_minus_m0": float(m1_scores.dropna().mean() - m0_scores.dropna().mean()),
        "m1_numerically_best": float(m1_scores.dropna().mean() > m0_scores.dropna().mean()),
        "paired_m1_minus_m0": delta_mean,
        "paired_delta_se": delta_se,
        "paired_delta_ci_low": delta_mean - 1.96 * delta_se if paired_n > 1 else float("nan"),
        "paired_delta_ci_high": delta_mean + 1.96 * delta_se if paired_n > 1 else float("nan"),
        "paired_n_participants": paired_n,
        "m1_second_eigenvalue": magnitude["second_eigenvalue"],
        "m1_first_eigenvalue": magnitude["first_eigenvalue"],
        "m1_second_to_first_eigenvalue_ratio": magnitude["second_to_first_eigenvalue_ratio"],
        "m1_second_loading_norm": magnitude["second_loading_norm"],
        "m1_second_loading_fraction": magnitude["second_loading_fraction"],
        "m1_second_dimension_bootstrap_stability": second_stability,
        "bootstrap_samples": int(bootstrap_samples),
        **task.world_kwargs,
    }


def make_exact_model_world(
    exact_world_id: ExactWorldId,
    *,
    seed: int,
    n_datasets: int,
    participants_per_dataset: int,
    sessions_per_participant: int,
    min_windows_per_session: int,
    max_windows_per_session: int,
    technical_missingness_rate: float = 0.0,
    exact_signal_scale: float = 1.0,
) -> pd.DataFrame:
    if exact_world_id not in EXACT_MODEL_WORLD_IDS:
        raise ValueError(f"unsupported exact world: {exact_world_id}")
    if n_datasets < 2:
        raise ValueError("at least two datasets are required")
    if participants_per_dataset < 2:
        raise ValueError("at least two participants per dataset are required")
    if not 1 <= min_windows_per_session <= max_windows_per_session:
        raise ValueError("invalid window range")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    tasks = ("stroop", "flanker", "sart", "task_switch")
    sources = [f"{exact_world_id}_source_{index + 1}" for index in range(n_datasets)]
    checksum = "sha256:" + hashlib.sha256(f"{exact_world_id}:{seed}".encode("utf-8")).hexdigest()
    for source_index, source in enumerate(sources):
        for participant_index in range(participants_per_dataset):
            participant_id = f"{source}_p{participant_index:03d}"
            for session_number in range(1, sessions_per_participant + 1):
                n_windows = int(rng.integers(min_windows_per_session, max_windows_per_session + 1))
                for window_number in range(1, n_windows + 1):
                    task_id = tasks[(source_index + participant_index + session_number + window_number) % len(tasks)]
                    standardised, truth = _exact_standardised_features(
                        exact_world_id,
                        rng,
                        exact_signal_scale=exact_signal_scale,
                    )
                    row = _canonical_minimal_row(
                        exact_world_id=exact_world_id,
                        checksum=checksum,
                        source=source,
                        participant_id=participant_id,
                        session_number=session_number,
                        window_number=window_number,
                        task_id=task_id,
                        standardised=standardised,
                        truth=truth,
                    )
                    if rng.random() < technical_missingness_rate:
                        for feature in ("median_rt_ms", "mean_response_speed", "throughput_proxy"):
                            row[feature] = np.nan
                    rows.append(row)
    return pd.DataFrame(rows)


def make_m2_self_check_world(
    *,
    seed: int,
    n_datasets: int,
    participants_per_dataset: int,
    sessions_per_participant: int,
    min_windows_per_session: int,
    max_windows_per_session: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Generate canonical rows from a known M2 likelihood in raw feature units."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    tasks = ("stroop", "flanker", "sart", "task_switch")
    sources = [f"M2_self_check_source_{index + 1}" for index in range(n_datasets)]
    checksum = "sha256:" + hashlib.sha256(f"m2-self-check:{seed}".encode("utf-8")).hexdigest()
    raw_base = np.array([0.78, 710.0, 1.42, 0.24, 1.10])
    raw_scale = np.array([0.060, 58.0, 0.125, 0.050, 0.165])
    coefficients = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.15, -0.10, 0.13, -0.08, 0.11],
            [0.85, -0.82, 0.90, 0.65, 0.82],
        ],
        dtype=float,
    )
    residual_sd = np.array([0.16, 0.16, 0.16, 0.16, 0.16], dtype=float)
    linear_direction = coefficients[1] / np.linalg.norm(coefficients[1])
    for source_index, source in enumerate(sources):
        for participant_index in range(participants_per_dataset):
            participant_id = f"{source}_p{participant_index:03d}"
            for session_number in range(1, sessions_per_participant + 1):
                n_windows = int(rng.integers(min_windows_per_session, max_windows_per_session + 1))
                for window_number in range(1, n_windows + 1):
                    task_id = tasks[(source_index + participant_index + session_number + window_number) % len(tasks)]
                    z = float(rng.normal())
                    basis = np.array([1.0, z, z**2 - 1.0])
                    standardised = basis @ coefficients + rng.normal(0.0, residual_sd, size=5)
                    raw = raw_base + standardised * raw_scale
                    rows.append(
                        _canonical_raw_row(
                            exact_world_id="M2_self_check",
                            checksum=checksum,
                            source=source,
                            participant_id=participant_id,
                            session_number=session_number,
                            window_number=window_number,
                            task_id=task_id,
                            raw_features=raw,
                            truth={
                                "latent_1": z,
                                "latent_2": 0.0,
                                "readiness": z,
                                "component": None,
                            },
                        )
                    )
    return pd.DataFrame(rows), {
        "raw_base": raw_base,
        "raw_scale": raw_scale,
        "coefficients": coefficients,
        "residual_variance": residual_sd**2,
        "linear_direction": linear_direction,
    }


def _run_tasks(
    tasks: Sequence[ExactRecoveryTask],
    *,
    config: dict[str, Any],
    workers: int,
    executor: str,
) -> list[dict[str, Any]]:
    if executor == "serial" or workers <= 1 or len(tasks) <= 1:
        return [run_exact_task(task, config) for task in tasks]
    pool_class = ProcessPoolExecutor if executor == "process" else ThreadPoolExecutor
    results: list[dict[str, Any]] = []
    with pool_class(max_workers=workers) as pool:
        futures = [pool.submit(run_exact_task, task, config) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda item: int(item["run_summary"]["task_index"]))


def _run_ew2_em_tasks(
    tasks: Sequence[ExactRecoveryTask],
    *,
    config: dict[str, Any],
    workers: int,
    executor: str,
    em_quadrature_points: int,
    em_max_iter: int,
) -> list[dict[str, Any]]:
    if executor == "serial" or workers <= 1 or len(tasks) <= 1:
        return [
            run_ew2_em_task(
                task,
                config,
                em_quadrature_points=em_quadrature_points,
                em_max_iter=em_max_iter,
            )
            for task in tasks
        ]
    pool_class = ProcessPoolExecutor if executor == "process" else ThreadPoolExecutor
    results: list[dict[str, Any]] = []
    with pool_class(max_workers=workers) as pool:
        futures = [
            pool.submit(
                run_ew2_em_task,
                task,
                config,
                em_quadrature_points=em_quadrature_points,
                em_max_iter=em_max_iter,
            )
            for task in tasks
        ]
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda item: int(item["run_summary"]["task_index"]))


def _run_m2_em_convergence_tasks(
    tasks: Sequence[ExactRecoveryTask],
    *,
    config: dict[str, Any],
    workers: int,
    executor: str,
    em_quadrature_points: int,
    max_iters: Sequence[int],
) -> list[dict[str, Any]]:
    if executor == "serial" or workers <= 1 or len(tasks) <= 1:
        rows = [
            row
            for task in tasks
            for row in run_m2_em_convergence_task(
                task,
                config,
                em_quadrature_points=em_quadrature_points,
                max_iters=max_iters,
            )
        ]
        return sorted(rows, key=lambda item: (int(item["task_index"]), int(item["max_iter"])))
    pool_class = ProcessPoolExecutor if executor == "process" else ThreadPoolExecutor
    rows: list[dict[str, Any]] = []
    with pool_class(max_workers=workers) as pool:
        futures = [
            pool.submit(
                run_m2_em_convergence_task,
                task,
                config,
                em_quadrature_points=em_quadrature_points,
                max_iters=max_iters,
            )
            for task in tasks
        ]
        for future in as_completed(futures):
            rows.extend(future.result())
    return sorted(rows, key=lambda item: (int(item["task_index"]), int(item["max_iter"])))


def _append_em_m2_candidate(
    frame_without_truth: pd.DataFrame,
    *,
    split: Any,
    model_scores: pd.DataFrame,
    participant_scores: pd.DataFrame,
    random_state: int,
    em_quadrature_points: int,
    em_max_iter: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame_without_truth.loc[list(split.train_indices)].copy()
    test = frame_without_truth.loc[list(split.test_indices)].copy()
    model = M2LatentCurveEMModel(
        feature_columns=CORE_SYNTHETIC_FEATURES,
        random_state=random_state,
        quadrature_points=em_quadrature_points,
        max_em_iter=em_max_iter,
    ).fit(train)
    sample_scores = model.score_samples(test).astype(float)
    valid_scores = sample_scores.dropna()
    observed_feature_counts = _observed_feature_counts(test, CORE_SYNTHETIC_FEATURES)
    total_observed_features = int(observed_feature_counts.sum())
    total_log_density = float(valid_scores.sum()) if not valid_scores.empty else float("nan")
    participant_means = _participant_log_density_means(test, sample_scores)
    per_feature = (
        float(total_log_density / total_observed_features)
        if total_observed_features and np.isfinite(total_log_density)
        else float("nan")
    )
    em_row = {
        "model_id": model.model_id,
        "primary_metric": PRIMARY_METRIC,
        "heldout_log_likelihood_total": total_log_density,
        "heldout_log_density_mean_per_window": float(valid_scores.mean()),
        "heldout_log_density_participant_weighted": float(participant_means.mean()),
        "heldout_log_density_mean_per_observed_feature": per_feature,
        "n_valid_windows": int(valid_scores.shape[0]),
        "n_test_participants": int(participant_means.shape[0]),
        "n_observed_feature_values": total_observed_features,
        "diagnostic_mean_residual_variance": float(np.mean(model.residual_variance_)),
        "diagnostic_em_n_iter": float(model.em_n_iter_ or 0),
        "diagnostic_em_converged": float(bool(model.em_converged_)),
        "diagnostic_em_final_likelihood_change": float(
            model.em_final_likelihood_change_ or float("nan")
        ),
        "diagnostic_em_final_parameter_change": float(
            model.em_final_parameter_change_ or float("nan")
        ),
        "diagnostic_em_train_log_likelihood": float(
            model.em_train_log_likelihood_ or float("nan")
        ),
    }
    augmented_model_scores = pd.concat(
        [model_scores.copy(), pd.DataFrame([em_row])],
        ignore_index=True,
    )
    augmented_participant_scores = participant_scores.copy()
    augmented_participant_scores[model.model_id] = participant_means
    return augmented_model_scores, augmented_participant_scores


def _aggregate_results(results: Sequence[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    run_summary = pd.DataFrame([result["run_summary"] for result in results])
    model_scores = pd.DataFrame([row for result in results for row in result["model_rows"]])
    participant_score_differences = pd.DataFrame(
        [row for result in results for row in result["participant_difference_rows"]]
    )
    return {
        "run_summary": run_summary,
        "model_scores": model_scores,
        "participant_score_differences": participant_score_differences,
        "recovery_matrix": _recovery_matrix(run_summary),
        "correct_model_recovery": _correct_model_recovery(run_summary),
        "selection_reason_counts": _selection_reason_counts(run_summary),
    }


def _aggregate_ew2_em_results(results: Sequence[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    run_summary = pd.DataFrame([result["run_summary"] for result in results])
    model_scores = pd.DataFrame([row for result in results for row in result["model_rows"]])
    participant_score_differences = pd.DataFrame(
        [row for result in results for row in result["participant_difference_rows"]]
    )
    return {
        "run_summary": run_summary,
        "model_scores": model_scores,
        "participant_score_differences": participant_score_differences,
    }


def _write_outputs(outputs: dict[str, pd.DataFrame], *, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in outputs.items():
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    return paths


def _write_ew2_em_outputs(outputs: dict[str, pd.DataFrame], *, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in outputs.items():
        path = output_dir / f"ew2_em_diagnostic_{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    return paths


def _ew2_em_summary(run_summary: pd.DataFrame, model_scores: pd.DataFrame) -> dict[str, Any]:
    n_runs = int(run_summary.shape[0])
    selected_counts = run_summary["selected_model_id"].value_counts(sort=False).to_dict()
    numerical_best_counts = run_summary["numerical_best_model_id"].value_counts(sort=False).to_dict()
    score_pivot = model_scores.pivot_table(
        index="run_id",
        columns="model_id",
        values=PRIMARY_METRIC,
        aggfunc="first",
    )
    em_minus_m1 = score_pivot[EM_M2_DIAGNOSTIC_MODEL_ID] - score_pivot[
        "M1_continuous_control_manifold"
    ]
    em_minus_standard_m2 = score_pivot[EM_M2_DIAGNOSTIC_MODEL_ID] - score_pivot[
        "M2_nonlinear_vigilance"
    ]
    em_selected_rate = float(run_summary["em_m2_selected"].mean()) if n_runs else float("nan")
    em_numerical_best_rate = (
        float(run_summary["em_m2_numerical_best"].mean()) if n_runs else float("nan")
    )
    return {
        "n_runs": n_runs,
        "em_quadrature_points": int(run_summary["em_quadrature_points"].iloc[0])
        if n_runs and "em_quadrature_points" in run_summary
        else None,
        "em_max_iter": int(run_summary["em_max_iter"].iloc[0])
        if n_runs and "em_max_iter" in run_summary
        else None,
        "selected_counts": {str(key): int(value) for key, value in selected_counts.items()},
        "numerical_best_counts": {
            str(key): int(value) for key, value in numerical_best_counts.items()
        },
        "standard_m2_selected_rate": float(run_summary["standard_m2_selected"].mean())
        if n_runs
        else float("nan"),
        "em_m2_selected_rate": em_selected_rate,
        "em_m2_numerical_best_rate": em_numerical_best_rate,
        "mean_m1_log_density": float(
            score_pivot["M1_continuous_control_manifold"].mean()
        ),
        "mean_standard_m2_log_density": float(score_pivot["M2_nonlinear_vigilance"].mean()),
        "mean_em_m2_log_density": float(score_pivot[EM_M2_DIAGNOSTIC_MODEL_ID].mean()),
        "mean_em_minus_m1": float(em_minus_m1.mean()),
        "mean_em_minus_standard_m2": float(em_minus_standard_m2.mean()),
        "decision": _ew2_em_decision(em_selected_rate, em_numerical_best_rate),
    }


def _ew2_em_decision(em_selected_rate: float, em_numerical_best_rate: float) -> str:
    if em_selected_rate >= 0.8:
        return "em_m2_repairs_ew2_exact_selection"
    if em_numerical_best_rate >= 0.8:
        return "em_m2_improves_ew2_numerical_fit_but_selection_remains_conservative"
    if em_numerical_best_rate >= 0.5:
        return "em_m2_partially_improves_ew2_numerical_fit"
    return "em_m2_does_not_repair_ew2_exact_fit"


def _m2_em_convergence_summary_frame(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for max_iter, group in detail.groupby("max_iter", sort=True):
        rows.append(
            {
                "max_iter": int(max_iter),
                "n_runs": int(group.shape[0]),
                "convergence_rate": float(group["em_converged"].mean()),
                "iteration_cap_hit_rate": float(group["em_hit_iteration_cap"].mean()),
                "mean_em_n_iter": float(group["em_n_iter"].mean()),
                "mean_train_log_likelihood": float(group["em_train_log_likelihood"].mean()),
                "mean_final_likelihood_change": float(
                    group["em_final_likelihood_change"].mean()
                ),
                "median_final_likelihood_change": float(
                    group["em_final_likelihood_change"].median()
                ),
                "mean_final_parameter_change": float(
                    group["em_final_parameter_change"].mean()
                ),
                "median_final_parameter_change": float(
                    group["em_final_parameter_change"].median()
                ),
                "mean_m1_log_density": float(group["m1_mean_log_density"].mean()),
                "mean_em_log_density": float(group["em_mean_log_density"].mean()),
                "mean_em_minus_m1": float(group["em_minus_m1"].mean()),
                "em_beats_m1_rate": float(group["em_beats_m1"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _m2_em_convergence_summary_payload(summary_frame: pd.DataFrame) -> dict[str, Any]:
    if summary_frame.empty:
        return {"decision": "no_m2_em_convergence_rows"}
    ordered = summary_frame.sort_values("max_iter").copy()
    best_row = ordered.sort_values(
        ["mean_em_log_density", "max_iter"],
        ascending=[False, True],
        kind="mergesort",
    ).iloc[0]
    baseline_row = ordered.iloc[0]
    heldout_gain_over_lowest_cap = float(
        best_row["mean_em_log_density"] - baseline_row["mean_em_log_density"]
    )
    beats_m1_rate_drop = float(
        baseline_row["em_beats_m1_rate"] - best_row["em_beats_m1_rate"]
    )
    decision = _m2_em_convergence_decision(
        baseline_max_iter=int(baseline_row["max_iter"]),
        best_max_iter=int(best_row["max_iter"]),
        heldout_gain_over_lowest_cap=heldout_gain_over_lowest_cap,
        baseline_beats_m1_rate=float(baseline_row["em_beats_m1_rate"]),
        best_beats_m1_rate=float(best_row["em_beats_m1_rate"]),
        baseline_convergence_rate=float(baseline_row["convergence_rate"]),
    )
    return {
        "decision": decision,
        "baseline_max_iter": int(baseline_row["max_iter"]),
        "best_mean_heldout_max_iter": int(best_row["max_iter"]),
        "heldout_gain_over_lowest_cap": heldout_gain_over_lowest_cap,
        "beats_m1_rate_drop_from_lowest_to_best": beats_m1_rate_drop,
        "rows": [
            {
                key: (int(value) if key in {"max_iter", "n_runs"} else float(value))
                for key, value in row.items()
            }
            for row in ordered.to_dict(orient="records")
        ],
    }


def _m2_em_convergence_decision(
    *,
    baseline_max_iter: int,
    best_max_iter: int,
    heldout_gain_over_lowest_cap: float,
    baseline_beats_m1_rate: float,
    best_beats_m1_rate: float,
    baseline_convergence_rate: float,
) -> str:
    if baseline_beats_m1_rate < 0.8 and best_beats_m1_rate >= 0.8:
        return "higher_em_cap_required_for_m2_recovery"
    if best_max_iter > baseline_max_iter and heldout_gain_over_lowest_cap > 0.01:
        return "increase_em_iteration_cap_before_confirmation"
    if baseline_convergence_rate < 0.8:
        return "lowest_cap_performance_stable_but_tolerance_not_met"
    return "lowest_cap_converges_and_performance_stable"


def _m1_second_dimension_magnitude(model: M1ContinuousManifoldModel) -> dict[str, float]:
    if model.eigenvalues_ is None or model.loadings_ is None or model.loadings_.shape[1] < 2:
        return {
            "first_eigenvalue": float("nan"),
            "second_eigenvalue": float("nan"),
            "second_to_first_eigenvalue_ratio": float("nan"),
            "second_loading_norm": float("nan"),
            "second_loading_fraction": float("nan"),
        }
    first = float(model.eigenvalues_[0])
    second = float(model.eigenvalues_[1])
    loading_norms = np.linalg.norm(model.loadings_, axis=0)
    second_norm = float(loading_norms[1])
    total_norm = float(np.sum(loading_norms))
    return {
        "first_eigenvalue": first,
        "second_eigenvalue": second,
        "second_to_first_eigenvalue_ratio": second / first if first else float("nan"),
        "second_loading_norm": second_norm,
        "second_loading_fraction": second_norm / total_norm if total_norm else float("nan"),
    }


def _m1_second_dimension_bootstrap_stability(
    train: pd.DataFrame,
    *,
    base_model: M1ContinuousManifoldModel,
    seed: int,
    bootstrap_samples: int,
) -> float:
    if bootstrap_samples <= 0 or base_model.loadings_ is None or base_model.loadings_.shape[1] < 2:
        return float("nan")
    base = base_model.loadings_[:, 1]
    base_norm = float(np.linalg.norm(base))
    if base_norm == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    groups = list(train.groupby(["source_dataset", "participant_id"], sort=True).groups.values())
    if not groups:
        return float("nan")
    similarities: list[float] = []
    for _ in range(int(bootstrap_samples)):
        sampled_group_indices = rng.integers(0, len(groups), size=len(groups))
        sampled_indices = [
            index
            for group_index in sampled_group_indices
            for index in groups[int(group_index)]
        ]
        sample = train.loc[sampled_indices].reset_index(drop=True)
        try:
            boot = M1ContinuousManifoldModel(
                feature_columns=CORE_SYNTHETIC_FEATURES,
                random_state=int(rng.integers(0, 2**31 - 1)),
            ).fit(sample)
        except Exception:
            continue
        if boot.loadings_ is None or boot.loadings_.shape[1] < 2:
            continue
        candidate = boot.loadings_[:, 1]
        candidate_norm = float(np.linalg.norm(candidate))
        if candidate_norm == 0:
            continue
        similarities.append(abs(float(np.dot(base, candidate) / (base_norm * candidate_norm))))
    return float(np.mean(similarities)) if similarities else float("nan")


def _ew0_m1_nesting_summary(detail: pd.DataFrame) -> dict[str, Any]:
    n_runs = int(detail.shape[0])
    mean_delta = float(detail["m1_minus_m0"].mean()) if n_runs else float("nan")
    mean_paired_delta = float(detail["paired_m1_minus_m0"].mean()) if n_runs else float("nan")
    mean_second_ratio = (
        float(detail["m1_second_to_first_eigenvalue_ratio"].mean()) if n_runs else float("nan")
    )
    mean_second_fraction = (
        float(detail["m1_second_loading_fraction"].mean()) if n_runs else float("nan")
    )
    mean_stability = (
        float(detail["m1_second_dimension_bootstrap_stability"].mean())
        if n_runs
        else float("nan")
    )
    m1_win_rate = float(detail["m1_numerically_best"].mean()) if n_runs else float("nan")
    substantive_second_dim_rate = float(
        (
            (detail["m1_second_to_first_eigenvalue_ratio"] >= 0.25)
            & (detail["m1_second_dimension_bootstrap_stability"] >= 0.60)
        ).mean()
    ) if n_runs else float("nan")
    return {
        "n_runs": n_runs,
        "m1_numerically_best_rate": m1_win_rate,
        "mean_m1_minus_m0": mean_delta,
        "mean_paired_m1_minus_m0": mean_paired_delta,
        "mean_second_to_first_eigenvalue_ratio": mean_second_ratio,
        "mean_second_loading_fraction": mean_second_fraction,
        "mean_second_dimension_bootstrap_stability": mean_stability,
        "substantive_second_dimension_rate": substantive_second_dim_rate,
        "decision": _ew0_m1_nesting_decision(
            m1_win_rate=m1_win_rate,
            mean_delta=mean_delta,
            substantive_second_dim_rate=substantive_second_dim_rate,
        ),
    }


def _ew0_m1_nesting_decision(
    *,
    m1_win_rate: float,
    mean_delta: float,
    substantive_second_dim_rate: float,
) -> str:
    if m1_win_rate >= 0.8 and mean_delta > 0.05 and substantive_second_dim_rate >= 0.5:
        return "m1_recovers_substantive_second_dimension_under_ew0"
    if m1_win_rate >= 0.8:
        return "m1_numerically_exploits_ew0_without_stable_second_dimension"
    return "m0_m1_practically_ambiguous_under_ew0"


def _recovery_matrix(run_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for world_id in EXACT_MODEL_WORLD_IDS:
        world_rows = run_summary[run_summary["exact_world_id"] == world_id]
        n_runs = int(world_rows.shape[0])
        aligned_model_id = EXACT_WORLD_MODEL_ALIGNMENT[world_id]
        for model_id in STATIC_MODEL_IDS:
            n_selected = int((world_rows["selected_model_id"] == model_id).sum())
            rows.append(
                {
                    "exact_world_id": world_id,
                    "aligned_model_id": aligned_model_id,
                    "selected_model_id": model_id,
                    "n_runs": n_runs,
                    "n_selected": n_selected,
                    "selected_model_proportion": n_selected / n_runs if n_runs else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _correct_model_recovery(run_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for world_id in EXACT_MODEL_WORLD_IDS:
        world_rows = run_summary[run_summary["exact_world_id"] == world_id]
        n_runs = int(world_rows.shape[0])
        n_correct = int(world_rows["correct_model_selected"].sum())
        rate = n_correct / n_runs if n_runs else float("nan")
        ci_low, ci_high = binomial_wilson_interval(n_correct, n_runs)
        rows.append(
            {
                "exact_world_id": world_id,
                "aligned_model_id": EXACT_WORLD_MODEL_ALIGNMENT[world_id],
                "n_runs": n_runs,
                "n_aligned_model_selected": n_correct,
                "aligned_model_recovery_rate": rate,
                "ci_method": "wilson_95",
                "ci_low": ci_low,
                "ci_high": ci_high,
                "decision_band": recovery_decision_band(rate),
            }
        )
    return pd.DataFrame(rows)


def _selection_reason_counts(run_summary: pd.DataFrame) -> pd.DataFrame:
    if run_summary.empty:
        return pd.DataFrame(
            columns=["exact_world_id", "selection_reason", "selected_model_id", "n_runs"]
        )
    return (
        run_summary.groupby(
            ["exact_world_id", "selection_reason", "selected_model_id"],
            dropna=False,
            sort=True,
        )
        .size()
        .reset_index(name="n_runs")
    )


def _participant_difference_rows(
    *,
    task: ExactRecoveryTask,
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
        rows.append(
            {
                "run_id": task.run_id,
                "task_index": task.task_index,
                "exact_world_id": task.exact_world_id,
                "numerical_best_model_id": numerical_best_model_id,
                "comparison_model_id": model_id,
                "participant_weighted_delta_best_minus_model": mean,
                "paired_n_participants": n,
                "paired_delta_sd": sd,
                "paired_delta_se": se,
                "paired_delta_ci_low": mean - 1.96 * se if n > 1 else float("nan"),
                "paired_delta_ci_high": mean + 1.96 * se if n > 1 else float("nan"),
            }
        )
    return rows


def _participant_difference_rows_for_models(
    *,
    task: ExactRecoveryTask,
    participant_scores: pd.DataFrame,
    numerical_best_model_id: str,
    model_ids: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if numerical_best_model_id not in participant_scores:
        return rows
    best = participant_scores[numerical_best_model_id]
    for model_id in model_ids:
        if model_id == numerical_best_model_id or model_id not in participant_scores:
            continue
        diff = (best - participant_scores[model_id]).dropna()
        n = int(diff.shape[0])
        mean = float(diff.mean()) if n else float("nan")
        sd = float(diff.std(ddof=1)) if n > 1 else float("nan")
        se = sd / np.sqrt(n) if n > 1 else float("nan")
        rows.append(
            {
                "run_id": task.run_id,
                "task_index": task.task_index,
                "exact_world_id": task.exact_world_id,
                "numerical_best_model_id": numerical_best_model_id,
                "comparison_model_id": model_id,
                "participant_weighted_delta_best_minus_model": mean,
                "paired_n_participants": n,
                "paired_delta_sd": sd,
                "paired_delta_se": se,
                "paired_delta_ci_low": mean - 1.96 * se if n > 1 else float("nan"),
                "paired_delta_ci_high": mean + 1.96 * se if n > 1 else float("nan"),
            }
        )
    return rows


def _exact_standardised_features(
    exact_world_id: str,
    rng: np.random.Generator,
    *,
    exact_signal_scale: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    scale = float(exact_signal_scale)
    if exact_world_id == "EW0_m0_exact":
        z = rng.normal()
        loadings = scale * np.array([0.95, -0.88, 0.90, -0.62, 0.84])
        x = z * loadings + rng.normal(0.0, 0.32, size=5)
        return x, {"latent_1": z, "latent_2": 0.0, "readiness": z, "component": None}
    if exact_world_id == "EW1_m1_exact":
        z = rng.normal(size=2)
        loadings = scale * np.array(
            [
                [0.88, -0.22],
                [-0.78, 0.55],
                [0.82, -0.35],
                [-0.42, 0.72],
                [0.80, -0.48],
            ]
        )
        x = z @ loadings.T + rng.normal(0.0, 0.28, size=5)
        return x, {"latent_1": z[0], "latent_2": z[1], "readiness": z[0], "component": None}
    if exact_world_id == "EW2_m2_exact":
        z = rng.normal()
        basis = np.array([1.0, z, z**2 - 1.0])
        coefficients = scale * np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.18, -0.10, 0.15, -0.08, 0.12],
                [0.72, -0.68, 0.76, 0.50, 0.70],
            ]
        )
        x = basis @ coefficients + rng.normal(0.0, 0.18, size=5)
        return x, {"latent_1": z, "latent_2": 0.0, "readiness": z, "component": None}
    if exact_world_id in {"EW3_m3_exact", "EW4_m4_exact"}:
        if exact_world_id == "EW3_m3_exact":
            weights = np.array([0.36, 0.34, 0.30])
            means = scale * np.array(
                [
                    [1.35, -1.10, 1.20, -0.85, 1.15],
                    [-1.15, 1.20, -1.05, 0.95, -1.10],
                    [0.10, 0.35, 0.05, -0.15, 0.20],
                ]
            )
        else:
            weights = np.array([0.28, 0.27, 0.24, 0.21])
            means = scale * np.array(
                [
                    [1.35, -1.10, 1.18, -0.88, 1.15],
                    [-1.20, 1.25, -1.10, 0.92, -1.05],
                    [0.45, 0.55, 0.35, -0.35, 0.30],
                    [-0.55, -0.35, -0.45, 0.55, -0.55],
                ]
            )
        component = int(rng.choice(len(weights), p=weights))
        x = means[component] + rng.normal(0.0, 0.20, size=5)
        return x, {
            "latent_1": x[0],
            "latent_2": x[1],
            "readiness": x[2],
            "component": component,
        }
    raise ValueError(f"unsupported exact world: {exact_world_id}")


def _canonical_minimal_row(
    *,
    exact_world_id: str,
    checksum: str,
    source: str,
    participant_id: str,
    session_number: int,
    window_number: int,
    task_id: str,
    standardised: np.ndarray,
    truth: dict[str, Any],
) -> dict[str, Any]:
    base = np.array([0.78, 710.0, 1.42, 0.24, 1.10])
    scale = np.array([0.055, 55.0, 0.12, 0.045, 0.16])
    accuracy, median_rt, response_speed, rt_cv, throughput = base + standardised * scale
    total_trials = 36
    start_trial = 1 + (window_number - 1) * 32
    return {
        "source_dataset": source,
        "source_version": f"{exact_world_id}-v1",
        "participant_id": participant_id,
        "session_id": f"s{session_number:02d}",
        "task_id": task_id,
        "block_id": f"b{1 + (window_number - 1) // 2:02d}",
        "window_id": f"w{window_number:02d}",
        "window_start_trial": start_trial,
        "window_end_trial": start_trial + total_trials - 1,
        "n_trials_total": total_trials,
        "n_trials_valid": total_trials,
        "source_file_or_table": "synthetic.exact_model_diagnostic",
        "source_commit_or_release": "m2.6b-exact-model-diagnostic-v1",
        "source_hash_if_available": checksum,
        "preprocessing_version": "synthetic-exact-model-v1",
        "feature_version": "canonical-window-v1",
        "accuracy": float(accuracy),
        "median_rt_ms": float(median_rt),
        "mean_response_speed": float(response_speed),
        "rt_cv": float(rt_cv),
        "throughput_proxy": float(throughput),
        "trial_count": total_trials,
        "practice_or_session_index": session_number,
        "condition_mix": "exact_model",
        "synthetic_exact_world_id": exact_world_id,
        "synthetic_aligned_model_id": EXACT_WORLD_MODEL_ALIGNMENT[exact_world_id],
        "synthetic_latent_1": float(truth["latent_1"]),
        "synthetic_latent_2": float(truth["latent_2"]),
        "synthetic_readiness": float(truth["readiness"]),
        "synthetic_component_id": (
            f"component_{truth['component']}"
            if truth["component"] is not None
            else "not_applicable"
        ),
    }


def _canonical_raw_row(
    *,
    exact_world_id: str,
    checksum: str,
    source: str,
    participant_id: str,
    session_number: int,
    window_number: int,
    task_id: str,
    raw_features: np.ndarray,
    truth: dict[str, Any],
) -> dict[str, Any]:
    total_trials = 36
    start_trial = 1 + (window_number - 1) * 32
    accuracy, median_rt, response_speed, rt_cv, throughput = raw_features
    return {
        "source_dataset": source,
        "source_version": f"{exact_world_id}-v1",
        "participant_id": participant_id,
        "session_id": f"s{session_number:02d}",
        "task_id": task_id,
        "block_id": f"b{1 + (window_number - 1) // 2:02d}",
        "window_id": f"w{window_number:02d}",
        "window_start_trial": start_trial,
        "window_end_trial": start_trial + total_trials - 1,
        "n_trials_total": total_trials,
        "n_trials_valid": total_trials,
        "source_file_or_table": "synthetic.exact_model_diagnostic",
        "source_commit_or_release": "m2-self-check-v1",
        "source_hash_if_available": checksum,
        "preprocessing_version": "synthetic-m2-self-check-v1",
        "feature_version": "canonical-window-v1",
        "accuracy": float(accuracy),
        "median_rt_ms": float(median_rt),
        "mean_response_speed": float(response_speed),
        "rt_cv": float(rt_cv),
        "throughput_proxy": float(throughput),
        "trial_count": total_trials,
        "practice_or_session_index": session_number,
        "condition_mix": "m2_self_check",
        "synthetic_exact_world_id": exact_world_id,
        "synthetic_aligned_model_id": "M2_nonlinear_vigilance",
        "synthetic_latent_1": float(truth["latent_1"]),
        "synthetic_latent_2": float(truth["latent_2"]),
        "synthetic_readiness": float(truth["readiness"]),
        "synthetic_component_id": "not_applicable",
    }


def _oracle_m2_scores(frame: pd.DataFrame, oracle: dict[str, Any], *, adapter: Any) -> tuple[np.ndarray, np.ndarray]:
    if adapter is None or adapter.means_ is None or adapter.scales_ is None:
        raise ValueError("a fitted TrainingFeatureAdapter is required for oracle M2 scoring")
    values = adapter.observed_standardised(frame).to_numpy(dtype=float)
    observed_mask = np.isfinite(values)
    feature_means = adapter.means_.loc[list(CORE_SYNTHETIC_FEATURES)].to_numpy(dtype=float)
    feature_scales = adapter.scales_.loc[list(CORE_SYNTHETIC_FEATURES)].to_numpy(dtype=float)
    raw_to_adapter_scale = oracle["raw_scale"] / feature_scales
    coefficients = np.array(oracle["coefficients"], dtype=float).copy()
    coefficients[0] = (
        (oracle["raw_base"] - feature_means) / feature_scales
        + raw_to_adapter_scale * coefficients[0]
    )
    coefficients[1] = raw_to_adapter_scale * coefficients[1]
    coefficients[2] = raw_to_adapter_scale * coefficients[2]
    residual_variance = np.maximum(
        (raw_to_adapter_scale**2) * oracle["residual_variance"],
        1e-10,
    )
    linear_norm = float(np.linalg.norm(coefficients[1]))
    linear_direction = coefficients[1] / linear_norm if linear_norm else coefficients[1]
    nodes, weights = np.polynomial.hermite.hermgauss(201)
    nodes = np.sqrt(2.0) * nodes
    log_weights = np.log(weights / np.sqrt(np.pi))
    basis = np.column_stack([np.ones_like(nodes), nodes, nodes**2 - 1.0])
    means_by_node = basis @ coefficients
    scores = np.full(values.shape[0], np.nan, dtype=float)
    for row_index, row in enumerate(values):
        mask = observed_mask[row_index]
        if not bool(mask.any()):
            continue
        residual = row[mask][None, :] - means_by_node[:, mask]
        node_logs = -0.5 * np.sum(
            np.log(2.0 * np.pi)
            + np.log(residual_variance[mask])[None, :]
            + (residual**2) / residual_variance[mask][None, :],
            axis=1,
        )
        scores[row_index] = _logsumexp(log_weights + node_logs)
    return scores, linear_direction


def _fit_and_score_m2_fixed_projection(
    train: pd.DataFrame,
    test: pd.DataFrame,
    projection: np.ndarray,
    *,
    quadrature_points: int,
) -> pd.Series:
    adapter = M2NonlinearVigilanceModel(
        feature_columns=CORE_SYNTHETIC_FEATURES,
        random_state=0,
        quadrature_points=quadrature_points,
    ).fit(train).adapter_
    if adapter is None:
        raise ValueError("M2 adapter was not fitted")
    train_values = adapter.transform(train).values
    projection = np.asarray(projection, dtype=float)
    projection = projection / float(np.linalg.norm(projection))
    coefficients, residual_variance = _fit_quadratic_given_projection_for_diagnostic(
        train_values,
        projection,
    )
    test_values = adapter.observed_standardised(test).to_numpy(dtype=float)
    observed_mask = adapter.observed_standardised(test).notna().to_numpy(dtype=bool)
    nodes, weights = np.polynomial.hermite.hermgauss(quadrature_points)
    nodes = np.sqrt(2.0) * nodes
    log_weights = np.log(weights / np.sqrt(np.pi))
    scores = _diagnostic_m2_log_density_by_row(
        test_values,
        observed_mask,
        coefficients,
        residual_variance,
        nodes,
        log_weights,
    )
    return pd.Series(scores, index=test.index)


def _fit_quadratic_given_projection_for_diagnostic(
    x: np.ndarray,
    projection: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    z = x @ projection
    z_scale = float(np.std(z, ddof=0)) or 1.0
    z = (z - float(np.mean(z))) / z_scale
    design = np.column_stack([np.ones_like(z), z, z**2 - 1.0])
    coefficients = np.linalg.pinv(design) @ x
    residuals = x - design @ coefficients
    residual_variance = np.maximum(np.var(residuals, axis=0), 1e-8)
    return coefficients, residual_variance


def _diagnostic_m2_log_density_by_row(
    values: np.ndarray,
    observed_mask: np.ndarray,
    coefficients: np.ndarray,
    residual_variance: np.ndarray,
    nodes: np.ndarray,
    log_weights: np.ndarray,
) -> np.ndarray:
    basis = np.column_stack([np.ones_like(nodes), nodes, nodes**2 - 1.0])
    means_by_node = basis @ coefficients
    output = np.full((values.shape[0], nodes.shape[0]), np.nan, dtype=float)
    for mask in np.unique(observed_mask, axis=0):
        row_selector = np.all(observed_mask == mask, axis=1)
        if not bool(mask.any()):
            continue
        subset = values[row_selector][:, mask]
        subset_means = means_by_node[:, mask]
        subset_variance = np.maximum(residual_variance[mask], 1e-10)
        residual = subset[:, None, :] - subset_means[None, :, :]
        node_logs = -0.5 * np.sum(
            np.log(2.0 * np.pi)
            + np.log(subset_variance)[None, None, :]
            + (residual**2) / subset_variance[None, None, :],
            axis=2,
        )
        output[row_selector] = log_weights[None, :] + node_logs
    return _logsumexp_axis1_for_diagnostic(output)


def _logsumexp_axis1_for_diagnostic(values: np.ndarray) -> np.ndarray:
    maximum = np.max(values, axis=1)
    finite = np.isfinite(maximum)
    output = np.full(values.shape[0], np.nan, dtype=float)
    if finite.any():
        stable = values[finite] - maximum[finite, None]
        output[finite] = maximum[finite] + np.log(np.sum(np.exp(stable), axis=1))
    output[~finite] = maximum[~finite]
    return output


def _m2_self_check_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "mean_oracle_m2_log_density": float(frame["oracle_m2_mean_log_density"].mean()),
        "mean_fitted_m2_log_density": float(frame["fitted_m2_mean_log_density"].mean()),
        "mean_fitted_m2_high_quadrature_log_density": float(
            frame["fitted_m2_high_quadrature_mean_log_density"].mean()
        ),
        "mean_fitted_m2_projection_search_log_density": float(
            frame["fitted_m2_projection_search_mean_log_density"].mean()
        ),
        "mean_fitted_m2_em_log_density": float(
            frame["fitted_m2_em_mean_log_density"].mean()
        ),
        "mean_fitted_m2_oracle_projection_log_density": float(
            frame["fitted_m2_oracle_projection_mean_log_density"].mean()
        ),
        "mean_fitted_m1_log_density": float(frame["fitted_m1_mean_log_density"].mean()),
        "mean_oracle_minus_m1": float(frame["oracle_minus_m1"].mean()),
        "mean_fitted_m2_minus_m1": float(frame["fitted_m2_minus_m1"].mean()),
        "mean_fitted_m2_high_quadrature_minus_m1": float(
            frame["fitted_m2_high_quadrature_minus_m1"].mean()
        ),
        "mean_fitted_m2_projection_search_minus_m1": float(
            frame["fitted_m2_projection_search_minus_m1"].mean()
        ),
        "mean_fitted_m2_em_minus_m1": float(frame["fitted_m2_em_minus_m1"].mean()),
        "mean_fitted_m2_oracle_projection_minus_m1": float(
            frame["fitted_m2_oracle_projection_minus_m1"].mean()
        ),
        "mean_oracle_minus_fitted_m2": float(frame["oracle_minus_fitted_m2"].mean()),
        "mean_oracle_minus_fitted_m2_high_quadrature": float(
            frame["oracle_minus_fitted_m2_high_quadrature"].mean()
        ),
        "mean_oracle_minus_fitted_m2_projection_search": float(
            frame["oracle_minus_fitted_m2_projection_search"].mean()
        ),
        "mean_oracle_minus_fitted_m2_em": float(
            frame["oracle_minus_fitted_m2_em"].mean()
        ),
        "mean_oracle_minus_fitted_m2_oracle_projection": float(
            frame["oracle_minus_fitted_m2_oracle_projection"].mean()
        ),
        "m2_beats_m1_rate": float((frame["fitted_m2_minus_m1"] > 0.0).mean()),
        "m2_high_quadrature_beats_m1_rate": float(
            (frame["fitted_m2_high_quadrature_minus_m1"] > 0.0).mean()
        ),
        "m2_projection_search_beats_m1_rate": float(
            (frame["fitted_m2_projection_search_minus_m1"] > 0.0).mean()
        ),
        "m2_em_beats_m1_rate": float((frame["fitted_m2_em_minus_m1"] > 0.0).mean()),
        "m2_oracle_projection_beats_m1_rate": float(
            (frame["fitted_m2_oracle_projection_minus_m1"] > 0.0).mean()
        ),
        "oracle_beats_m1_rate": float((frame["oracle_minus_m1"] > 0.0).mean()),
        "mean_m2_projection_alignment_abs": float(frame["m2_projection_alignment_abs"].mean()),
        "mean_m2_projection_search_alignment_abs": float(
            frame["m2_projection_search_alignment_abs"].mean()
        ),
        "mean_projection_search_candidates_evaluated": float(
            frame["m2_projection_search_candidates_evaluated"].mean()
        ),
        "mean_m2_em_n_iter": float(frame["m2_em_n_iter"].mean()),
        "interpretation": _m2_self_check_interpretation(frame),
    }


def _m2_self_check_interpretation(frame: pd.DataFrame) -> str:
    oracle_beats_m1 = float((frame["oracle_minus_m1"] > 0.0).mean())
    fitted_beats_m1 = float((frame["fitted_m2_minus_m1"] > 0.0).mean())
    high_quadrature_beats_m1 = float(
        (frame["fitted_m2_high_quadrature_minus_m1"] > 0.0).mean()
    )
    projection_search_beats_m1 = float(
        (frame["fitted_m2_projection_search_minus_m1"] > 0.0).mean()
    )
    em_beats_m1 = float((frame["fitted_m2_em_minus_m1"] > 0.0).mean())
    oracle_projection_beats_m1 = float(
        (frame["fitted_m2_oracle_projection_minus_m1"] > 0.0).mean()
    )
    if oracle_beats_m1 >= 0.8 and fitted_beats_m1 < 0.5:
        if em_beats_m1 >= 0.8:
            return "latent_curve_em_m2_recovers_self_check_data"
        if oracle_projection_beats_m1 >= 0.8 and projection_search_beats_m1 < 0.5:
            return "m2_projection_estimation_failure"
        if projection_search_beats_m1 >= 0.8:
            return "projection_search_m2_recovers_self_check_data"
        if high_quadrature_beats_m1 >= 0.8:
            return "fitted_m2_needs_higher_quadrature"
        return "oracle_m2_wins_but_fitted_m2_loses"
    if oracle_beats_m1 < 0.5:
        return "m2_generator_not_distinct_from_m1_under_oracle_scoring"
    if fitted_beats_m1 >= 0.8:
        return "fitted_m2_recovers_self_check_data"
    return "ambiguous_m2_self_check"


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + float(np.log(np.sum(np.exp(values - maximum))))


def _summary_payload(correct_model_recovery: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "exact_world_id": str(row["exact_world_id"]),
            "aligned_model_id": str(row["aligned_model_id"]),
            "recovery_rate": round(float(row["aligned_model_recovery_rate"]), 3),
            "n_runs": int(row["n_runs"]),
            "decision_band": str(row["decision_band"]),
        }
        for _, row in correct_model_recovery.iterrows()
    ]


def _validate_exact_config(config: dict[str, Any]) -> None:
    if config.get("study", {}).get("id") != EXACT_MODEL_STUDY_ID:
        raise ValueError(f"study.id must be {EXACT_MODEL_STUDY_ID}")
    worlds = tuple(config.get("exact_worlds", ()))
    if worlds != EXACT_MODEL_WORLD_IDS:
        raise ValueError("exact_worlds must match the frozen EW0-EW4 order")
    if tuple(config.get("models", ())) != STATIC_MODEL_IDS:
        raise ValueError("models must match STATIC_MODEL_IDS")


def _resolve_workers(workers: int | None) -> int:
    if workers is None:
        return 1
    return max(1, int(workers))


def _run_id(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _child_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8], 16)


def _parse_int_grid(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("iteration grid must contain at least one integer")
    if any(value < 1 for value in values):
        raise ValueError("iteration grid values must be positive")
    return values


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run M2.6b exact-model recovery diagnostic.")
    parser.add_argument("--config", default=str(EXACT_CONFIG_PATH))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--executor", choices=("serial", "thread", "process"), default="thread")
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--task-count", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--m2-self-check", action="store_true")
    parser.add_argument("--m2-self-check-replicates", type=int, default=12)
    parser.add_argument("--ew2-em-diagnostic", action="store_true")
    parser.add_argument("--ew2-em-replicates", type=int, default=10)
    parser.add_argument("--ew2-em-quadrature-points", type=int, default=201)
    parser.add_argument("--ew2-em-max-iter", type=int, default=80)
    parser.add_argument("--m2-em-convergence-diagnostic", action="store_true")
    parser.add_argument("--m2-em-convergence-replicates", type=int, default=12)
    parser.add_argument("--m2-em-convergence-max-iters", default="80,120,200")
    parser.add_argument("--m2-em-convergence-quadrature-points", type=int, default=201)
    parser.add_argument("--ew0-m1-nesting-diagnostic", action="store_true")
    parser.add_argument("--ew0-m1-nesting-replicates", type=int, default=12)
    parser.add_argument("--ew0-m1-bootstrap-samples", type=int, default=16)
    args = parser.parse_args(argv)

    config = load_exact_config(args.config)
    if args.m2_self_check:
        result = run_m2_self_check(
            n_replicates=args.m2_self_check_replicates,
            output_dir=args.output_dir or DEFAULT_OUTPUT_DIR / "m2_self_check",
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.ew2_em_diagnostic:
        result = run_ew2_em_diagnostic(
            config=config,
            output_dir=args.output_dir or DEFAULT_EW2_EM_OUTPUT_DIR,
            n_replicates=args.ew2_em_replicates,
            workers=args.workers,
            executor=args.executor,
            em_quadrature_points=args.ew2_em_quadrature_points,
            em_max_iter=args.ew2_em_max_iter,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.m2_em_convergence_diagnostic:
        result = run_m2_em_convergence_diagnostic(
            config=config,
            output_dir=args.output_dir or DEFAULT_M2_EM_CONVERGENCE_OUTPUT_DIR,
            n_replicates=args.m2_em_convergence_replicates,
            workers=args.workers,
            executor=args.executor,
            em_quadrature_points=args.m2_em_convergence_quadrature_points,
            max_iters=_parse_int_grid(args.m2_em_convergence_max_iters),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.ew0_m1_nesting_diagnostic:
        result = run_ew0_m1_nesting_diagnostic(
            config=config,
            output_dir=args.output_dir or DEFAULT_EW0_M1_NESTING_OUTPUT_DIR,
            n_replicates=args.ew0_m1_nesting_replicates,
            bootstrap_samples=args.ew0_m1_bootstrap_samples,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.dry_run:
        print(json.dumps(preflight_exact_model_recovery(config, workers=args.workers), indent=2))
        return 0
    result = run_exact_model_recovery(
        config_path=args.config,
        output_dir=args.output_dir,
        workers=args.workers,
        executor=args.executor,
        task_start=args.task_start,
        task_count=args.task_count,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
