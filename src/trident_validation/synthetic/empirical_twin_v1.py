"""M2.7 empirical-twin V1 generator smoke.

The generator preserves known W0-W4 structural truth and adds frozen empirical
background nuisance structure. It is a smoke/pipeline validation tool, not a
scientific recovery estimator.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from trident_validation.config import load_yaml_config
from trident_validation.models.static_tournament_v2 import (
    STATIC_TOURNAMENT_V2_CONTRACT,
    build_static_model_suite_v2,
)
from trident_validation.provenance import get_git_commit, hash_file, hash_mapping
from trident_validation.schema import STRUCTURAL_FEATURES_BY_FLAG, validate_window_schema
from trident_validation.splits import participant_train_test_split
from trident_validation.synthetic.fixtures import CORE_SYNTHETIC_FEATURES
from trident_validation.synthetic.recovery import (
    PRIMARY_METRIC,
    assert_no_ground_truth_columns,
    strip_ground_truth_columns,
)
from trident_validation.synthetic.selection_v2 import select_preferred_model_v2
from trident_validation.synthetic.worlds import (
    STATIC_SYNTHETIC_WORLD_IDS,
    WORLD_MODEL_ALIGNMENT,
    StaticWorldId,
    make_static_synthetic_world,
)


EMPIRICAL_TWIN_V1_ID = "empirical_twin_v1"
EMPIRICAL_BACKGROUND_CONTRACT_ID = "EMPIRICAL_BACKGROUND_CONTRACT_V1"
DEFAULT_CONFIG_PATH = Path("config/empirical_twin_v1.yaml")
DEFAULT_OUTPUT_DIR = Path("reports/generated/empirical_twin_v1_smoke")
FEATURES = tuple(CORE_SYNTHETIC_FEATURES)
PAIRED_FEATURE_ALIASES = {"median_rt_ms": "mean_rt_ms"}


@dataclass(frozen=True)
class BackgroundSpec:
    """Aggregate empirical-background inputs for generation."""

    acdc_dir: Path
    paired_dir: Path
    template_summary: pd.DataFrame
    feature_summary: pd.DataFrame
    acdc_variance: pd.DataFrame
    acdc_cov_between: pd.DataFrame
    acdc_cov_window: pd.DataFrame
    acdc_temporal: pd.DataFrame
    acdc_missingness: pd.DataFrame
    paired_variance: pd.DataFrame
    paired_cov_session: pd.DataFrame
    paired_stability: pd.DataFrame
    paired_cross_task_within: pd.DataFrame | None


@dataclass(frozen=True)
class EmpiricalTwinRun:
    """One generated known-truth empirical-twin dataset and audits."""

    run_id: str
    world_id: str
    replicate_index: int
    dataset: pd.DataFrame
    generator_audit: pd.DataFrame
    support_audit: pd.DataFrame
    generation_summary: dict[str, Any]


def run_empirical_twin_v1_smoke(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path | None = None,
    run_tournament_smoke: bool | None = None,
) -> dict[str, Any]:
    """Generate small empirical twins and optionally run a V2 tournament smoke."""

    config_path = Path(config_path)
    config = load_yaml_config(config_path)
    _validate_config(config)
    target_dir = Path(output_dir or config["outputs"]["directory"])
    target_dir.mkdir(parents=True, exist_ok=True)

    background = load_background_spec(
        Path(config["inputs"]["acdc_preflight_dir"]),
        Path(config["inputs"]["paired_preflight_dir"]),
    )
    generator_config = dict(config["generator"])
    worlds = tuple(str(world) for world in generator_config["worlds"])
    replicates = int(generator_config["replicates_per_world"])
    runs: list[EmpiricalTwinRun] = []
    for world_id in worlds:
        if world_id not in STATIC_SYNTHETIC_WORLD_IDS:
            raise ValueError(f"unsupported empirical-twin world: {world_id}")
        for replicate_index in range(replicates):
            print(
                f"M2.7 empirical-twin generation smoke | {world_id} | replicate {replicate_index}",
                flush=True,
            )
            runs.append(
                generate_empirical_twin_dataset(
                    world_id=world_id,  # type: ignore[arg-type]
                    replicate_index=replicate_index,
                    background=background,
                    config=generator_config,
                )
            )

    tournament_enabled = (
        bool(config["tournament_smoke"]["enabled"])
        if run_tournament_smoke is None
        else bool(run_tournament_smoke)
    )
    tournament_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    if tournament_enabled:
        for run in runs:
            print(
                f"M2.7 static V2 tournament smoke | {run.world_id} | replicate {run.replicate_index}",
                flush=True,
            )
            smoke = run_static_tournament_v2_smoke(
                run,
                test_fraction=float(config["tournament_smoke"]["test_fraction"]),
                practical_equivalence_margin=float(
                    config["tournament_smoke"]["practical_equivalence_margin"]
                ),
                paired_ci_z=float(config["tournament_smoke"]["paired_ci_z"]),
            )
            tournament_rows.extend(smoke["model_rows"])
            split_rows.append(smoke["split_row"])

    paths = write_empirical_twin_v1_outputs(
        runs,
        output_dir=target_dir,
        config=config,
        config_path=config_path,
        tournament_rows=tournament_rows,
        split_rows=split_rows,
    )
    return {
        "study_id": str(config["study"]["id"]),
        "empirical_background_contract": EMPIRICAL_BACKGROUND_CONTRACT_ID,
        "empirical_background_contract_commit": str(
            config["contract"]["empirical_background_contract_commit"]
        ),
        "static_tournament_contract": STATIC_TOURNAMENT_V2_CONTRACT.contract_id,
        "n_generated_runs": len(runs),
        "n_generated_rows": int(sum(run.dataset.shape[0] for run in runs)),
        "tournament_smoke_enabled": tournament_enabled,
        "formal_claims_allowed": False,
        "paths": {key: str(value) for key, value in paths.items()},
    }


def load_background_spec(acdc_dir: str | Path, paired_dir: str | Path) -> BackgroundSpec:
    """Load aggregate-only empirical-background preflight outputs."""

    acdc = Path(acdc_dir)
    paired = Path(paired_dir)
    return BackgroundSpec(
        acdc_dir=acdc,
        paired_dir=paired,
        template_summary=_read_csv(acdc / "template_summary.csv"),
        feature_summary=_read_csv(acdc / "feature_summary.csv"),
        acdc_variance=_read_csv(acdc / "variance_decomposition.csv"),
        acdc_cov_between=_read_csv(acdc / "covariance_between_person.csv"),
        acdc_cov_window=_read_csv(acdc / "covariance_within_session.csv"),
        acdc_temporal=_read_csv(acdc / "temporal_summary.csv"),
        acdc_missingness=_read_csv(acdc / "missingness_summary.csv"),
        paired_variance=_read_csv(paired / "paired_session_variance_decomposition.csv"),
        paired_cov_session=_read_csv(paired / "paired_session_covariance_session.csv"),
        paired_stability=_read_csv(
            paired / "paired_session_repeated_person_stability.csv",
            required=False,
        ),
        paired_cross_task_within=_read_csv(
            paired / "paired_session_cross_task_covariance_within_person.csv",
            required=False,
        ),
    )


def generate_empirical_twin_dataset(
    *,
    world_id: StaticWorldId,
    replicate_index: int,
    background: BackgroundSpec,
    config: dict[str, Any],
) -> EmpiricalTwinRun:
    """Generate one known-truth empirical twin with nuisance background."""

    seed = _child_seed(config["seed"], world_id, replicate_index, "dataset")
    rng = np.random.default_rng(seed)
    templates = _select_templates(background.template_summary, int(config["n_templates"]))
    base = make_static_synthetic_world(
        world_id,
        seed=seed,
        n_datasets=int(config["n_templates"]),
        participants_per_dataset=int(config["participants_per_template"]),
        sessions_per_participant=int(config["sessions_per_participant"]),
        min_windows_per_session=int(config["windows_per_session"]),
        max_windows_per_session=int(config["windows_per_session"]),
        technical_missingness_rate=0.0,
    )
    base_sources = sorted(base["source_dataset"].astype(str).unique())
    template_by_source = {
        source: templates.iloc[index % templates.shape[0]]
        for index, source in enumerate(base_sources)
    }

    for source, template in template_by_source.items():
        mask = base["source_dataset"].astype(str) == source
        base.loc[mask, "source_dataset"] = str(template["source_dataset"])
        base.loc[mask, "task_id"] = str(template["task_id"])
        base.loc[mask, "source_version"] = "empirical_twin_v1_smoke"
        base.loc[mask, "source_file_or_table"] = "synthetic.empirical_twin_v1"
        base.loc[mask, "source_commit_or_release"] = EMPIRICAL_BACKGROUND_CONTRACT_ID
        base.loc[mask, "preprocessing_version"] = "empirical-twin-v1-generator"
        base.loc[mask, "feature_version"] = "canonical-window-v1"
    _reset_task_context_columns(base)

    generated = base.copy()
    support_rows: list[dict[str, Any]] = []
    structural_fraction = float(config.get("structural_signal_fraction", 0.35))
    use_covariance = bool(config.get("use_covariance", True))
    session_trend = str(config.get("session_order_context_trend", "off")).lower() == "on"
    if session_trend:
        support_rows.append(
            _support_row(
                world_id,
                replicate_index,
                "session_order_context_trend",
                "sensitivity_enabled",
                "not_primary_generator_mode",
            )
        )

    for source, source_frame in generated.groupby("source_dataset", sort=False):
        template = _template_for_generated_source(templates, source)
        task = str(template["task_id"])
        source_index = source_frame.index
        means = _template_means(template)
        between_cov, between_mode = _covariance_matrix(
            background.acdc_cov_between,
            source=str(source),
            task=task,
            features=FEATURES,
            use_covariance=use_covariance,
        )
        window_cov, window_mode = _covariance_matrix(
            background.acdc_cov_window,
            source=str(source),
            task=task,
            features=FEATURES,
            use_covariance=use_covariance,
        )
        session_cov, session_mode = _paired_session_covariance(
            background,
            task=task,
            features=FEATURES,
            use_covariance=use_covariance,
        )
        lag1 = _feature_lookup(
            background.acdc_temporal,
            source=str(source),
            task=task,
            column="lag1_mean_autocorrelation",
            default=0.0,
            status_column="lag1_support_status",
        )
        fatigue = _feature_lookup(
            background.acdc_temporal,
            source=str(source),
            task=task,
            column="fatigue_mean_slope",
            default=0.0,
            status_column="fatigue_support_status",
        )
        missingness = _feature_lookup(
            background.acdc_missingness,
            source=str(source),
            task=task,
            column="missing_rate",
            default=0.0,
        )
        trial_counts = _trial_count_targets(template)

        support_rows.extend(
            [
                _support_row(world_id, replicate_index, "between_covariance", between_mode, str(source)),
                _support_row(world_id, replicate_index, "window_covariance", window_mode, str(source)),
                _support_row(world_id, replicate_index, "session_covariance", session_mode, task),
            ]
        )
        _apply_background_to_source(
            generated,
            source_index=source_index,
            means=means,
            between_cov=between_cov,
            session_cov=session_cov,
            window_cov=window_cov,
            lag1=lag1,
            fatigue=fatigue if session_trend is False else fatigue,
            missingness=missingness,
            trial_counts=trial_counts,
            structural_signal_fraction=structural_fraction,
            rng=rng,
        )

    generated = _finalise_generated_bounds(generated)
    validate_window_schema(strip_ground_truth_columns(generated))
    run_id = _run_id(EMPIRICAL_TWIN_V1_ID, world_id, replicate_index, seed)
    audit = build_generator_audit(
        generated,
        background=background,
        templates=templates,
        world_id=world_id,
        replicate_index=replicate_index,
    )
    summary = {
        "run_id": run_id,
        "world_id": world_id,
        "aligned_model_id": WORLD_MODEL_ALIGNMENT[world_id],
        "replicate_index": int(replicate_index),
        "dataset_seed": int(seed),
        "n_rows": int(generated.shape[0]),
        "n_participants": int(
            generated.loc[:, ["source_dataset", "participant_id"]].drop_duplicates().shape[0]
        ),
        "n_sessions": int(
            generated.loc[:, ["source_dataset", "participant_id", "session_id"]]
            .drop_duplicates()
            .shape[0]
        ),
        "n_sources": int(generated["source_dataset"].nunique()),
        "n_tasks": int(generated["task_id"].nunique()),
        "session_order_context_trend": "off",
        "formal_claims_allowed": False,
        "model_recovery_claim_allowed": False,
    }
    return EmpiricalTwinRun(
        run_id=run_id,
        world_id=world_id,
        replicate_index=replicate_index,
        dataset=generated,
        generator_audit=audit,
        support_audit=pd.DataFrame(support_rows),
        generation_summary=summary,
    )


def run_static_tournament_v2_smoke(
    run: EmpiricalTwinRun,
    *,
    test_fraction: float,
    practical_equivalence_margin: float,
    paired_ci_z: float,
) -> dict[str, Any]:
    """Run the static V2 scoring chain on one tiny generated dataset."""

    model_frame = strip_ground_truth_columns(run.dataset)
    assert_no_ground_truth_columns(model_frame)
    split_seed = _child_seed(run.run_id, "split")
    model_seed = _child_seed(run.run_id, "model")
    split = participant_train_test_split(
        model_frame,
        test_size=test_fraction,
        seed=split_seed,
    )
    frozen = _fit_score_models_without_truth_v2(
        model_frame,
        split_indices=(tuple(split.train_indices), tuple(split.test_indices)),
        random_state=model_seed,
    )
    selection = select_preferred_model_v2(
        frozen["model_scores"],
        frozen["participant_scores"],
        practical_equivalence_margin=practical_equivalence_margin,
        paired_ci_z=paired_ci_z,
    )
    uncertainty = selection.uncertainty_by_model.set_index("model_id", drop=False)
    model_rows: list[dict[str, Any]] = []
    for _, row in frozen["model_scores"].iterrows():
        model_id = str(row["model_id"])
        model_row = {
            "run_id": run.run_id,
            "world_id": run.world_id,
            "replicate_index": run.replicate_index,
            "aligned_model_id": WORLD_MODEL_ALIGNMENT[run.world_id],
            "model_id": model_id,
            "selected_model_id": selection.selected_model_id,
            "numerical_best_model_id": selection.numerical_best_model_id,
            "selection_reason": selection.selection_reason,
            "same_tier_ambiguous": selection.same_tier_ambiguous,
            "ambiguous_model_ids": ";".join(selection.ambiguous_model_ids),
            "pipeline_complete": True,
            "truth_leakage": "none_detected",
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
        model_rows.append(model_row)
    split_row = {
        "run_id": run.run_id,
        "world_id": run.world_id,
        "replicate_index": run.replicate_index,
        "n_train_rows": len(split.train_indices),
        "n_test_rows": len(split.test_indices),
        "n_train_participants": _n_participants(model_frame.loc[list(split.train_indices)]),
        "n_test_participants": _n_participants(model_frame.loc[list(split.test_indices)]),
        "participant_isolated": True,
        "checkpoint_resume_path_validated": "not_applicable_smoke_in_memory",
    }
    return {"model_rows": model_rows, "split_row": split_row}


def build_generator_audit(
    generated: pd.DataFrame,
    *,
    background: BackgroundSpec,
    templates: pd.DataFrame,
    world_id: str,
    replicate_index: int,
) -> pd.DataFrame:
    """Compare target and realised background quantities for smoke auditing."""

    rows: list[dict[str, Any]] = []
    for _, template in templates.iterrows():
        source = str(template["source_dataset"])
        task = str(template["task_id"])
        frame = generated[
            (generated["source_dataset"].astype(str) == source)
            & (generated["task_id"].astype(str) == task)
        ].copy()
        if frame.empty:
            continue
        for feature in FEATURES:
            rows.append(
                _audit_row(
                    world_id,
                    replicate_index,
                    source,
                    task,
                    "source_task_mean",
                    feature,
                    _as_float(template.get(f"{feature}_mean")),
                    _as_float(frame[feature].mean(skipna=True)),
                )
            )
            rows.append(
                _audit_row(
                    world_id,
                    replicate_index,
                    source,
                    task,
                    "missingness_rate",
                    feature,
                    _target_lookup(
                        background.acdc_missingness,
                        source=source,
                        task=task,
                        feature=feature,
                        column="missing_rate",
                    ),
                    float(frame[feature].isna().mean()),
                )
            )
            rows.append(
                _audit_row(
                    world_id,
                    replicate_index,
                    source,
                    task,
                    "between_person_variance",
                    feature,
                    _target_lookup(
                        background.acdc_variance,
                        source=source,
                        task=task,
                        feature=feature,
                        column="between_participant_variance",
                    ),
                    _realised_between_variance(frame, feature),
                )
            )
            rows.append(
                _audit_row(
                    world_id,
                    replicate_index,
                    source,
                    task,
                    "session_within_person_variance",
                    feature,
                    _paired_target_variance(background.paired_variance, task, feature),
                    _realised_session_variance(frame, feature),
                )
            )
            rows.append(
                _audit_row(
                    world_id,
                    replicate_index,
                    source,
                    task,
                    "window_within_session_variance",
                    feature,
                    _target_lookup(
                        background.acdc_variance,
                        source=source,
                        task=task,
                        feature=feature,
                        column="window_within_session_variance",
                    ),
                    _realised_window_variance(frame, feature),
                )
            )
            rows.append(
                _audit_row(
                    world_id,
                    replicate_index,
                    source,
                    task,
                    "lag1_autocorrelation",
                    feature,
                    _target_lookup(
                        background.acdc_temporal,
                        source=source,
                        task=task,
                        feature=feature,
                        column="lag1_mean_autocorrelation",
                    ),
                    _realised_lag1(frame, feature),
                )
            )
        rows.append(
            _audit_row(
                world_id,
                replicate_index,
                source,
                task,
                "trial_count_p50",
                "n_trials_valid",
                _as_float(template.get("p50_n_trials_valid")),
                _as_float(frame["n_trials_valid"].median(skipna=True)),
            )
        )
    return pd.DataFrame(rows)


def write_empirical_twin_v1_outputs(
    runs: Sequence[EmpiricalTwinRun],
    *,
    output_dir: Path,
    config: dict[str, Any],
    config_path: Path,
    tournament_rows: Sequence[dict[str, Any]],
    split_rows: Sequence[dict[str, Any]],
) -> dict[str, Path]:
    """Write generated smoke datasets and aggregate audit outputs."""

    output_dir.mkdir(parents=True, exist_ok=True)
    generated = pd.concat([run.dataset for run in runs], ignore_index=True)
    generated_path = output_dir / "empirical_twin_v1_generated_windows.csv.gz"
    generated.to_csv(generated_path, index=False)
    generator_audit = pd.concat([run.generator_audit for run in runs], ignore_index=True)
    audit_path = output_dir / "empirical_twin_v1_generator_audit.csv"
    generator_audit.to_csv(audit_path, index=False)
    support = pd.concat([run.support_audit for run in runs], ignore_index=True)
    support_path = output_dir / "empirical_twin_v1_support_audit.csv"
    support.to_csv(support_path, index=False)
    summary = pd.DataFrame([run.generation_summary for run in runs])
    summary_path = output_dir / "empirical_twin_v1_generation_summary.csv"
    summary.to_csv(summary_path, index=False)
    tournament_path = output_dir / "empirical_twin_v1_tournament_smoke_model_scores.csv"
    pd.DataFrame(tournament_rows).to_csv(tournament_path, index=False)
    split_path = output_dir / "empirical_twin_v1_tournament_smoke_split_audit.csv"
    pd.DataFrame(split_rows).to_csv(split_path, index=False)
    manifest = {
        "study_id": str(config["study"]["id"]),
        "generator_id": EMPIRICAL_TWIN_V1_ID,
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "empirical_background_contract": EMPIRICAL_BACKGROUND_CONTRACT_ID,
        "empirical_background_contract_commit": str(
            config["contract"]["empirical_background_contract_commit"]
        ),
        "static_tournament_contract": STATIC_TOURNAMENT_V2_CONTRACT.to_dict(),
        "session_order_context_trend": _switch_label(
            config["contract"]["session_order_context_trend"]
        ),
        "paired_cross_task_session_covariance": _switch_label(
            config["generator"].get("paired_cross_task_session_covariance", "off")
        ),
        "formal_claims_allowed": False,
        "model_recovery_claim_allowed": False,
        "validation_repo_commit": _repo_commit_or_unknown(Path.cwd()),
        "config_path": str(config_path),
        "config_hash": hash_file(config_path),
        "config_content_hash": hash_mapping(config),
        "n_generated_runs": len(runs),
        "n_generated_rows": int(generated.shape[0]),
        "generated_truth_columns": [
            column for column in generated.columns if str(column).startswith("synthetic_")
        ],
        "outputs": {
            "generated_windows": str(generated_path),
            "generator_audit": str(audit_path),
            "support_audit": str(support_path),
            "generation_summary": str(summary_path),
            "tournament_smoke_model_scores": str(tournament_path),
            "tournament_smoke_split_audit": str(split_path),
        },
    }
    manifest_path = output_dir / "empirical_twin_v1_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    report_path = output_dir / "empirical_twin_v1_smoke_report.md"
    report_path.write_text(_smoke_report(summary, tournament_rows), encoding="utf-8")
    return {
        "generated_windows": generated_path,
        "generator_audit": audit_path,
        "support_audit": support_path,
        "generation_summary": summary_path,
        "tournament_smoke_model_scores": tournament_path,
        "tournament_smoke_split_audit": split_path,
        "manifest": manifest_path,
        "report": report_path,
    }


def _fit_score_models_without_truth_v2(
    frame_without_truth: pd.DataFrame,
    *,
    split_indices: tuple[tuple[int, ...], tuple[int, ...]],
    random_state: int,
) -> dict[str, pd.DataFrame]:
    assert_no_ground_truth_columns(frame_without_truth)
    train_indices, test_indices = split_indices
    train = frame_without_truth.loc[list(train_indices)].copy()
    test = frame_without_truth.loc[list(test_indices)].copy()
    model_rows: list[dict[str, Any]] = []
    participant_series: dict[str, pd.Series] = {}
    observed_counts = test.loc[:, list(FEATURES)].notna().sum(axis=1)
    total_observed = int(observed_counts.sum())
    for model in build_static_model_suite_v2(
        feature_columns=FEATURES,
        random_state=random_state,
    ):
        fitted = model.fit(train)
        sample_scores = fitted.score_samples(test).astype(float)
        holdout = fitted.score_holdout(test)
        valid = sample_scores.dropna()
        total = float(valid.sum()) if not valid.empty else float("nan")
        participant_means = _participant_scores(test, sample_scores)
        participant_series[fitted.model_id] = participant_means
        diagnostics = holdout.diagnostics or {}
        model_rows.append(
            {
                "model_id": fitted.model_id,
                "primary_metric": holdout.primary_metric or PRIMARY_METRIC,
                "heldout_log_likelihood_total": total,
                "heldout_log_density_mean_per_window": float(valid.mean()),
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
    return {
        "model_scores": pd.DataFrame(model_rows),
        "participant_scores": pd.concat(participant_series, axis=1),
    }


def _apply_background_to_source(
    frame: pd.DataFrame,
    *,
    source_index: pd.Index,
    means: dict[str, float],
    between_cov: np.ndarray,
    session_cov: np.ndarray,
    window_cov: np.ndarray,
    lag1: dict[str, float],
    fatigue: dict[str, float],
    missingness: dict[str, float],
    trial_counts: dict[str, float],
    structural_signal_fraction: float,
    rng: np.random.Generator,
) -> None:
    source_frame = frame.loc[source_index].copy()
    feature_matrix = source_frame.loc[:, list(FEATURES)].astype(float)
    standardised = (feature_matrix - feature_matrix.mean()) / feature_matrix.std(ddof=0).replace(0, 1)
    total_sd = np.sqrt(
        np.maximum(np.diag(between_cov) + np.diag(session_cov) + np.diag(window_cov), 1e-9)
    )
    structural = standardised.to_numpy(dtype=float) * total_sd * structural_signal_fraction
    person_effects: dict[tuple[str, str], np.ndarray] = {}
    session_effects: dict[tuple[str, str, str], np.ndarray] = {}
    mean_vector = np.array([means[feature] for feature in FEATURES], dtype=float)
    values = np.zeros((source_frame.shape[0], len(FEATURES)), dtype=float)
    source_positions = {index: pos for pos, index in enumerate(source_frame.index)}

    for participant_key, participant_group in source_frame.groupby(
        ["source_dataset", "participant_id"],
        sort=False,
    ):
        person_effects[participant_key] = rng.multivariate_normal(
            np.zeros(len(FEATURES)),
            between_cov,
        )
        for session_key, session_group in participant_group.groupby(
            ["source_dataset", "participant_id", "session_id"],
            sort=False,
        ):
            session_effects[session_key] = rng.multivariate_normal(
                np.zeros(len(FEATURES)),
                session_cov,
            )
            ar_state = np.zeros(len(FEATURES), dtype=float)
            ordered = session_group.sort_values("window_start_trial")
            for window_number, (row_index, row) in enumerate(ordered.iterrows()):
                innovation = rng.multivariate_normal(np.zeros(len(FEATURES)), window_cov)
                rho = np.array(
                    [np.clip(lag1.get(feature, 0.0), -0.8, 0.8) for feature in FEATURES],
                    dtype=float,
                )
                ar_state = rho * ar_state + innovation
                fatigue_vector = np.array(
                    [fatigue.get(feature, 0.0) * window_number for feature in FEATURES],
                    dtype=float,
                )
                pos = source_positions[row_index]
                values[pos, :] = (
                    mean_vector
                    + structural[pos, :]
                    + person_effects[participant_key]
                    + session_effects[session_key]
                    + ar_state
                    + fatigue_vector
                )

    for feature_index, feature in enumerate(FEATURES):
        frame.loc[source_frame.index, feature] = values[:, feature_index]
        miss_rate = float(np.clip(missingness.get(feature, 0.0), 0.0, 0.75))
        if miss_rate > 0:
            mask = rng.random(source_frame.shape[0]) < miss_rate
            frame.loc[source_frame.index[mask], feature] = np.nan
    counts = rng.normal(
        loc=trial_counts["mean"],
        scale=max(trial_counts["sd"], 1.0),
        size=source_frame.shape[0],
    )
    counts = np.clip(counts, trial_counts["p10"], trial_counts["p90"])
    valid = np.maximum(np.rint(counts), 1).astype(int)
    frame.loc[source_frame.index, "n_trials_valid"] = valid
    frame.loc[source_frame.index, "n_trials_total"] = np.maximum(valid, valid + 1)
    starts = frame.loc[source_frame.index, "window_start_trial"].astype(int)
    frame.loc[source_frame.index, "window_end_trial"] = starts + valid - 1
    frame.loc[source_frame.index, "trial_count"] = valid


def _select_templates(template_summary: pd.DataFrame, n_templates: int) -> pd.DataFrame:
    eligible = template_summary.copy()
    eligible["n_rows"] = pd.to_numeric(eligible["n_rows"], errors="coerce")
    eligible["n_participants"] = pd.to_numeric(eligible["n_participants"], errors="coerce")
    eligible = eligible[
        (eligible["n_rows"] >= 100)
        & (eligible["n_participants"] >= 50)
        & (eligible["task_id"].isin(["Stroop", "Flanker", "SART"]))
    ].copy()
    if eligible.shape[0] < n_templates:
        raise ValueError("not enough eligible ACDC templates for empirical-twin smoke")
    selected_rows = []
    for task in ("Stroop", "Flanker", "SART"):
        task_rows = eligible[eligible["task_id"] == task].sort_values(
            ["n_participants", "n_rows"],
            ascending=[False, False],
        )
        if not task_rows.empty:
            selected_rows.append(task_rows.iloc[0])
        if len(selected_rows) == n_templates:
            break
    if len(selected_rows) < n_templates:
        remaining = eligible.drop(index=[row.name for row in selected_rows]).sort_values(
            ["n_participants", "n_rows"],
            ascending=[False, False],
        )
        selected_rows.extend(
            remaining.iloc[index] for index in range(n_templates - len(selected_rows))
        )
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def _template_for_generated_source(templates: pd.DataFrame, source: object) -> pd.Series:
    matched = templates[templates["source_dataset"].astype(str) == str(source)]
    if matched.empty:
        raise ValueError(f"generated source has no empirical template: {source}")
    return matched.iloc[0]


def _template_means(template: pd.Series) -> dict[str, float]:
    return {
        feature: _as_float(template.get(f"{feature}_mean"), default=0.0)
        for feature in FEATURES
    }


def _trial_count_targets(template: pd.Series) -> dict[str, float]:
    mean = _as_float(template.get("mean_n_trials_valid"), default=50.0)
    sd = _as_float(template.get("sd_n_trials_valid"), default=5.0)
    return {
        "mean": mean,
        "sd": sd if np.isfinite(sd) and sd > 0 else 5.0,
        "p10": _as_float(template.get("p10_n_trials_valid"), default=max(1.0, mean - 2 * sd)),
        "p90": _as_float(template.get("p90_n_trials_valid"), default=mean + 2 * sd),
    }


def _covariance_matrix(
    covariance: pd.DataFrame,
    *,
    source: str,
    task: str,
    features: Sequence[str],
    use_covariance: bool,
) -> tuple[np.ndarray, str]:
    rows = covariance[
        (covariance["source_dataset"].astype(str) == str(source))
        & (covariance["task_id"].astype(str) == str(task))
        & (covariance["support_status"].astype(str) == "estimated")
    ]
    return _matrix_from_covariance_rows(rows, features, use_covariance=use_covariance)


def _paired_session_covariance(
    background: BackgroundSpec,
    *,
    task: str,
    features: Sequence[str],
    use_covariance: bool,
) -> tuple[np.ndarray, str]:
    rows = background.paired_cov_session[
        (background.paired_cov_session["task_id"].astype(str) == str(task))
        & (background.paired_cov_session["support_status"].astype(str) == "estimated")
    ].copy()
    rows["feature_a"] = rows["feature_a"].map(_from_paired_feature_name)
    rows["feature_b"] = rows["feature_b"].map(_from_paired_feature_name)
    matrix, mode = _matrix_from_covariance_rows(rows, features, use_covariance=use_covariance)
    if np.allclose(matrix, 0.0):
        diag = [
            max(_paired_target_variance(background.paired_variance, task, feature), 0.0)
            for feature in features
        ]
        matrix = np.diag(diag)
        mode = "paired_session_diagonal_variance"
    return matrix, mode


def _matrix_from_covariance_rows(
    rows: pd.DataFrame,
    features: Sequence[str],
    *,
    use_covariance: bool,
) -> tuple[np.ndarray, str]:
    matrix = np.zeros((len(features), len(features)), dtype=float)
    feature_index = {feature: index for index, feature in enumerate(features)}
    for _, row in rows.iterrows():
        a = str(row["feature_a"])
        b = str(row["feature_b"])
        if a not in feature_index or b not in feature_index:
            continue
        value = _as_float(row["covariance"], default=0.0)
        matrix[feature_index[a], feature_index[b]] = value
        matrix[feature_index[b], feature_index[a]] = value
    if not use_covariance:
        matrix = np.diag(np.maximum(np.diag(matrix), 0.0))
        return _ensure_psd(matrix), "diagonal_by_config"
    if np.allclose(matrix, 0.0):
        return matrix, "unsupported_zero_neutral"
    min_eigen = float(np.linalg.eigvalsh((matrix + matrix.T) / 2).min())
    if min_eigen < -1e-8:
        matrix = np.diag(np.maximum(np.diag(matrix), 0.0))
        return _ensure_psd(matrix), "diagonal_fallback_non_psd"
    return _ensure_psd(matrix), "support_eligible_covariance"


def _ensure_psd(matrix: np.ndarray) -> np.ndarray:
    symmetric = (matrix + matrix.T) / 2
    min_eigen = float(np.linalg.eigvalsh(symmetric).min()) if symmetric.size else 0.0
    if min_eigen < 0:
        symmetric = symmetric + np.eye(symmetric.shape[0]) * abs(min_eigen)
    return symmetric + np.eye(symmetric.shape[0]) * 1e-9


def _feature_lookup(
    frame: pd.DataFrame,
    *,
    source: str,
    task: str,
    column: str,
    default: float,
    status_column: str | None = None,
) -> dict[str, float]:
    result = {}
    for feature in FEATURES:
        value = _target_lookup(
            frame,
            source=source,
            task=task,
            feature=feature,
            column=column,
            status_column=status_column,
        )
        result[feature] = default if not np.isfinite(value) else float(value)
    return result


def _target_lookup(
    frame: pd.DataFrame,
    *,
    source: str,
    task: str,
    feature: str,
    column: str,
    status_column: str | None = None,
) -> float:
    rows = frame[
        (frame["source_dataset"].astype(str) == str(source))
        & (frame["task_id"].astype(str) == str(task))
        & (frame["feature"].astype(str) == str(feature))
    ]
    if status_column and status_column in rows and not rows.empty:
        rows = rows[rows[status_column].astype(str) == "estimated"]
    if rows.empty or column not in rows:
        return float("nan")
    return _as_float(rows.iloc[0][column])


def _paired_target_variance(frame: pd.DataFrame, task: str, feature: str) -> float:
    paired_feature = PAIRED_FEATURE_ALIASES.get(feature, feature)
    rows = frame[
        (frame["task_id"].astype(str) == str(task))
        & (frame["feature"].astype(str) == paired_feature)
        & (frame["session_support_status"].astype(str) == "estimated")
    ]
    if rows.empty:
        return 0.0
    return max(_as_float(rows.iloc[0]["session_within_participant_variance"], default=0.0), 0.0)


def _from_paired_feature_name(feature: object) -> str:
    value = str(feature)
    if value == "mean_rt_ms":
        return "median_rt_ms"
    return value


def _finalise_generated_bounds(frame: pd.DataFrame) -> pd.DataFrame:
    bounded = frame.copy()
    bounded["accuracy"] = bounded["accuracy"].clip(0.01, 0.999)
    bounded["median_rt_ms"] = bounded["median_rt_ms"].clip(150.0, 3000.0)
    bounded["mean_response_speed"] = bounded["mean_response_speed"].clip(0.05, 8.0)
    bounded["rt_cv"] = bounded["rt_cv"].clip(0.005, 2.0)
    bounded["throughput_proxy"] = bounded["throughput_proxy"].clip(0.01, 8.0)
    return bounded


def _reset_task_context_columns(frame: pd.DataFrame) -> None:
    task = frame["task_id"].astype(str)
    frame["has_conflict_cost"] = task.isin(["Stroop", "Flanker"])
    frame["has_post_error"] = task.isin(["Stroop", "Flanker"])
    frame["has_vigilance"] = task == "SART"
    frame["has_switch_structure"] = False
    frame["congruency_mix"] = np.where(frame["has_conflict_cost"], "balanced", "not_applicable")
    frame["switch_rate"] = np.nan
    frame["lure_rate"] = np.where(frame["has_vigilance"], 0.12, np.nan)
    for flag, columns in STRUCTURAL_FEATURES_BY_FLAG.items():
        if flag not in frame.columns:
            continue
        absent = ~frame[flag].astype(bool)
        for column in columns:
            if column in frame.columns:
                frame.loc[absent, column] = np.nan


def _realised_between_variance(frame: pd.DataFrame, feature: str) -> float:
    means = frame.groupby(["source_dataset", "participant_id"], sort=True)[feature].mean()
    return _as_float(means.var(ddof=1))


def _realised_session_variance(frame: pd.DataFrame, feature: str) -> float:
    session = frame.groupby(["source_dataset", "participant_id", "session_id"], sort=True)[feature].mean()
    participant = session.groupby(level=[0, 1]).transform("mean")
    deviations = (session - participant).dropna()
    return _as_float(deviations.var(ddof=1))


def _realised_window_variance(frame: pd.DataFrame, feature: str) -> float:
    session_mean = frame.groupby(["source_dataset", "participant_id", "session_id"], sort=True)[
        feature
    ].transform("mean")
    residual = (frame[feature] - session_mean).dropna()
    return _as_float(residual.var(ddof=1))


def _realised_lag1(frame: pd.DataFrame, feature: str) -> float:
    values = []
    for _, group in frame.groupby(["source_dataset", "participant_id", "session_id"], sort=True):
        series = group.sort_values("window_start_trial")[feature].dropna()
        if series.shape[0] >= 3 and series.var(ddof=0) > 0:
            previous = series.to_numpy(dtype=float)[:-1]
            current = series.to_numpy(dtype=float)[1:]
            if previous.std() > 0 and current.std() > 0:
                values.append(float(np.corrcoef(previous, current)[0, 1]))
    return float(np.nanmean(values)) if values else float("nan")


def _participant_scores(test: pd.DataFrame, scores: pd.Series) -> pd.Series:
    table = test.loc[:, ["source_dataset", "participant_id"]].copy()
    table["heldout_log_density"] = scores.reindex(test.index).to_numpy(dtype=float)
    return table.dropna(subset=["heldout_log_density"]).groupby(
        ["source_dataset", "participant_id"],
        sort=True,
    )["heldout_log_density"].mean()


def _n_participants(frame: pd.DataFrame) -> int:
    return int(frame.loc[:, ["source_dataset", "participant_id"]].drop_duplicates().shape[0])


def _audit_row(
    world_id: str,
    replicate_index: int,
    source: str,
    task: str,
    parameter: str,
    feature: str,
    target: float,
    realised: float,
) -> dict[str, Any]:
    return {
        "world_id": world_id,
        "replicate_index": int(replicate_index),
        "source_dataset": source,
        "task_id": task,
        "parameter": parameter,
        "feature": feature,
        "target": target,
        "realised": realised,
        "absolute_delta": (
            abs(realised - target)
            if np.isfinite(realised) and np.isfinite(target)
            else float("nan")
        ),
    }


def _support_row(
    world_id: str,
    replicate_index: int,
    quantity: str,
    status: str,
    detail: str,
) -> dict[str, Any]:
    return {
        "world_id": str(world_id),
        "replicate_index": int(replicate_index),
        "quantity": quantity,
        "support_status": status,
        "detail": detail,
    }


def _read_csv(path: Path, *, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path)


def _as_float(value: object, *, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _run_id(*parts: object) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:16]


def _child_seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:8], 16)


def _repo_commit_or_unknown(repo_root: Path) -> str:
    try:
        return get_git_commit(repo_root)
    except Exception:
        return "unknown"


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("contract", {}).get("empirical_background_contract") != EMPIRICAL_BACKGROUND_CONTRACT_ID:
        raise ValueError("empirical_twin_v1 requires EMPIRICAL_BACKGROUND_CONTRACT_V1")
    if config.get("contract", {}).get("static_tournament_contract") != STATIC_TOURNAMENT_V2_CONTRACT.contract_id:
        raise ValueError("empirical_twin_v1 requires static_tournament_v2")
    if not _is_switch_off(config.get("contract", {}).get("session_order_context_trend")):
        raise ValueError("primary empirical_twin_v1 smoke requires session_order_context_trend: off")


def _is_switch_off(value: object) -> bool:
    return str(value).strip().lower() in {"off", "false", "0", "none", ""}


def _switch_label(value: object) -> str:
    return "off" if _is_switch_off(value) else str(value)


def _smoke_report(summary: pd.DataFrame, tournament_rows: Sequence[dict[str, Any]]) -> str:
    return "\n".join(
        [
            "# M2.7 Empirical-Twin V1 Smoke",
            "",
            "formal_claims_allowed: `false`",
            "model_recovery_claim_allowed: `false`",
            f"generated_runs: {summary.shape[0]}",
            f"generated_rows: {int(summary['n_rows'].sum()) if not summary.empty else 0}",
            f"tournament_model_score_rows: {len(tournament_rows)}",
            "",
            "This smoke validates generation, audits and end-to-end static V2 scoring only.",
            "Do not interpret recovery rates from this run.",
            "",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run M2.7 empirical-twin V1 smoke.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-tournament-smoke", action="store_true")
    args = parser.parse_args(argv)
    result = run_empirical_twin_v1_smoke(
        config_path=args.config,
        output_dir=args.output_dir,
        run_tournament_smoke=not args.no_tournament_smoke,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "paths"}, indent=2))
    for name, path in result["paths"].items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
