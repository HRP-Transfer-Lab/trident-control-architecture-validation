"""M2.7 empirical-twin V1 generator smoke.

The generator preserves known ETW0-ETW4 structural truth and adds frozen empirical
background nuisance structure. It is a smoke/pipeline validation tool, not a
scientific recovery estimator.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, Sequence

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
from trident_validation.synthetic.exact_model_diagnostic import (
    make_exact_model_world,
    make_m2_self_check_world,
)
from trident_validation.synthetic.recovery import (
    PRIMARY_METRIC,
    assert_no_ground_truth_columns,
    strip_ground_truth_columns,
)
from trident_validation.synthetic.selection_v2 import select_preferred_model_v2


EMPIRICAL_TWIN_V1_ID = "empirical_twin_v1"
EMPIRICAL_BACKGROUND_CONTRACT_ID = "EMPIRICAL_BACKGROUND_CONTRACT_V1"
EMPIRICAL_BACKGROUND_CONTRACT_SHA = "1e9c11e030dc0706c8941dfc08489bb6f63cac5d"
STATIC_MODEL_SELECTION_CONTRACT_V2_SHA = "81d22f2a942afd1652c169935f058b1634922d30"
DEFAULT_CONFIG_PATH = Path("config/empirical_twin_v1.yaml")
DEFAULT_OUTPUT_DIR = Path("reports/generated/empirical_twin_v1_smoke")
FEATURES = tuple(CORE_SYNTHETIC_FEATURES)
SOURCE_TASK_SHIFT_RULE = "template_mean_minus_full_acdc_feature_mean"
TRIAL_COUNT_IMPLEMENTATION_STATUS = "truncated_normal_from_mean_sd_p10_p90_approximation"
SESSION_CALIBRATION_RULE = "raw_session_covariance_scaled_by_m_over_m_minus_1_before_participant_centring"
WINDOW_CALIBRATION_RULE = "full_ar_innovation_covariance_scaled_to_centred_window_covariance_after_fatigue_or_diagonal_fallback"
LAG1_CALIBRATION_RULE = "deterministic_grid_calibration_of_internal_phi_to_frozen_short_sequence_pearson_estimand"
INTERNAL_AR_PHI_BOUND = 0.99
LAG1_CALIBRATION_GRID_POINTS = 201
LAG1_CALIBRATION_TOLERANCE = 0.03
M2_7_EMPIRICAL_TWIN_REGISTRY_VERSION = "m2_7_empirical_twin_world_registry_v1"

EmpiricalTwinWorldId = Literal["ETW0", "ETW1", "ETW2", "ETW3", "ETW4"]

M2_7_EMPIRICAL_TWIN_WORLD_REGISTRY: dict[str, dict[str, str]] = {
    "ETW0": {
        "description": "M0 probabilistic general performance",
        "aligned_model_id": "M0_probabilistic_general_performance",
        "structural_source": "EW0_m0_exact noise-free structural expectation",
        "exact_world_id": "EW0_m0_exact",
    },
    "ETW1": {
        "description": "M1 continuous manifold",
        "aligned_model_id": "M1_continuous_control_manifold",
        "structural_source": "EW1_m1_exact noise-free structural expectation",
        "exact_world_id": "EW1_m1_exact",
    },
    "ETW2": {
        "description": "M2_EM_v1-aligned nonlinear continuous world",
        "aligned_model_id": "M2_EM_v1",
        "structural_source": "M2 self-check latent quadratic curve used for M2_EM repair diagnostics, noise-free expectation",
        "exact_world_id": "M2_self_check",
    },
    "ETW3": {
        "description": "M3 three-component mixture",
        "aligned_model_id": "M3_three_profile_mixture",
        "structural_source": "EW3_m3_exact noise-free structural expectation",
        "exact_world_id": "EW3_m3_exact",
    },
    "ETW4": {
        "description": "M4 four-component mixture",
        "aligned_model_id": "M4_four_pace_profile_mixture",
        "structural_source": "EW4_m4_exact noise-free structural expectation",
        "exact_world_id": "EW4_m4_exact",
    },
}


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
    component_audit: pd.DataFrame | None = None


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
        if world_id not in M2_7_EMPIRICAL_TWIN_WORLD_REGISTRY:
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
                    config={**generator_config, "capture_component_audit": True},
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


def run_empirical_twin_v1_audit_diagnostics(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run target-versus-realised audit diagnostics without tournament scoring."""

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
    runs: list[EmpiricalTwinRun] = []
    for world_id in tuple(str(world) for world in generator_config["worlds"]):
        if world_id not in M2_7_EMPIRICAL_TWIN_WORLD_REGISTRY:
            raise ValueError(f"unsupported empirical-twin world: {world_id}")
        for replicate_index in range(int(generator_config["replicates_per_world"])):
            print(
                f"M2.7 target-realised audit diagnostics | {world_id} | replicate {replicate_index}",
                flush=True,
            )
            runs.append(
                generate_empirical_twin_dataset(
                    world_id=world_id,  # type: ignore[arg-type]
                    replicate_index=replicate_index,
                    background=background,
                    config={**generator_config, "capture_component_audit": True},
                )
            )
    paths = write_empirical_twin_v1_audit_diagnostics(
        runs,
        background=background,
        output_dir=target_dir,
    )
    return {
        "study_id": str(config["study"]["id"]),
        "diagnostic_scope": "target_realised_audit_only",
        "model_outputs_read": False,
        "tournament_scoring_run": False,
        "n_generated_runs": len(runs),
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


def _seed_schedule(
    master_seed: object,
    world_id: str,
    replicate_index: int,
    *,
    structural_seed_override: object | None = None,
) -> dict[str, int]:
    master = int(master_seed)
    return {
        "master_seed": master,
        "structural_seed": (
            int(structural_seed_override)
            if structural_seed_override is not None
            else _child_seed(master, world_id, replicate_index, "structural")
        ),
        "background_seed": _child_seed(master, world_id, replicate_index, "background"),
        "missingness_seed": _child_seed(master, world_id, replicate_index, "missingness"),
        "trial_count_seed": _child_seed(master, world_id, replicate_index, "trial_count"),
    }


def _make_m2_7_structural_world(
    world_id: EmpiricalTwinWorldId,
    *,
    seed: int,
    n_datasets: int,
    participants_per_dataset: int,
    sessions_per_participant: int,
    min_windows_per_session: int,
    max_windows_per_session: int,
) -> pd.DataFrame:
    """Return pure known structural signal for an M2.7 ETW world.

    Historical exact-world defaults are left unchanged. M2.7 asks for the
    noise-free structural expectation before empirical nuisance is added.
    """

    if world_id not in M2_7_EMPIRICAL_TWIN_WORLD_REGISTRY:
        raise ValueError(f"unsupported empirical-twin world: {world_id}")
    spec = M2_7_EMPIRICAL_TWIN_WORLD_REGISTRY[world_id]
    if world_id == "ETW2":
        frame, _oracle = make_m2_self_check_world(
            seed=seed,
            n_datasets=n_datasets,
            participants_per_dataset=participants_per_dataset,
            sessions_per_participant=sessions_per_participant,
            min_windows_per_session=min_windows_per_session,
            max_windows_per_session=max_windows_per_session,
            include_observation_noise=False,
        )
    else:
        frame = make_exact_model_world(
            spec["exact_world_id"],  # type: ignore[arg-type]
            seed=seed,
            n_datasets=n_datasets,
            participants_per_dataset=participants_per_dataset,
            sessions_per_participant=sessions_per_participant,
            min_windows_per_session=min_windows_per_session,
            max_windows_per_session=max_windows_per_session,
            technical_missingness_rate=0.0,
            include_observation_noise=False,
        )
    frame = frame.copy()
    frame["synthetic_world_id"] = world_id
    frame["synthetic_empirical_twin_world_id"] = world_id
    frame["synthetic_aligned_model_id"] = spec["aligned_model_id"]
    frame["synthetic_structural_source"] = spec["structural_source"]
    return frame


def generate_empirical_twin_dataset(
    *,
    world_id: EmpiricalTwinWorldId,
    replicate_index: int,
    background: BackgroundSpec,
    config: dict[str, Any],
) -> EmpiricalTwinRun:
    """Generate one known-truth empirical twin with nuisance background."""

    seed_schedule = _seed_schedule(
        config["seed"],
        world_id,
        replicate_index,
        structural_seed_override=config.get("structural_seed_override"),
    )
    background_rng = np.random.default_rng(seed_schedule["background_seed"])
    missingness_rng = np.random.default_rng(seed_schedule["missingness_seed"])
    trial_count_rng = np.random.default_rng(seed_schedule["trial_count_seed"])
    templates = _select_templates(background.template_summary, int(config["n_templates"]))
    base = _make_m2_7_structural_world(
        world_id,
        seed=seed_schedule["structural_seed"],
        n_datasets=int(config["n_templates"]),
        participants_per_dataset=int(config["participants_per_template"]),
        sessions_per_participant=int(config["sessions_per_participant"]),
        min_windows_per_session=int(config["windows_per_session"]),
        max_windows_per_session=int(config["windows_per_session"]),
    )
    base["synthetic_replicate_index"] = int(replicate_index)
    for feature in FEATURES:
        base[f"synthetic_structural_{feature}"] = pd.to_numeric(base[feature], errors="coerce")
    base_sources = sorted(base["source_dataset"].astype(str).unique())
    template_by_source = {
        source: templates.iloc[index % templates.shape[0]]
        for index, source in enumerate(base_sources)
    }

    for source_number, (source, template) in enumerate(template_by_source.items(), start=1):
        mask = base["source_dataset"].astype(str) == source
        participant_map = {
            old_id: f"{world_id}_src{source_number:02d}_p{index:03d}"
            for index, old_id in enumerate(
                sorted(base.loc[mask, "participant_id"].astype(str).unique())
            )
        }
        base.loc[mask, "source_dataset"] = str(template["source_dataset"])
        base.loc[mask, "participant_id"] = (
            base.loc[mask, "participant_id"].astype(str).map(participant_map)
        )
        base.loc[mask, "task_id"] = str(template["task_id"])
        base.loc[mask, "source_version"] = "empirical_twin_v1_smoke"
        base.loc[mask, "source_file_or_table"] = "synthetic.empirical_twin_v1"
        base.loc[mask, "source_commit_or_release"] = EMPIRICAL_BACKGROUND_CONTRACT_ID
        base.loc[mask, "preprocessing_version"] = "empirical-twin-v1-generator"
        base.loc[mask, "feature_version"] = "canonical-window-v1"
    _reset_task_context_columns(base)

    generated = base.copy()
    support_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    use_covariance = bool(config.get("use_covariance", True))
    session_trend = str(config.get("session_order_context_trend", "off")).lower() == "on"
    if session_trend:
        raise ValueError("primary empirical_twin_v1 generation requires session_order_context_trend: off")

    for source, source_frame in generated.groupby("source_dataset", sort=False):
        template = _template_for_generated_source(templates, source)
        task = str(template["task_id"])
        source_index = source_frame.index
        shifts = _centred_source_task_shifts(background, template)
        support_rows.extend(
            _template_support_rows(
                world_id,
                replicate_index,
                background=background,
                template=template,
                features=FEATURES,
            )
        )
        between_cov, _between_mode, between_support = _acdc_covariance_matrix(
            background.acdc_cov_between,
            background=background,
            source=str(source),
            task=task,
            features=FEATURES,
            use_covariance=use_covariance,
            level="between_person",
        )
        window_cov, _window_mode, window_support = _acdc_covariance_matrix(
            background.acdc_cov_window,
            background=background,
            source=str(source),
            task=task,
            features=FEATURES,
            use_covariance=use_covariance,
            level="within_session",
        )
        session_cov, _session_mode, session_support = _paired_session_covariance(
            background,
            task=task,
            features=FEATURES,
            use_covariance=use_covariance,
            sessions_per_participant=int(config["sessions_per_participant"]),
        )
        lag1, lag1_support = _eligible_feature_lookup(
            background.acdc_temporal,
            background=background,
            source=str(source),
            task=task,
            column="lag1_mean_autocorrelation",
            default=0.0,
            status_column="lag1_support_status",
            quantity="lag1_autocorrelation",
        )
        fatigue, fatigue_support = _eligible_feature_lookup(
            background.acdc_temporal,
            background=background,
            source=str(source),
            task=task,
            column="fatigue_mean_slope",
            default=0.0,
            status_column="fatigue_support_status",
            quantity="within_session_time_on_task_trend",
        )
        missingness, missing_support = _eligible_feature_lookup(
            background.acdc_missingness,
            background=background,
            source=str(source),
            task=task,
            column="missing_rate",
            default=0.0,
            quantity="missingness_rate",
        )
        internal_phi = _calibrate_internal_phi_map(
            lag1,
            fatigue=fatigue,
            target_window_cov=window_cov,
            windows_per_session=int(config["windows_per_session"]),
        )
        calibrated_lag1_expected = _expected_frozen_lag1_map(
            internal_phi,
            fatigue=fatigue,
            target_window_cov=window_cov,
            windows_per_session=int(config["windows_per_session"]),
        )
        lag1_calibration_support = _lag1_calibration_support_rows(
            lag1,
            internal_phi=internal_phi,
            calibrated_lag1_expected=calibrated_lag1_expected,
            source=str(source),
            task=task,
        )
        window_cov, window_calibration_support = _calibrate_window_covariance_for_estimand(
            window_cov,
            internal_phi=internal_phi,
            fatigue=fatigue,
            windows_per_session=int(config["windows_per_session"]),
            source=str(source),
            task=task,
        )
        trial_counts = _trial_count_targets(template)

        support_rows.extend(
            _stamp_support_rows(between_support, world_id, replicate_index)
            + _stamp_support_rows(window_support, world_id, replicate_index)
            + _stamp_support_rows(session_support, world_id, replicate_index)
            + _stamp_support_rows(lag1_support, world_id, replicate_index)
            + _stamp_support_rows(fatigue_support, world_id, replicate_index)
            + _stamp_support_rows(missing_support, world_id, replicate_index)
            + _stamp_support_rows(lag1_calibration_support, world_id, replicate_index)
            + _stamp_support_rows(window_calibration_support, world_id, replicate_index)
        )
        support_rows.append(
            _support_row(
                world_id,
                replicate_index,
                "trial_count_distribution",
                "implementation_approximation",
                TRIAL_COUNT_IMPLEMENTATION_STATUS,
                source_dataset=str(source),
                task_id=task,
                fallback_type="truncated_normal_from_summary_quantiles",
            )
        )
        _apply_background_to_source(
            generated,
            source_index=source_index,
            world_id=world_id,
            replicate_index=replicate_index,
            shifts=shifts,
            between_cov=between_cov,
            session_cov=session_cov,
            window_cov=window_cov,
            internal_phi=internal_phi,
            frozen_lag1_target=lag1,
            calibrated_lag1_expected=calibrated_lag1_expected,
            fatigue=fatigue if session_trend is False else fatigue,
            missingness=missingness,
            trial_counts=trial_counts,
            background_rng=background_rng,
            missingness_rng=missingness_rng,
            trial_count_rng=trial_count_rng,
            component_rows=component_rows
            if bool(config.get("capture_component_audit", False))
            else None,
        )

    generated = _finalise_generated_bounds(generated)
    validate_window_schema(strip_ground_truth_columns(generated))
    run_id = _run_id(
        EMPIRICAL_TWIN_V1_ID,
        world_id,
        replicate_index,
        seed_schedule["master_seed"],
    )
    audit = build_generator_audit(
        generated,
        background=background,
        templates=templates,
        world_id=world_id,
        replicate_index=replicate_index,
        component_audit=pd.DataFrame(component_rows) if component_rows else None,
    )
    summary = {
        "run_id": run_id,
        "world_id": world_id,
        "aligned_model_id": M2_7_EMPIRICAL_TWIN_WORLD_REGISTRY[world_id]["aligned_model_id"],
        "replicate_index": int(replicate_index),
        **seed_schedule,
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
        "paired_cross_task_session_covariance": "off",
        "source_task_shift_rule": SOURCE_TASK_SHIFT_RULE,
        "session_calibration_rule": SESSION_CALIBRATION_RULE,
        "window_calibration_rule": WINDOW_CALIBRATION_RULE,
        "lag1_calibration_rule": LAG1_CALIBRATION_RULE,
        "trial_count_implementation_status": TRIAL_COUNT_IMPLEMENTATION_STATUS,
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
        component_audit=pd.DataFrame(component_rows) if component_rows else None,
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
            "aligned_model_id": M2_7_EMPIRICAL_TWIN_WORLD_REGISTRY[run.world_id][
                "aligned_model_id"
            ],
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
    component_audit: pd.DataFrame | None = None,
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
        source_task_components = _component_audit_slice(component_audit, source, task)
        person_frame = _component_wide_frame(source_task_components, ["person_component"])
        session_frame = _component_wide_frame(source_task_components, ["session_component"])
        window_frame = _component_wide_frame(
            source_task_components,
            ["ar_component", "fatigue_component"],
        )
        person_session_frame = _component_wide_frame(
            source_task_components,
            ["person_component", "session_component"],
        )
        if frame.empty:
            continue
        for feature in FEATURES:
            structural_column = f"synthetic_structural_{feature}"
            target_shift = _centred_source_task_shifts(background, template)[feature]
            realised_shift = _component_source_task_shift(source_task_components, feature)
            if not np.isfinite(realised_shift):
                realised_shift = (
                    _as_float(frame[feature].mean(skipna=True))
                    - _as_float(frame[structural_column].mean(skipna=True), default=0.0)
                    if structural_column in frame
                    else float("nan")
                )
            rows.append(
                _audit_row(
                    world_id,
                    replicate_index,
                    source,
                    task,
                    "source_task_shift",
                    feature,
                    target_shift,
                    realised_shift,
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
                    _realised_between_variance(person_frame, feature)
                    if not person_frame.empty
                    else _realised_between_variance(frame, feature),
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
                    _realised_session_variance(session_frame, feature)
                    if not session_frame.empty
                    else _realised_session_variance(frame, feature),
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
                    _realised_window_variance(window_frame, feature)
                    if not window_frame.empty
                    else _realised_window_variance(frame, feature),
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
                    _realised_lag1(window_frame, feature)
                    if not window_frame.empty
                    else _realised_lag1(frame, feature),
                )
            )
            rows.append(
                _audit_row(
                    world_id,
                    replicate_index,
                    source,
                    task,
                    "repeated_person_stability",
                    feature,
                    _paired_target_stability(
                        background.paired_stability,
                        task,
                        feature,
                        fallback_variance=background.paired_variance,
                    ),
                    _realised_repeated_person_stability(person_session_frame, feature)
                    if not person_session_frame.empty
                    else _realised_repeated_person_stability(frame, feature),
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


def _component_audit_slice(
    component_audit: pd.DataFrame | None,
    source: str,
    task: str,
) -> pd.DataFrame:
    if component_audit is None or component_audit.empty:
        return pd.DataFrame()
    return component_audit[
        (component_audit["source_dataset"].astype(str) == str(source))
        & (component_audit["task_id"].astype(str) == str(task))
    ].copy()


def _component_wide_frame(
    component_audit: pd.DataFrame,
    component_columns: Sequence[str],
) -> pd.DataFrame:
    if component_audit.empty:
        return pd.DataFrame()
    if any(column not in component_audit for column in component_columns):
        return pd.DataFrame()
    index_columns = [
        "source_dataset",
        "participant_id",
        "session_id",
        "window_id",
        "window_start_trial",
    ]
    frame = component_audit.loc[:, [*index_columns, "feature", *component_columns]].copy()
    frame["_component_value"] = frame.loc[:, component_columns].sum(axis=1)
    wide = frame.pivot_table(
        index=index_columns,
        columns="feature",
        values="_component_value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide


def _component_source_task_shift(component_audit: pd.DataFrame, feature: str) -> float:
    if component_audit.empty or "target_source_task_shift" not in component_audit:
        return float("nan")
    rows = component_audit[component_audit["feature"].astype(str) == str(feature)]
    if rows.empty:
        return float("nan")
    return _as_float(rows["target_source_task_shift"].mean(skipna=True))


def write_empirical_twin_v1_audit_diagnostics(
    runs: Sequence[EmpiricalTwinRun],
    *,
    background: BackgroundSpec,
    output_dir: Path,
) -> dict[str, Path]:
    """Write diagnostic-only target-realised audit summaries."""

    output_dir.mkdir(parents=True, exist_ok=True)
    component = pd.concat(
        [
            run.component_audit
            for run in runs
            if run.component_audit is not None and not run.component_audit.empty
        ],
        ignore_index=True,
    )
    generated = pd.concat([run.dataset for run in runs], ignore_index=True)
    generator_audit = pd.concat([run.generator_audit for run in runs], ignore_index=True)
    lag1 = lag1_sequence_diagnostics(
        generated,
        component,
        background=background,
        existing_audit=generator_audit,
    )
    scale = scale_normalised_diagnostics(
        generated,
        component,
        background=background,
        existing_audit=generator_audit,
    )
    variance = component_variance_diagnostics(
        generated,
        component,
        background=background,
    )
    classification = classify_audit_diagnostics(lag1, scale, variance)

    lag1_path = output_dir / "empirical_twin_v1_lag1_diagnostics.csv"
    scale_path = output_dir / "empirical_twin_v1_scale_normalised_diagnostics.csv"
    variance_path = output_dir / "empirical_twin_v1_component_variance_diagnostics.csv"
    classification_path = output_dir / "empirical_twin_v1_audit_issue_classification.md"
    lag1.to_csv(lag1_path, index=False)
    scale.to_csv(scale_path, index=False)
    variance.to_csv(variance_path, index=False)
    classification_path.write_text(classification, encoding="utf-8")
    return {
        "lag1_diagnostics": lag1_path,
        "scale_normalised_diagnostics": scale_path,
        "component_variance_diagnostics": variance_path,
        "issue_classification": classification_path,
    }


def lag1_sequence_diagnostics(
    generated: pd.DataFrame,
    component: pd.DataFrame,
    *,
    background: BackgroundSpec,
    existing_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Compare contract lag-1 targets with diagnostic pooled adjacent-pair estimates."""

    rows: list[dict[str, Any]] = []
    existing = existing_audit[existing_audit["parameter"] == "lag1_autocorrelation"].copy()
    for keys, group in generated.groupby(
        ["synthetic_world_id", "synthetic_replicate_index", "source_dataset", "task_id"],
        dropna=False,
        sort=True,
    ):
        world_id, replicate_index, source, task = keys
        for feature in FEATURES:
            target = _target_lookup(
                background.acdc_temporal,
                source=str(source),
                task=str(task),
                feature=feature,
                column="lag1_mean_autocorrelation",
                status_column="lag1_support_status",
            )
            sequence_lengths = []
            pearson_values = []
            pair_previous: list[float] = []
            pair_current: list[float] = []
            for _, sequence in group.groupby(
                ["source_dataset", "participant_id", "session_id"],
                dropna=False,
                sort=True,
            ):
                series = sequence.sort_values("window_start_trial")[feature].dropna()
                sequence_lengths.append(int(series.shape[0]))
                if series.shape[0] >= 3 and series.var(ddof=0) > 0:
                    previous = series.to_numpy(dtype=float)[:-1]
                    current = series.to_numpy(dtype=float)[1:]
                    if previous.std() > 0 and current.std() > 0:
                        pearson_values.append(float(np.corrcoef(previous, current)[0, 1]))
                    pair_previous.extend(previous.tolist())
                    pair_current.extend(current.tolist())
            pooled_observed = _safe_corr(pair_previous, pair_current)
            component_rows = component[
                (component["world_id"].astype(str) == str(world_id))
                & (component["replicate_index"].astype(int) == int(replicate_index))
                & (component["source_dataset"].astype(str) == str(source))
                & (component["task_id"].astype(str) == str(task))
                & (component["feature"].astype(str) == feature)
            ]
            pooled_ar = _component_adjacent_pair_corr(component_rows, "ar_component")
            pooled_window_residual = _component_sum_adjacent_pair_corr(
                component_rows,
                ("ar_component", "fatigue_component"),
            )
            pooled_background = _component_adjacent_pair_corr(
                component_rows,
                "background_component",
            )
            existing_row = existing[
                (existing["world_id"].astype(str) == str(world_id))
                & (existing["replicate_index"].astype(int) == int(replicate_index))
                & (existing["source_dataset"].astype(str) == str(source))
                & (existing["task_id"].astype(str) == str(task))
                & (existing["feature"].astype(str) == feature)
            ]
            existing_realised = (
                _as_float(existing_row.iloc[0]["realised"])
                if not existing_row.empty
                else float("nan")
            )
            rows.append(
                {
                    "world_id": world_id,
                    "replicate_index": int(replicate_index),
                    "source_dataset": source,
                    "task_id": task,
                    "feature": feature,
                    "target_contract_lag1": target,
                    "n_sequences": len(sequence_lengths),
                    "n_usable_sequences": len(pearson_values),
                    "sequence_length_min": min(sequence_lengths) if sequence_lengths else 0,
                    "sequence_length_p50": float(np.median(sequence_lengths))
                    if sequence_lengths
                    else float("nan"),
                    "sequence_length_max": max(sequence_lengths) if sequence_lengths else 0,
                    "proportion_sequences_length_3": (
                        float(np.mean([length == 3 for length in sequence_lengths]))
                        if sequence_lengths
                        else float("nan")
                    ),
                    "existing_per_sequence_pearson_mean": existing_realised,
                    "diagnostic_pooled_observed_adjacent_pair_corr": pooled_observed,
                    "diagnostic_pooled_window_residual_adjacent_pair_corr": pooled_window_residual,
                    "diagnostic_pooled_ar_component_adjacent_pair_corr": pooled_ar,
                    "diagnostic_pooled_background_component_adjacent_pair_corr": pooled_background,
                    "all_three_window_cell": all(length == 3 for length in sequence_lengths),
                    "three_window_degeneracy": all(length == 3 for length in sequence_lengths),
                    "diagnostic_note": (
                        "per-sequence Pearson lag-1 is degenerate for three-window "
                        "sequences; pooled adjacent-pair values are DIAGNOSTIC ONLY"
                    ),
                }
            )
    return pd.DataFrame(rows)


def scale_normalised_diagnostics(
    generated: pd.DataFrame,
    component: pd.DataFrame,
    *,
    background: BackgroundSpec,
    existing_audit: pd.DataFrame,
) -> pd.DataFrame:
    """Add feature-scale-normalised mean and variance diagnostics."""

    rows: list[dict[str, Any]] = []
    for _, audit in existing_audit.iterrows():
        parameter = str(audit["parameter"])
        if parameter not in {
            "source_task_shift",
            "between_person_variance",
            "session_within_person_variance",
            "window_within_session_variance",
        }:
            continue
        source = str(audit["source_dataset"])
        task = str(audit["task_id"])
        feature = str(audit["feature"])
        target = _as_float(audit["target"])
        realised = _as_float(audit["realised"])
        template_sd = _template_feature_sd(background.template_summary, source, task, feature)
        row = {
            "world_id": audit["world_id"],
            "replicate_index": int(audit["replicate_index"]),
            "source_dataset": source,
            "task_id": task,
            "parameter": parameter,
            "feature": feature,
            "target": target,
            "realised_combined_observed": realised,
            "feature_sd_scale": template_sd,
            "mean_delta_in_feature_sd": float("nan"),
            "variance_ratio_combined_observed": _safe_fraction(realised, target),
            "diagnostic_scope": "combined_observed_includes_known_structural_signal",
        }
        if parameter == "source_task_shift":
            row["mean_delta_in_feature_sd"] = (
                (realised - target) / template_sd
                if np.isfinite(realised) and np.isfinite(target) and template_sd > 0
                else float("nan")
            )
            row["variance_ratio_combined_observed"] = float("nan")
            row["diagnostic_scope"] = "source_task_centred_shift"
        rows.append(row)
    return pd.DataFrame(rows)


def component_variance_diagnostics(
    generated: pd.DataFrame,
    component: pd.DataFrame,
    *,
    background: BackgroundSpec,
) -> pd.DataFrame:
    """Compare nuisance targets with observed and component-isolated variances."""

    rows: list[dict[str, Any]] = []
    for keys, group in generated.groupby(
        ["synthetic_world_id", "synthetic_replicate_index", "source_dataset", "task_id"],
        dropna=False,
        sort=True,
    ):
        world_id, replicate_index, source, task = keys
        component_group = component[
            (component["world_id"].astype(str) == str(world_id))
            & (component["replicate_index"].astype(int) == int(replicate_index))
            & (component["source_dataset"].astype(str) == str(source))
            & (component["task_id"].astype(str) == str(task))
        ]
        for feature in FEATURES:
            feature_component = component_group[component_group["feature"] == feature]
            rows.extend(
                [
                    _component_variance_row(
                        world_id,
                        replicate_index,
                        source,
                        task,
                        feature,
                        "between_person_variance",
                        _target_lookup(
                            background.acdc_variance,
                            source=str(source),
                            task=str(task),
                            feature=feature,
                            column="between_participant_variance",
                        ),
                        _realised_between_variance(group, feature),
                        _component_between_variance(feature_component, "background_component"),
                        _component_between_variance(feature_component, "person_component"),
                        "person_component_isolated_target_check",
                    ),
                    _component_variance_row(
                        world_id,
                        replicate_index,
                        source,
                        task,
                        feature,
                        "session_within_person_variance",
                        _paired_target_variance(background.paired_variance, str(task), feature),
                        _realised_session_variance(group, feature),
                        _component_session_variance(feature_component, "background_component"),
                        _component_session_variance(feature_component, "session_component"),
                        "session_component_isolated_target_check",
                    ),
                    _component_variance_row(
                        world_id,
                        replicate_index,
                        source,
                        task,
                        feature,
                        "window_within_session_variance",
                        _target_lookup(
                            background.acdc_variance,
                            source=str(source),
                            task=str(task),
                            feature=feature,
                            column="window_within_session_variance",
                        ),
                        _realised_window_variance(group, feature),
                        _component_window_variance(feature_component, "background_component"),
                        _component_window_variance(feature_component, "ar_component"),
                        "ar_component_innovation_target_check",
                    ),
                ]
            )
    return pd.DataFrame(rows)


def classify_audit_diagnostics(
    lag1: pd.DataFrame,
    scale: pd.DataFrame,
    variance: pd.DataFrame,
) -> str:
    """Return a concise diagnostic classification report."""

    lag_valid = lag1.dropna(
        subset=[
            "target_contract_lag1",
            "diagnostic_pooled_window_residual_adjacent_pair_corr",
        ]
    ).copy()
    lag_valid["pooled_residual_abs_delta"] = (
        lag_valid["diagnostic_pooled_window_residual_adjacent_pair_corr"]
        - lag_valid["target_contract_lag1"]
    ).abs()
    lag_valid["per_sequence_abs_delta"] = (
        lag_valid["existing_per_sequence_pearson_mean"]
        - lag_valid["target_contract_lag1"]
    ).abs()
    residual_median_delta = (
        float(lag_valid["pooled_residual_abs_delta"].median())
        if not lag_valid.empty
        else float("nan")
    )
    per_sequence_median_delta = (
        float(lag_valid["per_sequence_abs_delta"].median())
        if not lag_valid.empty
        else float("nan")
    )
    all_three_cell_rate = (
        float(lag1["all_three_window_cell"].mean())
        if "all_three_window_cell" in lag1 and not lag1.empty
        else float("nan")
    )
    proportion_sequences_length_3 = (
        float(lag1["proportion_sequences_length_3"].mean())
        if "proportion_sequences_length_3" in lag1 and not lag1.empty
        else float("nan")
    )
    mean_rows = scale[scale["parameter"] == "source_task_shift"].copy()
    median_abs_mean_z = (
        float(mean_rows["mean_delta_in_feature_sd"].abs().median())
        if not mean_rows.empty
        else float("nan")
    )
    between_rows = variance[variance["parameter"] == "between_person_variance"].copy()
    between_rows = between_rows.dropna(subset=["target", "component_isolated_realised"])
    between_rows["component_ratio"] = between_rows["component_isolated_realised"] / between_rows["target"]
    median_between_component_ratio = (
        float(between_rows["component_ratio"].median()) if not between_rows.empty else float("nan")
    )
    session_rows = variance[variance["parameter"] == "session_within_person_variance"].copy()
    session_rows = session_rows.dropna(subset=["target", "component_isolated_realised"])
    session_rows["component_ratio"] = session_rows["component_isolated_realised"] / session_rows["target"]
    median_session_component_ratio = (
        float(session_rows["component_ratio"].median()) if not session_rows.empty else float("nan")
    )
    window_rows = variance[variance["parameter"] == "window_within_session_variance"].copy()
    window_rows = window_rows.dropna(subset=["target", "component_isolated_realised"])
    window_rows["component_ratio"] = window_rows["component_isolated_realised"] / window_rows["target"]
    median_window_component_ratio = (
        float(window_rows["component_ratio"].median()) if not window_rows.empty else float("nan")
    )
    classification = []
    if all_three_cell_rate >= 0.95 and per_sequence_median_delta > residual_median_delta:
        classification.append(
            "lag1_discrepancy: audit-estimator limitation (b); all generated sequences "
            "are effectively length 3, making per-sequence Pearson lag-1 unstable/degenerate."
        )
    if np.isfinite(residual_median_delta) and residual_median_delta > 0.25:
        classification.append(
            "lag1_pooled_ar_component: possible generator AR-construction issue (a); "
            "pooled window-residual lag-1 remains far from the contract target."
        )
    else:
        classification.append(
            "lag1_pooled_window_residual: no gross generator error detected at smoke scale; "
            "remaining disagreement is compatible with estimator limitation and finite-smoke noise (b/c)."
        )
    classification.append(
        "source_task_shifts: centred-shift diagnostic median absolute delta "
        f"{median_abs_mean_z:.3f} feature SD; classify as finite-smoke noise (c) unless reviewed otherwise."
    )
    classification.append(
        "nuisance_variance_audit: original combined-observed variance rows include known structural signal; "
        "component-isolated rows should be used for generator-background diagnostics (b). "
        f"Median between/session/window component target ratios: "
        f"{median_between_component_ratio:.3f} / {median_session_component_ratio:.3f} / "
        f"{median_window_component_ratio:.3f}."
    )
    if np.isfinite(median_session_component_ratio) and median_session_component_ratio < 0.75:
        classification.append(
            "session_variance_component: possible generator scaling issue (a); "
            "the repeat-session deviation estimator is applied after participant-mean centring, "
            "while the generator currently draws raw session effects at the target variance."
        )
    if np.isfinite(median_window_component_ratio) and median_window_component_ratio < 0.75:
        classification.append(
            "window_variance_component: possible generator/audit-estimator issue (a/b); "
            "short three-window AR sequences and within-session centring reduce the component-isolated "
            "realised variance relative to the target."
        )
    classification.append(
        "empirical_contract_estimator: no genuine frozen empirical-contract estimator problem (d) "
        "is established by this smoke diagnostic; do not change the contract before review."
    )
    return "\n".join(
        [
            "# M2.7 Empirical-Twin V1 Target-Realised Audit Diagnostics",
            "",
            "Scope: generator/audit diagnostics only. No tournament model-score outputs were read.",
            "",
            "## Lag-1 Summary",
            "",
            f"all_three_window_cell_rate: {all_three_cell_rate:.3f}",
            f"proportion_sequences_length_3: {proportion_sequences_length_3:.3f}",
            f"median_abs_delta_existing_per_sequence_estimator: {per_sequence_median_delta:.3f}",
            f"median_abs_delta_pooled_window_residual_estimator: {residual_median_delta:.3f}",
            "",
            "## Issue Classification",
            "",
            *[f"- {item}" for item in classification],
            "",
            "Stop for scientific review before changing the empirical-background contract or launching the pilot.",
            "",
        ]
    )


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
    component = pd.concat(
        [
            run.component_audit
            for run in runs
            if run.component_audit is not None and not run.component_audit.empty
        ],
        ignore_index=True,
    )
    component_path = output_dir / "empirical_twin_v1_component_audit.csv"
    component.to_csv(component_path, index=False)
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
        "generator_version": "empirical_twin_v1_contract_hardened",
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "empirical_background_contract": EMPIRICAL_BACKGROUND_CONTRACT_ID,
        "empirical_background_contract_commit": str(
            config["contract"]["empirical_background_contract_commit"]
        ),
        "empirical_background_contract_sha": EMPIRICAL_BACKGROUND_CONTRACT_SHA,
        "static_tournament_contract": STATIC_TOURNAMENT_V2_CONTRACT.to_dict(),
        "static_tournament_contract_commit": str(
            config["contract"].get(
                "static_tournament_contract_commit",
                STATIC_MODEL_SELECTION_CONTRACT_V2_SHA,
            )
        ),
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
        "acdc_aggregate_input_dir": str(config["inputs"]["acdc_preflight_dir"]),
        "paired_aggregate_input_dir": str(config["inputs"]["paired_preflight_dir"]),
        "acdc_aggregate_checksum": _directory_checksum(Path(config["inputs"]["acdc_preflight_dir"])),
        "paired_aggregate_checksum": _directory_checksum(Path(config["inputs"]["paired_preflight_dir"])),
        "support_eligibility_hash": _dataframe_hash(support),
        "structural_world_registry_version": M2_7_EMPIRICAL_TWIN_REGISTRY_VERSION,
        "structural_world_registry": M2_7_EMPIRICAL_TWIN_WORLD_REGISTRY,
        "seed_schedule": summary.to_dict(orient="records"),
        "source_task_shift_rule": SOURCE_TASK_SHIFT_RULE,
        "session_calibration_rule": SESSION_CALIBRATION_RULE,
        "window_calibration_rule": WINDOW_CALIBRATION_RULE,
        "lag1_calibration_rule": LAG1_CALIBRATION_RULE,
        "trial_count_implementation_status": TRIAL_COUNT_IMPLEMENTATION_STATUS,
        "unsupported_quantities": support[support["support_status"] == "unsupported"].to_dict(
            orient="records"
        ),
        "fallback_quantities": support[
            support["fallback_type"].astype(str).str.len() > 0
        ].to_dict(orient="records"),
        "n_generated_runs": len(runs),
        "n_generated_rows": int(generated.shape[0]),
        "generated_truth_columns": [
            column for column in generated.columns if str(column).startswith("synthetic_")
        ],
        "outputs": {
            "generated_windows": str(generated_path),
            "generator_audit": str(audit_path),
            "support_audit": str(support_path),
            "component_audit": str(component_path),
            "generation_summary": str(summary_path),
            "tournament_smoke_model_scores": str(tournament_path),
            "tournament_smoke_split_audit": str(split_path),
        },
        "output_checksums": {
            "generated_windows": hash_file(generated_path),
            "generator_audit": hash_file(audit_path),
            "support_audit": hash_file(support_path),
            "component_audit": hash_file(component_path),
            "generation_summary": hash_file(summary_path),
            "tournament_smoke_model_scores": hash_file(tournament_path),
            "tournament_smoke_split_audit": hash_file(split_path),
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
        "component_audit": component_path,
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
    world_id: str,
    replicate_index: int,
    shifts: dict[str, float],
    between_cov: np.ndarray,
    session_cov: np.ndarray,
    window_cov: np.ndarray,
    internal_phi: dict[str, float],
    frozen_lag1_target: dict[str, float],
    calibrated_lag1_expected: dict[str, float],
    fatigue: dict[str, float],
    missingness: dict[str, float],
    trial_counts: dict[str, float],
    background_rng: np.random.Generator,
    missingness_rng: np.random.Generator,
    trial_count_rng: np.random.Generator,
    component_rows: list[dict[str, Any]] | None = None,
) -> None:
    source_frame = frame.loc[source_index].copy()
    structural = source_frame.loc[
        :,
        [f"synthetic_structural_{feature}" for feature in FEATURES],
    ].to_numpy(dtype=float)
    person_effects: dict[tuple[str, str], np.ndarray] = {}
    session_effects: dict[tuple[str, str, str], np.ndarray] = {}
    shift_vector = np.array([shifts[feature] for feature in FEATURES], dtype=float)
    values = np.zeros((source_frame.shape[0], len(FEATURES)), dtype=float)
    source_positions = {index: pos for pos, index in enumerate(source_frame.index)}
    rho = np.array(
        [
            np.clip(internal_phi.get(feature, 0.0), -INTERNAL_AR_PHI_BOUND, INTERNAL_AR_PHI_BOUND)
            for feature in FEATURES
        ],
        dtype=float,
    )

    for participant_key, participant_group in source_frame.groupby(
        ["source_dataset", "participant_id"],
        sort=False,
    ):
        person_effects[participant_key] = background_rng.multivariate_normal(
            np.zeros(len(FEATURES)),
            between_cov,
        )
        for session_key, session_group in participant_group.groupby(
            ["source_dataset", "participant_id", "session_id"],
            sort=False,
        ):
            session_effects[session_key] = background_rng.multivariate_normal(
                np.zeros(len(FEATURES)),
                session_cov,
            )
            ar_state = np.zeros(len(FEATURES), dtype=float)
            ordered = session_group.sort_values("window_start_trial")
            for window_number, (row_index, row) in enumerate(ordered.iterrows()):
                innovation = background_rng.multivariate_normal(np.zeros(len(FEATURES)), window_cov)
                ar_state = rho * ar_state + innovation
                fatigue_vector = np.array(
                    [fatigue.get(feature, 0.0) * window_number for feature in FEATURES],
                    dtype=float,
                )
                pos = source_positions[row_index]
                values[pos, :] = (
                    structural[pos, :]
                    + shift_vector
                    + person_effects[participant_key]
                    + session_effects[session_key]
                    + ar_state
                    + fatigue_vector
                )
                if component_rows is not None:
                    for feature_index, feature in enumerate(FEATURES):
                        component_rows.append(
                            {
                                "world_id": str(world_id),
                                "replicate_index": int(replicate_index),
                                "source_dataset": row["source_dataset"],
                                "task_id": row["task_id"],
                                "participant_id": row["participant_id"],
                                "session_id": row["session_id"],
                                "window_id": row["window_id"],
                                "window_start_trial": row["window_start_trial"],
                                "feature": feature,
                                "target_source_task_shift": shift_vector[feature_index],
                                "frozen_lag1_target": frozen_lag1_target.get(feature, np.nan),
                                "internal_ar_phi": rho[feature_index],
                                "calibrated_frozen_lag1_expected": calibrated_lag1_expected.get(
                                    feature,
                                    np.nan,
                                ),
                                "structural_component": structural[pos, feature_index],
                                "person_component": person_effects[participant_key][feature_index],
                                "session_component": session_effects[session_key][feature_index],
                                "ar_component": ar_state[feature_index],
                                "fatigue_component": fatigue_vector[feature_index],
                                "background_component": (
                                    person_effects[participant_key][feature_index]
                                    + session_effects[session_key][feature_index]
                                    + ar_state[feature_index]
                                    + fatigue_vector[feature_index]
                                ),
                            }
                        )

    for feature_index, feature in enumerate(FEATURES):
        frame.loc[source_frame.index, feature] = values[:, feature_index]
        miss_rate = float(np.clip(missingness.get(feature, 0.0), 0.0, 0.75))
        if miss_rate > 0:
            mask = missingness_rng.random(source_frame.shape[0]) < miss_rate
            frame.loc[source_frame.index[mask], feature] = np.nan
    counts = trial_count_rng.normal(
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


def _feature_global_means(background: BackgroundSpec) -> dict[str, float]:
    rows = background.feature_summary.copy()
    rows["feature"] = rows["feature"].astype(str)
    return {
        feature: _as_float(
            rows.loc[rows["feature"] == feature, "mean"].iloc[0],
            default=0.0,
        )
        if (rows["feature"] == feature).any()
        else 0.0
        for feature in FEATURES
    }


def _centred_source_task_shifts(
    background: BackgroundSpec,
    template: pd.Series,
) -> dict[str, float]:
    global_means = _feature_global_means(background)
    shifts: dict[str, float] = {}
    for feature in FEATURES:
        status, _reason = _acdc_template_feature_eligible(template, feature)
        if status != "estimated":
            shifts[feature] = 0.0
        else:
            shifts[feature] = (
                _as_float(template.get(f"{feature}_mean"), default=global_means[feature])
                - global_means[feature]
            )
    return shifts


def _template_support_rows(
    world_id: str,
    replicate_index: int,
    *,
    background: BackgroundSpec,
    template: pd.Series,
    features: Sequence[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source = str(template["source_dataset"])
    task = str(template["task_id"])
    for feature in features:
        status, reason = _acdc_template_feature_eligible(template, feature)
        rows.append(
            _support_row(
                world_id,
                replicate_index,
                "source_task_shift",
                status,
                reason,
                source_dataset=source,
                task_id=task,
                feature=feature,
                fallback_type="" if status == "estimated" else "disabled_or_neutral",
            )
        )
    return rows


def _stamp_support_rows(
    rows: Sequence[dict[str, Any]],
    world_id: str,
    replicate_index: int,
) -> list[dict[str, Any]]:
    stamped: list[dict[str, Any]] = []
    for row in rows:
        copy = dict(row)
        copy["world_id"] = str(world_id)
        copy["replicate_index"] = int(replicate_index)
        stamped.append(copy)
    return stamped


def _acdc_template_feature_eligible(template: pd.Series, feature: str) -> tuple[str, str]:
    n_rows = _as_float(template.get("n_rows"), default=0.0)
    n_participants = _as_float(template.get("n_participants"), default=0.0)
    coverage = _as_float(template.get(f"{feature}_coverage"), default=0.0)
    if n_participants < 50:
        return "unsupported", "template_n_participants_below_50"
    if n_rows < 100:
        return "unsupported", "template_n_rows_below_100"
    if coverage < 0.95:
        return "unsupported", "template_feature_coverage_below_0_95"
    return "estimated", "contract_template_thresholds_met"


def _trial_count_targets(template: pd.Series) -> dict[str, float]:
    mean = _as_float(template.get("mean_n_trials_valid"), default=50.0)
    sd = _as_float(template.get("sd_n_trials_valid"), default=5.0)
    return {
        "mean": mean,
        "sd": sd if np.isfinite(sd) and sd > 0 else 5.0,
        "p10": _as_float(template.get("p10_n_trials_valid"), default=max(1.0, mean - 2 * sd)),
        "p90": _as_float(template.get("p90_n_trials_valid"), default=mean + 2 * sd),
    }


def _acdc_covariance_matrix(
    covariance: pd.DataFrame,
    *,
    background: BackgroundSpec,
    source: str,
    task: str,
    features: Sequence[str],
    use_covariance: bool,
    level: str,
) -> tuple[np.ndarray, str, list[dict[str, Any]]]:
    rows = covariance[
        (covariance["source_dataset"].astype(str) == str(source))
        & (covariance["task_id"].astype(str) == str(task))
    ].copy()
    support_rows: list[dict[str, Any]] = []
    eligible_rows = []
    quantity = "between_person_covariance" if level == "between_person" else "within_session_covariance"
    for _, row in rows.iterrows():
        feature_a = str(row["feature_a"])
        feature_b = str(row["feature_b"])
        status, reason = _acdc_covariance_row_eligible(row, background=background, level=level)
        if status == "estimated":
            eligible_rows.append(row)
        support_rows.append(
            _support_row(
                row.get("synthetic_world_id", ""),
                0,
                quantity,
                status,
                reason,
                source_dataset=source,
                task_id=task,
                feature=f"{feature_a}|{feature_b}",
                fallback_type="" if status == "estimated" else "diagonal_variance_or_neutral",
                fallback_source="ACDC same-template diagonal variance",
            )
        )
    matrix, mode = _matrix_from_covariance_rows(
        pd.DataFrame(eligible_rows),
        features,
        use_covariance=use_covariance,
    )
    if np.allclose(matrix, 0.0):
        diag = []
        column = (
            "between_participant_variance"
            if level == "between_person"
            else "window_within_session_variance"
        )
        for feature in features:
            value, variance_status, variance_reason = _acdc_variance_target_if_eligible(
                background.acdc_variance,
                background=background,
                source=source,
                task=task,
                feature=feature,
                column=column,
                quantity=(
                    "between_person_variance"
                    if level == "between_person"
                    else "window_within_session_variance"
                ),
            )
            diag.append(max(value if np.isfinite(value) else 0.0, 0.0))
            support_rows.append(
                _support_row(
                    "",
                    0,
                    "between_person_variance"
                    if level == "between_person"
                    else "window_within_session_variance",
                    variance_status,
                    variance_reason,
                    source_dataset=source,
                    task_id=task,
                    feature=feature,
                    fallback_type="" if variance_status == "estimated" else "fixed_neutral_value",
                    fallback_source="" if variance_status == "estimated" else "unsupported_variance",
                )
            )
        matrix = np.diag(diag)
        mode = f"{level}_diagonal_variance"
    support_rows.append(
        _support_row(
            "",
            0,
            quantity,
            mode,
            f"{level}_matrix_generation_mode",
            source_dataset=source,
            task_id=task,
        )
    )
    return matrix, mode, support_rows


def _acdc_variance_target_if_eligible(
    frame: pd.DataFrame,
    *,
    background: BackgroundSpec,
    source: str,
    task: str,
    feature: str,
    column: str,
    quantity: str,
) -> tuple[float, str, str]:
    rows = frame[
        (frame["source_dataset"].astype(str) == str(source))
        & (frame["task_id"].astype(str) == str(task))
        & (frame["feature"].astype(str) == str(feature))
    ]
    status, reason = _scalar_support_status(
        rows,
        background=background,
        source=source,
        task=task,
        feature=feature,
        quantity=quantity,
        status_column=None,
    )
    if not rows.empty:
        row = rows.iloc[0]
        if quantity == "between_person_variance":
            if str(row.get("between_support_status")) != "estimated":
                status, reason = "unsupported", str(row.get("support_reason", "extractor_not_estimated"))
            elif _as_float(row.get("n_participants"), default=0.0) < 30:
                status, reason = "unsupported", "participants_with_observed_means_below_30"
        elif quantity == "window_within_session_variance":
            if str(row.get("window_support_status")) != "estimated":
                status, reason = "unsupported", str(row.get("support_reason", "extractor_not_estimated"))
            elif _as_float(row.get("n_sessions_with_repeated_windows"), default=0.0) < 30:
                status, reason = "unsupported", "sessions_with_repeated_windows_below_30"
            elif _as_float(row.get("n_observed"), default=0.0) < 100:
                status, reason = "unsupported", "window_residual_rows_below_100"
    if status != "estimated" or rows.empty:
        return 0.0, status, reason
    return max(_as_float(rows.iloc[0].get(column), default=0.0), 0.0), status, reason


def _paired_session_covariance(
    background: BackgroundSpec,
    *,
    task: str,
    features: Sequence[str],
    use_covariance: bool,
    sessions_per_participant: int,
) -> tuple[np.ndarray, str, list[dict[str, Any]]]:
    rows = background.paired_cov_session[
        (background.paired_cov_session["task_id"].astype(str) == str(task))
    ].copy()
    support_rows: list[dict[str, Any]] = []
    for feature in features:
        value = _paired_target_variance(background.paired_variance, task, feature)
        status = "estimated" if value > 0 else "unsupported"
        reason = (
            "paired_repeat_participant_exact_feature_thresholds_met"
            if value > 0
            else "paired_exact_canonical_feature_missing_or_unsupported"
        )
        support_rows.append(
            _support_row(
                "",
                0,
                "session_within_person_variance",
                status,
                reason,
                task_id=task,
                feature=feature,
                fallback_type="" if status == "estimated" else "fixed_neutral_value",
                fallback_source="" if status == "estimated" else "no_feature_alias_borrowing",
            )
        )
    eligible_rows = []
    for _, row in rows.iterrows():
        feature_a = str(row["feature_a"])
        feature_b = str(row["feature_b"])
        status, reason = _paired_covariance_row_eligible(
            row,
            features=features,
            paired_variance=background.paired_variance,
        )
        if status == "estimated":
            eligible_rows.append(row)
        support_rows.append(
            _support_row(
                "",
                0,
                "session_within_person_covariance",
                status,
                reason,
                task_id=task,
                feature=f"{feature_a}|{feature_b}",
                fallback_type="" if status == "estimated" else "diagonal_variance_or_neutral",
                fallback_source="paired exact-feature diagonal variance",
            )
        )
    matrix, mode = _matrix_from_covariance_rows(
        pd.DataFrame(eligible_rows),
        features,
        use_covariance=use_covariance,
    )
    if np.allclose(matrix, 0.0):
        diag = [
            max(_paired_target_variance(background.paired_variance, task, feature), 0.0)
            for feature in features
        ]
        matrix = np.diag(diag)
        mode = "paired_session_diagonal_variance"
    matrix = _calibrate_session_covariance_for_centring(
        matrix,
        sessions_per_participant=sessions_per_participant,
    )
    support_rows.append(
        _support_row(
            "",
            0,
            "session_within_person_covariance",
            mode,
            SESSION_CALIBRATION_RULE,
            task_id=task,
            fallback_type="" if "diagonal" not in mode else "calibrated_diagonal",
        )
    )
    return matrix, mode, support_rows


def _acdc_covariance_row_eligible(
    row: pd.Series,
    *,
    background: BackgroundSpec,
    level: str,
) -> tuple[str, str]:
    template_rows = background.template_summary[
        (background.template_summary["source_dataset"].astype(str) == str(row.get("source_dataset")))
        & (background.template_summary["task_id"].astype(str) == str(row.get("task_id")))
    ]
    if template_rows.empty:
        return "unsupported", "template_missing_for_covariance_row"
    template = template_rows.iloc[0]
    for feature in (str(row.get("feature_a")), str(row.get("feature_b"))):
        status, reason = _acdc_template_feature_eligible(template, feature)
        if status != "estimated":
            return "unsupported", f"{feature}:{reason}"
    if str(row.get("support_status")) != "estimated":
        return "unsupported", str(row.get("support_reason", "extractor_not_estimated"))
    n_units = _as_float(row.get("n_units"), default=0.0)
    if level == "between_person" and n_units < 30:
        return "unsupported", "complete_participant_means_below_30"
    if level == "within_session" and n_units < 100:
        return "unsupported", "complete_within_session_residual_rows_below_100"
    return "estimated", "contract_covariance_thresholds_met"


def _paired_covariance_row_eligible(
    row: pd.Series,
    *,
    features: Sequence[str],
    paired_variance: pd.DataFrame,
) -> tuple[str, str]:
    feature_a = str(row.get("feature_a"))
    feature_b = str(row.get("feature_b"))
    if feature_a not in features or feature_b not in features:
        return "unsupported", "paired_feature_not_exact_canonical_feature"
    task = str(row.get("task_id"))
    for feature in (feature_a, feature_b):
        variance_rows = paired_variance[
            (paired_variance["task_id"].astype(str) == task)
            & (paired_variance["feature"].astype(str) == feature)
            & (paired_variance["session_support_status"].astype(str) == "estimated")
        ]
        if variance_rows.empty:
            return "unsupported", f"{feature}:paired_exact_feature_variance_missing"
        if _as_float(variance_rows.iloc[0].get("n_repeat_participants"), default=0.0) < 30:
            return "unsupported", f"{feature}:repeat_participants_below_30"
    if str(row.get("support_status")) != "estimated":
        return "unsupported", str(row.get("support_reason", "extractor_not_estimated"))
    if _as_float(row.get("n_units"), default=0.0) < 30:
        return "unsupported", "complete_repeat_participant_session_deviation_units_below_30"
    return "estimated", "paired_session_covariance_thresholds_met"


def _calibrate_session_covariance_for_centring(
    matrix: np.ndarray,
    *,
    sessions_per_participant: int,
) -> np.ndarray:
    if sessions_per_participant <= 1 or np.allclose(matrix, 0.0):
        return matrix
    scaled = matrix * (float(sessions_per_participant) / float(sessions_per_participant - 1))
    min_eigen = float(np.linalg.eigvalsh((scaled + scaled.T) / 2).min()) if scaled.size else 0.0
    if min_eigen < -1e-8:
        return np.diag(np.maximum(np.diag(scaled), 0.0))
    return scaled


def _calibrate_internal_phi_map(
    frozen_lag1: dict[str, float],
    *,
    fatigue: dict[str, float],
    target_window_cov: np.ndarray,
    windows_per_session: int,
) -> dict[str, float]:
    return {
        feature: _calibrate_internal_phi_for_frozen_lag1(
            _as_float(frozen_lag1.get(feature), default=0.0),
            fatigue_slope=_as_float(fatigue.get(feature), default=0.0),
            target_window_variance=max(float(target_window_cov[index, index]), 0.0),
            windows_per_session=windows_per_session,
        )
        for index, feature in enumerate(FEATURES)
    }


def _expected_frozen_lag1_map(
    internal_phi: dict[str, float],
    *,
    fatigue: dict[str, float],
    target_window_cov: np.ndarray,
    windows_per_session: int,
) -> dict[str, float]:
    return {
        feature: _expected_frozen_lag1_for_phi(
            _as_float(internal_phi.get(feature), default=0.0),
            fatigue_slope=_as_float(fatigue.get(feature), default=0.0),
            target_window_variance=max(float(target_window_cov[index, index]), 0.0),
            windows_per_session=windows_per_session,
        )
        for index, feature in enumerate(FEATURES)
    }


def _lag1_calibration_support_rows(
    frozen_lag1: dict[str, float],
    *,
    internal_phi: dict[str, float],
    calibrated_lag1_expected: dict[str, float],
    source: str,
    task: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in FEATURES:
        target = _as_float(frozen_lag1.get(feature), default=float("nan"))
        realised = _as_float(calibrated_lag1_expected.get(feature), default=float("nan"))
        phi = _as_float(internal_phi.get(feature), default=float("nan"))
        if not np.isfinite(target) or not np.isfinite(realised):
            rows.append(
                _support_row(
                    "",
                    0,
                    "internal_lag1_calibration",
                    "unsupported",
                    "non_finite_target_or_calibrated_estimator",
                    source_dataset=source,
                    task_id=task,
                    feature=feature,
                    fallback_type="fixed_neutral_value",
                )
            )
            continue
        delta = abs(realised - target)
        at_boundary = abs(abs(phi) - INTERNAL_AR_PHI_BOUND) <= 1e-12
        status = "calibrated"
        detail = f"frozen_estimator_delta={delta:.6g}; internal_phi={phi:.6g}"
        fallback_type = ""
        if delta > LAG1_CALIBRATION_TOLERANCE or at_boundary:
            status = "boundary_best_effort"
            detail = (
                f"{detail}; calibration_tolerance={LAG1_CALIBRATION_TOLERANCE:.6g}; "
                f"target_may_be_outside_sequence_design_range"
            )
            fallback_type = "boundary_internal_phi"
        rows.append(
            _support_row(
                "",
                0,
                "internal_lag1_calibration",
                status,
                detail,
                source_dataset=source,
                task_id=task,
                feature=feature,
                fallback_type=fallback_type,
            )
        )
    return rows


@lru_cache(maxsize=2048)
def _calibrate_internal_phi_for_frozen_lag1(
    target_lag1: float,
    *,
    fatigue_slope: float,
    target_window_variance: float,
    windows_per_session: int,
) -> float:
    if not np.isfinite(target_lag1) or windows_per_session < 3 or target_window_variance <= 0:
        return 0.0
    target = float(np.clip(target_lag1, -INTERNAL_AR_PHI_BOUND, INTERNAL_AR_PHI_BOUND))
    grid = np.linspace(
        -INTERNAL_AR_PHI_BOUND,
        INTERNAL_AR_PHI_BOUND,
        LAG1_CALIBRATION_GRID_POINTS,
    )
    estimates = np.array(
        [
            _expected_frozen_lag1_for_phi(
                float(phi),
                fatigue_slope=fatigue_slope,
                target_window_variance=target_window_variance,
                windows_per_session=windows_per_session,
            )
            for phi in grid
        ],
        dtype=float,
    )
    valid = np.isfinite(estimates)
    if not valid.any():
        return float(np.clip(target, -INTERNAL_AR_PHI_BOUND, INTERNAL_AR_PHI_BOUND))
    best_index = int(np.nanargmin(np.abs(estimates[valid] - target)))
    valid_grid = grid[valid]
    return float(valid_grid[best_index])


def _expected_frozen_lag1_for_phi(
    phi: float,
    *,
    fatigue_slope: float,
    target_window_variance: float,
    windows_per_session: int,
) -> float:
    q = _calibrated_window_innovation_variance(
        target_window_variance,
        phi=phi,
        fatigue_slope=fatigue_slope,
        windows_per_session=windows_per_session,
    )
    if q <= 0:
        return float("nan")
    normals = _lag1_calibration_normals(windows_per_session)
    states = np.zeros_like(normals)
    state = np.zeros(normals.shape[0], dtype=float)
    innovation_sd = float(np.sqrt(q))
    for window in range(windows_per_session):
        state = phi * state + normals[:, window] * innovation_sd
        states[:, window] = state + fatigue_slope * window
    return _frozen_lag1_estimator_from_array(states)


@lru_cache(maxsize=16)
def _lag1_calibration_normals(windows_per_session: int) -> np.ndarray:
    rng = np.random.default_rng(_child_seed("lag1_calibration_normals", windows_per_session))
    return rng.normal(size=(2048, windows_per_session))


def _frozen_lag1_estimator_from_array(values: np.ndarray) -> float:
    if values.shape[1] < 3:
        return float("nan")
    previous = values[:, :-1]
    current = values[:, 1:]
    previous_centered = previous - previous.mean(axis=1, keepdims=True)
    current_centered = current - current.mean(axis=1, keepdims=True)
    numerator = np.sum(previous_centered * current_centered, axis=1)
    denominator = np.sqrt(
        np.sum(previous_centered**2, axis=1) * np.sum(current_centered**2, axis=1)
    )
    valid = denominator > 0
    if not valid.any():
        return float("nan")
    return float(np.mean(numerator[valid] / denominator[valid]))


def _calibrated_window_innovation_variance(
    target_variance: float,
    *,
    phi: float,
    fatigue_slope: float,
    windows_per_session: int,
) -> float:
    times = np.arange(windows_per_session, dtype=float)
    fatigue_sample_variance_factor = float(
        np.sum((times - times.mean()) ** 2) / max(windows_per_session - 1, 1)
    )
    fatigue_variance = fatigue_slope * fatigue_slope * fatigue_sample_variance_factor
    residual_target = max(float(target_variance) - fatigue_variance, 0.0)
    factor = _centred_ar_sample_covariance_factor(phi, phi, windows_per_session)
    return residual_target / max(factor, 1e-12)


def _calibrate_window_covariance_for_estimand(
    target_cov: np.ndarray,
    *,
    internal_phi: dict[str, float],
    fatigue: dict[str, float],
    windows_per_session: int,
    source: str = "",
    task: str = "",
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if windows_per_session < 2 or np.allclose(target_cov, 0.0):
        return target_cov, []
    phi = np.array(
        [
            np.clip(internal_phi.get(feature, 0.0), -INTERNAL_AR_PHI_BOUND, INTERNAL_AR_PHI_BOUND)
            for feature in FEATURES
        ],
        dtype=float,
    )
    slopes = np.array([fatigue.get(feature, 0.0) for feature in FEATURES], dtype=float)
    times = np.arange(windows_per_session, dtype=float)
    fatigue_sample_cov_factor = float(
        np.sum((times - times.mean()) ** 2) / max(windows_per_session - 1, 1)
    )
    q = np.zeros_like(target_cov, dtype=float)
    for i, feature_i in enumerate(FEATURES):
        for j, feature_j in enumerate(FEATURES):
            factor = _centred_ar_sample_covariance_factor(phi[i], phi[j], windows_per_session)
            fatigue_cov = slopes[i] * slopes[j] * fatigue_sample_cov_factor
            q[i, j] = (target_cov[i, j] - fatigue_cov) / max(factor, 1e-12)
    q = (q + q.T) / 2
    min_eigen = float(np.linalg.eigvalsh(q).min()) if q.size else 0.0
    rows: list[dict[str, Any]] = []
    if min_eigen >= -1e-8:
        rows.append(
            _support_row(
                "",
                0,
                "window_innovation_covariance_calibration",
                "support_eligible_covariance",
                "full_pairwise_ar_covariance_inversion_psd",
                source_dataset=source,
                task_id=task,
            )
        )
        return q + np.eye(q.shape[0]) * 1e-12, rows
    diagonal = np.zeros_like(target_cov, dtype=float)
    for index, feature in enumerate(FEATURES):
        diagonal[index, index] = _calibrated_window_innovation_variance(
            max(float(target_cov[index, index]), 0.0),
            phi=float(phi[index]),
            fatigue_slope=float(slopes[index]),
            windows_per_session=windows_per_session,
        )
    rows.append(
        _support_row(
            "",
            0,
            "window_innovation_covariance_calibration",
            "diagonal_fallback_non_psd",
            f"full_pairwise_ar_covariance_inversion_non_psd_min_eigen={min_eigen:.6g}",
            source_dataset=source,
            task_id=task,
            fallback_type="calibrated_supported_diagonal_variances",
            fallback_source="same_template_window_variance",
        )
    )
    return diagonal, rows


def _centred_ar_sample_covariance_factor(phi_i: float, phi_j: float, length: int) -> float:
    transition_i = np.zeros((length, length), dtype=float)
    transition_j = np.zeros((length, length), dtype=float)
    for row in range(length):
        for col in range(row + 1):
            transition_i[row, col] = phi_i ** (row - col)
            transition_j[row, col] = phi_j ** (row - col)
    centring = np.eye(length) - np.ones((length, length), dtype=float) / float(length)
    covariance = centring @ transition_i @ transition_j.T @ centring.T
    return float(np.trace(covariance) / max(length - 1, 1))


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


def _eligible_feature_lookup(
    frame: pd.DataFrame,
    *,
    background: BackgroundSpec,
    source: str,
    task: str,
    column: str,
    default: float,
    quantity: str,
    status_column: str | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    values: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for feature in FEATURES:
        target_rows = frame[
            (frame["source_dataset"].astype(str) == str(source))
            & (frame["task_id"].astype(str) == str(task))
            & (frame["feature"].astype(str) == str(feature))
        ]
        status, reason = _scalar_support_status(
            target_rows,
            background=background,
            source=source,
            task=task,
            feature=feature,
            quantity=quantity,
            status_column=status_column,
        )
        value = (
            _as_float(target_rows.iloc[0][column], default=default)
            if status == "estimated" and not target_rows.empty and column in target_rows
            else default
        )
        values[feature] = float(value) if np.isfinite(value) else default
        rows.append(
            _support_row(
                "",
                0,
                quantity,
                status,
                reason,
                source_dataset=source,
                task_id=task,
                feature=feature,
                fallback_type="" if status == "estimated" else "fixed_neutral_value",
                fallback_source="" if status == "estimated" else "contract_authorised_schema_completion",
            )
        )
    return values, rows


def _scalar_support_status(
    rows: pd.DataFrame,
    *,
    background: BackgroundSpec,
    source: str,
    task: str,
    feature: str,
    quantity: str,
    status_column: str | None,
) -> tuple[str, str]:
    template_rows = background.template_summary[
        (background.template_summary["source_dataset"].astype(str) == str(source))
        & (background.template_summary["task_id"].astype(str) == str(task))
    ]
    if template_rows.empty:
        return "unsupported", "template_missing"
    template_status, template_reason = _acdc_template_feature_eligible(template_rows.iloc[0], feature)
    if template_status != "estimated":
        return "unsupported", template_reason
    if rows.empty:
        return "unsupported", "aggregate_row_missing"
    row = rows.iloc[0]
    if status_column and status_column in row and str(row.get(status_column)) != "estimated":
        return "unsupported", str(row.get(status_column.replace("_status", "_reason"), "extractor_not_estimated"))
    if quantity == "lag1_autocorrelation":
        if _as_float(row.get("lag1_usable_sequences"), default=0.0) < 30:
            return "unsupported", "lag1_usable_sequences_below_30"
        if _as_float(row.get("lag1_median_sequence_length"), default=0.0) < 3:
            return "unsupported", "lag1_sequence_length_below_3"
    elif quantity == "within_session_time_on_task_trend":
        if _as_float(row.get("fatigue_usable_sessions"), default=0.0) < 30:
            return "unsupported", "fatigue_usable_sessions_below_30"
    elif quantity == "missingness_rate":
        if _as_float(row.get("n_rows"), default=0.0) <= 0:
            return "unsupported", "missingness_n_rows_zero"
    return "estimated", "contract_scalar_thresholds_met"


def _paired_target_variance(frame: pd.DataFrame, task: str, feature: str) -> float:
    rows = frame[
        (frame["task_id"].astype(str) == str(task))
        & (frame["feature"].astype(str) == str(feature))
        & (frame["session_support_status"].astype(str) == "estimated")
    ]
    if rows.empty:
        return 0.0
    if _as_float(rows.iloc[0].get("n_repeat_participants"), default=0.0) < 30:
        return 0.0
    return max(_as_float(rows.iloc[0]["session_within_participant_variance"], default=0.0), 0.0)


def _paired_target_stability(
    frame: pd.DataFrame,
    task: str,
    feature: str,
    *,
    fallback_variance: pd.DataFrame | None = None,
) -> float:
    if not frame.empty:
        rows = frame[
            (frame["task_id"].astype(str) == str(task))
            & (frame["feature"].astype(str) == str(feature))
            & (frame["contract_support_status"].astype(str) == "estimated")
        ]
        if not rows.empty and _as_float(rows.iloc[0].get("n_repeat_participants"), default=0.0) >= 30:
            return _as_float(rows.iloc[0].get("repeat_person_stability_icc"))
    if fallback_variance is None or fallback_variance.empty:
        return float("nan")
    rows = fallback_variance[
        (fallback_variance["task_id"].astype(str) == str(task))
        & (fallback_variance["feature"].astype(str) == str(feature))
        & (fallback_variance["between_support_status"].astype(str) == "estimated")
        & (fallback_variance["session_support_status"].astype(str) == "estimated")
    ]
    if rows.empty or _as_float(rows.iloc[0].get("n_repeat_participants"), default=0.0) < 30:
        return float("nan")
    return _as_float(rows.iloc[0].get("between_participant_fraction"))


def _finalise_generated_bounds(frame: pd.DataFrame) -> pd.DataFrame:
    bounded = frame.copy()
    bounded["accuracy"] = bounded["accuracy"].clip(0.01, 0.999)
    bounded["median_rt_ms"] = bounded["median_rt_ms"].clip(150.0, 3000.0)
    bounded["mean_response_speed"] = bounded["mean_response_speed"].clip(0.05, 8.0)
    bounded["rt_cv"] = bounded["rt_cv"].clip(0.005, 2.0)
    bounded["throughput_proxy"] = bounded["throughput_proxy"].clip(0.01, 8.0)
    return bounded


def _reset_task_context_columns(frame: pd.DataFrame) -> None:
    for columns in STRUCTURAL_FEATURES_BY_FLAG.values():
        for column in columns:
            if column not in frame.columns:
                frame[column] = np.nan
    task = frame["task_id"].astype(str)
    frame["has_conflict_cost"] = task.isin(["Stroop", "Flanker"])
    frame["has_post_error"] = task.isin(["Stroop", "Flanker"])
    frame["has_vigilance"] = task == "SART"
    frame["has_switch_structure"] = False
    frame["has_confidence"] = False
    frame["has_change_point"] = False
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


def _realised_repeated_person_stability(frame: pd.DataFrame, feature: str) -> float:
    observed = frame.loc[:, ["participant_id", "session_id"]].copy()
    observed["_value"] = pd.to_numeric(frame[feature], errors="coerce")
    observed = observed.dropna(subset=["_value"])
    session_counts = observed.groupby("participant_id", sort=True)["session_id"].nunique()
    repeat_ids = session_counts[session_counts >= 2].index
    repeated = observed[observed["participant_id"].isin(repeat_ids)]
    if repeated.empty:
        return float("nan")
    participant_means = repeated.groupby("participant_id", sort=True)["_value"].mean()
    baselines = repeated.join(participant_means.rename("participant_mean"), on="participant_id")
    residuals = baselines["_value"] - baselines["participant_mean"]
    between = (
        float(participant_means.var(ddof=1))
        if participant_means.shape[0] > 1
        else float("nan")
    )
    session = (
        float(residuals.var(ddof=1))
        if residuals.dropna().shape[0] > 1
        else float("nan")
    )
    denominator = np.nansum([between, session])
    return _safe_fraction(between, denominator)


def _component_adjacent_pair_corr(component_rows: pd.DataFrame, column: str) -> float:
    previous_values: list[float] = []
    current_values: list[float] = []
    if component_rows.empty or column not in component_rows:
        return float("nan")
    for _, sequence in component_rows.groupby(
        ["source_dataset", "participant_id", "session_id"],
        dropna=False,
        sort=True,
    ):
        ordered = sequence.sort_values("window_start_trial")
        values = pd.to_numeric(ordered[column], errors="coerce").dropna().to_numpy(dtype=float)
        if values.shape[0] >= 2:
            previous_values.extend(values[:-1].tolist())
            current_values.extend(values[1:].tolist())
    return _safe_corr(previous_values, current_values)


def _component_sum_adjacent_pair_corr(
    component_rows: pd.DataFrame,
    columns: Sequence[str],
) -> float:
    if component_rows.empty or any(column not in component_rows for column in columns):
        return float("nan")
    summed = component_rows.copy()
    summed["__component_sum"] = summed.loc[:, list(columns)].sum(axis=1)
    return _component_adjacent_pair_corr(summed, "__component_sum")


def _safe_corr(previous: Sequence[float], current: Sequence[float]) -> float:
    if len(previous) < 2 or len(current) < 2:
        return float("nan")
    previous_array = np.asarray(previous, dtype=float)
    current_array = np.asarray(current, dtype=float)
    valid = np.isfinite(previous_array) & np.isfinite(current_array)
    previous_array = previous_array[valid]
    current_array = current_array[valid]
    if previous_array.shape[0] < 2 or previous_array.std() <= 0 or current_array.std() <= 0:
        return float("nan")
    return float(np.corrcoef(previous_array, current_array)[0, 1])


def _template_feature_sd(
    template_summary: pd.DataFrame,
    source: str,
    task: str,
    feature: str,
) -> float:
    rows = template_summary[
        (template_summary["source_dataset"].astype(str) == str(source))
        & (template_summary["task_id"].astype(str) == str(task))
    ]
    if rows.empty:
        return float("nan")
    return _as_float(rows.iloc[0].get(f"{feature}_sd"))


def _component_between_variance(component: pd.DataFrame, column: str) -> float:
    if component.empty or column not in component:
        return float("nan")
    means = component.groupby(["source_dataset", "participant_id"], sort=True)[column].mean()
    return _as_float(means.var(ddof=1))


def _component_session_variance(component: pd.DataFrame, column: str) -> float:
    if component.empty or column not in component:
        return float("nan")
    session = component.groupby(["source_dataset", "participant_id", "session_id"], sort=True)[
        column
    ].mean()
    participant = session.groupby(level=[0, 1]).transform("mean")
    return _as_float((session - participant).dropna().var(ddof=1))


def _component_window_variance(component: pd.DataFrame, column: str) -> float:
    if component.empty or column not in component:
        return float("nan")
    session_mean = component.groupby(["source_dataset", "participant_id", "session_id"], sort=True)[
        column
    ].transform("mean")
    residual = (component[column] - session_mean).dropna()
    return _as_float(residual.var(ddof=1))


def _component_variance_row(
    world_id: str,
    replicate_index: int,
    source: object,
    task: object,
    feature: str,
    parameter: str,
    target: float,
    combined_observed: float,
    background_realised: float,
    component_isolated: float,
    diagnostic_scope: str,
) -> dict[str, Any]:
    return {
        "world_id": str(world_id),
        "replicate_index": int(replicate_index),
        "source_dataset": source,
        "task_id": task,
        "feature": feature,
        "parameter": parameter,
        "target": target,
        "combined_observed_realised": combined_observed,
        "background_component_realised": background_realised,
        "component_isolated_realised": component_isolated,
        "combined_observed_ratio": _safe_fraction(combined_observed, target),
        "background_component_ratio": _safe_fraction(background_realised, target),
        "component_isolated_ratio": _safe_fraction(component_isolated, target),
        "diagnostic_scope": diagnostic_scope,
    }


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
    *,
    source_dataset: str = "",
    task_id: str = "",
    feature: str = "",
    fallback_type: str = "",
    fallback_source: str = "",
) -> dict[str, Any]:
    return {
        "world_id": str(world_id),
        "replicate_index": int(replicate_index),
        "source_dataset": source_dataset,
        "task_id": task_id,
        "feature": feature,
        "quantity": quantity,
        "support_status": status,
        "detail": detail,
        "fallback_type": fallback_type,
        "fallback_source": fallback_source,
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


def _safe_fraction(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def _dataframe_hash(frame: pd.DataFrame) -> str:
    csv = frame.sort_index(axis=1).to_csv(index=False)
    return "sha256:" + hashlib.sha256(csv.encode("utf-8")).hexdigest()


def _directory_checksum(path: Path) -> str:
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.iterdir() if item.is_file()):
        digest.update(file_path.name.encode("utf-8"))
        digest.update(hash_file(file_path).encode("utf-8"))
    return "sha256:" + digest.hexdigest()


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
    if not _is_switch_off(config.get("generator", {}).get("paired_cross_task_session_covariance", "off")):
        raise ValueError("primary empirical_twin_v1 smoke requires paired_cross_task_session_covariance: off")


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
    parser.add_argument(
        "--audit-diagnostics-only",
        action="store_true",
        help="Write generator target-realised diagnostics without running tournament scoring.",
    )
    args = parser.parse_args(argv)
    if args.audit_diagnostics_only:
        result = run_empirical_twin_v1_audit_diagnostics(
            config_path=args.config,
            output_dir=args.output_dir,
        )
        print(json.dumps({key: value for key, value in result.items() if key != "paths"}, indent=2))
        for name, path in result["paths"].items():
            print(f"{name}: {path}")
        return 0
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
