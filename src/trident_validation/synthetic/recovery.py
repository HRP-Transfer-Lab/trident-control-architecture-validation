"""Synthetic W0-W4 model-recovery and falsification gate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from itertools import combinations, permutations
from math import comb, sqrt
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from trident_validation.models.static_tournament import (
    STATIC_MODEL_IDS,
    build_static_model_suite,
)
from trident_validation.splits import (
    SplitDefinition,
    assert_no_participant_overlap,
    participant_train_test_split,
)
from trident_validation.synthetic.fixtures import CORE_SYNTHETIC_FEATURES
from trident_validation.synthetic.worlds import (
    STATIC_SYNTHETIC_WORLD_IDS,
    WORLD_MODEL_ALIGNMENT,
    StaticWorldId,
    make_static_synthetic_world,
)


GROUND_TRUTH_COLUMN_PREFIXES = ("synthetic_",)
GROUND_TRUTH_COLUMN_MARKERS = ("ground_truth",)
GROUND_TRUTH_COLUMN_STARTS = ("true_", "known_")
NULL_OR_CONTINUOUS_WORLD_IDS = (
    "W0_general_performance",
    "W1_continuous_manifold",
    "W2_nonlinear_vigilance",
)
DISCRETE_MODEL_IDS = (
    "M3_three_profile_mixture",
    "M4_four_pace_profile_mixture",
)
PRIMARY_METRIC = "heldout_log_density_mean_per_window"


@dataclass(frozen=True)
class RecoverySimulationConfig:
    """Configuration for the synthetic model-recovery gate."""

    seed: int = 20260807
    n_replicates: int = 8
    n_stress_replicates: int = 2
    n_datasets: int = 3
    participants_per_dataset: int = 24
    sessions_per_participant: int = 2
    windows_per_session: int = 3
    test_size: float = 0.25
    feature_columns: Sequence[str] = CORE_SYNTHETIC_FEATURES
    practical_equivalence_margin: float = 0.01
    paired_ci_z: float = 1.96

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_columns", tuple(self.feature_columns))
        if self.n_replicates < 1:
            raise ValueError("n_replicates must be positive")
        if self.n_stress_replicates < 0:
            raise ValueError("n_stress_replicates must be non-negative")
        if self.n_datasets < 2:
            raise ValueError("at least two synthetic source datasets are required")
        if self.participants_per_dataset < 1:
            raise ValueError("participants_per_dataset must be positive")
        if self.sessions_per_participant < 1:
            raise ValueError("sessions_per_participant must be positive")
        if self.windows_per_session < 1:
            raise ValueError("windows_per_session must be positive")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1")
        if self.practical_equivalence_margin < 0:
            raise ValueError("practical_equivalence_margin must be non-negative")
        if self.paired_ci_z < 0:
            raise ValueError("paired_ci_z must be non-negative")


@dataclass(frozen=True)
class FrozenModelPredictions:
    """Model outputs generated without access to synthetic truth columns."""

    model_scores: pd.DataFrame
    participant_scores: pd.DataFrame
    probabilities_by_model: dict[str, pd.DataFrame]
    split: SplitDefinition


@dataclass(frozen=True)
class ModelSelection:
    """Registered primary-metric selection with paired uncertainty."""

    selected_model_id: str
    numerical_best_model_id: str
    primary_metric: str
    selection_reason: str
    uncertainty_by_model: pd.DataFrame
    per_window_winner: str
    participant_weighted_winner: str
    per_observed_feature_winner: str
    normalisation_sensitive: bool


@dataclass(frozen=True)
class SyntheticRecoveryOutputs:
    """Machine-readable recovery outputs plus internal audit tables."""

    synthetic_model_recovery: pd.DataFrame
    synthetic_recovery_matrix: pd.DataFrame
    synthetic_false_discrete_rate: pd.DataFrame
    synthetic_stress_recovery: pd.DataFrame
    run_summary: pd.DataFrame
    split_audit: pd.DataFrame


def run_synthetic_model_recovery(
    config: RecoverySimulationConfig | None = None,
) -> SyntheticRecoveryOutputs:
    """Run repeated W0-W4 synthetic recovery and stress tests."""

    config = config or RecoverySimulationConfig()
    model_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []

    baseline_kwargs = _baseline_world_kwargs(config)
    for world_id in STATIC_SYNTHETIC_WORLD_IDS:
        for replicate_index in range(config.n_replicates):
            run = _run_one_recovery_dataset(
                world_id=world_id,
                replicate_index=replicate_index,
                analysis_set="baseline",
                stress_family="none",
                stress_level="baseline",
                stress_value=1.0,
                world_kwargs=baseline_kwargs,
                config=config,
            )
            model_rows.extend(run["model_rows"])
            run_rows.append(run["run_row"])
            split_rows.append(run["split_row"])

    if config.n_stress_replicates:
        for scenario in _stress_scenarios(config):
            for world_id in STATIC_SYNTHETIC_WORLD_IDS:
                for replicate_index in range(config.n_stress_replicates):
                    run = _run_one_recovery_dataset(
                        world_id=world_id,
                        replicate_index=replicate_index,
                        analysis_set="stress",
                        stress_family=scenario["stress_family"],
                        stress_level=scenario["stress_level"],
                        stress_value=float(scenario["stress_value"]),
                        world_kwargs=scenario["world_kwargs"],
                        config=config,
                    )
                    model_rows.extend(run["model_rows"])
                    run_rows.append(run["run_row"])
                    split_rows.append(run["split_row"])

    model_recovery = pd.DataFrame(model_rows)
    run_summary = pd.DataFrame(run_rows)
    split_audit = pd.DataFrame(split_rows)
    recovery_matrix = build_recovery_matrix(run_summary)
    false_discrete_rate = compute_false_discrete_rate(run_summary)
    stress_recovery = build_stress_recovery(run_summary)
    return SyntheticRecoveryOutputs(
        synthetic_model_recovery=model_recovery,
        synthetic_recovery_matrix=recovery_matrix,
        synthetic_false_discrete_rate=false_discrete_rate,
        synthetic_stress_recovery=stress_recovery,
        run_summary=run_summary,
        split_audit=split_audit,
    )


def fit_score_models_without_truth(
    frame_without_truth: pd.DataFrame,
    split: SplitDefinition,
    *,
    feature_columns: Sequence[str],
    random_state: int,
) -> FrozenModelPredictions:
    """Fit M0-M4 and freeze predictions on a truth-stripped frame."""

    assert_no_ground_truth_columns(frame_without_truth)
    assert_no_participant_overlap(frame_without_truth, split)
    train = frame_without_truth.loc[list(split.train_indices)].copy()
    test = frame_without_truth.loc[list(split.test_indices)].copy()
    assert_no_ground_truth_columns(train)
    assert_no_ground_truth_columns(test)

    model_rows: list[dict[str, Any]] = []
    participant_series: dict[str, pd.Series] = {}
    probabilities_by_model: dict[str, pd.DataFrame] = {}
    observed_feature_counts = _observed_feature_counts(test, feature_columns)
    total_observed_features = int(observed_feature_counts.sum())

    for model in build_static_model_suite(
        feature_columns=feature_columns,
        random_state=random_state,
    ):
        fitted = model.fit(train)
        sample_scores = fitted.score_samples(test).astype(float)
        holdout = fitted.score_holdout(test)
        valid_scores = sample_scores.dropna()
        total_log_density = float(valid_scores.sum()) if not valid_scores.empty else float("nan")
        participant_means = _participant_log_density_means(test, sample_scores)
        participant_series[fitted.model_id] = participant_means
        per_feature = (
            float(total_log_density / total_observed_features)
            if total_observed_features and np.isfinite(total_log_density)
            else float("nan")
        )
        probabilities = fitted.predict_proba(test)
        if probabilities is not None:
            probabilities_by_model[fitted.model_id] = probabilities.copy(deep=True)

        diagnostics = holdout.diagnostics or {}
        model_rows.append(
            {
                "model_id": fitted.model_id,
                "primary_metric": holdout.primary_metric or PRIMARY_METRIC,
                "heldout_log_likelihood_total": total_log_density,
                "heldout_log_density_mean_per_window": float(valid_scores.mean()),
                "heldout_log_density_participant_weighted": float(participant_means.mean()),
                "heldout_log_density_mean_per_observed_feature": per_feature,
                "n_valid_windows": int(valid_scores.shape[0]),
                "n_test_participants": int(participant_means.shape[0]),
                "n_observed_feature_values": total_observed_features,
                **{f"diagnostic_{key}": float(value) for key, value in diagnostics.items()},
            }
        )

    participant_scores = pd.concat(participant_series, axis=1)
    return FrozenModelPredictions(
        model_scores=pd.DataFrame(model_rows),
        participant_scores=participant_scores,
        probabilities_by_model=probabilities_by_model,
        split=split,
    )


def select_preferred_model(
    model_scores: pd.DataFrame,
    participant_scores: pd.DataFrame,
    *,
    practical_equivalence_margin: float = 0.01,
    paired_ci_z: float = 1.96,
) -> ModelSelection:
    """Select the simplest distinguishable model under the primary metric."""

    assert_no_ground_truth_columns(model_scores)
    if model_scores.empty:
        raise ValueError("model_scores must not be empty")
    scores = model_scores.copy()
    scores["complexity_rank"] = scores["model_id"].map(_model_complexity_rank)
    valid = scores[np.isfinite(scores[PRIMARY_METRIC])].copy()
    if valid.empty:
        raise ValueError("no finite primary model scores are available")
    valid = valid.sort_values(
        [PRIMARY_METRIC, "complexity_rank"],
        ascending=[False, True],
        kind="mergesort",
    )
    numerical_best = str(valid.iloc[0]["model_id"])
    uncertainty = _paired_uncertainty_against_best(
        participant_scores,
        best_model_id=numerical_best,
        practical_equivalence_margin=practical_equivalence_margin,
        paired_ci_z=paired_ci_z,
    )
    uncertainty_by_model = uncertainty.set_index("model_id", drop=False)

    selected_model_id = numerical_best
    selection_reason = "numerical_best"
    valid_model_ids = set(valid["model_id"])
    for model_id in STATIC_MODEL_IDS:
        if model_id not in valid_model_ids:
            continue
        row = uncertainty_by_model.loc[model_id]
        if not bool(row["meaningfully_worse_than_numerical_best"]):
            selected_model_id = model_id
            if model_id == numerical_best:
                selection_reason = "numerical_best"
            else:
                selection_reason = "simpler_model_not_meaningfully_distinguishable"
            break

    winners = {
        "per_window": _metric_winner(scores, PRIMARY_METRIC),
        "participant_weighted": _metric_winner(
            scores,
            "heldout_log_density_participant_weighted",
        ),
        "per_observed_feature": _metric_winner(
            scores,
            "heldout_log_density_mean_per_observed_feature",
        ),
    }
    normalisation_sensitive = len(set(winners.values())) > 1
    return ModelSelection(
        selected_model_id=selected_model_id,
        numerical_best_model_id=numerical_best,
        primary_metric=PRIMARY_METRIC,
        selection_reason=selection_reason,
        uncertainty_by_model=uncertainty,
        per_window_winner=winners["per_window"],
        participant_weighted_winner=winners["participant_weighted"],
        per_observed_feature_winner=winners["per_observed_feature"],
        normalisation_sensitive=normalisation_sensitive,
    )


def build_recovery_matrix(run_summary: pd.DataFrame) -> pd.DataFrame:
    """Build a long-form 5 by 5 true-world by selected-model matrix."""

    baseline = run_summary[run_summary["analysis_set"] == "baseline"].copy()
    rows: list[dict[str, Any]] = []
    for world_id in STATIC_SYNTHETIC_WORLD_IDS:
        world_rows = baseline[baseline["true_world_id"] == world_id]
        n_runs = int(world_rows.shape[0])
        aligned_model_id = WORLD_MODEL_ALIGNMENT[world_id]
        correct_rate = (
            float(world_rows["correct_model_selected"].mean())
            if n_runs
            else float("nan")
        )
        for model_id in STATIC_MODEL_IDS:
            selected_count = int((world_rows["selected_model_id"] == model_id).sum())
            rows.append(
                {
                    "true_world_id": world_id,
                    "aligned_model_id": aligned_model_id,
                    "selected_model_id": model_id,
                    "n_runs": n_runs,
                    "n_selected": selected_count,
                    "selection_rate": selected_count / n_runs if n_runs else float("nan"),
                    "is_correct_cell": model_id == aligned_model_id,
                    "correct_model_recovery_rate": correct_rate,
                }
            )
    return pd.DataFrame(rows)


def compute_false_discrete_rate(run_summary: pd.DataFrame) -> pd.DataFrame:
    """Compute P(M3 or M4 selected | W0/W1/W2)."""

    rows: list[dict[str, Any]] = []
    for group_keys, group in run_summary.groupby(
        ["analysis_set", "stress_family", "stress_level"],
        dropna=False,
        sort=True,
    ):
        analysis_set, stress_family, stress_level = group_keys
        null_rows = group[group["true_world_id"].isin(NULL_OR_CONTINUOUS_WORLD_IDS)]
        rows.append(
            _false_discrete_row(
                null_rows,
                analysis_set=str(analysis_set),
                stress_family=str(stress_family),
                stress_level=str(stress_level),
                true_world_id="W0_W1_W2_combined",
            )
        )
        for world_id in NULL_OR_CONTINUOUS_WORLD_IDS:
            world_rows = null_rows[null_rows["true_world_id"] == world_id]
            rows.append(
                _false_discrete_row(
                    world_rows,
                    analysis_set=str(analysis_set),
                    stress_family=str(stress_family),
                    stress_level=str(stress_level),
                    true_world_id=world_id,
                )
            )
    return pd.DataFrame(rows)


def build_stress_recovery(run_summary: pd.DataFrame) -> pd.DataFrame:
    """Summarise recovery as a function of each stress condition."""

    stress = run_summary[run_summary["analysis_set"] == "stress"].copy()
    rows: list[dict[str, Any]] = []
    if stress.empty:
        return pd.DataFrame(
            columns=[
                "stress_family",
                "stress_level",
                "stress_value",
                "true_world_id",
                "aligned_model_id",
                "selected_model_id",
                "n_runs",
                "n_selected",
                "selection_rate",
                "correct_model_recovery_rate",
                "false_discrete_rate_for_null_world",
                "normalisation_sensitive_rate",
            ]
        )
    for group_keys, world_rows in stress.groupby(
        ["stress_family", "stress_level", "stress_value", "true_world_id"],
        sort=True,
        dropna=False,
    ):
        stress_family, stress_level, stress_value, world_id = group_keys
        n_runs = int(world_rows.shape[0])
        correct_rate = float(world_rows["correct_model_selected"].mean())
        normalisation_rate = float(world_rows["normalisation_sensitive"].mean())
        if world_id in NULL_OR_CONTINUOUS_WORLD_IDS:
            false_rate = float(world_rows["selected_model_id"].isin(DISCRETE_MODEL_IDS).mean())
        else:
            false_rate = float("nan")
        for model_id in STATIC_MODEL_IDS:
            selected_count = int((world_rows["selected_model_id"] == model_id).sum())
            rows.append(
                {
                    "stress_family": stress_family,
                    "stress_level": stress_level,
                    "stress_value": float(stress_value),
                    "true_world_id": world_id,
                    "aligned_model_id": WORLD_MODEL_ALIGNMENT[str(world_id)],
                    "selected_model_id": model_id,
                    "n_runs": n_runs,
                    "n_selected": selected_count,
                    "selection_rate": selected_count / n_runs if n_runs else float("nan"),
                    "correct_model_recovery_rate": correct_rate,
                    "false_discrete_rate_for_null_world": false_rate,
                    "normalisation_sensitive_rate": normalisation_rate,
                }
            )
    return pd.DataFrame(rows)


def ground_truth_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    """Return columns that must be hidden during fitting and selection."""

    columns: list[str] = []
    for column in frame.columns:
        lowered = str(column).lower()
        if lowered.startswith(GROUND_TRUTH_COLUMN_PREFIXES):
            columns.append(column)
        elif lowered.startswith(GROUND_TRUTH_COLUMN_STARTS):
            columns.append(column)
        elif any(marker in lowered for marker in GROUND_TRUTH_COLUMN_MARKERS):
            columns.append(column)
    return tuple(columns)


def strip_ground_truth_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove synthetic truth columns before any model-facing operation."""

    columns = ground_truth_columns(frame)
    if not columns:
        return frame.copy()
    return frame.drop(columns=list(columns)).copy()


def assert_no_ground_truth_columns(frame: pd.DataFrame) -> None:
    """Fail if a model-facing table exposes known synthetic truth."""

    leaked = ground_truth_columns(frame)
    if leaked:
        raise ValueError(
            "ground-truth columns are not allowed before prediction freeze: "
            + ", ".join(leaked)
        )


def best_label_alignment(
    true_labels: Sequence[object],
    predicted_labels: Sequence[object],
) -> dict[str, Any]:
    """Match neutral predicted labels to truth labels without assuming names."""

    truth = [str(value) for value in true_labels]
    predicted = [str(value) for value in predicted_labels]
    if len(truth) != len(predicted):
        raise ValueError("true and predicted labels must have the same length")
    if not truth:
        return {
            "mapping": {},
            "matched_accuracy": float("nan"),
            "n_true_components": 0,
            "n_predicted_components": 0,
            "unmatched_predicted_components": 0,
        }
    true_unique = sorted(set(truth))
    pred_unique = sorted(set(predicted))
    best_mapping: dict[str, str] = {}
    best_matches = -1
    if len(pred_unique) <= len(true_unique):
        for true_choice in permutations(true_unique, len(pred_unique)):
            mapping = dict(zip(pred_unique, true_choice, strict=True))
            matches = sum(mapping[pred] == actual for actual, pred in zip(truth, predicted, strict=True))
            if matches > best_matches:
                best_matches = matches
                best_mapping = mapping
    else:
        for pred_subset in combinations(pred_unique, len(true_unique)):
            for true_choice in permutations(true_unique):
                mapping = dict(zip(pred_subset, true_choice, strict=True))
                matches = sum(
                    mapping.get(pred) == actual
                    for actual, pred in zip(truth, predicted, strict=True)
                )
                if matches > best_matches:
                    best_matches = matches
                    best_mapping = mapping
    return {
        "mapping": best_mapping,
        "matched_accuracy": best_matches / len(truth),
        "n_true_components": len(true_unique),
        "n_predicted_components": len(pred_unique),
        "unmatched_predicted_components": len(set(pred_unique).difference(best_mapping)),
    }


def adjusted_rand_index(
    true_labels: Sequence[object],
    predicted_labels: Sequence[object],
) -> float:
    """Compute ARI without depending on external clustering packages."""

    truth = [str(value) for value in true_labels]
    predicted = [str(value) for value in predicted_labels]
    if len(truth) != len(predicted):
        raise ValueError("true and predicted labels must have the same length")
    n = len(truth)
    if n < 2:
        return float("nan")
    contingency: dict[tuple[str, str], int] = {}
    true_counts: dict[str, int] = {}
    pred_counts: dict[str, int] = {}
    for actual, pred in zip(truth, predicted, strict=True):
        contingency[(actual, pred)] = contingency.get((actual, pred), 0) + 1
        true_counts[actual] = true_counts.get(actual, 0) + 1
        pred_counts[pred] = pred_counts.get(pred, 0) + 1

    sum_comb = sum(comb(value, 2) for value in contingency.values())
    sum_true = sum(comb(value, 2) for value in true_counts.values())
    sum_pred = sum(comb(value, 2) for value in pred_counts.values())
    total_pairs = comb(n, 2)
    expected = (sum_true * sum_pred / total_pairs) if total_pairs else 0.0
    maximum = 0.5 * (sum_true + sum_pred)
    denominator = maximum - expected
    if denominator == 0:
        return 1.0 if sum_comb == maximum else 0.0
    return float((sum_comb - expected) / denominator)


def write_synthetic_recovery_outputs(
    outputs: SyntheticRecoveryOutputs,
    *,
    output_dir: str | Path = "reports",
    config: RecoverySimulationConfig | None = None,
) -> dict[str, Path]:
    """Write CSV outputs and the Milestone 2.5 report."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "synthetic_model_recovery": output_path / "synthetic_model_recovery.csv",
        "synthetic_recovery_matrix": output_path / "synthetic_recovery_matrix.csv",
        "synthetic_false_discrete_rate": output_path / "synthetic_false_discrete_rate.csv",
        "synthetic_stress_recovery": output_path / "synthetic_stress_recovery.csv",
        "report": output_path / "milestone_2_5_synthetic_recovery.md",
    }
    outputs.synthetic_model_recovery.to_csv(paths["synthetic_model_recovery"], index=False)
    outputs.synthetic_recovery_matrix.to_csv(paths["synthetic_recovery_matrix"], index=False)
    outputs.synthetic_false_discrete_rate.to_csv(paths["synthetic_false_discrete_rate"], index=False)
    outputs.synthetic_stress_recovery.to_csv(paths["synthetic_stress_recovery"], index=False)
    paths["report"].write_text(
        render_synthetic_recovery_report(outputs, config=config),
        encoding="utf-8",
    )
    return paths


def render_synthetic_recovery_report(
    outputs: SyntheticRecoveryOutputs,
    *,
    config: RecoverySimulationConfig | None = None,
) -> str:
    """Render a concise, frozen synthetic recovery report."""

    config = config or RecoverySimulationConfig()
    baseline = outputs.run_summary[outputs.run_summary["analysis_set"] == "baseline"].copy()
    false_rows = outputs.synthetic_false_discrete_rate
    false_overall = false_rows[
        (false_rows["analysis_set"] == "baseline")
        & (false_rows["true_world_id"] == "W0_W1_W2_combined")
    ]
    false_rate = (
        float(false_overall.iloc[0]["false_discrete_rate"])
        if not false_overall.empty
        else float("nan")
    )
    correct_rows = []
    for world_id, group in baseline.groupby("true_world_id", sort=True):
        selected_counts = group["selected_model_id"].value_counts().sort_index()
        selected_distribution = "; ".join(
            f"{model_id}={int(count)}" for model_id, count in selected_counts.items()
        )
        correct_rows.append(
            [
                world_id,
                WORLD_MODEL_ALIGNMENT[str(world_id)],
                f"{float(group['correct_model_selected'].mean()):.3f}",
                selected_distribution,
            ]
        )

    matrix_counts = outputs.synthetic_recovery_matrix.pivot(
        index="true_world_id",
        columns="selected_model_id",
        values="n_selected",
    ).fillna(0)
    matrix_rows = [
        [world_id, *[str(int(matrix_counts.loc[world_id, model_id])) for model_id in STATIC_MODEL_IDS]]
        for world_id in STATIC_SYNTHETIC_WORLD_IDS
    ]

    component = outputs.synthetic_model_recovery[
        outputs.synthetic_model_recovery["component_ari"].notna()
    ].copy()
    component_rows = []
    if not component.empty:
        grouped = component.groupby(["true_world_id", "model_id"], sort=True)
        for (world_id, model_id), group in grouped:
            component_rows.append(
                [
                    world_id,
                    model_id,
                    f"{float(group['component_ari'].mean()):.3f}",
                    f"{float(group['posterior_entropy_mean'].mean()):.3f}",
                    f"{float(group['component_size_l1'].mean()):.3f}",
                ]
            )

    stress_rows = []
    stress = outputs.synthetic_stress_recovery.copy()
    if not stress.empty:
        stress_world = stress.drop_duplicates(
            ["stress_family", "stress_level", "true_world_id"]
        )
        grouped = stress_world.groupby(["stress_family", "stress_level", "stress_value"], sort=True)
        for (family, level, value), group in grouped:
            stress_rows.append(
                [
                    family,
                    level,
                    f"{float(value):.3f}",
                    f"{float(group['correct_model_recovery_rate'].mean()):.3f}",
                    f"{float(group['normalisation_sensitive_rate'].mean()):.3f}",
                ]
            )

    normalisation_sensitive_count = int(baseline["normalisation_sensitive"].sum())
    baseline_runs = int(baseline.shape[0])
    poor_worlds = [
        row[0]
        for row in correct_rows
        if row[2] != "nan" and float(row[2]) < 0.5
    ]
    if poor_worlds:
        failure_mode = (
            "Recovery is weak for "
            + ", ".join(poor_worlds)
            + ". The gate records these as failures or ambiguities rather than "
            "retuning the synthetic worlds."
        )
    else:
        failure_mode = (
            "No baseline true world fell below 0.50 correct recovery in this run. "
            "This is a reporting result, not a tuning target."
        )

    lines = [
        "# Milestone 2.5 Synthetic Model-Recovery and Falsification Gate",
        "",
        "## Scope",
        "",
        "This run uses only fully synthetic W0-W4 worlds and the existing M0-M4 static tournament. It does not use public datasets, M5, cPCA, CVQ, CVAE, HMMs, graph models or Attention Coach transport.",
        "",
        "Model fitting, scoring, participant splitting and model selection operate on a frame with all synthetic truth columns removed. Truth columns are joined back only after model scores, posterior probabilities and the selected model are frozen for evaluation.",
        "",
        "The registered primary metric is held-out log density per window. When the participant-level paired comparison cannot meaningfully distinguish a more complex numerical winner from a simpler candidate, the simpler model is retained.",
        "",
        "## Run Configuration",
        "",
        _markdown_table(
            ["Field", "Value"],
            [
                ["seed", str(config.seed)],
                ["baseline_replicates_per_world", str(config.n_replicates)],
                ["stress_replicates_per_world_level", str(config.n_stress_replicates)],
                ["participants_per_dataset", str(config.participants_per_dataset)],
                ["sessions_per_participant", str(config.sessions_per_participant)],
                ["windows_per_session", str(config.windows_per_session)],
                ["test_size", f"{config.test_size:.3f}"],
                ["practical_equivalence_margin", f"{config.practical_equivalence_margin:.3f}"],
            ],
        ),
        "",
        "## Baseline Recovery Matrix",
        "",
        _markdown_table(["True world", *STATIC_MODEL_IDS], matrix_rows),
        "",
        "## Correct Recovery Rates",
        "",
        _markdown_table(
            ["True world", "Aligned model", "Correct recovery rate", "Selected-model counts"],
            correct_rows,
        ),
        "",
        "## Falsification Check",
        "",
        f"False discrete-structure rate P(M3 or M4 selected | W0/W1/W2): {false_rate:.3f}.",
        "",
        f"Normalisation-sensitive baseline selections: {normalisation_sensitive_count} of {baseline_runs}. These cases are flagged in the CSV outputs rather than silently resolved.",
        "",
        "## W3/W4 Component Recovery",
        "",
        _markdown_table(
            ["True world", "Model", "Mean ARI", "Mean posterior entropy", "Mean component-size L1"],
            component_rows
            or [["not_applicable", "not_applicable", "nan", "nan", "nan"]],
        ),
        "",
        "Recovered components are reported only with neutral component identifiers.",
        "",
        "## Stress Recovery",
        "",
        _markdown_table(
            ["Stress family", "Level", "Value", "Mean correct recovery", "Normalisation-sensitive rate"],
            stress_rows
            or [["not_run", "not_run", "nan", "nan", "nan"]],
        ),
        "",
        "## Failure Mode",
        "",
        failure_mode,
        "",
        "## Machine-Readable Outputs",
        "",
        "- `reports/synthetic_model_recovery.csv`",
        "- `reports/synthetic_recovery_matrix.csv`",
        "- `reports/synthetic_false_discrete_rate.csv`",
        "- `reports/synthetic_stress_recovery.csv`",
    ]
    return "\n".join(lines) + "\n"


def _run_one_recovery_dataset(
    *,
    world_id: StaticWorldId,
    replicate_index: int,
    analysis_set: str,
    stress_family: str,
    stress_level: str,
    stress_value: float,
    world_kwargs: dict[str, Any],
    config: RecoverySimulationConfig,
) -> dict[str, Any]:
    run_id = _run_id(
        config.seed,
        analysis_set,
        stress_family,
        stress_level,
        world_id,
        replicate_index,
    )
    dataset_seed = _child_seed(run_id, "dataset")
    split_seed = _child_seed(run_id, "split")
    model_seed = _child_seed(run_id, "model")
    truth_frame = make_static_synthetic_world(
        world_id,
        seed=dataset_seed,
        **world_kwargs,
    )
    model_frame = strip_ground_truth_columns(truth_frame)
    assert_no_ground_truth_columns(model_frame)
    split = participant_train_test_split(
        model_frame,
        test_size=config.test_size,
        seed=split_seed,
    )
    frozen = fit_score_models_without_truth(
        model_frame,
        split,
        feature_columns=config.feature_columns,
        random_state=model_seed,
    )
    selection = select_preferred_model(
        frozen.model_scores,
        frozen.participant_scores,
        practical_equivalence_margin=config.practical_equivalence_margin,
        paired_ci_z=config.paired_ci_z,
    )

    # Synthetic truth is revealed only after model scores and predictions above are frozen.
    truth = _truth_for_test_rows(truth_frame, split)
    component_metrics = _component_recovery_after_freeze(
        frozen.probabilities_by_model,
        truth,
    )
    aligned_model_id = WORLD_MODEL_ALIGNMENT[world_id]
    run_common = {
        "run_id": run_id,
        "analysis_set": analysis_set,
        "stress_family": stress_family,
        "stress_level": stress_level,
        "stress_value": stress_value,
        "replicate_index": replicate_index,
        "dataset_seed": dataset_seed,
        "split_seed": split_seed,
        "model_seed": model_seed,
        "true_world_id": world_id,
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
        "participants_per_dataset": int(world_kwargs["participants_per_dataset"]),
        "sessions_per_participant": int(world_kwargs["sessions_per_participant"]),
        "min_windows_per_session": int(world_kwargs["min_windows_per_session"]),
        "max_windows_per_session": int(world_kwargs["max_windows_per_session"]),
        "observation_noise_scale": float(world_kwargs["observation_noise_scale"]),
        "source_shift_scale": float(world_kwargs["source_shift_scale"]),
        "technical_missingness_rate": float(world_kwargs["technical_missingness_rate"]),
        "latent_separation_scale": float(world_kwargs["latent_separation_scale"]),
    }
    uncertainty = selection.uncertainty_by_model.set_index("model_id", drop=False)
    model_rows = []
    for _, row in frozen.model_scores.iterrows():
        model_id = str(row["model_id"])
        row_dict = {
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
            row_dict.update(
                {
                    key: uncertainty.loc[model_id, key]
                    for key in uncertainty.columns
                    if key != "model_id"
                }
            )
        row_dict.update(_empty_component_metrics())
        if model_id in component_metrics:
            row_dict.update(component_metrics[model_id])
        model_rows.append(row_dict)

    split_row = {
        "run_id": run_id,
        "analysis_set": analysis_set,
        "stress_family": stress_family,
        "stress_level": stress_level,
        "true_world_id": world_id,
        **_split_audit(model_frame, split),
    }
    return {
        "model_rows": model_rows,
        "run_row": run_common,
        "split_row": split_row,
    }


def _baseline_world_kwargs(config: RecoverySimulationConfig) -> dict[str, Any]:
    return {
        "n_datasets": config.n_datasets,
        "participants_per_dataset": config.participants_per_dataset,
        "sessions_per_participant": config.sessions_per_participant,
        "min_windows_per_session": config.windows_per_session,
        "max_windows_per_session": config.windows_per_session,
        "observation_noise_scale": 1.0,
        "source_shift_scale": 1.0,
        "technical_missingness_rate": 0.01,
        "latent_separation_scale": 1.0,
    }


def _stress_scenarios(config: RecoverySimulationConfig) -> list[dict[str, Any]]:
    base = _baseline_world_kwargs(config)
    scenarios: list[dict[str, Any]] = []

    for value in (12, 24, 48):
        kwargs = {**base, "participants_per_dataset": value}
        scenarios.append(_stress_scenario("participant_count", str(value), value, kwargs))

    for value in (2, 4, 8):
        kwargs = {
            **base,
            "sessions_per_participant": 1,
            "min_windows_per_session": value,
            "max_windows_per_session": value,
        }
        scenarios.append(_stress_scenario("windows_per_participant", str(value), value, kwargs))

    for value in (0.75, 1.0, 1.5, 2.0):
        kwargs = {**base, "observation_noise_scale": value}
        scenarios.append(_stress_scenario("observation_noise", str(value), value, kwargs))

    for value in (0.0, 1.0, 2.0):
        kwargs = {**base, "source_shift_scale": value}
        scenarios.append(_stress_scenario("source_dataset_shift", str(value), value, kwargs))

    for value in (0.0, 0.05, 0.15):
        kwargs = {**base, "technical_missingness_rate": value}
        scenarios.append(_stress_scenario("missingness", str(value), value, kwargs))

    for value in (0.6, 1.0, 1.4):
        kwargs = {**base, "latent_separation_scale": value}
        scenarios.append(_stress_scenario("latent_profile_separation", str(value), value, kwargs))

    return scenarios


def _stress_scenario(
    family: str,
    level: str,
    value: float,
    world_kwargs: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stress_family": family,
        "stress_level": level,
        "stress_value": float(value),
        "world_kwargs": world_kwargs,
    }


def _paired_uncertainty_against_best(
    participant_scores: pd.DataFrame,
    *,
    best_model_id: str,
    practical_equivalence_margin: float,
    paired_ci_z: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id in STATIC_MODEL_IDS:
        if model_id not in participant_scores.columns:
            continue
        diffs = participant_scores[best_model_id] - participant_scores[model_id]
        diffs = diffs.dropna()
        n_pairs = int(diffs.shape[0])
        mean_delta = float(diffs.mean()) if n_pairs else float("nan")
        sd_delta = float(diffs.std(ddof=1)) if n_pairs > 1 else float("nan")
        se_delta = sd_delta / sqrt(n_pairs) if n_pairs > 1 else float("nan")
        ci_low = mean_delta - paired_ci_z * se_delta if n_pairs > 1 else float("nan")
        ci_high = mean_delta + paired_ci_z * se_delta if n_pairs > 1 else float("nan")
        meaningfully_worse = (
            model_id != best_model_id
            and n_pairs > 1
            and mean_delta > practical_equivalence_margin
            and ci_low > 0
        )
        rows.append(
            {
                "model_id": model_id,
                "paired_n_participants": n_pairs,
                "paired_delta_best_minus_model": mean_delta,
                "paired_delta_se": se_delta,
                "paired_delta_ci_low": ci_low,
                "paired_delta_ci_high": ci_high,
                "practical_equivalence_margin": practical_equivalence_margin,
                "meaningfully_worse_than_numerical_best": bool(meaningfully_worse),
            }
        )
    return pd.DataFrame(rows)


def _metric_winner(model_scores: pd.DataFrame, metric: str) -> str:
    ranked = model_scores.copy()
    ranked["complexity_rank"] = ranked["model_id"].map(_model_complexity_rank)
    ranked = ranked[np.isfinite(ranked[metric])].sort_values(
        [metric, "complexity_rank"],
        ascending=[False, True],
        kind="mergesort",
    )
    if ranked.empty:
        return "not_available"
    return str(ranked.iloc[0]["model_id"])


def _participant_log_density_means(
    test_frame: pd.DataFrame,
    sample_scores: pd.Series,
) -> pd.Series:
    table = test_frame.loc[:, ["source_dataset", "participant_id"]].copy()
    table["heldout_log_density"] = sample_scores.reindex(test_frame.index).to_numpy(dtype=float)
    table = table.dropna(subset=["heldout_log_density"])
    return table.groupby(["source_dataset", "participant_id"], sort=True)[
        "heldout_log_density"
    ].mean()


def _observed_feature_counts(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
) -> pd.Series:
    return frame.loc[:, list(feature_columns)].notna().sum(axis=1)


def _truth_for_test_rows(
    truth_frame: pd.DataFrame,
    split: SplitDefinition,
) -> pd.DataFrame:
    columns = [
        "synthetic_world_id",
        "synthetic_aligned_model_id",
        "synthetic_component_id",
    ]
    return truth_frame.loc[list(split.test_indices), columns].copy()


def _component_recovery_after_freeze(
    probabilities_by_model: dict[str, pd.DataFrame],
    truth: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    if "synthetic_component_id" not in truth:
        return {}
    truth_labels = truth["synthetic_component_id"].astype(str)
    valid_truth = truth_labels != "not_applicable"
    if not valid_truth.any():
        return {}

    results: dict[str, dict[str, Any]] = {}
    for model_id, probabilities in probabilities_by_model.items():
        if model_id not in DISCRETE_MODEL_IDS:
            continue
        aligned_probabilities = probabilities.loc[truth.index]
        hard_indices = aligned_probabilities.to_numpy(dtype=float).argmax(axis=1)
        predicted_labels = pd.Series(
            [f"component_{int(index)}" for index in hard_indices],
            index=truth.index,
        )
        valid_index = valid_truth & predicted_labels.notna()
        true_values = truth_labels.loc[valid_index].tolist()
        predicted_values = predicted_labels.loc[valid_index].tolist()
        probabilities_valid = aligned_probabilities.loc[valid_index].to_numpy(dtype=float)
        match = best_label_alignment(true_values, predicted_values)
        ari = adjusted_rand_index(true_values, predicted_values)
        entropy = _posterior_entropy(probabilities_valid)
        size_l1, size_score = _component_size_recovery(true_values, predicted_values, match["mapping"])
        results[model_id] = {
            "component_ari": ari,
            "component_matching_accuracy": float(match["matched_accuracy"]),
            "component_label_mapping": _format_label_mapping(match["mapping"]),
            "n_true_components": int(match["n_true_components"]),
            "n_predicted_components": int(match["n_predicted_components"]),
            "unmatched_predicted_components": int(match["unmatched_predicted_components"]),
            "posterior_entropy_mean": entropy["mean_entropy"],
            "posterior_entropy_normalised": entropy["normalised_entropy"],
            "component_size_l1": size_l1,
            "component_size_recovery_score": size_score,
        }
    return results


def _component_size_recovery(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    mapping: dict[str, str],
) -> tuple[float, float]:
    n = len(true_labels)
    if not n:
        return float("nan"), float("nan")
    true_counts = pd.Series(true_labels, dtype="object").value_counts(normalize=True)
    mapped_predictions = [
        mapping.get(label, "__unmatched_predicted_component__")
        for label in predicted_labels
    ]
    predicted_counts = pd.Series(mapped_predictions, dtype="object").value_counts(normalize=True)
    l1 = 0.0
    for label, true_proportion in true_counts.items():
        l1 += abs(float(predicted_counts.get(label, 0.0)) - float(true_proportion))
    l1 += float(predicted_counts.get("__unmatched_predicted_component__", 0.0))
    return float(l1), float(max(0.0, 1.0 - l1 / 2.0))


def _posterior_entropy(probabilities: np.ndarray) -> dict[str, float]:
    if probabilities.size == 0:
        return {"mean_entropy": float("nan"), "normalised_entropy": float("nan")}
    clipped = np.clip(probabilities, 1e-300, 1.0)
    entropy_by_row = -np.sum(clipped * np.log(clipped), axis=1)
    normaliser = np.log(probabilities.shape[1]) if probabilities.shape[1] > 1 else 1.0
    return {
        "mean_entropy": float(np.mean(entropy_by_row)),
        "normalised_entropy": float(np.mean(entropy_by_row) / normaliser),
    }


def _empty_component_metrics() -> dict[str, Any]:
    return {
        "component_ari": float("nan"),
        "component_matching_accuracy": float("nan"),
        "component_label_mapping": "",
        "n_true_components": float("nan"),
        "n_predicted_components": float("nan"),
        "unmatched_predicted_components": float("nan"),
        "posterior_entropy_mean": float("nan"),
        "posterior_entropy_normalised": float("nan"),
        "component_size_l1": float("nan"),
        "component_size_recovery_score": float("nan"),
    }


def _format_label_mapping(mapping: dict[str, str]) -> str:
    return ";".join(f"{pred}->{truth}" for pred, truth in sorted(mapping.items()))


def _false_discrete_row(
    frame: pd.DataFrame,
    *,
    analysis_set: str,
    stress_family: str,
    stress_level: str,
    true_world_id: str,
) -> dict[str, Any]:
    n_runs = int(frame.shape[0])
    n_discrete = int(frame["selected_model_id"].isin(DISCRETE_MODEL_IDS).sum())
    return {
        "analysis_set": analysis_set,
        "stress_family": stress_family,
        "stress_level": stress_level,
        "true_world_id": true_world_id,
        "n_runs": n_runs,
        "n_discrete_selected": n_discrete,
        "false_discrete_rate": n_discrete / n_runs if n_runs else float("nan"),
    }


def _split_audit(frame_without_truth: pd.DataFrame, split: SplitDefinition) -> dict[str, int]:
    train_groups = _participant_groups(frame_without_truth.loc[list(split.train_indices)])
    test_groups = _participant_groups(frame_without_truth.loc[list(split.test_indices)])
    overlap = train_groups.intersection(test_groups)
    return {
        "n_train_rows": len(split.train_indices),
        "n_test_rows": len(split.test_indices),
        "n_train_participants": len(train_groups),
        "n_test_participants": len(test_groups),
        "n_participant_overlap": len(overlap),
        "participant_isolated": len(overlap) == 0,
    }


def _participant_groups(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(
        frame.loc[:, ["source_dataset", "participant_id"]]
        .astype(str)
        .agg(tuple, axis=1)
        .tolist()
    )


def _model_complexity_rank(model_id: str) -> int:
    try:
        return STATIC_MODEL_IDS.index(model_id)
    except ValueError:
        return len(STATIC_MODEL_IDS)


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
    parser = argparse.ArgumentParser(
        description="Run synthetic W0-W4 model recovery and falsification gate.",
    )
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--stress-replicates", type=int, default=2)
    parser.add_argument("--participants-per-dataset", type=int, default=24)
    parser.add_argument("--sessions-per-participant", type=int, default=2)
    parser.add_argument("--windows-per-session", type=int, default=3)
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args(argv)

    config = RecoverySimulationConfig(
        seed=args.seed,
        n_replicates=args.replicates,
        n_stress_replicates=args.stress_replicates,
        participants_per_dataset=args.participants_per_dataset,
        sessions_per_participant=args.sessions_per_participant,
        windows_per_session=args.windows_per_session,
        test_size=args.test_size,
    )
    outputs = run_synthetic_model_recovery(config)
    paths = write_synthetic_recovery_outputs(outputs, output_dir=args.output_dir, config=config)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
