"""Exploratory M6.1b HCP-YA transversal sensitivity runner.

This module implements post-M6.1 robustness checks only. It writes
participant-free summaries and never writes Subject IDs, fold assignments,
participant-level scores, predictions or residuals.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trident_validation.config import ConfigValidationError, load_yaml_config
from trident_validation.mechanistic.hcp_transversal_analysis import (
    _bootstrap_ci,
    _config_hash,
    _fit_ridge_gaussian,
    _gaussian_log_density,
    _prediction_metrics,
    _predict_ridge_gaussian,
    _read_extract,
    _standardise_columns,
    make_participant_folds,
    validate_hcp_transversal_analysis_config,
)
from trident_validation.provenance import get_git_commit, hash_file, hash_mapping


@dataclass(frozen=True)
class HCPSensitivityResult:
    """Participant-free M6.1b sensitivity result."""

    summary: dict[str, Any]
    model_summary: pd.DataFrame
    contrast_summary: pd.DataFrame
    k_robustness_summary: pd.DataFrame
    demand_pattern_summary: pd.DataFrame
    manifest: dict[str, Any]
    report_markdown: str


def run_hcp_transversal_sensitivity(
    config_path: str | Path = "config/hcp_ya_transversal_sensitivity_v1.yaml",
    *,
    repo_root: str | Path | None = None,
    data_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    write_outputs: bool = True,
) -> HCPSensitivityResult:
    """Run the post-M6.1 exploratory sensitivity addendum."""

    config_file, root, config = _load_config(config_path, repo_root)
    validate_hcp_transversal_sensitivity_config(config)
    base_config_path = _resolve_repo_path(root, config["inputs"]["base_analysis_config_path"])
    base_config = load_yaml_config(base_config_path)
    validate_hcp_transversal_analysis_config(base_config)

    extract_path = Path(data_path) if data_path is not None else _resolve_repo_path(root, config["inputs"]["hcp_extract_path"])
    if not extract_path.exists():
        raise ConfigValidationError(f"HCP extract does not exist: {extract_path}")
    data = _read_extract(extract_path, config, root)
    _validate_extract_for_sensitivity(data, config)
    folds = make_participant_folds(
        data[config["inputs"]["participant_id_column"]],
        n_folds=int(config["validation"]["n_folds"]),
        seed=int(config["validation"]["split_seed"]),
    )

    model_summary, participant_scores = _run_sensitivity_models(data, config, folds)
    contrast_summary = _sensitivity_contrasts(participant_scores, config)
    k_robustness_summary = _k_robustness_summary(contrast_summary)
    demand_pattern_summary = _demand_pattern_summary(contrast_summary)
    summary = {
        "analysis_id": config["analysis"]["id"],
        "status": "m6_1b_sensitivity_complete_exploratory_no_confirmatory_claim",
        "question": config["analysis"]["question"],
        "formulated_after_m6_1_outcome_inspection": True,
        "changes_frozen_m6_1_analysis": False,
        "n_subjects": int(data[config["inputs"]["participant_id_column"]].nunique()),
        "cohort_mode": config["inputs"]["cohort_mode"],
        "split_strategy": config["validation"]["split_strategy"],
        "n_folds": int(config["validation"]["n_folds"]),
        "split_seed": int(config["validation"]["split_seed"]),
        "bootstrap_seed": int(config["validation"]["bootstrap_seed"]),
        "bootstrap_iterations": int(config["validation"]["bootstrap_iterations"]),
        "formal_claims_allowed": False,
        "confirmed_g_claim_allowed": False,
        "confirmed_c_signal_claim_allowed": False,
        "confirmed_v_mechanism_claim_allowed": False,
        "representational_operator_claim_allowed": False,
        "strategic_policy_claim_allowed": False,
        "participant_level_outputs_written": False,
        "participant_level_data_in_git_allowed": False,
        "wm_task_2bk_acc_used": False,
        "relational_task_acc_used": False,
        "pmat_rt_used": False,
        "config_hash": _config_hash(root, config_file),
        "config_content_hash": hash_mapping(config),
        "base_analysis_config_hash": hash_file(base_config_path),
        "input_hash": hash_file(extract_path),
        "git_commit": get_git_commit(root),
    }
    manifest = _manifest(
        summary,
        config,
        config_file,
        root,
        extract_path,
        folds,
        model_summary,
        contrast_summary,
        k_robustness_summary,
        demand_pattern_summary,
    )
    report = _render_report(summary, model_summary, contrast_summary, k_robustness_summary, demand_pattern_summary)
    if write_outputs:
        _write_outputs(
            root,
            config,
            output_dir,
            summary,
            model_summary,
            contrast_summary,
            k_robustness_summary,
            demand_pattern_summary,
            manifest,
            report,
        )
    return HCPSensitivityResult(
        summary=summary,
        model_summary=model_summary,
        contrast_summary=contrast_summary,
        k_robustness_summary=k_robustness_summary,
        demand_pattern_summary=demand_pattern_summary,
        manifest=manifest,
        report_markdown=report,
    )


def validate_hcp_transversal_sensitivity_config(config: dict[str, Any]) -> None:
    """Validate the post-M6.1 sensitivity config and claim boundaries."""

    analysis = _required_mapping(config, "analysis")
    if analysis.get("id") != "hcp_ya_transversal_sensitivity_v1":
        raise ConfigValidationError("analysis.id must be hcp_ya_transversal_sensitivity_v1")
    if analysis.get("status") != "post_m6_1_exploratory_sensitivity_addendum":
        raise ConfigValidationError("analysis.status must be post_m6_1_exploratory_sensitivity_addendum")
    for field in (
        "changes_frozen_m6_1_analysis",
        "formal_claims_allowed",
        "trident_validation_claim_allowed",
        "transfer_outcomes_allowed",
        "confirmed_g_claim_allowed",
        "confirmed_c_signal_claim_allowed",
        "confirmed_v_mechanism_claim_allowed",
        "representational_operator_claim_allowed",
        "strategic_policy_claim_allowed",
        "bottleneck_claim_allowed",
    ):
        if analysis.get(field) is not False:
            raise ConfigValidationError(f"analysis.{field} must be false")
    if analysis.get("formulated_after_m6_1_outcome_inspection") is not True:
        raise ConfigValidationError("sensitivity addendum must declare post-M6.1 formulation")
    if analysis.get("model_fitting_enabled") is not True:
        raise ConfigValidationError("analysis.model_fitting_enabled must be true")

    inputs = _required_mapping(config, "inputs")
    if inputs.get("cohort_mode") != "hcp_100_unrelated":
        raise ConfigValidationError("sensitivity requires cohort_mode=hcp_100_unrelated")
    if inputs.get("family_id_column") is not None:
        raise ConfigValidationError("hcp_100_unrelated sensitivity must not require Family_ID")
    for field in (
        "participant_level_data_in_git_allowed",
        "participant_level_predictions_written",
        "participant_level_scores_written",
    ):
        if inputs.get(field) is not False:
            raise ConfigValidationError(f"inputs.{field} must be false")
    provenance = _required_mapping(inputs, "cohort_provenance")
    if provenance.get("official_hcp_100_unrelated_subjects_group_declared") is not True:
        raise ConfigValidationError("official HCP 100 Unrelated Subjects provenance must be declared")

    validation = _required_mapping(config, "validation")
    if validation.get("split_strategy") != "participant_isolated_official_hcp_100_unrelated":
        raise ConfigValidationError("split strategy must be explicit official unrelated participant isolation")
    if validation.get("n_folds") != 5:
        raise ConfigValidationError("sensitivity requires 5 folds")
    if validation.get("all_scaling_inside_training_folds") is not True:
        raise ConfigValidationError("all scaling must occur inside training folds")
    if validation.get("primary_metric") != "participant_isolated_heldout_predictive_log_density_per_participant":
        raise ConfigValidationError("primary metric must be held-out predictive log density")
    if int(validation.get("bootstrap_iterations", 0)) != 1000:
        raise ConfigValidationError("sensitivity requires 1000 bootstrap iterations")

    forbidden = set(_as_list(config["forbidden_columns"]["never_in_k_variant"]))
    variants = _as_list_of_mappings(config.get("k_sensitivity_variants"), "k_sensitivity_variants")
    variant_ids = [variant.get("id") for variant in variants]
    if len(variant_ids) != len(set(variant_ids)):
        raise ConfigValidationError("k_sensitivity_variants must have unique ids")
    if "registered_full_k" not in variant_ids:
        raise ConfigValidationError("registered_full_k variant is required")
    for variant in variants:
        columns = _as_list(variant.get("columns"))
        if len(columns) < 2:
            raise ConfigValidationError(f"{variant.get('id')} must contain at least two indicators")
        overlap = sorted(set(columns).intersection(forbidden))
        if overlap:
            raise ConfigValidationError(f"{variant.get('id')} contains forbidden columns: {', '.join(overlap)}")
    full = next(variant for variant in variants if variant["id"] == "registered_full_k")
    if _as_list(full["columns"]) != ["ProcSpeed_Unadj", "PicSeq_Unadj", "ReadEng_Unadj", "PicVocab_Unadj"]:
        raise ConfigValidationError("registered_full_k must match frozen M6.1 K columns")

    never_used = set(_as_list(config["forbidden_columns"]["never_used"]))
    outcomes = _required_mapping(config, "outcome_domains")
    for domain, spec in outcomes.items():
        columns = set(_as_list(spec.get("columns")))
        if columns.intersection(never_used):
            raise ConfigValidationError(f"{domain} uses forbidden diagnostic/blocked columns")

    boundaries = _required_mapping(config, "interpretation_boundaries")
    for field in (
        "confirms_g",
        "confirms_cognitive_control",
        "confirms_vigilance_mechanism",
        "tests_binding_operator",
        "tests_relational_operator",
        "tests_wm_updating_operator",
        "tests_strategy",
        "tests_bottleneck",
        "single_task_residual_to_capacity_allowed",
    ):
        if boundaries.get(field) is not False:
            raise ConfigValidationError(f"interpretation_boundaries.{field} must be false")


def _run_sensitivity_models(
    data: pd.DataFrame,
    config: dict[str, Any],
    folds: list[np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    participant_col = config["inputs"]["participant_id_column"]
    all_summaries: list[pd.DataFrame] = []
    all_scores: list[pd.DataFrame] = []
    for variant in config["k_sensitivity_variants"]:
        for domain, spec in config["outcome_domains"].items():
            outcome = _as_list(spec["columns"])[0]
            required = _required_source_columns(config, variant) + [outcome]
            domain_data = data[[participant_col] + required].dropna(subset=required).copy()
            summaries, scores = _run_variant_domain(domain_data, config, variant, domain, outcome, folds)
            all_summaries.append(summaries)
            all_scores.append(scores)
    return pd.concat(all_summaries, ignore_index=True), pd.concat(all_scores, ignore_index=True)


def _run_variant_domain(
    domain_data: pd.DataFrame,
    config: dict[str, Any],
    variant: dict[str, Any],
    domain: str,
    outcome: str,
    folds: list[np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    participant_col = config["inputs"]["participant_id_column"]
    score_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    models = [
        {"id": "S0_intercept", "predictors": []},
        {"id": "S1_K_variant", "predictors": ["K"]},
    ]
    if variant["id"] == config["model_families"]["demand_pattern_reference"]["k_variant"]:
        models.extend(config["model_families"]["demand_pattern_reference"]["models"])
    for fold_index, test_subjects in enumerate(folds):
        is_test = domain_data[participant_col].astype(str).isin(set(test_subjects))
        train = domain_data.loc[~is_test].copy()
        test = domain_data.loc[is_test].copy()
        if train.empty or test.empty:
            raise ConfigValidationError(f"empty train/test split for {domain}/{variant['id']}")
        train_features, test_features = _fold_features(train, test, config, variant)
        y_train = train[outcome].to_numpy(dtype=float)
        y_test = test[outcome].to_numpy(dtype=float)
        for model in models:
            model_id = model["id"]
            predictors = _as_list(model.get("predictors"))
            fit = _fit_ridge_gaussian(
                train_features[predictors].to_numpy(dtype=float),
                y_train,
                float(config["validation"]["ridge_alpha"]),
            )
            pred = _predict_ridge_gaussian(fit, test_features[predictors].to_numpy(dtype=float))
            log_density = _gaussian_log_density(y_test, pred, fit["sigma2"])
            for row_index, subject in enumerate(test[participant_col].astype(str).to_numpy()):
                score_rows.append(
                    {
                        "domain": domain,
                        "k_variant": variant["id"],
                        "model_id": model_id,
                        "subject": subject,
                        "fold": fold_index,
                        "heldout_log_density": float(log_density[row_index]),
                    }
                )
                prediction_rows.append(
                    {
                        "domain": domain,
                        "k_variant": variant["id"],
                        "model_id": model_id,
                        "y": float(y_test[row_index]),
                        "prediction": float(pred[row_index]),
                        "heldout_log_density": float(log_density[row_index]),
                    }
                )
    scores = pd.DataFrame(score_rows)
    predictions = pd.DataFrame(prediction_rows)
    summaries = []
    for (k_variant, model_id), group in predictions.groupby(["k_variant", "model_id"], sort=False):
        metrics = _prediction_metrics(
            group["y"].to_numpy(),
            group["prediction"].to_numpy(),
            group["heldout_log_density"].to_numpy(),
        )
        n_participants = scores.loc[
            (scores["k_variant"] == k_variant) & (scores["model_id"] == model_id),
            "subject",
        ].nunique()
        summaries.append(
            {
                "domain": domain,
                "k_variant": k_variant,
                "model_id": model_id,
                "n_participants": int(n_participants),
                **metrics,
            }
        )
    return pd.DataFrame(summaries), scores


def _fold_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: dict[str, Any],
    variant: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = pd.DataFrame(index=train.index)
    test_out = pd.DataFrame(index=test.index)
    k_train, k_test = _standardise_columns(train, test, _as_list(variant["columns"]))
    c_col = _as_list(config["constructs"]["C_candidate"]["columns"])[0]
    c_train, c_test = _standardise_columns(train, test, [c_col])
    v_train, v_test = _standardise_columns(train, test, _as_list(config["constructs"]["V_candidate"]["columns"]))
    train_out["K"] = k_train.mean(axis=1)
    test_out["K"] = k_test.mean(axis=1)
    train_out["C"] = c_train[:, 0]
    test_out["C"] = c_test[:, 0]
    train_out["V"] = v_train.mean(axis=1)
    test_out["V"] = v_test.mean(axis=1)
    return train_out, test_out


def _sensitivity_contrasts(participant_scores: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    boot_seed = int(config["validation"]["bootstrap_seed"])
    boot_iters = int(config["validation"]["bootstrap_iterations"])
    contrast_index = 0
    for (domain, k_variant), group in participant_scores.groupby(["domain", "k_variant"], sort=False):
        wide = group.pivot(index="subject", columns="model_id", values="heldout_log_density")
        rows.append(
            _contrast_row(
                wide,
                domain,
                k_variant,
                "K_variant_increment",
                "S1_K_variant",
                "S0_intercept",
                "k_robustness",
                boot_seed + contrast_index,
                boot_iters,
            )
        )
        contrast_index += 1
        if k_variant == config["model_families"]["demand_pattern_reference"]["k_variant"]:
            for contrast, spec in config["model_families"]["demand_pattern_reference"]["contrasts"].items():
                rows.append(
                    _contrast_row(
                        wide,
                        domain,
                        k_variant,
                        contrast,
                        spec["comparison_model"],
                        spec["reference_model"],
                        "demand_pattern_diagnostic",
                        boot_seed + contrast_index,
                        boot_iters,
                    )
                )
                contrast_index += 1
    return pd.DataFrame(rows)


def _contrast_row(
    wide: pd.DataFrame,
    domain: str,
    k_variant: str,
    contrast: str,
    comparison: str,
    reference: str,
    status: str,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    delta = wide[comparison] - wide[reference]
    ci = _bootstrap_ci(delta.to_numpy(dtype=float), seed, iterations)
    return {
        "domain": domain,
        "k_variant": k_variant,
        "contrast": contrast,
        "comparison_model": comparison,
        "reference_model": reference,
        "status": status,
        "n_participants": int(delta.notna().sum()),
        "mean_delta_log_density_per_participant": float(delta.mean()),
        "total_delta_log_density": float(delta.sum()),
        "bootstrap_ci_low": ci[0],
        "bootstrap_ci_high": ci[1],
        "fraction_positive_participant_deltas": float((delta > 0).mean()),
    }


def _k_robustness_summary(contrast_summary: pd.DataFrame) -> pd.DataFrame:
    rows = contrast_summary.loc[contrast_summary["contrast"] == "K_variant_increment"].copy()
    return rows.sort_values(["domain", "mean_delta_log_density_per_participant"], ascending=[True, False]).reset_index(
        drop=True
    )


def _demand_pattern_summary(contrast_summary: pd.DataFrame) -> pd.DataFrame:
    rows = contrast_summary.loc[contrast_summary["status"] == "demand_pattern_diagnostic"].copy()
    return rows.sort_values(["contrast", "domain"]).reset_index(drop=True)


def _validate_extract_for_sensitivity(data: pd.DataFrame, config: dict[str, Any]) -> None:
    participant_col = config["inputs"]["participant_id_column"]
    if participant_col not in data.columns:
        raise ConfigValidationError("extract is missing participant column")
    if data[participant_col].duplicated().any():
        raise ConfigValidationError("extract contains duplicate participants")
    n_subjects = int(data[participant_col].nunique())
    expected = int(config["inputs"]["expected_n_subjects"])
    if n_subjects != expected:
        raise ConfigValidationError(f"expected {expected} HCP unrelated subjects, got {n_subjects}")
    required = set()
    for variant in config["k_sensitivity_variants"]:
        required.update(_as_list(variant["columns"]))
    required.update(_as_list(config["constructs"]["C_candidate"]["columns"]))
    required.update(_as_list(config["constructs"]["V_candidate"]["columns"]))
    for spec in config["outcome_domains"].values():
        required.update(_as_list(spec["columns"]))
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ConfigValidationError("extract is missing required sensitivity columns: " + ", ".join(missing))


def _required_source_columns(config: dict[str, Any], variant: dict[str, Any]) -> list[str]:
    return (
        _as_list(variant["columns"])
        + _as_list(config["constructs"]["C_candidate"]["columns"])
        + _as_list(config["constructs"]["V_candidate"]["columns"])
    )


def _manifest(
    summary: dict[str, Any],
    config: dict[str, Any],
    config_file: Path,
    root: Path,
    extract_path: Path,
    folds: list[np.ndarray],
    model_summary: pd.DataFrame,
    contrast_summary: pd.DataFrame,
    k_robustness_summary: pd.DataFrame,
    demand_pattern_summary: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "analysis_id": summary["analysis_id"],
        "status": summary["status"],
        "git_commit": summary["git_commit"],
        "config_path": str(config_file),
        "config_hash": summary["config_hash"],
        "config_content_hash": summary["config_content_hash"],
        "input_path": str(extract_path),
        "input_hash": summary["input_hash"],
        "cohort_mode": summary["cohort_mode"],
        "n_subjects": summary["n_subjects"],
        "split": {
            "name": summary["split_strategy"],
            "n_folds": summary["n_folds"],
            "seed": summary["split_seed"],
            "fold_sizes": [int(len(fold)) for fold in folds],
        },
        "bootstrap": {
            "seed": summary["bootstrap_seed"],
            "iterations": summary["bootstrap_iterations"],
        },
        "ridge_alpha": float(config["validation"]["ridge_alpha"]),
        "k_variant_ids": [variant["id"] for variant in config["k_sensitivity_variants"]],
        "output_hashes": {
            "model_summary_content": hash_mapping({"rows": model_summary.to_dict(orient="records")}),
            "contrast_summary_content": hash_mapping({"rows": contrast_summary.to_dict(orient="records")}),
            "k_robustness_summary_content": hash_mapping({"rows": k_robustness_summary.to_dict(orient="records")}),
            "demand_pattern_summary_content": hash_mapping({"rows": demand_pattern_summary.to_dict(orient="records")}),
        },
        "participant_level_outputs_written": False,
        "claims": {
            "formal_claims_allowed": False,
            "g_confirmed": False,
            "c_signal_confirmed": False,
            "v_mechanism_confirmed": False,
            "representational_operator_claim_allowed": False,
            "strategic_policy_claim_allowed": False,
            "bottleneck_tests_run": False,
        },
    }


def _write_outputs(
    root: Path,
    config: dict[str, Any],
    output_dir: str | Path | None,
    summary: dict[str, Any],
    model_summary: pd.DataFrame,
    contrast_summary: pd.DataFrame,
    k_robustness_summary: pd.DataFrame,
    demand_pattern_summary: pd.DataFrame,
    manifest: dict[str, Any],
    report: str,
) -> None:
    out_dir = Path(output_dir) if output_dir is not None else root / config["outputs"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "sensitivity_summary.json", summary)
    model_summary.to_csv(out_dir / "model_summary.csv", index=False)
    contrast_summary.to_csv(out_dir / "contrast_summary.csv", index=False)
    k_robustness_summary.to_csv(out_dir / "k_robustness_summary.csv", index=False)
    demand_pattern_summary.to_csv(out_dir / "demand_pattern_summary.csv", index=False)
    _write_json(out_dir / "manifest.json", manifest)
    (out_dir / "sensitivity_report.md").write_text(report, encoding="utf-8", newline="\n")

    (root / config["outputs"]["report_md"]).write_text(report, encoding="utf-8", newline="\n")
    _write_json(root / config["outputs"]["participant_free_summary_json"], _participant_free_payload(result_summary=summary, model_summary=model_summary, contrast_summary=contrast_summary, k_robustness_summary=k_robustness_summary, demand_pattern_summary=demand_pattern_summary))
    _write_json(root / config["outputs"]["manifest_json"], manifest)


def _participant_free_payload(
    *,
    result_summary: dict[str, Any],
    model_summary: pd.DataFrame,
    contrast_summary: pd.DataFrame,
    k_robustness_summary: pd.DataFrame,
    demand_pattern_summary: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "summary": result_summary,
        "model_summary": model_summary.to_dict(orient="records"),
        "contrast_summary": contrast_summary.to_dict(orient="records"),
        "k_robustness_summary": k_robustness_summary.to_dict(orient="records"),
        "demand_pattern_summary": demand_pattern_summary.to_dict(orient="records"),
    }


def _render_report(
    summary: dict[str, Any],
    model_summary: pd.DataFrame,
    contrast_summary: pd.DataFrame,
    k_robustness_summary: pd.DataFrame,
    demand_pattern_summary: pd.DataFrame,
) -> str:
    lines = [
        "# M6.1b HCP-YA Transversal Sensitivity Report",
        "",
        "**Status:** exploratory post-M6.1 sensitivity complete",
        "",
        "**N:** 100 official HCP unrelated subjects",
        "",
        "This addendum was formulated after M6.1 outcome inspection. It is a robustness and demand-pattern diagnostic only. It does not confirm g, cognitive control, vigilance, binding, relational processing, WM updating, strategic policy variables, bottlenecks, Trident-G, APC, PACE, criticality or transfer.",
        "",
        "## Summary",
        "",
        f"- Cohort mode: `{summary['cohort_mode']}`",
        f"- Split strategy: `{summary['split_strategy']}`",
        f"- Split seed: {summary['split_seed']}",
        f"- Bootstrap seed: {summary['bootstrap_seed']}",
        f"- Bootstrap iterations: {summary['bootstrap_iterations']}",
        "- Participant-level outputs written: false",
        "- Frozen M6.1 analysis changed: false",
        "",
        "## K Robustness",
        "",
        "| Domain | K variant | Mean delta log density | 95% bootstrap CI | Fraction positive |",
        "|---|---|---:|---|---:|",
    ]
    for row in k_robustness_summary.itertuples(index=False):
        lines.append(
            f"| {row.domain} | {row.k_variant} | {_fmt(row.mean_delta_log_density_per_participant)} | "
            f"[{_fmt(row.bootstrap_ci_low)}, {_fmt(row.bootstrap_ci_high)}] | "
            f"{_fmt(row.fraction_positive_participant_deltas)} |"
        )
    lines.extend(
        [
            "",
            "## C/V Demand-Pattern Diagnostics",
            "",
            "| Domain | Contrast | Mean delta log density | 95% bootstrap CI | Fraction positive |",
            "|---|---|---:|---|---:|",
        ]
    )
    for row in demand_pattern_summary.itertuples(index=False):
        lines.append(
            f"| {row.domain} | {row.contrast} | {_fmt(row.mean_delta_log_density_per_participant)} | "
            f"[{_fmt(row.bootstrap_ci_low)}, {_fmt(row.bootstrap_ci_high)}] | "
            f"{_fmt(row.fraction_positive_participant_deltas)} |"
        )
    lines.extend(
        [
            "",
            "## Model Descriptives",
            "",
            "| Domain | K variant | Model | Mean held-out log density | RMSE | MAE | Held-out R2 | Held-out r |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in model_summary.itertuples(index=False):
        lines.append(
            f"| {row.domain} | {row.k_variant} | {row.model_id} | "
            f"{_fmt(row.heldout_log_density_mean_per_participant)} | {_fmt(row.rmse)} | "
            f"{_fmt(row.mae)} | {_fmt(row.heldout_r2)} | {_fmt(row.heldout_correlation)} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- `WM_Task_2bk_Acc` used: false",
            "- `Relational_Task_Acc` used: false",
            "- `PMAT24_A_RTCR` used: false",
            "- Binding/relational/WM-updating operators tested: false",
            "- Strategic A/T/PC tested: false",
            "- Single-task residual converted to capacity: false",
        ]
    )
    _ = contrast_summary
    return "\n".join(lines) + "\n"


def _load_config(
    config_path: str | Path,
    repo_root: str | Path | None,
) -> tuple[Path, Path, dict[str, Any]]:
    config_file = Path(config_path)
    root = Path(repo_root) if repo_root is not None else config_file.resolve().parents[1]
    config = load_yaml_config(root / config_file if not config_file.is_absolute() else config_file)
    return config_file, root, config


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict) or not value:
        raise ConfigValidationError(f"{key} must be a non-empty mapping")
    return value


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _as_list_of_mappings(value: Any, key: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ConfigValidationError(f"{key} must be a non-empty list")
    if not all(isinstance(item, dict) for item in value):
        raise ConfigValidationError(f"{key} entries must be mappings")
    return value


def _resolve_repo_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _fmt(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.6f}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/hcp_ya_transversal_sensitivity_v1.yaml")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    result = run_hcp_transversal_sensitivity(
        args.config,
        data_path=args.data_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(result.summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
