"""Frozen M6.1 HCP-YA transversal K/C/V analysis runner.

The runner implements the pre-registered exploratory HCP-YA N=100
transversal test. It writes participant-free summaries only: participant-level
held-out scores are kept in memory for paired contrasts and are never written.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from trident_validation.config import ConfigValidationError, load_yaml_config
from trident_validation.mechanistic.hcp_extract_schema import (
    canonicalize_hcp_extract_columns,
    load_hcp_extract_schema,
)
from trident_validation.provenance import get_git_commit, hash_file, hash_mapping


@dataclass(frozen=True)
class HCPTransversalAnalysisResult:
    """Participant-free M6.1 analysis result."""

    summary: dict[str, Any]
    model_summary: pd.DataFrame
    contrast_summary: pd.DataFrame
    domain_weight_summary: pd.DataFrame
    residual_diagnostics: pd.DataFrame
    diagnostic_summary: dict[str, Any]
    manifest: dict[str, Any]
    report_markdown: str


@dataclass(frozen=True)
class HCPTransversalAnalysisPlan:
    """Participant-free locked HCP analysis plan."""

    summary: dict[str, Any]
    domain_plan: pd.DataFrame
    model_plan: pd.DataFrame
    layer_specific_plan: pd.DataFrame
    report_markdown: str


def run_hcp_transversal_analysis_plan(
    config_path: str | Path = "config/hcp_ya_transversal_analysis_v1.yaml",
    *,
    repo_root: str | Path | None = None,
    preflight_summary_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    write_outputs: bool = True,
) -> HCPTransversalAnalysisPlan:
    """Build the locked HCP-YA analysis plan/status report without fitting."""

    config_file, root, config = _load_config(config_path, repo_root)
    validate_hcp_transversal_analysis_config(config)
    preflight_path = _preflight_path(root, config, preflight_summary_path)
    preflight = _load_preflight_summary(preflight_path)
    gate = _preflight_gate_status(config, preflight)
    domain_plan = _domain_plan(config)
    model_plan = _model_plan(config)
    layer_plan = _layer_specific_plan(config)
    summary = {
        "analysis_id": config["analysis"]["id"],
        "status": gate["status"],
        "question": config["analysis"]["question"],
        "model_fitting_enabled": bool(config["analysis"]["model_fitting_enabled"]),
        "model_fitting_allowed_now": False,
        "requires_pre_outcome_freeze_commit": True,
        "preflight_summary_path": str(preflight_path),
        "preflight_status": gate["preflight_status"],
        "preflight_support_passed": gate["preflight_support_passed"],
        "blocked_reason": gate["blocked_reason"],
        "formal_claims_allowed": False,
        "trident_validation_claim_allowed": False,
        "transfer_outcomes_allowed": False,
        "pace_ontology_claim_allowed": False,
        "dynamic_regime_allowed": False,
        "neural_criticality_claim_allowed": False,
        "cusp_claim_allowed": False,
        "predictive_calibration_claim_allowed": False,
        "t_commit_claim_allowed": False,
        "ordinary_participant_folds_allowed": False,
        "primary_metric": config["validation"]["primary_metric"],
        "config_hash": _config_hash(root, config_file),
        "config_content_hash": hash_mapping(config),
        "git_commit": get_git_commit(root),
    }
    report = _render_plan_report(summary, domain_plan, model_plan, layer_plan)
    if write_outputs:
        out_dir = _output_dir(root, config, output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_json(out_dir / "analysis_plan_summary.json", summary)
        domain_plan.to_csv(out_dir / "domain_plan.csv", index=False)
        model_plan.to_csv(out_dir / "model_plan.csv", index=False)
        layer_plan.to_csv(out_dir / "layer_specific_plan.csv", index=False)
        (out_dir / "analysis_plan_report.md").write_text(report, encoding="utf-8", newline="\n")
        report_path = root / config["outputs"]["plan_report_md"]
        report_path.write_text(report, encoding="utf-8", newline="\n")
    return HCPTransversalAnalysisPlan(summary, domain_plan, model_plan, layer_plan, report)


def run_hcp_transversal_analysis(
    config_path: str | Path = "config/hcp_ya_transversal_analysis_v1.yaml",
    *,
    repo_root: str | Path | None = None,
    data_path: str | Path | None = None,
    preflight_summary_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    write_outputs: bool = True,
) -> HCPTransversalAnalysisResult:
    """Run the frozen M6.1 analysis once and write participant-free outputs."""

    config_file, root, config = _load_config(config_path, repo_root)
    validate_hcp_transversal_analysis_config(config)
    if config["analysis"].get("model_fitting_enabled") is not True:
        raise ConfigValidationError("analysis.model_fitting_enabled must be true for --run-analysis")
    preflight_path = _preflight_path(root, config, preflight_summary_path)
    preflight = _load_preflight_summary(preflight_path)
    gate = _preflight_gate_status(config, preflight)
    if gate["status"] != "ready_to_run_frozen_analysis":
        raise ConfigValidationError(f"support preflight does not permit analysis run: {gate['blocked_reason']}")

    extract_path = Path(data_path) if data_path is not None else _resolve_repo_path(root, config["inputs"]["hcp_extract_path"])
    if not extract_path.exists():
        raise ConfigValidationError(f"HCP extract does not exist: {extract_path}")
    data = _read_extract(extract_path, config, root)
    _validate_extract_for_analysis(data, config)
    folds = make_participant_folds(
        data[config["inputs"]["participant_id_column"]],
        n_folds=int(config["validation"]["n_folds"]),
        seed=int(config["validation"]["split_seed"]),
    )

    all_model_rows: list[pd.DataFrame] = []
    all_score_rows: list[pd.DataFrame] = []
    for domain, spec in _analysis_domains(config).items():
        outcome = _as_list(spec["columns"])[0]
        model_summary, participant_scores = _run_domain_models(data, config, domain, outcome, folds)
        all_model_rows.append(model_summary)
        all_score_rows.append(participant_scores)

    participant_scores_all = pd.concat(all_score_rows, ignore_index=True)
    model_summary = pd.concat(all_model_rows, ignore_index=True)
    contrast_summary = _contrast_summary(participant_scores_all, config)
    domain_weight_summary, domain_weight_scores = _run_domain_weight_models(data, config, folds)
    residual_diagnostics = _residual_diagnostics(participant_scores_all, config)
    diagnostic_summary = _diagnostic_summary(data, config)

    summary = {
        "analysis_id": config["analysis"]["id"],
        "status": "m6_1_analysis_complete_exploratory_no_confirmatory_claim",
        "question": config["analysis"]["question"],
        "n_subjects": int(data[config["inputs"]["participant_id_column"]].nunique()),
        "cohort_mode": config["inputs"]["cohort_mode"],
        "split_strategy": config["validation"]["split_strategy"],
        "n_folds": int(config["validation"]["n_folds"]),
        "split_seed": int(config["validation"]["split_seed"]),
        "bootstrap_seed": int(config["validation"]["bootstrap_seed"]),
        "bootstrap_iterations": int(config["validation"]["bootstrap_iterations"]),
        "model_fitting_enabled": True,
        "formal_claims_allowed": False,
        "trident_validation_claim_allowed": False,
        "transfer_outcomes_allowed": False,
        "participant_level_outputs_written": False,
        "participant_level_data_in_git_allowed": False,
        "m6_2_status": config["m6_2_layer_specific_capacity_gate"]["status"],
        "m6_3_status": config["m6_3_bottleneck_gate"]["status"],
        "config_hash": _config_hash(root, config_file),
        "config_content_hash": hash_mapping(config),
        "input_hash": hash_file(extract_path),
        "preflight_summary_path": str(preflight_path),
        "preflight_status": gate["preflight_status"],
        "git_commit": get_git_commit(root),
    }
    manifest = _manifest(
        summary,
        config,
        config_file,
        root,
        extract_path,
        preflight_path,
        folds,
        model_summary,
        contrast_summary,
        domain_weight_summary,
        residual_diagnostics,
        diagnostic_summary,
    )
    report = _render_analysis_report(
        summary,
        model_summary,
        contrast_summary,
        domain_weight_summary,
        residual_diagnostics,
        diagnostic_summary,
    )
    if write_outputs:
        _write_analysis_outputs(
            root,
            config,
            output_dir,
            summary,
            model_summary,
            contrast_summary,
            domain_weight_summary,
            residual_diagnostics,
            diagnostic_summary,
            manifest,
            report,
        )
    return HCPTransversalAnalysisResult(
        summary=summary,
        model_summary=model_summary,
        contrast_summary=contrast_summary,
        domain_weight_summary=domain_weight_summary,
        residual_diagnostics=residual_diagnostics,
        diagnostic_summary=diagnostic_summary,
        manifest=manifest,
        report_markdown=report,
    )


def validate_hcp_transversal_analysis_config(config: dict[str, Any]) -> None:
    """Validate frozen analysis config and anti-circularity rules."""

    analysis = _required_mapping(config, "analysis")
    if analysis.get("id") != "hcp_ya_transversal_analysis_v1":
        raise ConfigValidationError("analysis.id must be hcp_ya_transversal_analysis_v1")
    if analysis.get("status") != "m6_1_pre_outcome_frozen_analysis_protocol":
        raise ConfigValidationError("analysis.status must be m6_1_pre_outcome_frozen_analysis_protocol")
    for field in (
        "formal_claims_allowed",
        "trident_validation_claim_allowed",
        "transfer_outcomes_allowed",
        "pace_ontology_claim_allowed",
        "dynamic_regime_allowed",
        "neural_criticality_claim_allowed",
        "cusp_claim_allowed",
        "predictive_calibration_claim_allowed",
        "t_commit_claim_allowed",
    ):
        if analysis.get(field) is not False:
            raise ConfigValidationError(f"analysis.{field} must be false")
    if analysis.get("model_fitting_enabled") is not True:
        raise ConfigValidationError("analysis.model_fitting_enabled must be true after pre-outcome freeze")
    if analysis.get("requires_pre_outcome_freeze_commit") is not True:
        raise ConfigValidationError("analysis must require a pre-outcome freeze commit")

    inputs = _required_mapping(config, "inputs")
    if inputs.get("participant_level_data_in_git_allowed") is not False:
        raise ConfigValidationError("participant-level data in Git must be false")
    if inputs.get("cohort_mode") != "hcp_100_unrelated":
        raise ConfigValidationError("M6.1 currently requires cohort_mode=hcp_100_unrelated")
    provenance = _required_mapping(inputs, "cohort_provenance")
    if provenance.get("official_hcp_100_unrelated_subjects_group_declared") is not True:
        raise ConfigValidationError("official HCP 100 Unrelated Subjects provenance must be declared")
    if provenance.get("exported_from_official_group") is not True:
        raise ConfigValidationError("hcp_100_unrelated provenance must state exported_from_official_group=true")
    if inputs.get("family_id_column") is not None:
        raise ConfigValidationError("hcp_100_unrelated analysis must not require Family_ID")
    family_policy = _required_mapping(inputs, "family_structure_policy")
    if family_policy.get("ordinary_participant_folds_allowed") is not False:
        raise ConfigValidationError("ordinary participant folds are not allowed")

    validation = _required_mapping(config, "validation")
    if validation.get("split_strategy") != "participant_isolated_official_hcp_100_unrelated":
        raise ConfigValidationError("split strategy must be the explicit HCP 100-unrelated participant-isolated mode")
    if validation.get("n_folds") != 5:
        raise ConfigValidationError("M6.1 requires 5 folds")
    if validation.get("all_scaling_inside_training_folds") is not True:
        raise ConfigValidationError("all scaling must occur inside training folds")
    if validation.get("primary_metric") != "participant_isolated_heldout_predictive_log_density_per_participant":
        raise ConfigValidationError("primary metric must be participant-isolated held-out predictive density")
    if validation.get("bootstrap_iterations") != 1000:
        raise ConfigValidationError("M6.1 requires 1000 bootstrap iterations")

    constructs = _required_mapping(config, "constructs")
    if _as_list(constructs["K_candidate"].get("columns")) != [
        "ProcSpeed_Unadj",
        "PicSeq_Unadj",
        "ReadEng_Unadj",
        "PicVocab_Unadj",
    ]:
        raise ConfigValidationError("K_candidate columns changed from the frozen source variables")
    if constructs["K_candidate"].get("aggregation") != "unweighted_arithmetic_mean_after_training_fold_standardisation":
        raise ConfigValidationError("K_candidate must use unweighted post-standardisation averaging")
    if _as_list(constructs["C_candidate"].get("columns")) != ["Flanker_Unadj"]:
        raise ConfigValidationError("C_candidate must be Flanker_Unadj only")
    if _as_list(constructs["V_candidate"].get("columns")) != ["SCPT_SEN", "SCPT_SPEC"]:
        raise ConfigValidationError("V_candidate must use SCPT_SEN and SCPT_SPEC only")
    if constructs["V_candidate"].get("aggregation") != "unweighted_arithmetic_mean_after_training_fold_standardisation":
        raise ConfigValidationError("V_candidate must use unweighted post-standardisation averaging")

    predictors = {
        "K_candidate": set(_as_list(constructs["K_candidate"].get("columns"))),
        "C_candidate": set(_as_list(constructs["C_candidate"].get("columns"))),
        "V_candidate": set(_as_list(constructs["V_candidate"].get("columns"))),
    }
    forbidden_k = {
        "Flanker_Unadj",
        "CardSort_Unadj",
        "ListSort_Unadj",
        "WM_Task_2bk_Acc",
        "Relational_Task_Acc",
        "PMAT24_A_CR",
        "PMAT24_A_RTCR",
        "CogTotalComp_Unadj",
        "CogTotalComp_AgeAdj",
        "CogFluidComp_Unadj",
        "CogFluidComp_AgeAdj",
        "CogCrystalComp_Unadj",
        "CogCrystalComp_AgeAdj",
    }
    overlap_k = predictors["K_candidate"].intersection(forbidden_k)
    if overlap_k:
        raise ConfigValidationError("K contains forbidden HCP overlap/global columns: " + ", ".join(sorted(overlap_k)))

    outcomes = _required_mapping(config, "outcome_domains")
    cv_columns = predictors["C_candidate"] | predictors["V_candidate"]
    for domain, spec in outcomes.items():
        anti = _required_mapping(spec, "anti_circularity")
        if anti.get("exclude_outcome_from_K") is not True:
            raise ConfigValidationError(f"{domain} must exclude outcome columns from K")
        if anti.get("forbid_overlap_with_C_or_V") is not True:
            raise ConfigValidationError(f"{domain} must forbid overlap with C/V")
        outcome_columns = set(_as_list(spec.get("columns"))) | set(_as_list(spec.get("secondary_columns")))
        if outcome_columns.intersection(cv_columns):
            raise ConfigValidationError(f"{domain} outcome overlaps with C/V predictors")
        if outcome_columns.intersection(predictors["K_candidate"]):
            raise ConfigValidationError(f"{domain} outcome overlaps with K predictors")
    if outcomes["wm_nback"].get("analysis_eligible") is not False:
        raise ConfigValidationError("WM_Task_2bk_Acc must remain scientifically ineligible")

    expected_models = [
        ("M0_intercept", []),
        ("M1_K", ["K_candidate"]),
        ("M2_K_plus_C", ["K_candidate", "C_candidate"]),
        ("M3_K_plus_V", ["K_candidate", "V_candidate"]),
        ("M4_K_plus_C_plus_V", ["K_candidate", "C_candidate", "V_candidate"]),
        ("M5_K_plus_C_plus_V_plus_C_by_V", ["K_candidate", "C_candidate", "V_candidate", "C_by_V"]),
    ]
    observed_models = [(model["id"], _as_list(model.get("predictors"))) for model in config["model_set"]]
    if observed_models != expected_models:
        raise ConfigValidationError("model_set must exactly match the frozen M0-M5 sequence")

    contrasts = _required_mapping(config, "registered_contrasts")
    expected_contrasts = {
        "K_increment": ("M1_K", "M0_intercept"),
        "C_increment_beyond_K": ("M2_K_plus_C", "M1_K"),
        "V_increment_beyond_K": ("M3_K_plus_V", "M1_K"),
        "joint_CV_increment_beyond_K": ("M4_K_plus_C_plus_V", "M1_K"),
        "interaction_increment": ("M5_K_plus_C_plus_V_plus_C_by_V", "M4_K_plus_C_plus_V"),
    }
    observed_contrasts = {
        name: (spec.get("comparison_model"), spec.get("reference_model"))
        for name, spec in contrasts.items()
    }
    if observed_contrasts != expected_contrasts:
        raise ConfigValidationError("registered contrasts changed")
    if contrasts["interaction_increment"].get("status") != "secondary_only":
        raise ConfigValidationError("interaction increment must remain secondary only")

    domain_weight = _required_mapping(config, "domain_weight_test")
    if domain_weight.get("enabled") is not True:
        raise ConfigValidationError("domain-weight test must be enabled")
    if domain_weight.get("status") != "secondary_low_precision":
        raise ConfigValidationError("domain-weight test must remain secondary low-precision")

    m6_2 = _required_mapping(config, "m6_2_layer_specific_capacity_gate")
    if m6_2.get("status") != "blocked_pending_independent_second_WM_indicator":
        raise ConfigValidationError("M6.2 gate must remain blocked pending an independent second WM indicator")
    m6_3 = _required_mapping(config, "m6_3_bottleneck_gate")
    if m6_3.get("status") != "blocked_until_m6_2_establishes_defensible_independent_layer_specific_WM_candidate":
        raise ConfigValidationError("M6.3 gate must remain blocked")


def make_participant_folds(subjects: pd.Series, *, n_folds: int, seed: int) -> list[np.ndarray]:
    """Create deterministic participant-isolated folds."""

    unique_subjects = np.array(sorted({str(subject) for subject in subjects.dropna()}), dtype=object)
    if len(unique_subjects) < n_folds:
        raise ConfigValidationError("not enough participants for requested folds")
    rng = np.random.default_rng(seed)
    shuffled = unique_subjects.copy()
    rng.shuffle(shuffled)
    return [fold for fold in np.array_split(shuffled, n_folds) if len(fold) > 0]


def _run_domain_models(
    data: pd.DataFrame,
    config: dict[str, Any],
    domain: str,
    outcome: str,
    folds: list[np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    participant_col = config["inputs"]["participant_id_column"]
    required_columns = _construct_source_columns(config) + [outcome]
    domain_data = data[[participant_col] + required_columns].dropna(subset=required_columns).copy()
    score_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for fold_index, test_subjects in enumerate(folds):
        is_test = domain_data[participant_col].astype(str).isin(set(test_subjects))
        train = domain_data.loc[~is_test].copy()
        test = domain_data.loc[is_test].copy()
        if train.empty or test.empty:
            raise ConfigValidationError(f"empty train/test split for {domain}")
        train_features, test_features = _fold_constructs(train, test, config)
        y_train = train[outcome].to_numpy(dtype=float)
        y_test = test[outcome].to_numpy(dtype=float)
        for model in config["model_set"]:
            model_id = model["id"]
            predictors = _feature_columns_for_model(model)
            fit = _fit_ridge_gaussian(train_features[predictors].to_numpy(dtype=float), y_train, float(config["validation"]["ridge_alpha"]))
            pred = _predict_ridge_gaussian(fit, test_features[predictors].to_numpy(dtype=float))
            log_density = _gaussian_log_density(y_test, pred, fit["sigma2"])
            for row_index, subject in enumerate(test[participant_col].astype(str).to_numpy()):
                score_rows.append(
                    {
                        "domain": domain,
                        "model_id": model_id,
                        "subject": subject,
                        "fold": fold_index,
                        "heldout_log_density": float(log_density[row_index]),
                    }
                )
                prediction_rows.append(
                    {
                        "domain": domain,
                        "model_id": model_id,
                        "y": float(y_test[row_index]),
                        "prediction": float(pred[row_index]),
                        "residual": float(y_test[row_index] - pred[row_index]),
                        "heldout_log_density": float(log_density[row_index]),
                    }
                )
    scores = pd.DataFrame(score_rows)
    predictions = pd.DataFrame(prediction_rows)
    summaries = []
    for model_id, group in predictions.groupby("model_id", sort=False):
        metrics = _prediction_metrics(group["y"].to_numpy(), group["prediction"].to_numpy(), group["heldout_log_density"].to_numpy())
        summaries.append(
            {
                "domain": domain,
                "model_id": model_id,
                "n_participants": int(scores.loc[scores["model_id"] == model_id, "subject"].nunique()),
                **metrics,
            }
        )
    return pd.DataFrame(summaries), scores


def _run_domain_weight_models(
    data: pd.DataFrame,
    config: dict[str, Any],
    folds: list[np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    participant_col = config["inputs"]["participant_id_column"]
    domains = _as_list(config["domain_weight_test"]["domains"])
    outcomes = {domain: _as_list(config["outcome_domains"][domain]["columns"])[0] for domain in domains}
    required_columns = _construct_source_columns(config) + list(outcomes.values())
    stack_source = data[[participant_col] + required_columns].dropna(subset=required_columns).copy()
    score_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for fold_index, test_subjects in enumerate(folds):
        is_test = stack_source[participant_col].astype(str).isin(set(test_subjects))
        train = stack_source.loc[~is_test].copy()
        test = stack_source.loc[is_test].copy()
        train_features, test_features = _fold_constructs(train, test, config)
        train_stack, test_stack = _stack_domain_rows(train, test, train_features, test_features, outcomes)
        train_stack, test_stack = _standardise_stacked_outcomes(train_stack, test_stack)
        for model_id, predictors in {
            "common_slope": ["domain_wm_list_sorting", "domain_reasoning_pmat", "K", "C", "V"],
            "domain_specific_slope": [
                "domain_wm_list_sorting",
                "domain_reasoning_pmat",
                "K",
                "C",
                "V",
                "domain_wm_list_sorting_x_K",
                "domain_wm_list_sorting_x_C",
                "domain_wm_list_sorting_x_V",
                "domain_reasoning_pmat_x_K",
                "domain_reasoning_pmat_x_C",
                "domain_reasoning_pmat_x_V",
            ],
        }.items():
            fit = _fit_ridge_gaussian(
                train_stack[predictors].to_numpy(dtype=float),
                train_stack["y_standardised"].to_numpy(dtype=float),
                float(config["validation"]["ridge_alpha"]),
            )
            pred = _predict_ridge_gaussian(fit, test_stack[predictors].to_numpy(dtype=float))
            y_test = test_stack["y_standardised"].to_numpy(dtype=float)
            log_density = _gaussian_log_density(y_test, pred, fit["sigma2"])
            for row_index, row in enumerate(test_stack.itertuples(index=False)):
                score_rows.append(
                    {
                        "model_id": model_id,
                        "subject": row.subject,
                        "fold": fold_index,
                        "domain": row.domain,
                        "heldout_log_density": float(log_density[row_index]),
                    }
                )
                prediction_rows.append(
                    {
                        "model_id": model_id,
                        "y": float(y_test[row_index]),
                        "prediction": float(pred[row_index]),
                        "heldout_log_density": float(log_density[row_index]),
                    }
                )
    scores = pd.DataFrame(score_rows)
    predictions = pd.DataFrame(prediction_rows)
    rows = []
    for model_id, group in predictions.groupby("model_id", sort=False):
        participant_density = scores.loc[scores["model_id"] == model_id].groupby("subject")["heldout_log_density"].mean()
        metrics = _prediction_metrics(group["y"].to_numpy(), group["prediction"].to_numpy(), group["heldout_log_density"].to_numpy())
        rows.append(
            {
                "comparison": "common_vs_domain_specific_slope",
                "model_id": model_id,
                "n_participants": int(len(participant_density)),
                "heldout_log_density_mean_per_participant": float(participant_density.mean()),
                **metrics,
            }
        )
    participant_delta = (
        scores.groupby(["subject", "model_id"])["heldout_log_density"].mean().unstack()
    )
    delta = participant_delta["domain_specific_slope"] - participant_delta["common_slope"]
    ci = _bootstrap_ci(delta.to_numpy(dtype=float), int(config["validation"]["bootstrap_seed"]), int(config["validation"]["bootstrap_iterations"]))
    rows.append(
        {
            "comparison": "common_vs_domain_specific_slope",
            "model_id": "domain_specific_minus_common",
            "n_participants": int(delta.notna().sum()),
            "heldout_log_density_mean_delta_per_participant": float(delta.mean()),
            "bootstrap_ci_low": ci[0],
            "bootstrap_ci_high": ci[1],
            "status": "secondary_low_precision",
        }
    )
    return pd.DataFrame(rows), scores


def _fold_constructs(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    constructs = config["constructs"]
    train_out = pd.DataFrame(index=train.index)
    test_out = pd.DataFrame(index=test.index)
    k_train, k_test = _standardise_columns(
        train,
        test,
        _as_list(constructs["K_candidate"]["columns"]),
    )
    v_train, v_test = _standardise_columns(
        train,
        test,
        _as_list(constructs["V_candidate"]["columns"]),
    )
    c_column = _as_list(constructs["C_candidate"]["columns"])[0]
    c_train, c_test = _standardise_columns(train, test, [c_column])
    train_out["K"] = k_train.mean(axis=1)
    test_out["K"] = k_test.mean(axis=1)
    train_out["C"] = c_train[:, 0]
    test_out["C"] = c_test[:, 0]
    train_out["V"] = v_train.mean(axis=1)
    test_out["V"] = v_test.mean(axis=1)
    train_out["C_by_V"] = train_out["C"] * train_out["V"]
    test_out["C_by_V"] = test_out["C"] * test_out["V"]
    return train_out, test_out


def _standardise_columns(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    train_values = train[columns].to_numpy(dtype=float)
    test_values = test[columns].to_numpy(dtype=float)
    means = train_values.mean(axis=0)
    sds = train_values.std(axis=0, ddof=0)
    if np.any(~np.isfinite(sds)) or np.any(sds <= 0):
        raise ConfigValidationError("training-fold standardisation encountered a zero/non-finite SD")
    return (train_values - means) / sds, (test_values - means) / sds


def _feature_columns_for_model(model: dict[str, Any]) -> list[str]:
    mapping = {
        "K_candidate": "K",
        "C_candidate": "C",
        "V_candidate": "V",
        "C_by_V": "C_by_V",
    }
    return [mapping[predictor] for predictor in _as_list(model.get("predictors"))]


def _fit_ridge_gaussian(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, Any]:
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    design = np.column_stack([np.ones(len(y)), x])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    pred = design @ beta
    sigma2 = max(float(np.mean((y - pred) ** 2)), 1e-8)
    return {"beta": beta, "sigma2": sigma2}


def _predict_ridge_gaussian(fit: dict[str, Any], x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    design = np.column_stack([np.ones(x.shape[0]), x])
    return design @ fit["beta"]


def _gaussian_log_density(y: np.ndarray, pred: np.ndarray, sigma2: float) -> np.ndarray:
    return -0.5 * (np.log(2.0 * math.pi * sigma2) + ((y - pred) ** 2) / sigma2)


def _prediction_metrics(y: np.ndarray, pred: np.ndarray, log_density: np.ndarray) -> dict[str, float | None]:
    residual = y - pred
    sst = float(np.sum((y - y.mean()) ** 2))
    corr = None
    if np.std(y) > 0 and np.std(pred) > 0:
        corr = float(np.corrcoef(y, pred)[0, 1])
    return {
        "heldout_log_density_total": float(np.sum(log_density)),
        "heldout_log_density_mean_per_participant": float(np.mean(log_density)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "heldout_r2": float(1.0 - np.sum(residual**2) / sst) if sst > 0 else None,
        "heldout_correlation": corr,
        "heldout_residual_variance": float(np.var(residual, ddof=0)),
        "heldout_outcome_variance": float(np.var(y, ddof=0)),
    }


def _contrast_summary(participant_scores: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    boot_seed = int(config["validation"]["bootstrap_seed"])
    boot_iters = int(config["validation"]["bootstrap_iterations"])
    for domain in participant_scores["domain"].drop_duplicates():
        wide = participant_scores.loc[participant_scores["domain"] == domain].pivot(
            index="subject",
            columns="model_id",
            values="heldout_log_density",
        )
        for index, (contrast, spec) in enumerate(config["registered_contrasts"].items()):
            comparison = spec["comparison_model"]
            reference = spec["reference_model"]
            delta = wide[comparison] - wide[reference]
            ci = _bootstrap_ci(delta.to_numpy(dtype=float), boot_seed + index, boot_iters)
            rows.append(
                {
                    "domain": domain,
                    "contrast": contrast,
                    "comparison_model": comparison,
                    "reference_model": reference,
                    "status": spec.get("status", "primary"),
                    "n_participants": int(delta.notna().sum()),
                    "mean_delta_log_density_per_participant": float(delta.mean()),
                    "total_delta_log_density": float(delta.sum()),
                    "bootstrap_ci_low": ci[0],
                    "bootstrap_ci_high": ci[1],
                    "fraction_positive_participant_deltas": float((delta > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_ci(values: np.ndarray, seed: int, iterations: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=float)
    for draw_index in range(iterations):
        sample = rng.choice(values, size=len(values), replace=True)
        draws[draw_index] = sample.mean()
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _stack_domain_rows(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    outcomes: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_rows = []
    test_rows = []
    participant_col = "Subject"
    for frame, features, destination in ((train, train_features, train_rows), (test, test_features, test_rows)):
        for domain, outcome in outcomes.items():
            domain_dummy_wm = 1.0 if domain == "wm_list_sorting" else 0.0
            domain_dummy_reasoning = 1.0 if domain == "reasoning_pmat" else 0.0
            for row_index, source_row in enumerate(frame.itertuples(index=False)):
                k = float(features.iloc[row_index]["K"])
                c = float(features.iloc[row_index]["C"])
                v = float(features.iloc[row_index]["V"])
                destination.append(
                    {
                        "subject": str(getattr(source_row, participant_col)),
                        "domain": domain,
                        "y_raw": float(getattr(source_row, outcome)),
                        "K": k,
                        "C": c,
                        "V": v,
                        "domain_wm_list_sorting": domain_dummy_wm,
                        "domain_reasoning_pmat": domain_dummy_reasoning,
                        "domain_wm_list_sorting_x_K": domain_dummy_wm * k,
                        "domain_wm_list_sorting_x_C": domain_dummy_wm * c,
                        "domain_wm_list_sorting_x_V": domain_dummy_wm * v,
                        "domain_reasoning_pmat_x_K": domain_dummy_reasoning * k,
                        "domain_reasoning_pmat_x_C": domain_dummy_reasoning * c,
                        "domain_reasoning_pmat_x_V": domain_dummy_reasoning * v,
                    }
                )
    return pd.DataFrame(train_rows), pd.DataFrame(test_rows)


def _standardise_stacked_outcomes(
    train_stack: pd.DataFrame,
    test_stack: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train_stack.copy()
    test = test_stack.copy()
    train["y_standardised"] = np.nan
    test["y_standardised"] = np.nan
    for domain in train["domain"].unique():
        train_mask = train["domain"] == domain
        test_mask = test["domain"] == domain
        mean = float(train.loc[train_mask, "y_raw"].mean())
        sd = float(train.loc[train_mask, "y_raw"].std(ddof=0))
        if not np.isfinite(sd) or sd <= 0:
            raise ConfigValidationError(f"zero/non-finite training outcome SD for stacked domain {domain}")
        train.loc[train_mask, "y_standardised"] = (train.loc[train_mask, "y_raw"] - mean) / sd
        test.loc[test_mask, "y_standardised"] = (test.loc[test_mask, "y_raw"] - mean) / sd
    return train, test


def _residual_diagnostics(participant_scores: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for domain in participant_scores["domain"].drop_duplicates():
        for model_id in ("M0_intercept", "M4_K_plus_C_plus_V"):
            model_rows = participant_scores[
                (participant_scores["domain"] == domain) & (participant_scores["model_id"] == model_id)
            ]
            rows.append(
                {
                    "domain": domain,
                    "model_id": model_id,
                    "n_participants": int(model_rows["subject"].nunique()),
                    "mean_heldout_log_density": float(model_rows["heldout_log_density"].mean()),
                    "diagnostic_label": "descriptive_heldout_unexplained_outcome_variation_not_latent_capacity",
                }
            )
    return pd.DataFrame(rows)


def _diagnostic_summary(data: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    pmat_rt = config["diagnostic_outcomes"]["pmat_rt"]["column"]
    return {
        "pmat_rt_column": pmat_rt,
        "pmat_rt_available": pmat_rt in data.columns,
        "pmat_rt_nonmissing": int(data[pmat_rt].notna().sum()) if pmat_rt in data.columns else 0,
        "pmat_rt_used_in_reasoning_composite": False,
        "wm_task_2bk_acc_used": False,
        "relational_task_acc_used": False,
    }


def _domain_plan(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    k_columns = set(_as_list(config["constructs"]["K_candidate"]["columns"]))
    for domain, spec in config["outcome_domains"].items():
        outcome_columns = set(_as_list(spec.get("columns"))) | set(_as_list(spec.get("secondary_columns")))
        rows.append(
            {
                "domain": domain,
                "primary": bool(spec.get("primary", False)),
                "role": spec.get("role", "unspecified"),
                "analysis_eligible": bool(spec.get("analysis_eligible", True)),
                "status": str(spec.get("status", "analysis_eligible" if spec.get("analysis_eligible", True) else "blocked")),
                "outcome_columns": "|".join(sorted(outcome_columns)),
                "k_columns_after_exclusion": "|".join(sorted(k_columns.difference(outcome_columns))),
                "c_candidate_columns": "|".join(_as_list(config["constructs"]["C_candidate"]["columns"])),
                "v_columns": "|".join(_as_list(config["constructs"]["V_candidate"]["columns"])),
                "anti_circularity_checked": True,
            }
        )
    return pd.DataFrame(rows)


def _model_plan(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for domain in _analysis_domains(config):
        for index, model in enumerate(config["model_set"]):
            rows.append(
                {
                    "domain": domain,
                    "model_order": index,
                    "model_id": model["id"],
                    "predictors": "|".join(_as_list(model.get("predictors"))),
                    "primary_metric": config["validation"]["primary_metric"],
                }
            )
    return pd.DataFrame(rows)


def _layer_specific_plan(config: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gate": "M6.2",
                "status": config["m6_2_layer_specific_capacity_gate"]["status"],
                "allowed_descriptive_residual_diagnostic": bool(
                    config["m6_2_layer_specific_capacity_gate"]["descriptive_residual_variance_allowed"]
                ),
                "latent_capacity_claim_allowed": False,
            },
            {
                "gate": "M6.3",
                "status": config["m6_3_bottleneck_gate"]["status"],
                "allowed_descriptive_residual_diagnostic": False,
                "latent_capacity_claim_allowed": False,
            },
        ]
    )


def _analysis_domains(config: dict[str, Any]) -> dict[str, Any]:
    return {
        domain: spec
        for domain, spec in config["outcome_domains"].items()
        if bool(spec.get("analysis_eligible", True)) and spec.get("role") in {"primary", "secondary_near_domain"}
    }


def _validate_extract_for_analysis(data: pd.DataFrame, config: dict[str, Any]) -> None:
    participant_col = config["inputs"]["participant_id_column"]
    if participant_col not in data.columns:
        raise ConfigValidationError("extract is missing participant column")
    if data[participant_col].duplicated().any():
        raise ConfigValidationError("extract contains duplicate participants")
    n_subjects = int(data[participant_col].nunique())
    expected = int(config["inputs"]["expected_n_subjects"])
    if n_subjects != expected:
        raise ConfigValidationError(f"expected {expected} HCP unrelated subjects, got {n_subjects}")
    required = _construct_source_columns(config)
    for spec in _analysis_domains(config).values():
        required.extend(_as_list(spec["columns"]))
    missing = sorted(set(required).difference(data.columns))
    if missing:
        raise ConfigValidationError("extract is missing required analysis columns: " + ", ".join(missing))


def _construct_source_columns(config: dict[str, Any]) -> list[str]:
    constructs = config["constructs"]
    return (
        _as_list(constructs["K_candidate"]["columns"])
        + _as_list(constructs["C_candidate"]["columns"])
        + _as_list(constructs["V_candidate"]["columns"])
    )


def _preflight_gate_status(config: dict[str, Any], preflight: dict[str, Any] | None) -> dict[str, Any]:
    required = config["analysis"]["required_preflight_status"]
    if preflight is None:
        return {
            "status": "blocked_no_preflight_summary",
            "preflight_status": None,
            "preflight_support_passed": False,
            "blocked_reason": "preflight summary is missing",
        }
    preflight_status = preflight.get("status")
    support_passed = bool(preflight.get("support_passed", False))
    cohort_ok = preflight.get("cohort_mode") == "hcp_100_unrelated"
    n_ok = int(preflight.get("input_unique_subjects", 0)) == int(config["inputs"]["expected_n_subjects"])
    if preflight_status == required and support_passed and cohort_ok and n_ok:
        return {
            "status": "ready_to_run_frozen_analysis",
            "preflight_status": preflight_status,
            "preflight_support_passed": True,
            "blocked_reason": None,
        }
    return {
        "status": "blocked_preflight_not_passed",
        "preflight_status": preflight_status,
        "preflight_support_passed": support_passed,
        "blocked_reason": f"requires status {required}, support_passed true, hcp_100_unrelated, and expected N",
    }


def _manifest(
    summary: dict[str, Any],
    config: dict[str, Any],
    config_file: Path,
    root: Path,
    extract_path: Path,
    preflight_path: Path,
    folds: list[np.ndarray],
    model_summary: pd.DataFrame,
    contrast_summary: pd.DataFrame,
    domain_weight_summary: pd.DataFrame,
    residual_diagnostics: pd.DataFrame,
    diagnostic_summary: dict[str, Any],
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
        "preflight_summary_path": str(preflight_path),
        "preflight_hash": hash_file(preflight_path) if preflight_path.exists() else None,
        "cohort_mode": summary["cohort_mode"],
        "n_subjects": summary["n_subjects"],
        "split": {
            "name": summary["split_strategy"],
            "n_folds": summary["n_folds"],
            "seed": summary["split_seed"],
            "fold_sizes": [int(len(fold)) for fold in folds],
        },
        "seeds": {
            "split_seed": summary["split_seed"],
            "bootstrap_seed": summary["bootstrap_seed"],
        },
        "ridge_alpha": float(config["validation"]["ridge_alpha"]),
        "bootstrap_iterations": summary["bootstrap_iterations"],
        "model_ids": [model["id"] for model in config["model_set"]],
        "output_hashes": {
            "model_summary_content": hash_mapping({"rows": model_summary.to_dict(orient="records")}),
            "contrast_summary_content": hash_mapping({"rows": contrast_summary.to_dict(orient="records")}),
            "domain_weight_summary_content": hash_mapping({"rows": domain_weight_summary.to_dict(orient="records")}),
            "residual_diagnostics_content": hash_mapping({"rows": residual_diagnostics.to_dict(orient="records")}),
            "diagnostic_summary_content": hash_mapping(diagnostic_summary),
        },
        "participant_level_outputs_written": False,
        "claims": {
            "formal_claims_allowed": False,
            "trident_validation_claim_allowed": False,
            "g_confirmed": False,
            "c_signal_confirmed": False,
            "v_mechanism_confirmed": False,
            "w_specific_estimated": False,
            "bottleneck_tests_run": False,
        },
    }


def _write_analysis_outputs(
    root: Path,
    config: dict[str, Any],
    output_dir: str | Path | None,
    summary: dict[str, Any],
    model_summary: pd.DataFrame,
    contrast_summary: pd.DataFrame,
    domain_weight_summary: pd.DataFrame,
    residual_diagnostics: pd.DataFrame,
    diagnostic_summary: dict[str, Any],
    manifest: dict[str, Any],
    report: str,
) -> None:
    out_dir = _output_dir(root, config, output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "analysis_summary.json", summary)
    model_summary.to_csv(out_dir / "model_summary.csv", index=False)
    contrast_summary.to_csv(out_dir / "contrast_summary.csv", index=False)
    domain_weight_summary.to_csv(out_dir / "domain_weight_summary.csv", index=False)
    residual_diagnostics.to_csv(out_dir / "residual_diagnostics.csv", index=False)
    _write_json(out_dir / "diagnostic_summary.json", diagnostic_summary)
    _write_json(out_dir / "manifest.json", manifest)
    (out_dir / "analysis_report.md").write_text(report, encoding="utf-8", newline="\n")
    tracked_report = root / config["outputs"]["analysis_report_md"]
    tracked_summary = root / config["outputs"]["participant_free_summary_json"]
    tracked_manifest = root / config["outputs"]["manifest_json"]
    tracked_report.write_text(report, encoding="utf-8", newline="\n")
    _write_json(tracked_summary, _participant_free_result_payload(summary, model_summary, contrast_summary, domain_weight_summary, residual_diagnostics, diagnostic_summary))
    _write_json(tracked_manifest, manifest)


def _participant_free_result_payload(
    summary: dict[str, Any],
    model_summary: pd.DataFrame,
    contrast_summary: pd.DataFrame,
    domain_weight_summary: pd.DataFrame,
    residual_diagnostics: pd.DataFrame,
    diagnostic_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "summary": summary,
        "model_summary": model_summary.to_dict(orient="records"),
        "contrast_summary": contrast_summary.to_dict(orient="records"),
        "domain_weight_summary": domain_weight_summary.to_dict(orient="records"),
        "residual_diagnostics": residual_diagnostics.to_dict(orient="records"),
        "diagnostic_summary": diagnostic_summary,
    }


def _render_plan_report(
    summary: dict[str, Any],
    domain_plan: pd.DataFrame,
    model_plan: pd.DataFrame,
    layer_plan: pd.DataFrame,
) -> str:
    lines = [
        "# M6.1 HCP-YA Transversal Analysis Plan",
        "",
        "**Status:** pre-outcome frozen analysis protocol",
        "",
        f"**Model fitting enabled after freeze:** {str(summary['model_fitting_enabled']).lower()}",
        "",
        "No Trident-G/APC/PACE validation claim, transfer claim, g confirmation, C_signal confirmation, V mechanism confirmation, W-specific capacity claim, bottleneck claim, neural-criticality claim or cusp claim is authorised.",
        "",
        "## Question",
        "",
        summary["question"],
        "",
        "## Gate Status",
        "",
        f"- Status: `{summary['status']}`",
        f"- Preflight status: `{summary['preflight_status']}`",
        f"- Preflight support passed: {str(summary['preflight_support_passed']).lower()}",
        "",
        "## Domain Plan",
        "",
        "| Domain | Role | Primary | Analysis eligible | Status | Outcome columns | K columns after exclusion |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for row in domain_plan.itertuples(index=False):
        lines.append(
            f"| {row.domain} | {row.role} | {str(row.primary).lower()} | "
            f"{str(row.analysis_eligible).lower()} | {row.status} | {row.outcome_columns} | "
            f"{row.k_columns_after_exclusion} |"
        )
    lines.extend(["", "## Model Sequence", "", "| Domain | Order | Model | Predictors |", "|---|---:|---|---|"])
    for row in model_plan.itertuples(index=False):
        lines.append(f"| {row.domain} | {row.model_order} | {row.model_id} | {row.predictors or 'intercept'} |")
    lines.extend(
        [
            "",
            "## M6.2/M6.3 Gates",
            "",
            "| Gate | Status | Latent capacity claim allowed |",
            "|---|---|---:|",
        ]
    )
    for row in layer_plan.itertuples(index=False):
        lines.append(f"| {row.gate} | {row.status} | {str(row.latent_capacity_claim_allowed).lower()} |")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Participant-level HCP data read by this plan: false",
            "- Ordinary participant folds allowed: false",
            "- Outcome columns reused to construct K/C_candidate/V: false",
            "- M5 interaction is secondary only and low precision",
            "- N=100 exploratory unrelated-subject test only",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_analysis_report(
    summary: dict[str, Any],
    model_summary: pd.DataFrame,
    contrast_summary: pd.DataFrame,
    domain_weight_summary: pd.DataFrame,
    residual_diagnostics: pd.DataFrame,
    diagnostic_summary: dict[str, Any],
) -> str:
    lines = [
        "# M6.1 HCP-YA Transversal K/C Candidate/V Analysis Report",
        "",
        "**Status:** exploratory external-data analysis complete",
        "",
        "**N:** 100 official HCP unrelated subjects",
        "",
        "This is an exploratory HCP unrelated-subject test with N=100. It does not confirm g, cognitive control as a latent mechanism, vigilance as a latent mechanism, Trident-G, APC, PACE, W-specific capacity, bottlenecks, criticality or transfer.",
        "",
        "## Summary",
        "",
        f"- Cohort mode: `{summary['cohort_mode']}`",
        f"- Split strategy: `{summary['split_strategy']}`",
        f"- Folds: {summary['n_folds']}",
        f"- Split seed: {summary['split_seed']}",
        f"- Bootstrap seed: {summary['bootstrap_seed']}",
        f"- Bootstrap iterations: {summary['bootstrap_iterations']}",
        "- Participant-level outputs written: false",
        "",
        "## Model Summary",
        "",
        "| Domain | Model | Mean held-out log density | RMSE | MAE | Held-out R2 | Held-out r |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in model_summary.itertuples(index=False):
        lines.append(
            f"| {row.domain} | {row.model_id} | {_fmt(row.heldout_log_density_mean_per_participant)} | "
            f"{_fmt(row.rmse)} | {_fmt(row.mae)} | {_fmt(row.heldout_r2)} | {_fmt(row.heldout_correlation)} |"
        )
    lines.extend(
        [
            "",
            "## Registered Contrasts",
            "",
            "| Domain | Contrast | Mean delta | 95% bootstrap CI | Fraction positive | Status |",
            "|---|---|---:|---|---:|---|",
        ]
    )
    for row in contrast_summary.itertuples(index=False):
        lines.append(
            f"| {row.domain} | {row.contrast} | {_fmt(row.mean_delta_log_density_per_participant)} | "
            f"[{_fmt(row.bootstrap_ci_low)}, {_fmt(row.bootstrap_ci_high)}] | "
            f"{_fmt(row.fraction_positive_participant_deltas)} | {row.status} |"
        )
    lines.extend(
        [
            "",
            "## Domain-Weight Comparison",
            "",
            "| Model/result | Mean held-out log density or delta | 95% bootstrap CI | Status |",
            "|---|---:|---|---|",
        ]
    )
    for row in domain_weight_summary.itertuples(index=False):
        value = getattr(row, "heldout_log_density_mean_delta_per_participant", None)
        if value is None or pd.isna(value):
            value = getattr(row, "heldout_log_density_mean_per_participant", None)
        ci_low = getattr(row, "bootstrap_ci_low", None)
        ci_high = getattr(row, "bootstrap_ci_high", None)
        status = getattr(row, "status", "secondary")
        lines.append(
            f"| {row.model_id} | {_fmt(value)} | [{_fmt(ci_low)}, {_fmt(ci_high)}] | {status} |"
        )
    lines.extend(
        [
            "",
            "## Descriptive Residual Diagnostics",
            "",
            "These residual diagnostics are descriptive unexplained outcome variation only. They are not W, WM capacity, a layer-specific factor or a bottleneck variable.",
            "",
            "| Domain | Model | Mean held-out log density | Diagnostic label |",
            "|---|---|---:|---|",
        ]
    )
    for row in residual_diagnostics.itertuples(index=False):
        lines.append(f"| {row.domain} | {row.model_id} | {_fmt(row.mean_heldout_log_density)} | {row.diagnostic_label} |")
    lines.extend(
        [
            "",
            "## Diagnostic Fields",
            "",
            f"- PMAT24_A_RTCR available: {str(diagnostic_summary['pmat_rt_available']).lower()}",
            f"- PMAT24_A_RTCR non-missing: {diagnostic_summary['pmat_rt_nonmissing']}",
            "- PMAT RT used in reasoning composite: false",
            "- WM_Task_2bk_Acc used: false",
            "- Relational_Task_Acc used: false",
            "",
            "## Gates",
            "",
            f"- M6.2 status: `{summary['m6_2_status']}`",
            f"- M6.3 status: `{summary['m6_3_status']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.6f}"


def _load_config(
    config_path: str | Path,
    repo_root: str | Path | None,
) -> tuple[Path, Path, dict[str, Any]]:
    config_file = Path(config_path)
    root = Path(repo_root) if repo_root is not None else config_file.resolve().parents[1]
    config = load_yaml_config(root / config_file if not config_file.is_absolute() else config_file)
    return config_file, root, config


def _preflight_path(
    root: Path,
    config: dict[str, Any],
    preflight_summary_path: str | Path | None,
) -> Path:
    return (
        Path(preflight_summary_path)
        if preflight_summary_path is not None
        else _resolve_repo_path(root, config["analysis"]["current_preflight_summary_path"])
    )


def _read_extract(path: Path, config: dict[str, Any], root: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        data = pd.read_csv(path)
    elif suffix in {".tsv", ".tab"}:
        data = pd.read_csv(path, sep="\t")
    elif suffix == ".parquet":
        data = pd.read_parquet(path)
    else:
        raise ConfigValidationError(f"unsupported HCP extract extension: {path.suffix}")
    schema_path = config["inputs"].get("extract_schema_path")
    if schema_path:
        schema = load_hcp_extract_schema(schema_path, repo_root=root)
        data = canonicalize_hcp_extract_columns(data, schema)
    return data


def _load_preflight_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _output_dir(root: Path, config: dict[str, Any], output_dir: str | Path | None) -> Path:
    return Path(output_dir) if output_dir is not None else root / config["outputs"]["output_dir"]


def _config_hash(root: Path, config_file: Path) -> str:
    return hash_file(root / config_file) if not config_file.is_absolute() else hash_file(config_file)


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


def _resolve_repo_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/hcp_ya_transversal_analysis_v1.yaml")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--preflight-summary", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--run-analysis", action="store_true")
    args = parser.parse_args(argv)
    if args.plan_only == args.run_analysis:
        raise ConfigValidationError("choose exactly one of --plan-only or --run-analysis")
    if args.plan_only:
        result = run_hcp_transversal_analysis_plan(
            args.config,
            preflight_summary_path=args.preflight_summary,
            output_dir=args.output_dir,
        )
        print(json.dumps(result.summary, indent=2, sort_keys=True))
        return
    result = run_hcp_transversal_analysis(
        args.config,
        data_path=args.data_path,
        preflight_summary_path=args.preflight_summary,
        output_dir=args.output_dir,
    )
    print(json.dumps(result.summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
