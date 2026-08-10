"""Stage 1B speed-accuracy policy candidate analysis.

This module tests whether a cross-task speed-accuracy policy candidate is
reproducible beyond K, C_signal and V. It is a prospective amendment formulated
after Stage 1-3 review, but before inspecting T_policy outcomes.

The candidate is behavioural only. It is not T_commit, Predictive Calibration,
an optimality measure, a PACE state, a Trident-G state, a DDM boundary, a
dynamic regime or criticality evidence.
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
from trident_validation.mechanistic.stage1_k_c_v import (
    _fit_ridge_gaussian,
    _gaussian_log_density,
    _participant_folds,
    _required_mapping,
    _resolve_repo_path,
    _write_json,
)
from trident_validation.provenance import get_git_commit, hash_file, hash_mapping


TASKS = ("Stroop", "Flanker")


@dataclass(frozen=True)
class Stage1BResult:
    """Participant-free Stage 1B policy result."""

    summary: dict[str, Any]
    support_preflight: pd.DataFrame
    cross_task_contrasts: pd.DataFrame
    persistence_contrasts: pd.DataFrame
    residual_diagnostics: pd.DataFrame
    negative_controls: pd.DataFrame
    report_markdown: str


def run_stage1b_sat_policy_analysis(
    config_path: str | Path = "config/public_stage1b_sat_policy_v1.yaml",
    *,
    repo_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    write_outputs: bool = True,
) -> Stage1BResult:
    """Run the pre-registered Stage 1B speed-accuracy policy candidate test."""

    config_file = Path(config_path)
    root = Path(repo_root) if repo_root is not None else config_file.resolve().parents[1]
    config = load_yaml_config(root / config_file if not config_file.is_absolute() else config_file)
    _validate_config(config)

    input_path = _resolve_repo_path(root, config["inputs"]["paired_session_features_path"])
    observed_hash = hash_file(input_path)
    expected_hash = "sha256:" + str(config["inputs"]["paired_session_features_sha256"]).lower()
    if observed_hash.lower() != expected_hash:
        raise ConfigValidationError(
            f"paired session feature checksum mismatch: expected {expected_hash}, got {observed_hash}"
        )

    raw = pd.read_parquet(input_path)
    support = policy_support_preflight(raw, config)
    if not bool(support.attrs["support_gate_passed"]):
        raise ConfigValidationError("Stage 1B policy support gate failed")

    forbidden_used = _forbidden_columns_used(config)
    cross_scores = _run_cross_task_transport(raw, config)
    persistence_scores = _run_next_session_persistence(raw, config)
    cross_contrasts = _summarise_contrasts(
        cross_scores,
        config,
        contrast_specs={
            "policy_minus_base": ("policy", "base"),
            "speed_negative_control_minus_base": ("speed_negative_control", "base"),
            "accuracy_negative_control_minus_base": ("accuracy_negative_control", "base"),
        },
        target_filter=("T_flanker", "T_stroop"),
        unit_label="session_rows",
    )
    cross_mean = _mean_cross_task_policy_contrast(cross_scores, config)
    cross_contrasts = pd.concat([cross_contrasts, cross_mean], ignore_index=True)
    persistence_contrasts = _summarise_contrasts(
        persistence_scores,
        config,
        contrast_specs={
            "policy_persistence_minus_base": ("policy", "base"),
            "speed_persistence_negative_control_minus_base": ("speed_negative_control", "base"),
            "accuracy_persistence_negative_control_minus_base": ("accuracy_negative_control", "base"),
        },
        target_filter=("T_common_next",),
        unit_label="adjacent_pairs",
    )
    residual = _residual_policy_diagnostics(raw, config)
    decision = _make_decision(cross_contrasts, persistence_contrasts, residual, forbidden_used, config)
    negative_controls = _negative_control_summary(cross_contrasts, persistence_contrasts)

    summary = {
        "analysis_id": config["analysis"]["id"],
        "status": "completed",
        "question": config["analysis"]["question"],
        "decision": decision["decision"],
        "decision_reason": decision["reason"],
        "hypothesis_timing": config["analysis"]["hypothesis_timing"],
        "formal_claims_allowed": False,
        "trident_validation_claim_allowed": False,
        "transfer_outcomes_used": False,
        "pace_ontology_claim_allowed": False,
        "dynamic_regime_columns_used": False,
        "neural_criticality_claim_allowed": False,
        "cusp_claim_allowed": False,
        "t_commit_claim_allowed": False,
        "predictive_calibration_claim_allowed": False,
        "optimality_claim_allowed": False,
        "forbidden_columns_present": [c for c in config["forbidden_columns"] if c in raw.columns],
        "forbidden_columns_used": sorted(forbidden_used),
        "input_hash": observed_hash,
        "input_rows": int(len(raw)),
        "input_participants": int(raw[config["identity_columns"]["participant_id"]].nunique()),
        "support_gate_passed": bool(support.attrs["support_gate_passed"]),
        "cross_task_rows": int(
            cross_scores[["participant_id", "fold", "row_index"]].drop_duplicates().shape[0]
        ),
        "cross_task_participants": int(cross_scores["participant_id"].nunique()),
        "adjacent_pairs": int(persistence_scores["row_index"].nunique()),
        "adjacent_pair_participants": int(persistence_scores["participant_id"].nunique()),
        "n_folds": int(config["validation"]["n_folds"]),
        "config_hash": hash_file(root / config_file) if not config_file.is_absolute() else hash_file(config_file),
        "config_content_hash": hash_mapping(config),
        "git_commit": get_git_commit(root),
        "residual_T_common_variance_fraction_after_K_C_signal_V": float(
            residual.loc[residual["diagnostic"] == "residual_variance_fraction", "value"].iloc[0]
        ),
        "primary_cross_task_contrast": _contrast_summary(
            cross_contrasts,
            "mean_cross_task_transport",
            "policy_minus_base",
        ),
        "primary_next_session_contrast": _contrast_summary(
            persistence_contrasts,
            "T_common_next",
            "policy_persistence_minus_base",
        ),
    }
    report = _render_report(
        summary,
        support,
        cross_contrasts,
        persistence_contrasts,
        residual,
        negative_controls,
    )

    if write_outputs:
        out_dir = Path(output_dir) if output_dir is not None else root / config["outputs"]["output_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_json(out_dir / "stage1b_summary.json", summary)
        support.to_csv(out_dir / "support_preflight.csv", index=False)
        cross_contrasts.to_csv(out_dir / "cross_task_contrasts.csv", index=False)
        persistence_contrasts.to_csv(out_dir / "persistence_contrasts.csv", index=False)
        residual.to_csv(out_dir / "residual_diagnostics.csv", index=False)
        negative_controls.to_csv(out_dir / "negative_controls.csv", index=False)
        _write_json(out_dir / "manifest.json", _manifest(summary, config, root))
        (out_dir / "stage1b_report.md").write_text(report, encoding="utf-8", newline="\n")
        report_path = root / str(config["outputs"]["report_md"])
        report_path.write_text(report, encoding="utf-8", newline="\n")

    return Stage1BResult(
        summary=summary,
        support_preflight=support,
        cross_task_contrasts=cross_contrasts,
        persistence_contrasts=persistence_contrasts,
        residual_diagnostics=residual,
        negative_controls=negative_controls,
        report_markdown=report,
    )


def policy_support_preflight(raw: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Return schema/missingness support only; no hypothesis associations."""

    participant_col = config["identity_columns"]["participant_id"]
    rows: list[dict[str, Any]] = []
    required = config["support_preflight"]["required_observables"]
    all_policy_columns: list[str] = []
    for task in TASKS:
        accuracy = required[task]["accuracy"]
        mean_rt = required[task]["mean_rt_ms"]
        all_policy_columns.extend([accuracy, mean_rt])
        task_complete = raw.dropna(subset=[accuracy, mean_rt]) if accuracy in raw and mean_rt in raw else raw.iloc[0:0]
        rows.append(
            {
                "task": task,
                "accuracy_column": accuracy,
                "mean_rt_column": mean_rt,
                "accuracy_present": accuracy in raw.columns,
                "mean_rt_present": mean_rt in raw.columns,
                "accuracy_nonmissing": int(raw[accuracy].notna().sum()) if accuracy in raw else 0,
                "mean_rt_nonmissing": int(raw[mean_rt].notna().sum()) if mean_rt in raw else 0,
                "complete_speed_accuracy_rows": int(len(task_complete)),
                "complete_speed_accuracy_participants": int(task_complete[participant_col].nunique()) if participant_col in task_complete else 0,
            }
        )
    complete_both = raw.dropna(subset=all_policy_columns)
    rows.append(
        {
            "task": "Stroop_Flanker_common",
            "accuracy_column": "both",
            "mean_rt_column": "both",
            "accuracy_present": True,
            "mean_rt_present": True,
            "accuracy_nonmissing": int(raw[all_policy_columns].notna().all(axis=1).sum()),
            "mean_rt_nonmissing": int(raw[all_policy_columns].notna().all(axis=1).sum()),
            "complete_speed_accuracy_rows": int(len(complete_both)),
            "complete_speed_accuracy_participants": int(complete_both[participant_col].nunique()),
        }
    )
    support = pd.DataFrame(rows)
    common = support.loc[support["task"] == "Stroop_Flanker_common"].iloc[0]
    support.attrs["support_gate_passed"] = bool(
        common["complete_speed_accuracy_rows"] >= int(config["support_preflight"]["minimum_complete_policy_rows"])
        and common["complete_speed_accuracy_participants"]
        >= int(config["support_preflight"]["minimum_complete_policy_participants"])
    )
    return support


def _validate_config(config: dict[str, Any]) -> None:
    analysis = _required_mapping(config, "analysis")
    if analysis.get("id") != "public_stage1b_sat_policy_v1":
        raise ConfigValidationError("analysis.id must be public_stage1b_sat_policy_v1")
    if analysis.get("status") != "pre_outcome_registered_analysis":
        raise ConfigValidationError("analysis.status must be pre_outcome_registered_analysis")
    for field in (
        "formal_claims_allowed",
        "trident_validation_claim_allowed",
        "transfer_outcomes_allowed",
        "pace_ontology_claim_allowed",
        "dynamic_regime_allowed",
        "neural_criticality_claim_allowed",
        "cusp_claim_allowed",
        "t_commit_claim_allowed",
        "predictive_calibration_claim_allowed",
        "optimality_claim_allowed",
    ):
        if analysis.get(field) is not False:
            raise ConfigValidationError(f"analysis.{field} must be false")
    policy = _required_mapping(config, "policy_construction")
    if policy.get("fold_fitted_scaling_only") is not True:
        raise ConfigValidationError("policy scaling must be fold-fitted only")
    if policy.get("task_weights_fitted_from_data") is not False:
        raise ConfigValidationError("policy task weights must not be fitted from data")
    forbidden = set(config.get("forbidden_columns", ()))
    for group_name, group in _required_mapping(config, "feature_groups").items():
        columns = group.get("columns")
        if not isinstance(columns, dict) or not columns:
            raise ConfigValidationError(f"feature_groups.{group_name}.columns must be a mapping")
        blocked = forbidden.intersection(columns)
        if blocked:
            raise ConfigValidationError(
                f"feature_groups.{group_name}.columns includes forbidden columns: "
                + ", ".join(sorted(blocked))
            )
    if config["validation"].get("participant_isolated") is not True:
        raise ConfigValidationError("participant-isolated validation is required")


def _run_cross_task_transport(raw: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = _complete_session_rows(raw, config)
    fold_ids = _participant_folds(
        rows[config["identity_columns"]["participant_id"]].astype(str),
        int(config["validation"]["n_folds"]),
        int(config["validation"]["split_seed"]),
    )
    scored: list[pd.DataFrame] = []
    for fold in range(int(config["validation"]["n_folds"])):
        train_raw = rows.loc[fold_ids != fold].copy()
        test_raw = rows.loc[fold_ids == fold].copy()
        scaler = _fit_raw_scaler(train_raw, _raw_feature_columns(config))
        train = _session_design(train_raw, scaler, config)
        test = _session_design(test_raw, scaler, config)
        scored.extend(
            _score_direction(
                fold,
                train,
                test,
                target="T_flanker",
                base_predictors=("K_without_Flanker_general", "C_signal", "V"),
                policy_predictor="T_stroop",
                speed_predictor="speed_stroop",
                accuracy_predictor="accuracy_stroop",
                config=config,
            )
        )
        scored.extend(
            _score_direction(
                fold,
                train,
                test,
                target="T_stroop",
                base_predictors=("K_without_Stroop_general", "C_signal", "V"),
                policy_predictor="T_flanker",
                speed_predictor="speed_flanker",
                accuracy_predictor="accuracy_flanker",
                config=config,
            )
        )
    return pd.concat(scored, ignore_index=True)


def _run_next_session_persistence(raw: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    pairs = _adjacent_pairs(raw, config)
    fold_ids = _participant_folds(
        pairs["participant_id"].astype(str),
        int(config["validation"]["n_folds"]),
        int(config["validation"]["split_seed"]),
    )
    scored: list[pd.DataFrame] = []
    for fold in range(int(config["validation"]["n_folds"])):
        train_pairs = pairs.loc[fold_ids != fold].copy()
        test_pairs = pairs.loc[fold_ids == fold].copy()
        train_sessions = _pair_side_raw(train_pairs, ("current", "next"), config)
        scaler = _fit_raw_scaler(train_sessions, _raw_feature_columns(config))
        train = _pair_design(train_pairs, scaler, config)
        test = _pair_design(test_pairs, scaler, config)
        scored.extend(
            _score_target(
                fold,
                train,
                test,
                target="T_common_next",
                model_specs={
                    "base": ("K_current", "C_signal_current", "V_current"),
                    "policy": ("K_current", "C_signal_current", "V_current", "T_common_current"),
                    "speed_negative_control": ("K_current", "C_signal_current", "V_current", "speed_common_current"),
                    "accuracy_negative_control": ("K_current", "C_signal_current", "V_current", "accuracy_common_current"),
                },
                config=config,
            )
        )
    return pd.concat(scored, ignore_index=True)


def _score_direction(
    fold: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target: str,
    base_predictors: tuple[str, ...],
    policy_predictor: str,
    speed_predictor: str,
    accuracy_predictor: str,
    config: dict[str, Any],
) -> list[pd.DataFrame]:
    return _score_target(
        fold,
        train,
        test,
        target=target,
        model_specs={
            "base": base_predictors,
            "policy": (*base_predictors, policy_predictor),
            "speed_negative_control": (*base_predictors, speed_predictor),
            "accuracy_negative_control": (*base_predictors, accuracy_predictor),
        },
        config=config,
    )


def _score_target(
    fold: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target: str,
    model_specs: dict[str, tuple[str, ...]],
    config: dict[str, Any],
) -> list[pd.DataFrame]:
    out: list[pd.DataFrame] = []
    alpha = float(config["validation"]["ridge_alpha"])
    y_train = train[target].to_numpy(dtype=float)
    y_test = test[target].to_numpy(dtype=float)
    for model_id, predictors in model_specs.items():
        train_x, test_x = _design_matrices(train, test, predictors)
        fit = _fit_ridge_gaussian(train_x, y_train, alpha)
        prediction = test_x @ fit["coef"]
        log_density = _gaussian_log_density(y_test, prediction, fit["sigma2"])
        out.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "participant_id": test["participant_id"].astype(str).to_numpy(),
                    "row_index": test["row_index"].to_numpy(),
                    "target_id": target,
                    "model_id": model_id,
                    "log_density": log_density,
                    "squared_error": (y_test - prediction) ** 2,
                    "n_train": len(train),
                    "n_test": len(test),
                }
            )
        )
    return out


def _summarise_contrasts(
    scored: pd.DataFrame,
    config: dict[str, Any],
    *,
    contrast_specs: dict[str, tuple[str, str]],
    target_filter: tuple[str, ...],
    unit_label: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target in target_filter:
        wide = (
            scored.loc[scored["target_id"] == target]
            .pivot(index=["participant_id", "fold", "row_index"], columns="model_id", values="log_density")
            .reset_index()
        )
        for contrast_id, (num, den) in contrast_specs.items():
            rows.append(
                _bootstrap_delta_row(
                    wide,
                    target_id=target,
                    contrast_id=contrast_id,
                    numerator=num,
                    denominator=den,
                    config=config,
                    unit_label=unit_label,
                )
            )
    return pd.DataFrame(rows)


def _mean_cross_task_policy_contrast(scored: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for target in ("T_flanker", "T_stroop"):
        wide = (
            scored.loc[scored["target_id"] == target]
            .pivot(index=["participant_id", "fold", "row_index"], columns="model_id", values="log_density")
            .reset_index()
        )
        wide["delta"] = wide["policy"] - wide["base"]
        rows.append(wide[["participant_id", "delta"]])
    combined = pd.concat(rows, ignore_index=True)
    return pd.DataFrame(
        [
            _bootstrap_delta_values(
                combined,
                target_id="mean_cross_task_transport",
                contrast_id="policy_minus_base",
                config=config,
                unit_label="directional_session_rows",
            )
        ]
    )


def _bootstrap_delta_row(
    wide: pd.DataFrame,
    *,
    target_id: str,
    contrast_id: str,
    numerator: str,
    denominator: str,
    config: dict[str, Any],
    unit_label: str,
) -> dict[str, Any]:
    values = wide.loc[:, ["participant_id"]].copy()
    values["delta"] = wide[numerator] - wide[denominator]
    return _bootstrap_delta_values(
        values,
        target_id=target_id,
        contrast_id=contrast_id,
        config=config,
        unit_label=unit_label,
    )


def _bootstrap_delta_values(
    values: pd.DataFrame,
    *,
    target_id: str,
    contrast_id: str,
    config: dict[str, Any],
    unit_label: str,
) -> dict[str, Any]:
    participant_delta = values.groupby("participant_id", as_index=False)["delta"].mean()
    deltas = participant_delta["delta"].to_numpy(dtype=float)
    rng = np.random.default_rng(int(config["validation"]["bootstrap_seed"]))
    n_boot = int(config["validation"]["bootstrap_iterations"])
    alpha = (1.0 - float(config["validation"]["ci"])) / 2.0
    boot = np.array([np.mean(rng.choice(deltas, size=len(deltas), replace=True)) for _ in range(n_boot)])
    return {
        "target_id": target_id,
        "contrast_id": contrast_id,
        "n_participants": int(len(deltas)),
        unit_label: int(len(values)),
        "mean_delta_log_density": float(np.mean(values["delta"])),
        "participant_mean_delta_log_density": float(np.mean(deltas)),
        "median_participant_delta_log_density": float(np.median(deltas)),
        "ci_lower": float(np.quantile(boot, alpha)),
        "ci_upper": float(np.quantile(boot, 1.0 - alpha)),
        "positive_participant_rate": float(np.mean(deltas > 0.0)),
    }


def _residual_policy_diagnostics(raw: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = _complete_session_rows(raw, config)
    fold_ids = _participant_folds(
        rows[config["identity_columns"]["participant_id"]].astype(str),
        int(config["validation"]["n_folds"]),
        int(config["validation"]["split_seed"]),
    )
    diagnostics: list[pd.DataFrame] = []
    for fold in range(int(config["validation"]["n_folds"])):
        train_raw = rows.loc[fold_ids != fold].copy()
        test_raw = rows.loc[fold_ids == fold].copy()
        scaler = _fit_raw_scaler(train_raw, _raw_feature_columns(config))
        train = _session_design(train_raw, scaler, config)
        test = _session_design(test_raw, scaler, config)
        train_x, test_x = _design_matrices(train, test, ("K", "C_signal", "V"))
        fit = _fit_ridge_gaussian(train_x, train["T_common"].to_numpy(dtype=float), float(config["validation"]["ridge_alpha"]))
        prediction = test_x @ fit["coef"]
        diagnostics.append(
            pd.DataFrame(
                {
                    "participant_id": test["participant_id"].astype(str),
                    "T_common": test["T_common"].to_numpy(dtype=float),
                    "residual": test["T_common"].to_numpy(dtype=float) - prediction,
                    "K": test["K"].to_numpy(dtype=float),
                    "C_signal": test["C_signal"].to_numpy(dtype=float),
                    "V": test["V"].to_numpy(dtype=float),
                }
            )
        )
    all_diag = pd.concat(diagnostics, ignore_index=True)
    total_var = float(np.var(all_diag["T_common"], ddof=0))
    residual_var = float(np.var(all_diag["residual"], ddof=0))
    rows_out = [
        {
            "diagnostic": "residual_variance_fraction",
            "value": residual_var / total_var if total_var > 0 else 0.0,
        }
    ]
    for predictor in ("K", "C_signal", "V"):
        rows_out.append(
            {
                "diagnostic": f"correlation_T_common_with_{predictor}",
                "value": float(np.corrcoef(all_diag["T_common"], all_diag[predictor])[0, 1]),
            }
        )
    return pd.DataFrame(rows_out)


def _negative_control_summary(
    cross_contrasts: pd.DataFrame,
    persistence_contrasts: pd.DataFrame,
) -> pd.DataFrame:
    cross = cross_contrasts.loc[
        cross_contrasts["contrast_id"].isin(
            ("speed_negative_control_minus_base", "accuracy_negative_control_minus_base")
        )
    ].copy()
    cross["test_family"] = "cross_task_transport"
    persistence = persistence_contrasts.loc[
        persistence_contrasts["contrast_id"].isin(
            (
                "speed_persistence_negative_control_minus_base",
                "accuracy_persistence_negative_control_minus_base",
            )
        )
    ].copy()
    persistence["test_family"] = "next_session_persistence"
    return pd.concat([cross, persistence], ignore_index=True)


def _make_decision(
    cross_contrasts: pd.DataFrame,
    persistence_contrasts: pd.DataFrame,
    residual: pd.DataFrame,
    forbidden_used: set[str],
    config: dict[str, Any],
) -> dict[str, str]:
    rule = config["decision_rule"]
    residual_fraction = float(
        residual.loc[residual["diagnostic"] == "residual_variance_fraction", "value"].iloc[0]
    )
    cross = _contrast_lookup(cross_contrasts, "mean_cross_task_transport", "policy_minus_base")
    temporal = _contrast_lookup(persistence_contrasts, "T_common_next", "policy_persistence_minus_base")
    if forbidden_used:
        return {"decision": "inconclusive", "reason": "forbidden columns were used"}
    cross_supported = (
        cross["participant_mean_delta_log_density"] > float(rule["primary_mean_delta_minimum"])
        and cross["ci_lower"] >= float(rule["primary_ci_lower_tolerance"])
    )
    temporal_supported = (
        temporal["participant_mean_delta_log_density"] > float(rule["primary_mean_delta_minimum"])
        and temporal["ci_lower"] >= float(rule["primary_ci_lower_tolerance"])
    )
    if (
        residual_fraction >= float(rule["residual_variance_fraction_minimum_for_support"])
        and cross_supported
        and temporal_supported
    ):
        return {
            "decision": "policy_dimension_supported",
            "reason": "residual policy variance, cross-task transport and next-session persistence all met the pre-registered support rule",
        }
    if residual_fraction <= float(rule["residual_variance_fraction_maximum_for_not_supported"]):
        return {
            "decision": "policy_dimension_not_supported",
            "reason": "T_common left too little residual variance after K/C_signal/V",
        }
    if (
        cross["ci_upper"] <= float(rule["disconfirm_ci_upper_maximum"])
        and temporal["ci_upper"] <= float(rule["disconfirm_ci_upper_maximum"])
    ):
        return {
            "decision": "policy_dimension_not_supported",
            "reason": "both primary policy contrasts failed the pre-registered held-out predictive criterion",
        }
    return {
        "decision": "inconclusive",
        "reason": "cross-task and temporal evidence did not jointly satisfy support or disconfirmation rules",
    }


def _complete_session_rows(raw: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    required = {
        config["identity_columns"]["participant_id"],
        config["identity_columns"]["session_id"],
        config["identity_columns"]["session_order"],
        *_raw_feature_columns(config),
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ConfigValidationError("input is missing required columns: " + ", ".join(missing))
    return raw.loc[:, sorted(required)].dropna(subset=_raw_feature_columns(config)).copy()


def _adjacent_pairs(raw: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows = _complete_session_rows(raw, config)
    participant_col = config["identity_columns"]["participant_id"]
    session_col = config["identity_columns"]["session_id"]
    order_col = config["identity_columns"]["session_order"]
    order_values = config["identity_columns"]["session_order_values"]
    rows["_session_order_index"] = rows[order_col].map(order_values)
    rows = rows.sort_values([participant_col, "_session_order_index", session_col])
    records: list[dict[str, Any]] = []
    feature_columns = _raw_feature_columns(config)
    for participant_id, group in rows.groupby(participant_col, sort=True):
        group_records = group.to_dict("records")
        for current, nxt in zip(group_records, group_records[1:], strict=False):
            if int(nxt["_session_order_index"]) - int(current["_session_order_index"]) != 1:
                continue
            record: dict[str, Any] = {"participant_id": participant_id}
            for column in feature_columns:
                record[f"current__{column}"] = float(current[column])
                record[f"next__{column}"] = float(nxt[column])
            records.append(record)
    paired = pd.DataFrame.from_records(records)
    if paired.empty:
        raise ConfigValidationError("no adjacent pairs are available for policy persistence")
    paired["row_index"] = np.arange(len(paired), dtype=int)
    return paired


def _pair_side_raw(pairs: pd.DataFrame, sides: tuple[str, ...], config: dict[str, Any]) -> pd.DataFrame:
    frames = []
    for side in sides:
        frame = pairs[[f"{side}__{column}" for column in _raw_feature_columns(config)]].copy()
        frame.columns = _raw_feature_columns(config)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _session_design(raw: pd.DataFrame, scaler: dict[str, tuple[float, float]], config: dict[str, Any]) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "participant_id": raw[config["identity_columns"]["participant_id"]].astype(str).to_numpy(),
            "row_index": np.arange(len(raw), dtype=int),
        }
    )
    _add_common_components(out, raw, scaler, config, suffix="")
    return out


def _pair_design(pairs: pd.DataFrame, scaler: dict[str, tuple[float, float]], config: dict[str, Any]) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "participant_id": pairs["participant_id"].astype(str).to_numpy(),
            "row_index": pairs["row_index"].to_numpy(),
        }
    )
    for side in ("current", "next"):
        raw_side = pairs[[f"{side}__{column}" for column in _raw_feature_columns(config)]].copy()
        raw_side.columns = _raw_feature_columns(config)
        _add_common_components(out, raw_side, scaler, config, suffix=f"_{side}")
    return out


def _add_common_components(
    out: pd.DataFrame,
    raw: pd.DataFrame,
    scaler: dict[str, tuple[float, float]],
    config: dict[str, Any],
    *,
    suffix: str,
) -> None:
    out[f"K{suffix}"] = _composite(raw, config["feature_groups"]["K"]["columns"], scaler)
    out[f"K_without_Stroop_general{suffix}"] = _composite(
        raw,
        _without_task_general(config["feature_groups"]["K"]["columns"], "Stroop"),
        scaler,
    )
    out[f"K_without_Flanker_general{suffix}"] = _composite(
        raw,
        _without_task_general(config["feature_groups"]["K"]["columns"], "Flanker"),
        scaler,
    )
    out[f"C_signal{suffix}"] = _composite(raw, config["feature_groups"]["C_signal"]["columns"], scaler)
    out[f"V{suffix}"] = _composite(raw, config["feature_groups"]["V"]["columns"], scaler)
    for task in TASKS:
        task_lower = task.lower()
        acc = config["policy_construction"]["accuracy_columns"][task]
        rt = config["policy_construction"]["mean_rt_columns"][task]
        accuracy = _z(raw[acc].to_numpy(dtype=float), scaler[acc])
        speed = -_z(raw[rt].to_numpy(dtype=float), scaler[rt])
        out[f"accuracy_{task_lower}{suffix}"] = accuracy
        out[f"speed_{task_lower}{suffix}"] = speed
        out[f"T_{task_lower}{suffix}"] = (accuracy - speed) / np.sqrt(2.0)
    out[f"T_common{suffix}"] = (out[f"T_stroop{suffix}"] + out[f"T_flanker{suffix}"]) / 2.0
    out[f"speed_common{suffix}"] = (out[f"speed_stroop{suffix}"] + out[f"speed_flanker{suffix}"]) / 2.0
    out[f"accuracy_common{suffix}"] = (out[f"accuracy_stroop{suffix}"] + out[f"accuracy_flanker{suffix}"]) / 2.0


def _fit_raw_scaler(raw: pd.DataFrame, columns: list[str]) -> dict[str, tuple[float, float]]:
    scaler: dict[str, tuple[float, float]] = {}
    for column in columns:
        values = raw[column].to_numpy(dtype=float)
        scale = float(np.std(values, ddof=0))
        scaler[column] = (float(np.mean(values)), scale if scale > 0 else 1.0)
    return scaler


def _composite(
    raw: pd.DataFrame,
    columns: dict[str, int],
    scaler: dict[str, tuple[float, float]],
) -> np.ndarray:
    oriented = [float(sign) * _z(raw[column].to_numpy(dtype=float), scaler[column]) for column, sign in columns.items()]
    return np.mean(np.vstack(oriented), axis=0)


def _z(values: np.ndarray, mean_scale: tuple[float, float]) -> np.ndarray:
    mean, scale = mean_scale
    return (values - mean) / scale


def _without_task_general(columns: dict[str, int], task: str) -> dict[str, int]:
    task_lower = task.lower()
    blocked = {
        f"{task_lower}_accuracy",
        f"{task_lower}_mean_rt_ms",
        f"{task_lower}_throughput",
    }
    return {column: sign for column, sign in columns.items() if column not in blocked}


def _design_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    predictors: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    train_x = np.column_stack([np.ones(len(train)), *[train[p].to_numpy(dtype=float) for p in predictors]])
    test_x = np.column_stack([np.ones(len(test)), *[test[p].to_numpy(dtype=float) for p in predictors]])
    if train_x.shape[1] > 1:
        means = train_x[:, 1:].mean(axis=0)
        sds = np.where(train_x[:, 1:].std(axis=0) == 0, 1.0, train_x[:, 1:].std(axis=0))
        train_x[:, 1:] = (train_x[:, 1:] - means) / sds
        test_x[:, 1:] = (test_x[:, 1:] - means) / sds
    return train_x, test_x


def _raw_feature_columns(config: dict[str, Any]) -> list[str]:
    columns: set[str] = set()
    for group in config["feature_groups"].values():
        columns.update(group["columns"])
    columns.update(config["policy_construction"]["accuracy_columns"].values())
    columns.update(config["policy_construction"]["mean_rt_columns"].values())
    return sorted(columns)


def _forbidden_columns_used(config: dict[str, Any]) -> set[str]:
    forbidden = set(config.get("forbidden_columns", ()))
    used: set[str] = set()
    for group in config["feature_groups"].values():
        used.update(group["columns"])
    used.update(config["policy_construction"]["accuracy_columns"].values())
    used.update(config["policy_construction"]["mean_rt_columns"].values())
    return used.intersection(forbidden)


def _contrast_lookup(contrasts: pd.DataFrame, target_id: str, contrast_id: str) -> pd.Series:
    row = contrasts.loc[
        (contrasts["target_id"] == target_id) & (contrasts["contrast_id"] == contrast_id)
    ]
    if len(row) != 1:
        raise RuntimeError(f"missing contrast {target_id}/{contrast_id}")
    return row.iloc[0]


def _contrast_summary(contrasts: pd.DataFrame, target_id: str, contrast_id: str) -> dict[str, float]:
    row = _contrast_lookup(contrasts, target_id, contrast_id)
    return {
        "participant_mean_delta_log_density": float(row["participant_mean_delta_log_density"]),
        "ci_lower": float(row["ci_lower"]),
        "ci_upper": float(row["ci_upper"]),
        "positive_participant_rate": float(row["positive_participant_rate"]),
    }


def _render_report(
    summary: dict[str, Any],
    support: pd.DataFrame,
    cross: pd.DataFrame,
    persistence: pd.DataFrame,
    residual: pd.DataFrame,
    negative: pd.DataFrame,
) -> str:
    lines = [
        "# M5 Stage 1B Speed-Accuracy Policy Candidate Real-Data Analysis",
        "",
        "**Status:** exploratory Stage 1B policy-candidate analysis",
        "",
        "**Formal claims allowed:** false",
        "",
        "No Trident-G/APC/PACE validation claim, transfer claim, T_commit claim, Predictive Calibration claim, optimality claim, neural-criticality claim or cusp claim is made.",
        "",
        "## Question",
        "",
        "Is there a reproducible cross-task speed-accuracy policy dimension beyond K, C_signal and V?",
        "",
        "## Timing",
        "",
        "This hypothesis was formulated after Stage 1-3 review and was pre-registered before inspecting T_policy_candidate outcomes.",
        "",
        "## Data-Support Preflight",
        "",
        f"- Input rows: {summary['input_rows']}",
        f"- Input participants: {summary['input_participants']}",
        f"- Input checksum: `{summary['input_hash']}`",
        f"- Support gate passed: {str(summary['support_gate_passed']).lower()}",
        "",
        "| Task | Accuracy column | Mean RT column | Complete rows | Complete participants |",
        "|---|---|---|---:|---:|",
    ]
    for row in support.itertuples(index=False):
        lines.append(
            f"| {row.task} | {row.accuracy_column} | {row.mean_rt_column} | "
            f"{row.complete_speed_accuracy_rows} | {row.complete_speed_accuracy_participants} |"
        )
    lines.extend(
        [
            "",
            "## Pre-Registered Decision",
            "",
            f"Decision: **{summary['decision']}**",
            "",
            summary["decision_reason"],
            "",
            "## Residual Policy Variance",
            "",
            "| Diagnostic | Value |",
            "|---|---:|",
        ]
    )
    for row in residual.itertuples(index=False):
        lines.append(f"| {row.diagnostic} | {row.value:.5f} |")
    lines.extend(
        [
            "",
            "## Cross-Task Transport",
            "",
            "| Target | Contrast | Participant-mean delta log density | 95% CI | Positive-participant rate |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in cross.itertuples(index=False):
        lines.append(
            f"| {row.target_id} | {row.contrast_id} | {row.participant_mean_delta_log_density:.5f} | "
            f"[{row.ci_lower:.5f}, {row.ci_upper:.5f}] | {row.positive_participant_rate:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Next-Session Persistence",
            "",
            "| Target | Contrast | Participant-mean delta log density | 95% CI | Positive-participant rate |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in persistence.itertuples(index=False):
        lines.append(
            f"| {row.target_id} | {row.contrast_id} | {row.participant_mean_delta_log_density:.5f} | "
            f"[{row.ci_lower:.5f}, {row.ci_upper:.5f}] | {row.positive_participant_rate:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Negative Controls",
            "",
            "| Test family | Target | Contrast | Participant-mean delta log density | 95% CI |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in negative.itertuples(index=False):
        lines.append(
            f"| {row.test_family} | {row.target_id} | {row.contrast_id} | "
            f"{row.participant_mean_delta_log_density:.5f} | [{row.ci_lower:.5f}, {row.ci_upper:.5f}] |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- T_commit claim: false",
            "- Predictive Calibration claim: false",
            "- Optimality claim: false",
            "- PACE/profile/probability/candidate columns used: false",
            "- Dynamic-regime variables used: false",
            "- Transfer outcomes used: false",
            "- Criticality/cusp interpretation: false",
            "",
            "This result only evaluates a behavioural cross-task speed-accuracy policy candidate in the paired public source.",
        ]
    )
    return "\n".join(lines) + "\n"


def _manifest(summary: dict[str, Any], config: dict[str, Any], root: Path) -> dict[str, Any]:
    return {
        "analysis_id": summary["analysis_id"],
        "git_commit": summary["git_commit"],
        "config_hash": summary["config_hash"],
        "config_content_hash": summary["config_content_hash"],
        "input_hash": summary["input_hash"],
        "participant_isolated": bool(config["validation"]["participant_isolated"]),
        "split_seed": int(config["validation"]["split_seed"]),
        "bootstrap_seed": int(config["validation"]["bootstrap_seed"]),
        "formal_claims_allowed": False,
        "outputs_are_participant_free": True,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/public_stage1b_sat_policy_v1.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    result = run_stage1b_sat_policy_analysis(args.config, output_dir=args.output_dir)
    print(json.dumps(result.summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
