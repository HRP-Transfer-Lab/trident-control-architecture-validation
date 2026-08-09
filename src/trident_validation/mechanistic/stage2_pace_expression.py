"""Stage 2 descriptive PACE-expression increment analysis.

This module implements the second public-data mechanism analysis from the
M5/M6 ladder. It tests whether fold-derived descriptive PACE-expression
indicators add held-out predictive structure beyond continuous K/C_signal/V.

PACE expressions are not imported from upstream profile columns, are not latent
ontology claims and are not forced four-profile assignments.
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
    _build_adjacent_pairs,
    _fit_ridge_gaussian,
    _fold_composites,
    _forbidden_columns_used,
    _gaussian_log_density,
    _participant_folds,
    _required_mapping,
    _resolve_repo_path,
    _write_json,
)
from trident_validation.provenance import get_git_commit, hash_file, hash_mapping


PACE_COLUMNS = ("regulated", "brittle", "compensatory", "overloaded")


@dataclass(frozen=True)
class Stage2Result:
    """Participant-free Stage 2 analysis result."""

    summary: dict[str, Any]
    model_scores: pd.DataFrame
    contrasts: pd.DataFrame
    expression_rates: pd.DataFrame
    report_markdown: str


def run_stage2_pace_expression_analysis(
    config_path: str | Path = "config/public_stage2_pace_expression_v1.yaml",
    *,
    repo_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    write_outputs: bool = True,
) -> Stage2Result:
    """Run the pre-registered Stage 2 descriptive PACE-expression test."""

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
    forbidden_used = _forbidden_columns_used(config)
    pairs = _build_adjacent_pairs(raw, config)
    fold_ids = _participant_folds(
        pairs["participant_id"].astype(str),
        int(config["validation"]["n_folds"]),
        int(config["validation"]["split_seed"]),
    )
    model_scores, scored_rows, expression_rates = _score_models(pairs, fold_ids, config)
    contrasts = _summarise_contrasts(scored_rows, config)
    decision = _make_decision(contrasts, forbidden_used, config)

    summary = {
        "analysis_id": config["analysis"]["id"],
        "status": "completed",
        "question": config["analysis"]["question"],
        "decision": decision["decision"],
        "decision_reason": decision["reason"],
        "formal_claims_allowed": False,
        "trident_validation_claim_allowed": False,
        "transfer_outcomes_used": False,
        "upstream_profile_columns_used": False,
        "pace_ontology_claim_allowed": False,
        "forced_four_profile_claim_allowed": False,
        "dynamic_regime_columns_used": False,
        "neural_criticality_claim_allowed": False,
        "cusp_claim_allowed": False,
        "input_rows": int(len(raw)),
        "complete_adjacent_pairs": int(len(pairs)),
        "participants_with_adjacent_pairs": int(pairs["participant_id"].nunique()),
        "n_folds": int(config["validation"]["n_folds"]),
        "forbidden_columns_present": [c for c in config["forbidden_columns"] if c in raw.columns],
        "forbidden_columns_used": sorted(forbidden_used),
        "input_hash": observed_hash,
        "config_hash": hash_file(root / config_file) if not config_file.is_absolute() else hash_file(config_file),
        "config_content_hash": hash_mapping(config),
        "git_commit": get_git_commit(root),
        "primary_contrast": _contrast_summary(
            contrasts,
            "Y_behavior_next",
            "continuous_K_C_V_plus_descriptive_PACE_expression_minus_continuous_K_C_V",
        ),
        "negative_control_contrast": _contrast_summary(
            contrasts,
            "Y_behavior_next",
            "PACE_expression_only_negative_control_minus_continuous_K_C_V",
        ),
        "expression_rate_summary": _expression_rate_summary(expression_rates),
    }
    report = _render_report(summary, model_scores, contrasts, expression_rates)

    if write_outputs:
        out_dir = Path(output_dir) if output_dir is not None else root / config["outputs"]["output_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_json(out_dir / "stage2_summary.json", summary)
        model_scores.to_csv(out_dir / "model_scores.csv", index=False)
        contrasts.to_csv(out_dir / "paired_contrasts.csv", index=False)
        expression_rates.to_csv(out_dir / "pace_expression_rates.csv", index=False)
        (out_dir / "stage2_report.md").write_text(report, encoding="utf-8", newline="\n")
        report_path = root / str(config["outputs"]["report_md"])
        report_path.write_text(report, encoding="utf-8", newline="\n")

    return Stage2Result(
        summary=summary,
        model_scores=model_scores,
        contrasts=contrasts,
        expression_rates=expression_rates,
        report_markdown=report,
    )


def _validate_config(config: dict[str, Any]) -> None:
    analysis = _required_mapping(config, "analysis")
    if analysis.get("id") != "public_stage2_pace_expression_v1":
        raise ConfigValidationError("analysis.id must be public_stage2_pace_expression_v1")
    if analysis.get("status") != "pre_outcome_registered_analysis":
        raise ConfigValidationError("analysis.status must be pre_outcome_registered_analysis")
    for field in (
        "formal_claims_allowed",
        "trident_validation_claim_allowed",
        "transfer_outcomes_allowed",
        "pace_ontology_claim_allowed",
        "forced_four_profile_claim_allowed",
        "dynamic_regime_allowed",
        "neural_criticality_claim_allowed",
        "cusp_claim_allowed",
    ):
        if analysis.get(field) is not False:
            raise ConfigValidationError(f"analysis.{field} must be false")

    pace = _required_mapping(config, "pace_expression")
    if pace.get("derived_inside_training_folds_only") is not True:
        raise ConfigValidationError("PACE expressions must be derived inside training folds")
    if pace.get("descriptive_only") is not True:
        raise ConfigValidationError("PACE expressions must be descriptive only")
    if tuple(pace.get("indicators", {}).keys()) != PACE_COLUMNS:
        raise ConfigValidationError("PACE expression indicators must be regulated/brittle/compensatory/overloaded")

    forbidden = set(config.get("forbidden_columns", ()))
    for group_name, group in _required_mapping(config, "feature_groups").items():
        columns = group.get("columns")
        if not isinstance(columns, dict) or not columns:
            raise ConfigValidationError(f"feature_groups.{group_name}.columns must be a mapping")
        forbidden_features = forbidden.intersection(columns)
        if forbidden_features:
            raise ConfigValidationError(
                f"feature_groups.{group_name}.columns includes forbidden columns: "
                + ", ".join(sorted(forbidden_features))
            )
    expected_models = {
        "continuous_K_C_V",
        "continuous_K_C_V_plus_descriptive_PACE_expression",
        "PACE_expression_only_negative_control",
    }
    if set(_required_mapping(config, "candidate_models")) != expected_models:
        raise ConfigValidationError("candidate_models must match the Stage 2 ladder")
    if config["validation"].get("participant_isolated") is not True:
        raise ConfigValidationError("participant-isolated validation is required")


def _score_models(
    pairs: pd.DataFrame,
    fold_ids: np.ndarray,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_rows: list[dict[str, Any]] = []
    scored_rows: list[pd.DataFrame] = []
    expression_rows: list[dict[str, Any]] = []
    n_folds = int(config["validation"]["n_folds"])
    alpha = float(config["validation"]["ridge_alpha"])

    for fold in range(n_folds):
        train_pairs = pairs.loc[fold_ids != fold].copy()
        test_pairs = pairs.loc[fold_ids == fold].copy()
        train_comp, test_comp = _fold_composites(train_pairs, test_pairs, config)
        thresholds = {
            "K": float(train_comp["K_current"].median()),
            "C_signal": float(train_comp["C_signal_current"].median()),
            "V": float(train_comp["V_current"].median()),
        }
        train_design = _add_pace_expression_indicators(train_comp, thresholds)
        test_design = _add_pace_expression_indicators(test_comp, thresholds)

        for expression in PACE_COLUMNS:
            expression_rows.append(
                {
                    "fold": fold,
                    "expression": expression,
                    "train_rate": float(train_design[expression].mean()),
                    "test_rate": float(test_design[expression].mean()),
                }
            )
        expression_rows.append(
            {
                "fold": fold,
                "expression": "none_of_four",
                "train_rate": float(1.0 - train_design[list(PACE_COLUMNS)].any(axis=1).mean()),
                "test_rate": float(1.0 - test_design[list(PACE_COLUMNS)].any(axis=1).mean()),
            }
        )

        for model_id, model in config["candidate_models"].items():
            predictors = list(model["predictors"])
            train_x, test_x = _design_matrices(train_design, test_design, predictors)
            for target_id, target in config["targets"].items():
                target_y_train = _target_vector(train_design, target["groups"])
                target_y_test = _target_vector(test_design, target["groups"])
                fit = _fit_ridge_gaussian(train_x, target_y_train, alpha)
                prediction = test_x @ fit["coef"]
                log_density = _gaussian_log_density(target_y_test, prediction, fit["sigma2"])
                model_rows.append(
                    {
                        "fold": fold,
                        "model_id": model_id,
                        "target_id": target_id,
                        "n_train_pairs": int(len(train_design)),
                        "n_test_pairs": int(len(test_design)),
                        "mean_log_density": float(np.mean(log_density)),
                        "mse": float(np.mean((target_y_test - prediction) ** 2)),
                        "sigma2_train": float(fit["sigma2"]),
                    }
                )
                scored_rows.append(
                    pd.DataFrame(
                        {
                            "fold": fold,
                            "row_index": test_design["row_index"].to_numpy(),
                            "model_id": model_id,
                            "target_id": target_id,
                            "log_density": log_density,
                            "squared_error": (target_y_test - prediction) ** 2,
                        }
                    )
                )

    return (
        pd.DataFrame(model_rows),
        pd.concat(scored_rows, ignore_index=True),
        pd.DataFrame(expression_rows),
    )


def _add_pace_expression_indicators(
    composites: pd.DataFrame,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    out = composites.copy()
    k_high = out["K_current"] >= thresholds["K"]
    c_high = out["C_signal_current"] >= thresholds["C_signal"]
    v_high = out["V_current"] >= thresholds["V"]
    out["regulated"] = (k_high & c_high & v_high).astype(float)
    out["brittle"] = (k_high & c_high & ~v_high).astype(float)
    out["compensatory"] = (~k_high & (c_high | v_high)).astype(float)
    out["overloaded"] = (~k_high & ~c_high & ~v_high).astype(float)
    return out


def _design_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    predictors: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    train_x = _raw_design_matrix(train, predictors)
    test_x = _raw_design_matrix(test, predictors)
    if train_x.shape[1] > 1:
        means = train_x[:, 1:].mean(axis=0)
        sds = np.where(train_x[:, 1:].std(axis=0) == 0, 1.0, train_x[:, 1:].std(axis=0))
        train_x[:, 1:] = (train_x[:, 1:] - means) / sds
        test_x[:, 1:] = (test_x[:, 1:] - means) / sds
    return train_x, test_x


def _raw_design_matrix(composites: pd.DataFrame, predictors: list[str]) -> np.ndarray:
    columns = [np.ones(len(composites))]
    for predictor in predictors:
        columns.append(composites[f"{predictor}_current"].to_numpy() if predictor in {"K", "C_signal", "V"} else composites[predictor].to_numpy())
    return np.column_stack(columns)


def _target_vector(composites: pd.DataFrame, groups: list[str]) -> np.ndarray:
    columns = [composites[f"{group}_next"].to_numpy(dtype=float) for group in groups]
    return np.mean(np.vstack(columns), axis=0)


def _summarise_contrasts(scored_rows: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    contrast_specs = {
        "continuous_K_C_V_plus_descriptive_PACE_expression_minus_continuous_K_C_V": (
            "continuous_K_C_V_plus_descriptive_PACE_expression",
            "continuous_K_C_V",
        ),
        "PACE_expression_only_negative_control_minus_continuous_K_C_V": (
            "PACE_expression_only_negative_control",
            "continuous_K_C_V",
        ),
    }
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(config["validation"]["bootstrap_seed"]))
    n_boot = int(config["validation"]["bootstrap_iterations"])
    alpha = (1.0 - float(config["validation"]["ci"])) / 2.0
    for target_id in config["targets"]:
        target_scores = scored_rows.loc[scored_rows["target_id"] == target_id]
        wide = target_scores.pivot(index=["fold", "row_index"], columns="model_id", values="log_density").reset_index()
        for contrast_id, (numerator, denominator) in contrast_specs.items():
            delta = (wide[numerator] - wide[denominator]).to_numpy(dtype=float)
            boot = np.array([np.mean(rng.choice(delta, size=len(delta), replace=True)) for _ in range(n_boot)])
            rows.append(
                {
                    "target_id": target_id,
                    "contrast_id": contrast_id,
                    "n_pairs": int(len(delta)),
                    "mean_delta_log_density": float(np.mean(delta)),
                    "median_delta_log_density": float(np.median(delta)),
                    "ci_lower": float(np.quantile(boot, alpha)),
                    "ci_upper": float(np.quantile(boot, 1.0 - alpha)),
                    "positive_pair_rate": float(np.mean(delta > 0.0)),
                }
            )
    return pd.DataFrame(rows)


def _make_decision(
    contrasts: pd.DataFrame,
    forbidden_used: set[str],
    config: dict[str, Any],
) -> dict[str, str]:
    rule = config["decision_rule"]
    primary = _contrast_lookup(
        contrasts,
        "Y_behavior_next",
        "continuous_K_C_V_plus_descriptive_PACE_expression_minus_continuous_K_C_V",
    )
    negative = _contrast_lookup(
        contrasts,
        "Y_behavior_next",
        "PACE_expression_only_negative_control_minus_continuous_K_C_V",
    )
    if forbidden_used:
        return {"decision": "inconclusive", "reason": "forbidden columns were used"}
    support = (
        primary["mean_delta_log_density"] > float(rule["primary_increment_mean_minimum"])
        and primary["ci_lower"] >= float(rule["primary_increment_ci_lower_tolerance"])
        and (
            not bool(rule["negative_control_must_not_exceed_continuous_baseline"])
            or negative["mean_delta_log_density"] <= 0.0
        )
    )
    if support:
        return {
            "decision": "pace_expression_increment_supported",
            "reason": "descriptive PACE indicators improved next-session behaviour beyond continuous K/C/V under the pre-registered tolerance",
        }
    if primary["ci_upper"] <= float(rule["disconfirm_primary_increment_ci_upper_maximum"]):
        return {
            "decision": "pace_expression_increment_not_supported",
            "reason": "descriptive PACE increment failed the pre-registered held-out predictive criterion",
        }
    return {
        "decision": "inconclusive",
        "reason": "PACE increment did not satisfy either the support or disconfirmation rule",
    }


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
        "mean_delta_log_density": float(row["mean_delta_log_density"]),
        "ci_lower": float(row["ci_lower"]),
        "ci_upper": float(row["ci_upper"]),
        "positive_pair_rate": float(row["positive_pair_rate"]),
    }


def _expression_rate_summary(expression_rates: pd.DataFrame) -> dict[str, float]:
    return {
        str(row.expression): float(row.test_rate)
        for row in expression_rates.groupby("expression", as_index=False)["test_rate"].mean().itertuples(index=False)
    }


def _render_report(
    summary: dict[str, Any],
    model_scores: pd.DataFrame,
    contrasts: pd.DataFrame,
    expression_rates: pd.DataFrame,
) -> str:
    score_table = (
        model_scores.groupby(["target_id", "model_id"], as_index=False)["mean_log_density"]
        .mean()
        .sort_values(["target_id", "model_id"])
    )
    expression_table = (
        expression_rates.groupby("expression", as_index=False)[["train_rate", "test_rate"]]
        .mean()
        .sort_values("expression")
    )
    lines = [
        "# M5 Stage 2 Descriptive PACE-Expression Real-Data Analysis",
        "",
        "**Status:** exploratory Stage 2 mechanism analysis",
        "",
        "**Formal claims allowed:** false",
        "",
        "No Trident-G/APC/PACE validation claim, transfer claim, neural-criticality claim or cusp claim is made.",
        "",
        "## Question",
        "",
        "Do regulated / brittle / compensatory / overloaded expressions add predictive structure beyond continuous K/C/V?",
        "",
        "## Design",
        "",
        "The analysis uses the frozen paired public session-level source. Adjacent current sessions predict next-session behaviour under participant-isolated 5-fold validation.",
        "",
        "Continuous K/C_signal/V composites are fold-scaled from observed task features. Descriptive PACE-expression indicators are derived inside each training fold from training-fold medians and then applied to held-out participants. They are threshold descriptors, not latent classes.",
        "",
        "The upstream profile/probability/candidate columns are present in the source but are not used.",
        "",
        "## Sample",
        "",
        f"- Input participant-session rows: {summary['input_rows']}",
        f"- Complete adjacent session pairs: {summary['complete_adjacent_pairs']}",
        f"- Participants with adjacent pairs: {summary['participants_with_adjacent_pairs']}",
        f"- Input checksum: `{summary['input_hash']}`",
        "",
        "## Pre-Registered Decision",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        summary["decision_reason"],
        "",
        "## Primary Predictive Contrasts",
        "",
        "| Target | Contrast | Mean delta log density | 95% CI | Positive-pair rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in contrasts.itertuples(index=False):
        lines.append(
            f"| {row.target_id} | {row.contrast_id} | {row.mean_delta_log_density:.5f} | "
            f"[{row.ci_lower:.5f}, {row.ci_upper:.5f}] | {row.positive_pair_rate:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Descriptive Expression Rates",
            "",
            "| Expression | Train rate | Held-out rate |",
            "|---|---:|---:|",
        ]
    )
    for row in expression_table.itertuples(index=False):
        lines.append(f"| {row.expression} | {row.train_rate:.3f} | {row.test_rate:.3f} |")
    lines.extend(
        [
            "",
            "## Mean Held-Out Log Density",
            "",
            "| Target | Model | Mean held-out log density |",
            "|---|---:|---:|",
        ]
    )
    for row in score_table.itertuples(index=False):
        lines.append(f"| {row.target_id} | {row.model_id} | {row.mean_log_density:.5f} |")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Upstream PACE/profile/probability/candidate columns used: false",
            "- PACE expression ontology claim: false",
            "- Forced four-profile claim: false",
            "- Dynamic-regime variables used: false",
            "- Transfer outcomes used: false",
            "- Criticality/cusp interpretation: false",
            "",
            "This Stage 2 result only tests a descriptive nonlinear increment over continuous K/C/V in the paired public source.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/public_stage2_pace_expression_v1.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    result = run_stage2_pace_expression_analysis(args.config, output_dir=args.output_dir)
    print(json.dumps(result.summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
