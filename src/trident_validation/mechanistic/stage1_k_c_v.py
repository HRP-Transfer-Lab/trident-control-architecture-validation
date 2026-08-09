"""Stage 1 real-data K / C_signal / vigilance separation analysis.

This module implements the first public-data mechanism analysis from the
M5/M6 ladder. It asks whether current-session general performance (K),
conflict-control signal (C_signal) and SART vigilance/readiness (V) can be
separated when predicting adjacent next-session behaviour.

The analysis is claim-bounded: it does not use PACE/profile labels, dynamic
regime labels, transfer outcomes, neural criticality language or cusp claims.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from math import erf, sqrt
from pathlib import Path
from typing import Any
import hashlib

import numpy as np
import pandas as pd

from trident_validation.config import ConfigValidationError, load_yaml_config
from trident_validation.provenance import get_git_commit, hash_file, hash_mapping


@dataclass(frozen=True)
class Stage1Result:
    """Participant-free Stage 1 analysis result."""

    summary: dict[str, Any]
    model_scores: pd.DataFrame
    contrasts: pd.DataFrame
    residual_separation: dict[str, float]
    report_markdown: str


def run_stage1_k_c_v_analysis(
    config_path: str | Path = "config/public_stage1_k_c_v_v1.yaml",
    *,
    repo_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    write_outputs: bool = True,
) -> Stage1Result:
    """Run the pre-registered Stage 1 K/C/V separation analysis."""

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
        pairs["participant_id"].astype(str), int(config["validation"]["n_folds"]), int(config["validation"]["split_seed"])
    )

    model_scores, scored_rows = _score_models(pairs, fold_ids, config)
    contrasts = _summarise_contrasts(scored_rows, config)
    residual = _residual_separation(scored_rows)
    decision = _make_decision(contrasts, residual, forbidden_used, config)

    summary = {
        "analysis_id": config["analysis"]["id"],
        "status": "completed",
        "question": config["analysis"]["question"],
        "decision": decision["decision"],
        "decision_reason": decision["reason"],
        "formal_claims_allowed": False,
        "trident_validation_claim_allowed": False,
        "transfer_outcomes_used": False,
        "pace_columns_used": False,
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
        "primary_contrasts": _primary_contrast_summary(contrasts),
        "residual_separation": residual,
    }
    report = _render_report(summary, model_scores, contrasts, residual, config)

    if write_outputs:
        out_dir = Path(output_dir) if output_dir is not None else root / config["outputs"]["output_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_json(out_dir / "stage1_summary.json", summary)
        model_scores.to_csv(out_dir / "model_scores.csv", index=False)
        contrasts.to_csv(out_dir / "paired_contrasts.csv", index=False)
        (out_dir / "stage1_report.md").write_text(report, encoding="utf-8", newline="\n")
        report_path = root / str(config["outputs"]["report_md"])
        report_path.write_text(report, encoding="utf-8", newline="\n")

    return Stage1Result(
        summary=summary,
        model_scores=model_scores,
        contrasts=contrasts,
        residual_separation=residual,
        report_markdown=report,
    )


def _validate_config(config: dict[str, Any]) -> None:
    analysis = _required_mapping(config, "analysis")
    if analysis.get("id") != "public_stage1_k_c_v_v1":
        raise ConfigValidationError("analysis.id must be public_stage1_k_c_v_v1")
    if analysis.get("status") != "pre_outcome_registered_analysis":
        raise ConfigValidationError("analysis.status must be pre_outcome_registered_analysis")
    for field in (
        "formal_claims_allowed",
        "trident_validation_claim_allowed",
        "transfer_outcomes_allowed",
        "pace_allowed",
        "dynamic_regime_allowed",
        "neural_criticality_claim_allowed",
        "cusp_claim_allowed",
    ):
        if analysis.get(field) is not False:
            raise ConfigValidationError(f"analysis.{field} must be false")

    groups = _required_mapping(config, "feature_groups")
    if set(groups) != {"K", "C_signal", "V"}:
        raise ConfigValidationError("feature_groups must define exactly K, C_signal and V")
    forbidden = set(config.get("forbidden_columns", ()))
    for group_name, group in groups.items():
        columns = group.get("columns")
        if not isinstance(columns, dict) or not columns:
            raise ConfigValidationError(f"feature_groups.{group_name}.columns must be a mapping")
        forbidden_features = forbidden.intersection(columns)
        if forbidden_features:
            raise ConfigValidationError(
                f"feature_groups.{group_name}.columns includes forbidden columns: "
                + ", ".join(sorted(forbidden_features))
            )
        for column, sign in columns.items():
            if int(sign) not in (-1, 1):
                raise ConfigValidationError(f"{column} orientation must be -1 or 1")

    models = _required_mapping(config, "candidate_models")
    expected_models = {
        "K_only",
        "K_plus_C_signal",
        "K_plus_V",
        "K_plus_C_signal_plus_V",
        "K_plus_C_signal_by_V",
    }
    if set(models) != expected_models:
        raise ConfigValidationError("candidate_models must match the Stage 1 ladder")
    if config["validation"].get("participant_isolated") is not True:
        raise ConfigValidationError("participant-isolated validation is required")


def _build_adjacent_pairs(raw: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    id_cols = config["identity_columns"]
    participant_col = id_cols["participant_id"]
    session_col = id_cols["session_id"]
    order_col = id_cols["session_order"]
    order_values = id_cols["session_order_values"]
    required_columns = {participant_col, session_col, order_col}
    for group in config["feature_groups"].values():
        required_columns.update(group["columns"])
    missing = sorted(required_columns.difference(raw.columns))
    if missing:
        raise ConfigValidationError("input is missing required columns: " + ", ".join(missing))

    df = raw.loc[:, sorted(required_columns)].copy()
    df["_session_order_index"] = df[order_col].map(order_values)
    if df["_session_order_index"].isna().any():
        bad = sorted(df.loc[df["_session_order_index"].isna(), order_col].dropna().astype(str).unique())
        raise ConfigValidationError("unknown session order values: " + ", ".join(bad))
    feature_columns = _all_feature_columns(config)
    df = df.dropna(subset=feature_columns)
    df = df.sort_values([participant_col, "_session_order_index", session_col])

    pairs: list[dict[str, Any]] = []
    for participant_id, group in df.groupby(participant_col, sort=True):
        records = group.to_dict("records")
        for current, nxt in zip(records, records[1:], strict=False):
            if int(nxt["_session_order_index"]) - int(current["_session_order_index"]) != 1:
                continue
            record: dict[str, Any] = {
                "participant_id": participant_id,
                "current_order": int(current["_session_order_index"]),
                "next_order": int(nxt["_session_order_index"]),
            }
            for column in feature_columns:
                record[f"current__{column}"] = float(current[column])
                record[f"next__{column}"] = float(nxt[column])
            pairs.append(record)

    paired = pd.DataFrame.from_records(pairs)
    if paired.empty:
        raise ConfigValidationError("no complete adjacent session pairs are available")
    minimum_sessions = int(config["pairing"]["minimum_observed_sessions_per_participant"])
    if minimum_sessions != 2:
        raise ConfigValidationError("Stage 1 adjacent-pair analysis requires minimum two sessions")
    return paired


def _participant_folds(participants: pd.Series, n_folds: int, seed: int) -> np.ndarray:
    unique = sorted(participants.unique())
    fold_by_participant = {
        participant: _stable_int_hash(f"{seed}:{participant}") % n_folds
        for participant in unique
    }
    folds = participants.map(fold_by_participant).to_numpy(dtype=int)
    if len(set(folds)) != n_folds:
        raise ConfigValidationError("participant fold assignment produced an empty fold")
    return folds


def _score_models(
    pairs: pd.DataFrame,
    fold_ids: np.ndarray,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_rows: list[dict[str, Any]] = []
    scored_rows: list[pd.DataFrame] = []
    n_folds = int(config["validation"]["n_folds"])
    alpha = float(config["validation"]["ridge_alpha"])

    for fold in range(n_folds):
        train_mask = fold_ids != fold
        test_mask = fold_ids == fold
        train_pairs = pairs.loc[train_mask].copy()
        test_pairs = pairs.loc[test_mask].copy()
        train_comp, test_comp = _fold_composites(train_pairs, test_pairs, config)

        for model_id, model in config["candidate_models"].items():
            predictors = list(model["predictors"])
            train_x, test_x = _design_matrices(train_comp, test_comp, predictors)
            for target_id, target in config["targets"].items():
                target_column = f"{target['group']}_next"
                fit = _fit_ridge_gaussian(train_x, train_comp[target_column].to_numpy(), alpha)
                test_y = test_comp[target_column].to_numpy()
                log_density = _gaussian_log_density(test_y, test_x @ fit["coef"], fit["sigma2"])
                model_rows.append(
                    {
                        "fold": fold,
                        "model_id": model_id,
                        "target_id": target_id,
                        "n_train_pairs": int(len(train_comp)),
                        "n_test_pairs": int(len(test_comp)),
                        "mean_log_density": float(np.mean(log_density)),
                        "mse": float(np.mean((test_y - test_x @ fit["coef"]) ** 2)),
                        "sigma2_train": float(fit["sigma2"]),
                    }
                )
                scored_rows.append(
                    pd.DataFrame(
                        {
                            "fold": fold,
                            "row_index": test_comp["row_index"].to_numpy(),
                            "model_id": model_id,
                            "target_id": target_id,
                            "log_density": log_density,
                            "squared_error": (test_y - test_x @ fit["coef"]) ** 2,
                        }
                    )
                )

        base = test_comp.loc[:, ["row_index", "K_current", "C_signal_current", "V_current"]].copy()
        base["fold"] = fold
        base["model_id"] = "__composites__"
        base["target_id"] = "__none__"
        base["log_density"] = np.nan
        base["squared_error"] = np.nan
        scored_rows.append(base)

    score_df = pd.DataFrame(model_rows)
    row_df = pd.concat(scored_rows, ignore_index=True)
    return score_df, row_df


def _fold_composites(
    train_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = _all_feature_columns(config)
    train_session_values = []
    for prefix in ("current__", "next__"):
        block = train_pairs[[prefix + col for col in feature_columns]].copy()
        block.columns = feature_columns
        train_session_values.append(block)
    train_sessions = pd.concat(train_session_values, ignore_index=True)
    means = train_sessions.mean(axis=0)
    sds = train_sessions.std(axis=0, ddof=0).replace(0, 1.0)

    return (
        _apply_composites(train_pairs, config, means, sds),
        _apply_composites(test_pairs, config, means, sds),
    )


def _apply_composites(
    pairs: pd.DataFrame,
    config: dict[str, Any],
    means: pd.Series,
    sds: pd.Series,
) -> pd.DataFrame:
    out = pd.DataFrame({"row_index": np.arange(len(pairs), dtype=int)})
    for group_id, group in config["feature_groups"].items():
        for side in ("current", "next"):
            oriented = []
            for column, sign in group["columns"].items():
                z = (pairs[f"{side}__{column}"].to_numpy(dtype=float) - means[column]) / sds[column]
                oriented.append(float(sign) * z)
            out[f"{group_id}_{side}"] = np.mean(np.vstack(oriented), axis=0)
    return out


def _design_matrices(
    train_composites: pd.DataFrame,
    test_composites: pd.DataFrame,
    predictors: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    train_x = _raw_design_matrix(train_composites, predictors)
    test_x = _raw_design_matrix(test_composites, predictors)
    if train_x.shape[1] > 1:
        means = train_x[:, 1:].mean(axis=0)
        sds = np.where(train_x[:, 1:].std(axis=0) == 0, 1.0, train_x[:, 1:].std(axis=0))
        train_x[:, 1:] = (train_x[:, 1:] - means) / sds
        test_x[:, 1:] = (test_x[:, 1:] - means) / sds
    return train_x, test_x


def _raw_design_matrix(composites: pd.DataFrame, predictors: list[str]) -> np.ndarray:
    columns = [np.ones(len(composites))]
    for predictor in predictors:
        if predictor == "C_signal_by_V":
            values = composites["C_signal_current"].to_numpy() * composites["V_current"].to_numpy()
        else:
            values = composites[f"{predictor}_current"].to_numpy()
        columns.append(values)
    return np.column_stack(columns)


def _fit_ridge_gaussian(x: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, Any]:
    penalty = np.eye(x.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    residual = y - x @ coef
    dof = max(1, x.shape[0] - x.shape[1])
    sigma2 = max(float(np.sum(residual**2) / dof), 1e-8)
    return {"coef": coef, "sigma2": sigma2}


def _gaussian_log_density(y: np.ndarray, mean: np.ndarray, sigma2: float) -> np.ndarray:
    return -0.5 * (np.log(2.0 * np.pi * sigma2) + ((y - mean) ** 2) / sigma2)


def _summarise_contrasts(scored_rows: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    contrast_specs = {
        "K_plus_C_signal_minus_K_only": ("K_plus_C_signal", "K_only"),
        "K_plus_V_minus_K_only": ("K_plus_V", "K_only"),
        "K_plus_C_signal_plus_V_minus_K_only": ("K_plus_C_signal_plus_V", "K_only"),
        "K_plus_C_signal_by_V_minus_additive": ("K_plus_C_signal_by_V", "K_plus_C_signal_plus_V"),
    }
    rng = np.random.default_rng(int(config["validation"]["bootstrap_seed"]))
    n_boot = int(config["validation"]["bootstrap_iterations"])
    alpha = (1.0 - float(config["validation"]["ci"])) / 2.0
    scored = scored_rows.loc[scored_rows["model_id"] != "__composites__"].copy()
    for target_id in config["targets"]:
        target_scores = scored.loc[scored["target_id"] == target_id]
        index_cols = ["fold", "row_index"]
        wide = target_scores.pivot(index=index_cols, columns="model_id", values="log_density").reset_index()
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


def _residual_separation(scored_rows: pd.DataFrame) -> dict[str, float]:
    comp = (
        scored_rows.loc[scored_rows["model_id"] == "__composites__", ["fold", "row_index", "K_current", "C_signal_current", "V_current"]]
        .drop_duplicates(["fold", "row_index"])
        .copy()
    )
    x = np.column_stack([np.ones(len(comp)), comp["K_current"].to_numpy()])
    c = comp["C_signal_current"].to_numpy()
    v = comp["V_current"].to_numpy()
    beta_c = np.linalg.lstsq(x, c, rcond=None)[0]
    beta_v = np.linalg.lstsq(x, v, rcond=None)[0]
    c_resid = c - x @ beta_c
    v_resid = v - x @ beta_v
    c_var = float(np.var(c, ddof=0))
    v_var = float(np.var(v, ddof=0))
    c_resid_var = float(np.var(c_resid, ddof=0))
    v_resid_var = float(np.var(v_resid, ddof=0))
    corr = float(np.corrcoef(c_resid, v_resid)[0, 1])
    return {
        "C_signal_residual_variance_fraction_after_K": c_resid_var / c_var if c_var > 0 else 0.0,
        "V_residual_variance_fraction_after_K": v_resid_var / v_var if v_var > 0 else 0.0,
        "C_signal_V_residual_correlation_after_K": corr,
        "C_signal_V_residual_abs_correlation_after_K": abs(corr),
    }


def _make_decision(
    contrasts: pd.DataFrame,
    residual: dict[str, float],
    forbidden_used: set[str],
    config: dict[str, Any],
) -> dict[str, str]:
    rule = config["decision_rule"]
    c_resid = residual["C_signal_residual_variance_fraction_after_K"]
    v_resid = residual["V_residual_variance_fraction_after_K"]
    abs_corr = residual["C_signal_V_residual_abs_correlation_after_K"]
    c_increment = _contrast_lookup(contrasts, "C_signal_next", "K_plus_C_signal_minus_K_only")
    v_increment = _contrast_lookup(contrasts, "V_next", "K_plus_V_minus_K_only")

    if forbidden_used:
        return {"decision": "inconclusive", "reason": "forbidden columns were used"}

    support_residual = (
        c_resid >= float(rule["residual_variance_fraction_minimum_for_support"])
        and v_resid >= float(rule["residual_variance_fraction_minimum_for_support"])
        and abs_corr <= float(rule["residual_abs_correlation_maximum_for_support"])
    )
    support_increment = (
        c_increment["mean_delta_log_density"] > float(rule["unique_increment_mean_minimum"])
        and v_increment["mean_delta_log_density"] > float(rule["unique_increment_mean_minimum"])
        and c_increment["ci_lower"] >= float(rule["unique_increment_ci_lower_tolerance"])
        and v_increment["ci_lower"] >= float(rule["unique_increment_ci_lower_tolerance"])
    )
    if support_residual and support_increment:
        return {
            "decision": "separation_supported",
            "reason": "C_signal and V retain residual variance after K and each improves its own next-session target under participant-isolated scoring",
        }

    disconfirm_residual = (
        c_resid <= float(rule["disconfirm_residual_variance_fraction_maximum"])
        or v_resid <= float(rule["disconfirm_residual_variance_fraction_maximum"])
        or abs_corr >= float(rule["disconfirm_residual_abs_correlation_minimum"])
    )
    disconfirm_increment = (
        c_increment["ci_upper"] <= float(rule["disconfirm_increment_ci_upper_maximum"])
        and v_increment["ci_upper"] <= float(rule["disconfirm_increment_ci_upper_maximum"])
    )
    if disconfirm_residual or disconfirm_increment:
        return {
            "decision": "separation_not_supported",
            "reason": "pre-registered residual redundancy or unique-prediction failure threshold was met",
        }
    return {
        "decision": "inconclusive",
        "reason": "evidence did not satisfy either the support or disconfirmation rule",
    }


def _primary_contrast_summary(contrasts: pd.DataFrame) -> dict[str, dict[str, float]]:
    primary = {
        "C_signal_next__K_plus_C_signal_minus_K_only": ("C_signal_next", "K_plus_C_signal_minus_K_only"),
        "V_next__K_plus_V_minus_K_only": ("V_next", "K_plus_V_minus_K_only"),
        "K_next__K_plus_C_signal_plus_V_minus_K_only": ("K_next", "K_plus_C_signal_plus_V_minus_K_only"),
    }
    out = {}
    for key, (target, contrast) in primary.items():
        row = _contrast_lookup(contrasts, target, contrast)
        out[key] = {
            "mean_delta_log_density": float(row["mean_delta_log_density"]),
            "ci_lower": float(row["ci_lower"]),
            "ci_upper": float(row["ci_upper"]),
            "positive_pair_rate": float(row["positive_pair_rate"]),
        }
    return out


def _contrast_lookup(contrasts: pd.DataFrame, target_id: str, contrast_id: str) -> pd.Series:
    row = contrasts.loc[
        (contrasts["target_id"] == target_id) & (contrasts["contrast_id"] == contrast_id)
    ]
    if len(row) != 1:
        raise RuntimeError(f"missing contrast {target_id}/{contrast_id}")
    return row.iloc[0]


def _render_report(
    summary: dict[str, Any],
    model_scores: pd.DataFrame,
    contrasts: pd.DataFrame,
    residual: dict[str, float],
    config: dict[str, Any],
) -> str:
    score_table = (
        model_scores.groupby(["target_id", "model_id"], as_index=False)["mean_log_density"]
        .mean()
        .sort_values(["target_id", "model_id"])
    )
    lines = [
        "# M5 Stage 1 K / C_signal / Vigilance Real-Data Analysis",
        "",
        "**Status:** exploratory Stage 1 mechanism analysis",
        "",
        "**Formal claims allowed:** false",
        "",
        "No Trident-G/APC/PACE validation claim, transfer claim, neural-criticality claim or cusp claim is made.",
        "",
        "## Question",
        "",
        "Can control and readiness be separated?",
        "",
        "## Design",
        "",
        "The analysis uses the frozen paired public session-level source only. Adjacent sessions are paired within participant, and current-session K/C_signal/V composites predict next-session K/C_signal/V targets under participant-isolated 5-fold validation.",
        "",
        "Forbidden profile/probability/candidate columns are present in the source but are not used.",
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
        "## Residual Separation",
        "",
        f"- C_signal residual variance fraction after K: {residual['C_signal_residual_variance_fraction_after_K']:.4f}",
        f"- V residual variance fraction after K: {residual['V_residual_variance_fraction_after_K']:.4f}",
        f"- Absolute residual C_signal/V correlation after K: {residual['C_signal_V_residual_abs_correlation_after_K']:.4f}",
        "",
        "## Primary Predictive Contrasts",
        "",
        "| Target | Contrast | Mean delta log density | 95% CI | Positive-pair rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in contrasts.itertuples(index=False):
        if row.contrast_id in {
            "K_plus_C_signal_minus_K_only",
            "K_plus_V_minus_K_only",
            "K_plus_C_signal_plus_V_minus_K_only",
        }:
            lines.append(
                f"| {row.target_id} | {row.contrast_id} | {row.mean_delta_log_density:.5f} | "
                f"[{row.ci_lower:.5f}, {row.ci_upper:.5f}] | {row.positive_pair_rate:.3f} |"
            )
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
            "- PACE/profile/candidate columns used: false",
            "- Dynamic-regime variables used: false",
            "- Transfer outcomes used: false",
            "- Criticality/cusp interpretation: false",
            "",
            "This Stage 1 result may inform whether the programme should proceed to the descriptive PACE-expression increment, but it does not validate Trident-G or any latent ontology.",
        ]
    )
    return "\n".join(lines) + "\n"


def _all_feature_columns(config: dict[str, Any]) -> list[str]:
    columns: list[str] = []
    for group in config["feature_groups"].values():
        columns.extend(group["columns"].keys())
    return sorted(set(columns))


def _forbidden_columns_used(config: dict[str, Any]) -> set[str]:
    configured = set()
    for group in config["feature_groups"].values():
        configured.update(group["columns"].keys())
    for model in config["candidate_models"].values():
        configured.update(model["predictors"])
    return configured.intersection(config["forbidden_columns"])


def _stable_int_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return (centre - half, centre + half)


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict) or not value:
        raise ConfigValidationError(f"{key} must be a non-empty mapping")
    return value


def _resolve_repo_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/public_stage1_k_c_v_v1.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    result = run_stage1_k_c_v_analysis(args.config, output_dir=args.output_dir)
    print(json.dumps(result.summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
