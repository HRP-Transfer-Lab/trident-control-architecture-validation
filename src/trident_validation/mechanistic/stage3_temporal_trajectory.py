"""Stage 3 temporal-trajectory increment analysis.

This module implements a conservative first temporal test from the M5/M6
ladder. It asks whether prior within-person change from online to lab1 adds
held-out prediction of lab2 behaviour beyond static lab1 K/C_signal/V.

It does not assign adaptive/locked/scattered regimes and does not authorise
criticality, cusp, transfer or Trident-G validation claims.
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
    _forbidden_columns_used,
    _gaussian_log_density,
    _participant_folds,
    _required_mapping,
    _resolve_repo_path,
    _write_json,
)
from trident_validation.provenance import get_git_commit, hash_file, hash_mapping


@dataclass(frozen=True)
class Stage3Result:
    """Participant-free Stage 3 temporal analysis result."""

    summary: dict[str, Any]
    model_scores: pd.DataFrame
    contrasts: pd.DataFrame
    delta_summary: pd.DataFrame
    report_markdown: str


def run_stage3_temporal_trajectory_analysis(
    config_path: str | Path = "config/public_stage3_temporal_trajectory_v1.yaml",
    *,
    repo_root: str | Path | None = None,
    output_dir: str | Path | None = None,
    write_outputs: bool = True,
) -> Stage3Result:
    """Run the pre-registered Stage 3 temporal-trajectory increment test."""

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
    sequences = _build_three_session_sequences(raw, config)
    fold_ids = _participant_folds(
        sequences["participant_id"].astype(str),
        int(config["validation"]["n_folds"]),
        int(config["validation"]["split_seed"]),
    )
    model_scores, scored_rows, delta_summary = _score_models(sequences, fold_ids, config)
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
        "pace_ontology_claim_allowed": False,
        "adaptive_locked_scattered_regime_claim_allowed": False,
        "regime_labels_used": False,
        "dynamic_regime_columns_used": False,
        "neural_criticality_claim_allowed": False,
        "cusp_claim_allowed": False,
        "input_rows": int(len(raw)),
        "complete_three_session_sequences": int(len(sequences)),
        "participants_with_three_session_sequences": int(sequences["participant_id"].nunique()),
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
            "continuous_autoregressive_delta_minus_static_K_C_V",
        ),
        "negative_control_contrast": _contrast_summary(
            contrasts,
            "Y_behavior_next",
            "delta_only_negative_control_minus_static_K_C_V",
        ),
    }
    report = _render_report(summary, model_scores, contrasts, delta_summary)

    if write_outputs:
        out_dir = Path(output_dir) if output_dir is not None else root / config["outputs"]["output_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_json(out_dir / "stage3_summary.json", summary)
        model_scores.to_csv(out_dir / "model_scores.csv", index=False)
        contrasts.to_csv(out_dir / "paired_contrasts.csv", index=False)
        delta_summary.to_csv(out_dir / "delta_summary.csv", index=False)
        (out_dir / "stage3_report.md").write_text(report, encoding="utf-8", newline="\n")
        report_path = root / str(config["outputs"]["report_md"])
        report_path.write_text(report, encoding="utf-8", newline="\n")

    return Stage3Result(
        summary=summary,
        model_scores=model_scores,
        contrasts=contrasts,
        delta_summary=delta_summary,
        report_markdown=report,
    )


def _validate_config(config: dict[str, Any]) -> None:
    analysis = _required_mapping(config, "analysis")
    if analysis.get("id") != "public_stage3_temporal_trajectory_v1":
        raise ConfigValidationError("analysis.id must be public_stage3_temporal_trajectory_v1")
    if analysis.get("status") != "pre_outcome_registered_analysis":
        raise ConfigValidationError("analysis.status must be pre_outcome_registered_analysis")
    for field in (
        "formal_claims_allowed",
        "trident_validation_claim_allowed",
        "transfer_outcomes_allowed",
        "pace_ontology_claim_allowed",
        "adaptive_locked_scattered_regime_claim_allowed",
        "neural_criticality_claim_allowed",
        "cusp_claim_allowed",
    ):
        if analysis.get(field) is not False:
            raise ConfigValidationError(f"analysis.{field} must be false")
    trajectory = _required_mapping(config, "trajectory")
    if trajectory.get("regime_labels_allowed") is not False:
        raise ConfigValidationError("regime labels are not allowed in this Stage 3 increment test")
    if trajectory.get("required_session_sequence") != ["online", "lab1", "lab2"]:
        raise ConfigValidationError("Stage 3 requires online/lab1/lab2 sequence")

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
        "static_K_C_V",
        "continuous_autoregressive_delta",
        "delta_only_negative_control",
    }
    if set(_required_mapping(config, "candidate_models")) != expected_models:
        raise ConfigValidationError("candidate_models must match the Stage 3 temporal increment protocol")
    if config["validation"].get("participant_isolated") is not True:
        raise ConfigValidationError("participant-isolated validation is required")


def _build_three_session_sequences(raw: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    id_cols = config["identity_columns"]
    participant_col = id_cols["participant_id"]
    session_col = id_cols["session_id"]
    order_col = id_cols["session_order"]
    sequence = list(config["trajectory"]["required_session_sequence"])
    required_columns = {participant_col, session_col, order_col}
    for group in config["feature_groups"].values():
        required_columns.update(group["columns"])
    missing = sorted(required_columns.difference(raw.columns))
    if missing:
        raise ConfigValidationError("input is missing required columns: " + ", ".join(missing))

    feature_columns = _all_feature_columns(config)
    df = raw.loc[:, sorted(required_columns)].dropna(subset=feature_columns).copy()
    rows: list[dict[str, Any]] = []
    for participant_id, group in df.groupby(participant_col, sort=True):
        by_session = {
            str(record[order_col]): record
            for record in group.sort_values([order_col, session_col]).to_dict("records")
        }
        if not all(session in by_session for session in sequence):
            continue
        previous = by_session[sequence[0]]
        current = by_session[sequence[1]]
        nxt = by_session[sequence[2]]
        row: dict[str, Any] = {"participant_id": participant_id}
        for column in feature_columns:
            row[f"previous__{column}"] = float(previous[column])
            row[f"current__{column}"] = float(current[column])
            row[f"next__{column}"] = float(nxt[column])
        rows.append(row)
    sequences_df = pd.DataFrame.from_records(rows)
    if sequences_df.empty:
        raise ConfigValidationError("no complete three-session sequences are available")
    return sequences_df


def _score_models(
    sequences: pd.DataFrame,
    fold_ids: np.ndarray,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_rows: list[dict[str, Any]] = []
    scored_rows: list[pd.DataFrame] = []
    delta_rows: list[dict[str, Any]] = []
    n_folds = int(config["validation"]["n_folds"])
    alpha = float(config["validation"]["ridge_alpha"])
    for fold in range(n_folds):
        train_sequences = sequences.loc[fold_ids != fold].copy()
        test_sequences = sequences.loc[fold_ids == fold].copy()
        train_design, test_design = _fold_trajectory_composites(train_sequences, test_sequences, config)
        for group in ("K", "C_signal", "V"):
            delta_rows.append(
                {
                    "fold": fold,
                    "component": f"delta_{group}",
                    "train_mean": float(train_design[f"delta_{group}"].mean()),
                    "train_sd": float(train_design[f"delta_{group}"].std(ddof=0)),
                    "test_mean": float(test_design[f"delta_{group}"].mean()),
                    "test_sd": float(test_design[f"delta_{group}"].std(ddof=0)),
                }
            )

        for model_id, model in config["candidate_models"].items():
            train_x, test_x = _design_matrices(train_design, test_design, list(model["predictors"]))
            for target_id, target in config["targets"].items():
                y_train = _target_vector(train_design, target["groups"])
                y_test = _target_vector(test_design, target["groups"])
                fit = _fit_ridge_gaussian(train_x, y_train, alpha)
                prediction = test_x @ fit["coef"]
                log_density = _gaussian_log_density(y_test, prediction, fit["sigma2"])
                model_rows.append(
                    {
                        "fold": fold,
                        "model_id": model_id,
                        "target_id": target_id,
                        "n_train_sequences": int(len(train_design)),
                        "n_test_sequences": int(len(test_design)),
                        "mean_log_density": float(np.mean(log_density)),
                        "mse": float(np.mean((y_test - prediction) ** 2)),
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
                            "squared_error": (y_test - prediction) ** 2,
                        }
                    )
                )
    return (
        pd.DataFrame(model_rows),
        pd.concat(scored_rows, ignore_index=True),
        pd.DataFrame(delta_rows),
    )


def _fold_trajectory_composites(
    train_sequences: pd.DataFrame,
    test_sequences: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = _all_feature_columns(config)
    train_session_values = []
    for prefix in ("previous__", "current__", "next__"):
        block = train_sequences[[prefix + col for col in feature_columns]].copy()
        block.columns = feature_columns
        train_session_values.append(block)
    train_sessions = pd.concat(train_session_values, ignore_index=True)
    means = train_sessions.mean(axis=0)
    sds = train_sessions.std(axis=0, ddof=0).replace(0, 1.0)
    return (
        _apply_trajectory_composites(train_sequences, config, means, sds),
        _apply_trajectory_composites(test_sequences, config, means, sds),
    )


def _apply_trajectory_composites(
    sequences: pd.DataFrame,
    config: dict[str, Any],
    means: pd.Series,
    sds: pd.Series,
) -> pd.DataFrame:
    out = pd.DataFrame({"row_index": np.arange(len(sequences), dtype=int)})
    for group_id, group in config["feature_groups"].items():
        for side in ("previous", "current", "next"):
            oriented = []
            for column, sign in group["columns"].items():
                z = (sequences[f"{side}__{column}"].to_numpy(dtype=float) - means[column]) / sds[column]
                oriented.append(float(sign) * z)
            out[f"{group_id}_{side}"] = np.mean(np.vstack(oriented), axis=0)
        out[f"delta_{group_id}"] = out[f"{group_id}_current"] - out[f"{group_id}_previous"]
    return out


def _design_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    predictors: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    train_x = np.column_stack([np.ones(len(train)), *[train[p].to_numpy() for p in predictors]])
    test_x = np.column_stack([np.ones(len(test)), *[test[p].to_numpy() for p in predictors]])
    if train_x.shape[1] > 1:
        means = train_x[:, 1:].mean(axis=0)
        sds = np.where(train_x[:, 1:].std(axis=0) == 0, 1.0, train_x[:, 1:].std(axis=0))
        train_x[:, 1:] = (train_x[:, 1:] - means) / sds
        test_x[:, 1:] = (test_x[:, 1:] - means) / sds
    return train_x, test_x


def _target_vector(composites: pd.DataFrame, groups: list[str]) -> np.ndarray:
    columns = [composites[f"{group}_next"].to_numpy(dtype=float) for group in groups]
    return np.mean(np.vstack(columns), axis=0)


def _summarise_contrasts(scored_rows: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    contrast_specs = {
        "continuous_autoregressive_delta_minus_static_K_C_V": (
            "continuous_autoregressive_delta",
            "static_K_C_V",
        ),
        "delta_only_negative_control_minus_static_K_C_V": (
            "delta_only_negative_control",
            "static_K_C_V",
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
                    "n_sequences": int(len(delta)),
                    "mean_delta_log_density": float(np.mean(delta)),
                    "median_delta_log_density": float(np.median(delta)),
                    "ci_lower": float(np.quantile(boot, alpha)),
                    "ci_upper": float(np.quantile(boot, 1.0 - alpha)),
                    "positive_sequence_rate": float(np.mean(delta > 0.0)),
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
        "continuous_autoregressive_delta_minus_static_K_C_V",
    )
    negative = _contrast_lookup(
        contrasts,
        "Y_behavior_next",
        "delta_only_negative_control_minus_static_K_C_V",
    )
    if forbidden_used:
        return {"decision": "inconclusive", "reason": "forbidden columns were used"}
    support = (
        primary["mean_delta_log_density"] > float(rule["primary_increment_mean_minimum"])
        and primary["ci_lower"] >= float(rule["primary_increment_ci_lower_tolerance"])
        and (
            not bool(rule["negative_control_must_not_exceed_static_baseline"])
            or negative["mean_delta_log_density"] <= 0.0
        )
    )
    if support:
        return {
            "decision": "temporal_increment_supported",
            "reason": "prior within-person trajectory improved lab2 behaviour beyond static lab1 K/C/V under the pre-registered tolerance",
        }
    if primary["ci_upper"] <= float(rule["disconfirm_primary_increment_ci_upper_maximum"]):
        return {
            "decision": "temporal_increment_not_supported",
            "reason": "prior within-person trajectory failed the pre-registered held-out predictive criterion",
        }
    return {
        "decision": "inconclusive",
        "reason": "temporal increment did not satisfy either the support or disconfirmation rule",
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
        "positive_sequence_rate": float(row["positive_sequence_rate"]),
    }


def _render_report(
    summary: dict[str, Any],
    model_scores: pd.DataFrame,
    contrasts: pd.DataFrame,
    delta_summary: pd.DataFrame,
) -> str:
    score_table = (
        model_scores.groupby(["target_id", "model_id"], as_index=False)["mean_log_density"]
        .mean()
        .sort_values(["target_id", "model_id"])
    )
    delta_table = (
        delta_summary.groupby("component", as_index=False)[["train_mean", "train_sd", "test_mean", "test_sd"]]
        .mean()
        .sort_values("component")
    )
    lines = [
        "# M5 Stage 3 Temporal-Trajectory Real-Data Analysis",
        "",
        "**Status:** exploratory Stage 3 temporal increment analysis",
        "",
        "**Formal claims allowed:** false",
        "",
        "No Trident-G/APC/PACE validation claim, transfer claim, adaptive/locked/scattered regime claim, neural-criticality claim or cusp claim is made.",
        "",
        "## Question",
        "",
        "Does prior within-person trajectory add predictive structure beyond static K/C/V?",
        "",
        "## Design",
        "",
        "The analysis uses complete online -> lab1 -> lab2 sequences from the frozen paired public session-level source. Lab1 static K/C_signal/V predicts lab2 behaviour. The temporal candidate adds online-to-lab1 within-person deltas.",
        "",
        "No regime labels are estimated or interpreted. This is a trajectory-increment test only.",
        "",
        "The upstream profile/probability/candidate columns are present in the source but are not used.",
        "",
        "## Sample",
        "",
        f"- Input participant-session rows: {summary['input_rows']}",
        f"- Complete three-session sequences: {summary['complete_three_session_sequences']}",
        f"- Participants with complete sequences: {summary['participants_with_three_session_sequences']}",
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
        "| Target | Contrast | Mean delta log density | 95% CI | Positive-sequence rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in contrasts.itertuples(index=False):
        lines.append(
            f"| {row.target_id} | {row.contrast_id} | {row.mean_delta_log_density:.5f} | "
            f"[{row.ci_lower:.5f}, {row.ci_upper:.5f}] | {row.positive_sequence_rate:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Delta Diagnostics",
            "",
            "| Component | Train mean | Train SD | Held-out mean | Held-out SD |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in delta_table.itertuples(index=False):
        lines.append(f"| {row.component} | {row.train_mean:.4f} | {row.train_sd:.4f} | {row.test_mean:.4f} | {row.test_sd:.4f} |")
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
            "- Regime labels used: false",
            "- Adaptive/locked/scattered regime claim: false",
            "- Upstream PACE/profile/probability/candidate columns used: false",
            "- Transfer outcomes used: false",
            "- Criticality/cusp interpretation: false",
            "",
            "This Stage 3 result only tests whether a simple prior within-person trajectory increment is useful beyond static K/C/V in the paired public source.",
        ]
    )
    return "\n".join(lines) + "\n"


def _all_feature_columns(config: dict[str, Any]) -> list[str]:
    columns: list[str] = []
    for group in config["feature_groups"].values():
        columns.extend(group["columns"].keys())
    return sorted(set(columns))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/public_stage3_temporal_trajectory_v1.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    result = run_stage3_temporal_trajectory_analysis(args.config, output_dir=args.output_dir)
    print(json.dumps(result.summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
