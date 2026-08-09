"""M2.8 K-only versus static APC generator smoke.

This module is engineering scaffolding for the first mechanistic
identifiability gate. It generates known-truth synthetic data for MECH0 and
MECH1, strips truth columns before placeholder scoring, and reports audits. It
does not fit public data and does not authorise Trident-G, APC, PACE, transfer,
neural-criticality or cusp claims.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from trident_validation.config import ConfigValidationError, load_yaml_config
from trident_validation.mechanistic.identifiability_plan import (
    dataframe_hash,
    load_mechanistic_identifiability_plan,
)
from trident_validation.provenance import hash_file, hash_mapping
from trident_validation.splits import assert_no_participant_overlap, participant_train_test_split
from trident_validation.synthetic.recovery import assert_no_ground_truth_columns, strip_ground_truth_columns


K_APC_GATE_ID = "K_only_vs_APC"
K_APC_TRUTH_FAMILIES = ("MECH0", "MECH1")
MECHANISTIC_OBSERVED_FEATURES = (
    "signal_recovery_score",
    "evidence_update_score",
    "commit_control_score",
    "predictive_calibration_score",
    "accuracy",
    "mean_response_speed",
    "rt_cv",
    "throughput_proxy",
)
TRUTH_COLUMNS = (
    "synthetic_truth_family_id",
    "synthetic_truth_family_label",
    "synthetic_gate_id",
    "synthetic_K",
    "synthetic_C_signal",
    "synthetic_A_evidence",
    "synthetic_T_commit",
    "synthetic_PC_calibration",
)


@dataclass(frozen=True)
class KAPCSmokeOutputs:
    """M2.8 K/APC smoke outputs."""

    generation_audit: pd.DataFrame
    model_scores: pd.DataFrame
    split_audit: pd.DataFrame
    report: str
    manifest: dict[str, Any]


def generate_k_apc_unit(schedule_row: pd.Series | dict[str, Any]) -> pd.DataFrame:
    """Generate one known-truth MECH0 or MECH1 participant/block table."""

    row = dict(schedule_row)
    family_id = str(row["truth_family_id"])
    if family_id not in K_APC_TRUTH_FAMILIES:
        raise ValueError(f"K/APC smoke supports only {K_APC_TRUTH_FAMILIES}, got {family_id}")
    rng_structural = np.random.default_rng(int(row["structural_seed"]))
    rng_nuisance = np.random.default_rng(int(row["nuisance_seed"]))
    rng_observation = np.random.default_rng(int(row["observation_seed"]))
    participants = int(row["participants"])
    sessions = int(row["sessions_per_participant"])
    blocks = int(row["blocks_per_session"])

    k_person = rng_structural.normal(0.0, 1.0, size=participants)
    if family_id == "MECH1":
        apc_person = rng_structural.normal(0.0, 1.0, size=(participants, 4))
    else:
        apc_person = np.full((participants, 4), np.nan)

    rows: list[dict[str, Any]] = []
    for participant_index in range(participants):
        participant_id = f"{family_id.lower()}_p{participant_index:03d}"
        for session_index in range(sessions):
            session_shift = rng_nuisance.normal(0.0, 0.25)
            for block_index in range(blocks):
                block_shift = rng_nuisance.normal(0.0, 0.18)
                time_on_task = block_index / max(blocks - 1, 1)
                fatigue = -0.08 * time_on_task + rng_nuisance.normal(0.0, 0.05)
                k_value = float(k_person[participant_index])
                if family_id == "MECH1":
                    c_signal, a_evidence, t_commit, pc_calibration = (
                        apc_person[participant_index]
                        + rng_structural.normal(0.0, 0.35, size=4)
                    )
                else:
                    c_signal = a_evidence = t_commit = pc_calibration = np.nan

                observed = _observed_features(
                    family_id=family_id,
                    k_value=k_value,
                    c_signal=float(c_signal) if np.isfinite(c_signal) else np.nan,
                    a_evidence=float(a_evidence) if np.isfinite(a_evidence) else np.nan,
                    t_commit=float(t_commit) if np.isfinite(t_commit) else np.nan,
                    pc_calibration=float(pc_calibration) if np.isfinite(pc_calibration) else np.nan,
                    session_shift=float(session_shift),
                    block_shift=float(block_shift),
                    fatigue=float(fatigue),
                    rng=rng_observation,
                )
                rows.append(
                    {
                        "source_dataset": "m2_8_mechanistic_synthetic",
                        "source_version": "k_apc_smoke_v1",
                        "participant_id": participant_id,
                        "session_id": f"s{session_index + 1:02d}",
                        "task_id": "mechanistic_k_apc",
                        "block_id": f"b{block_index + 1:02d}",
                        "window_id": f"w{block_index + 1:02d}",
                        "window_start_trial": block_index * 32 + 1,
                        "window_end_trial": (block_index + 1) * 32,
                        "n_trials_total": 32,
                        "n_trials_valid": 32,
                        "time_on_task": float(time_on_task),
                        "trial_count": 32,
                        "condition_mix": "synthetic_mechanistic",
                        "synthetic_truth_family_id": family_id,
                        "synthetic_truth_family_label": str(row["truth_family_label"]),
                        "synthetic_gate_id": str(row["gate_id"]),
                        "synthetic_K": k_value,
                        "synthetic_C_signal": c_signal,
                        "synthetic_A_evidence": a_evidence,
                        "synthetic_T_commit": t_commit,
                        "synthetic_PC_calibration": pc_calibration,
                        **observed,
                    }
                )
    return pd.DataFrame(rows)


def run_k_apc_smoke(
    config_path: str | Path = "config/mechanistic_identifiability_v1.yaml",
    *,
    output_dir: str | Path = "reports/generated/m2_8_k_apc_smoke",
) -> KAPCSmokeOutputs:
    """Run the bounded first-gate M2.8 generator/scoring smoke."""

    config_path = Path(config_path)
    output_dir = Path(output_dir)
    config = load_yaml_config(config_path)
    plan = load_mechanistic_identifiability_plan(config_path)
    gate_schedule = plan.schedule[
        (plan.schedule["gate_id"] == K_APC_GATE_ID)
        & plan.schedule["truth_family_id"].isin(K_APC_TRUTH_FAMILIES)
    ].copy()
    if gate_schedule.empty:
        raise ConfigValidationError("K_only_vs_APC schedule contains no MECH0/MECH1 units")

    audit_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    total_generated_rows = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    for schedule_row in gate_schedule.to_dict(orient="records"):
        unit = generate_k_apc_unit(schedule_row)
        total_generated_rows += int(unit.shape[0])
        unit_path = output_dir / f"{schedule_row['unit_id']}_windows.csv"
        _atomic_write_csv(unit_path, unit)
        audit_rows.extend(_variable_audit(unit, schedule_row, unit_path))
        scoring_frame = strip_ground_truth_columns(unit)
        assert_no_ground_truth_columns(scoring_frame)
        split = participant_train_test_split(
            scoring_frame,
            test_size=float(schedule_row["participant_holdout_fraction"]),
            seed=int(schedule_row["split_seed"]),
            participant_columns=("source_dataset", "participant_id"),
        )
        assert_no_participant_overlap(scoring_frame, split)
        model_scores = score_capacity_vs_apc_placeholders(
            scoring_frame,
            split,
            feature_columns=MECHANISTIC_OBSERVED_FEATURES,
        )
        for model_row in model_scores.to_dict(orient="records"):
            model_rows.append({**_unit_fields(schedule_row), **model_row})
        split_rows.append(
            {
                **_unit_fields(schedule_row),
                "participant_isolated": True,
                "n_participant_overlap": 0,
                "n_train_rows": len(split.train_indices),
                "n_test_rows": len(split.test_indices),
                "n_train_participants": int(
                    scoring_frame.loc[list(split.train_indices), "participant_id"].nunique()
                ),
                "n_test_participants": int(
                    scoring_frame.loc[list(split.test_indices), "participant_id"].nunique()
                ),
                "truth_columns_in_scoring_frame": 0,
            }
        )

    generation_audit = pd.DataFrame(audit_rows)
    model_scores = pd.DataFrame(model_rows)
    split_audit = pd.DataFrame(split_rows)
    report = k_apc_smoke_report(generation_audit, model_scores, split_audit)
    manifest = {
        "study_id": "m2_8_k_apc_smoke",
        "status": "engineering_generator_smoke_only",
        "formal_claims_allowed": False,
        "trident_validation_claim_allowed": False,
        "real_transfer_outcomes_allowed": False,
        "gate_id": K_APC_GATE_ID,
        "truth_family_ids": list(K_APC_TRUTH_FAMILIES),
        "config_path": str(config_path.as_posix()),
        "config_hash": hash_mapping(config),
        "schedule_hash": plan.schedule_hash,
        "n_units": int(gate_schedule.shape[0]),
        "n_rows": int(total_generated_rows),
        "placeholder_models": [
            "SCORE0_capacity_only_rank1",
            "SCORE1_static_continuous_apc_rank5",
        ],
        "model_winner_interpretation_allowed": False,
    }
    _atomic_write_csv(output_dir / "generation_audit.csv", generation_audit)
    _atomic_write_csv(output_dir / "model_scores.csv", model_scores)
    _atomic_write_csv(output_dir / "split_audit.csv", split_audit)
    (output_dir / "M2_8_K_APC_SMOKE.md").write_text(report, encoding="utf-8")
    manifest["output_hashes"] = {
        "generation_audit": hash_file(output_dir / "generation_audit.csv"),
        "model_scores": hash_file(output_dir / "model_scores.csv"),
        "split_audit": hash_file(output_dir / "split_audit.csv"),
        "report": hash_file(output_dir / "M2_8_K_APC_SMOKE.md"),
    }
    _atomic_write_json(output_dir / "manifest.json", manifest)
    return KAPCSmokeOutputs(
        generation_audit=generation_audit,
        model_scores=model_scores,
        split_audit=split_audit,
        report=report,
        manifest=manifest,
    )


def score_capacity_vs_apc_placeholders(
    frame_without_truth: pd.DataFrame,
    split,
    *,
    feature_columns: Iterable[str] = MECHANISTIC_OBSERVED_FEATURES,
) -> pd.DataFrame:
    """Score placeholder rank models on a truth-stripped participant split."""

    assert_no_ground_truth_columns(frame_without_truth)
    assert_no_participant_overlap(frame_without_truth, split)
    features = tuple(feature_columns)
    train = frame_without_truth.loc[list(split.train_indices), list(features)]
    test = frame_without_truth.loc[list(split.test_indices), list(features)]
    rows = []
    for model_id, rank in (
        ("SCORE0_capacity_only_rank1", 1),
        ("SCORE1_static_continuous_apc_rank5", 5),
    ):
        result = _rank_gaussian_score(train, test, rank=rank)
        rows.append(
            {
                "model_id": model_id,
                "model_status": "placeholder_engineering_scaffold",
                "rank": rank,
                "heldout_log_density_mean_per_row": result["heldout_log_density_mean_per_row"],
                "train_reconstruction_mse": result["train_reconstruction_mse"],
                "test_reconstruction_mse": result["test_reconstruction_mse"],
                "n_train_rows": int(train.shape[0]),
                "n_test_rows": int(test.shape[0]),
                "n_features": len(features),
                "truth_columns_received": 0,
            }
        )
    return pd.DataFrame(rows)


def k_apc_smoke_report(
    generation_audit: pd.DataFrame,
    model_scores: pd.DataFrame,
    split_audit: pd.DataFrame,
) -> str:
    """Create a concise non-claim smoke report."""

    rows = [
        "# M2.8 K/APC Generator Smoke",
        "",
        "**Status:** engineering generator smoke only",
        "",
        "No confirmatory Trident-G, APC, PACE, neural-criticality, cusp or transfer claim is made.",
        "",
        "## Scope",
        "",
        "This smoke covers only the first M2.8 identifiability gate:",
        "",
        "```text",
        "K_only_vs_APC",
        "MECH0: K -> Y_behavior",
        "MECH1: K + C_signal + A_evidence + T_commit + PC_calibration -> Y_behavior",
        "```",
        "",
        "The scorer is a placeholder engineering scaffold and is not a frozen M2.8 model tournament.",
        "",
        "## Generation Audit",
        "",
        generation_audit.to_csv(index=False, lineterminator="\n"),
        "",
        "## Split Audit",
        "",
        split_audit.to_csv(index=False, lineterminator="\n"),
        "",
        "## Placeholder Scores",
        "",
        model_scores.to_csv(index=False, lineterminator="\n"),
    ]
    return "\n".join(rows)


def _observed_features(
    *,
    family_id: str,
    k_value: float,
    c_signal: float,
    a_evidence: float,
    t_commit: float,
    pc_calibration: float,
    session_shift: float,
    block_shift: float,
    fatigue: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    noise = rng.normal(0.0, 0.28, size=8)
    nuisance = session_shift + block_shift + fatigue
    if family_id == "MECH0":
        latent = np.repeat(k_value, 4)
    else:
        latent = np.array([c_signal, a_evidence, t_commit, pc_calibration], dtype=float)
    signal = 0.35 * k_value + 0.95 * latent[0] + 0.40 * nuisance + noise[0]
    evidence = 0.35 * k_value + 0.95 * latent[1] + 0.35 * nuisance + noise[1]
    commit = 0.35 * k_value + 0.95 * latent[2] + 0.30 * nuisance + noise[2]
    calibration = 0.35 * k_value + 0.95 * latent[3] + 0.25 * nuisance + noise[3]
    composite = (signal + evidence + commit + calibration) / 4.0
    accuracy = float(np.clip(0.74 + 0.08 * composite + noise[4] * 0.02, 0.35, 0.995))
    response_speed = float(np.clip(1.45 + 0.15 * composite + 0.06 * commit + noise[5] * 0.04, 0.4, 3.2))
    rt_cv = float(np.clip(0.24 - 0.025 * composite - 0.015 * calibration + noise[6] * 0.03, 0.03, 0.9))
    throughput = float(accuracy * response_speed * 100.0 + noise[7] * 0.5)
    return {
        "signal_recovery_score": float(signal),
        "evidence_update_score": float(evidence),
        "commit_control_score": float(commit),
        "predictive_calibration_score": float(calibration),
        "accuracy": accuracy,
        "mean_response_speed": response_speed,
        "median_rt_ms": float(1000.0 / response_speed),
        "rt_cv": rt_cv,
        "throughput_proxy": throughput,
    }


def _rank_gaussian_score(train: pd.DataFrame, test: pd.DataFrame, *, rank: int) -> dict[str, float]:
    train_values = train.to_numpy(dtype=float)
    test_values = test.to_numpy(dtype=float)
    means = train_values.mean(axis=0)
    scales = train_values.std(axis=0, ddof=1)
    scales = np.where(scales <= 1e-8, 1.0, scales)
    train_z = (train_values - means) / scales
    test_z = (test_values - means) / scales
    _, _, vh = np.linalg.svd(train_z, full_matrices=False)
    rank = min(rank, vh.shape[0] - 1)
    components = vh[:rank, :]
    train_recon = train_z @ components.T @ components
    test_recon = test_z @ components.T @ components
    train_resid = train_z - train_recon
    test_resid = test_z - test_recon
    residual_var = np.var(train_resid, axis=0, ddof=1)
    residual_var = np.maximum(residual_var, 0.05)
    log_density = -0.5 * (
        np.log(2.0 * math.pi * residual_var) + (test_resid**2 / residual_var)
    ).sum(axis=1)
    return {
        "heldout_log_density_mean_per_row": float(log_density.mean()),
        "train_reconstruction_mse": float(np.mean(train_resid**2)),
        "test_reconstruction_mse": float(np.mean(test_resid**2)),
    }


def _variable_audit(
    unit: pd.DataFrame,
    schedule_row: dict[str, Any],
    unit_path: Path,
) -> list[dict[str, Any]]:
    rows = []
    for variable_id, column in (
        ("K", "synthetic_K"),
        ("C_signal", "synthetic_C_signal"),
        ("A_evidence", "synthetic_A_evidence"),
        ("T_commit", "synthetic_T_commit"),
        ("PC_calibration", "synthetic_PC_calibration"),
    ):
        values = unit[column].dropna()
        rows.append(
            {
                **_unit_fields(schedule_row),
                "variable_id": variable_id,
                "n_rows": int(unit.shape[0]),
                "n_observed_truth_values": int(values.shape[0]),
                "truth_mean": float(values.mean()) if not values.empty else float("nan"),
                "truth_variance": float(values.var(ddof=1)) if values.shape[0] > 1 else float("nan"),
                "unit_hash": hash_file(unit_path),
            }
        )
    rows.append(
        {
            **_unit_fields(schedule_row),
            "variable_id": "observed_feature_rank",
            "n_rows": int(unit.shape[0]),
            "n_observed_truth_values": int(unit.shape[0]),
            "truth_mean": float("nan"),
            "truth_variance": float(np.linalg.matrix_rank(unit.loc[:, MECHANISTIC_OBSERVED_FEATURES])),
            "unit_hash": hash_file(unit_path),
        }
    )
    return rows


def _unit_fields(schedule_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "unit_id": str(schedule_row["unit_id"]),
        "gate_id": str(schedule_row["gate_id"]),
        "truth_family_id": str(schedule_row["truth_family_id"]),
        "replicate_index": int(schedule_row["replicate_index"]),
    }


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the first M2.8 engineering smoke."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/mechanistic_identifiability_v1.yaml")
    parser.add_argument("--output-dir", default="reports/generated/m2_8_k_apc_smoke")
    args = parser.parse_args(argv)
    outputs = run_k_apc_smoke(args.config, output_dir=args.output_dir)
    print(
        json.dumps(
            {
                "study_id": outputs.manifest["study_id"],
                "status": outputs.manifest["status"],
                "gate_id": outputs.manifest["gate_id"],
                "n_units": outputs.manifest["n_units"],
                "n_rows": outputs.manifest["n_rows"],
                "model_winner_interpretation_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
