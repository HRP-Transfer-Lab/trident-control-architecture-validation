from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from trident_validation.config import ConfigValidationError, load_yaml_config
import trident_validation.mechanistic.stage1b_sat_policy as stage1b
from trident_validation.mechanistic.stage1b_sat_policy import (
    _run_cross_task_transport,
    _validate_config,
    _without_task_general,
    policy_support_preflight,
    run_stage1b_sat_policy_analysis,
)
from trident_validation.provenance import hash_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/public_stage1b_sat_policy_v1.yaml"


def test_stage1b_config_is_policy_candidate_and_claim_bounded():
    config = load_yaml_config(CONFIG_PATH)
    _validate_config(config)

    assert config["analysis"]["hypothesis_timing"] == "formulated_after_stage1_to_stage3_review"
    assert config["analysis"]["t_commit_claim_allowed"] is False
    assert config["analysis"]["predictive_calibration_claim_allowed"] is False
    assert config["analysis"]["optimality_claim_allowed"] is False
    assert config["policy_construction"]["fold_fitted_scaling_only"] is True
    assert config["policy_construction"]["task_weights_fitted_from_data"] is False
    configured_columns = {
        column
        for group in config["feature_groups"].values()
        for column in group["columns"]
    }
    assert configured_columns.isdisjoint(config["forbidden_columns"])


def test_stage1b_rejects_forbidden_profile_column_as_feature():
    config = load_yaml_config(CONFIG_PATH)
    config["feature_groups"]["K"]["columns"]["control_profile"] = 1

    with pytest.raises(ConfigValidationError, match="forbidden columns"):
        _validate_config(config)


def test_stage1b_registered_local_source_path_resolves_when_available():
    config = load_yaml_config(CONFIG_PATH)
    source_path = (ROOT / config["inputs"]["paired_session_features_path"]).resolve()

    if source_path.exists():
        assert hash_file(source_path).lower() == (
            "sha256:" + config["inputs"]["paired_session_features_sha256"].lower()
        )


def test_policy_support_preflight_uses_availability_only():
    config = load_yaml_config(CONFIG_PATH)
    df = _synthetic_policy_frame(n_participants=60, sessions=("online", "lab1"))

    support = policy_support_preflight(df, config)

    assert support.attrs["support_gate_passed"] is True
    common = support.loc[support["task"] == "Stroop_Flanker_common"].iloc[0]
    assert common["complete_speed_accuracy_rows"] == 120
    assert common["complete_speed_accuracy_participants"] == 60
    assert "correlation" not in " ".join(support.columns)


def test_cross_task_k_leakage_guard_removes_target_task_general_features():
    config = load_yaml_config(CONFIG_PATH)
    k_columns = config["feature_groups"]["K"]["columns"]

    flanker_guarded = _without_task_general(k_columns, "Flanker")
    stroop_guarded = _without_task_general(k_columns, "Stroop")

    assert "flanker_accuracy" not in flanker_guarded
    assert "flanker_mean_rt_ms" not in flanker_guarded
    assert "flanker_throughput" not in flanker_guarded
    assert "stroop_accuracy" in flanker_guarded
    assert "stroop_accuracy" not in stroop_guarded
    assert "stroop_mean_rt_ms" not in stroop_guarded
    assert "stroop_throughput" not in stroop_guarded
    assert "flanker_accuracy" in stroop_guarded


def test_cross_task_transport_is_participant_isolated_and_target_bounded():
    config = load_yaml_config(CONFIG_PATH)
    config["validation"]["bootstrap_iterations"] = 20
    df = _synthetic_policy_frame(n_participants=60, sessions=("online", "lab1"))

    scored = _run_cross_task_transport(df, config)

    assert set(scored["target_id"]) == {"T_flanker", "T_stroop"}
    assert set(scored["model_id"]) == {
        "base",
        "policy",
        "speed_negative_control",
        "accuracy_negative_control",
    }
    for fold in sorted(scored["fold"].unique()):
        test_participants = set(scored.loc[scored["fold"] == fold, "participant_id"])
        train_participants = set(scored.loc[scored["fold"] != fold, "participant_id"])
        assert test_participants.union(train_participants)
        # Each participant is scored in only one held-out fold.
        assert test_participants.isdisjoint(
            set(
                scored.loc[
                    (scored["fold"] != fold)
                    & scored["participant_id"].isin(test_participants),
                    "participant_id",
                ]
            )
        )


def test_stage1b_runner_is_deterministic_and_participant_free(monkeypatch):
    config = load_yaml_config(CONFIG_PATH)
    df = _synthetic_policy_frame(n_participants=70, sessions=("online", "lab1", "lab2"))
    work_dir = ROOT / "reports" / "generated" / "test_stage1b_sat_policy"
    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = work_dir / "paired.parquet"
    input_path.write_text("synthetic parquet placeholder", encoding="utf-8")
    monkeypatch.setattr(stage1b.pd, "read_parquet", lambda path: df.copy())
    config["inputs"]["paired_session_features_path"] = str(input_path)
    config["inputs"]["paired_session_features_sha256"] = hash_file(input_path).replace("sha256:", "").upper()
    config["validation"]["bootstrap_iterations"] = 20
    config["outputs"]["output_dir"] = str(work_dir / "out")
    config["outputs"]["report_md"] = str(work_dir / "report.md")
    config_path = work_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    first = run_stage1b_sat_policy_analysis(
        config_path,
        repo_root=ROOT,
        output_dir=work_dir / "out",
        write_outputs=True,
    )
    second = run_stage1b_sat_policy_analysis(
        config_path,
        repo_root=ROOT,
        output_dir=work_dir / "out2",
        write_outputs=False,
    )

    assert first.summary["decision"] == second.summary["decision"]
    assert first.summary["primary_cross_task_contrast"] == second.summary["primary_cross_task_contrast"]
    assert first.summary["primary_next_session_contrast"] == second.summary["primary_next_session_contrast"]
    assert first.summary["formal_claims_allowed"] is False
    assert first.summary["t_commit_claim_allowed"] is False
    assert first.summary["forbidden_columns_used"] == []
    assert "participant_id" not in first.cross_task_contrasts.columns
    assert "participant_id" not in first.persistence_contrasts.columns
    assert (work_dir / "out" / "stage1b_summary.json").exists()
    assert (work_dir / "report.md").exists()


def _synthetic_policy_frame(
    n_participants: int,
    sessions: tuple[str, ...],
) -> pd.DataFrame:
    rng = np.random.default_rng(2468)
    rows = []
    for participant in range(n_participants):
        k = rng.normal()
        c = rng.normal()
        v = rng.normal()
        policy = rng.normal()
        for index, session_type in enumerate(sessions):
            policy_session = 0.75 * policy + rng.normal(scale=0.2) + 0.1 * index
            k_session = k + rng.normal(scale=0.2)
            c_session = c + rng.normal(scale=0.2)
            v_session = v + rng.normal(scale=0.2)
            rows.append(
                {
                    "participant_id": f"P{participant:03d}",
                    "session_id": f"P{participant:03d}:{session_type}",
                    "session_type": session_type,
                    "control_profile": "forbidden",
                    "control_profile_probability": 0.99,
                    "low_engagement_candidate": False,
                    "high_engagement_candidate": False,
                    "stroop_accuracy": 0.80 + 0.06 * k_session + 0.08 * policy_session + rng.normal(scale=0.03),
                    "flanker_accuracy": 0.82 + 0.06 * k_session + 0.08 * policy_session + rng.normal(scale=0.03),
                    "stroop_mean_rt_ms": 620 - 28 * k_session + 32 * policy_session + rng.normal(scale=12),
                    "flanker_mean_rt_ms": 540 - 24 * k_session + 28 * policy_session + rng.normal(scale=12),
                    "stroop_throughput": 1.4 + 0.18 * k_session + rng.normal(scale=0.05),
                    "flanker_throughput": 1.6 + 0.18 * k_session + rng.normal(scale=0.05),
                    "stroop_interference_rt_ms": 120 - 18 * c_session + rng.normal(scale=8),
                    "flanker_interference_rt_ms": 45 - 9 * c_session + rng.normal(scale=5),
                    "stroop_interference_accuracy": 0.08 - 0.025 * c_session + rng.normal(scale=0.01),
                    "flanker_interference_accuracy": 0.06 - 0.02 * c_session + rng.normal(scale=0.01),
                    "sart_commission_rate": 0.30 - 0.05 * v_session + rng.normal(scale=0.015),
                    "sart_omission_rate": 0.05 - 0.02 * v_session + rng.normal(scale=0.01),
                    "sart_anticipatory_rate": 0.04 - 0.02 * v_session + rng.normal(scale=0.01),
                    "sart_go_rt_cv": 0.35 - 0.04 * v_session + rng.normal(scale=0.02),
                    "sart_pre_failure_speeding_ms": 80 - 10 * v_session + rng.normal(scale=8),
                }
            )
    return pd.DataFrame(rows)
