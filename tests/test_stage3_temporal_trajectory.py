from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from trident_validation.config import ConfigValidationError, load_yaml_config
import trident_validation.mechanistic.stage3_temporal_trajectory as stage3
from trident_validation.mechanistic.stage3_temporal_trajectory import (
    _build_three_session_sequences,
    _validate_config,
    run_stage3_temporal_trajectory_analysis,
)
from trident_validation.mechanistic.stage1_k_c_v import _participant_folds
from trident_validation.provenance import hash_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/public_stage3_temporal_trajectory_v1.yaml"


def test_stage3_config_is_temporal_and_claim_bounded():
    config = load_yaml_config(CONFIG_PATH)
    _validate_config(config)

    assert config["analysis"]["formal_claims_allowed"] is False
    assert config["analysis"]["adaptive_locked_scattered_regime_claim_allowed"] is False
    assert config["analysis"]["neural_criticality_claim_allowed"] is False
    assert config["trajectory"]["regime_labels_allowed"] is False
    assert config["trajectory"]["required_session_sequence"] == ["online", "lab1", "lab2"]
    configured_columns = {
        column
        for group in config["feature_groups"].values()
        for column in group["columns"]
    }
    assert configured_columns.isdisjoint(config["forbidden_columns"])


def test_stage3_rejects_forbidden_profile_column_as_feature():
    config = load_yaml_config(CONFIG_PATH)
    config["feature_groups"]["K"]["columns"]["control_profile"] = 1

    with pytest.raises(ConfigValidationError, match="forbidden columns"):
        _validate_config(config)


def test_stage3_registered_local_source_path_resolves_when_available():
    config = load_yaml_config(CONFIG_PATH)
    source_path = (ROOT / config["inputs"]["paired_session_features_path"]).resolve()

    if source_path.exists():
        assert hash_file(source_path).lower() == (
            "sha256:" + config["inputs"]["paired_session_features_sha256"].lower()
        )


def test_stage3_builds_complete_three_session_sequences_and_isolated_folds():
    config = load_yaml_config(CONFIG_PATH)
    df = _synthetic_stage3_frame(n_participants=20)

    sequences = _build_three_session_sequences(df, config)
    folds = _participant_folds(sequences["participant_id"], 4, 99)

    assert len(sequences) == 20
    assert sequences["participant_id"].nunique() == 20
    assert {"previous__stroop_accuracy", "current__stroop_accuracy", "next__stroop_accuracy"}.issubset(
        sequences.columns
    )
    for fold in set(folds):
        test_participants = set(sequences.loc[folds == fold, "participant_id"])
        train_participants = set(sequences.loc[folds != fold, "participant_id"])
        assert test_participants.isdisjoint(train_participants)


def test_stage3_runner_produces_participant_free_summary(monkeypatch):
    config = load_yaml_config(CONFIG_PATH)
    df = _synthetic_stage3_frame(n_participants=80)
    work_dir = ROOT / "reports" / "generated" / "test_stage3_temporal_trajectory"
    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = work_dir / "paired.parquet"
    input_path.write_text("synthetic parquet placeholder", encoding="utf-8")
    monkeypatch.setattr(stage3.pd, "read_parquet", lambda path: df.copy())
    config["inputs"]["paired_session_features_path"] = str(input_path)
    config["inputs"]["paired_session_features_sha256"] = hash_file(input_path).replace("sha256:", "").upper()
    config["validation"]["bootstrap_iterations"] = 50
    config["outputs"]["output_dir"] = str(work_dir / "out")
    config["outputs"]["report_md"] = str(work_dir / "report.md")
    config_path = work_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = run_stage3_temporal_trajectory_analysis(
        config_path,
        repo_root=ROOT,
        output_dir=work_dir / "out",
        write_outputs=True,
    )

    assert result.summary["formal_claims_allowed"] is False
    assert result.summary["regime_labels_used"] is False
    assert result.summary["adaptive_locked_scattered_regime_claim_allowed"] is False
    assert result.summary["forbidden_columns_used"] == []
    assert result.summary["complete_three_session_sequences"] == 80
    assert set(result.model_scores["model_id"]) == set(config["candidate_models"])
    assert "Y_behavior_next" in set(result.contrasts["target_id"])
    assert "participant_id" not in result.model_scores.columns
    assert "participant_id" not in result.contrasts.columns
    assert (work_dir / "out" / "stage3_summary.json").exists()
    assert (work_dir / "report.md").exists()


def _synthetic_stage3_frame(n_participants: int) -> pd.DataFrame:
    rng = np.random.default_rng(789)
    rows = []
    for participant in range(n_participants):
        k = rng.normal()
        c = rng.normal()
        v = rng.normal()
        dk = rng.normal(scale=0.4)
        dc = rng.normal(scale=0.4)
        dv = rng.normal(scale=0.4)
        states = {
            "online": (k, c, v),
            "lab1": (k + dk, c + dc, v + dv),
            "lab2": (k + dk + 0.6 * dk, c + dc + 0.6 * dc, v + dv + 0.6 * dv),
        }
        for session_type, (ks, cs, vs) in states.items():
            noise = rng.normal(scale=0.08, size=15)
            row = {
                "participant_id": f"P{participant:03d}",
                "session_id": f"P{participant:03d}:{session_type}",
                "session_type": session_type,
                "control_profile": "forbidden",
                "control_profile_probability": 0.99,
                "low_engagement_candidate": False,
                "high_engagement_candidate": False,
                "stroop_accuracy": 0.85 + 0.05 * ks + noise[0],
                "flanker_accuracy": 0.86 + 0.05 * ks + noise[1],
                "stroop_mean_rt_ms": 600 - 30 * ks + noise[2],
                "flanker_mean_rt_ms": 520 - 25 * ks + noise[3],
                "stroop_throughput": 1.4 + 0.2 * ks + noise[4],
                "flanker_throughput": 1.6 + 0.2 * ks + noise[5],
                "stroop_interference_rt_ms": 120 - 20 * cs + noise[6],
                "flanker_interference_rt_ms": 45 - 10 * cs + noise[7],
                "stroop_interference_accuracy": 0.08 - 0.03 * cs + noise[8],
                "flanker_interference_accuracy": 0.06 - 0.02 * cs + noise[9],
                "sart_commission_rate": 0.30 - 0.06 * vs + noise[10],
                "sart_omission_rate": 0.05 - 0.02 * vs + noise[11],
                "sart_anticipatory_rate": 0.04 - 0.02 * vs + noise[12],
                "sart_go_rt_cv": 0.35 - 0.04 * vs + noise[13],
                "sart_pre_failure_speeding_ms": 80 - 12 * vs + noise[14],
            }
            rows.append(row)
    return pd.DataFrame(rows)
