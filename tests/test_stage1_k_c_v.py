from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from trident_validation.config import ConfigValidationError, load_yaml_config
import trident_validation.mechanistic.stage1_k_c_v as stage1
from trident_validation.mechanistic.stage1_k_c_v import (
    run_stage1_k_c_v_analysis,
    _build_adjacent_pairs,
    _participant_folds,
    _validate_config,
)
from trident_validation.provenance import hash_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/public_stage1_k_c_v_v1.yaml"


def test_stage1_config_is_claim_bounded_and_uses_only_k_c_v():
    config = load_yaml_config(CONFIG_PATH)
    _validate_config(config)

    assert config["analysis"]["formal_claims_allowed"] is False
    assert config["analysis"]["transfer_outcomes_allowed"] is False
    assert config["analysis"]["pace_allowed"] is False
    assert config["analysis"]["dynamic_regime_allowed"] is False
    assert set(config["feature_groups"]) == {"K", "C_signal", "V"}
    configured_columns = {
        column
        for group in config["feature_groups"].values()
        for column in group["columns"]
    }
    assert configured_columns.isdisjoint(config["forbidden_columns"])


def test_stage1_registered_local_source_path_resolves_when_available():
    config = load_yaml_config(CONFIG_PATH)
    source_path = (ROOT / config["inputs"]["paired_session_features_path"]).resolve()

    if source_path.exists():
        assert hash_file(source_path).lower() == (
            "sha256:" + config["inputs"]["paired_session_features_sha256"].lower()
        )


def test_stage1_rejects_forbidden_profile_column_as_feature():
    config = load_yaml_config(CONFIG_PATH)
    config["feature_groups"]["K"]["columns"]["control_profile"] = 1

    with pytest.raises(ConfigValidationError, match="feature_groups.K.columns"):
        _validate_config(config)


def test_stage1_builds_adjacent_pairs_and_participant_isolated_folds():
    config = load_yaml_config(CONFIG_PATH)
    df = _synthetic_stage1_frame(n_participants=12)

    pairs = _build_adjacent_pairs(df, config)
    folds = _participant_folds(pairs["participant_id"], 3, 11)

    assert len(pairs) == 12
    assert pairs["participant_id"].nunique() == 12
    for fold in set(folds):
        test_participants = set(pairs.loc[folds == fold, "participant_id"])
        train_participants = set(pairs.loc[folds != fold, "participant_id"])
        assert test_participants.isdisjoint(train_participants)


def test_stage1_runner_produces_participant_free_summary(monkeypatch):
    config = load_yaml_config(CONFIG_PATH)
    df = _synthetic_stage1_frame(n_participants=60)
    work_dir = ROOT / "reports" / "generated" / "test_stage1_k_c_v"
    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = work_dir / "paired.parquet"
    input_path.write_text("synthetic parquet placeholder", encoding="utf-8")
    monkeypatch.setattr(stage1.pd, "read_parquet", lambda path: df.copy())
    config["inputs"]["paired_session_features_path"] = str(input_path)
    config["inputs"]["paired_session_features_sha256"] = hash_file(input_path).replace("sha256:", "").upper()
    config["validation"]["n_folds"] = 5
    config["validation"]["bootstrap_iterations"] = 50
    config["outputs"]["output_dir"] = str(work_dir / "out")
    config["outputs"]["report_md"] = str(work_dir / "report.md")
    config_path = work_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = run_stage1_k_c_v_analysis(
        config_path,
        repo_root=ROOT,
        output_dir=work_dir / "out",
        write_outputs=True,
    )

    assert result.summary["formal_claims_allowed"] is False
    assert result.summary["transfer_outcomes_used"] is False
    assert result.summary["forbidden_columns_used"] == []
    assert result.summary["complete_adjacent_pairs"] == 60
    assert set(result.model_scores["model_id"]) == set(config["candidate_models"])
    assert {"C_signal_next", "V_next", "K_next"}.issubset(set(result.contrasts["target_id"]))
    assert "participant_id" not in result.model_scores.columns
    assert "participant_id" not in result.contrasts.columns
    assert (work_dir / "out" / "stage1_summary.json").exists()
    assert (work_dir / "report.md").exists()


def _synthetic_stage1_frame(n_participants: int) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    rows = []
    for participant in range(n_participants):
        k = rng.normal()
        c = rng.normal()
        v = rng.normal()
        for session_index, session_type in enumerate(("online", "lab1")):
            drift = 0.15 * session_index
            noise = rng.normal(scale=0.15, size=15)
            row = {
                "participant_id": f"P{participant:03d}",
                "session_id": f"P{participant:03d}:{session_type}",
                "session_type": session_type,
                "control_profile": "forbidden",
                "control_profile_probability": 0.99,
                "low_engagement_candidate": False,
                "high_engagement_candidate": False,
                "stroop_accuracy": 0.85 + 0.05 * k + drift + noise[0],
                "flanker_accuracy": 0.86 + 0.05 * k + drift + noise[1],
                "stroop_mean_rt_ms": 600 - 30 * k - 5 * drift + noise[2],
                "flanker_mean_rt_ms": 520 - 25 * k - 5 * drift + noise[3],
                "stroop_throughput": 1.4 + 0.2 * k + drift + noise[4],
                "flanker_throughput": 1.6 + 0.2 * k + drift + noise[5],
                "stroop_interference_rt_ms": 120 - 20 * c + noise[6],
                "flanker_interference_rt_ms": 45 - 10 * c + noise[7],
                "stroop_interference_accuracy": 0.08 - 0.03 * c + noise[8],
                "flanker_interference_accuracy": 0.06 - 0.02 * c + noise[9],
                "sart_commission_rate": 0.30 - 0.06 * v + noise[10],
                "sart_omission_rate": 0.05 - 0.02 * v + noise[11],
                "sart_anticipatory_rate": 0.04 - 0.02 * v + noise[12],
                "sart_go_rt_cv": 0.35 - 0.04 * v + noise[13],
                "sart_pre_failure_speeding_ms": 80 - 12 * v + noise[14],
            }
            rows.append(row)
    return pd.DataFrame(rows)
