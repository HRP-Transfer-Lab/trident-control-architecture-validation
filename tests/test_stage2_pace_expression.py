from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from trident_validation.config import ConfigValidationError, load_yaml_config
import trident_validation.mechanistic.stage2_pace_expression as stage2
from trident_validation.mechanistic.stage2_pace_expression import (
    PACE_COLUMNS,
    _add_pace_expression_indicators,
    _validate_config,
    run_stage2_pace_expression_analysis,
)
from trident_validation.provenance import hash_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/public_stage2_pace_expression_v1.yaml"


def test_stage2_config_is_descriptive_and_claim_bounded():
    config = load_yaml_config(CONFIG_PATH)
    _validate_config(config)

    assert config["analysis"]["formal_claims_allowed"] is False
    assert config["analysis"]["pace_ontology_claim_allowed"] is False
    assert config["analysis"]["forced_four_profile_claim_allowed"] is False
    assert config["analysis"]["dynamic_regime_allowed"] is False
    assert config["pace_expression"]["derived_inside_training_folds_only"] is True
    assert tuple(config["pace_expression"]["indicators"]) == PACE_COLUMNS
    configured_columns = {
        column
        for group in config["feature_groups"].values()
        for column in group["columns"]
    }
    assert configured_columns.isdisjoint(config["forbidden_columns"])


def test_stage2_rejects_forbidden_profile_column_as_feature():
    config = load_yaml_config(CONFIG_PATH)
    config["feature_groups"]["K"]["columns"]["control_profile_probability"] = 1

    with pytest.raises(ConfigValidationError, match="forbidden columns"):
        _validate_config(config)


def test_stage2_registered_local_source_path_resolves_when_available():
    config = load_yaml_config(CONFIG_PATH)
    source_path = (ROOT / config["inputs"]["paired_session_features_path"]).resolve()

    if source_path.exists():
        assert hash_file(source_path).lower() == (
            "sha256:" + config["inputs"]["paired_session_features_sha256"].lower()
        )


def test_pace_expression_indicators_are_not_forced_four_profile_assignments():
    composites = pd.DataFrame(
        {
            "K_current": [1.0, 1.0, -1.0, -1.0, 1.0],
            "C_signal_current": [1.0, 1.0, 1.0, -1.0, -1.0],
            "V_current": [1.0, -1.0, 1.0, -1.0, 1.0],
        }
    )
    labelled = _add_pace_expression_indicators(
        composites,
        {"K": 0.0, "C_signal": 0.0, "V": 0.0},
    )

    assert labelled["regulated"].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert labelled["brittle"].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0]
    assert labelled["compensatory"].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]
    assert labelled["overloaded"].tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]
    assert labelled.loc[4, list(PACE_COLUMNS)].sum() == 0.0


def test_stage2_runner_produces_participant_free_summary(monkeypatch):
    config = load_yaml_config(CONFIG_PATH)
    df = _synthetic_stage2_frame(n_participants=70)
    work_dir = ROOT / "reports" / "generated" / "test_stage2_pace_expression"
    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = work_dir / "paired.parquet"
    input_path.write_text("synthetic parquet placeholder", encoding="utf-8")
    monkeypatch.setattr(stage2.pd, "read_parquet", lambda path: df.copy())
    config["inputs"]["paired_session_features_path"] = str(input_path)
    config["inputs"]["paired_session_features_sha256"] = hash_file(input_path).replace("sha256:", "").upper()
    config["validation"]["bootstrap_iterations"] = 50
    config["outputs"]["output_dir"] = str(work_dir / "out")
    config["outputs"]["report_md"] = str(work_dir / "report.md")
    config_path = work_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = run_stage2_pace_expression_analysis(
        config_path,
        repo_root=ROOT,
        output_dir=work_dir / "out",
        write_outputs=True,
    )

    assert result.summary["formal_claims_allowed"] is False
    assert result.summary["upstream_profile_columns_used"] is False
    assert result.summary["forced_four_profile_claim_allowed"] is False
    assert result.summary["forbidden_columns_used"] == []
    assert result.summary["complete_adjacent_pairs"] == 70
    assert set(result.model_scores["model_id"]) == set(config["candidate_models"])
    assert "Y_behavior_next" in set(result.contrasts["target_id"])
    assert "participant_id" not in result.model_scores.columns
    assert "participant_id" not in result.contrasts.columns
    assert (work_dir / "out" / "stage2_summary.json").exists()
    assert (work_dir / "report.md").exists()


def _synthetic_stage2_frame(n_participants: int) -> pd.DataFrame:
    rng = np.random.default_rng(456)
    rows = []
    for participant in range(n_participants):
        k = rng.normal()
        c = rng.normal()
        v = rng.normal()
        pace_bonus = 0.35 if k > 0 and c > 0 and v > 0 else -0.20 if k < 0 and c < 0 and v < 0 else 0.0
        for session_index, session_type in enumerate(("online", "lab1")):
            next_shift = pace_bonus if session_index == 1 else 0.0
            noise = rng.normal(scale=0.12, size=15)
            row = {
                "participant_id": f"P{participant:03d}",
                "session_id": f"P{participant:03d}:{session_type}",
                "session_type": session_type,
                "control_profile": "forbidden",
                "control_profile_probability": 0.99,
                "low_engagement_candidate": False,
                "high_engagement_candidate": False,
                "stroop_accuracy": 0.85 + 0.05 * k + next_shift + noise[0],
                "flanker_accuracy": 0.86 + 0.05 * k + next_shift + noise[1],
                "stroop_mean_rt_ms": 600 - 30 * k - 40 * next_shift + noise[2],
                "flanker_mean_rt_ms": 520 - 25 * k - 35 * next_shift + noise[3],
                "stroop_throughput": 1.4 + 0.2 * k + next_shift + noise[4],
                "flanker_throughput": 1.6 + 0.2 * k + next_shift + noise[5],
                "stroop_interference_rt_ms": 120 - 20 * c - 20 * next_shift + noise[6],
                "flanker_interference_rt_ms": 45 - 10 * c - 15 * next_shift + noise[7],
                "stroop_interference_accuracy": 0.08 - 0.03 * c - 0.05 * next_shift + noise[8],
                "flanker_interference_accuracy": 0.06 - 0.02 * c - 0.05 * next_shift + noise[9],
                "sart_commission_rate": 0.30 - 0.06 * v - 0.08 * next_shift + noise[10],
                "sart_omission_rate": 0.05 - 0.02 * v - 0.05 * next_shift + noise[11],
                "sart_anticipatory_rate": 0.04 - 0.02 * v - 0.05 * next_shift + noise[12],
                "sart_go_rt_cv": 0.35 - 0.04 * v - 0.08 * next_shift + noise[13],
                "sart_pre_failure_speeding_ms": 80 - 12 * v - 15 * next_shift + noise[14],
            }
            rows.append(row)
    return pd.DataFrame(rows)
