from pathlib import Path
import json
import uuid

import pandas as pd
import pytest

from trident_validation.config import load_yaml_config
from trident_validation.synthetic import empirical_twin_pilot
from trident_validation.synthetic.empirical_twin_v1 import load_background_spec, _dataframe_hash


ROOT = Path(__file__).resolve().parents[1]


def _scratch_dir() -> Path:
    path = ROOT / "reports/generated/test_empirical_twin_pilot" / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def _pilot_config(base_path: Path) -> dict:
    config = load_yaml_config(ROOT / "config/empirical_twin_v1_pilot.yaml")
    config["pilot_design"]["schedule_csv"] = str(base_path / "pilot_schedule.csv")
    config["pilot_design"]["schedule_json"] = str(base_path / "pilot_schedule.json")
    config["outputs"]["directory"] = str(base_path / "pilot_outputs")
    return config


def test_pilot_template_eligibility_is_matched_stroop_flanker_only():
    background = load_background_spec(
        ROOT / "reports/generated/empirical_twin_preflight_v1_full_acdc",
        ROOT / "reports/generated/empirical_twin_preflight_v1_full_acdc_paired_session",
    )

    by_task = empirical_twin_pilot._eligible_templates_by_task(
        background.template_summary,
        ("Stroop", "Flanker"),
    )

    assert by_task["Stroop"].shape[0] == 34
    assert by_task["Flanker"].shape[0] == 12
    assert set(by_task) == {"Stroop", "Flanker"}
    assert "Simon" not in set(pd.concat(by_task.values())["task_id"])
    assert "SART" not in set(pd.concat(by_task.values())["task_id"])
    with pytest.raises(ValueError, match="no contract-eligible templates"):
        empirical_twin_pilot._eligible_templates_by_task(
            background.template_summary,
            ("SART",),
        )


def test_prepare_pilot_schedule_is_immutable_task_balanced_and_hashed():
    config = _pilot_config(_scratch_dir())
    output_dir = Path(config["outputs"]["directory"])

    schedule, schedule_hash = empirical_twin_pilot.prepare_pilot_schedule(
        config,
        config_path=ROOT / "config/empirical_twin_v1_pilot.yaml",
        output_dir=output_dir,
    )

    assert schedule.shape[0] == 50
    assert set(schedule["world_id"]) == {"ETW0", "ETW1", "ETW2", "ETW3", "ETW4"}
    assert schedule.groupby("world_id").size().eq(10).all()
    assert schedule["empirical_background_contract_sha"].eq(
        "1e9c11e030dc0706c8941dfc08489bb6f63cac5d"
    ).all()
    assert schedule["static_model_contract_sha"].eq(
        "81d22f2a942afd1652c169935f058b1634922d30"
    ).all()
    assert schedule["generator_sha"].eq("9699427176961a3064e9db8da23a3b287a441b1b").all()

    replicate_pairs = schedule.drop_duplicates("replicate_index").sort_values("replicate_index")
    assert replicate_pairs.shape[0] == 10
    assert replicate_pairs["stroop_template_id"].nunique() == 10
    assert replicate_pairs["flanker_template_id"].nunique() == 10
    assert replicate_pairs["stroop_template_task_id"].eq("Stroop").all()
    assert replicate_pairs["flanker_template_task_id"].eq("Flanker").all()

    for _, group in schedule.groupby("replicate_index"):
        assert group["stroop_template_id"].nunique() == 1
        assert group["flanker_template_id"].nunique() == 1
        assert set(group["world_id"]) == {"ETW0", "ETW1", "ETW2", "ETW3", "ETW4"}
        task_sets = {
            tuple(record["task_id"] for record in json.loads(value))
            for value in group["template_records_json"]
        }
        assert task_sets == {("Stroop", "Flanker")}

    schedule_path = Path(config["pilot_design"]["schedule_csv"])
    manifest = json.loads((output_dir / "pilot_manifest.json").read_text(encoding="utf-8"))
    assert schedule_path.exists()
    assert Path(config["pilot_design"]["schedule_json"]).exists()
    assert schedule_hash == _dataframe_hash(pd.read_csv(schedule_path))
    assert manifest["schedule_hash"] == schedule_hash
    assert manifest["matched_support_intersection"] == ["Stroop", "Flanker"]


def test_checkpoint_checksum_validation_detects_tampering():
    path = _scratch_dir() / "checkpoint.json"
    schedule_hash = "sha256:test"
    payload = {
        "status": "complete",
        "unit_id": "ETW0_replicate_000",
        "schedule_hash": schedule_hash,
        "runtime": {"unit_runtime_seconds": 1.0},
    }

    empirical_twin_pilot._atomic_write_checkpoint(path, payload)
    assert empirical_twin_pilot._checkpoint_valid(path, schedule_hash)

    data = json.loads(path.read_text(encoding="utf-8"))
    data["status"] = "failed"
    path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")

    assert not empirical_twin_pilot._checkpoint_valid(path, schedule_hash)
