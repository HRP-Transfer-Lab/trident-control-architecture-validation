import json
from pathlib import Path

import pytest

from trident_validation.config import (
    ConfigValidationError,
    load_config_directory,
    require_registered_upstream_commit,
    validate_upstream_registry,
)
from trident_validation.provenance import build_run_manifest, hash_file, write_run_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_config_directory_and_upstream_registry_validate():
    configs = load_config_directory(ROOT / "config")
    registry = validate_upstream_registry(configs["upstream_evidence"])

    assert "flow_zone_zone_validation" in registry
    record = require_registered_upstream_commit(
        registry,
        "HRP-Transfer-Lab/flow-zone-zone-validation",
        "2d8d479befd73155d2215c1fef80a60cd27eaa5b",
    )
    assert record.import_policy["mode"] == "reference_only"


def test_unregistered_upstream_commit_fails_loudly():
    configs = load_config_directory(ROOT / "config")
    registry = validate_upstream_registry(configs["upstream_evidence"])

    with pytest.raises(ConfigValidationError):
        require_registered_upstream_commit(
            registry,
            "HRP-Transfer-Lab/flow-zone-zone-validation",
            "0" * 40,
        )


def test_hash_file_has_algorithm_prefix():
    digest = hash_file(ROOT / "config" / "model_tournament.yaml")

    assert digest.startswith("sha256:")
    assert len(digest) == len("sha256:") + 64


def test_run_manifest_is_deterministic_with_fixed_timestamp():
    kwargs = dict(
        protocol="flow-zone-validation-v1",
        repo_root=ROOT,
        config_paths={
            "datasets": ROOT / "config" / "datasets.yaml",
            "model_tournament": ROOT / "config" / "model_tournament.yaml",
        },
        upstream={
            "flow_zone_zone_validation": "2d8d479befd73155d2215c1fef80a60cd27eaa5b",
        },
        seed=20260807,
        split={
            "name": "participant_group_kfold",
            "seed": 20260807,
            "fold_index": 0,
            "n_splits": 5,
            "train_participants": ["must_not_appear"],
            "test_participants": ["must_not_appear"],
        },
        models=["M0_general_performance"],
        model_metadata={
            "M0_probabilistic_general_performance": {
                "imputation": {
                    "imputation_strategy": "training_feature_mean",
                    "fitted_on_n_rows": 10,
                },
                "train_participants": ["must_not_appear"],
            }
        },
        timestamp_utc="2026-08-07T00:00:00+00:00",
        include_dirty=False,
    )

    manifest_a = build_run_manifest(**kwargs)
    manifest_b = build_run_manifest(**kwargs)
    output_dir = ROOT / ".test-artifacts"
    output_path = output_dir / "manifest.json"
    try:
        write_run_manifest(output_path, manifest_a)
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    finally:
        if output_path.exists():
            output_path.unlink()
        if output_dir.exists():
            output_dir.rmdir()

    assert manifest_a == manifest_b
    assert payload["split"]["name"] == "participant_group_kfold"
    assert "train_participants" not in payload["split"]
    assert "test_participants" not in payload["split"]
    model_metadata = payload["model_metadata"]["M0_probabilistic_general_performance"]
    assert model_metadata["imputation"]["imputation_strategy"] == "training_feature_mean"
    assert "train_participants" not in model_metadata
