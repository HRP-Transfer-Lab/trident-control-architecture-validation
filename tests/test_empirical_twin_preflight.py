from pathlib import Path
import uuid

import pytest

from trident_validation.schema import validate_window_schema
from trident_validation.synthetic import empirical_twin_preflight as preflight
from trident_validation.synthetic.fixtures import make_synthetic_window_table


ROOT = Path(__file__).resolve().parents[1]


def test_empirical_twin_preflight_smoke_writes_outputs():
    output_dir = ROOT / "reports" / "generated" / "test_empirical_twin_preflight" / uuid.uuid4().hex

    result = preflight.run_empirical_twin_preflight(output_dir=output_dir)

    assert result["study_id"] == "empirical_twin_preflight_v1"
    assert result["input_mode"] == "fallback_smoke_fixture"
    assert result["formal_claims_allowed"] is False
    assert result["n_templates"] > 0
    assert (output_dir / "template_summary.csv").exists()
    assert (output_dir / "template_covariance.csv").exists()
    assert (output_dir / "variance_decomposition.csv").exists()
    assert (output_dir / "temporal_summary.csv").exists()
    assert (output_dir / "missingness_summary.csv").exists()
    assert (output_dir / "empirical_twin_preflight_summary.json").exists()


def test_empirical_twin_preflight_accepts_input_table(tmp_path):
    frame = make_synthetic_window_table(
        seed=123,
        n_datasets=3,
        participants_per_dataset=2,
        sessions_per_participant=1,
        min_windows_per_session=1,
        max_windows_per_session=1,
    )
    input_path = tmp_path / "canonical_windows.csv"
    output_dir = tmp_path / "preflight_outputs"
    frame.to_csv(input_path, index=False)

    result = preflight.run_empirical_twin_preflight(
        input_tables=[input_path],
        output_dir=output_dir,
        allow_fallback_fixture=False,
    )

    assert result["input_mode"] == "empirical_window_tables"
    assert result["n_rows"] == frame.shape[0]
    assert (output_dir / "template_summary.csv").exists()


def test_empirical_twin_preflight_accepts_flowzone_cognitive_windows(tmp_path):
    frame = _minimal_flowzone_cognitive_windows()
    input_path = tmp_path / "cognitive_windows.csv"
    output_dir = tmp_path / "preflight_outputs"
    frame.to_csv(input_path, index=False)

    result = preflight.run_empirical_twin_preflight(
        input_tables=[input_path],
        output_dir=output_dir,
        allow_fallback_fixture=False,
    )

    assert result["input_mode"] == "empirical_window_tables"
    assert result["n_rows"] == frame.shape[0]
    assert result["n_templates"] == 2
    assert (output_dir / "template_summary.csv").exists()


def test_flowzone_cognitive_windows_adapter_outputs_canonical_schema():
    canonical = preflight.canonicalise_empirical_window_table(
        _minimal_flowzone_cognitive_windows()
    )

    report = validate_window_schema(canonical)

    assert report.n_rows == 4
    assert set(canonical["source_dataset"]) == {"acdc_web"}
    assert set(canonical["task_id"]) == {"Stroop", "SART"}
    assert canonical["has_vigilance"].sum() == 2


def test_empirical_twin_preflight_no_fallback_requires_input():
    with pytest.raises(ValueError, match="no empirical window tables provided"):
        preflight.run_empirical_twin_preflight(
            input_tables=[],
            allow_fallback_fixture=False,
        )


def test_configured_input_paths_accept_dict_entries(tmp_path):
    paths = preflight._configured_input_paths(
        [{"path": "windows.csv"}],
        config_base_dir=tmp_path,
    )

    assert paths == [tmp_path / "windows.csv"]


def test_empirical_nuisance_rejects_truth_columns():
    frame = make_synthetic_window_table(
        seed=123,
        n_datasets=3,
        participants_per_dataset=2,
        sessions_per_participant=1,
        min_windows_per_session=1,
        max_windows_per_session=1,
    )
    frame["synthetic_world_id"] = "not_allowed"

    with pytest.raises(ValueError, match="ground-truth columns"):
        preflight.estimate_empirical_nuisance(frame)


def test_empirical_nuisance_estimates_source_task_templates():
    frame = make_synthetic_window_table(
        seed=123,
        n_datasets=3,
        participants_per_dataset=3,
        sessions_per_participant=1,
        min_windows_per_session=1,
        max_windows_per_session=2,
    )

    nuisance = preflight.estimate_empirical_nuisance(frame)

    assert set(nuisance) == {
        "template_summary",
        "feature_summary",
        "template_covariance",
        "variance_decomposition",
        "temporal_summary",
        "missingness_summary",
    }
    assert {"source_dataset", "task_id", "n_rows", "n_participants"}.issubset(
        nuisance["template_summary"].columns
    )
    assert nuisance["feature_summary"]["feature"].nunique() == 5


def _minimal_flowzone_cognitive_windows():
    import pandas as pd

    return pd.DataFrame(
        [
            {
                "window_id": "w1",
                "dataset_id": "acdc_web",
                "participant_id": "p1",
                "task_family": "Stroop",
                "block_raw": "b1",
                "window_size": 80,
                "window_index": 0,
                "n_trials": 80,
                "n_valid_correct_rt": 76,
                "accuracy": 0.90,
                "median_rt_ms": 640.0,
                "mean_rt_ms": 660.0,
                "rt_cv": 0.21,
                "throughput_proxy": 1.41,
                "control_cost_supported": True,
                "control_cost_rt_ms": 70.0,
                "control_cost_acc": 0.04,
                "pes_supported": True,
                "post_error_slowing_ms": 25.0,
                "error_burstiness": 0.12,
            },
            {
                "window_id": "w2",
                "dataset_id": "acdc_web",
                "participant_id": "p1",
                "task_family": "Stroop",
                "block_raw": "b1",
                "window_size": 80,
                "window_index": 1,
                "n_trials": 80,
                "n_valid_correct_rt": 74,
                "accuracy": 0.88,
                "median_rt_ms": 670.0,
                "mean_rt_ms": 690.0,
                "rt_cv": 0.23,
                "throughput_proxy": 1.31,
                "control_cost_supported": True,
                "control_cost_rt_ms": 82.0,
                "control_cost_acc": 0.05,
                "pes_supported": True,
                "post_error_slowing_ms": 30.0,
                "error_burstiness": 0.14,
            },
            {
                "window_id": "w3",
                "dataset_id": "acdc_web",
                "participant_id": "p2",
                "task_family": "SART",
                "block_raw": "b1",
                "window_size": 80,
                "window_index": 0,
                "n_trials": 80,
                "n_valid_correct_rt": 72,
                "accuracy": 0.86,
                "median_rt_ms": 600.0,
                "mean_rt_ms": 620.0,
                "rt_cv": 0.25,
                "throughput_proxy": 1.43,
                "nonresponse_rate": 0.08,
                "slow_tail_rate": 0.12,
                "fast_error_rate": 0.03,
                "rt_drift": -0.01,
            },
            {
                "window_id": "w4",
                "dataset_id": "acdc_web",
                "participant_id": "p2",
                "task_family": "SART",
                "block_raw": "b1",
                "window_size": 80,
                "window_index": 1,
                "n_trials": 80,
                "n_valid_correct_rt": 70,
                "accuracy": 0.84,
                "median_rt_ms": 625.0,
                "mean_rt_ms": 640.0,
                "rt_cv": 0.27,
                "throughput_proxy": 1.34,
                "nonresponse_rate": 0.10,
                "slow_tail_rate": 0.13,
                "fast_error_rate": 0.04,
                "rt_drift": -0.02,
            },
        ]
    )
