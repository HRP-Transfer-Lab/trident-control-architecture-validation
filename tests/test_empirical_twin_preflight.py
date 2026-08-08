from pathlib import Path
import json
import uuid

import pandas as pd
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
    assert (output_dir / "template_support.csv").exists()
    assert (output_dir / "covariance_raw.csv").exists()
    assert (output_dir / "covariance_between_person.csv").exists()
    assert (output_dir / "covariance_session.csv").exists()
    assert (output_dir / "covariance_within_session.csv").exists()
    assert (output_dir / "variance_decomposition.csv").exists()
    assert (output_dir / "temporal_summary.csv").exists()
    assert (output_dir / "missingness_summary.csv").exists()
    assert (output_dir / "empirical_twin_preflight_provenance.json").exists()
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
    provenance = json.loads(
        (output_dir / "empirical_twin_preflight_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert provenance["input_tables"][0]["checksum"].startswith("sha256:")


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
        "template_support",
        "feature_summary",
        "covariance_raw",
        "covariance_between_person",
        "covariance_session",
        "covariance_within_session",
        "variance_decomposition",
        "temporal_summary",
        "missingness_summary",
    }
    assert {"source_dataset", "task_id", "n_rows", "n_participants"}.issubset(
        nuisance["template_summary"].columns
    )
    assert nuisance["feature_summary"]["feature"].nunique() == 5


def test_nested_variance_decomposition_reports_session_and_window_components():
    frame = _nested_canonical_fixture()

    nuisance = preflight.estimate_empirical_nuisance(frame, feature_columns=("accuracy",))
    variance = nuisance["variance_decomposition"]
    row = variance[(variance["source_dataset"] == "source_a") & (variance["task_id"] == "stroop")].iloc[0]

    assert row["between_support_status"] == "estimated"
    assert row["session_support_status"] == "estimated"
    assert row["window_support_status"] == "estimated"
    assert row["between_participant_variance"] > 0
    assert row["session_within_participant_variance"] >= 0
    assert row["window_within_session_variance"] >= 0


def test_sparse_template_reports_unsupported_components_without_pooling():
    frame = _nested_canonical_fixture().iloc[[0]].copy()

    nuisance = preflight.estimate_empirical_nuisance(frame, feature_columns=("accuracy",))
    variance = nuisance["variance_decomposition"].iloc[0]
    support = nuisance["template_support"].iloc[0]

    assert variance["between_support_status"] == "unsupported"
    assert variance["session_support_status"] == "unsupported"
    assert variance["window_support_status"] == "unsupported"
    assert support["variance_component_support"] == "unsupported"


def test_temporal_autocorrelation_does_not_cross_tasks_or_sessions():
    frame = _cross_task_temporal_fixture()

    nuisance = preflight.estimate_empirical_nuisance(frame, feature_columns=("accuracy",))
    temporal = nuisance["temporal_summary"]
    stroop = temporal[temporal["task_id"] == "stroop"].iloc[0]
    flanker = temporal[temporal["task_id"] == "flanker"].iloc[0]

    assert stroop["lag1_support_status"] == "unsupported"
    assert flanker["lag1_support_status"] == "unsupported"


def test_covariance_levels_separate_between_person_and_within_session():
    frame = _nested_canonical_fixture()

    nuisance = preflight.estimate_empirical_nuisance(
        frame,
        feature_columns=("accuracy", "median_rt_ms"),
    )

    assert set(nuisance["covariance_between_person"]["covariance_level"]) == {
        "between_person"
    }
    assert set(nuisance["covariance_within_session"]["covariance_level"]) == {
        "within_session"
    }
    assert (
        nuisance["covariance_between_person"]["support_status"] == "estimated"
    ).any()


def test_report_outputs_do_not_include_participant_ids(tmp_path):
    frame = _nested_canonical_fixture()
    input_path = tmp_path / "canonical_windows.csv"
    output_dir = tmp_path / "preflight_outputs"
    frame.to_csv(input_path, index=False)

    preflight.run_empirical_twin_preflight(
        input_tables=[input_path],
        output_dir=output_dir,
        allow_fallback_fixture=False,
    )

    report = (output_dir / "empirical_twin_preflight_report.md").read_text(
        encoding="utf-8"
    )
    summary = (output_dir / "empirical_twin_preflight_summary.json").read_text(
        encoding="utf-8"
    )

    assert "p001" not in report
    assert "p001" not in summary


def test_empirical_twin_preflight_is_deterministic_on_fixed_input(tmp_path):
    frame = _nested_canonical_fixture()
    first = preflight.estimate_empirical_nuisance(frame, feature_columns=("accuracy",))
    second = preflight.estimate_empirical_nuisance(frame, feature_columns=("accuracy",))

    pd.testing.assert_frame_equal(
        first["variance_decomposition"],
        second["variance_decomposition"],
    )
    pd.testing.assert_frame_equal(first["temporal_summary"], second["temporal_summary"])


def test_paired_session_adapter_excludes_profile_columns_and_keeps_session_level():
    frame = _paired_session_fixture()

    long_frame, audit = preflight.canonicalise_paired_session_table(frame)

    assert set(long_frame["task_id"]) == {"Stroop", "Flanker", "SART"}
    assert "control_profile" not in long_frame.columns
    assert "control_profile_probability" not in long_frame.columns
    forbidden = audit[audit["audit_item"] == "forbidden_columns_excluded"].iloc[0]
    assert int(forbidden["value"]) == 2
    assert "control_profile" in forbidden["detail"]
    emitted = audit[audit["audit_item"] == "session_level_rows_emitted"].iloc[0]
    assert "no window semantics created" in emitted["detail"]


def test_paired_session_background_estimates_repeated_session_components():
    outputs = preflight.estimate_paired_session_background(_paired_session_fixture())

    assert {
        "paired_session_adapter_audit",
        "paired_session_support",
        "paired_session_variance_decomposition",
        "paired_session_covariance_session",
        "paired_session_practice_summary",
        "paired_session_cross_task_covariance",
    }.issubset(outputs)
    variance = outputs["paired_session_variance_decomposition"]
    row = variance[
        (variance["task_id"] == "Stroop") & (variance["feature"] == "accuracy")
    ].iloc[0]
    assert row["session_support_status"] == "estimated"
    assert row["window_support_status"] == "unsupported"
    unavailable = variance[
        (variance["task_id"] == "SART") & (variance["feature"] == "accuracy")
    ].iloc[0]
    assert unavailable["n_observed"] == 0
    assert unavailable["session_support_status"] == "unsupported"
    assert unavailable["support_reason"] == "feature_structurally_unavailable_for_task"
    support = outputs["paired_session_support"]
    assert set(support["window_level_support"]) == {"unsupported"}


def test_paired_session_cli_outputs_are_aggregate_only(tmp_path):
    windows = make_synthetic_window_table(
        seed=123,
        n_datasets=3,
        participants_per_dataset=2,
        sessions_per_participant=1,
        min_windows_per_session=1,
        max_windows_per_session=1,
    )
    window_path = tmp_path / "windows.csv"
    paired_path = tmp_path / "paired_sessions.csv"
    output_dir = tmp_path / "outputs"
    windows.to_csv(window_path, index=False)
    _paired_session_fixture().to_csv(paired_path, index=False)

    result = preflight.run_empirical_twin_preflight(
        input_tables=[window_path],
        paired_session_tables=[paired_path],
        output_dir=output_dir,
        allow_fallback_fixture=False,
    )

    assert result["paired_session_background_included"] is True
    assert (output_dir / "paired_session_variance_decomposition.csv").exists()
    report = (output_dir / "empirical_twin_preflight_report.md").read_text(
        encoding="utf-8"
    )
    assert "paired_p1" not in report


def _minimal_flowzone_cognitive_windows():
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


def _nested_canonical_fixture():
    rows = []
    for participant_index, person_shift in enumerate([0.0, 0.3], start=1):
        for session_index, session_shift in enumerate([0.0, 0.1], start=1):
            for window_index, window_shift in enumerate([0.0, 0.02, -0.01], start=1):
                accuracy = 0.75 + person_shift + session_shift + window_shift
                rows.append(
                    {
                        "source_dataset": "source_a",
                        "source_version": "test",
                        "participant_id": f"p{participant_index:03d}",
                        "session_id": f"s{session_index:02d}",
                        "task_id": "stroop",
                        "block_id": "b01",
                        "window_id": f"p{participant_index}_s{session_index}_w{window_index}",
                        "window_start_trial": 1 + (window_index - 1) * 20,
                        "window_end_trial": window_index * 20,
                        "n_trials_total": 20,
                        "n_trials_valid": 20,
                        "source_file_or_table": "test",
                        "source_commit_or_release": "test",
                        "source_hash_if_available": "sha256:test",
                        "preprocessing_version": "test",
                        "feature_version": "canonical-window-v1",
                        "accuracy": accuracy,
                        "median_rt_ms": 700.0 - 100.0 * accuracy,
                        "mean_response_speed": 1000.0 / (700.0 - 100.0 * accuracy),
                        "rt_cv": 0.2,
                        "throughput_proxy": accuracy * 100.0,
                        "trial_count": 20,
                        "practice_or_session_index": session_index,
                        "time_on_task": float(window_index),
                        "condition_mix": "mixed",
                        "congruency_mix": "balanced",
                        "switch_rate": float("nan"),
                        "lure_rate": float("nan"),
                        "difficulty_level": 1,
                        "soa_or_foreperiod": float("nan"),
                        "response_mapping": "test",
                        "input_device": "test",
                        "timing_quality": "test",
                        "browser_focus_flags": "none",
                        "has_conflict_cost": True,
                        "has_post_error": False,
                        "has_vigilance": False,
                        "has_switch_structure": False,
                        "has_confidence": False,
                        "has_change_point": False,
                        "conflict_cost_rt": 50.0,
                        "conflict_cost_accuracy": 0.03,
                        "post_error_adjustment": float("nan"),
                        "error_burstiness": float("nan"),
                        "recovery_slope": float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def _cross_task_temporal_fixture():
    frame = _nested_canonical_fixture().iloc[:4].copy()
    frame["participant_id"] = "p001"
    frame["session_id"] = ["s01", "s01", "s02", "s02"]
    frame["task_id"] = ["stroop", "flanker", "stroop", "flanker"]
    frame["window_id"] = [f"w{index}" for index in range(4)]
    frame["window_start_trial"] = [1, 21, 1, 21]
    frame["window_end_trial"] = [20, 40, 20, 40]
    frame["accuracy"] = [0.1, 0.9, 0.2, 0.8]
    return frame


def _paired_session_fixture():
    rows = []
    for participant_id, person_shift in [("paired_p1", 0.00), ("paired_p2", 0.10)]:
        for session_index, session_type in enumerate(["online", "lab1"], start=1):
            practice = 0.03 * (session_index - 1)
            rows.append(
                {
                    "participant_id": participant_id,
                    "session_id": f"{participant_id}_{session_type}",
                    "session_type": session_type,
                    "dataset_id": "paired",
                    "stroop_accuracy": 0.80 + person_shift + practice,
                    "stroop_mean_rt_ms": 700.0 - 20.0 * person_shift - 10.0 * practice,
                    "stroop_interference_rt_ms": 90.0 - 5.0 * practice,
                    "stroop_interference_accuracy": 0.05 - 0.01 * practice,
                    "stroop_throughput": 1.1 + person_shift + practice,
                    "flanker_accuracy": 0.82 + person_shift + practice,
                    "flanker_mean_rt_ms": 650.0 - 20.0 * person_shift - 10.0 * practice,
                    "flanker_interference_rt_ms": 70.0 - 5.0 * practice,
                    "flanker_interference_accuracy": 0.04 - 0.01 * practice,
                    "flanker_throughput": 1.2 + person_shift + practice,
                    "sart_commission_rate": 0.20 - 0.02 * practice,
                    "sart_omission_rate": 0.08 - 0.01 * practice,
                    "sart_anticipatory_rate": 0.02,
                    "sart_go_mean_rt_ms": 600.0 - 10.0 * practice,
                    "sart_go_rt_cv": 0.25 - 0.01 * practice,
                    "sart_pre_failure_speeding_ms": -25.0 + practice,
                    "control_profile": "forbidden_label",
                    "control_profile_probability": 0.99,
                }
            )
    return pd.DataFrame(rows)
