from pathlib import Path
import uuid

import numpy as np
import pandas as pd
import pytest

from trident_validation.models.base import FeatureRequirements, TournamentModel
from trident_validation.models.static_tournament import STATIC_MODEL_IDS
from trident_validation.synthetic import CORE_SYNTHETIC_FEATURES
from trident_validation.synthetic import formal_recovery
from trident_validation.synthetic.recovery import best_label_alignment


ROOT = Path(__file__).resolve().parents[1]


def _small_config() -> dict:
    return {
        "study": {"id": "formal_synthetic_recovery_v1"},
        "worlds": [
            "W0_general_performance",
            "W1_continuous_manifold",
            "W2_nonlinear_vigilance",
            "W3_three_profile_mixture",
            "W4_four_pace_mixture",
        ],
        "models": list(STATIC_MODEL_IDS),
        "baseline": {
            "replicates_per_world": 1,
            "n_datasets": 3,
            "participants_per_dataset": 6,
            "sessions_per_participant": 1,
            "min_windows_per_session": 1,
            "max_windows_per_session": 1,
            "observation_noise_scale": 1.0,
            "source_shift_scale": 1.0,
            "technical_missingness_rate": 0.01,
            "latent_separation_scale": 1.0,
        },
        "split": {
            "policy": "participant_isolated",
            "train_fraction": 0.75,
            "test_fraction": 0.25,
        },
        "selection": {
            "primary_metric": "heldout_log_density_mean_per_window",
            "practical_equivalence_margin": 0.01,
            "paired_ci_z": 1.96,
            "tie_resolution": "simplest_model_not_meaningfully_distinguishable_from_numerical_best",
        },
        "stress": {
            "replicates_per_world_per_level": 0,
            "one_factor_at_a_time": True,
            "participant_count_per_dataset": [6],
            "observation_noise_multiplier": [1.0],
            "missingness": [0.01],
            "source_dataset_shift_multiplier": [1.0],
            "latent_profile_separation_multiplier": [1.0],
            "windows_per_session_condition": {
                "baseline": {
                    "min_windows_per_session": 1,
                    "max_windows_per_session": 1,
                }
            },
        },
        "seeds": {"master_seed": 20260807},
        "outputs": {
            "compressed_replicate_audit": "reports/generated/formal_synthetic_recovery_v1_replicates.csv.gz",
            "manifest": "manifests/formal_synthetic_recovery_v1.json",
            "report": "reports/formal_synthetic_recovery_v1.md",
        },
    }


def test_formal_seed_schedule_is_deterministic_and_has_child_seeds():
    config = _small_config()

    first = formal_recovery.build_formal_seed_schedule(config)
    second = formal_recovery.build_formal_seed_schedule(config)

    assert first == second
    assert len(first) == 5
    assert len({task.run_id for task in first}) == 5
    assert all(isinstance(task.dataset_seed, int) for task in first)
    assert first[0].dataset_seed != first[0].split_seed
    assert first[0].split_seed != first[0].model_seed


def test_formal_task_slicing_preserves_schedule_order():
    config = _small_config()
    tasks = formal_recovery.build_formal_seed_schedule(config)

    shard = formal_recovery._slice_tasks(tasks, task_start=1, task_count=3)

    assert [task.task_index for task in shard] == [1, 2, 3]
    assert formal_recovery._slice_tasks(tasks, task_start=10, task_count=3) == []
    assert formal_recovery._slice_tasks(tasks, task_start=3, task_count=None) == tasks[3:]


def test_formal_clean_tree_enforcement_rejects_dirty_status(monkeypatch):
    monkeypatch.setattr(formal_recovery, "get_git_dirty", lambda repo_root=".": True)

    class Result:
        stdout = " M some_file.py\n"
        returncode = 0

    monkeypatch.setattr(formal_recovery.subprocess, "run", lambda *args, **kwargs: Result())

    with pytest.raises(formal_recovery.FormalRecoveryError):
        formal_recovery.ensure_clean_git_tree(ROOT)


def test_formal_aggregation_and_confidence_intervals_are_correct():
    run_summary = pd.DataFrame(
        [
            {
                "analysis_set": "baseline",
                "stress_family": "none",
                "stress_level": "baseline",
                "stress_value": "baseline",
                "true_world_id": "W0_general_performance",
                "selected_model_id": "M0_probabilistic_general_performance",
                "correct_model_selected": True,
                "normalisation_sensitive": False,
            },
            {
                "analysis_set": "baseline",
                "stress_family": "none",
                "stress_level": "baseline",
                "stress_value": "baseline",
                "true_world_id": "W0_general_performance",
                "selected_model_id": "M3_three_profile_mixture",
                "correct_model_selected": False,
                "normalisation_sensitive": True,
            },
            {
                "analysis_set": "baseline",
                "stress_family": "none",
                "stress_level": "baseline",
                "stress_value": "baseline",
                "true_world_id": "W1_continuous_manifold",
                "selected_model_id": "M4_four_pace_profile_mixture",
                "correct_model_selected": False,
                "normalisation_sensitive": False,
            },
        ]
    )

    matrix = formal_recovery.formal_recovery_matrix(run_summary)
    correct = formal_recovery.correct_model_recovery(run_summary)
    false_rate = formal_recovery.formal_false_discrete_rate(run_summary)

    assert len(matrix) == 25
    assert correct.loc[
        correct["true_world_id"] == "W0_general_performance",
        "aligned_model_recovery_rate",
    ].iloc[0] == 0.5
    assert correct.loc[
        correct["true_world_id"] == "W0_general_performance",
        "ci_low",
    ].iloc[0] < 0.5
    pooled = false_rate[false_rate["true_world_id"] == "W0_W1_W2_combined"].iloc[0]
    assert pooled["n_runs"] == 3
    assert pooled["n_discrete_selected"] == 2
    assert np.isclose(pooled["false_discrete_rate"], 2 / 3)
    assert formal_recovery.recovery_decision_band(0.70) == "strong_recovery"
    assert formal_recovery.false_discrete_decision_band(0.21) == "concerning"


def test_formal_preflight_reports_expected_scale_and_paths():
    config = _small_config()

    preflight = formal_recovery.preflight_formal_recovery(
        config,
        output_dir="reports/generated",
        checkpoint_dir="reports/generated/checkpoints/formal_synthetic_recovery_v1",
        workers=2,
    )

    assert preflight["total_synthetic_datasets"] == 5
    assert preflight["total_model_fits"] == 25
    assert preflight["available_workers"] == 2
    assert preflight["output_paths"]["synthetic_recovery_matrix"].endswith(
        "synthetic_recovery_matrix.csv"
    )
    assert preflight["checkpoint_path"].endswith("formal_synthetic_recovery_v1")
    assert preflight["resume_supported"] is True
    assert preflight["estimated_runtime"] == "not_available_without_machine_specific_benchmark"
    assert preflight["seed_schedule_hash"].startswith("sha256:")


def test_formal_checkpoint_resume_skips_completed_tasks(monkeypatch):
    config = _small_config()
    tasks = formal_recovery.build_formal_seed_schedule(config)[:2]
    checkpoint_dir = ROOT / "reports" / "generated" / "test_checkpoints" / uuid.uuid4().hex
    calls = []

    def fake_run(task, config):
        calls.append(task.run_id)
        return formal_recovery.FormalRecoveryResult(
            task_index=task.task_index,
            run_summary={
                "run_id": task.run_id,
                "task_index": task.task_index,
                "analysis_set": task.analysis_set,
                "stress_family": task.stress_family,
                "stress_level": task.stress_level,
                "stress_value": task.stress_value,
                "true_world_id": task.world_id,
                "selected_model_id": "M0_probabilistic_general_performance",
                "correct_model_selected": True,
                "normalisation_sensitive": False,
            },
            model_rows=[],
            participant_difference_rows=[],
            component_rows=[],
            split_audit={
                "run_id": task.run_id,
                "task_index": task.task_index,
                "participant_isolated": True,
            },
        )

    monkeypatch.setattr(formal_recovery, "run_formal_task", fake_run)

    first = formal_recovery._run_tasks(
        tasks,
        config=config,
        workers=1,
        checkpoint_dir=checkpoint_dir,
        resume=True,
        progress_interval_seconds=0,
    )
    second = formal_recovery._run_tasks(
        tasks,
        config=config,
        workers=1,
        checkpoint_dir=checkpoint_dir,
        resume=True,
        progress_interval_seconds=0,
    )

    assert [result.task_index for result in first] == [0, 1]
    assert [result.task_index for result in second] == [0, 1]
    assert calls == [tasks[0].run_id, tasks[1].run_id]
    assert len(list((checkpoint_dir / "complete").glob("*.json"))) == 2


def test_formal_benchmark_reports_projected_runtime(monkeypatch):
    config = _small_config()

    def fake_run_tasks(tasks, **kwargs):
        return []

    monkeypatch.setattr(formal_recovery, "_run_tasks", fake_run_tasks)

    result = formal_recovery.benchmark_formal_recovery(
        config,
        samples=2,
        workers=2,
        executor="serial",
    )

    assert result["benchmark_synthetic_datasets"] == 2
    assert result["benchmark_model_fits"] == 10
    assert result["projected_total_runtime"]["basis"] == "benchmark_seconds_per_synthetic_dataset"


def test_formal_task_does_not_expose_truth_columns_to_model_fitting(monkeypatch):
    seen_fit_columns = []

    class SentinelModel(TournamentModel):
        def __init__(self, model_id: str, score: float):
            self.model_id = model_id
            self.score = score
            self.random_state = 0

        def feature_requirements(self):
            return FeatureRequirements(tuple(CORE_SYNTHETIC_FEATURES))

        def fit(self, frame, y=None, groups=None):
            assert formal_recovery.strip_ground_truth_columns(frame).equals(frame)
            seen_fit_columns.append(tuple(frame.columns))
            return self

        def score_samples(self, frame):
            assert formal_recovery.strip_ground_truth_columns(frame).equals(frame)
            return pd.Series(self.score, index=frame.index)

        def predict_representation(self, frame):
            return pd.DataFrame(index=frame.index)

        def get_model_metadata(self):
            return {"model_id": self.model_id}

    fake_models = [
        SentinelModel("M0_probabilistic_general_performance", 0.00),
        SentinelModel("M1_continuous_control_manifold", 0.01),
        SentinelModel("M2_nonlinear_vigilance", 0.02),
        SentinelModel("M3_three_profile_mixture", 0.03),
        SentinelModel("M4_four_pace_profile_mixture", 0.04),
    ]
    monkeypatch.setattr(
        formal_recovery,
        "build_static_model_suite",
        lambda **kwargs: fake_models,
        raising=False,
    )
    monkeypatch.setattr(
        "trident_validation.synthetic.recovery.build_static_model_suite",
        lambda **kwargs: fake_models,
    )
    config = _small_config()
    task = formal_recovery.build_formal_seed_schedule(config)[0]

    result = formal_recovery.run_formal_task(task, config)

    assert seen_fit_columns
    assert all("synthetic_world_id" not in columns for columns in seen_fit_columns)
    assert result.split_audit["participant_isolated"]


def test_formal_component_recovery_is_permutation_invariant_and_neutral():
    truth = ["component_0", "component_1", "component_1", "component_0"]
    predicted = ["component_2", "component_0", "component_0", "component_2"]

    match = best_label_alignment(truth, predicted)

    assert match["matched_accuracy"] == 1.0
    assert set(match["mapping"].values()) == {"component_0", "component_1"}
    assert "PACE" not in ";".join(match["mapping"].values())
    assert "Trident" not in ";".join(match["mapping"].values())


def test_formal_serial_and_parallel_results_match_for_small_fixture():
    config = _small_config()
    tasks = formal_recovery.build_formal_seed_schedule(config)[:2]

    serial = formal_recovery.aggregate_formal_results(
        formal_recovery._run_tasks(tasks, config=config, workers=1)
    )
    parallel = formal_recovery.aggregate_formal_results(
        formal_recovery._run_tasks(tasks, config=config, workers=2)
    )

    pd.testing.assert_frame_equal(serial.run_summary, parallel.run_summary)
    pd.testing.assert_frame_equal(serial.model_audit, parallel.model_audit)
