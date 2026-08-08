import numpy as np
import pandas as pd

from trident_validation.models.base import FeatureRequirements, TournamentModel
from trident_validation.synthetic import (
    CORE_SYNTHETIC_FEATURES,
    make_static_synthetic_world,
)
from trident_validation.synthetic import recovery
from trident_validation.synthetic.recovery import (
    RecoverySimulationConfig,
    adjusted_rand_index,
    best_label_alignment,
    build_recovery_matrix,
    compute_false_discrete_rate,
    run_synthetic_model_recovery,
    select_preferred_model,
)


def test_synthetic_model_recovery_is_reproducible_and_participant_isolated():
    config = RecoverySimulationConfig(
        seed=1001,
        n_replicates=1,
        n_stress_replicates=0,
        participants_per_dataset=8,
        windows_per_session=2,
    )

    first = run_synthetic_model_recovery(config)
    second = run_synthetic_model_recovery(config)

    pd.testing.assert_frame_equal(
        first.synthetic_model_recovery,
        second.synthetic_model_recovery,
    )
    pd.testing.assert_frame_equal(first.run_summary, second.run_summary)
    assert first.split_audit["participant_isolated"].all()
    assert first.split_audit["n_participant_overlap"].eq(0).all()


def test_recovery_matrix_and_false_discrete_rate_aggregate_selected_models():
    run_summary = pd.DataFrame(
        [
            {
                "analysis_set": "baseline",
                "stress_family": "none",
                "stress_level": "baseline",
                "true_world_id": "W0_general_performance",
                "selected_model_id": "M0_probabilistic_general_performance",
                "correct_model_selected": True,
            },
            {
                "analysis_set": "baseline",
                "stress_family": "none",
                "stress_level": "baseline",
                "true_world_id": "W0_general_performance",
                "selected_model_id": "M3_three_profile_mixture",
                "correct_model_selected": False,
            },
            {
                "analysis_set": "baseline",
                "stress_family": "none",
                "stress_level": "baseline",
                "true_world_id": "W1_continuous_manifold",
                "selected_model_id": "M4_four_pace_profile_mixture",
                "correct_model_selected": False,
            },
        ]
    )

    matrix = build_recovery_matrix(run_summary)
    false_rate = compute_false_discrete_rate(run_summary)

    assert len(matrix) == 25
    w0_m3 = matrix[
        (matrix["true_world_id"] == "W0_general_performance")
        & (matrix["selected_model_id"] == "M3_three_profile_mixture")
    ].iloc[0]
    assert w0_m3["n_selected"] == 1
    assert w0_m3["selection_rate"] == 0.5
    combined = false_rate[
        (false_rate["analysis_set"] == "baseline")
        & (false_rate["true_world_id"] == "W0_W1_W2_combined")
    ].iloc[0]
    assert combined["n_runs"] == 3
    assert combined["n_discrete_selected"] == 2
    assert np.isclose(combined["false_discrete_rate"], 2 / 3)


def test_selection_keeps_simpler_model_when_top_score_is_not_distinguishable():
    model_scores = pd.DataFrame(
        [
            {
                "model_id": "M0_probabilistic_general_performance",
                "heldout_log_density_mean_per_window": 1.000,
                "heldout_log_density_participant_weighted": 1.000,
                "heldout_log_density_mean_per_observed_feature": 0.200,
            },
            {
                "model_id": "M1_continuous_control_manifold",
                "heldout_log_density_mean_per_window": 1.004,
                "heldout_log_density_participant_weighted": 1.004,
                "heldout_log_density_mean_per_observed_feature": 0.201,
            },
            {
                "model_id": "M2_nonlinear_vigilance",
                "heldout_log_density_mean_per_window": 0.900,
                "heldout_log_density_participant_weighted": 0.900,
                "heldout_log_density_mean_per_observed_feature": 0.180,
            },
        ]
    )
    participant_scores = pd.DataFrame(
        {
            "M0_probabilistic_general_performance": [1.000, 1.000, 1.000, 1.000],
            "M1_continuous_control_manifold": [1.004, 1.004, 1.004, 1.004],
            "M2_nonlinear_vigilance": [0.900, 0.900, 0.900, 0.900],
        }
    )

    selection = select_preferred_model(
        model_scores,
        participant_scores,
        practical_equivalence_margin=0.01,
    )

    assert selection.numerical_best_model_id == "M1_continuous_control_manifold"
    assert selection.selected_model_id == "M0_probabilistic_general_performance"
    assert selection.selection_reason == "simpler_model_not_meaningfully_distinguishable"


def test_neutral_label_matching_is_permutation_invariant():
    truth = ["component_0", "component_0", "component_1", "component_1", "component_2"]
    predicted = ["component_2", "component_2", "component_0", "component_0", "component_1"]

    match = best_label_alignment(truth, predicted)
    ari = adjusted_rand_index(truth, predicted)

    assert match["mapping"] == {
        "component_0": "component_1",
        "component_1": "component_2",
        "component_2": "component_0",
    }
    assert match["matched_accuracy"] == 1.0
    assert ari == 1.0
    assert "PACE" not in ";".join(match["mapping"])
    assert "Trident" not in ";".join(match["mapping"])


def test_recovery_fitting_and_selection_do_not_receive_ground_truth_columns(monkeypatch):
    seen_fit_columns = []

    class SentinelModel(TournamentModel):
        def __init__(self, model_id: str, score: float):
            self.model_id = model_id
            self.score = score
            self.random_state = 0

        def feature_requirements(self):
            return FeatureRequirements(tuple(CORE_SYNTHETIC_FEATURES))

        def fit(self, frame, y=None, groups=None):
            leaked = recovery.ground_truth_columns(frame)
            assert leaked == ()
            seen_fit_columns.append(tuple(frame.columns))
            return self

        def score_samples(self, frame):
            leaked = recovery.ground_truth_columns(frame)
            assert leaked == ()
            return pd.Series(self.score, index=frame.index)

        def predict_representation(self, frame):
            return pd.DataFrame(index=frame.index)

        def get_model_metadata(self):
            return {"model_id": self.model_id}

    fake_models = [
        SentinelModel("M0_probabilistic_general_performance", 0.0),
        SentinelModel("M1_continuous_control_manifold", 0.0),
        SentinelModel("M2_nonlinear_vigilance", 1.0),
        SentinelModel("M3_three_profile_mixture", 0.0),
        SentinelModel("M4_four_pace_profile_mixture", 0.0),
    ]
    monkeypatch.setattr(recovery, "build_static_model_suite", lambda **kwargs: fake_models)
    frame = make_static_synthetic_world(
        "W0_general_performance",
        seed=222,
        participants_per_dataset=8,
        min_windows_per_session=2,
        max_windows_per_session=2,
    )
    frame["synthetic_aligned_model_id"] = "M4_four_pace_profile_mixture"
    frame["known_ground_truth_label"] = "leak_if_seen"
    assert "known_ground_truth_label" in recovery.ground_truth_columns(frame)
    assert "known_ground_truth_label" not in recovery.strip_ground_truth_columns(frame).columns
    config = RecoverySimulationConfig(
        seed=2002,
        n_replicates=1,
        n_stress_replicates=0,
        participants_per_dataset=8,
        windows_per_session=2,
    )

    run = recovery._run_one_recovery_dataset(
        world_id="W0_general_performance",
        replicate_index=0,
        analysis_set="baseline",
        stress_family="none",
        stress_level="baseline",
        stress_value=1.0,
        world_kwargs={
            "n_datasets": 3,
            "participants_per_dataset": 8,
            "sessions_per_participant": 2,
            "min_windows_per_session": 2,
            "max_windows_per_session": 2,
            "observation_noise_scale": 1.0,
            "source_shift_scale": 1.0,
            "technical_missingness_rate": 0.01,
            "latent_separation_scale": 1.0,
        },
        config=config,
    )

    assert seen_fit_columns
    assert all("synthetic_world_id" not in columns for columns in seen_fit_columns)
    assert run["run_row"]["selected_model_id"] == "M2_nonlinear_vigilance"
