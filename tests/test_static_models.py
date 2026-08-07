import numpy as np
import pandas as pd

from trident_validation.models import (
    M1ContinuousManifoldModel,
    M2NonlinearVigilanceModel,
    M3ThreeProfileMixtureModel,
    M4FourProfileMixtureModel,
    TournamentModel,
    build_static_model_suite,
    results_to_frame,
    score_static_models_on_split,
)
from trident_validation.splits import participant_train_test_split
from trident_validation.synthetic import CORE_SYNTHETIC_FEATURES, make_static_synthetic_world


def test_m1_to_m4_implement_tournament_contract():
    frame = make_static_synthetic_world(
        "W1_continuous_manifold",
        seed=301,
        participants_per_dataset=12,
        min_windows_per_session=2,
        max_windows_per_session=2,
    )
    split = participant_train_test_split(frame, test_size=0.25, seed=1)
    train = frame.loc[list(split.train_indices)]
    test = frame.loc[list(split.test_indices)]
    models = [
        M1ContinuousManifoldModel(CORE_SYNTHETIC_FEATURES, random_state=1),
        M2NonlinearVigilanceModel(CORE_SYNTHETIC_FEATURES, random_state=2),
        M3ThreeProfileMixtureModel(CORE_SYNTHETIC_FEATURES, random_state=3),
        M4FourProfileMixtureModel(CORE_SYNTHETIC_FEATURES, random_state=4),
    ]

    for model in models:
        assert isinstance(model, TournamentModel)
        fitted = model.fit(train)
        scores = fitted.score_samples(test)
        representation = fitted.predict_representation(test)
        holdout = fitted.score_holdout(test)
        metadata = fitted.get_model_metadata()

        assert len(scores) == len(test)
        assert np.isfinite(scores.dropna()).all()
        assert len(representation) == len(test)
        assert holdout.primary_metric == "heldout_log_density_mean_per_window"
        assert np.isfinite(holdout.primary_value)
        assert metadata["primary_metric"] == "heldout_log_density_mean_per_window"
        assert metadata["random_state"] == model.random_state
        assert "imputation" in metadata


def test_mixture_probabilities_are_neutral_and_sum_to_one():
    frame = make_static_synthetic_world(
        "W4_four_pace_mixture",
        seed=302,
        participants_per_dataset=14,
        min_windows_per_session=2,
        max_windows_per_session=2,
    )
    split = participant_train_test_split(frame, test_size=0.25, seed=2)
    train = frame.loc[list(split.train_indices)]
    test = frame.loc[list(split.test_indices)]

    model = M4FourProfileMixtureModel(CORE_SYNTHETIC_FEATURES, random_state=9).fit(train)
    probabilities = model.predict_proba(test)
    metadata = model.get_model_metadata()

    assert probabilities.shape == (len(test), 4)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert metadata["component_labels"] == [
        "component_0",
        "component_1",
        "component_2",
        "component_3",
    ]
    assert set(metadata["component_means"]) == set(metadata["component_labels"])


def test_static_tournament_scores_all_registered_static_models_on_one_split():
    frame = make_static_synthetic_world(
        "W2_nonlinear_vigilance",
        seed=303,
        participants_per_dataset=12,
        min_windows_per_session=2,
        max_windows_per_session=2,
    )
    split = participant_train_test_split(frame, test_size=0.25, seed=3)

    results = score_static_models_on_split(
        frame,
        split,
        feature_columns=CORE_SYNTHETIC_FEATURES,
        random_state=20260807,
    )
    summary = results_to_frame(results)

    assert len(results) == 5
    assert set(summary["model_id"]) == {
        model.model_id
        for model in build_static_model_suite(
            feature_columns=CORE_SYNTHETIC_FEATURES,
            random_state=20260807,
        )
    }
    assert summary["primary_metric"].eq("heldout_log_density_mean_per_window").all()
    assert np.isfinite(summary["primary_value"]).all()


def test_static_models_do_not_mutate_canonical_holdout_data():
    frame = make_static_synthetic_world(
        "W3_three_profile_mixture",
        seed=304,
        participants_per_dataset=12,
        min_windows_per_session=2,
        max_windows_per_session=2,
    )
    split = participant_train_test_split(frame, test_size=0.25, seed=4)
    train = frame.loc[list(split.train_indices)]
    test = frame.loc[list(split.test_indices)].copy()
    canonical = test.copy(deep=True)

    model = M3ThreeProfileMixtureModel(CORE_SYNTHETIC_FEATURES, random_state=5).fit(train)
    _ = model.score_holdout(test)

    pd.testing.assert_frame_equal(test, canonical)

