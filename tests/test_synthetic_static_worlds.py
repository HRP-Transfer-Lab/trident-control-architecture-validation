import numpy as np

from trident_validation.models import STATIC_MODEL_IDS, score_static_models_on_split
from trident_validation.schema import validate_window_schema
from trident_validation.splits import participant_train_test_split
from trident_validation.synthetic import (
    CORE_SYNTHETIC_FEATURES,
    STATIC_SYNTHETIC_WORLD_IDS,
    WORLD_MODEL_ALIGNMENT,
    make_all_static_synthetic_worlds,
    make_static_synthetic_world,
)


def test_static_synthetic_worlds_pass_canonical_schema_and_have_neutral_truth():
    for world_id in STATIC_SYNTHETIC_WORLD_IDS:
        frame = make_static_synthetic_world(
            world_id,
            seed=401,
            participants_per_dataset=10,
            min_windows_per_session=2,
            max_windows_per_session=2,
        )
        report = validate_window_schema(frame)

        assert report.n_sources == 3
        assert frame["synthetic_world_id"].nunique() == 1
        assert frame["synthetic_world_id"].iloc[0] == world_id
        assert frame["synthetic_aligned_model_id"].iloc[0] == WORLD_MODEL_ALIGNMENT[world_id]
        assert set(frame["synthetic_component_id"].dropna()).issubset(
            {"component_0", "component_1", "component_2", "component_3", "not_applicable"}
        )


def test_all_static_world_generation_is_reproducible():
    worlds_a = make_all_static_synthetic_worlds(
        seed=402,
        participants_per_dataset=8,
        min_windows_per_session=2,
        max_windows_per_session=2,
    )
    worlds_b = make_all_static_synthetic_worlds(
        seed=402,
        participants_per_dataset=8,
        min_windows_per_session=2,
        max_windows_per_session=2,
    )

    assert set(worlds_a) == set(STATIC_SYNTHETIC_WORLD_IDS)
    for world_id in STATIC_SYNTHETIC_WORLD_IDS:
        assert worlds_a[world_id].equals(worlds_b[world_id])


def test_w0_and_w1_truth_do_not_contain_discrete_components():
    for world_id in ("W0_general_performance", "W1_continuous_manifold", "W2_nonlinear_vigilance"):
        frame = make_static_synthetic_world(
            world_id,
            seed=403,
            participants_per_dataset=8,
            min_windows_per_session=2,
            max_windows_per_session=2,
        )

        assert set(frame["synthetic_component_id"]) == {"not_applicable"}
        assert np.isfinite(frame["synthetic_latent_1"]).all()


def test_w3_and_w4_truth_have_expected_neutral_component_counts():
    w3 = make_static_synthetic_world(
        "W3_three_profile_mixture",
        seed=404,
        participants_per_dataset=24,
        min_windows_per_session=2,
        max_windows_per_session=2,
    )
    w4 = make_static_synthetic_world(
        "W4_four_pace_mixture",
        seed=405,
        participants_per_dataset=24,
        min_windows_per_session=2,
        max_windows_per_session=2,
    )

    assert set(w3["synthetic_component_id"]) == {"component_0", "component_1", "component_2"}
    assert set(w4["synthetic_component_id"]) == {
        "component_0",
        "component_1",
        "component_2",
        "component_3",
    }


def test_every_static_candidate_scores_every_static_synthetic_world():
    for index, world_id in enumerate(STATIC_SYNTHETIC_WORLD_IDS):
        frame = make_static_synthetic_world(
            world_id,
            seed=500 + index,
            participants_per_dataset=8,
            min_windows_per_session=2,
            max_windows_per_session=2,
        )
        split = participant_train_test_split(frame, test_size=0.25, seed=600 + index)
        results = score_static_models_on_split(
            frame,
            split,
            feature_columns=CORE_SYNTHETIC_FEATURES,
            random_state=700 + index,
        )

        assert {result.model_id for result in results} == set(STATIC_MODEL_IDS)
        assert all(np.isfinite(result.primary_value) for result in results)
        assert frame["synthetic_aligned_model_id"].iloc[0] == WORLD_MODEL_ALIGNMENT[world_id]
