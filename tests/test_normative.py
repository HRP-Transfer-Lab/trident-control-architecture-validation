import numpy as np
import pandas as pd
import pytest

from trident_validation.normative import (
    N0RawNormativeModel,
    N1SimpleNormativeResidualizer,
    NormativeResidualizer,
    ProspectiveLeakageError,
)
from trident_validation.splits import participant_train_test_split
from trident_validation.synthetic import CORE_SYNTHETIC_FEATURES, make_synthetic_window_table


def test_normative_residualizer_fits_train_and_transforms_untouched_participants():
    frame = make_synthetic_window_table(seed=99)
    split = participant_train_test_split(frame, test_size=0.25, seed=1)
    train = frame.loc[list(split.train_indices)].copy()
    test = frame.loc[list(split.test_indices)].copy()

    residualizer = NormativeResidualizer(feature_columns=CORE_SYNTHETIC_FEATURES).fit(train)
    transformed = residualizer.transform(test)

    for feature in CORE_SYNTHETIC_FEATURES:
        assert f"{feature}_raw" in transformed
        assert f"{feature}_expected" in transformed
        assert f"{feature}_population_z" in transformed
        assert f"{feature}_deviation" in transformed
        assert f"{feature}_uncertainty" in transformed
    assert set(residualizer.training_participant_offsets_["accuracy"]).isdisjoint(
        set(
            test[["source_dataset", "participant_id"]]
            .astype(str)
            .agg(tuple, axis=1)
            .tolist()
        )
    )


def test_normative_residualizer_preserves_missing_feature_cells():
    frame = make_synthetic_window_table(seed=100)
    split = participant_train_test_split(frame, test_size=0.2, seed=2)
    train = frame.loc[list(split.train_indices)].copy()
    test = frame.loc[list(split.test_indices)].copy()
    missing_index = test.index[0]
    test.loc[missing_index, "median_rt_ms"] = np.nan

    transformed = NormativeResidualizer(feature_columns=["median_rt_ms"]).fit(train).transform(test)

    assert np.isnan(transformed.loc[missing_index, "median_rt_ms_raw"])
    assert np.isnan(transformed.loc[missing_index, "median_rt_ms_population_z"])
    assert np.isnan(transformed.loc[missing_index, "median_rt_ms_deviation"])
    assert not np.isnan(transformed.loc[missing_index, "median_rt_ms_expected"])


def test_personal_baseline_uses_prior_sessions_not_same_or_future_sessions():
    train = _minimal_window_table(
        participant_id="train_p",
        values=[10.0, 10.0, 10.0],
    )
    test = _minimal_window_table(
        participant_id="test_p",
        values=[10.0, 20.0, 999.0],
    )
    changed_future = _minimal_window_table(
        participant_id="test_p",
        values=[10.0, 20.0, -999.0],
    )

    residualizer = NormativeResidualizer(feature_columns=["accuracy"]).fit(train)
    transformed = residualizer.transform(test)
    transformed_changed = residualizer.transform(changed_future)

    first_session = transformed[transformed["session_id"] == "s01"]
    second_session = transformed[transformed["session_id"] == "s02"]
    assert (first_session["accuracy_prior_session_count"] == 0).all()
    assert (first_session["accuracy_deviation"] == 0.0).all()
    assert (second_session["accuracy_prior_session_count"] == 1).all()
    assert (second_session["accuracy_deviation"] == 10.0).all()
    assert transformed.loc[transformed["session_id"] != "s03", "accuracy_deviation"].tolist() == (
        transformed_changed.loc[transformed_changed["session_id"] != "s03", "accuracy_deviation"].tolist()
    )


def test_n0_and_n1_normative_interfaces_are_distinct_and_neutral():
    frame = make_synthetic_window_table(seed=101)
    n0 = N0RawNormativeModel(feature_columns=["accuracy"]).fit(frame)
    n1 = N1SimpleNormativeResidualizer(feature_columns=["accuracy"]).fit(frame)

    raw = n0.transform(frame.head(3))
    residualised = n1.transform(frame.head(3))

    assert n0.get_metadata()["normative_id"] == "N0_raw_no_normalisation"
    assert n1.get_metadata()["normative_id"] == "N1_simple_median_residualiser"
    assert raw["accuracy_deviation"].equals(raw["accuracy_raw"])
    assert "strict_prospective" in n1.get_metadata()["prospective_baseline_mode"]
    assert "accuracy_expected" in residualised.columns


def test_n1_rejects_non_prospective_baseline_mode():
    train = _minimal_window_table(participant_id="train_p", values=[10.0, 11.0])
    residualizer = NormativeResidualizer(feature_columns=["accuracy"]).fit(train)

    with pytest.raises(ProspectiveLeakageError, match="strict_prospective"):
        residualizer.transform(train, personal_baseline_mode="retrospective_full_participant")


def test_external_prior_with_current_or_future_session_fails():
    train = _minimal_window_table(participant_id="train_p", values=[10.0, 10.0, 10.0])
    target = _minimal_window_table(participant_id="test_p", values=[20.0]).copy()
    target["session_id"] = "s02"
    target["practice_or_session_index"] = 2
    future_prior = _minimal_window_table(participant_id="test_p", values=[999.0]).copy()
    future_prior["session_id"] = "s03"
    future_prior["practice_or_session_index"] = 3

    residualizer = NormativeResidualizer(feature_columns=["accuracy"]).fit(train)

    with pytest.raises(ProspectiveLeakageError, match="current or future"):
        residualizer.transform_with_prior_observations(target, future_prior)


def test_external_prior_from_past_session_is_allowed():
    train = _minimal_window_table(participant_id="train_p", values=[10.0, 10.0, 10.0])
    target = _minimal_window_table(participant_id="test_p", values=[20.0]).copy()
    target["session_id"] = "s02"
    target["practice_or_session_index"] = 2
    prior = _minimal_window_table(participant_id="test_p", values=[12.0]).copy()
    prior["session_id"] = "s01"
    prior["practice_or_session_index"] = 1

    residualizer = NormativeResidualizer(feature_columns=["accuracy"]).fit(train)
    transformed = residualizer.transform_with_prior_observations(target, prior)

    assert transformed["accuracy_prior_session_count"].tolist() == [1]


def _minimal_window_table(participant_id: str, values: list[float]) -> pd.DataFrame:
    rows = []
    for session_number, value in enumerate(values, start=1):
        rows.append(
            {
                "source_dataset": "source",
                "source_version": "v1",
                "participant_id": participant_id,
                "session_id": f"s{session_number:02d}",
                "task_id": "stroop",
                "block_id": "b01",
                "window_id": "w01",
                "window_start_trial": 1,
                "window_end_trial": 10,
                "n_trials_total": 10,
                "n_trials_valid": 10,
                "source_file_or_table": "synthetic",
                "source_commit_or_release": "synthetic",
                "source_hash_if_available": "sha256:" + "0" * 64,
                "preprocessing_version": "v1",
                "feature_version": "v1",
                "practice_or_session_index": session_number,
                "accuracy": value,
                "has_conflict_cost": True,
                "has_post_error": True,
                "has_vigilance": False,
                "has_switch_structure": False,
                "has_confidence": False,
                "has_change_point": False,
            }
        )
    return pd.DataFrame(rows)
