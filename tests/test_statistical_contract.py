import numpy as np
import pandas as pd

from trident_validation.models import M0ProbabilisticPCAModel, TournamentModel, TrainingFeatureAdapter
from trident_validation.splits import assert_no_participant_overlap, participant_train_test_split
from trident_validation.synthetic import CORE_SYNTHETIC_FEATURES, make_synthetic_window_table


def test_probabilistic_m0_implements_formal_tournament_contract():
    frame = make_synthetic_window_table(seed=202)
    split = participant_train_test_split(frame, test_size=0.2, seed=20260807)
    train = frame.loc[list(split.train_indices)].copy()
    test = frame.loc[list(split.test_indices)].copy()

    model = M0ProbabilisticPCAModel(
        feature_columns=CORE_SYNTHETIC_FEATURES,
        random_state=20260807,
    ).fit(train)
    assert isinstance(model, TournamentModel)

    scores = model.score_samples(test)
    representation = model.predict_representation(test)
    holdout = model.score_holdout(test)
    metadata = model.get_model_metadata()

    assert len(scores) == len(test)
    assert np.isfinite(scores.dropna()).all()
    assert list(representation.columns) == ["M0_probabilistic_general_performance_factor_1"]
    assert holdout.primary_metric == "heldout_log_density_mean_per_window"
    assert holdout.primary_value == holdout.metrics["heldout_log_density_mean_per_window"]
    assert "heldout_reconstruction_mse" in holdout.diagnostics
    assert metadata["latent_labels"] == "neutral_numeric_factor"
    assert metadata["random_state"] == 20260807


def test_probabilistic_m0_imputation_and_scaling_are_training_fitted_only():
    frame = make_synthetic_window_table(seed=203)
    split = participant_train_test_split(frame, test_size=0.3, seed=13)
    assert_no_participant_overlap(frame, split)
    train = frame.loc[list(split.train_indices)].copy()
    test = frame.loc[list(split.test_indices)].copy()
    test_before = test.copy(deep=True)
    test.loc[:, "accuracy"] = 9999.0
    test.loc[:, "median_rt_ms"] = np.nan

    model = M0ProbabilisticPCAModel(
        feature_columns=CORE_SYNTHETIC_FEATURES,
        random_state=13,
    ).fit(train)
    metadata_before = model.get_model_metadata()
    scores = model.score_samples(test)
    metadata_after = model.get_model_metadata()

    assert np.isfinite(scores).all()
    assert metadata_before == metadata_after
    adapter_meta = metadata_after["imputation"]
    assert adapter_meta["fitted_on_n_rows"] == len(train)
    expected_participant_groups = (
        train[["source_dataset", "participant_id"]].astype(str).drop_duplicates().shape[0]
    )
    assert adapter_meta["fitted_on_n_participant_groups"] == expected_participant_groups
    assert adapter_meta["training_feature_means"]["accuracy"] != 9999.0
    assert test_before.loc[:, CORE_SYNTHETIC_FEATURES].equals(
        frame.loc[list(split.test_indices), CORE_SYNTHETIC_FEATURES]
    )


def test_model_adapter_preserves_canonical_nan_and_records_structural_missingness():
    frame = make_synthetic_window_table(seed=204)
    canonical = frame.copy(deep=True)
    adapter = TrainingFeatureAdapter(
        feature_columns=["accuracy", "lapse_rate"],
        min_observed_features_per_window=1,
    ).fit(frame)

    prepared = adapter.transform(frame)
    metadata = adapter.get_metadata()

    assert frame.equals(canonical)
    assert np.isnan(frame.loc[~frame["has_vigilance"], "lapse_rate"]).all()
    assert not np.isnan(prepared.values).any()
    assert metadata["training_structural_missing_counts"]["lapse_rate"] > 0
    assert metadata["imputation_strategy"] == "training_feature_mean"


def test_factor_fitting_uses_train_participants_not_extreme_holdout_participants():
    frame = make_synthetic_window_table(seed=205)
    split = participant_train_test_split(frame, test_size=0.25, seed=21)
    train = frame.loc[list(split.train_indices)].copy()
    test = frame.loc[list(split.test_indices)].copy()
    extreme_holdout = test.copy()
    extreme_holdout.loc[:, "accuracy"] = -1000.0
    extreme_holdout.loc[:, "throughput_proxy"] = 100000.0

    train_only_model = M0ProbabilisticPCAModel(
        feature_columns=CORE_SYNTHETIC_FEATURES,
        random_state=21,
    ).fit(train)
    leaky_reference_model = M0ProbabilisticPCAModel(
        feature_columns=CORE_SYNTHETIC_FEATURES,
        random_state=21,
    ).fit(pd.concat([train, extreme_holdout], ignore_index=True))

    train_only_meta = train_only_model.get_model_metadata()
    leaky_meta = leaky_reference_model.get_model_metadata()

    assert train_only_meta["imputation"]["fitted_on_n_rows"] == len(train)
    assert train_only_meta["imputation"]["training_feature_means"] != leaky_meta["imputation"][
        "training_feature_means"
    ]
    assert train_only_meta["loadings"] != leaky_meta["loadings"]
