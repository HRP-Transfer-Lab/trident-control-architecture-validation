from trident_validation.models import M0GeneralPerformanceModel
from trident_validation.splits import participant_train_test_split
from trident_validation.synthetic import CORE_SYNTHETIC_FEATURES, make_synthetic_window_table


def test_m0_general_performance_fits_predicts_and_scores_synthetic_holdout():
    frame = make_synthetic_window_table(seed=77)
    split = participant_train_test_split(frame, test_size=0.2, seed=20260807)
    train = frame.loc[list(split.train_indices)]
    test = frame.loc[list(split.test_indices)]

    model = M0GeneralPerformanceModel(feature_columns=CORE_SYNTHETIC_FEATURES).fit(train)
    predictions = model.predict(test)
    score = model.score_holdout(test)
    metadata = model.get_model_metadata()

    assert list(predictions.columns) == ["M0_general_performance_score"]
    assert len(predictions) == len(test)
    assert score.model_id == "M0_general_performance"
    assert score.metrics["heldout_reconstruction_mse"] >= 0
    assert metadata["labels"] == "neutral_numeric_projection"


def test_m0_is_deterministic_for_same_training_data():
    frame = make_synthetic_window_table(seed=78)
    split = participant_train_test_split(frame, test_size=0.2, seed=4)
    train = frame.loc[list(split.train_indices)]
    test = frame.loc[list(split.test_indices)]

    model_a = M0GeneralPerformanceModel(feature_columns=CORE_SYNTHETIC_FEATURES).fit(train)
    model_b = M0GeneralPerformanceModel(feature_columns=CORE_SYNTHETIC_FEATURES).fit(train)

    assert model_a.predict(test).equals(model_b.predict(test))
    assert model_a.get_model_metadata() == model_b.get_model_metadata()

