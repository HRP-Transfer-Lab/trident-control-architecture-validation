import numpy as np
import pandas as pd
import pytest

from trident_validation.schema import SchemaValidationError, validate_window_schema
from trident_validation.synthetic import make_synthetic_window_table


def test_synthetic_window_table_passes_contract():
    frame = make_synthetic_window_table(seed=123)

    report = validate_window_schema(frame)

    assert report.n_sources == 3
    assert report.n_participants == 120
    assert report.n_rows >= 3 * 40 * 2 * 2
    assert report.structural_missing_cells > 0


def test_schema_rejects_duplicate_canonical_keys():
    frame = make_synthetic_window_table(seed=123)
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)

    with pytest.raises(SchemaValidationError, match="duplicate"):
        validate_window_schema(duplicated)


def test_schema_rejects_invalid_trial_counts():
    frame = make_synthetic_window_table(seed=123)
    frame.loc[0, "n_trials_valid"] = frame.loc[0, "n_trials_total"] + 1

    with pytest.raises(SchemaValidationError, match="n_trials_valid"):
        validate_window_schema(frame)


def test_structural_missingness_must_remain_missing_when_flag_false():
    frame = make_synthetic_window_table(seed=123)
    target_index = frame.index[~frame["has_vigilance"]][0]
    assert np.isnan(frame.loc[target_index, "lapse_rate"])

    frame.loc[target_index, "lapse_rate"] = 0.0

    with pytest.raises(SchemaValidationError, match="lapse_rate"):
        validate_window_schema(frame)

