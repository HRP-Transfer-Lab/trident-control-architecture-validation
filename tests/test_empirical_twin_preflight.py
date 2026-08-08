from pathlib import Path
import uuid

import pytest

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
