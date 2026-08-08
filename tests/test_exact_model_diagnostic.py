from pathlib import Path
import uuid

import pandas as pd

from trident_validation.models.adapters import TrainingFeatureAdapter
from trident_validation.models.m2_nonlinear import (
    M2LatentCurveEMModel,
    M2ProjectionSearchVigilanceModel,
)
from trident_validation.models.static_tournament import STATIC_MODEL_IDS
from trident_validation.synthetic import exact_model_diagnostic as exact
from trident_validation.synthetic.recovery import ground_truth_columns, strip_ground_truth_columns


ROOT = Path(__file__).resolve().parents[1]


def _small_config() -> dict:
    return {
        "study": {"id": "exact_model_recovery_v1"},
        "exact_worlds": list(exact.EXACT_MODEL_WORLD_IDS),
        "models": list(STATIC_MODEL_IDS),
        "baseline": {
            "replicates_per_world": 1,
            "n_datasets": 3,
            "participants_per_dataset": 8,
            "sessions_per_participant": 1,
            "min_windows_per_session": 1,
            "max_windows_per_session": 1,
            "technical_missingness_rate": 0.0,
            "exact_signal_scale": 1.0,
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
        "seeds": {"master_seed": 20260808},
        "outputs": {"directory": "reports/generated/exact_model_recovery_v1"},
    }


def test_exact_seed_schedule_is_deterministic():
    config = _small_config()

    first = exact.build_exact_seed_schedule(config)
    second = exact.build_exact_seed_schedule(config)

    assert first == second
    assert len(first) == 5
    assert len({task.run_id for task in first}) == 5
    assert [task.exact_world_id for task in first] == list(exact.EXACT_MODEL_WORLD_IDS)


def test_exact_world_frame_hides_truth_columns_cleanly():
    frame = exact.make_exact_model_world(
        "EW3_m3_exact",
        seed=123,
        n_datasets=3,
        participants_per_dataset=8,
        sessions_per_participant=1,
        min_windows_per_session=1,
        max_windows_per_session=1,
    )

    assert "synthetic_exact_world_id" in ground_truth_columns(frame)
    stripped = strip_ground_truth_columns(frame)
    assert "synthetic_exact_world_id" not in stripped.columns
    assert {"source_dataset", "participant_id", *exact.CORE_SYNTHETIC_FEATURES}.issubset(stripped.columns)


def test_exact_model_smoke_run_outputs_recovery_tables():
    config = _small_config()
    tasks = exact.build_exact_seed_schedule(config)[:2]
    results = [exact.run_exact_task(task, config) for task in tasks]
    outputs = exact._aggregate_results(results)

    assert set(outputs) == {
        "run_summary",
        "model_scores",
        "participant_score_differences",
        "recovery_matrix",
        "correct_model_recovery",
        "selection_reason_counts",
    }
    assert isinstance(outputs["run_summary"], pd.DataFrame)
    assert outputs["model_scores"].shape[0] == 2 * len(STATIC_MODEL_IDS)


def test_m2_self_check_world_has_oracle_scores():
    frame, oracle = exact.make_m2_self_check_world(
        seed=123,
        n_datasets=3,
        participants_per_dataset=8,
        sessions_per_participant=1,
        min_windows_per_session=1,
        max_windows_per_session=1,
    )

    adapter = TrainingFeatureAdapter(feature_columns=exact.CORE_SYNTHETIC_FEATURES).fit(
        strip_ground_truth_columns(frame)
    )
    scores, linear_direction = exact._oracle_m2_scores(frame, oracle, adapter=adapter)

    assert scores.shape[0] == frame.shape[0]
    assert pd.Series(scores).notna().all()
    assert linear_direction.shape[0] == len(exact.CORE_SYNTHETIC_FEATURES)
    assert "synthetic_exact_world_id" in ground_truth_columns(frame)


def test_projection_search_m2_fits_self_check_frame():
    frame, _ = exact.make_m2_self_check_world(
        seed=123,
        n_datasets=3,
        participants_per_dataset=8,
        sessions_per_participant=1,
        min_windows_per_session=1,
        max_windows_per_session=1,
    )
    model_frame = strip_ground_truth_columns(frame)
    model = M2ProjectionSearchVigilanceModel(
        feature_columns=exact.CORE_SYNTHETIC_FEATURES,
        random_state=123,
        quadrature_points=51,
        n_random_projection_candidates=8,
    ).fit(model_frame)

    scores = model.score_samples(model_frame)

    assert scores.notna().all()
    assert model.projection_search_candidates_evaluated_ >= len(exact.CORE_SYNTHETIC_FEATURES)


def test_latent_curve_em_m2_fits_self_check_frame():
    frame, _ = exact.make_m2_self_check_world(
        seed=123,
        n_datasets=3,
        participants_per_dataset=8,
        sessions_per_participant=1,
        min_windows_per_session=1,
        max_windows_per_session=1,
    )
    model_frame = strip_ground_truth_columns(frame)
    model = M2LatentCurveEMModel(
        feature_columns=exact.CORE_SYNTHETIC_FEATURES,
        random_state=123,
        quadrature_points=31,
        max_em_iter=2,
    ).fit(model_frame)

    scores = model.score_samples(model_frame)

    assert scores.notna().all()
    assert model.em_n_iter_ >= 1
    assert model.em_converged_ is not None
    assert model.em_final_likelihood_change_ is not None
    assert model.em_final_parameter_change_ is not None


def test_m2_self_check_smoke_writes_summary():
    output_dir = ROOT / "reports" / "generated" / "test_m2_self_check" / uuid.uuid4().hex

    result = exact.run_m2_self_check(
        seed=123,
        n_replicates=1,
        n_datasets=3,
        participants_per_dataset=8,
        sessions_per_participant=1,
        min_windows_per_session=1,
        max_windows_per_session=1,
        output_dir=output_dir,
    )

    assert result["study_id"] == "m2_self_check_v1"
    assert result["summary"]["interpretation"] in {
        "oracle_m2_wins_but_fitted_m2_loses",
        "m2_generator_not_distinct_from_m1_under_oracle_scoring",
        "m2_projection_estimation_failure",
        "fitted_m2_needs_higher_quadrature",
        "projection_search_m2_recovers_self_check_data",
        "latent_curve_em_m2_recovers_self_check_data",
        "fitted_m2_recovers_self_check_data",
        "ambiguous_m2_self_check",
    }
    assert (output_dir / "m2_self_check.csv").exists()
    assert (output_dir / "m2_self_check_summary.json").exists()


def test_ew2_em_diagnostic_smoke_writes_summary():
    output_dir = ROOT / "reports" / "generated" / "test_ew2_em_diagnostic" / uuid.uuid4().hex

    result = exact.run_ew2_em_diagnostic(
        config=_small_config(),
        output_dir=output_dir,
        n_replicates=1,
        workers=1,
        executor="serial",
        em_quadrature_points=31,
        em_max_iter=2,
    )

    assert result["study_id"] == "ew2_exact_with_m2_em_diagnostic_v1"
    assert result["n_runs"] == 1
    assert "decision" in result["summary"]
    assert (output_dir / "ew2_em_diagnostic_run_summary.csv").exists()
    assert (output_dir / "ew2_em_diagnostic_model_scores.csv").exists()
    assert (output_dir / "ew2_em_diagnostic_summary.json").exists()


def test_m2_em_convergence_diagnostic_smoke_writes_summary():
    output_dir = ROOT / "reports" / "generated" / "test_m2_em_convergence" / uuid.uuid4().hex

    result = exact.run_m2_em_convergence_diagnostic(
        config=_small_config(),
        output_dir=output_dir,
        n_replicates=1,
        workers=1,
        executor="serial",
        em_quadrature_points=31,
        max_iters=(2, 3),
    )

    assert result["study_id"] == "m2_em_convergence_diagnostic_v1"
    assert result["n_runs"] == 1
    assert result["max_iters"] == [2, 3]
    assert "decision" in result["summary"]
    assert (output_dir / "m2_em_convergence_detail.csv").exists()
    assert (output_dir / "m2_em_convergence_summary.csv").exists()
    assert (output_dir / "m2_em_convergence_summary.json").exists()


def test_ew0_m1_nesting_diagnostic_smoke_writes_summary():
    output_dir = ROOT / "reports" / "generated" / "test_ew0_m1_nesting" / uuid.uuid4().hex

    result = exact.run_ew0_m1_nesting_diagnostic(
        config=_small_config(),
        output_dir=output_dir,
        n_replicates=1,
        bootstrap_samples=2,
    )

    assert result["study_id"] == "ew0_m1_nesting_diagnostic_v1"
    assert result["n_runs"] == 1
    assert "decision" in result["summary"]
    assert (output_dir / "ew0_m1_nesting_detail.csv").exists()
    assert (output_dir / "ew0_m1_nesting_summary.json").exists()
