from pathlib import Path
import json
import uuid

import numpy as np
import pandas as pd

from trident_validation.synthetic import empirical_twin_v1
from trident_validation.synthetic.recovery import ground_truth_columns, strip_ground_truth_columns


ROOT = Path(__file__).resolve().parents[1]


def test_empirical_twin_generation_preserves_truth_but_strips_model_inputs():
    background = empirical_twin_v1.load_background_spec(
        ROOT / "reports/generated/empirical_twin_preflight_v1_full_acdc",
        ROOT / "reports/generated/empirical_twin_preflight_v1_full_acdc_paired_session",
    )
    run = empirical_twin_v1.generate_empirical_twin_dataset(
        world_id="ETW1",
        replicate_index=0,
        background=background,
        config={
            "seed": 20260809,
            "n_templates": 2,
            "participants_per_template": 4,
            "sessions_per_participant": 2,
            "windows_per_session": 3,
            "use_covariance": True,
            "session_order_context_trend": "off",
        },
    )

    assert run.dataset["synthetic_world_id"].eq("ETW1").all()
    assert run.generation_summary["aligned_model_id"] == "M1_continuous_control_manifold"
    assert ground_truth_columns(run.dataset)
    assert not ground_truth_columns(strip_ground_truth_columns(run.dataset))
    assert run.dataset.groupby(["source_dataset", "participant_id"])["session_id"].nunique().min() == 2
    assert run.dataset.groupby(["source_dataset", "participant_id", "session_id"]).size().min() == 3
    assert "control_profile" not in run.dataset.columns
    assert "control_profile_probability" not in run.dataset.columns


def test_empirical_twin_generator_audit_contains_required_background_levels():
    background = empirical_twin_v1.load_background_spec(
        ROOT / "reports/generated/empirical_twin_preflight_v1_full_acdc",
        ROOT / "reports/generated/empirical_twin_preflight_v1_full_acdc_paired_session",
    )
    run = empirical_twin_v1.generate_empirical_twin_dataset(
        world_id="ETW0",
        replicate_index=0,
        background=background,
        config={
            "seed": 20260810,
            "n_templates": 2,
            "participants_per_template": 4,
            "sessions_per_participant": 2,
            "windows_per_session": 3,
            "use_covariance": True,
            "session_order_context_trend": "off",
        },
    )

    parameters = set(run.generator_audit["parameter"])
    assert {
        "source_task_shift",
        "between_person_variance",
        "session_within_person_variance",
        "window_within_session_variance",
        "lag1_autocorrelation",
        "missingness_rate",
        "trial_count_p50",
    }.issubset(parameters)
    session_rows = run.generator_audit[
        run.generator_audit["parameter"] == "session_within_person_variance"
    ]
    assert session_rows["target"].fillna(0).max() > 0
    assert session_rows["realised"].fillna(0).max() > 0


def test_empirical_twin_smoke_writes_outputs_without_claims():
    output_dir = (
        ROOT
        / "reports"
        / "generated"
        / "test_empirical_twin_v1"
        / uuid.uuid4().hex
    )
    result = empirical_twin_v1.run_empirical_twin_v1_smoke(
        output_dir=output_dir,
        run_tournament_smoke=False,
    )

    assert result["empirical_background_contract"] == "EMPIRICAL_BACKGROUND_CONTRACT_V1"
    assert result["formal_claims_allowed"] is False
    assert result["n_generated_runs"] == 5
    assert (output_dir / "empirical_twin_v1_generated_windows.csv.gz").exists()
    assert (output_dir / "empirical_twin_v1_generator_audit.csv").exists()
    manifest = pd.read_json(output_dir / "empirical_twin_v1_manifest.json", typ="series")
    assert manifest["session_order_context_trend"] == "off"
    assert manifest["static_tournament_contract_commit"] == "81d22f2"


def test_empirical_twin_audit_diagnostics_report_lag1_degeneracy_without_tournament():
    output_dir = (
        ROOT
        / "reports"
        / "generated"
        / "test_empirical_twin_v1_diagnostics"
        / uuid.uuid4().hex
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "study": {
            "id": "empirical_twin_v1_diagnostic_test",
            "stage": "M2.7_generator_smoke",
            "status": "smoke_only",
        },
        "contract": {
            "empirical_background_contract": "EMPIRICAL_BACKGROUND_CONTRACT_V1",
            "empirical_background_contract_commit": "1e9c11e030dc0706c8941dfc08489bb6f63cac5d",
            "static_tournament_contract": "static_tournament_v2",
            "static_tournament_contract_commit": "81d22f2",
            "session_order_context_trend": "off",
            "formal_claims_allowed": False,
        },
        "inputs": {
            "acdc_preflight_dir": str(
                ROOT / "reports/generated/empirical_twin_preflight_v1_full_acdc"
            ),
            "paired_preflight_dir": str(
                ROOT
                / "reports/generated/empirical_twin_preflight_v1_full_acdc_paired_session"
            ),
        },
        "generator": {
            "seed": 20260809,
            "worlds": ["ETW0"],
            "replicates_per_world": 1,
            "n_templates": 2,
            "participants_per_template": 4,
            "sessions_per_participant": 2,
            "windows_per_session": 3,
            "use_covariance": True,
            "paired_cross_task_session_covariance": "off",
        },
        "tournament_smoke": {"enabled": False},
        "outputs": {"directory": str(output_dir)},
    }
    config_path = output_dir / "diagnostic_test_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    result = empirical_twin_v1.run_empirical_twin_v1_audit_diagnostics(
        config_path=config_path,
        output_dir=output_dir,
    )

    assert result["model_outputs_read"] is False
    assert result["tournament_scoring_run"] is False
    lag1 = pd.read_csv(output_dir / "empirical_twin_v1_lag1_diagnostics.csv")
    assert lag1["three_window_degeneracy"].astype(bool).all()
    assert {
        "existing_per_sequence_pearson_mean",
        "proportion_sequences_length_3",
        "all_three_window_cell",
        "diagnostic_pooled_observed_adjacent_pair_corr",
        "diagnostic_pooled_window_residual_adjacent_pair_corr",
        "diagnostic_pooled_ar_component_adjacent_pair_corr",
    }.issubset(lag1.columns)
    variance = pd.read_csv(output_dir / "empirical_twin_v1_component_variance_diagnostics.csv")
    assert {
        "combined_observed_realised",
        "background_component_realised",
        "component_isolated_realised",
    }.issubset(variance.columns)
    classification = (
        output_dir / "empirical_twin_v1_audit_issue_classification.md"
    ).read_text(encoding="utf-8")
    assert "No tournament model-score outputs were read" in classification


def test_etw2_registry_is_m2_em_v1_aligned():
    registry = empirical_twin_v1.M2_7_EMPIRICAL_TWIN_WORLD_REGISTRY

    assert list(registry) == ["ETW0", "ETW1", "ETW2", "ETW3", "ETW4"]
    assert registry["ETW2"]["aligned_model_id"] == "M2_EM_v1"
    assert "M2 self-check" in registry["ETW2"]["structural_source"]


def test_background_seed_does_not_change_structural_truth():
    background = empirical_twin_v1.load_background_spec(
        ROOT / "reports/generated/empirical_twin_preflight_v1_full_acdc",
        ROOT / "reports/generated/empirical_twin_preflight_v1_full_acdc_paired_session",
    )
    config = {
        "seed": 20260809,
        "n_templates": 2,
        "participants_per_template": 4,
        "sessions_per_participant": 2,
        "windows_per_session": 3,
        "use_covariance": True,
        "session_order_context_trend": "off",
    }
    first = empirical_twin_v1.generate_empirical_twin_dataset(
        world_id="ETW2",
        replicate_index=0,
        background=background,
        config=config,
    )
    second = empirical_twin_v1.generate_empirical_twin_dataset(
        world_id="ETW2",
        replicate_index=1,
        background=background,
        config={**config, "seed": config["seed"]},
    )

    structural_cols = [f"synthetic_structural_{feature}" for feature in empirical_twin_v1.FEATURES]
    first_structural = first.dataset[structural_cols].reset_index(drop=True)
    second_structural = second.dataset[structural_cols].reset_index(drop=True)
    assert not first_structural.equals(second_structural)

    same_structural = empirical_twin_v1._make_m2_7_structural_world(
        "ETW2",
        seed=first.generation_summary["structural_seed"],
        n_datasets=2,
        participants_per_dataset=4,
        sessions_per_participant=2,
        min_windows_per_session=3,
        max_windows_per_session=3,
    )
    changed_background = empirical_twin_v1.generate_empirical_twin_dataset(
        world_id="ETW2",
        replicate_index=0,
        background=background,
        config={**config, "seed": config["seed"] + 1, "structural_seed_override": first.generation_summary["structural_seed"]},
    )
    assert same_structural.shape[0] == first.dataset.shape[0]
    assert first.dataset[structural_cols].reset_index(drop=True).equals(
        same_structural[[column.replace("synthetic_structural_", "") for column in structural_cols]]
        .rename(columns={column.replace("synthetic_structural_", ""): column for column in structural_cols})
        .reset_index(drop=True)
    )
    assert not first.dataset[list(empirical_twin_v1.FEATURES)].equals(
        changed_background.dataset[list(empirical_twin_v1.FEATURES)]
    )


def test_paired_session_median_rt_is_unsupported_not_borrowed():
    background = empirical_twin_v1.load_background_spec(
        ROOT / "reports/generated/empirical_twin_preflight_v1_full_acdc",
        ROOT / "reports/generated/empirical_twin_preflight_v1_full_acdc_paired_session",
    )
    run = empirical_twin_v1.generate_empirical_twin_dataset(
        world_id="ETW0",
        replicate_index=0,
        background=background,
        config={
            "seed": 20260809,
            "n_templates": 2,
            "participants_per_template": 4,
            "sessions_per_participant": 2,
            "windows_per_session": 3,
            "use_covariance": True,
            "session_order_context_trend": "off",
        },
    )

    median_support = run.support_audit[
        (run.support_audit["quantity"] == "session_within_person_variance")
        & (run.support_audit["feature"] == "median_rt_ms")
    ]
    assert not median_support.empty
    assert set(median_support["support_status"]) == {"unsupported"}
    assert set(median_support["fallback_source"]) == {"no_feature_alias_borrowing"}


def test_session_covariance_calibration_recovers_participant_centred_estimand():
    target = np.diag([2.0, 5.0])
    calibrated = empirical_twin_v1._calibrate_session_covariance_for_centring(
        target,
        sessions_per_participant=3,
    )
    rng = np.random.default_rng(123)
    draws = rng.multivariate_normal(np.zeros(2), calibrated, size=(20000, 3))
    centred = draws - draws.mean(axis=1, keepdims=True)
    realised = np.var(centred.reshape(-1, 2), axis=0, ddof=1)

    assert np.allclose(realised, np.diag(target), rtol=0.04)


def test_between_person_component_recovers_target_variance_at_large_n():
    target = np.diag([0.4, 1.5, 0.2, 0.08, 0.6])
    rng = np.random.default_rng(321)
    draws = rng.multivariate_normal(
        np.zeros(len(empirical_twin_v1.FEATURES)),
        target,
        size=20000,
    )
    realised = np.var(draws, axis=0, ddof=1)

    assert np.allclose(realised, np.diag(target), rtol=0.04)


def test_window_ar_calibration_recovers_centred_variance_without_double_counting_fatigue():
    target = np.diag([1.25, 0.75, 0.5, 0.25, 0.1])
    lag1 = {feature: 0.35 for feature in empirical_twin_v1.FEATURES}
    fatigue = {feature: 0.01 for feature in empirical_twin_v1.FEATURES}
    length = 12
    calibrated = empirical_twin_v1._calibrate_window_covariance_for_estimand(
        target,
        lag1=lag1,
        fatigue=fatigue,
        windows_per_session=length,
    )
    rng = np.random.default_rng(456)
    realised_rows = []
    for _ in range(12000):
        state = np.zeros(len(empirical_twin_v1.FEATURES))
        rows = []
        for window in range(length):
            state = 0.35 * state + rng.multivariate_normal(
                np.zeros(len(empirical_twin_v1.FEATURES)),
                calibrated,
            )
            rows.append(state + 0.01 * window)
        matrix = np.asarray(rows)
        centred = matrix - matrix.mean(axis=0, keepdims=True)
        realised_rows.append(np.var(centred, axis=0, ddof=1))
    realised = np.mean(realised_rows, axis=0)

    assert np.allclose(realised, np.diag(target), rtol=0.05)


def test_primary_config_rejects_cross_task_session_covariance():
    config = {
        "contract": {
            "empirical_background_contract": "EMPIRICAL_BACKGROUND_CONTRACT_V1",
            "static_tournament_contract": "static_tournament_v2",
            "session_order_context_trend": "off",
        },
        "generator": {"paired_cross_task_session_covariance": "on"},
    }

    try:
        empirical_twin_v1._validate_config(config)
    except ValueError as exc:
        assert "paired_cross_task_session_covariance" in str(exc)
    else:
        raise AssertionError("cross-task paired covariance must be off in primary mode")
