from pathlib import Path
import json
import uuid

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
        world_id="W1_continuous_manifold",
        replicate_index=0,
        background=background,
        config={
            "seed": 20260809,
            "n_templates": 2,
            "participants_per_template": 4,
            "sessions_per_participant": 2,
            "windows_per_session": 3,
            "structural_signal_fraction": 0.35,
            "use_covariance": True,
            "session_order_context_trend": "off",
        },
    )

    assert run.dataset["synthetic_world_id"].eq("W1_continuous_manifold").all()
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
        world_id="W0_general_performance",
        replicate_index=0,
        background=background,
        config={
            "seed": 20260810,
            "n_templates": 2,
            "participants_per_template": 4,
            "sessions_per_participant": 2,
            "windows_per_session": 3,
            "structural_signal_fraction": 0.35,
            "use_covariance": True,
            "session_order_context_trend": "off",
        },
    )

    parameters = set(run.generator_audit["parameter"])
    assert {
        "source_task_mean",
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
            "empirical_background_contract_commit": "1e9c11e",
            "static_tournament_contract": "static_tournament_v2",
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
            "worlds": ["W0_general_performance"],
            "replicates_per_world": 1,
            "n_templates": 2,
            "participants_per_template": 4,
            "sessions_per_participant": 2,
            "windows_per_session": 3,
            "structural_signal_fraction": 0.35,
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
