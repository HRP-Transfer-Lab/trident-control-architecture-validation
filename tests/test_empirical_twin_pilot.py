from pathlib import Path
import json
import uuid

import pandas as pd
import pytest

from trident_validation.config import load_yaml_config
from trident_validation.models.static_tournament_v2 import STATIC_V2_MODEL_IDS
from trident_validation.synthetic import empirical_twin_pilot
from trident_validation.synthetic.empirical_twin_v1 import load_background_spec, _dataframe_hash


ROOT = Path(__file__).resolve().parents[1]


def _scratch_dir() -> Path:
    path = ROOT / "reports/generated/test_empirical_twin_pilot" / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def _pilot_config(base_path: Path) -> dict:
    config = load_yaml_config(ROOT / "config/empirical_twin_v1_pilot.yaml")
    config["pilot_design"]["schedule_csv"] = str(base_path / "pilot_schedule.csv")
    config["pilot_design"]["schedule_json"] = str(base_path / "pilot_schedule.json")
    config["outputs"]["directory"] = str(base_path / "pilot_outputs")
    return config


def test_pilot_template_eligibility_is_matched_stroop_flanker_only():
    background = load_background_spec(
        ROOT / "reports/generated/empirical_twin_preflight_v1_full_acdc",
        ROOT / "reports/generated/empirical_twin_preflight_v1_full_acdc_paired_session",
    )

    by_task = empirical_twin_pilot._eligible_templates_by_task(
        background.template_summary,
        ("Stroop", "Flanker"),
    )

    assert by_task["Stroop"].shape[0] == 34
    assert by_task["Flanker"].shape[0] == 12
    assert set(by_task) == {"Stroop", "Flanker"}
    assert "Simon" not in set(pd.concat(by_task.values())["task_id"])
    assert "SART" not in set(pd.concat(by_task.values())["task_id"])
    with pytest.raises(ValueError, match="no contract-eligible templates"):
        empirical_twin_pilot._eligible_templates_by_task(
            background.template_summary,
            ("SART",),
        )


def test_prepare_pilot_schedule_is_immutable_task_balanced_and_hashed():
    config = _pilot_config(_scratch_dir())
    output_dir = Path(config["outputs"]["directory"])

    schedule, schedule_hash = empirical_twin_pilot.prepare_pilot_schedule(
        config,
        config_path=ROOT / "config/empirical_twin_v1_pilot.yaml",
        output_dir=output_dir,
    )

    assert schedule.shape[0] == 50
    assert schedule["task_index"].nunique() == 50
    assert sorted(schedule["task_index"].tolist()) == list(range(50))
    assert set(schedule["world_id"]) == {"ETW0", "ETW1", "ETW2", "ETW3", "ETW4"}
    assert schedule.groupby("world_id").size().eq(10).all()
    assert schedule["empirical_background_contract_sha"].eq(
        "1e9c11e030dc0706c8941dfc08489bb6f63cac5d"
    ).all()
    assert schedule["static_model_contract_sha"].eq(
        "81d22f2a942afd1652c169935f058b1634922d30"
    ).all()
    assert schedule["generator_sha"].eq("9699427176961a3064e9db8da23a3b287a441b1b").all()

    replicate_pairs = schedule.drop_duplicates("replicate_index").sort_values("replicate_index")
    assert replicate_pairs.shape[0] == 10
    assert replicate_pairs["stroop_template_id"].nunique() == 10
    assert replicate_pairs["flanker_template_id"].nunique() == 10
    assert replicate_pairs["stroop_template_task_id"].eq("Stroop").all()
    assert replicate_pairs["flanker_template_task_id"].eq("Flanker").all()

    for _, group in schedule.groupby("replicate_index"):
        assert group["stroop_template_id"].nunique() == 1
        assert group["flanker_template_id"].nunique() == 1
        assert set(group["world_id"]) == {"ETW0", "ETW1", "ETW2", "ETW3", "ETW4"}
        task_sets = {
            tuple(record["task_id"] for record in json.loads(value))
            for value in group["template_records_json"]
        }
        assert task_sets == {("Stroop", "Flanker")}

    schedule_path = Path(config["pilot_design"]["schedule_csv"])
    manifest = json.loads((output_dir / "pilot_manifest.json").read_text(encoding="utf-8"))
    assert schedule_path.exists()
    assert Path(config["pilot_design"]["schedule_json"]).exists()
    assert schedule_hash == _dataframe_hash(pd.read_csv(schedule_path))
    assert manifest["schedule_hash"] == schedule_hash
    assert manifest["matched_support_intersection"] == ["Stroop", "Flanker"]


def test_checkpoint_checksum_validation_detects_tampering():
    path = _scratch_dir() / "checkpoint.json"
    schedule_hash = "sha256:test"
    payload = {
        "status": "complete",
        "unit_id": "ETW0_replicate_000",
        "schedule_hash": schedule_hash,
        "runtime": {"unit_runtime_seconds": 1.0},
    }

    empirical_twin_pilot._atomic_write_checkpoint(path, payload)
    assert empirical_twin_pilot._checkpoint_valid(path, schedule_hash)

    data = json.loads(path.read_text(encoding="utf-8"))
    data["status"] = "failed"
    path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")

    assert not empirical_twin_pilot._checkpoint_valid(path, schedule_hash)


def test_pilot_report_machinery_freezes_required_sections_from_mock_outputs():
    selections = _mock_selections()
    participant_scores = _mock_participant_scores()
    diagnostics = _mock_fit_diagnostics()
    runtime = _mock_runtime()
    generator_audit = _mock_generator_audit()

    contrasts = empirical_twin_pilot.paired_contrasts(participant_scores)
    world_selection = empirical_twin_pilot.world_by_model_selection_summary(selections)
    false_discrete = empirical_twin_pilot.false_discrete_pilot_summary(selections)
    etw0 = empirical_twin_pilot.etw0_m0_m1_diagnostics(participant_scores, diagnostics)
    etw2 = empirical_twin_pilot.etw2_m2_em_diagnostics(participant_scores, diagnostics)
    discrete = empirical_twin_pilot.discrete_contrast_summary(contrasts)
    failure_runtime = empirical_twin_pilot.failure_runtime_summary(diagnostics, runtime)
    background = empirical_twin_pilot.background_realism_summary(generator_audit)
    runtime_summary = empirical_twin_pilot.runtime_summary_table(runtime)

    assert {"numerical_best_wilson95_low", "selected_wilson95_high"}.issubset(world_selection.columns)
    assert world_selection["exploratory_n_per_world"].eq(10).all()
    assert false_discrete.loc[
        false_discrete["scope"] == "combined_continuous_ETW0_ETW2",
        "diagnostic_label",
    ].eq("pilot_false_discrete_selection_diagnostic_not_confirmatory_error_rate").all()
    assert "M1_minus_M0_participant_isolated_heldout_density" in set(etw0["metric"])
    assert "M1_second_first_eigenvalue_ratio" in set(etw0["metric"])
    assert "M2_EM_v1_minus_M1_participant_isolated_heldout_density" in set(etw2["metric"])
    assert "em_converged_rate" in set(etw2["metric"])
    assert set(empirical_twin_pilot.FROZEN_DISCRETE_CONTRASTS).issubset(
        set(zip(discrete["world_id"], discrete["contrast"]))
    )
    assert "fit_failures_by_world_model" in set(failure_runtime["summary"])
    assert {
        "source_task_shift",
        "between_person",
        "session",
        "window",
        "lag1",
        "missingness",
        "trial_count",
        "repeated_person_stability",
    }.issubset(set(background["parameter_family"]))

    report = empirical_twin_pilot.pilot_report(
        model_scores=_mock_model_scores(),
        selections=selections,
        contrasts=contrasts,
        diagnostics=diagnostics,
        runtime_summary=runtime_summary,
        background_summary=background,
        world_selection_summary=world_selection,
        false_discrete_summary=false_discrete,
        etw0_diagnostics=etw0,
        etw2_diagnostics=etw2,
        discrete_summary=discrete,
        failure_runtime=failure_runtime,
    )

    for section in (
        "EXPLORATORY / INFORMATIVE PILOT",
        "n = 10 replicates/world",
        "No confirmatory architecture-recovery claim",
        "No Trident-G/APC/PACE/neural-criticality/cusp validation claim",
        "World-By-Model Selection",
        "False-Discrete Pilot Diagnostic",
        "ETW0 M0/M1 Diagnostics",
        "ETW2 M2_EM_v1 Diagnostics",
        "ETW3/ETW4 Discrete Contrasts",
        "Failures And Runtime",
        "Background Realism",
    ):
        assert section in report


def _mock_selections() -> pd.DataFrame:
    rows = []
    selected = {
        "ETW0": ("M0_probabilistic_general_performance", "M3_three_profile_mixture"),
        "ETW1": ("M1_continuous_control_manifold", "M1_continuous_control_manifold"),
        "ETW2": ("M2_EM_v1", "M4_four_pace_profile_mixture"),
        "ETW3": ("M3_three_profile_mixture", "M1_continuous_control_manifold"),
        "ETW4": ("M4_four_pace_profile_mixture", "M4_four_pace_profile_mixture"),
    }
    numerical = {
        "ETW0": ("M1_continuous_control_manifold", "M3_three_profile_mixture"),
        "ETW1": ("M1_continuous_control_manifold", "M2_EM_v1"),
        "ETW2": ("M2_EM_v1", "M4_four_pace_profile_mixture"),
        "ETW3": ("M3_three_profile_mixture", "M1_continuous_control_manifold"),
        "ETW4": ("M4_four_pace_profile_mixture", "M3_three_profile_mixture"),
    }
    for world_id in selected:
        for replicate_index in range(2):
            rows.append(
                {
                    "unit_id": f"{world_id}_replicate_{replicate_index:03d}",
                    "world_id": world_id,
                    "replicate_index": replicate_index,
                    "selection_status": "complete",
                    "selected_model_id": selected[world_id][replicate_index],
                    "numerical_best_model_id": numerical[world_id][replicate_index],
                    "same_tier_ambiguous": replicate_index == 1,
                    "normalisation_sensitive": world_id in {"ETW0", "ETW2"} and replicate_index == 1,
                }
            )
    return pd.DataFrame(rows)


def _mock_participant_scores() -> pd.DataFrame:
    base_scores = {
        "ETW0": {
            "M0_probabilistic_general_performance": -2.00,
            "M1_continuous_control_manifold": -1.95,
            "M2_EM_v1": -1.98,
            "M3_three_profile_mixture": -2.05,
            "M4_four_pace_profile_mixture": -2.08,
        },
        "ETW2": {
            "M0_probabilistic_general_performance": -2.20,
            "M1_continuous_control_manifold": -2.05,
            "M2_EM_v1": -1.90,
            "M3_three_profile_mixture": -2.00,
            "M4_four_pace_profile_mixture": -1.98,
        },
        "ETW3": {
            "M1_continuous_control_manifold": -2.10,
            "M2_EM_v1": -2.06,
            "M3_three_profile_mixture": -1.93,
        },
        "ETW4": {
            "M1_continuous_control_manifold": -2.20,
            "M2_EM_v1": -2.15,
            "M3_three_profile_mixture": -1.97,
            "M4_four_pace_profile_mixture": -1.92,
        },
    }
    rows = []
    for world_id, model_scores in base_scores.items():
        for replicate_index in range(2):
            for participant_index in range(3):
                for model_id, value in model_scores.items():
                    rows.append(
                        {
                            "unit_id": f"{world_id}_replicate_{replicate_index:03d}",
                            "world_id": world_id,
                            "replicate_index": replicate_index,
                            "model_id": model_id,
                            "participant_key": f"p{participant_index}",
                            "heldout_log_density": value + replicate_index * 0.01 + participant_index * 0.001,
                        }
                    )
    return pd.DataFrame(rows)


def _mock_fit_diagnostics() -> pd.DataFrame:
    rows = []
    for world_id in ("ETW0", "ETW1", "ETW2", "ETW3", "ETW4"):
        for model_id in STATIC_V2_MODEL_IDS:
            for replicate_index in range(2):
                row = {
                    "unit_id": f"{world_id}_replicate_{replicate_index:03d}",
                    "world_id": world_id,
                    "replicate_index": replicate_index,
                    "model_id": model_id,
                    "fit_status": "success",
                    "runtime_seconds": 0.5 + replicate_index,
                }
                if world_id == "ETW0" and model_id == "M1_continuous_control_manifold":
                    row.update(
                        {
                            "second_first_eigenvalue_ratio": 0.22 + replicate_index * 0.01,
                            "second_loading_fraction": 0.31 + replicate_index * 0.01,
                        }
                    )
                if world_id == "ETW2" and model_id == "M2_EM_v1":
                    row.update(
                        {
                            "em_n_iter": 12 + replicate_index,
                            "em_converged": replicate_index == 0,
                            "em_iteration_cap_flag": replicate_index == 1,
                            "em_final_likelihood_change": 0.001 + replicate_index * 0.001,
                            "em_final_parameter_change": 0.002 + replicate_index * 0.001,
                            "residual_variance_mean": 0.18,
                            "residual_variance_min": 0.11,
                            "residual_variance_max": 0.25,
                            "runtime_seconds": 3.5 + replicate_index,
                        }
                    )
                if world_id == "ETW3" and model_id == "M4_four_pace_profile_mixture" and replicate_index == 1:
                    row["fit_status"] = "failed"
                rows.append(row)
    return pd.DataFrame(rows)


def _mock_runtime() -> pd.DataFrame:
    rows = []
    for world_id in ("ETW0", "ETW1", "ETW2", "ETW3", "ETW4"):
        for replicate_index in range(2):
            rows.append(
                {
                    "unit_id": f"{world_id}_replicate_{replicate_index:03d}",
                    "world_id": world_id,
                    "replicate_index": replicate_index,
                    "unit_runtime_seconds": 10.0 + replicate_index,
                    "generation_runtime_seconds": 3.0,
                    "tournament_runtime_seconds": 7.0 + replicate_index,
                    "status": "failed"
                    if world_id == "ETW3" and replicate_index == 1
                    else "complete",
                }
            )
    return pd.DataFrame(rows)


def _mock_generator_audit() -> pd.DataFrame:
    parameters = [
        "source_task_shift",
        "between_person_variance",
        "session_within_person_variance",
        "window_within_session_variance",
        "window_within_session_covariance",
        "lag1_autocorrelation",
        "missingness_rate",
        "trial_count_p50",
        "repeated_person_stability",
    ]
    return pd.DataFrame(
        [
            {
                "world_id": "ETW0",
                "replicate_index": 0,
                "source_dataset": "mock",
                "task_id": "Stroop",
                "parameter": parameter,
                "feature": "accuracy" if "covariance" not in parameter else "accuracy__median_rt_ms",
                "target": 0.10,
                "realised": 0.12,
                "absolute_delta": 0.02,
            }
            for parameter in parameters
        ]
    )


def _mock_model_scores() -> pd.DataFrame:
    rows = []
    for world_id in ("ETW0", "ETW1", "ETW2", "ETW3", "ETW4"):
        for model_id in STATIC_V2_MODEL_IDS:
            rows.append(
                {
                    "unit_id": f"{world_id}_replicate_000",
                    "world_id": world_id,
                    "replicate_index": 0,
                    "model_id": model_id,
                    "heldout_log_density_mean_per_window": -2.0,
                }
            )
    return pd.DataFrame(rows)
