from pathlib import Path
import uuid

import pandas as pd
import pytest

from trident_validation.mechanistic.identifiability_plan import load_mechanistic_identifiability_plan
from trident_validation.mechanistic.k_apc_smoke import (
    K_APC_GATE_ID,
    MECHANISTIC_OBSERVED_FEATURES,
    generate_k_apc_unit,
    load_k_apc_scoring_contract,
    run_k_apc_smoke,
    score_capacity_vs_apc_registered,
)
from trident_validation.splits import participant_train_test_split
from trident_validation.synthetic.recovery import strip_ground_truth_columns


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "config/mechanistic_identifiability_v1.yaml"
SCORING_PATH = ROOT / "config/mechanistic_k_apc_scoring_v1.yaml"


def _scratch_dir() -> Path:
    path = ROOT / "reports/generated/test_k_apc_smoke" / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def _schedule_row(family_id: str) -> pd.Series:
    plan = load_mechanistic_identifiability_plan(PLAN_PATH, repo_root=ROOT)
    match = plan.schedule[
        (plan.schedule["gate_id"] == K_APC_GATE_ID)
        & (plan.schedule["truth_family_id"] == family_id)
    ]
    return match.iloc[0]


def test_k_apc_generator_is_deterministic_for_same_schedule_row():
    row = _schedule_row("MECH1")

    first = generate_k_apc_unit(row)
    second = generate_k_apc_unit(row)

    pd.testing.assert_frame_equal(first, second)
    assert first.shape[0] == 48 * 2 * 16
    assert set(MECHANISTIC_OBSERVED_FEATURES).issubset(first.columns)


def test_mech0_has_capacity_truth_only_and_mech1_has_apc_truth():
    mech0 = generate_k_apc_unit(_schedule_row("MECH0"))
    mech1 = generate_k_apc_unit(_schedule_row("MECH1"))

    for column in (
        "synthetic_C_signal",
        "synthetic_A_evidence",
        "synthetic_T_commit",
        "synthetic_PC_calibration",
    ):
        assert mech0[column].isna().all()
        assert mech1[column].notna().all()
        assert mech1[column].var() > 0.5
    assert mech0["synthetic_K"].var() > 0.5
    assert mech1["synthetic_K"].var() > 0.5


def test_registered_scorer_rejects_truth_columns_and_accepts_stripped_data():
    frame = generate_k_apc_unit(_schedule_row("MECH1"))
    stripped = strip_ground_truth_columns(frame)
    split = participant_train_test_split(
        stripped,
        test_size=0.25,
        seed=123,
        participant_columns=("source_dataset", "participant_id"),
    )

    with pytest.raises(ValueError, match="ground-truth columns are not allowed"):
        score_capacity_vs_apc_registered(frame, split)

    scores = score_capacity_vs_apc_registered(stripped, split)
    assert scores["truth_columns_received"].eq(0).all()
    assert set(scores["model_id"]) == {
        "SCORE0_capacity_only_rank1",
        "SCORE1_static_continuous_apc_rank5",
    }
    assert scores["selection_status"].eq("engineering_smoke_selection_only").all()
    assert "complexity_adjusted_heldout_log_density_mean_per_row" in scores.columns


def test_registered_scorer_smoke_separates_k_only_from_apc_truth():
    selected = {}
    for family_id in ("MECH0", "MECH1"):
        row = _schedule_row(family_id)
        frame = strip_ground_truth_columns(generate_k_apc_unit(row))
        split = participant_train_test_split(
            frame,
            test_size=float(row["participant_holdout_fraction"]),
            seed=int(row["split_seed"]),
            participant_columns=("source_dataset", "participant_id"),
        )
        scores = score_capacity_vs_apc_registered(frame, split)
        selected[family_id] = scores["selected_model_id"].iloc[0]

    assert selected == {
        "MECH0": "SCORE0_capacity_only_rank1",
        "MECH1": "SCORE1_static_continuous_apc_rank5",
    }


def test_k_apc_scoring_contract_blocks_real_transfer_and_cusp_claims():
    contract = load_k_apc_scoring_contract(SCORING_PATH)

    assert contract.registry_id == "mechanistic_k_apc_scoring_v1"
    assert contract.primary_metric == "complexity_adjusted_heldout_log_density_mean_per_row"
    assert contract.feature_columns == MECHANISTIC_OBSERVED_FEATURES


def test_k_apc_smoke_runs_first_gate_only_and_writes_audits():
    outputs = run_k_apc_smoke(PLAN_PATH, output_dir=_scratch_dir(), scoring_config_path=SCORING_PATH)

    assert outputs.manifest["gate_id"] == K_APC_GATE_ID
    assert outputs.manifest["n_units"] == 4
    assert outputs.manifest["n_rows"] == 4 * 48 * 2 * 16
    assert outputs.manifest["scoring_contract_id"] == "mechanistic_k_apc_scoring_v1"
    assert outputs.manifest["model_winner_interpretation_allowed"] is False
    assert set(outputs.generation_audit["truth_family_id"]) == {"MECH0", "MECH1"}
    assert set(outputs.split_audit["gate_id"]) == {K_APC_GATE_ID}
    assert outputs.split_audit["participant_isolated"].eq(True).all()
    assert outputs.split_audit["truth_columns_in_scoring_frame"].eq(0).all()
    assert {
        "K",
        "C_signal",
        "A_evidence",
        "T_commit",
        "PC_calibration",
        "observed_feature_rank",
    }.issubset(set(outputs.generation_audit["variable_id"]))
    assert "No confirmatory Trident-G" in outputs.report
