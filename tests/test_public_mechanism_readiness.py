from pathlib import Path

from trident_validation.mechanistic.public_readiness import run_public_mechanism_readiness


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/public_mechanism_readiness_v1.yaml"


def test_public_mechanism_readiness_is_aggregate_only_and_claim_bounded():
    outputs = run_public_mechanism_readiness(CONFIG_PATH, repo_root=ROOT)

    manifest = outputs.manifest
    assert manifest["status"] == "aggregate_support_preflight_only"
    assert manifest["formal_claims_allowed"] is False
    assert manifest["model_fitting_allowed"] is False
    assert manifest["real_transfer_outcomes_allowed"] is False
    assert manifest["trident_validation_claim_allowed"] is False


def test_public_mechanism_readiness_preserves_variable_order_and_boundaries():
    outputs = run_public_mechanism_readiness(CONFIG_PATH, repo_root=ROOT)
    support = outputs.support_matrix

    assert support["variable_id"].tolist() == [
        "K",
        "V",
        "C_signal",
        "A_evidence",
        "T_commit",
        "PC_calibration",
        "R_dynamic",
        "P_pace",
        "Y_behavior",
        "Transfer_external",
    ]
    assert support.loc[support["variable_id"] == "Transfer_external", "support_status"].iloc[0] == "forbidden"
    assert support.loc[support["variable_id"] == "P_pace", "support_status"].iloc[0] == "descriptive_only"
    assert support["model_fitting_allowed"].eq(False).all()
    assert support["real_transfer_outcomes_allowed"].eq(False).all()


def test_public_mechanism_readiness_records_current_support_limitations():
    outputs = run_public_mechanism_readiness(CONFIG_PATH, repo_root=ROOT)
    support = outputs.support_matrix.set_index("variable_id")

    assert support.loc["K", "support_status"] == "supported"
    assert support.loc["Y_behavior", "support_status"] == "supported"
    assert support.loc["V", "support_status"] == "partial"
    assert "SART" in support.loc["V", "paired_task_support"]
    assert support.loc["C_signal", "support_status"] == "partial"
    assert "conflict_cost" in support.loc["C_signal", "available_features"]
    assert support.loc["A_evidence", "support_status"] == "unsupported"
    assert support.loc["PC_calibration", "support_status"] == "unsupported"
    assert support.loc["R_dynamic", "support_status"] == "partial_temporal_not_regime"
