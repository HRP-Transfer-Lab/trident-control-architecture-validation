"""Locked HCP-YA transversal analysis scaffold.

This module builds a participant-free analysis plan for the HCP-YA transversal
K/C/V test. It intentionally does not fit models. Model fitting requires a
passed HCP support preflight and a later analysis-freeze commit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from trident_validation.config import ConfigValidationError, load_yaml_config
from trident_validation.provenance import get_git_commit, hash_file, hash_mapping


@dataclass(frozen=True)
class HCPTransversalAnalysisPlan:
    """Participant-free locked HCP analysis plan."""

    summary: dict[str, Any]
    domain_plan: pd.DataFrame
    model_plan: pd.DataFrame
    layer_specific_plan: pd.DataFrame
    report_markdown: str


def run_hcp_transversal_analysis_plan(
    config_path: str | Path = "config/hcp_ya_transversal_analysis_v1.yaml",
    *,
    repo_root: str | Path | None = None,
    preflight_summary_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    write_outputs: bool = True,
) -> HCPTransversalAnalysisPlan:
    """Build the locked HCP-YA analysis plan/status report."""

    config_file = Path(config_path)
    root = Path(repo_root) if repo_root is not None else config_file.resolve().parents[1]
    config = load_yaml_config(root / config_file if not config_file.is_absolute() else config_file)
    validate_hcp_transversal_analysis_config(config)

    preflight_path = (
        Path(preflight_summary_path)
        if preflight_summary_path is not None
        else _resolve_repo_path(root, config["analysis"]["current_preflight_summary_path"])
    )
    preflight = _load_preflight_summary(preflight_path)
    gate = _preflight_gate_status(config, preflight)
    domain_plan = _domain_plan(config)
    model_plan = _model_plan(config)
    layer_plan = _layer_specific_plan(config)
    summary = {
        "analysis_id": config["analysis"]["id"],
        "status": gate["status"],
        "question": config["analysis"]["question"],
        "model_fitting_enabled": False,
        "model_fitting_allowed_now": False,
        "requires_later_analysis_freeze_commit": True,
        "preflight_summary_path": str(preflight_path),
        "preflight_status": gate["preflight_status"],
        "preflight_support_passed": gate["preflight_support_passed"],
        "blocked_reason": gate["blocked_reason"],
        "formal_claims_allowed": False,
        "trident_validation_claim_allowed": False,
        "transfer_outcomes_allowed": False,
        "pace_ontology_claim_allowed": False,
        "dynamic_regime_allowed": False,
        "neural_criticality_claim_allowed": False,
        "cusp_claim_allowed": False,
        "predictive_calibration_claim_allowed": False,
        "t_commit_claim_allowed": False,
        "ordinary_participant_folds_allowed": False,
        "primary_metric": config["validation"]["primary_metric"],
        "config_hash": hash_file(root / config_file) if not config_file.is_absolute() else hash_file(config_file),
        "config_content_hash": hash_mapping(config),
        "git_commit": get_git_commit(root),
    }
    report = _render_report(summary, domain_plan, model_plan, layer_plan)
    if write_outputs:
        out_dir = Path(output_dir) if output_dir is not None else root / config["outputs"]["output_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_json(out_dir / "analysis_plan_summary.json", summary)
        domain_plan.to_csv(out_dir / "domain_plan.csv", index=False)
        model_plan.to_csv(out_dir / "model_plan.csv", index=False)
        layer_plan.to_csv(out_dir / "layer_specific_plan.csv", index=False)
        (out_dir / "analysis_plan_report.md").write_text(report, encoding="utf-8", newline="\n")
        report_path = root / config["outputs"]["plan_report_md"]
        report_path.write_text(report, encoding="utf-8", newline="\n")
    return HCPTransversalAnalysisPlan(summary, domain_plan, model_plan, layer_plan, report)


def validate_hcp_transversal_analysis_config(config: dict[str, Any]) -> None:
    """Validate locked analysis scaffold and anti-circularity rules."""

    analysis = _required_mapping(config, "analysis")
    if analysis.get("id") != "hcp_ya_transversal_analysis_v1":
        raise ConfigValidationError("analysis.id must be hcp_ya_transversal_analysis_v1")
    if analysis.get("status") != "locked_pending_passed_support_preflight":
        raise ConfigValidationError("analysis.status must be locked_pending_passed_support_preflight")
    for field in (
        "formal_claims_allowed",
        "trident_validation_claim_allowed",
        "transfer_outcomes_allowed",
        "pace_ontology_claim_allowed",
        "dynamic_regime_allowed",
        "neural_criticality_claim_allowed",
        "cusp_claim_allowed",
        "predictive_calibration_claim_allowed",
        "t_commit_claim_allowed",
        "model_fitting_enabled",
    ):
        if analysis.get(field) is not False:
            raise ConfigValidationError(f"analysis.{field} must be false")
    if analysis.get("requires_later_analysis_freeze_commit") is not True:
        raise ConfigValidationError("analysis must require a later analysis-freeze commit")

    family_policy = _required_mapping(config["inputs"], "family_structure_policy")
    if family_policy.get("ordinary_participant_folds_allowed") is not False:
        raise ConfigValidationError("ordinary participant folds are not allowed")
    if config["inputs"].get("participant_level_data_in_git_allowed") is not False:
        raise ConfigValidationError("participant-level data in Git must be false")
    if config["validation"].get("all_scaling_inside_training_folds") is not True:
        raise ConfigValidationError("all scaling must occur inside training folds")
    if config["validation"].get("outcome_specific_K_exclusion_required") is not True:
        raise ConfigValidationError("outcome-specific K exclusion is required")

    predictors = _required_mapping(config, "coordinate_sources")
    outcomes = _required_mapping(config, "outcome_domains")
    cv_columns = set(_as_list(predictors["C_signal"].get("columns"))) | set(_as_list(predictors["V"].get("columns")))
    for domain, spec in outcomes.items():
        anti = _required_mapping(spec, "anti_circularity")
        if anti.get("exclude_outcome_from_K") is not True:
            raise ConfigValidationError(f"{domain} must exclude outcome columns from K")
        if anti.get("forbid_overlap_with_C_or_V") is not True:
            raise ConfigValidationError(f"{domain} must forbid overlap with C/V")
        outcome_columns = set(_as_list(spec.get("columns"))) | set(_as_list(spec.get("secondary_columns")))
        overlap = outcome_columns.intersection(cv_columns)
        if overlap:
            raise ConfigValidationError(
                f"{domain} outcome overlaps with C_signal/V predictors: " + ", ".join(sorted(overlap))
            )

    layer = _required_mapping(config, "layer_specific_residual_tests")
    if layer.get("no_outcome_reuse_for_specific_factor") is not True:
        raise ConfigValidationError("layer-specific residual tests must forbid outcome reuse")
    for key in ("wm_specific", "reasoning_specific"):
        block = _required_mapping(layer, key)
        if block.get("status") != "blocked_until_independent_indicators_registered":
            raise ConfigValidationError(f"{key} must remain blocked until independent indicators are registered")


def _domain_plan(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    k_columns = set(_as_list(config["coordinate_sources"]["K"]["columns"]))
    for domain, spec in config["outcome_domains"].items():
        outcome_columns = set(_as_list(spec.get("columns"))) | set(_as_list(spec.get("secondary_columns")))
        rows.append(
            {
                "domain": domain,
                "primary": bool(spec.get("primary", False)),
                "outcome_columns": "|".join(sorted(outcome_columns)),
                "k_columns_after_exclusion": "|".join(sorted(k_columns.difference(outcome_columns))),
                "c_signal_columns": "|".join(_as_list(config["coordinate_sources"]["C_signal"]["columns"])),
                "v_columns": "|".join(_as_list(config["coordinate_sources"]["V"]["columns"])),
                "anti_circularity_checked": True,
            }
        )
    return pd.DataFrame(rows)


def _model_plan(config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for domain in config["outcome_domains"]:
        for index, model in enumerate(config["primary_model_sequence"]):
            rows.append(
                {
                    "domain": domain,
                    "model_order": index,
                    "model_id": model["id"],
                    "predictors": "|".join(_as_list(model.get("predictors"))),
                    "primary_metric": config["validation"]["primary_metric"],
                }
            )
    return pd.DataFrame(rows)


def _layer_specific_plan(config: dict[str, Any]) -> pd.DataFrame:
    layer = config["layer_specific_residual_tests"]
    return pd.DataFrame(
        [
            {
                "layer": "working_memory",
                "status": layer["wm_specific"]["status"],
                "candidate_indicators": "|".join(_as_list(layer["wm_specific"].get("candidate_indicators"))),
                "outcome_reuse_allowed": False,
            },
            {
                "layer": "reasoning",
                "status": layer["reasoning_specific"]["status"],
                "candidate_indicators": "|".join(_as_list(layer["reasoning_specific"].get("candidate_indicators"))),
                "outcome_reuse_allowed": False,
            },
        ]
    )


def _load_preflight_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _preflight_gate_status(config: dict[str, Any], preflight: dict[str, Any] | None) -> dict[str, Any]:
    required = config["analysis"]["required_preflight_status"]
    if preflight is None:
        return {
            "status": "blocked_no_preflight_summary",
            "preflight_status": None,
            "preflight_support_passed": False,
            "blocked_reason": "preflight summary is missing",
        }
    preflight_status = preflight.get("status")
    support_passed = bool(preflight.get("support_passed", False))
    if preflight_status == required and support_passed:
        return {
            "status": "ready_to_freeze_analysis_config",
            "preflight_status": preflight_status,
            "preflight_support_passed": True,
            "blocked_reason": "model fitting remains disabled until a later analysis-freeze commit",
        }
    return {
        "status": "blocked_preflight_not_passed",
        "preflight_status": preflight_status,
        "preflight_support_passed": support_passed,
        "blocked_reason": f"requires preflight status {required} with support_passed true",
    }


def _render_report(
    summary: dict[str, Any],
    domain_plan: pd.DataFrame,
    model_plan: pd.DataFrame,
    layer_plan: pd.DataFrame,
) -> str:
    lines = [
        "# M6 HCP-YA Transversal Analysis Plan",
        "",
        "**Status:** locked pending passed support preflight",
        "",
        "**Model fitting enabled:** false",
        "",
        "No Trident-G/APC/PACE validation claim, transfer claim, Predictive Calibration claim, T_commit claim, dynamic-regime claim, neural-criticality claim or cusp claim is made.",
        "",
        "## Question",
        "",
        summary["question"],
        "",
        "## Gate Status",
        "",
        f"- Status: `{summary['status']}`",
        f"- Preflight status: `{summary['preflight_status']}`",
        f"- Preflight support passed: {str(summary['preflight_support_passed']).lower()}",
        f"- Blocked reason: {summary['blocked_reason']}",
        "",
        "## Domain Plan",
        "",
        "| Domain | Primary | Outcome columns | K columns after exclusion |",
        "|---|---:|---|---|",
    ]
    for row in domain_plan.itertuples(index=False):
        lines.append(f"| {row.domain} | {str(row.primary).lower()} | {row.outcome_columns} | {row.k_columns_after_exclusion} |")
    lines.extend(
        [
            "",
            "## Model Sequence",
            "",
            "| Domain | Order | Model | Predictors |",
            "|---|---:|---|---|",
        ]
    )
    for row in model_plan.itertuples(index=False):
        lines.append(f"| {row.domain} | {row.model_order} | {row.model_id} | {row.predictors} |")
    lines.extend(
        [
            "",
            "## Layer-Specific Residual Gate",
            "",
            "| Layer | Status | Candidate indicators | Outcome reuse allowed |",
            "|---|---|---|---:|",
        ]
    )
    for row in layer_plan.itertuples(index=False):
        lines.append(
            f"| {row.layer} | {row.status} | {row.candidate_indicators or 'none'} | "
            f"{str(row.outcome_reuse_allowed).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Participant-level HCP data read by this plan: false",
            "- Ordinary participant folds allowed: false",
            "- Outcome columns reused to construct same-domain K/C/V: false",
            "- Layer-specific residual factors registered: false",
            "- NKI replication required for stronger transport claim: true",
        ]
    )
    return "\n".join(lines) + "\n"


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict) or not value:
        raise ConfigValidationError(f"{key} must be a non-empty mapping")
    return value


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _resolve_repo_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/hcp_ya_transversal_analysis_v1.yaml")
    parser.add_argument("--preflight-summary", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.plan_only:
        raise ConfigValidationError("HCP transversal analysis scaffold currently supports --plan-only only")
    result = run_hcp_transversal_analysis_plan(
        args.config,
        preflight_summary_path=args.preflight_summary,
        output_dir=args.output_dir,
    )
    print(json.dumps(result.summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
