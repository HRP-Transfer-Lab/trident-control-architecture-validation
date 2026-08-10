"""HCP-YA transversal K/C/V data-support preflight.

This module validates whether a local HCP-YA behavioural extract can support
the prospective transversal K/C/V strategy. It does not fit models.
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
class HCPTransversalPreflightResult:
    """Participant-free HCP-YA transversal preflight output."""

    summary: dict[str, Any]
    column_support: pd.DataFrame
    domain_support: pd.DataFrame
    split_support: dict[str, Any]
    report_markdown: str


def run_hcp_transversal_preflight(
    config_path: str | Path = "config/hcp_ya_transversal_v1.yaml",
    *,
    repo_root: str | Path | None = None,
    data_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    write_outputs: bool = True,
) -> HCPTransversalPreflightResult:
    """Run support-only preflight for the HCP-YA transversal strategy."""

    config_file = Path(config_path)
    root = Path(repo_root) if repo_root is not None else config_file.resolve().parents[1]
    config = load_yaml_config(root / config_file if not config_file.is_absolute() else config_file)
    validate_hcp_transversal_config(config)

    extract_path = Path(data_path) if data_path is not None else _resolve_repo_path(root, config["inputs"]["hcp_extract_path"])
    if not extract_path.exists():
        summary = _missing_data_summary(config, config_file, root, extract_path)
        column_support = pd.DataFrame()
        domain_support = pd.DataFrame()
        split_support = {
            "status": "data_file_missing",
            "family_isolated_cv_feasible": False,
            "unrelated_only_feasible": False,
            "ordinary_participant_folds_allowed": False,
        }
        report = _render_report(summary, column_support, domain_support, split_support)
        if write_outputs:
            _write_outputs(config, root, output_dir, summary, column_support, domain_support, split_support, report)
        return HCPTransversalPreflightResult(summary, column_support, domain_support, split_support, report)

    observed_hash = hash_file(extract_path)
    expected_hash = config["inputs"].get("hcp_extract_sha256")
    if expected_hash and observed_hash.lower() != str(expected_hash).lower():
        raise ConfigValidationError(
            f"HCP extract checksum mismatch: expected {expected_hash}, got {observed_hash}"
        )

    data = _read_table(extract_path)
    column_support = _column_support(data, config)
    domain_support = _domain_support(data, config)
    split_support = _split_support(data, config)
    support_passed = bool(
        column_support["available"].all()
        and domain_support["support_passed"].all()
        and (split_support["family_isolated_cv_feasible"] or split_support["unrelated_only_feasible"])
        and not split_support["ordinary_participant_folds_allowed"]
    )
    summary = {
        "analysis_id": config["analysis"]["id"],
        "status": "data_support_passed_ready_to_freeze_analysis" if support_passed else "data_support_incomplete",
        "question": config["analysis"]["question"],
        "model_fitting_allowed": False,
        "formal_claims_allowed": False,
        "trident_validation_claim_allowed": False,
        "transfer_outcomes_allowed": False,
        "pace_ontology_claim_allowed": False,
        "dynamic_regime_allowed": False,
        "neural_criticality_claim_allowed": False,
        "cusp_claim_allowed": False,
        "predictive_calibration_claim_allowed": False,
        "t_commit_claim_allowed": False,
        "participant_level_data_in_git_allowed": False,
        "input_path": str(extract_path),
        "input_hash": observed_hash,
        "input_rows": int(len(data)),
        "input_columns": int(len(data.columns)),
        "support_passed": support_passed,
        "config_hash": hash_file(root / config_file) if not config_file.is_absolute() else hash_file(config_file),
        "config_content_hash": hash_mapping(config),
        "git_commit": get_git_commit(root),
    }
    report = _render_report(summary, column_support, domain_support, split_support)
    if write_outputs:
        _write_outputs(config, root, output_dir, summary, column_support, domain_support, split_support, report)
    return HCPTransversalPreflightResult(summary, column_support, domain_support, split_support, report)


def validate_hcp_transversal_config(config: dict[str, Any]) -> None:
    """Validate claim boundaries and anti-circularity rules."""

    analysis = _required_mapping(config, "analysis")
    if analysis.get("id") != "hcp_ya_transversal_v1":
        raise ConfigValidationError("analysis.id must be hcp_ya_transversal_v1")
    if analysis.get("status") != "data_support_preflight_protocol":
        raise ConfigValidationError("analysis.status must be data_support_preflight_protocol")
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
    ):
        if analysis.get(field) is not False:
            raise ConfigValidationError(f"analysis.{field} must be false")

    inputs = _required_mapping(config, "inputs")
    if inputs.get("participant_level_data_in_git_allowed") is not False:
        raise ConfigValidationError("participant-level data must not be allowed in Git")
    family_policy = _required_mapping(inputs, "family_structure_policy")
    if family_policy.get("ordinary_participant_folds_allowed") is not False:
        raise ConfigValidationError("ordinary participant folds are not allowed for HCP-YA primary testing")

    predictors = _required_mapping(config, "predictor_sources")
    outcomes = _required_mapping(config, "outcome_domains")
    predictor_columns = {
        variable: set(_as_list(spec.get("columns")))
        for variable, spec in predictors.items()
    }
    for domain, spec in outcomes.items():
        outcome_columns = set(_as_list(spec.get("columns"))) | set(_as_list(spec.get("secondary_columns")))
        anti = _required_mapping(spec, "anti_circularity")
        if anti.get("exclude_from_K_when_target") is not True:
            raise ConfigValidationError(f"{domain} must exclude outcome columns from K when target")
        if anti.get("must_not_overlap_C_or_V") is not True:
            raise ConfigValidationError(f"{domain} must forbid C/V outcome overlap")
        overlap_cv = outcome_columns.intersection(predictor_columns.get("C_signal", set()) | predictor_columns.get("V", set()))
        if overlap_cv:
            raise ConfigValidationError(
                f"{domain} outcome overlaps with C_signal/V predictors: " + ", ".join(sorted(overlap_cv))
            )
    decision = _required_mapping(config, "decision_boundary")
    if decision.get("model_fitting_allowed_by_this_config") is not False:
        raise ConfigValidationError("this config must not authorise model fitting")


def _column_support(data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    roles: list[dict[str, Any]] = []
    participant_column = config["inputs"]["participant_id_column"]
    roles.append(_column_row("identity", "participant_id", participant_column, data))
    family_column = config["inputs"].get("family_id_column")
    if family_column:
        roles.append(_column_row("identity", "family_id", family_column, data))
    unrelated_column = config["inputs"].get("unrelated_indicator_column")
    if unrelated_column:
        roles.append(_column_row("identity", "unrelated_indicator", unrelated_column, data))

    for variable, spec in config["predictor_sources"].items():
        for column in _as_list(spec.get("columns")):
            roles.append(_column_row("predictor", variable, column, data))
    for domain, spec in config["outcome_domains"].items():
        for column in _as_list(spec.get("columns")):
            roles.append(_column_row("outcome", domain, column, data))
        for column in _as_list(spec.get("secondary_columns")):
            roles.append(_column_row("secondary_outcome", domain, column, data))
    return pd.DataFrame(roles)


def _domain_support(data: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    participant_col = config["inputs"]["participant_id_column"]
    thresholds = config["support_thresholds"]
    for domain, spec in config["outcome_domains"].items():
        outcome_columns = _as_list(spec.get("columns"))
        k_columns = _domain_k_columns(config, domain)
        required_columns = (
            [participant_col]
            + k_columns
            + _as_list(config["predictor_sources"]["C_signal"]["columns"])
            + _as_list(config["predictor_sources"]["V"]["columns"])
            + outcome_columns
        )
        available = [column for column in required_columns if column in data.columns]
        complete = data.dropna(subset=available) if len(available) == len(required_columns) else data.iloc[0:0]
        rows.append(
            {
                "domain": domain,
                "primary": bool(spec.get("primary", False)),
                "outcome_columns": "|".join(outcome_columns),
                "k_columns_after_exclusion": "|".join(k_columns),
                "required_columns": "|".join(required_columns),
                "missing_columns": "|".join(sorted(set(required_columns).difference(data.columns))),
                "complete_participants": int(complete[participant_col].nunique()) if participant_col in complete else 0,
                "complete_rows": int(len(complete)),
                "support_passed": bool(
                    len(available) == len(required_columns)
                    and complete[participant_col].nunique()
                    >= int(thresholds["minimum_complete_participants_per_domain"])
                ),
            }
        )
    return pd.DataFrame(rows)


def _split_support(data: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    inputs = config["inputs"]
    participant_col = inputs["participant_id_column"]
    family_col = inputs.get("family_id_column")
    unrelated_col = inputs.get("unrelated_indicator_column")
    thresholds = config["support_thresholds"]
    n_folds = int(config["validation"]["n_folds"])
    n_participants = int(data[participant_col].nunique()) if participant_col in data else 0
    family_available = bool(family_col and family_col in data.columns)
    n_families = int(data[family_col].nunique()) if family_available else 0
    family_feasible = bool(
        family_available
        and n_families >= n_folds * int(thresholds["minimum_families_per_fold"])
    )
    unrelated_available = bool(unrelated_col and unrelated_col in data.columns)
    unrelated_count = int(data.loc[data[unrelated_col].astype(bool), participant_col].nunique()) if unrelated_available and participant_col in data else 0
    unrelated_feasible = bool(
        unrelated_available
        and unrelated_count >= int(thresholds["minimum_unrelated_participants"])
    )
    return {
        "participant_column_available": participant_col in data.columns,
        "n_participants": n_participants,
        "family_id_column": family_col,
        "family_id_available": family_available,
        "n_families": n_families,
        "family_isolated_cv_feasible": family_feasible,
        "unrelated_indicator_column": unrelated_col,
        "unrelated_indicator_available": unrelated_available,
        "n_unrelated_participants": unrelated_count,
        "unrelated_only_feasible": unrelated_feasible,
        "ordinary_participant_folds_allowed": False,
    }


def _domain_k_columns(config: dict[str, Any], domain: str) -> list[str]:
    k_columns = set(_as_list(config["predictor_sources"]["K"]["columns"]))
    outcome_spec = config["outcome_domains"][domain]
    outcome_columns = set(_as_list(outcome_spec.get("columns"))) | set(_as_list(outcome_spec.get("secondary_columns")))
    return sorted(k_columns.difference(outcome_columns))


def _missing_data_summary(
    config: dict[str, Any],
    config_file: Path,
    root: Path,
    extract_path: Path,
) -> dict[str, Any]:
    return {
        "analysis_id": config["analysis"]["id"],
        "status": "data_file_missing",
        "question": config["analysis"]["question"],
        "model_fitting_allowed": False,
        "formal_claims_allowed": False,
        "trident_validation_claim_allowed": False,
        "input_path": str(extract_path),
        "support_passed": False,
        "config_hash": hash_file(root / config_file) if not config_file.is_absolute() else hash_file(config_file),
        "config_content_hash": hash_mapping(config),
        "git_commit": get_git_commit(root),
    }


def _render_report(
    summary: dict[str, Any],
    column_support: pd.DataFrame,
    domain_support: pd.DataFrame,
    split_support: dict[str, Any],
) -> str:
    lines = [
        "# M6 HCP-YA Transversal K/C/V Preflight",
        "",
        "**Status:** data-support preflight only",
        "",
        "**Model fitting allowed:** false",
        "",
        "No Trident-G/APC/PACE validation claim, transfer claim, Predictive Calibration claim, T_commit claim, dynamic-regime claim, neural-criticality claim or cusp claim is made.",
        "",
        "## Question",
        "",
        "Do lower-level K/C/V coordinates transport across attention/control, working memory and reasoning?",
        "",
        "## Summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Support passed: {str(summary['support_passed']).lower()}",
        f"- Input path: `{summary['input_path']}`",
    ]
    if "input_hash" in summary:
        lines.append(f"- Input checksum: `{summary['input_hash']}`")
    lines.extend(
        [
            "",
            "## Split Safeguard",
            "",
            f"- Family-isolated CV feasible: {str(split_support.get('family_isolated_cv_feasible', False)).lower()}",
            f"- Unrelated-only feasible: {str(split_support.get('unrelated_only_feasible', False)).lower()}",
            f"- Ordinary participant folds allowed: {str(split_support.get('ordinary_participant_folds_allowed', False)).lower()}",
            "",
        ]
    )
    if not column_support.empty:
        lines.extend(
            [
                "## Column Support",
                "",
                "| Role | Variable/domain | Column | Available | Nonmissing |",
                "|---|---|---|---:|---:|",
            ]
        )
        for row in column_support.itertuples(index=False):
            lines.append(f"| {row.role} | {row.variable_or_domain} | {row.column} | {str(row.available).lower()} | {row.nonmissing} |")
        lines.append("")
    if not domain_support.empty:
        lines.extend(
            [
                "## Domain Support",
                "",
                "| Domain | Primary | Complete participants | Missing columns | Support passed |",
                "|---|---:|---:|---|---:|",
            ]
        )
        for row in domain_support.itertuples(index=False):
            lines.append(
                f"| {row.domain} | {str(row.primary).lower()} | {row.complete_participants} | "
                f"{row.missing_columns or 'none'} | {str(row.support_passed).lower()} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Boundary",
            "",
            "- Participant-level HCP data in Git: false",
            "- Outcome columns reused to construct same-domain K/C/V: false by config validation",
            "- Model outcomes interpreted: false",
            "- Stage 1-3 reports changed: false",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_outputs(
    config: dict[str, Any],
    root: Path,
    output_dir: str | Path | None,
    summary: dict[str, Any],
    column_support: pd.DataFrame,
    domain_support: pd.DataFrame,
    split_support: dict[str, Any],
    report: str,
) -> None:
    out_dir = Path(output_dir) if output_dir is not None else root / config["outputs"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "preflight_summary.json", summary)
    _write_json(out_dir / "split_support.json", split_support)
    column_support.to_csv(out_dir / "column_support.csv", index=False)
    domain_support.to_csv(out_dir / "domain_support.csv", index=False)
    (out_dir / "preflight_report.md").write_text(report, encoding="utf-8", newline="\n")
    report_path = root / config["outputs"]["preflight_report_md"]
    report_path.write_text(report, encoding="utf-8", newline="\n")


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ConfigValidationError(f"unsupported HCP extract extension: {path.suffix}")


def _column_row(role: str, variable_or_domain: str, column: str, data: pd.DataFrame) -> dict[str, Any]:
    return {
        "role": role,
        "variable_or_domain": variable_or_domain,
        "column": column,
        "available": column in data.columns,
        "nonmissing": int(data[column].notna().sum()) if column in data.columns else 0,
    }


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
    parser.add_argument("--config", default="config/hcp_ya_transversal_v1.yaml")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.preflight_only:
        raise ConfigValidationError("HCP transversal runner currently supports --preflight-only only")
    result = run_hcp_transversal_preflight(
        args.config,
        data_path=args.data_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(result.summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
