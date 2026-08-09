"""Aggregate-only public mechanism readiness preflight.

This is the handoff from synthetic identifiability scaffolding toward real
public-data analysis. It reads aggregate preflight outputs only, maps them to
the frozen M3 variable registry, and reports support/limitations before any
mechanism model is fit.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from trident_validation.config import ConfigValidationError, load_yaml_config
from trident_validation.mechanistic.variable_registry import REQUIRED_VARIABLE_IDS, load_variable_registry
from trident_validation.provenance import hash_file, hash_mapping


@dataclass(frozen=True)
class PublicMechanismReadiness:
    """Aggregate-only readiness output."""

    support_matrix: pd.DataFrame
    manifest: dict[str, Any]
    report: str


def run_public_mechanism_readiness(
    config_path: str | Path = "config/public_mechanism_readiness_v1.yaml",
    *,
    repo_root: str | Path | None = None,
) -> PublicMechanismReadiness:
    """Build and write the aggregate-only public mechanism support preflight."""

    config_path = Path(config_path)
    root = Path(repo_root) if repo_root is not None else config_path.resolve().parents[1]
    config = load_yaml_config(config_path)
    _validate_config(config)
    registry = load_variable_registry(root / str(config["registry"]["variable_registry_path"]))
    acdc_dir = root / str(config["inputs"]["full_acdc_aggregate_dir"])
    paired_dir = root / str(config["inputs"]["paired_aggregate_dir"])
    aggregates = _load_aggregates(acdc_dir, paired_dir)
    support = build_support_matrix(config, aggregates)
    if tuple(support["variable_id"]) != REQUIRED_VARIABLE_IDS:
        raise ConfigValidationError("support matrix must preserve the frozen variable order")

    outputs = config["outputs"]
    support_path = root / str(outputs["support_matrix_csv"])
    manifest_path = root / str(outputs["manifest_json"])
    report_path = root / str(outputs["report_md"])
    report = public_readiness_report(support)
    _atomic_write_csv(support_path, support)
    manifest = {
        "study_id": "public_mechanism_readiness_v1",
        "status": "aggregate_support_preflight_only",
        "formal_claims_allowed": False,
        "model_fitting_allowed": False,
        "real_transfer_outcomes_allowed": False,
        "trident_validation_claim_allowed": False,
        "variable_registry_id": registry.registry_id,
        "variable_registry_commit": str(config["registry"]["variable_registry_commit"]),
        "config_path": _display_path(config_path, root),
        "config_hash": hash_mapping(config),
        "input_hashes": {
            "full_acdc_template_support": hash_file(acdc_dir / "template_support.csv"),
            "full_acdc_temporal_summary": hash_file(acdc_dir / "temporal_summary.csv"),
            "full_acdc_feature_summary": hash_file(acdc_dir / "feature_summary.csv"),
            "paired_session_support": hash_file(paired_dir / "paired_session_support.csv"),
            "paired_session_feature_summary": hash_file(paired_dir / "paired_session_feature_summary.csv"),
            "paired_session_repeated_person_stability": hash_file(
                paired_dir / "paired_session_repeated_person_stability.csv"
            ),
        },
        "n_variables": int(support.shape[0]),
        "support_counts": support["support_status"].value_counts().sort_index().to_dict(),
    }
    _atomic_write_json(manifest_path, manifest)
    report_path.write_text(report, encoding="utf-8")
    manifest["output_hashes"] = {
        "support_matrix": hash_file(support_path),
        "report": hash_file(report_path),
    }
    _atomic_write_json(manifest_path, manifest)
    return PublicMechanismReadiness(support_matrix=support, manifest=manifest, report=report)


def build_support_matrix(config: dict[str, Any], aggregates: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Map aggregate support outputs onto the frozen M3 variable registry."""

    thresholds = config["support_thresholds"]
    eligible_acdc = _eligible_acdc_templates(aggregates["acdc_template_support"], thresholds)
    paired_support = aggregates["paired_support"]
    paired_features = aggregates["paired_feature_summary"]
    temporal = aggregates["acdc_temporal"]
    acdc_tasks = _task_summary(eligible_acdc)
    paired_tasks = _paired_task_summary(paired_support)

    rows = [
        _row(
            "K",
            "supported",
            "ACDC supports between-person/static observed behaviour across eligible templates; paired source supports repeat/session stability for Stroop/Flanker/SART.",
            acdc_tasks=acdc_tasks,
            paired_tasks=paired_tasks,
            features="accuracy|median_rt_ms|mean_response_speed|rt_cv|throughput_proxy|mean_rt_ms",
        ),
        _row(
            "V",
            "partial",
            "Paired SART supports session-level vigilance/readiness features with repeat participants; Full ACDC has no eligible SART window template.",
            acdc_tasks="none_for_SART",
            paired_tasks=_paired_task_summary(paired_support[paired_support["task_id"] == "SART"]),
            features=_available_features(paired_features, "SART"),
        ),
        _row(
            "C_signal",
            "partial",
            "Paired Stroop/Flanker provide conflict-cost session summaries; Full ACDC core aggregates do not provide direct conflict-cost window features.",
            acdc_tasks="Stroop|Flanker_core_only",
            paired_tasks=_paired_task_summary(paired_support[paired_support["task_id"].isin(["Stroop", "Flanker"])]),
            features=_available_features(paired_features, ("Stroop", "Flanker"), prefix="conflict"),
        ),
        _row(
            "A_evidence",
            "unsupported",
            "Current aggregate sources do not expose post-error, change-point or evidence-weighting observables required for primary Evidence updating analysis.",
            acdc_tasks="none",
            paired_tasks="none",
            features="none",
        ),
        _row(
            "T_commit",
            "partial_proxy_only",
            "RT/speed/accuracy can describe decision timing, but no deadline or explicit speed-accuracy manipulation is available for primary Commit-threshold identification.",
            acdc_tasks=acdc_tasks,
            paired_tasks=paired_tasks,
            features="median_rt_ms|mean_rt_ms|mean_response_speed|accuracy",
        ),
        _row(
            "PC_calibration",
            "unsupported",
            "No confidence, prediction-error or source-reliability observables are available in the current aggregate sources.",
            acdc_tasks="none",
            paired_tasks="none",
            features="none",
        ),
        _row(
            "R_dynamic",
            "partial_temporal_not_regime",
            "ACDC aggregate temporal support gives lag-1 and fatigue/time-on-task summaries, but this is not enough to identify dynamic regimes.",
            acdc_tasks=_temporal_task_summary(temporal),
            paired_tasks="paired_source_session_summary_no_window_sequences",
            features="lag1|fatigue",
        ),
        _row(
            "P_pace",
            "descriptive_only",
            "PACE remains a descriptive phenotype until independently supported; no forced four-profile ontology is authorised.",
            acdc_tasks="not_primary",
            paired_tasks="not_primary",
            features="speed_accuracy_variability_patterns_only",
        ),
        _row(
            "Y_behavior",
            "supported",
            "Observed behaviour is available through ACDC core window features and paired session summaries.",
            acdc_tasks=acdc_tasks,
            paired_tasks=paired_tasks,
            features="accuracy|rt|speed|variability|throughput|conflict_costs|vigilance_errors",
        ),
        _row(
            "Transfer_external",
            "forbidden",
            "Real wrapper-transfer outcomes remain external and must not be inspected or used until a later prospective prediction freeze.",
            acdc_tasks="not_applicable",
            paired_tasks="not_applicable",
            features="none",
        ),
    ]
    support = pd.DataFrame(rows)
    support["model_fitting_allowed"] = False
    support["real_transfer_outcomes_allowed"] = False
    support["trident_validation_claim_allowed"] = False
    return support


def public_readiness_report(support: pd.DataFrame) -> str:
    """Render the public mechanism readiness preflight."""

    return "\n".join(
        [
            "# M5 Public Mechanism Readiness Preflight",
            "",
            "**Status:** aggregate support preflight only",
            "",
            "No mechanism model is fit. No confirmatory Trident-G, APC, PACE,",
            "neural-criticality, cusp or transfer claim is made.",
            "",
            "## Purpose",
            "",
            "This preflight maps available public aggregate evidence onto the frozen",
            "M3 variable registry before any real-data mechanism analysis.",
            "",
            "## Variable Support Matrix",
            "",
            "```csv",
            support.to_csv(index=False, lineterminator="\n"),
            "```",
            "",
            "## Interpretation Boundary",
            "",
            "Supported means the current public aggregates can supply observables for",
            "a prospective analysis design. It does not mean the latent variable is",
            "real, validated or preferred. Partial support must remain explicitly",
            "limited in any later model. Unsupported and forbidden variables must not",
            "be silently estimated from proxies.",
        ]
    )


def _load_aggregates(acdc_dir: Path, paired_dir: Path) -> dict[str, pd.DataFrame]:
    required = {
        "acdc_template_support": acdc_dir / "template_support.csv",
        "acdc_temporal": acdc_dir / "temporal_summary.csv",
        "acdc_feature_summary": acdc_dir / "feature_summary.csv",
        "paired_support": paired_dir / "paired_session_support.csv",
        "paired_feature_summary": paired_dir / "paired_session_feature_summary.csv",
        "paired_stability": paired_dir / "paired_session_repeated_person_stability.csv",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise ConfigValidationError("missing aggregate preflight outputs: " + ", ".join(missing))
    return {key: pd.read_csv(path) for key, path in required.items()}


def _validate_config(config: dict[str, Any]) -> None:
    registry = _required_mapping(config, "registry")
    inputs = _required_mapping(config, "inputs")
    if registry.get("id") != "public_mechanism_readiness_v1":
        raise ConfigValidationError("registry.id must be public_mechanism_readiness_v1")
    for field in (
        "formal_claims_allowed",
        "real_transfer_outcomes_allowed",
        "trident_validation_claim_allowed",
        "neural_criticality_claim_allowed",
        "cusp_required",
        "model_fitting_allowed",
    ):
        if registry.get(field) is not False:
            raise ConfigValidationError(f"registry.{field} must be false")
    if inputs.get("aggregate_only") is not True:
        raise ConfigValidationError("public readiness preflight must be aggregate-only")
    if inputs.get("raw_participant_data_allowed") is not False:
        raise ConfigValidationError("raw participant data must not be loaded")


def _eligible_acdc_templates(template_support: pd.DataFrame, thresholds: dict[str, Any]) -> pd.DataFrame:
    frame = template_support.copy()
    return frame[
        (pd.to_numeric(frame["n_participants"]) >= int(thresholds["full_acdc_template_min_participants"]))
        & (pd.to_numeric(frame["n_rows"]) >= int(thresholds["full_acdc_template_min_rows"]))
        & (
            pd.to_numeric(frame["feature_coverage_min"])
            >= float(thresholds["full_acdc_template_min_feature_coverage"])
        )
    ].copy()


def _task_summary(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "none"
    counts = frame.groupby("task_id").size().sort_index()
    return "|".join(f"{task}:{int(count)}" for task, count in counts.items())


def _paired_task_summary(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "none"
    return "|".join(
        f"{row.task_id}:participants={int(row.n_participants)},repeat={int(row.n_repeat_participants)}"
        for row in frame.sort_values("task_id").itertuples(index=False)
    )


def _temporal_task_summary(frame: pd.DataFrame) -> str:
    estimated = frame[frame["lag1_support_status"] == "estimated"]
    if estimated.empty:
        return "none"
    counts = estimated.groupby("task_id")["source_dataset"].nunique().sort_index()
    return "|".join(f"{task}:lag1_templates={int(count)}" for task, count in counts.items())


def _available_features(
    feature_summary: pd.DataFrame,
    tasks: str | tuple[str, ...],
    *,
    prefix: str | None = None,
) -> str:
    task_values = (tasks,) if isinstance(tasks, str) else tasks
    frame = feature_summary[feature_summary["task_id"].isin(task_values)].copy()
    frame = frame[pd.to_numeric(frame["n_observed"]) >= 100]
    if prefix is not None:
        frame = frame[frame["feature"].str.startswith(prefix)]
    if frame.empty:
        return "none"
    return "|".join(sorted(frame["feature"].unique().tolist()))


def _row(
    variable_id: str,
    support_status: str,
    support_reason: str,
    *,
    acdc_tasks: str,
    paired_tasks: str,
    features: str,
) -> dict[str, Any]:
    return {
        "variable_id": variable_id,
        "support_status": support_status,
        "acdc_task_support": acdc_tasks,
        "paired_task_support": paired_tasks,
        "available_features": features,
        "support_reason": support_reason,
    }


def _required_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = config.get(key)
    if not isinstance(value, dict) or not value:
        raise ConfigValidationError(f"{key} must be a non-empty mapping")
    return value


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _display_path(path: Path, root: Path) -> str:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    try:
        return str(resolved_path.relative_to(resolved_root).as_posix())
    except ValueError:
        return str(path.as_posix())


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for aggregate-only public mechanism readiness."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/public_mechanism_readiness_v1.yaml")
    args = parser.parse_args(argv)
    outputs = run_public_mechanism_readiness(args.config)
    print(
        json.dumps(
            {
                "study_id": outputs.manifest["study_id"],
                "status": outputs.manifest["status"],
                "support_counts": outputs.manifest["support_counts"],
                "model_fitting_allowed": False,
                "real_transfer_outcomes_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
