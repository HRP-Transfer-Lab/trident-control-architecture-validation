"""M2.7 empirical-twin nuisance preflight."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import pandas as pd

from trident_validation.config import load_yaml_config
from trident_validation.models.static_tournament_v2 import STATIC_TOURNAMENT_V2_CONTRACT
from trident_validation.schema import validate_window_schema
from trident_validation.synthetic.fixtures import CORE_SYNTHETIC_FEATURES, make_synthetic_window_table
from trident_validation.synthetic.recovery import ground_truth_columns


PREFLIGHT_STUDY_ID = "empirical_twin_preflight_v1"
DEFAULT_CONFIG_PATH = Path("config/empirical_twin_preflight_v1.yaml")
DEFAULT_OUTPUT_DIR = Path("reports/generated/empirical_twin_preflight_v1")
TEMPLATE_COLUMNS = ("source_dataset", "task_id")


def run_empirical_twin_preflight(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run nuisance-only preflight for M2.7 empirical-twin design."""

    start = time.perf_counter()
    config = load_yaml_config(config_path)
    _validate_preflight_config(config)
    frame, input_mode = _load_or_make_preflight_frame(config)
    report = validate_window_schema(frame)
    _assert_no_truth_inputs(frame)
    feature_columns = tuple(config["nuisance_features"]["core_features"])
    nuisance = estimate_empirical_nuisance(frame, feature_columns=feature_columns)
    target_dir = Path(output_dir or config["outputs"]["directory"])
    paths = _write_preflight_outputs(
        nuisance,
        output_dir=target_dir,
        schema_report=report,
        input_mode=input_mode,
    )
    summary = {
        "study_id": PREFLIGHT_STUDY_ID,
        "input_mode": input_mode,
        "static_tournament_contract": STATIC_TOURNAMENT_V2_CONTRACT.contract_id,
        "n_rows": int(report.n_rows),
        "n_participants": int(report.n_participants),
        "n_sources": int(report.n_sources),
        "n_tasks": int(report.n_tasks),
        "n_templates": int(nuisance["template_summary"].shape[0]),
        "formal_claims_allowed": False,
        "runtime_seconds": round(float(time.perf_counter() - start), 3),
    }
    summary_path = target_dir / "empirical_twin_preflight_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    paths["summary"] = summary_path
    return {
        **summary,
        "paths": {key: str(value) for key, value in paths.items()},
    }


def estimate_empirical_nuisance(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = CORE_SYNTHETIC_FEATURES,
) -> dict[str, pd.DataFrame]:
    """Estimate nuisance summaries without reading latent truth columns."""

    _assert_no_truth_inputs(frame)
    features = tuple(feature_columns)
    template_summary = _template_summary(frame, features)
    feature_summary = _feature_summary(frame, features)
    covariance = _template_covariance(frame, features)
    variance = _variance_decomposition(frame, features)
    temporal = _temporal_summary(frame, features)
    missingness = _missingness_summary(frame, features)
    return {
        "template_summary": template_summary,
        "feature_summary": feature_summary,
        "template_covariance": covariance,
        "variance_decomposition": variance,
        "temporal_summary": temporal,
        "missingness_summary": missingness,
    }


def _load_or_make_preflight_frame(config: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    input_paths = [Path(path) for path in config["inputs"].get("empirical_window_tables", [])]
    if input_paths:
        frames = [_read_window_table(path) for path in input_paths]
        return pd.concat(frames, ignore_index=True), "configured_empirical_tables"
    fixture = config["inputs"]["fallback_smoke_fixture"]
    if not bool(fixture.get("enabled", False)):
        raise ValueError("no empirical_window_tables configured and fallback smoke fixture disabled")
    return (
        make_synthetic_window_table(
            seed=int(fixture["seed"]),
            n_datasets=int(fixture["n_datasets"]),
            participants_per_dataset=int(fixture["participants_per_dataset"]),
            sessions_per_participant=int(fixture["sessions_per_participant"]),
            min_windows_per_session=int(fixture["min_windows_per_session"]),
            max_windows_per_session=int(fixture["max_windows_per_session"]),
        ),
        "fallback_smoke_fixture",
    )


def _read_window_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"empirical window table not found: {path}")
    suffixes = "".join(path.suffixes[-2:]).lower()
    if path.suffix.lower() == ".csv" or suffixes == ".csv.gz":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"unsupported empirical table format: {path}")


def _template_summary(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(list(TEMPLATE_COLUMNS), dropna=False, sort=True):
        source, task = keys
        row: dict[str, Any] = {
            "source_dataset": str(source),
            "task_id": str(task),
            "n_rows": int(group.shape[0]),
            "n_participants": int(
                group.loc[:, ["source_dataset", "participant_id"]].drop_duplicates().shape[0]
            ),
            "mean_n_trials_valid": float(pd.to_numeric(group["n_trials_valid"]).mean()),
            "sd_n_trials_valid": float(pd.to_numeric(group["n_trials_valid"]).std(ddof=1)),
        }
        for feature in features:
            values = pd.to_numeric(group[feature], errors="coerce")
            row[f"{feature}_mean"] = float(values.mean())
            row[f"{feature}_sd"] = float(values.std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def _feature_summary(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in features:
        values = pd.to_numeric(frame[feature], errors="coerce")
        rows.append(
            {
                "feature": feature,
                "n_observed": int(values.notna().sum()),
                "missing_rate": float(values.isna().mean()),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "p10": float(values.quantile(0.10)),
                "p50": float(values.quantile(0.50)),
                "p90": float(values.quantile(0.90)),
            }
        )
    return pd.DataFrame(rows)


def _template_covariance(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(list(TEMPLATE_COLUMNS), dropna=False, sort=True):
        source, task = keys
        matrix = group.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
        for feature_a in features:
            for feature_b in features:
                pair = pd.DataFrame(
                    {
                        "feature_a": matrix[feature_a],
                        "feature_b": matrix[feature_b],
                    }
                ).dropna()
                if pair.shape[0] < 2:
                    covariance = float("nan")
                    correlation = float("nan")
                else:
                    covariance = float(pair.cov().iloc[0, 1])
                    correlation = (
                        float(pair.corr().iloc[0, 1])
                        if float(pair["feature_a"].std(ddof=1)) > 0
                        and float(pair["feature_b"].std(ddof=1)) > 0
                        else float("nan")
                    )
                rows.append(
                    {
                        "source_dataset": str(source),
                        "task_id": str(task),
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "covariance": covariance,
                        "correlation": correlation,
                    }
                )
    return pd.DataFrame(rows)


def _variance_decomposition(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    participant_keys = ["source_dataset", "participant_id"]
    for feature in features:
        values = pd.to_numeric(frame[feature], errors="coerce")
        work = frame.loc[:, participant_keys].copy()
        work[feature] = values
        observed = work.dropna(subset=[feature])
        participant_means = observed.groupby(participant_keys, sort=True)[feature].mean()
        between = float(participant_means.var(ddof=1))
        with_means = observed.join(
            participant_means.rename("participant_mean"),
            on=participant_keys,
        )
        within = float((with_means[feature] - with_means["participant_mean"]).var(ddof=1))
        total = float(values.var(ddof=1))
        rows.append(
            {
                "feature": feature,
                "total_variance": total,
                "between_participant_variance": between,
                "within_participant_variance": within,
                "between_participant_fraction": between / total if total else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _temporal_summary(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in features:
        rows.append(
            {
                "feature": feature,
                "mean_lag1_autocorrelation": _mean_lag1_autocorrelation(frame, feature),
                "practice_slope": _mean_linear_slope(frame, feature, "practice_or_session_index"),
                "time_on_task_slope": _mean_linear_slope(frame, feature, "time_on_task"),
            }
        )
    return pd.DataFrame(rows)


def _mean_lag1_autocorrelation(frame: pd.DataFrame, feature: str) -> float:
    correlations: list[float] = []
    grouped = frame.sort_values(
        ["source_dataset", "participant_id", "session_id", "window_start_trial"]
    ).groupby(["source_dataset", "participant_id", "session_id"], sort=True)
    for _, group in grouped:
        values = pd.to_numeric(group[feature], errors="coerce").dropna().to_numpy(dtype=float)
        if values.shape[0] < 3 or float(np.std(values[:-1])) == 0 or float(np.std(values[1:])) == 0:
            continue
        correlations.append(float(np.corrcoef(values[:-1], values[1:])[0, 1]))
    return float(np.mean(correlations)) if correlations else float("nan")


def _mean_linear_slope(frame: pd.DataFrame, feature: str, predictor: str) -> float:
    if predictor not in frame.columns:
        return float("nan")
    slopes: list[float] = []
    grouped = frame.groupby(["source_dataset", "participant_id"], sort=True)
    for _, group in grouped:
        x = pd.to_numeric(group[predictor], errors="coerce")
        y = pd.to_numeric(group[feature], errors="coerce")
        valid = x.notna() & y.notna()
        if int(valid.sum()) < 2 or float(np.std(x[valid])) == 0:
            continue
        slope = np.polyfit(x[valid].to_numpy(dtype=float), y[valid].to_numpy(dtype=float), deg=1)[0]
        slopes.append(float(slope))
    return float(np.mean(slopes)) if slopes else float("nan")


def _missingness_summary(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(list(TEMPLATE_COLUMNS), dropna=False, sort=True):
        source, task = keys
        for feature in features:
            rows.append(
                {
                    "source_dataset": str(source),
                    "task_id": str(task),
                    "feature": feature,
                    "n_rows": int(group.shape[0]),
                    "missing_rate": float(group[feature].isna().mean()),
                }
            )
    return pd.DataFrame(rows)


def _write_preflight_outputs(
    nuisance: dict[str, pd.DataFrame],
    *,
    output_dir: Path,
    schema_report: Any,
    input_mode: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in nuisance.items():
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    report_path = output_dir / "empirical_twin_preflight_report.md"
    report_path.write_text(
        _preflight_markdown_report(
            nuisance,
            schema_report=schema_report,
            input_mode=input_mode,
        ),
        encoding="utf-8",
    )
    paths["report"] = report_path
    return paths


def _preflight_markdown_report(
    nuisance: dict[str, pd.DataFrame],
    *,
    schema_report: Any,
    input_mode: str,
) -> str:
    return "\n".join(
        [
            "# M2.7 Empirical-Twin Preflight",
            "",
            f"input_mode: `{input_mode}`",
            f"static_contract: `{STATIC_TOURNAMENT_V2_CONTRACT.contract_id}`",
            "formal_claims_allowed: `false`",
            "",
            "## Schema",
            "",
            f"rows: {schema_report.n_rows}",
            f"participants: {schema_report.n_participants}",
            f"sources: {schema_report.n_sources}",
            f"tasks: {schema_report.n_tasks}",
            "",
            "## Outputs",
            "",
            *[f"- `{name}`: {frame.shape[0]} rows" for name, frame in nuisance.items()],
            "",
            "Empirical data are used for nuisance estimation only. No latent truth columns are accepted.",
        ]
    )


def _assert_no_truth_inputs(frame: pd.DataFrame) -> None:
    truth_columns = ground_truth_columns(frame)
    if truth_columns:
        raise ValueError(
            "empirical nuisance input must not contain ground-truth columns: "
            + ", ".join(truth_columns)
        )


def _validate_preflight_config(config: dict[str, Any]) -> None:
    if config.get("study", {}).get("id") != PREFLIGHT_STUDY_ID:
        raise ValueError(f"study.id must be {PREFLIGHT_STUDY_ID}")
    contract = config.get("contract", {})
    if contract.get("static_tournament_contract") != STATIC_TOURNAMENT_V2_CONTRACT.contract_id:
        raise ValueError("preflight must use static_tournament_v2")
    if not bool(contract.get("nuisance_only_from_empirical_data")):
        raise ValueError("nuisance_only_from_empirical_data must be true")
    if bool(contract.get("formal_claims_allowed")):
        raise ValueError("preflight config must not allow formal claims")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run M2.7 empirical-twin nuisance preflight.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    result = run_empirical_twin_preflight(
        config_path=args.config,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
