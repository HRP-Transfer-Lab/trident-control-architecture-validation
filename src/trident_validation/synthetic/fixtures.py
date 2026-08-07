"""Fully synthetic canonical window-table fixture."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


CORE_SYNTHETIC_FEATURES = (
    "accuracy",
    "median_rt_ms",
    "mean_response_speed",
    "rt_cv",
    "throughput_proxy",
)


def make_synthetic_window_table(
    *,
    seed: int = 20260807,
    n_datasets: int = 3,
    participants_per_dataset: int = 40,
    sessions_per_participant: int = 2,
    min_windows_per_session: int = 2,
    max_windows_per_session: int = 4,
) -> pd.DataFrame:
    """Generate a realistic synthetic canonical window table.

    The fixture includes stable person differences, source/task shifts,
    practice, fatigue, unequal valid trial counts, technical missingness and
    structural missingness flags. It contains no Trident/PACE discovery labels.
    """

    if n_datasets < 3:
        raise ValueError("Milestone 1 fixture requires at least 3 datasets")
    if participants_per_dataset < 1:
        raise ValueError("participants_per_dataset must be positive")
    if sessions_per_participant < 1:
        raise ValueError("sessions_per_participant must be positive")
    if not 1 <= min_windows_per_session <= max_windows_per_session:
        raise ValueError("invalid window range")

    rng = np.random.default_rng(seed)
    source_names = [f"synthetic_source_{index + 1}" for index in range(n_datasets)]
    source_shifts = {
        source: rng.normal(loc=0.0, scale=0.25)
        for source in source_names
    }
    tasks = ("stroop", "flanker", "sart", "task_switch")
    rows: list[dict[str, object]] = []
    checksum = "sha256:" + hashlib.sha256(f"synthetic:{seed}".encode("utf-8")).hexdigest()

    for source_index, source in enumerate(source_names):
        for participant_index in range(participants_per_dataset):
            participant_id = f"{source}_p{participant_index:03d}"
            person_trait = rng.normal(loc=0.0, scale=0.7)
            response_policy = rng.normal(loc=0.0, scale=0.35)
            for session_number in range(1, sessions_per_participant + 1):
                session_practice = 0.12 * (session_number - 1)
                n_windows = int(rng.integers(min_windows_per_session, max_windows_per_session + 1))
                session_id = f"s{session_number:02d}"
                for window_number in range(1, n_windows + 1):
                    task_id = tasks[(source_index + participant_index + session_number + window_number) % len(tasks)]
                    block_id = f"b{1 + (window_number - 1) // 2:02d}"
                    window_id = f"w{window_number:02d}"
                    start_trial = 1 + (window_number - 1) * 32
                    total_trials = int(rng.integers(28, 45))
                    invalid_trials = int(rng.binomial(total_trials, 0.04))
                    valid_trials = max(1, total_trials - invalid_trials)
                    time_on_task = (window_number - 1) * 65.0
                    fatigue = -0.015 * window_number + rng.normal(0.0, 0.05)
                    source_shift = source_shifts[source]
                    task_shift = {
                        "stroop": -0.05,
                        "flanker": 0.0,
                        "sart": -0.08,
                        "task_switch": -0.12,
                    }[task_id]
                    latent_performance = (
                        person_trait
                        + source_shift
                        + task_shift
                        + session_practice
                        + fatigue
                        + rng.normal(0.0, 0.25)
                    )

                    accuracy = float(np.clip(0.78 + 0.07 * latent_performance, 0.45, 0.99))
                    median_rt = float(np.clip(720 - 55 * latent_performance + 45 * response_policy, 260, 1400))
                    response_speed = float(1000.0 / median_rt)
                    rt_cv = float(np.clip(0.24 - 0.025 * latent_performance + rng.normal(0.0, 0.03), 0.05, 0.7))
                    throughput = float(accuracy * response_speed * 100.0)

                    has_conflict = task_id in {"stroop", "flanker"}
                    has_post_error = task_id in {"stroop", "flanker", "task_switch"}
                    has_vigilance = task_id == "sart"
                    has_switch = task_id == "task_switch"

                    row: dict[str, object] = {
                        "source_dataset": source,
                        "source_version": "synthetic-v1",
                        "participant_id": participant_id,
                        "session_id": session_id,
                        "task_id": task_id,
                        "block_id": block_id,
                        "window_id": window_id,
                        "window_start_trial": start_trial,
                        "window_end_trial": start_trial + total_trials - 1,
                        "n_trials_total": total_trials,
                        "n_trials_valid": valid_trials,
                        "source_file_or_table": "synthetic.infrastructure_fixture",
                        "source_commit_or_release": "synthetic-milestone1",
                        "source_hash_if_available": checksum,
                        "preprocessing_version": "synthetic-preprocessing-v1",
                        "feature_version": "canonical-window-v1",
                        "accuracy": accuracy,
                        "median_rt_ms": median_rt,
                        "mean_response_speed": response_speed,
                        "rt_cv": rt_cv,
                        "throughput_proxy": throughput,
                        "trial_count": valid_trials,
                        "practice_or_session_index": session_number,
                        "condition_mix": "mixed",
                        "congruency_mix": "balanced" if has_conflict else "not_applicable",
                        "switch_rate": float(rng.uniform(0.25, 0.45)) if has_switch else np.nan,
                        "lure_rate": float(rng.uniform(0.08, 0.22)) if has_vigilance else np.nan,
                        "difficulty_level": int(rng.integers(1, 4)),
                        "soa_or_foreperiod": float(rng.choice([500.0, 750.0, 1000.0])),
                        "time_on_task": time_on_task,
                        "response_mapping": "standard",
                        "input_device": rng.choice(["keyboard", "touchpad"]),
                        "timing_quality": rng.choice(["good", "acceptable"], p=[0.85, 0.15]),
                        "browser_focus_flags": "none",
                        "has_conflict_cost": has_conflict,
                        "has_post_error": has_post_error,
                        "has_vigilance": has_vigilance,
                        "has_switch_structure": has_switch,
                        "has_confidence": False,
                        "has_change_point": False,
                        "conflict_cost_rt": float(rng.normal(75 - 8 * latent_performance, 20)) if has_conflict else np.nan,
                        "conflict_cost_accuracy": float(rng.normal(0.04 - 0.01 * latent_performance, 0.015)) if has_conflict else np.nan,
                        "post_error_adjustment": float(rng.normal(35 + 8 * response_policy, 15)) if has_post_error else np.nan,
                        "error_burstiness": float(np.clip(rng.normal(0.12 - 0.02 * latent_performance, 0.03), 0.0, 1.0)) if has_post_error else np.nan,
                        "lag1": float(rng.normal(0.18, 0.05)),
                        "lag2": float(rng.normal(0.08, 0.03)),
                        "roughness": float(np.clip(rng.normal(0.22 - 0.02 * latent_performance, 0.04), 0.01, 1.0)),
                        "permutation_entropy": float(np.clip(rng.normal(0.65 - 0.03 * latent_performance, 0.06), 0.0, 1.0)),
                        "difference_entropy": float(np.clip(rng.normal(0.58 - 0.02 * latent_performance, 0.06), 0.0, 1.0)),
                        "temporal_drift": float(rng.normal(-0.01 * window_number, 0.03)),
                        "sign_change_rate": float(np.clip(rng.normal(0.35, 0.08), 0.0, 1.0)),
                        "large_update_rate": float(np.clip(rng.normal(0.08 + 0.02 * abs(response_policy), 0.03), 0.0, 1.0)),
                        "recovery_slope": float(rng.normal(0.04 + 0.02 * latent_performance, 0.03)) if has_post_error else np.nan,
                        "vigilance_engagement": float(np.clip(0.74 + 0.05 * latent_performance, 0.1, 1.0)) if has_vigilance else np.nan,
                        "inhibitory_stability": float(np.clip(0.7 + 0.04 * latent_performance, 0.1, 1.0)) if has_vigilance else np.nan,
                        "reciprocal_rt": response_speed if has_vigilance else np.nan,
                        "slow_tail_response_speed": float(response_speed * rng.uniform(0.7, 0.9)) if has_vigilance else np.nan,
                        "lapse_rate": float(np.clip(0.08 - 0.02 * latent_performance, 0.0, 0.5)) if has_vigilance else np.nan,
                        "false_start_rate": float(np.clip(0.03 + 0.01 * response_policy, 0.0, 0.3)) if has_vigilance else np.nan,
                        "vigilance_drift": float(rng.normal(-0.02 * window_number, 0.03)) if has_vigilance else np.nan,
                    }

                    if rng.random() < 0.015:
                        row["median_rt_ms"] = np.nan
                        row["mean_response_speed"] = np.nan
                    rows.append(row)

    return pd.DataFrame(rows)

