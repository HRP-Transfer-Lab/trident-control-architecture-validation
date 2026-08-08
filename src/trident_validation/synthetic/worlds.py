"""Static synthetic null worlds W0-W4."""

from __future__ import annotations

import hashlib
from typing import Literal

import numpy as np
import pandas as pd


StaticWorldId = Literal[
    "W0_general_performance",
    "W1_continuous_manifold",
    "W2_nonlinear_vigilance",
    "W3_three_profile_mixture",
    "W4_four_pace_mixture",
]

STATIC_SYNTHETIC_WORLD_IDS: tuple[StaticWorldId, ...] = (
    "W0_general_performance",
    "W1_continuous_manifold",
    "W2_nonlinear_vigilance",
    "W3_three_profile_mixture",
    "W4_four_pace_mixture",
)

WORLD_MODEL_ALIGNMENT = {
    "W0_general_performance": "M0_probabilistic_general_performance",
    "W1_continuous_manifold": "M1_continuous_control_manifold",
    "W2_nonlinear_vigilance": "M2_nonlinear_vigilance",
    "W3_three_profile_mixture": "M3_three_profile_mixture",
    "W4_four_pace_mixture": "M4_four_pace_profile_mixture",
}


def make_static_synthetic_world(
    world_id: StaticWorldId,
    *,
    seed: int = 20260807,
    n_datasets: int = 3,
    participants_per_dataset: int = 36,
    sessions_per_participant: int = 2,
    min_windows_per_session: int = 2,
    max_windows_per_session: int = 4,
    observation_noise_scale: float = 1.0,
    source_shift_scale: float = 1.0,
    technical_missingness_rate: float = 0.01,
    latent_separation_scale: float = 1.0,
) -> pd.DataFrame:
    """Generate one fully synthetic static ground-truth world.

    W0-W4 are infrastructure falsification worlds. They are not evidence for
    the Trident hypothesis and use only neutral component identifiers.
    """

    if world_id not in STATIC_SYNTHETIC_WORLD_IDS:
        raise ValueError(f"unsupported static world_id: {world_id}")
    if n_datasets < 2:
        raise ValueError("at least two datasets are required")
    if participants_per_dataset < 1:
        raise ValueError("participants_per_dataset must be positive")
    if sessions_per_participant < 1:
        raise ValueError("sessions_per_participant must be positive")
    if not 1 <= min_windows_per_session <= max_windows_per_session:
        raise ValueError("invalid window range")
    if observation_noise_scale < 0:
        raise ValueError("observation_noise_scale must be non-negative")
    if source_shift_scale < 0:
        raise ValueError("source_shift_scale must be non-negative")
    if not 0 <= technical_missingness_rate < 1:
        raise ValueError("technical_missingness_rate must be in [0, 1)")
    if latent_separation_scale < 0:
        raise ValueError("latent_separation_scale must be non-negative")
    rng = np.random.default_rng(seed)
    source_names = [f"{world_id}_source_{index + 1}" for index in range(n_datasets)]
    source_shifts = {
        source: rng.normal(0.0, 0.18 * source_shift_scale, size=5)
        for source in source_names
    }
    tasks = ("stroop", "flanker", "sart", "task_switch")
    checksum = "sha256:" + hashlib.sha256(f"{world_id}:{seed}".encode("utf-8")).hexdigest()
    rows: list[dict[str, object]] = []

    for source_index, source in enumerate(source_names):
        for participant_index in range(participants_per_dataset):
            participant_id = f"{source}_p{participant_index:03d}"
            person = _participant_truth(
                world_id,
                rng,
                latent_separation_scale=latent_separation_scale,
            )
            for session_number in range(1, sessions_per_participant + 1):
                n_windows = int(rng.integers(min_windows_per_session, max_windows_per_session + 1))
                for window_number in range(1, n_windows + 1):
                    task_id = tasks[(source_index + participant_index + session_number + window_number) % len(tasks)]
                    latent = _window_latent(
                        world_id,
                        person,
                        session_number,
                        window_number,
                        rng,
                        latent_separation_scale=latent_separation_scale,
                    )
                    feature_vector = _feature_vector(
                        world_id,
                        latent,
                        source_shifts[source],
                        task_id,
                        rng,
                        observation_noise_scale=observation_noise_scale,
                    )
                    rows.append(
                        _canonical_row(
                            world_id=world_id,
                            checksum=checksum,
                            source=source,
                            participant_id=participant_id,
                            session_number=session_number,
                            window_number=window_number,
                            task_id=task_id,
                            feature_vector=feature_vector,
                            latent=latent,
                            rng=rng,
                            technical_missingness_rate=technical_missingness_rate,
                        )
                    )
    return pd.DataFrame(rows)


def make_all_static_synthetic_worlds(
    *,
    seed: int = 20260807,
    **kwargs: object,
) -> dict[StaticWorldId, pd.DataFrame]:
    """Generate every static synthetic world with deterministic child seeds."""

    return {
        world_id: make_static_synthetic_world(
            world_id,
            seed=seed + index * 1009,
            **kwargs,
        )
        for index, world_id in enumerate(STATIC_SYNTHETIC_WORLD_IDS)
    }


def _participant_truth(
    world_id: StaticWorldId,
    rng: np.random.Generator,
    *,
    latent_separation_scale: float,
) -> dict[str, object]:
    trait = float(rng.normal(0.0, 0.65 * latent_separation_scale))
    tilt = float(rng.normal(0.0, 0.55 * latent_separation_scale))
    readiness = float(rng.normal(0.0, 0.9 * latent_separation_scale))
    component = None
    if world_id == "W3_three_profile_mixture":
        component = int(rng.choice(3, p=[0.36, 0.34, 0.30]))
    elif world_id == "W4_four_pace_mixture":
        component = int(rng.choice(4, p=[0.28, 0.27, 0.24, 0.21]))
    return {
        "trait": trait,
        "tilt": tilt,
        "readiness": readiness,
        "component": component,
    }


def _window_latent(
    world_id: StaticWorldId,
    person: dict[str, object],
    session_number: int,
    window_number: int,
    rng: np.random.Generator,
    *,
    latent_separation_scale: float,
) -> dict[str, object]:
    practice = 0.12 * (session_number - 1)
    fatigue = -0.035 * (window_number - 1)
    trait = float(person["trait"])
    tilt = float(person["tilt"])
    readiness = float(person["readiness"]) + rng.normal(0.0, 0.25) + practice + fatigue
    latent_1 = trait + practice + fatigue + rng.normal(0.0, 0.22)
    latent_2 = tilt + rng.normal(0.0, 0.24)
    component = person["component"]
    if world_id == "W3_three_profile_mixture" and component is not None:
        centres = np.array(
            [
                [0.9, -0.2],
                [-0.8, 0.7],
                [0.15, -1.0],
            ]
        ) * latent_separation_scale
        latent_1, latent_2 = centres[int(component)] + rng.normal(0.0, 0.25, size=2)
    elif world_id == "W4_four_pace_mixture" and component is not None:
        centres = np.array(
            [
                [0.9, -0.2],
                [-0.7, 0.85],
                [0.2, -1.1],
                [-1.0, -0.85],
            ]
        ) * latent_separation_scale
        latent_1, latent_2 = centres[int(component)] + rng.normal(0.0, 0.22, size=2)
    return {
        "latent_1": float(latent_1),
        "latent_2": float(latent_2),
        "readiness": float(readiness),
        "component": component,
    }


def _feature_vector(
    world_id: StaticWorldId,
    latent: dict[str, object],
    source_shift: np.ndarray,
    task_id: str,
    rng: np.random.Generator,
    *,
    observation_noise_scale: float,
) -> np.ndarray:
    task_shift = {
        "stroop": np.array([-0.04, 0.06, -0.05, 0.04, -0.03]),
        "flanker": np.array([0.02, -0.02, 0.02, -0.01, 0.03]),
        "sart": np.array([-0.03, 0.04, -0.04, 0.06, -0.02]),
        "task_switch": np.array([-0.08, 0.08, -0.07, 0.07, -0.05]),
    }[task_id]
    latent_1 = float(latent["latent_1"])
    latent_2 = float(latent["latent_2"])
    readiness = float(latent["readiness"])
    noise = rng.normal(
        0.0,
        np.array([0.025, 18.0, 0.00008, 0.018, 0.003]) * observation_noise_scale,
    )
    if world_id == "W0_general_performance":
        raw = np.array([
            0.78 + 0.075 * latent_1,
            710.0 - 60.0 * latent_1,
            1.42 + 0.12 * latent_1,
            0.24 - 0.03 * latent_1,
            1.10 + 0.18 * latent_1,
        ])
    elif world_id == "W1_continuous_manifold":
        raw = np.array([
            0.78 + 0.06 * latent_1 - 0.025 * latent_2,
            710.0 - 48.0 * latent_1 + 55.0 * latent_2,
            1.42 + 0.09 * latent_1 - 0.08 * latent_2,
            0.23 - 0.018 * latent_1 + 0.055 * abs(latent_2),
            1.10 + 0.14 * latent_1 - 0.05 * latent_2,
        ])
    elif world_id == "W2_nonlinear_vigilance":
        inverted = 1.0 - 0.55 * (readiness - 0.15) ** 2
        raw = np.array([
            0.74 + 0.07 * inverted,
            745.0 - 55.0 * inverted + 12.0 * readiness,
            1.36 + 0.11 * inverted,
            0.28 - 0.045 * inverted + 0.025 * abs(readiness),
            1.02 + 0.17 * inverted,
        ])
    elif world_id in {"W3_three_profile_mixture", "W4_four_pace_mixture"}:
        raw = np.array([
            0.77 + 0.07 * latent_1 - 0.02 * latent_2,
            720.0 - 52.0 * latent_1 + 65.0 * latent_2,
            1.38 + 0.10 * latent_1 - 0.09 * latent_2,
            0.24 - 0.025 * latent_1 + 0.055 * abs(latent_2),
            1.06 + 0.16 * latent_1 - 0.07 * latent_2,
        ])
    else:
        raise ValueError(f"unsupported world: {world_id}")
    shifted = raw + source_shift * np.array([0.03, 30.0, 0.04, 0.02, 0.05]) + task_shift
    shifted = shifted + noise
    shifted[0] = np.clip(shifted[0], 0.35, 0.995)
    shifted[1] = np.clip(shifted[1], 260.0, 1500.0)
    shifted[2] = np.clip(shifted[2], 0.35, 3.2)
    shifted[3] = np.clip(shifted[3], 0.04, 0.95)
    shifted[4] = np.clip(shifted[4], 0.2, 2.8)
    return shifted.astype(float)


def _canonical_row(
    *,
    world_id: StaticWorldId,
    checksum: str,
    source: str,
    participant_id: str,
    session_number: int,
    window_number: int,
    task_id: str,
    feature_vector: np.ndarray,
    latent: dict[str, object],
    rng: np.random.Generator,
    technical_missingness_rate: float,
) -> dict[str, object]:
    block_id = f"b{1 + (window_number - 1) // 2:02d}"
    start_trial = 1 + (window_number - 1) * 32
    total_trials = int(rng.integers(30, 46))
    valid_trials = max(1, total_trials - int(rng.binomial(total_trials, 0.045)))
    has_conflict = task_id in {"stroop", "flanker"}
    has_post_error = task_id in {"stroop", "flanker", "task_switch"}
    has_vigilance = task_id == "sart"
    has_switch = task_id == "task_switch"
    component = latent["component"]
    accuracy, median_rt, response_speed, rt_cv, throughput = feature_vector
    if rng.random() < technical_missingness_rate:
        median_rt = np.nan
        response_speed = np.nan
        throughput = np.nan
    return {
        "source_dataset": source,
        "source_version": f"{world_id}-synthetic-v1",
        "participant_id": participant_id,
        "session_id": f"s{session_number:02d}",
        "task_id": task_id,
        "block_id": block_id,
        "window_id": f"w{window_number:02d}",
        "window_start_trial": start_trial,
        "window_end_trial": start_trial + total_trials - 1,
        "n_trials_total": total_trials,
        "n_trials_valid": valid_trials,
        "source_file_or_table": "synthetic.static_worlds",
        "source_commit_or_release": "synthetic-milestone2",
        "source_hash_if_available": checksum,
        "preprocessing_version": "synthetic-static-preprocessing-v1",
        "feature_version": "canonical-window-v1",
        "accuracy": float(accuracy),
        "median_rt_ms": float(median_rt) if not np.isnan(median_rt) else np.nan,
        "mean_response_speed": float(response_speed) if not np.isnan(response_speed) else np.nan,
        "rt_cv": float(rt_cv),
        "throughput_proxy": float(throughput),
        "trial_count": valid_trials,
        "practice_or_session_index": session_number,
        "condition_mix": "synthetic_mixed",
        "congruency_mix": "balanced" if has_conflict else "not_applicable",
        "switch_rate": float(rng.uniform(0.24, 0.48)) if has_switch else np.nan,
        "lure_rate": float(rng.uniform(0.08, 0.24)) if has_vigilance else np.nan,
        "difficulty_level": int(rng.integers(1, 4)),
        "soa_or_foreperiod": float(rng.choice([500.0, 750.0, 1000.0])),
        "time_on_task": float((window_number - 1) * 65.0),
        "response_mapping": "standard",
        "input_device": rng.choice(["keyboard", "touchpad"]),
        "timing_quality": rng.choice(["good", "acceptable"], p=[0.86, 0.14]),
        "browser_focus_flags": "none",
        "has_conflict_cost": has_conflict,
        "has_post_error": has_post_error,
        "has_vigilance": has_vigilance,
        "has_switch_structure": has_switch,
        "has_confidence": False,
        "has_change_point": False,
        "conflict_cost_rt": float(rng.normal(75.0 - 6.0 * float(latent["latent_1"]), 18.0)) if has_conflict else np.nan,
        "conflict_cost_accuracy": float(rng.normal(0.04 - 0.01 * float(latent["latent_1"]), 0.012)) if has_conflict else np.nan,
        "post_error_adjustment": float(rng.normal(35.0 + 12.0 * abs(float(latent["latent_2"])), 14.0)) if has_post_error else np.nan,
        "error_burstiness": float(np.clip(rng.normal(0.12 - 0.02 * float(latent["latent_1"]), 0.03), 0.0, 1.0)) if has_post_error else np.nan,
        "lag1": float(rng.normal(0.18, 0.05)),
        "lag2": float(rng.normal(0.08, 0.03)),
        "roughness": float(np.clip(rng.normal(0.22 + 0.03 * abs(float(latent["latent_2"])), 0.04), 0.01, 1.0)),
        "permutation_entropy": float(np.clip(rng.normal(0.62 + 0.03 * abs(float(latent["latent_2"])), 0.06), 0.0, 1.0)),
        "difference_entropy": float(np.clip(rng.normal(0.56 + 0.02 * abs(float(latent["latent_2"])), 0.06), 0.0, 1.0)),
        "temporal_drift": float(rng.normal(-0.01 * window_number, 0.03)),
        "sign_change_rate": float(np.clip(rng.normal(0.34, 0.08), 0.0, 1.0)),
        "large_update_rate": float(np.clip(rng.normal(0.08 + 0.02 * abs(float(latent["latent_2"])), 0.03), 0.0, 1.0)),
        "recovery_slope": float(rng.normal(0.04 + 0.02 * float(latent["latent_1"]), 0.03)) if has_post_error else np.nan,
        "vigilance_engagement": float(np.clip(0.74 + 0.06 * float(latent["readiness"]), 0.1, 1.0)) if has_vigilance else np.nan,
        "inhibitory_stability": float(np.clip(0.70 + 0.04 * float(latent["readiness"]), 0.1, 1.0)) if has_vigilance else np.nan,
        "reciprocal_rt": float(1000.0 / median_rt) if has_vigilance and not np.isnan(median_rt) else np.nan,
        "slow_tail_response_speed": float((1000.0 / median_rt) * rng.uniform(0.7, 0.9)) if has_vigilance and not np.isnan(median_rt) else np.nan,
        "lapse_rate": float(np.clip(0.08 - 0.02 * float(latent["readiness"]), 0.0, 0.5)) if has_vigilance else np.nan,
        "false_start_rate": float(np.clip(0.03 + 0.01 * abs(float(latent["latent_2"])), 0.0, 0.3)) if has_vigilance else np.nan,
        "vigilance_drift": float(rng.normal(-0.02 * window_number, 0.03)) if has_vigilance else np.nan,
        "synthetic_world_id": world_id,
        "synthetic_ground_truth_family": world_id.removeprefix("W").split("_", 1)[1],
        "synthetic_aligned_model_id": WORLD_MODEL_ALIGNMENT[world_id],
        "synthetic_latent_1": float(latent["latent_1"]),
        "synthetic_latent_2": float(latent["latent_2"]),
        "synthetic_readiness": float(latent["readiness"]),
        "synthetic_component_id": f"component_{component}" if component is not None else "not_applicable",
    }

