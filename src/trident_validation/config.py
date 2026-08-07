"""Configuration loading and registry validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import yaml


class ConfigValidationError(ValueError):
    """Raised when repository configuration violates the protocol contract."""


_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")


@dataclass(frozen=True)
class UpstreamEvidenceRecord:
    """Pinned upstream evidence source used as reference-only prior evidence."""

    key: str
    repository: str
    branch: str
    role: str
    pinned_commit: str
    materials: tuple[dict[str, str], ...]
    import_policy: dict[str, Any]


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and require a mapping at the document root."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigValidationError(f"{config_path} must contain a YAML mapping")
    return loaded


def load_config_directory(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load every ``*.yaml`` file in a config directory keyed by stem."""

    config_dir = Path(path)
    return {
        config_file.stem: load_yaml_config(config_file)
        for config_file in sorted(config_dir.glob("*.yaml"))
    }


def validate_upstream_registry(config: dict[str, Any]) -> dict[str, UpstreamEvidenceRecord]:
    """Validate ``config/upstream_evidence.yaml``.

    The registry is intentionally strict because upstream evidence is a pinned
    reference, not a mutable training source.
    """

    entries = config.get("upstream_evidence")
    if not isinstance(entries, dict) or not entries:
        raise ConfigValidationError("upstream_evidence must be a non-empty mapping")

    validated: dict[str, UpstreamEvidenceRecord] = {}
    for key, record in entries.items():
        if not isinstance(record, dict):
            raise ConfigValidationError(f"{key} must be a mapping")

        missing = [
            field
            for field in ("repository", "branch", "role", "pinned_commit", "materials", "import_policy")
            if field not in record
        ]
        if missing:
            raise ConfigValidationError(f"{key} missing required fields: {', '.join(missing)}")

        pinned_commit = str(record["pinned_commit"])
        if not _COMMIT_RE.fullmatch(pinned_commit):
            raise ConfigValidationError(f"{key} pinned_commit must be a 40-character commit hash")

        materials = record["materials"]
        if not isinstance(materials, list) or not materials:
            raise ConfigValidationError(f"{key} materials must be a non-empty list")
        for index, material in enumerate(materials):
            if not isinstance(material, dict) or "path" not in material or "purpose" not in material:
                raise ConfigValidationError(
                    f"{key} materials[{index}] must contain path and purpose"
                )

        import_policy = record["import_policy"]
        if not isinstance(import_policy, dict):
            raise ConfigValidationError(f"{key} import_policy must be a mapping")
        if import_policy.get("mode") != "reference_only":
            raise ConfigValidationError(f"{key} import_policy.mode must be reference_only")
        if import_policy.get("do_not_modify_upstream_repo") is not True:
            raise ConfigValidationError(
                f"{key} must explicitly prohibit modifying the upstream repository"
            )

        validated[key] = UpstreamEvidenceRecord(
            key=str(key),
            repository=str(record["repository"]),
            branch=str(record["branch"]),
            role=str(record["role"]),
            pinned_commit=pinned_commit.lower(),
            materials=tuple(dict(item) for item in materials),
            import_policy=dict(import_policy),
        )

    return validated


def require_registered_upstream_commit(
    registry: dict[str, UpstreamEvidenceRecord],
    repository: str,
    commit: str,
) -> UpstreamEvidenceRecord:
    """Return the matching upstream record or fail loudly."""

    normalised_commit = commit.lower()
    for record in registry.values():
        if record.repository == repository and record.pinned_commit == normalised_commit:
            return record
    raise ConfigValidationError(
        f"{repository}@{commit} is not registered as pinned upstream evidence"
    )

