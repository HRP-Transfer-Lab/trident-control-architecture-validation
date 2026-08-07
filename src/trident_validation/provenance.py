"""Provenance and run-manifest helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib import metadata
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


class ProvenanceError(RuntimeError):
    """Raised when provenance cannot be generated safely."""


def hash_file(path: str | Path) -> str:
    """Return a SHA-256 digest with an explicit algorithm prefix."""

    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def hash_mapping(value: dict[str, Any]) -> str:
    """Hash a JSON-serialisable mapping deterministically."""

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return f"sha256:{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def get_git_commit(repo_root: str | Path = ".") -> str:
    """Return the current Git commit for ``repo_root``."""

    root = Path(repo_root)
    result = subprocess.run(
        [*_git_command_prefix(root), "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise ProvenanceError(result.stderr.strip() or "Could not read Git commit")
    return result.stdout.strip()


def get_git_dirty(repo_root: str | Path = ".") -> bool:
    """Return whether tracked or untracked files are present."""

    root = Path(repo_root)
    result = subprocess.run(
        [*_git_command_prefix(root), "status", "--porcelain"],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise ProvenanceError(result.stderr.strip() or "Could not read Git status")
    return bool(result.stdout.strip())


def package_versions(package_names: list[str] | tuple[str, ...]) -> dict[str, str | None]:
    """Return installed package versions for manifest provenance."""

    versions: dict[str, str | None] = {}
    for package_name in package_names:
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


def build_run_manifest(
    *,
    protocol: str,
    repo_root: str | Path,
    config_paths: dict[str, str | Path],
    upstream: dict[str, Any],
    seed: int,
    split: dict[str, Any],
    models: list[str],
    model_metadata: dict[str, Any] | None = None,
    input_paths: dict[str, str | Path] | None = None,
    output_paths: dict[str, str | Path] | None = None,
    timestamp_utc: str | None = None,
    include_dirty: bool = True,
) -> dict[str, Any]:
    """Build a machine-readable manifest without participant identifiers."""

    if timestamp_utc is None:
        timestamp_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    config_hashes = {
        name: hash_file(path)
        for name, path in sorted(config_paths.items(), key=lambda item: item[0])
    }
    input_hashes = {
        name: hash_file(path)
        for name, path in sorted((input_paths or {}).items(), key=lambda item: item[0])
    }
    output_hashes = {
        name: hash_file(path)
        for name, path in sorted((output_paths or {}).items(), key=lambda item: item[0])
    }

    manifest = {
        "protocol": protocol,
        "timestamp_utc": timestamp_utc,
        "git_commit": get_git_commit(repo_root),
        "configs": config_hashes,
        "upstream": upstream,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": Path(sys.executable).name,
        },
        "packages": package_versions(("numpy", "pandas", "PyYAML")),
        "seed": int(seed),
        "split": _sanitise_split_for_manifest(split),
        "models": list(models),
        "model_metadata": _sanitise_model_metadata(model_metadata or {}),
        "inputs": input_hashes,
        "outputs": output_hashes,
    }
    if include_dirty:
        manifest["git_dirty"] = get_git_dirty(repo_root)
    return manifest


def write_run_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    """Write a deterministic JSON manifest."""

    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _sanitise_split_for_manifest(split: dict[str, Any]) -> dict[str, Any]:
    """Remove any field likely to contain participant identifiers."""

    blocked = {
        "train_indices",
        "test_indices",
        "train_participants",
        "test_participants",
        "participant_ids",
        "groups",
    }
    return {
        key: value
        for key, value in sorted(split.items(), key=lambda item: item[0])
        if key not in blocked
    }


def _sanitise_model_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Remove obvious participant identifiers from model metadata."""

    blocked = {
        "participant_ids",
        "participants",
        "train_participants",
        "test_participants",
        "fit_participant_ids",
    }
    sanitised: dict[str, Any] = {}
    for key, value in sorted(metadata.items(), key=lambda item: item[0]):
        if key in blocked:
            continue
        if isinstance(value, dict):
            sanitised[key] = _sanitise_model_metadata(value)
        elif isinstance(value, list):
            sanitised[key] = [
                _sanitise_model_metadata(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            sanitised[key] = value
    return sanitised


def _git_command_prefix(root: Path) -> list[str]:
    """Use command-local safe.directory for Windows sandbox/admin split users."""

    safe_path = root.resolve().as_posix()
    return ["git", "-c", f"safe.directory={safe_path}"]
