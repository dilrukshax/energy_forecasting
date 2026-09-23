"""Experiment tracking and run versioning.

"Deployed ML models are part code, part data", so a result is only reproducible if you can say
*which* code and *which* data produced it. This module writes a manifest per run recording:

* a run id and timestamp,
* the git commit and whether the tree was dirty,
* a SHA-256 of the raw dataset,
* the full resolved configuration,
* library versions,
* the metrics and the selected features.

Each run gets its own directory under ``reports/runs/``, so results accumulate instead of
overwriting each other and two runs can be diffed. This is deliberately a flat-file
implementation rather than MLflow or W&B: it has no service to stand up, it works in CI, and
the manifest is a plain JSON file that a later job can compare against. Swapping in a hosted
tracker later means changing this module only.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from energy_forecast.config import Config, project_root
from energy_forecast.utils.logging import get_logger

logger = get_logger(__name__)


def file_digest(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 of a file, read in chunks so a large CSV is not loaded into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_digest(config: Config) -> str:
    """Return a stable SHA-256 over the resolved configuration.

    Two runs with the same config digest and the same data digest should produce the same
    result; if they do not, the difference is in the code or in an unseeded source of
    randomness, which narrows the search considerably.
    """
    payload = json.dumps(_config_payload(config), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _config_payload(config: Config) -> dict[str, Any]:
    """Return the configuration as a plain nested dictionary."""
    return {field_name: getattr(config, field_name)
            for field_name in Config.__dataclass_fields__}


def git_state() -> dict[str, str | None]:
    """Return the current commit and whether the working tree has uncommitted changes.

    Degrades gracefully: outside a git checkout, or without git installed, the fields are
    ``None`` rather than an exception. A run that cannot be traced to a commit is still worth
    recording - it just has to say so.
    """
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(args, cwd=project_root(),
                                           stderr=subprocess.DEVNULL, text=True).strip()
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            return None

    commit = run("git", "rev-parse", "HEAD")
    status = run("git", "status", "--porcelain")
    return {
        "commit": commit,
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else str(bool(status)),
    }


def environment() -> dict[str, str]:
    """Record the library versions a result depends on."""
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for module_name in ("numpy", "pandas", "sklearn", "tensorflow"):
        try:
            module = __import__(module_name)
            versions[module_name] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[module_name] = "not installed"
    return versions


@dataclass
class RunManifest:
    """Everything needed to reproduce and audit a single run."""

    run_id: str
    started_at: str
    config_digest: str
    data_digest: str
    data_path: str
    git: dict[str, str | None] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    data_quality: dict[str, Any] = field(default_factory=dict)
    selected_features: list[str] = field(default_factory=list)
    metrics: list[dict[str, Any]] = field(default_factory=list)
    trials: list[dict[str, Any]] = field(default_factory=list)
    best_model: str | None = None
    finished_at: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)


def new_run(config: Config, notes: str | None = None) -> RunManifest:
    """Open a manifest for a run that is about to start.

    Args:
        config: The resolved configuration the run will use.
        notes: Optional free-text description, e.g. what is being tried.

    Returns:
        A :class:`RunManifest` with the provenance fields populated.

    """
    started = datetime.now(UTC)
    data_path = config.raw_path
    manifest = RunManifest(
        run_id=started.strftime("%Y%m%dT%H%M%SZ"),
        started_at=started.isoformat(),
        config_digest=config_digest(config)[:16],
        data_digest=file_digest(data_path)[:16] if data_path.is_file() else "missing",
        data_path=str(data_path),
        git=git_state(),
        environment=environment(),
        config=_config_payload(config),
        notes=notes,
    )
    logger.info("run %s | config %s | data %s | commit %s",
                manifest.run_id, manifest.config_digest, manifest.data_digest,
                (manifest.git.get("commit") or "unknown")[:8])
    return manifest


def save_run(manifest: RunManifest, config: Config) -> Path:
    """Write the manifest to ``experiments/runs/<run_id>/manifest.json``.

    Also refreshes ``runs/latest.json`` so a downstream job has a stable path to read, while
    the per-run directories keep the history. Falls back to the reports directory when no
    experiments directory is configured.
    """
    manifest.finished_at = datetime.now(UTC).isoformat()

    base = config.outputs.get("experiments_dir") or config.outputs["reports_dir"]
    runs_dir = config.path(base) / "runs"
    run_dir = runs_dir / manifest.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    path = run_dir / "manifest.json"
    payload = manifest.to_dict()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)

    with open(runs_dir / "latest.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)

    logger.info("run manifest written to %s", path)
    return path


def compare_runs(base_dir: Path, metric: str = "mae") -> Any:
    """Return a table of every recorded run, best first.

    The point of accumulating manifests: "is the retrain better than what is deployed?" should
    be answerable by reading files, not by remembering.
    """
    import pandas as pd

    rows = []
    for manifest_path in sorted((base_dir / "runs").glob("*/manifest.json")):
        with open(manifest_path, encoding="utf-8") as handle:
            payload = json.load(handle)
        best = payload.get("best_model")
        scores = {m["model"]: m for m in payload.get("metrics", [])}
        rows.append({
            "run_id": payload.get("run_id"),
            "best_model": best,
            metric: scores.get(best, {}).get(metric),
            "config_digest": payload.get("config_digest"),
            "data_digest": payload.get("data_digest"),
            "commit": (payload.get("git", {}).get("commit") or "")[:8],
            "dirty": payload.get("git", {}).get("dirty"),
        })
    if not rows:
        return pd.DataFrame(columns=["run_id", "best_model", metric])
    return pd.DataFrame(rows).sort_values(metric, na_position="last").set_index("run_id")
