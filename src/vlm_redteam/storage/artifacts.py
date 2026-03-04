"""Artifact storage placeholder."""

from pathlib import Path


def ensure_artifact_dir(path: str | Path) -> Path:
    """Create artifact directory if needed."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
