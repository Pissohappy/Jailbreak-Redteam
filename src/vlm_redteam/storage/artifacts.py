"""Artifact storage helpers."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


class ArtifactStore:
    """Content-addressed local artifact store for run assets."""

    def __init__(self, runs_root: str | Path = "runs") -> None:
        self.runs_root = Path(runs_root)

    @staticmethod
    def _sha256_file(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as fp:
            for chunk in iter(lambda: fp.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def save_image(self, run_id: str, image_path: str | Path) -> tuple[str, str]:
        """Copy an image into the run artifact directory and deduplicate by sha256."""

        src = Path(image_path)
        if not src.exists() or not src.is_file():
            raise FileNotFoundError(f"Image not found: {src}")

        digest = self._sha256_file(src)
        artifacts_dir = self.runs_root / run_id / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        target = artifacts_dir / f"{digest}.png"
        if not target.exists():
            shutil.copy2(src, target)

        return str(target), digest
