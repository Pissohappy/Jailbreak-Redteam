"""Event log storage placeholder."""

from datetime import datetime, UTC
from pathlib import Path


def append_event(log_path: str | Path, message: str) -> None:
    """Append a timestamped event line to a local log file."""
    ts = datetime.now(UTC).isoformat()
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(log_path).open("a", encoding="utf-8") as fp:
        fp.write(f"{ts}\t{message}\n")
