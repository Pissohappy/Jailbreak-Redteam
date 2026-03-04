"""Append-only JSONL event logger."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EventLogger:
    """Writes timestamped JSON events for a run."""

    def __init__(self, run_id: str, runs_root: str | Path = "runs") -> None:
        self.run_id = run_id
        self.log_path = Path(runs_root) / run_id / "events.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event_type: str, payload_dict: dict[str, Any]) -> None:
        """Append one JSON event to the run log."""

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "event_type": event_type,
            "payload": payload_dict,
        }
        with self.log_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
