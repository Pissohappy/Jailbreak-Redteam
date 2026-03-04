import json
from pathlib import Path

from vlm_redteam.graph.state import stable_hash
from vlm_redteam.storage.artifacts import ArtifactStore
from vlm_redteam.storage.event_log import EventLogger


def test_stable_hash_is_deterministic() -> None:
    obj1 = {"b": [1, 2], "a": {"x": "y"}}
    obj2 = {"a": {"x": "y"}, "b": [1, 2]}

    assert stable_hash(obj1) == stable_hash(obj2)


def test_artifact_store_deduplicates(tmp_path: Path) -> None:
    run_id = "run-1"
    source = tmp_path / "img.png"
    source.write_bytes(b"fake image bytes")

    store = ArtifactStore(runs_root=tmp_path / "runs")
    path1, digest1 = store.save_image(run_id, source)
    path2, digest2 = store.save_image(run_id, source)

    assert digest1 == digest2
    assert path1 == path2
    assert Path(path1).exists()


def test_event_logger_writes_jsonl(tmp_path: Path) -> None:
    logger = EventLogger(run_id="run-xyz", runs_root=tmp_path / "runs")
    logger.log_event("judge", {"score": 0.9})

    lines = logger.log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    payload = json.loads(lines[0])
    assert payload["run_id"] == "run-xyz"
    assert payload["event_type"] == "judge"
    assert payload["payload"]["score"] == 0.9
    assert "timestamp" in payload
