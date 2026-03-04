import json
from pathlib import Path

from vlm_redteam.attacks.adapter import AttackAdapter
from vlm_redteam.attacks.registry import AttackRegistry
from vlm_redteam.graph.nodes.expand import expand_node
from vlm_redteam.graph.state import Branch
from vlm_redteam.storage.artifacts import ArtifactStore
from vlm_redteam.storage.event_log import EventLogger


class DummyAttack:
    cfg: dict = {}

    def generate_test_case(self, original_prompt, image_path, case_id, **kwargs):
        return {
            "case_id": case_id,
            "jailbreak_prompt": f"jb::{original_prompt}",
            "jailbreak_image_path": image_path,
            "original_prompt": original_prompt,
            "original_image_path": image_path,
        }


def test_expand_generates_candidates_and_logs(tmp_path: Path) -> None:
    img = tmp_path / "in.png"
    img.write_bytes(b"fake image")

    registry = AttackRegistry()
    registry.register("dummy", AttackAdapter(DummyAttack(), "dummy"), weight=1.0)

    run_id = "run-expand"
    event_logger = EventLogger(run_id=run_id, runs_root=tmp_path / "runs")
    artifact_store = ArtifactStore(runs_root=tmp_path / "runs")

    state = {
        "run_id": run_id,
        "task": {"goal": "goal-text", "image_path": str(img)},
        "round_idx": 1,
        "max_rounds": 3,
        "beam_width": 1,
        "per_branch_candidates": 2,
        "beam": [Branch(branch_id="b0", parent_id=None, round_idx=1)],
        "candidates": [],
        "done": False,
        "best": None,
        "stats": {
            "seed": 123,
            "attack_registry": registry,
            "event_logger": event_logger,
            "artifact_store": artifact_store,
            "runs_root": str(tmp_path / "runs"),
        },
    }

    out = expand_node(state)
    assert len(out["candidates"]) == 2

    first = out["candidates"][0]
    assert first.test_case["jailbreak_image_path"].startswith(str(tmp_path / "runs" / run_id / "artifacts"))

    events = event_logger.log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(events) == 4
    types = [json.loads(line)["event_type"] for line in events]
    assert types.count("ActionProposed") == 2
    assert types.count("TestCaseGenerated") == 2
