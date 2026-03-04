import json
from pathlib import Path

from vlm_redteam.graph.nodes.execute import execute_target_node
from vlm_redteam.graph.state import Candidate
from vlm_redteam.storage.event_log import EventLogger


def test_execute_target_node_fills_output_and_logs(tmp_path: Path) -> None:
    run_id = "run-exec"
    event_logger = EventLogger(run_id=run_id, runs_root=tmp_path / "runs")

    state = {
        "run_id": run_id,
        "task": {"goal": "original goal", "image_path": None},
        "round_idx": 0,
        "max_rounds": 1,
        "beam_width": 1,
        "per_branch_candidates": 2,
        "beam": [],
        "candidates": [
            Candidate(
                cand_id="c1",
                from_branch_id="b0",
                attack_name="a",
                params={"seed": 1},
                test_case={"jailbreak_prompt": "jb1", "jailbreak_image_path": None},
            ),
            Candidate(
                cand_id="c2",
                from_branch_id="b0",
                attack_name="a",
                params={"seed": 2},
                test_case={"jailbreak_prompt": "jb2", "jailbreak_image_path": None},
            ),
        ],
        "done": False,
        "best": None,
        "stats": {
            "event_logger": event_logger,
            "target_vllm_base_url": "",
            "target_vllm_model": "dummy-model",
            "temperature": 0.2,
            "max_tokens": 64,
            "enable_vision": False,
            "concurrency": 4,
            "runs_root": str(tmp_path / "runs"),
        },
    }

    out = execute_target_node(state)
    assert len(out["candidates"]) == 2
    assert out["candidates"][0].target_output.startswith("[dummy vllm output]")
    assert out["candidates"][1].target_output.startswith("[dummy vllm output]")

    events = event_logger.log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(events) == 2
    payload = [json.loads(line) for line in events]
    assert all(e["event_type"] == "TargetCalled" for e in payload)
    assert payload[0]["payload"]["cand_id"] == "c1"
    assert payload[1]["payload"]["cand_id"] == "c2"
