from vlm_redteam.graph.nodes.judge import judge_candidates
from vlm_redteam.models.judge_client import JudgeClient


def test_judge_disabled_returns_dummy() -> None:
    client = JudgeClient(base_url="", model="judge")
    out = client.judge_one(goal="g", jailbreak_prompt="p", target_output="o")
    assert out == {"success": False, "score": 0.0, "reason": "judge disabled"}


def test_parse_judge_output_json_and_fallback() -> None:
    client = JudgeClient(base_url="", model="judge")

    parsed_json = client.parse_judge_output('{"success": true, "score": 0.8, "reason": "ok"}')
    assert parsed_json == {"success": True, "score": 0.8, "reason": "ok"}

    parsed_text = client.parse_judge_output("Result: success=true; score=0.42; reason=close")
    assert parsed_text["success"] is True
    assert parsed_text["score"] == 0.42


def test_judge_candidates_populates_fields_when_disabled() -> None:
    state = {
        "run_id": "r1",
        "task": {"goal": "test-goal"},
        "round_idx": 0,
        "max_rounds": 1,
        "beam_width": 1,
        "per_branch_candidates": 1,
        "beam": [],
        "candidates": [
            {
                "cand_id": "c1",
                "from_branch_id": "b0",
                "attack_name": "a",
                "params": {},
                "test_case": {"jailbreak_prompt": "jb prompt"},
                "target_output": "model output",
            }
        ],
        "done": False,
        "best": None,
        "stats": {"judge_base_url": "", "judge_model": "judge"},
    }

    updated = judge_candidates(state)
    cand = updated["candidates"][0]
    assert cand["judge_success"] is False
    assert cand["judge_score"] == 0.0
    assert cand["judge_reason"] == "judge disabled"
