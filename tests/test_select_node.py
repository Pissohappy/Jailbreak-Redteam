from vlm_redteam.graph.nodes.select import select_beam_node


def test_select_beam_dedup_topk_and_success_termination() -> None:
    state = {
        "run_id": "r1",
        "task": {"goal": "goal"},
        "round_idx": 0,
        "max_rounds": 5,
        "beam_width": 2,
        "per_branch_candidates": 3,
        "beam": [],
        "candidates": [
            {
                "cand_id": "c1",
                "from_branch_id": "b0",
                "test_case": {"jailbreak_prompt": "p", "jailbreak_image_hash": "ih"},
                "target_output": "o",
                "judge_success": False,
                "judge_score": 0.5,
            },
            {
                "cand_id": "c2",
                "from_branch_id": "b0",
                "test_case": {"jailbreak_prompt": "p", "jailbreak_image_hash": "ih"},
                "target_output": "o",
                "judge_success": True,
                "judge_score": 0.8,
            },
            {
                "cand_id": "c3",
                "from_branch_id": "b0",
                "test_case": {"jailbreak_prompt": "p2", "jailbreak_image_hash": "ih2"},
                "target_output": "o2",
                "judge_success": False,
                "judge_score": 0.7,
            },
        ],
        "done": False,
        "best": None,
        "stats": {},
    }

    out = select_beam_node(state)
    assert out["done"] is True
    assert out["best"] is not None
    assert out["best"].judge_success is True
    assert out["round_idx"] == 1
    assert len(out["beam"]) == 2
    # c1/c2 dedupe by signature, keep higher score branch
    assert out["beam"][0].aggregate_score == 0.8


def test_select_beam_max_round_termination() -> None:
    state = {
        "run_id": "r2",
        "task": {"goal": "goal"},
        "round_idx": 1,
        "max_rounds": 2,
        "beam_width": 1,
        "per_branch_candidates": 1,
        "beam": [],
        "candidates": [
            {
                "cand_id": "c1",
                "from_branch_id": "b0",
                "test_case": {"jailbreak_prompt": "p", "jailbreak_image_hash": "ih"},
                "target_output": "o",
                "judge_success": False,
                "judge_score": 0.3,
            }
        ],
        "done": False,
        "best": None,
        "stats": {},
    }

    out = select_beam_node(state)
    assert out["done"] is True
    assert out["best"] is not None
    assert out["best"].judge_success is False
    assert out["best"].aggregate_score == 0.3


def test_select_beam_history_strategy_none_only_keeps_current_round() -> None:
    state = {
        "run_id": "r3",
        "task": {"goal": "goal"},
        "round_idx": 0,
        "max_rounds": 2,
        "beam_width": 1,
        "per_branch_candidates": 1,
        "beam": [{"branch_id": "b0", "history": [{"user_text": "old", "target_output": "old_resp"}]}],
        "candidates": [
            {
                "cand_id": "c1",
                "from_branch_id": "b0",
                "test_case": {"jailbreak_prompt": "new", "jailbreak_image_hash": "ih"},
                "target_output": "new_resp",
                "judge_success": False,
                "judge_score": 0.3,
            }
        ],
        "done": False,
        "best": None,
        "stats": {"history_strategy": "none"},
    }

    out = select_beam_node(state)
    assert len(out["beam"][0].history) == 1
    assert out["beam"][0].history[0]["user_text"] == "new"
