from vlm_redteam.graph.nodes.select import select_beam_node
from vlm_redteam.graph.state import Branch, Candidate


def _candidate(cand_id: str, score: float, success: bool = False, output: str = "out") -> Candidate:
    return Candidate(
        cand_id=cand_id,
        from_branch_id="b0",
        attack_name="a",
        params={"seed": 1},
        test_case={
            "jailbreak_prompt": "jb",
            "jailbreak_image_path": "/tmp/img.png",
            "jailbreak_image_hash": "h1",
        },
        target_output=output,
        judge_success=success,
        judge_score=score,
        judge_reason="r",
    )


def test_select_beam_dedup_and_topk_and_round_increment() -> None:
    state = {
        "run_id": "r",
        "task": {"goal": "g"},
        "round_idx": 0,
        "max_rounds": 3,
        "beam_width": 1,
        "per_branch_candidates": 2,
        "beam": [Branch(branch_id="b0", parent_id=None, round_idx=0, history=[{"old": 1}])],
        "candidates": [
            _candidate("c1", 0.2, output="same"),
            _candidate("c2", 0.9, output="same"),
        ],
        "done": False,
        "best": None,
        "stats": {},
    }

    out = select_beam_node(state)
    assert out["round_idx"] == 1
    assert out["done"] is False
    assert len(out["beam"]) == 1
    assert out["beam"][0].branch_id == "c2"
    assert out["beam"][0].aggregate_score == 0.9
    assert len(out["beam"][0].history) == 2


def test_select_beam_stops_on_success() -> None:
    state = {
        "run_id": "r",
        "task": {"goal": "g"},
        "round_idx": 0,
        "max_rounds": 5,
        "beam_width": 2,
        "per_branch_candidates": 2,
        "beam": [Branch(branch_id="b0", parent_id=None, round_idx=0)],
        "candidates": [_candidate("c1", 0.3, success=True)],
        "done": False,
        "best": None,
        "stats": {},
    }

    out = select_beam_node(state)
    assert out["done"] is True
    assert out["best"].branch_id == "c1"


def test_select_beam_stops_at_max_rounds() -> None:
    state = {
        "run_id": "r",
        "task": {"goal": "g"},
        "round_idx": 1,
        "max_rounds": 2,
        "beam_width": 2,
        "per_branch_candidates": 2,
        "beam": [Branch(branch_id="b0", parent_id=None, round_idx=1)],
        "candidates": [_candidate("c1", 0.4, success=False)],
        "done": False,
        "best": None,
        "stats": {},
    }

    out = select_beam_node(state)
    assert out["done"] is True
    assert out["best"].branch_id == "c1"
