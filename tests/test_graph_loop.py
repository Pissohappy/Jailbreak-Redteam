import pytest

pytest.importorskip("langgraph")

from vlm_redteam.attacks.adapter import AttackAdapter
from vlm_redteam.attacks.registry import AttackRegistry
from vlm_redteam.graph.build_graph import compile_graph
from vlm_redteam.graph.state import Branch


class DummyAttack:
    cfg: dict = {}

    def generate_test_case(self, original_prompt, image_path, case_id, **kwargs):
        return {
            "case_id": case_id,
            "jailbreak_prompt": f"jb::{original_prompt}",
            "jailbreak_image_path": None,
            "original_prompt": original_prompt,
            "original_image_path": image_path,
        }


def test_graph_loop_reaches_done_at_max_rounds() -> None:
    registry = AttackRegistry()
    registry.register("dummy", AttackAdapter(DummyAttack(), "dummy"), weight=1.0)

    state = {
        "run_id": "loop-run",
        "task": {"goal": "goal", "image_path": None},
        "round_idx": 0,
        "max_rounds": 1,
        "beam_width": 1,
        "per_branch_candidates": 1,
        "beam": [Branch(branch_id="b0", parent_id=None, round_idx=0)],
        "candidates": [],
        "done": False,
        "best": None,
        "stats": {
            "seed": 0,
            "attack_registry": registry,
            "runs_root": "runs",
            "target_vllm_base_url": "",
            "target_vllm_model": "dummy-model",
            "enable_vision": False,
            "temperature": 0.2,
            "max_tokens": 32,
            "concurrency": 4,
            "judge_base_url": "",
            "judge_model": "judge-model",
            "judge_concurrency": 4,
        },
    }

    app = compile_graph()
    out = app.invoke(state)
    assert out["done"] is True
    assert out["best"] is not None
    assert out["round_idx"] == 1
