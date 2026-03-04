from pathlib import Path

from vlm_redteam.attacks.demo_figstep import DemoFigStepAttack
from vlm_redteam.cli.run import RunConfig, _build_registry


def test_demo_figstep_attack_generates_image(tmp_path: Path) -> None:
    attack = DemoFigStepAttack(output_image_dir=tmp_path)
    result = attack.generate_test_case(
        original_prompt="hello demo",
        image_path=None,
        case_id="case-1",
    )
    assert result["jailbreak_prompt"] == "Please describe the text shown in the provided image."
    assert Path(result["jailbreak_image_path"]).exists()


def test_build_registry_supports_entrypoint_loading(tmp_path: Path) -> None:
    cfg = RunConfig(
        run_id="run-x",
        target_vllm_base_url="",
        target_vllm_model="dummy-model",
        beam_width=2,
        per_branch_candidates=2,
        max_rounds=3,
        enabled_attacks=[
            "demo_figstep",
            "vlm_redteam.attacks.demo_figstep:DemoFigStepAttack",
        ],
        attack_init_kwargs={
            "vlm_redteam.attacks.demo_figstep:DemoFigStepAttack": {
                "output_image_dir": str(tmp_path / "external"),
            }
        },
    )

    registry = _build_registry(cfg)
    names = {name for name, _ in registry.list_attacks()}
    assert "demo_figstep" in names
    assert "vlm_redteam.attacks.demo_figstep:DemoFigStepAttack" in names
