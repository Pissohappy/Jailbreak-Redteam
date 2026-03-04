from pathlib import Path

from vlm_redteam.attacks.demo_figstep import DemoFigStepAttack
from vlm_redteam.cli.run import (
    RunConfig,
    _build_registry,
    _infer_strategy_name_from_entrypoint,
    _resolve_init_kwargs_for_attack,
)


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


def test_infer_strategy_name_from_entrypoint() -> None:
    assert _infer_strategy_name_from_entrypoint("attacks_strategy.figstep.attack:FigStepAttack") == "figstep"
    assert _infer_strategy_name_from_entrypoint("vlm_redteam.attacks.demo_figstep:DemoFigStepAttack") is None


def test_resolve_init_kwargs_auto_loads_strategy_params(tmp_path: Path) -> None:
    cfg_root = tmp_path / "configs" / "attacks"
    cfg_root.mkdir(parents=True, exist_ok=True)
    (cfg_root / "figstep.yaml").write_text(
        "name: figstep\nparameters:\n  steps: 3\n  font_size: 70\n",
        encoding="utf-8",
    )

    entry = "attacks_strategy.figstep.attack:FigStepAttack"
    cfg = RunConfig(
        run_id="run-y",
        target_vllm_base_url="",
        target_vllm_model="dummy-model",
        beam_width=1,
        per_branch_candidates=1,
        max_rounds=1,
        enabled_attacks=[entry],
        strategy_configs_dir=str(cfg_root),
        attack_init_kwargs={entry: {"config": {"steps": 5}}},
    )

    kwargs = _resolve_init_kwargs_for_attack(entry, cfg)
    assert kwargs["config"]["font_size"] == 70
    assert kwargs["config"]["steps"] == 5
    assert kwargs["output_image_dir"].endswith("runs/run-y/generated_images")
