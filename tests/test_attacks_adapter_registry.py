from dataclasses import dataclass
from random import Random

from vlm_redteam.attacks.adapter import AttackAdapter
from vlm_redteam.attacks.registry import AttackRegistry


@dataclass
class DummyTestCase:
    case_id: str
    jailbreak_prompt: str
    jailbreak_image_path: str
    original_prompt: str
    original_image_path: str


class DummyCfg:
    strength = 1


class DummyAttack:
    def __init__(self) -> None:
        self.cfg = DummyCfg()
        self.calls: list[dict] = []

    def generate_test_case(self, original_prompt, image_path, case_id, **kwargs):
        self.calls.append(
            {
                "original_prompt": original_prompt,
                "image_path": image_path,
                "case_id": case_id,
                "kwargs": kwargs,
                "cfg_strength": self.cfg.strength,
            }
        )
        return DummyTestCase(
            case_id=case_id,
            jailbreak_prompt=f"jb::{original_prompt}",
            jailbreak_image_path=image_path or "",
            original_prompt=original_prompt,
            original_image_path=image_path or "",
        )


def test_attack_adapter_overrides_cfg_and_restores() -> None:
    attack = DummyAttack()
    adapter = AttackAdapter(attack_obj=attack, attack_name="dummy")

    out = adapter.generate(
        original_prompt="hello",
        image_path="/tmp/in.png",
        case_id="c-1",
        params={"strength": 7},
    )

    assert out["case_id"] == "c-1"
    assert out["jailbreak_prompt"] == "jb::hello"
    assert attack.calls[0]["cfg_strength"] == 7
    assert attack.cfg.strength == 1


def test_attack_registry_basic_and_sampling() -> None:
    registry = AttackRegistry()
    adapter1 = AttackAdapter(attack_obj=DummyAttack(), attack_name="a1")
    adapter2 = AttackAdapter(attack_obj=DummyAttack(), attack_name="a2")

    registry.register("a1", adapter1, weight=3.0)
    registry.register("a2", adapter2, weight=1.0)

    listed = dict(registry.list_attacks())
    assert listed == {"a1": 3.0, "a2": 1.0}
    assert registry.get("a1") is adapter1

    sampled = registry.sample_attacks(k=2, rng=Random(0))
    assert len(sampled) == 2
    assert all(name in {"a1", "a2"} for name in sampled)
