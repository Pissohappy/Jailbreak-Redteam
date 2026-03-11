from vlm_redteam.attacks.adapter import AttackAdapter
from vlm_redteam.attacks.registry import AttackRegistry
from vlm_redteam.graph.nodes.expand_strategies.base import ExpandContext
from vlm_redteam.graph.nodes.expand_strategies.llm_guided import LLMGuidedStrategy


class _DummyAttack:
    def generate_test_case(self, **kwargs):
        return {}


class _RecordingClient:
    model = "fake"

    def __init__(self) -> None:
        self.messages = None

    def generate_one_with_messages(self, messages, temperature, max_tokens):
        self.messages = messages
        return '{"selected_attacks": [{"name": "a1", "reason": "ok"}]}'


class _FallbackOnlyClient:
    model = "fake"

    def __init__(self) -> None:
        self.user_text = None

    def generate_one(self, user_text, image_path, temperature, max_tokens, history=None):
        self.user_text = user_text
        return '{"selected_attacks": [{"name": "a1", "reason": "ok"}]}'


def _build_registry() -> AttackRegistry:
    registry = AttackRegistry()
    registry.register("a1", AttackAdapter(_DummyAttack(), "a1"), weight=1.0)
    return registry


def test_llm_guided_prefers_system_and_user_messages() -> None:
    client = _RecordingClient()
    strategy = LLMGuidedStrategy(llm_client=client)

    selected = strategy.select_attacks(
        ExpandContext(
            run_id="r1",
            round_idx=1,
            branch={"history": []},
            original_prompt="goal",
            original_image_path=None,
            registry=_build_registry(),
            per_branch_candidates=1,
            seed=123,
            stats={},
        )
    )

    assert selected == [("a1", {"llm_reason": "ok"})]
    assert client.messages is not None
    assert client.messages[0]["role"] == "system"
    assert client.messages[1]["role"] == "user"


def test_llm_guided_falls_back_for_legacy_client_without_message_api() -> None:
    client = _FallbackOnlyClient()
    strategy = LLMGuidedStrategy(llm_client=client)

    selected = strategy.select_attacks(
        ExpandContext(
            run_id="r1",
            round_idx=1,
            branch={"history": []},
            original_prompt="goal",
            original_image_path=None,
            registry=_build_registry(),
            per_branch_candidates=1,
            seed=123,
            stats={},
        )
    )

    assert selected == [("a1", {"llm_reason": "ok"})]
    assert client.user_text is not None
    assert "## Target Goal" in client.user_text
