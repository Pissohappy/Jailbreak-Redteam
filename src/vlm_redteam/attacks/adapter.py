"""Adapter for integrating existing BaseAttack/TestCase style attacks."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


class AttackAdapter:
    """Wraps an existing attack object with a unified dict-based interface."""

    def __init__(self, attack_obj: Any, attack_name: str) -> None:
        self.attack_obj = attack_obj
        self.attack_name = attack_name

    def _snapshot_and_override_cfg(self, params: dict[str, Any]) -> list[tuple[str, Any, bool]]:
        """Override cfg fields and return a restore plan.

        Restore plan entries: (field_name, old_value, field_existed_before).
        """

        if not params:
            return []

        cfg = getattr(self.attack_obj, "cfg", None)
        if cfg is None:
            return []

        restore_plan: list[tuple[str, Any, bool]] = []
        for key, value in params.items():
            if isinstance(cfg, dict):
                existed = key in cfg
                old_value = cfg.get(key)
                cfg[key] = value
            else:
                existed = hasattr(cfg, key)
                old_value = getattr(cfg, key, None)
                setattr(cfg, key, value)
            restore_plan.append((key, old_value, existed))
        return restore_plan

    def _restore_cfg(self, restore_plan: list[tuple[str, Any, bool]]) -> None:
        """Restore cfg after generation to prevent cross-candidate contamination."""

        if not restore_plan:
            return

        cfg = getattr(self.attack_obj, "cfg", None)
        if cfg is None:
            return

        for key, old_value, existed in restore_plan:
            if isinstance(cfg, dict):
                if existed:
                    cfg[key] = old_value
                else:
                    cfg.pop(key, None)
            else:
                if existed:
                    setattr(cfg, key, old_value)
                elif hasattr(cfg, key):
                    delattr(cfg, key)

    @staticmethod
    def _test_case_to_dict(test_case: Any) -> dict[str, Any]:
        """Extract required fields from TestCase object."""

        if is_dataclass(test_case):
            raw = asdict(test_case)
        elif isinstance(test_case, dict):
            raw = dict(test_case)
        else:
            raw = {
                "case_id": getattr(test_case, "test_case_id"),
                "jailbreak_prompt": getattr(test_case, "prompt"),
                "jailbreak_image_path": getattr(test_case, "image_path"),
                "original_prompt": getattr(test_case, "original_prompt"),
                "original_image_path": getattr(test_case, "original_image_path"),
            }

        return {
            "case_id": raw.get("test_case_id"),
            "jailbreak_prompt": raw.get("prompt"),
            "jailbreak_image_path": raw.get("image_path"),
            "original_prompt": raw.get("original_prompt"),
            "original_image_path": raw.get("original_image_path"),
        }

    def generate(
        self,
        original_prompt: str,
        image_path: str | None,
        case_id: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate one adapted test-case dict while safely overriding cfg params."""

        restore_plan = self._snapshot_and_override_cfg(params)
        try:
            test_case = self.attack_obj.generate_test_case(
                original_prompt=original_prompt,
                image_path=image_path,
                case_id=case_id,
                **params,
            )
        finally:
            self._restore_cfg(restore_plan)

        return self._test_case_to_dict(test_case)
