"""CLI entrypoint for running the redteam pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from pydantic import BaseModel

from vlm_redteam.graph.build_graph import compile_graph


class RunConfig(BaseModel):
    """Runtime config schema."""

    run_id: str
    target_vllm_base_url: str
    target_vllm_model: str
    judge_base_url: str | None = None
    beam_width: int
    per_branch_candidates: int
    max_rounds: int


def load_config(path: Path) -> RunConfig:
    """Load and validate YAML config."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return RunConfig.model_validate(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VLM redteam graph")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    _ = load_config(Path(args.config))
    print("Loaded config OK")

    _ = compile_graph()
    print("Graph compiled OK")


if __name__ == "__main__":
    main()
