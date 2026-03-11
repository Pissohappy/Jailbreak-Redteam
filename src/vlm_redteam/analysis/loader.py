"""Data loader for experiment results."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AttackEvent:
    """Event record for an attack attempt."""

    event_type: str
    timestamp: str
    round_idx: int
    branch_id: str
    attack_name: str
    case_id: str | None = None
    strategy: str | None = None
    success: bool | None = None  # Whether this attack led to success
    score: float | None = None  # Score of this attack


@dataclass
class SampleResult:
    """Result for a single sample."""

    sample_id: int | str
    run_id: str
    goal: str
    image_path: str
    main_category: str
    subcategory: str
    success: bool
    aggregate_score: float | None
    error: str | None = None

    # Additional data from summary.json
    total_candidates: int | None = None
    best_branch_path: list[str] | None = None
    round_topk_scores: list[dict] | None = None

    # Attack events from events.jsonl
    attack_events: list[AttackEvent] = field(default_factory=list)


@dataclass
class ExperimentRun:
    """A single experiment run with timestamp."""

    experiment_name: str
    timestamp: str
    config_path: str
    dataset_path: str
    total_samples: int
    successful_samples: int
    success_rate: float
    results: list[SampleResult] = field(default_factory=list)


class ExperimentLoader:
    """Load and parse experiment results from directory structure."""

    def __init__(self, runs_root: str | Path = "runs"):
        self.runs_root = Path(runs_root)
        self._dataset_cache: dict[int | str, dict] = {}  # Cache for dataset lookup by id

    def load_dataset(self, dataset_path: str | Path) -> None:
        """Load a dataset file for category enrichment.

        The dataset should be a JSON file with items containing 'id', 'main_category', and 'subcategory'.

        Args:
            dataset_path: Path to the dataset JSON file
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        with open(dataset_path) as f:
            data = json.load(f)

        # Build lookup cache by id
        for item in data:
            item_id = item.get("id")
            if item_id is not None:
                self._dataset_cache[item_id] = item

        print(f"Loaded {len(self._dataset_cache)} samples from dataset for category enrichment")

    def list_experiments(self) -> list[str]:
        """List all experiment names in runs directory."""
        experiments = []
        if not self.runs_root.exists():
            return experiments

        for exp_dir in self.runs_root.iterdir():
            if exp_dir.is_dir() and not exp_dir.name.startswith("."):
                # Check if it has timestamp subdirectories (new format)
                has_timestamps = any(
                    d.is_dir() and self._is_timestamp(d.name)
                    for d in exp_dir.iterdir()
                )
                if has_timestamps:
                    experiments.append(exp_dir.name)
                # Also check for old format (batch_results.json in the directory)
                elif (exp_dir / "batch_results.json").exists():
                    experiments.append(exp_dir.name)

        return sorted(experiments)

    def list_runs(self, experiment_name: str) -> list[str]:
        """List all timestamp runs for an experiment."""
        exp_dir = self.runs_root / experiment_name
        if not exp_dir.exists():
            return []

        runs = []
        for run_dir in exp_dir.iterdir():
            if run_dir.is_dir() and self._is_timestamp(run_dir.name):
                runs.append(run_dir.name)

        # If no timestamp runs, check for old format (batch_results.json in experiment dir)
        if not runs and (exp_dir / "batch_results.json").exists():
            runs.append("")  # Empty string indicates old format

        return sorted(runs)

    def load_run(self, experiment_name: str, timestamp: str) -> ExperimentRun | None:
        """Load a specific experiment run."""
        # Handle old format (empty timestamp)
        if not timestamp:
            exp_dir = self.runs_root / experiment_name
            batch_results_path = exp_dir / "batch_results.json"
            if batch_results_path.exists():
                return self._load_from_batch_results(batch_results_path)
            return None

        # New format with timestamp
        run_dir = self.runs_root / experiment_name / timestamp
        if not run_dir.exists():
            return None

        # Load batch summary
        batch_summary_path = run_dir / "batch_summary.json"
        if batch_summary_path.exists():
            return self._load_from_batch_summary(batch_summary_path, run_dir)

        return None

    def load_latest_run(self, experiment_name: str) -> ExperimentRun | None:
        """Load the latest run for an experiment."""
        runs = self.list_runs(experiment_name)
        if not runs:
            return None

        return self.load_run(experiment_name, runs[-1])

    def load_all_runs(self, experiment_name: str) -> list[ExperimentRun]:
        """Load all runs for an experiment."""
        runs = []
        for timestamp in self.list_runs(experiment_name):
            run = self.load_run(experiment_name, timestamp)
            if run:
                runs.append(run)
        return runs

    def _is_timestamp(self, name: str) -> bool:
        """Check if directory name is a timestamp (YYYYMMDD_HHMMSS)."""
        import re
        return bool(re.match(r"^\d{8}_\d{6}$", name))

    def _enrich_with_dataset(self, result: SampleResult) -> None:
        """Enrich a sample result with category info from loaded dataset."""
        if result.sample_id in self._dataset_cache:
            item = self._dataset_cache[result.sample_id]
            if not result.main_category:
                result.main_category = item.get("main_category", "")
            if not result.subcategory:
                result.subcategory = item.get("subcategory", "")

    def _load_from_batch_summary(self, path: Path, run_dir: Path) -> ExperimentRun:
        """Load experiment run from batch_summary.json (new format)."""
        with open(path) as f:
            data = json.load(f)

        results = []
        for r in data.get("results", []):
            result = SampleResult(
                sample_id=r.get("sample_id", 0),
                run_id=r.get("run_id", ""),
                goal=r.get("goal", ""),
                image_path=r.get("image_path", ""),
                main_category=r.get("main_category", ""),
                subcategory=r.get("subcategory", ""),
                success=r.get("success", False),
                aggregate_score=r.get("aggregate_score"),
                error=r.get("error"),
            )
            # Enrich with dataset if category info is missing
            if not result.main_category or not result.subcategory:
                self._enrich_with_dataset(result)
            results.append(result)

        return ExperimentRun(
            experiment_name=data.get("experiment_name", ""),
            timestamp=data.get("timestamp", ""),
            config_path=data.get("config_path", ""),
            dataset_path=data.get("dataset_path", ""),
            total_samples=data.get("total_samples", len(results)),
            successful_samples=data.get("successful_samples", 0),
            success_rate=data.get("success_rate", 0.0),
            results=results,
        )

    def _load_from_batch_results(self, path: Path) -> ExperimentRun:
        """Load experiment run from batch_results.json (old format)."""
        with open(path) as f:
            results_data = json.load(f)

        results = []
        for r in results_data:
            result = SampleResult(
                sample_id=r.get("sample_id", 0),
                run_id=r.get("run_id", ""),
                goal=r.get("goal", ""),
                image_path=r.get("image_path", ""),
                main_category=r.get("main_category", ""),
                subcategory=r.get("subcategory", ""),
                success=r.get("success", False),
                aggregate_score=r.get("aggregate_score"),
                error=r.get("error"),
            )
            # Enrich with dataset if category info is missing
            if not result.main_category or not result.subcategory:
                self._enrich_with_dataset(result)
            results.append(result)

        # Extract experiment name from run_id pattern
        first_run_id = results[0].run_id if results else ""
        experiment_name = first_run_id.rsplit("_sample_", 1)[0] if "_sample_" in first_run_id else path.parent.name

        success_count = sum(1 for r in results if r.success)

        return ExperimentRun(
            experiment_name=experiment_name,
            timestamp="",
            config_path="",
            dataset_path="",
            total_samples=len(results),
            successful_samples=success_count,
            success_rate=success_count / len(results) if results else 0.0,
            results=results,
        )

    def load_with_sample_details(self, experiment_name: str, timestamp: str) -> ExperimentRun | None:
        """Load experiment run with detailed sample information from summary.json files."""
        run = self.load_run(experiment_name, timestamp)
        if not run:
            return None

        run_dir = self.runs_root / experiment_name / timestamp
        samples_dir = run_dir / "samples"

        if not samples_dir.exists():
            return run

        # Load additional details from each sample's summary.json
        for result in run.results:
            sample_dir = samples_dir / f"sample_{result.sample_id}"
            summary_path = sample_dir / "summary.json"

            if summary_path.exists():
                with open(summary_path) as f:
                    summary = json.load(f)

                result.total_candidates = summary.get("total_candidates")
                result.best_branch_path = summary.get("best_branch_path")
                result.round_topk_scores = summary.get("round_topk_scores")

        return run

    def load_with_attack_events(self, experiment_name: str, timestamp: str) -> ExperimentRun | None:
        """Load experiment run with attack event details from events.jsonl files."""
        run = self.load_with_sample_details(experiment_name, timestamp)
        if not run:
            return None

        run_dir = self.runs_root / experiment_name / timestamp
        samples_dir = run_dir / "samples"

        if not samples_dir.exists():
            return run

        # Load attack events from each sample's events.jsonl
        for result in run.results:
            sample_dir = samples_dir / f"sample_{result.sample_id}"
            events_path = sample_dir / "events.jsonl"

            if events_path.exists():
                result.attack_events = self._load_attack_events(events_path, result.success)

        return run

    def _load_attack_events(self, events_path: Path, sample_success: bool) -> list[AttackEvent]:
        """Load attack events from events.jsonl file."""
        attack_proposals = {}  # case_id -> AttackEvent (partial)
        current_round = 0  # Track current round from LLMGuidedSelection events

        with open(events_path) as f:
            for line in f:
                if not line.strip():
                    continue
                event = json.loads(line.strip())

                event_type = event.get("event_type", "")
                payload = event.get("payload", {})
                timestamp = event.get("timestamp", "")

                if event_type == "LLMGuidedSelection":
                    # Update current round from LLMGuidedSelection event
                    current_round = payload.get("round_idx", current_round)

                elif event_type == "ActionProposed":
                    # Record attack proposal with current round
                    case_id = payload.get("case_id", "")
                    attack_event = AttackEvent(
                        event_type=event_type,
                        timestamp=timestamp,
                        round_idx=current_round + 1,  # Round is 1-indexed for display
                        branch_id=payload.get("branch_id", ""),
                        attack_name=payload.get("attack_name", ""),
                        case_id=case_id,
                        strategy=payload.get("strategy", ""),
                    )
                    attack_proposals[case_id] = attack_event

                elif event_type == "TargetCalled":
                    # Target was called, mark this attack as executed
                    case_id = payload.get("cand_id", "")
                    if case_id in attack_proposals:
                        attack_proposals[case_id].success = False  # Will be updated if this leads to success

        # Determine which attack led to success (if sample was successful)
        if sample_success and attack_proposals:
            # If sample is successful, the last attack is typically the successful one
            case_ids = list(attack_proposals.keys())
            if case_ids:
                # Mark the last attack as successful
                last_case_id = case_ids[-1]
                attack_proposals[last_case_id].success = True

        return list(attack_proposals.values())
