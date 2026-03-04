"""
Unified data format definitions
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import uuid


def _convert_paths_to_strings(obj: Any) -> Any:
    """Recursively convert Path objects to strings for JSON serialization"""
    if isinstance(obj, Path):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: _convert_paths_to_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_paths_to_strings(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_convert_paths_to_strings(item) for item in obj)
    else:
        return obj


@dataclass
class TestCase:
    """Standardized test case format"""

    test_case_id: str
    prompt: str
    image_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.test_case_id:
            self.test_case_id = f"test_{uuid.uuid4().hex[:8]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        # Ensure image_path is a string
        image_path_str = str(self.image_path) if self.image_path is not None else None

        # Recursively convert Path objects in metadata
        metadata_clean = _convert_paths_to_strings(self.metadata)

        return {
            "test_case_id": self.test_case_id,
            "image_path": image_path_str,
            "prompt": self.prompt,
            "metadata": metadata_clean,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TestCase":
        """Create instance from dictionary"""
        return cls(**data)


@dataclass
class ModelResponse:
    """Standardized model response format"""

    test_case_id: str
    model_response: str
    model_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    reasoning_trace: Optional[str] = None
    final_answer: Optional[str] = None
    response_parse_status: Optional[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        # Recursively convert Path objects in metadata
        metadata_clean = _convert_paths_to_strings(self.metadata)

        return {
            "test_case_id": self.test_case_id,
            "model_response": self.model_response,
            "model_name": self.model_name,
            "metadata": metadata_clean,
            "reasoning_trace": self.reasoning_trace,
            "final_answer": self.final_answer,
            "response_parse_status": self.response_parse_status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelResponse":
        """Create instance from dictionary"""
        data_copy = data.copy()

        if "metadata" not in data_copy or data_copy["metadata"] is None:
            data_copy["metadata"] = {}

        data_copy.setdefault("reasoning_trace", None)
        data_copy.setdefault("final_answer", None)
        data_copy.setdefault("response_parse_status", None)

        return cls(**data_copy)


@dataclass
class EvaluationResult:
    """Standardized evaluation result format"""

    test_case_id: str
    judge_score: int
    judge_reason: str
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Optional fields for associating original data
    attack_method: Optional[str] = None
    original_prompt: Optional[str] = None
    jailbreak_prompt: Optional[str] = None
    image_path: Optional[str] = None
    model_response: Optional[str] = None
    model_name: Optional[str] = None
    defense_method: Optional[str] = None

    def __post_init__(self):
        # Add evaluation timestamp
        if "evaluation_time" not in self.metadata:
            self.metadata["evaluation_time"] = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        # Ensure image_path is a string
        image_path_str = str(self.image_path) if self.image_path is not None else None

        # Recursively convert Path objects in metadata
        metadata_clean = _convert_paths_to_strings(self.metadata)

        return {
            "test_case_id": self.test_case_id,
            "attack_method": self.attack_method,
            "original_prompt": self.original_prompt,
            "jailbreak_prompt": self.jailbreak_prompt,
            "image_path": image_path_str,
            "model_response": self.model_response,
            "model_name": self.model_name,
            "defense_method": self.defense_method,
            "judge_score": self.judge_score,
            "judge_reason": self.judge_reason,
            "success": self.success,
            "metadata": metadata_clean,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationResult":
        """Create instance from dictionary"""
        # Handle additional fields, such as evaluator_name
        data_copy = data.copy()

        # If evaluator_name exists, move it to metadata
        evaluator_name = data_copy.pop("evaluator_name", None)
        if evaluator_name is not None:
            if "metadata" not in data_copy:
                data_copy["metadata"] = {}
            data_copy["metadata"]["evaluator_name"] = evaluator_name

        return cls(**data_copy)


@dataclass
class PipelineConfig:
    """Unified Pipeline configuration class"""

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Test case generation configuration
    test_case_generation: Dict[str, Any] = field(default_factory=dict)

    # Model response generation configuration
    response_generation: Dict[str, Any] = field(default_factory=dict)

    # Evaluation configuration
    evaluation: Dict[str, Any] = field(default_factory=dict)

    # System configuration
    system: Dict[str, Any] = field(default_factory=dict)

    # Experiment configuration
    experiment: Dict[str, Any] = field(default_factory=dict)

    # Environment configuration
    environment: Dict[str, Any] = field(default_factory=dict)

    # Backward compatibility fields
    output_dir: str = "output/pipeline_results"
    max_workers: int = 4
    batch_size: int = 10  # Batch save size
    name: Optional[str] = None
    description: Optional[str] = None
    running_modes: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        result = {}

        # Add all configuration sections
        if self.metadata:
            result["metadata"] = self.metadata
        if self.test_case_generation:
            result["test_case_generation"] = self.test_case_generation
        if self.response_generation:
            result["response_generation"] = self.response_generation
        if self.evaluation:
            result["evaluation"] = self.evaluation
        if self.system:
            result["system"] = self.system
        if self.experiment:
            result["experiment"] = self.experiment
        if self.environment:
            result["environment"] = self.environment

        # Backward compatibility fields
        result.update(
            {
                "output_dir": self.output_dir,
                "max_workers": self.max_workers,
                "batch_size": self.batch_size,
            }
        )

        # Optional fields
        if self.name:
            result["name"] = self.name
        if self.description:
            result["description"] = self.description
        if self.running_modes:
            result["running_modes"] = self.running_modes

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create instance from dictionary"""
        # Extract all supported fields
        known_fields = {
            "metadata",
            "test_case_generation",
            "response_generation",
            "evaluation",
            "system",
            "experiment",
            "environment",
            "output_dir",
            "max_workers",
            "batch_size",
            "name",
            "description",
            "running_modes",
        }

        # Filter out known fields
        filtered_data = {k: v for k, v in data.items() if k in known_fields}

        return cls(**filtered_data)
