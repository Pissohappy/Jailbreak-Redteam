"""
Core abstract class definitions
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Type, TypeVar
import logging

from .data_formats import TestCase, ModelResponse, EvaluationResult
from dataclasses import fields, is_dataclass, MISSING

T = TypeVar("T")


class BaseComponent(ABC):
    """Component base class, provides common configuration and logging handling logic"""

    # Configuration class, subclasses can override
    CONFIG_CLASS: Optional[Type] = None

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize base component

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.name = self.__class__.__name__
        # Registry name injected by UnifiedRegistry (canonical component id, e.g. "llama_guard_3")
        self.registry_name = (
            config.get("_registry_name") if isinstance(config, dict) else None
        )

        # Initialize logger (using the module name where component is located)
        self.logger = logging.getLogger(self.__class__.__module__)

        # Automatically handle configuration conversion
        self._process_config(config)

    def _process_config(self, config: Dict[str, Any] = None) -> None:
        """Process configuration, convert to configuration object or keep as dictionary"""
        config = config or {}

        if self.CONFIG_CLASS is not None and is_dataclass(self.CONFIG_CLASS):
            # Convert configuration dictionary to configuration object
            cfg_dict = config
            allowed_fields = {f.name for f in fields(self.CONFIG_CLASS)}
            filtered = {k: v for k, v in cfg_dict.items() if k in allowed_fields}
            try:
                self.cfg = self.CONFIG_CLASS(**filtered)
            except TypeError as e:
                # Convert TypeError to ValueError for consistency
                raise ValueError(f"Invalid configuration: {e}") from e
        else:
            # No configuration class or not a dataclass, use dictionary directly
            self.cfg = config

        # Validate configuration
        self.validate_config()

    def validate_config(self) -> None:
        """Validate configuration, subclasses can override this method to add custom validation logic

        Default implementation checks required fields (if configuration class has required fields)
        """
        if hasattr(self, "cfg") and is_dataclass(self.cfg):
            # Check required fields
            for field in fields(self.cfg.__class__):
                if field.default is MISSING and field.default_factory is MISSING:
                    # This is a required field (no default value or default factory)
                    if (
                        not hasattr(self.cfg, field.name)
                        or getattr(self.cfg, field.name) is None
                    ):
                        raise ValueError(
                            f"Configuration field '{field.name}' is required but not set"
                        )
        elif hasattr(self, "cfg") and isinstance(self.cfg, dict):
            # For dictionary configuration, can add dictionary-specific validation
            pass
        # Subclasses can add more validation logic

    def _determine_load_model(self) -> bool:
        """Determine if local model needs to be loaded.

        Checks the load_model field in configuration (dict or dataclass).
        Returns False if load_model is not explicitly set.
        """
        if hasattr(self, "cfg"):
            config_obj = self.cfg
        else:
            config_obj = self.config

        if isinstance(config_obj, dict):
            return config_obj.get("load_model", False)
        elif hasattr(config_obj, "load_model"):
            return getattr(config_obj, "load_model", False)

        return False


class BaseAttack(BaseComponent, ABC):
    """Attack method base class (enhanced version)"""

    # Standard Metadata Keys
    META_KEY_ATTACK_METHOD = "attack_method"
    META_KEY_ORIGINAL_PROMPT = "original_prompt"
    META_KEY_JAILBREAK_PROMPT = "jailbreak_prompt"
    META_KEY_JAILBREAK_IMAGE = "jailbreak_image_path"
    META_KEY_TARGET_MODEL = "target_model"

    def __init__(
        self, config: Dict[str, Any] = None, output_image_dir: Optional[str] = None
    ):
        """
        Initialize attack method

        Args:
            config: Configuration dictionary, will be automatically converted to configuration object
            output_image_dir: Output image directory path
        """
        # Call parent class initialization (including logging and configuration handling)
        super().__init__(config)

        # Set output image directory
        from pathlib import Path

        self.output_image_dir = Path(output_image_dir) if output_image_dir else None

        # Determine if local model needs to be loaded
        self.load_model = self._determine_load_model()

    @abstractmethod
    def generate_test_case(
        self, original_prompt: str, image_path: str, case_id: str, **kwargs
    ) -> TestCase:
        """
        Generate test case

        Args:
            original_prompt: Original prompt
            image_path: Image path
            case_id: Test case ID
            **kwargs: Other parameters

        Returns:
            TestCase: Generated test case
        """
        pass

    def create_test_case(
        self,
        case_id: str,
        jailbreak_prompt: str,
        jailbreak_image_path: str,
        original_prompt: str,
        original_image_path: str = None,
        metadata: Dict[str, Any] = None,
        **kwargs,
    ) -> TestCase:
        """
        Create a standardized `TestCase` with populated metadata.

        Args:
            case_id: Unique test case ID
            jailbreak_prompt: The generated adversarial prompt
            jailbreak_image_path: Path to the generated adversarial image
            original_prompt: The original harmful prompt
            original_image_path: Path to the original image (optional)
            metadata: Additional metadata
            **kwargs: Other parameters to store in metadata

        Returns:
            TestCase object
        """
        attack_method = (
            getattr(self, "registry_name", None)
            or getattr(self, "name", None)
            or self.__class__.__name__.replace("Attack", "").lower()
        )

        base_metadata = {
            self.META_KEY_ATTACK_METHOD: attack_method,
            self.META_KEY_ORIGINAL_PROMPT: original_prompt,
            self.META_KEY_JAILBREAK_PROMPT: jailbreak_prompt,
            self.META_KEY_JAILBREAK_IMAGE: jailbreak_image_path,
        }

        if hasattr(self, "cfg") and hasattr(self.cfg, "target_model_name"):
            if self.cfg.target_model_name:
                base_metadata[self.META_KEY_TARGET_MODEL] = self.cfg.target_model_name

        if original_image_path:
            base_metadata["original_image_path"] = original_image_path

        # Merge additional metadata
        final_metadata = {**base_metadata, **(metadata or {}), **kwargs}

        return TestCase(
            test_case_id=str(case_id),
            prompt=jailbreak_prompt,
            image_path=jailbreak_image_path,
            metadata=final_metadata,
        )

    def __str__(self):
        return f"{self.name}(config={self.config})"


class BaseModel(BaseComponent, ABC):
    """Model base class (enhanced version)"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize model

        Args:
            config: Configuration dictionary, will be automatically converted to configuration object
        """
        # Call parent class initialization (including logging and configuration processing)
        super().__init__(config)

    @abstractmethod
    def generate_response(self, test_case: TestCase, **kwargs) -> ModelResponse:
        """
        Generate model response

        Args:
            test_case: Test case
            **kwargs: Other parameters

        Returns:
            ModelResponse: Model response
        """
        pass

    def __str__(self):
        return f"{self.name}(config={self.config})"


class BaseDefense(BaseComponent, ABC):
    """Defense method base class (enhanced version)"""

    # Metadata keys
    META_KEY_SHOULD_BLOCK = "should_return_default"
    META_KEY_BLOCKED = "defense_blocked"
    META_KEY_DEFAULT_RESP = "default_response"
    META_KEY_GENERATED_RESP = "defense_generated_response"
    META_KEY_GENERATED_RESP_USED = "defense_generated_response_used"

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize defense method

        Args:
            config: Configuration dictionary, will be automatically converted to configuration object
        """
        # Call parent class initialization (including logging and configuration handling)
        super().__init__(config)

        # Determine if local model needs to be loaded
        self.load_model = self._determine_load_model()

    @abstractmethod
    def apply_defense(self, test_case: TestCase, **kwargs) -> TestCase:
        """
        Apply defense method

        Args:
            test_case: Original test case
            **kwargs: Other parameters

        Returns:
            TestCase: Test case after applying defense
        """
        pass

    def create_defended_case(
        self,
        test_case: TestCase,
        defended_prompt: str = None,
        defended_image_path: str = None,
        metadata: Dict[str, Any] = None,
    ) -> TestCase:
        """
        Create a new `TestCase` after applying defense (common helper).

        This keeps downstream pipeline logic and metadata conventions consistent.
        """
        defense_method = (
            getattr(self, "registry_name", None)
            or getattr(self, "name", None)
            or self.__class__.__name__.replace("Defense", "").lower()
        )

        return TestCase(
            test_case_id=test_case.test_case_id,
            prompt=defended_prompt,
            image_path=defended_image_path,
            metadata={
                **(test_case.metadata or {}),
                "defense_method": defense_method,
                "defended_prompt": defended_prompt,
                "defended_image_path": defended_image_path,
                **(metadata or {}),
            },
        )

    def block_input(
        self,
        test_case: TestCase,
        default_response: str = "I'm sorry, I can't assist with that.",
        defended_prompt: str = None,
        defended_image_path: str = None,
        metadata: Dict[str, Any] = None,
    ) -> TestCase:
        """
        Mark the test case as 'blocked' by the defense.
        Pipeline will intercept this and return `default_response` immediately.
        """
        defended_case = self.create_defended_case(
            test_case=test_case,
            defended_prompt=defended_prompt if defended_prompt is not None else test_case.prompt,
            defended_image_path=defended_image_path
            if defended_image_path is not None
            else test_case.image_path,
            metadata={
                **(metadata or {}),
                self.META_KEY_SHOULD_BLOCK: True,
                self.META_KEY_BLOCKED: True,
                self.META_KEY_DEFAULT_RESP: default_response,
            },
        )
        return defended_case

    def reply_directly(
        self,
        test_case: TestCase,
        response_text: str,
        defended_prompt: str = None,
        defended_image_path: str = None,
        metadata: Dict[str, Any] = None,
    ) -> TestCase:
        """
        Defense generates the response directly (bypassing the target model).
        Pipeline will use `response_text` as the final output.
        """
        defended_case = self.create_defended_case(
            test_case=test_case,
            defended_prompt=defended_prompt if defended_prompt is not None else test_case.prompt,
            defended_image_path=defended_image_path
            if defended_image_path is not None
            else test_case.image_path,
            metadata={
                **(metadata or {}),
                self.META_KEY_GENERATED_RESP: response_text,
                self.META_KEY_GENERATED_RESP_USED: True,
            },
        )
        return defended_case

    def __str__(self):
        return f"{self.name}(config={self.config})"


class BaseEvaluator(BaseComponent, ABC):
    """Evaluator base class (enhanced version)"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize evaluator

        Args:
            config: Configuration dictionary, will be automatically converted to configuration object
        """
        # Call parent class initialization (including logging and configuration handling)
        super().__init__(config)

    @abstractmethod
    def evaluate_response(
        self, model_response: ModelResponse, **kwargs
    ) -> EvaluationResult:
        """
        Evaluate model response

        Args:
            model_response: Model response
            **kwargs: Other parameters

        Returns:
            EvaluationResult: Evaluation result
        """
        pass

    def __str__(self):
        return f"{self.name}(config={self.config})"


# Note: Old architecture's Registry class and global registry instance have been removed
# Please use UNIFIED_REGISTRY from core.unified_registry
