"""Model configuration for Mistral-7B NER fine-tuning."""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator
from pydantic.dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for Mistral-7B model and LoRA fine-tuning parameters.

    This class manages model loading and LoRA (Low-Rank Adaptation) configuration
    for efficient fine-tuning of Mistral-7B for Named Entity Recognition tasks.
    Supports both 4-bit and 8-bit quantization for memory-efficient training.

    Attributes:
        model_name (str | None): HuggingFace model identifier (default: uses Mistral-7B-v0.3).

        load_in_8bit (bool): Enable 8-bit quantization (~10GB VRAM usage).
        load_in_4bit (bool): Enable 4-bit quantization with QLoRA (~6GB VRAM usage).
            Note: Only one quantization method should be enabled at a time.
            If both are False, uses full precision (~24GB VRAM usage).

        device_map (str): Device mapping strategy for model layers (default: 'auto').
        trust_remote_code (bool): Allow executing remote code from HuggingFace models.
        use_cache (bool): Whether to use KV cache during inference (disabled for training).

        lora_r (int): LoRA rank - size of decomposition matrices (default: 32).
        lora_alpha (int): LoRA scaling parameter (default: 64, typically 2x rank).
        lora_dropout (float): Dropout probability for LoRA layers (default: 0.1).
        lora_bias (str): How to handle biases - 'none', 'all', or 'lora_only'.
        target_modules (list[str] | str | None): Which modules to apply LoRA to.
            If None, will default to attention and MLP projections.
        task_type (str): Task type for the model (default: 'TOKEN_CLS' for NER).

    Note:
        For optimal performance with limited VRAM:
        - Use load_in_4bit=True for GPUs with 8-12GB VRAM
        - Use load_in_8bit=True for GPUs with 12-16GB VRAM
        - Use full precision only with 24GB+ VRAM
    """

    model_name: str | None = None
    # Quantization settings - Choose one:
    # - 8-bit: Better performance, ~10GB VRAM usage
    # - 4-bit: More memory efficient, ~6GB VRAM usage
    # - Both false: Full precision, ~24GB VRAM usage
    load_in_8bit: bool = True  # Set to true for 8-bit quantization - DEFAULT
    load_in_4bit: bool = False  # Set to true for 4-bit quantization (QLoRA)
    device_map: str = "auto"
    trust_remote_code: bool = True
    use_cache: bool = False

    # LoRA configuration - Enhanced for better capacity
    lora_r: int = 32
    lora_alpha: int = 64  # Increased proportionally (2x rank)
    lora_dropout: float = 0.1
    lora_bias: Literal["none", "all", "lora_only"] = "none"
    target_modules: list[str] | str | None = None
    task_type: Literal["TOKEN_CLS"] = "TOKEN_CLS"

    @model_validator(mode="after")
    def validate_quantization(self) -> ModelConfig:
        """Ensure only one quantization method is enabled at a time."""
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError(
                "Cannot enable both 8-bit and 4-bit quantization simultaneously. "
                "Please set either load_in_8bit=True OR load_in_4bit=True, not both."
            )
        return self
