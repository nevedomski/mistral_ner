"""Advanced learning rate schedulers for NER fine-tuning."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from transformers import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from transformers import TrainingArguments

logger = logging.getLogger("mistral_ner")


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    training_args: TrainingArguments,
    num_training_steps: int,
    **scheduler_kwargs: Any,
) -> Any:
    """
    Create learning rate scheduler based on configuration.

    Args:
        optimizer: The optimizer to schedule
        scheduler_type: Type of scheduler ('linear', 'cosine', 'cosine_with_restarts', 'polynomial')
        training_args: Training arguments containing warmup configuration
        num_training_steps: Total number of training steps
        **scheduler_kwargs: Additional scheduler-specific parameters

    Returns:
        Learning rate scheduler
    """
    # Calculate warmup steps
    if training_args.warmup_ratio > 0:
        warmup_steps = int(training_args.warmup_ratio * num_training_steps)
    else:
        warmup_steps = training_args.warmup_steps if hasattr(training_args, "warmup_steps") else 0

    logger.info(
        f"Creating {scheduler_type} scheduler with {warmup_steps} warmup steps out of {num_training_steps} total steps"
    )

    if scheduler_type == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            **scheduler_kwargs,
        )
    elif scheduler_type == "cosine":
        num_cycles = scheduler_kwargs.pop("num_cycles", 0.5)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            **scheduler_kwargs,
        )
    elif scheduler_type == "cosine_with_restarts":
        num_cycles = scheduler_kwargs.pop("num_cycles", 1)
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
            **scheduler_kwargs,
        )
    elif scheduler_type == "polynomial":
        power = scheduler_kwargs.pop("power", 1.0)
        lr_end = scheduler_kwargs.pop("lr_end", 1e-7)
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            lr_end=lr_end,
            power=power,
            **scheduler_kwargs,
        )
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    logger.info(f"Successfully created {scheduler_type} scheduler")
    return scheduler


class LayerwiseLearningRateScheduler:
    """
    Implements layer-wise learning rate decay where earlier layers get smaller learning rates.

    This is particularly effective for transformer fine-tuning as earlier layers contain
    more general features while later layers are more task-specific.
    """

    def __init__(
        self,
        model: Any,
        base_lr: float,
        decay_factor: float = 0.9,
        num_layers: int | None = None,
    ) -> None:
        """
        Initialize layer-wise learning rate scheduler.

        Args:
            model: The model to apply layer-wise LR to
            base_lr: Base learning rate for the last layer
            decay_factor: Decay factor for earlier layers (0 < decay_factor < 1)
            num_layers: Number of layers (auto-detected if None)
        """
        self.model = model
        self.base_lr = base_lr
        self.decay_factor = decay_factor

        # Auto-detect number of layers if not provided
        if num_layers is None:
            self.num_layers = self._count_transformer_layers()
        else:
            self.num_layers = num_layers

        logger.info(f"Layer-wise LR scheduler: {self.num_layers} layers, decay_factor={decay_factor}")

    def _count_transformer_layers(self) -> int:
        """Count transformer layers in the model."""
        count = 0
        for name, _ in self.model.named_modules():
            if "layer" in name.lower() or "block" in name.lower():
                if any(x in name.lower() for x in ["attention", "mlp", "feed_forward"]):
                    continue
                count += 1
        return max(count, 12)  # Default to 12 if detection fails

    def get_layer_learning_rates(self) -> dict[str, float]:
        """
        Get learning rates for each layer.

        Returns:
            Dictionary mapping layer names to learning rates
        """
        layer_lrs = {}

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Determine layer index (simplified heuristic)
            layer_idx = self._get_layer_index(name)

            # Calculate learning rate (later layers get higher LR)
            lr_multiplier = self.decay_factor ** (self.num_layers - layer_idx - 1)
            layer_lr = self.base_lr * lr_multiplier

            layer_lrs[name] = layer_lr

        return layer_lrs

    def _get_layer_index(self, param_name: str) -> int:
        """Extract layer index from parameter name."""
        # This is a simplified heuristic - may need adjustment for different models
        import re

        # Look for patterns like "layer.0", "layers.0", "block.0", etc.
        patterns = [
            r"layer\.(\d+)",
            r"layers\.(\d+)",
            r"block\.(\d+)",
            r"h\.(\d+)",  # GPT-style
        ]

        for pattern in patterns:
            match = re.search(pattern, param_name)
            if match:
                return int(match.group(1))

        # Default to last layer if no pattern matches
        return self.num_layers - 1


def create_layerwise_optimizer_groups(
    model: Any,
    base_lr: float,
    weight_decay: float = 0.01,
    decay_factor: float = 0.9,
) -> list[dict[str, Any]]:
    """
    Create optimizer parameter groups with layer-wise learning rates.

    Args:
        model: The model
        base_lr: Base learning rate
        weight_decay: Weight decay value
        decay_factor: Learning rate decay factor for earlier layers

    Returns:
        List of parameter groups for optimizer
    """
    scheduler = LayerwiseLearningRateScheduler(model, base_lr, decay_factor)
    layer_lrs = scheduler.get_layer_learning_rates()

    # Group parameters by learning rate
    lr_groups: dict[float, list[Any]] = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        lr = layer_lrs.get(name, base_lr)
        if lr not in lr_groups:
            lr_groups[lr] = []
        lr_groups[lr].append(param)

    # Create parameter groups
    param_groups = []
    for lr, params in lr_groups.items():
        group = {
            "params": params,
            "lr": lr,
            "weight_decay": weight_decay,
        }
        param_groups.append(group)
        logger.info(f"Parameter group: lr={lr:.6f}, params={len(params)}")

    logger.info(f"Created {len(param_groups)} parameter groups with layer-wise learning rates")
    return param_groups


def create_advanced_scheduler_config(
    scheduler_type: str = "cosine",
    num_cycles: float = 0.5,
    power: float = 1.0,
    lr_end: float = 1e-7,
    **extra_kwargs: Any,
) -> dict[str, Any]:
    """
    Create advanced scheduler configuration.

    Args:
        scheduler_type: Type of scheduler
        num_cycles: Number of cycles for cosine schedulers
        power: Power for polynomial decay
        lr_end: End learning rate for polynomial decay
        **extra_kwargs: Additional configuration

    Returns:
        Scheduler configuration dictionary
    """
    config = {
        "type": scheduler_type,
        "num_cycles": num_cycles,
        "power": power,
        "lr_end": lr_end,
    }
    config.update(extra_kwargs)

    return config


# Predefined scheduler configurations
SCHEDULER_CONFIGS = {
    "cosine_aggressive": {
        "type": "cosine",
        "num_cycles": 1.0,  # More aggressive decay
    },
    "cosine_gentle": {
        "type": "cosine",
        "num_cycles": 0.25,  # Gentler decay
    },
    "cosine_restarts": {
        "type": "cosine_with_restarts",
        "num_cycles": 2,  # Two restart cycles
    },
    "polynomial_strong": {
        "type": "polynomial",
        "power": 2.0,  # Quadratic decay
        "lr_end": 1e-8,
    },
    "polynomial_gentle": {
        "type": "polynomial",
        "power": 0.5,  # Square root decay
        "lr_end": 1e-6,
    },
}


def get_scheduler_config(name: str) -> dict[str, Any]:
    """Get predefined scheduler configuration by name."""
    if name not in SCHEDULER_CONFIGS:
        available = list(SCHEDULER_CONFIGS.keys())
        raise ValueError(f"Unknown scheduler config '{name}'. Available: {available}")

    return SCHEDULER_CONFIGS[name].copy()
