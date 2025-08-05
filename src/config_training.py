"""Training configuration for Mistral NER fine-tuning."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field
from pydantic.dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration for training Mistral NER models with HuggingFace Transformers.

    This class manages all training hyperparameters and settings for fine-tuning
    Mistral-7B models on NER tasks. Supports various loss functions, learning rate
    schedulers, mixed precision training, and integration with Weights & Biases.

    Attributes:
        output_dir (str): Directory for saving model checkpoints during training.
        final_output_dir (str): Directory for saving the final merged model.
        num_train_epochs (int): Number of training epochs (default: 5).
        per_device_train_batch_size (int): Training batch size per GPU (default: 4).
        per_device_eval_batch_size (int): Evaluation batch size per GPU (default: 8).
        gradient_accumulation_steps (int): Steps to accumulate gradients (default: 8).
        gradient_checkpointing (bool): Enable gradient checkpointing for memory efficiency.

        optim (str): Optimizer to use (default: 'paged_adamw_32bit' for QLoRA).
        learning_rate (float): Initial learning rate (default: 0.0002).
        weight_decay (float): Weight decay for regularization (default: 0.001).
        warmup_ratio (float): Proportion of training for warmup (default: 0.03).
        max_grad_norm (float): Maximum gradient norm for clipping (default: 1.0).

        loss_type (str): Loss function - 'focal', 'cross_entropy', 'label_smoothing', or 'class_balanced'.
        focal_alpha (float | None): Class weights for focal loss (auto-computed if None).
        focal_gamma (float): Focusing parameter for focal loss (default: 2.0).
        label_smoothing (float): Smoothing factor for label smoothing loss (default: 0.1).
        class_balanced_beta (float): Re-weighting factor for class-balanced loss (default: 0.9999).

        lr_scheduler_type (str): Learning rate scheduler type - 'linear', 'cosine', etc.
        lr_scheduler_kwargs (dict): Additional scheduler parameters.

        eval_strategy (str): When to evaluate - 'epoch' or 'steps'.
        save_strategy (str): When to save checkpoints - 'epoch' or 'steps'.
        logging_steps (int): Log metrics every N steps (default: 10).
        save_total_limit (int): Maximum number of checkpoints to keep (default: 3).
        load_best_model_at_end (bool): Load best checkpoint after training.
        metric_for_best_model (str): Metric for selecting best model (default: 'eval_f1').

        fp16 (bool): Use 16-bit floating point precision (default: True).
        bf16 (bool): Use bfloat16 precision (requires Ampere GPUs).
        tf32 (bool): Use TensorFloat-32 on Ampere GPUs (default: True).

        early_stopping_patience (int): Epochs to wait before early stopping (default: 3).
        early_stopping_threshold (float): Minimum improvement threshold (default: 0.01).

        clear_cache_steps (int): Clear GPU cache every N steps (default: 50).
        resume_from_checkpoint (str | None): Path to resume training from.

        seed (int): Random seed for reproducibility (default: 42).
        data_seed (int): Random seed for data sampling (default: 42).

        push_to_hub (bool): Push model to HuggingFace Hub.
        hub_model_id (str | None): Model ID on HuggingFace Hub.
        merge_adapters_on_save (bool): Merge LoRA adapters when saving (default: True).

    Note:
        - Effective batch size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus
        - For 4-bit training, use 'paged_adamw_32bit' or 'paged_adamw_8bit' optimizers
        - Focal loss is recommended for imbalanced NER datasets
    """

    output_dir: str = "./mistral-ner-finetuned"
    final_output_dir: str = "./mistral-ner-finetuned-final"
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_32bit"
    learning_rate: float = 0.0002
    weight_decay: float = 0.001
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0

    # Loss function configuration
    focal_loss: bool = False  # Use focal loss if True
    focal_gamma: float = 2.0  # Focusing parameter for focal loss

    class_balanced_loss: bool = False  # Use class-balanced loss if True
    class_balanced_beta: float = 0.99  # Only used if class_balanced_loss=True
    class_balanced_weights: list[float] = Field(default_factory=list)  # Pre-computed class weights

    label_smoothing_loss: bool = False  # Use label smoothing if True
    label_smoothing: float = 0.1  # Only used if label_smoothing_loss=True

    # Learning rate scheduling
    lr_scheduler_type: Literal["linear", "cosine", "cosine_with_restarts", "polynomial"] = "cosine"
    lr_scheduler_kwargs: dict[str, Any] = Field(default_factory=dict)  # Additional scheduler parameters

    # Training strategy
    eval_strategy: Literal["epoch"] = "epoch"
    save_strategy: Literal["epoch"] = "epoch"
    logging_steps: int = 10
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: Literal["eval_f1"] = "eval_f1"
    greater_is_better: bool = True
    report_to: list[str] = Field(default_factory=lambda: ["wandb"])

    # Mixed precision training
    fp16: bool = True  # Use FP16 mixed precision (most GPUs)
    bf16: bool = False  # Use BF16 mixed precision (A100/H100 GPUs only)
    tf32: bool = True  # Enable TensorFloat-32 on Ampere GPUs for faster matmuls

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01

    # Memory management
    clear_cache_steps: int = 50

    # Resume from checkpoint
    resume_from_checkpoint: str | None = None

    # Seed
    seed: int = 42
    data_seed: int = 42

    # Hub
    push_to_hub: bool = False
    hub_model_id: str | None = None
    hub_strategy: Literal["every_save"] = "every_save"

    # Model saving
    merge_adapters_on_save: bool = True  # Merge LoRA adapters for deployment-ready model

    evaluate_on_test_set: bool = True  # Evaluate on test set after training
    compute_entity_level_metrics: bool = False  # Compute entity-level metrics like precision, recall, F1
