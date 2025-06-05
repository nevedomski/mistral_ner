"""Training loop and utilities for Mistral NER fine-tuning."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import wandb
from transformers import (
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.integrations import WandbCallback

from datasets import Dataset

from .evaluation import compute_metrics_factory, load_seqeval_metric
from .losses import compute_class_frequencies, create_loss_function
from .schedulers import create_scheduler
from .utils import check_gpu_memory, clear_gpu_cache, detect_mixed_precision_support

if TYPE_CHECKING:
    from transformers import DataCollatorForTokenClassification

    from .config import Config

logger = logging.getLogger("mistral_ner")


class MemoryCallback(TrainerCallback):
    """Callback to monitor and manage GPU memory during training."""

    def __init__(self, clear_cache_steps: int = 50):
        self.clear_cache_steps = clear_cache_steps

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any) -> None:
        """Clear cache periodically to prevent OOM."""
        if state.global_step % self.clear_cache_steps == 0:
            clear_gpu_cache()
            memory_stats = check_gpu_memory()
            logger.debug(f"Step {state.global_step} - GPU Memory: {memory_stats}")

    def on_epoch_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        """Clear cache at end of epoch."""
        clear_gpu_cache()


class CustomWandbCallback(WandbCallback):
    """Custom WandB callback with additional logging."""

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel | None = None,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Add custom metrics to WandB logs."""
        if self._wandb is not None and logs is not None:
            # Add GPU memory stats
            memory_stats = check_gpu_memory()
            if "error" not in memory_stats:
                for gpu_id, stats in memory_stats.items():
                    if isinstance(stats, dict):
                        logs[f"gpu/{gpu_id}/memory_used_gb"] = stats["allocated_gb"]
                        logs[f"gpu/{gpu_id}/memory_util_percent"] = stats["utilization_percent"]

        super().on_log(args, state, control, model=model, logs=logs, **kwargs)


class TrainingManager:
    """Manages the training process with enhanced features."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.seqeval_metric = load_seqeval_metric()

    def create_training_arguments(self) -> TrainingArguments:
        """Create training arguments from config."""
        # Detect mixed precision support
        mp_support = detect_mixed_precision_support()

        # Override config based on hardware support
        fp16 = self.config.training.fp16 and mp_support["fp16"]
        bf16 = self.config.training.bf16 and mp_support["bf16"]
        tf32 = self.config.training.tf32 and mp_support["tf32"]

        # Ensure only one of fp16/bf16 is True
        if bf16:
            fp16 = False

        logger.info(f"Mixed precision settings - fp16: {fp16}, bf16: {bf16}, tf32: {tf32}")

        # Create output directory
        Path(self.config.training.output_dir).mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=self.config.training.output_dir,
            num_train_epochs=self.config.training.num_train_epochs,
            per_device_train_batch_size=self.config.training.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.training.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            gradient_checkpointing=self.config.training.gradient_checkpointing,
            optim=self.config.training.optim,
            learning_rate=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            warmup_ratio=self.config.training.warmup_ratio,
            max_grad_norm=self.config.training.max_grad_norm,
            # Evaluation and saving
            eval_strategy=self.config.training.eval_strategy,
            save_strategy=self.config.training.save_strategy,
            logging_steps=self.config.training.logging_steps,
            save_total_limit=self.config.training.save_total_limit,
            load_best_model_at_end=self.config.training.load_best_model_at_end,
            metric_for_best_model=self.config.training.metric_for_best_model,
            greater_is_better=self.config.training.greater_is_better,
            # Mixed precision
            fp16=fp16,
            bf16=bf16,
            tf32=tf32,
            # Logging
            report_to=self.config.training.report_to if self.config.logging.use_wandb else [],
            disable_tqdm=self.config.logging.disable_tqdm,
            log_level=self.config.logging.log_level,
            # Other settings
            seed=self.config.training.seed,
            data_seed=self.config.training.data_seed,
            local_rank=self.config.training.local_rank,
            ddp_find_unused_parameters=self.config.training.ddp_find_unused_parameters,
            # Hub settings
            push_to_hub=self.config.training.push_to_hub,
            hub_model_id=self.config.training.hub_model_id,
            hub_strategy=self.config.training.hub_strategy,
        )

        return training_args

    def create_trainer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        data_collator: DataCollatorForTokenClassification,
        compute_metrics: Callable[[Any], dict[str, float]] | None = None,
    ) -> Trainer:
        """Create trainer with all components."""
        # Create training arguments
        training_args = self.create_training_arguments()

        # Create compute metrics function if not provided
        if compute_metrics is None:
            if self.config.training.use_enhanced_evaluation:
                from .evaluation import create_enhanced_compute_metrics

                compute_metrics = create_enhanced_compute_metrics(
                    self.config.data.label_names,
                    self.seqeval_metric,
                    compute_detailed=self.config.training.compute_entity_level_metrics,
                )
            else:
                compute_metrics = compute_metrics_factory(self.config.data.label_names, self.seqeval_metric)

        # Create callbacks
        callbacks = [
            MemoryCallback(clear_cache_steps=self.config.training.clear_cache_steps),
        ]

        # Add early stopping callback
        if self.config.training.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.training.early_stopping_patience,
                    early_stopping_threshold=self.config.training.early_stopping_threshold,
                )
            )

        # Add WandB callback if enabled
        if self.config.logging.use_wandb and "wandb" in self.config.training.report_to:
            callbacks.append(CustomWandbCallback())

        # Create custom trainer with focal loss support
        CustomTrainerClass = create_custom_trainer_class(self.config, train_dataset)
        trainer = CustomTrainerClass(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

        return trainer

    def train(self, trainer: Trainer, resume_from_checkpoint: str | None = None) -> dict[str, Any]:
        """Run training with error handling."""
        try:
            # Use resume_from_checkpoint from config if not provided
            if resume_from_checkpoint is None:
                resume_from_checkpoint = self.config.training.resume_from_checkpoint

            # Log training start
            logger.info("Starting training...")
            if resume_from_checkpoint:
                logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")

            # Start training
            train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

            # Log metrics
            logger.info(f"Training completed. Metrics: {train_result.metrics}")

            # Save final model
            if self.config.training.final_output_dir:
                logger.info(f"Saving final model to {self.config.training.final_output_dir}")
                trainer.save_model(self.config.training.final_output_dir)
                trainer.save_state()

            return train_result.metrics  # type: ignore[no-any-return]

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA out of memory during training: {e}")
            logger.error("Try reducing batch size or sequence length")
            raise

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            # Save checkpoint
            checkpoint_dir = os.path.join(self.config.training.output_dir, "interrupted_checkpoint")
            logger.info(f"Saving interrupted checkpoint to {checkpoint_dir}")
            trainer.save_model(checkpoint_dir)
            trainer.save_state()
            raise

        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise

        finally:
            # Clean up
            clear_gpu_cache()

            # Close WandB run if active
            if wandb.run is not None:
                wandb.finish()


def create_custom_trainer_class(config: Config, train_dataset: Dataset | None = None) -> type[Trainer]:
    """Create a custom Trainer class with additional features."""

    # Setup custom loss function if not using standard cross-entropy
    custom_loss_fn = None
    if config.training.loss_type != "cross_entropy" or config.training.use_class_weights:
        logger.info(f"Setting up custom loss function: {config.training.loss_type}")
        if config.training.use_class_weights:
            logger.info("Class weighting enabled")

        # Compute class frequencies if needed
        class_frequencies = None
        if (
            config.training.loss_type in ["class_balanced"]
            or (config.training.loss_type == "focal" and config.training.focal_alpha is None)
            or config.training.use_class_weights
        ):
            if train_dataset is not None:
                class_frequencies = compute_class_frequencies(train_dataset, "labels")
            else:
                logger.warning("No train_dataset provided for class frequency computation")

        # Create loss function with proper type handling
        if config.training.loss_type == "focal":
            focal_alpha: float | list[float] | None = config.training.focal_alpha
            # Auto-compute alpha from class frequencies if not specified
            if config.training.focal_alpha is None and class_frequencies is not None:
                # Compute inverse frequency weights
                total_samples = sum(class_frequencies)
                alpha_weights = [total_samples / (len(class_frequencies) * freq) for freq in class_frequencies]
                focal_alpha = alpha_weights
                logger.info(f"Auto-computed focal loss alpha weights: {alpha_weights}")

            custom_loss_fn = create_loss_function(
                loss_type=config.training.loss_type,
                num_labels=config.model.num_labels,
                gamma=config.training.focal_gamma,
                alpha=focal_alpha,
                class_frequencies=class_frequencies,
                class_weights=config.training.manual_class_weights if config.training.use_class_weights else None,
                auto_weight=config.training.use_class_weights
                and config.training.manual_class_weights is None
                and focal_alpha is None,
                weight_type=config.training.class_weight_type,
                class_weight_smoothing=config.training.class_weight_smoothing,
            )
        elif config.training.loss_type == "label_smoothing":
            custom_loss_fn = create_loss_function(
                loss_type=config.training.loss_type,
                num_labels=config.model.num_labels,
                smoothing=config.training.label_smoothing,
            )
        elif config.training.loss_type == "class_balanced":
            if class_frequencies is None:
                raise ValueError("class_frequencies required for class_balanced loss")
            custom_loss_fn = create_loss_function(
                loss_type=config.training.loss_type,
                num_labels=config.model.num_labels,
                class_frequencies=class_frequencies,
                beta=config.training.class_balanced_beta,
            )
        elif config.training.loss_type == "weighted_cross_entropy" or (
            config.training.loss_type == "cross_entropy" and config.training.use_class_weights
        ):
            # Handle weighted cross-entropy
            custom_loss_fn = create_loss_function(
                loss_type="weighted_cross_entropy",
                num_labels=config.model.num_labels,
                class_frequencies=class_frequencies,
                class_weights=config.training.manual_class_weights,
                auto_weight=config.training.use_class_weights and config.training.manual_class_weights is None,
                weight_type=config.training.class_weight_type,
                class_weight_smoothing=config.training.class_weight_smoothing,
            )
        else:
            custom_loss_fn = create_loss_function(
                loss_type=config.training.loss_type,
                num_labels=config.model.num_labels,
            )

    class CustomTrainer(Trainer):
        """Custom trainer with enhanced error handling, logging, and focal loss support."""

        def get_train_dataloader(self) -> torch.utils.data.DataLoader:
            """Override to use custom batch sampler if enabled."""
            if config.training.use_batch_balancing and train_dataset is not None:
                from torch.utils.data import DataLoader

                from .batch_balancing import BalancedBatchSampler, BatchCompositionLogger, EntityAwareBatchSampler

                # Create appropriate sampler
                if config.training.batch_balance_type == "balanced":
                    batch_sampler = BalancedBatchSampler(
                        dataset=train_dataset,
                        batch_size=self.args.per_device_train_batch_size,
                        min_positive_ratio=config.training.min_positive_ratio,
                        seed=self.args.seed,
                    )
                elif config.training.batch_balance_type == "entity_aware":
                    batch_sampler = EntityAwareBatchSampler(
                        dataset=train_dataset,
                        batch_size=self.args.per_device_train_batch_size,
                        seed=self.args.seed,
                    )
                else:
                    raise ValueError(f"Unknown batch_balance_type: {config.training.batch_balance_type}")

                # Initialize batch composition logger if enabled
                if config.training.log_batch_composition:
                    self.batch_logger = BatchCompositionLogger(log_every_n_batches=config.training.log_batch_every_n)

                # Create dataloader with custom batch sampler
                return DataLoader(
                    train_dataset,
                    batch_sampler=batch_sampler,
                    collate_fn=self.data_collator,
                    pin_memory=self.args.dataloader_pin_memory,
                    num_workers=self.args.dataloader_num_workers,
                )
            else:
                # Use default dataloader
                return super().get_train_dataloader()

        def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None) -> Any:
            """Create custom learning rate scheduler."""
            if config.training.lr_scheduler_type not in ["linear"]:
                # Use our custom scheduler
                if optimizer is None:
                    optimizer = self.optimizer

                scheduler = create_scheduler(
                    optimizer=optimizer,
                    scheduler_type=config.training.lr_scheduler_type,
                    training_args=self.args,
                    num_training_steps=num_training_steps,
                    **config.training.lr_scheduler_kwargs,
                )
                return scheduler
            else:
                # Use default transformers scheduler
                return super().create_scheduler(num_training_steps, optimizer)

        def compute_loss(
            self, model: PreTrainedModel, inputs: dict[str, Any], return_outputs: bool = False
        ) -> torch.Tensor | tuple[torch.Tensor, Any]:
            """Override compute_loss to use custom loss functions."""
            try:
                # Log batch composition if enabled
                if config.training.log_batch_composition and hasattr(self, "batch_logger"):
                    labels = inputs.get("labels")
                    if labels is not None:
                        self.batch_logger.log_batch(labels)

                if custom_loss_fn is not None:
                    # Use custom loss function
                    labels = inputs.get("labels")
                    if labels is None:
                        raise ValueError("No labels found in inputs for custom loss computation")

                    # Forward pass
                    outputs = model(**inputs)
                    logits = outputs.get("logits")
                    if logits is None:
                        raise ValueError("No logits found in model outputs")

                    # Compute custom loss
                    loss = custom_loss_fn(logits, labels)

                    if return_outputs:
                        return (loss, outputs)
                    return loss
                else:
                    # Use default loss computation
                    return super().compute_loss(model, inputs, return_outputs)

            except torch.cuda.OutOfMemoryError:
                logger.error("OOM in compute_loss, clearing cache and retrying...")
                clear_gpu_cache()
                # Try with reduced batch
                for key in ["input_ids", "attention_mask", "labels"]:
                    if key in inputs and hasattr(inputs[key], "shape"):
                        batch_size = inputs[key].shape[0]
                        if batch_size > 1:
                            # Reduce to half batch size
                            inputs[key] = inputs[key][: batch_size // 2]

                return super().compute_loss(model, inputs, return_outputs)

        def evaluation_loop(self, *args: Any, **kwargs: Any) -> Any:
            """Override evaluation loop to add memory management."""
            clear_gpu_cache()
            result = super().evaluation_loop(*args, **kwargs)
            clear_gpu_cache()
            return result

    return CustomTrainer


def run_training_pipeline(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    data_collator: DataCollatorForTokenClassification,
    config: Config,
) -> dict[str, Any]:
    """Run complete training pipeline."""
    # Setup WandB if enabled
    if config.logging.use_wandb:
        config.setup_wandb()
        from .utils import setup_wandb_logging

        setup_wandb_logging(config)

    # Create training manager
    training_manager = TrainingManager(config)

    # Create trainer
    trainer = training_manager.create_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # Run training
    metrics = training_manager.train(trainer)

    return metrics
