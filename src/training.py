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

from .config import Config
from .evaluation import compute_metrics_factory
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
            gpu_info = check_gpu_memory()
            logger.debug(f"Step {state.global_step}: {gpu_info}")

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any) -> None:
        """Log GPU memory usage."""
        if state.global_step % args.logging_steps == 0:
            gpu_info = check_gpu_memory()
            logger.info(f"GPU memory: {gpu_info}")

    def on_epoch_end(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any
    ) -> None:
        """Clear cache at end of epoch."""
        clear_gpu_cache()


class CustomWandbCallback(WandbCallback):
    """Custom WandB callback with additional features."""

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Log additional metrics to WandB."""
        super().on_log(args, state, control, logs, **kwargs)

        # Log GPU metrics if available
        if logs and self._wandb is not None:
            gpu_info = check_gpu_memory()
            # Add GPU stats to logs for each GPU
            for gpu_id, stats in gpu_info.items():
                if isinstance(stats, dict) and "allocated_gb" in stats:
                    logs[f"gpu/{gpu_id}/memory_used_gb"] = stats["allocated_gb"]
                    logs[f"gpu/{gpu_id}/memory_util_percent"] = stats.get("utilization_percent", 0)


class ConfigSaveCallback(TrainerCallback):
    """Callback to save config alongside model checkpoints."""

    def __init__(self, config: Config) -> None:
        """Initialize with config to save."""
        self.config = config

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Save config when model checkpoint is saved."""
        # Save config to the output directory
        output_dir = args.output_dir

        # If we're saving the best model at the end, also save to that directory
        if args.should_save:
            # For regular checkpoints, save in checkpoint directory
            if state.best_model_checkpoint:
                checkpoint_folder = Path(output_dir) / f"checkpoint-{state.global_step}"
                if checkpoint_folder.exists():
                    config_path = checkpoint_folder / "config.yaml"
                    self.config.to_yaml(config_path)
                    logger.info(f"Config saved to checkpoint: {config_path}")

            # Always save to main output directory
            config_path = Path(output_dir) / "config.yaml"
            self.config.to_yaml(config_path)
            logger.info(f"Config saved to: {config_path}")


class TrainingManager:
    """Manages the training process with enhanced features."""

    def __init__(self, config: Config) -> None:
        self.config = config

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
                    compute_detailed=self.config.training.compute_entity_level_metrics,
                )
            else:
                compute_metrics = compute_metrics_factory(self.config.data.label_names)

        # Create callbacks
        callbacks = [
            MemoryCallback(clear_cache_steps=self.config.training.clear_cache_steps),
            ConfigSaveCallback(config=self.config),
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

            # Save final model with adapters
            if self.config.training.final_output_dir:
                logger.info(f"Saving final model to {self.config.training.final_output_dir}")
                trainer.save_model(self.config.training.final_output_dir)
                trainer.save_state()
                # Save config to final output directory
                config_path = Path(self.config.training.final_output_dir) / "config.yaml"
                self.config.to_yaml(config_path)
                logger.info(f"Config saved to final output: {config_path}")

                # Check if we should merge and save the model
                if getattr(self.config.training, "merge_adapters_on_save", True):
                    # Save merged model to a separate directory
                    merged_output_dir = str(
                        Path(self.config.training.final_output_dir).parent
                        / f"{Path(self.config.training.final_output_dir).name}-merged"
                    )
                    logger.info(f"Merging LoRA adapters and saving to {merged_output_dir}")

                    from .model import merge_and_save_model

                    merge_and_save_model(trainer.model, trainer.tokenizer, merged_output_dir)

                    # Also save config to merged model directory
                    merged_config_path = Path(merged_output_dir) / "config.yaml"
                    self.config.to_yaml(merged_config_path)
                    logger.info(f"Config saved to merged model: {merged_config_path}")

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
            # Save config to interrupted checkpoint
            config_path = Path(checkpoint_dir) / "config.yaml"
            self.config.to_yaml(config_path)
            logger.info(f"Config saved to interrupted checkpoint: {config_path}")
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
    """Create a custom trainer class with custom loss function."""
    # Compute class frequencies for loss weighting
    class_frequencies = None
    if config.training.loss_type in ["focal", "class_balanced"] and config.training.focal_alpha is None:
        if train_dataset is not None:
            logger.info("Computing class frequencies for loss weighting...")
            class_frequencies = compute_class_frequencies(train_dataset)
            logger.info(f"Class frequencies: {class_frequencies}")
        else:
            logger.warning("Train dataset not provided, cannot compute class frequencies for loss weighting")

    # Create loss function
    loss_fn = create_loss_function(
        loss_type=config.training.loss_type,
        num_labels=len(config.data.label_names),
        alpha=config.training.focal_alpha,  # Note: focal_alpha -> alpha
        gamma=config.training.focal_gamma,  # Note: focal_gamma -> gamma
        smoothing=config.training.label_smoothing,  # Note: label_smoothing -> smoothing
        beta=config.training.class_balanced_beta,  # Note: class_balanced_beta -> beta
        class_frequencies=class_frequencies,
    )

    class CustomTrainer(Trainer):
        """Custom trainer with focal loss support."""

        def compute_loss(
            self,
            model: PreTrainedModel,
            inputs: dict[str, torch.Tensor],
            return_outputs: bool = False,
        ) -> torch.Tensor | tuple[torch.Tensor, Any]:
            """Compute custom loss."""
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits

            # Reshape for loss computation
            batch_size, seq_length = labels.shape
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)

            # Compute loss
            loss = loss_fn(logits_flat, labels_flat)

            return (loss, outputs) if return_outputs else loss

        def create_scheduler(
            self, num_training_steps: int, optimizer: torch.optim.Optimizer | None = None
        ) -> torch.optim.lr_scheduler._LRScheduler:
            """Create custom learning rate scheduler if specified."""
            if hasattr(config.training, "lr_scheduler_type") and config.training.lr_scheduler_type:
                if optimizer is None:
                    optimizer = self.optimizer
                return create_scheduler(
                    optimizer=optimizer,
                    scheduler_type=config.training.lr_scheduler_type,
                    num_training_steps=num_training_steps,
                    num_warmup_steps=int(num_training_steps * self.args.warmup_ratio),
                    **config.training.lr_scheduler_kwargs,
                )
            else:
                # Use default scheduler
                return super().create_scheduler(num_training_steps, optimizer)

        def evaluation_loop(self, *args: Any, **kwargs: Any) -> Any:
            """Custom evaluation loop with GPU memory management."""
            # Clear cache before evaluation
            clear_gpu_cache()

            # Run default evaluation
            result = super().evaluation_loop(*args, **kwargs)

            # Clear cache after evaluation
            clear_gpu_cache()

            return result

    return CustomTrainer


# Multi-dataset support functions
def create_multi_dataset_trainer_class(config: Config) -> type[Trainer]:
    """Create a custom trainer class for multi-dataset training."""
    from .datasets.samplers import DistributedMultiDatasetSampler, MultiDatasetSampler

    class MultiDatasetTrainer(Trainer):
        """Custom trainer for multi-dataset training."""

        def _get_train_sampler(self) -> MultiDatasetSampler | None:
            """Get custom sampler for multi-dataset training."""
            if self.train_dataset is None:
                return None

            # Extract dataset sizes from the combined dataset
            # This assumes the dataset was created by MultiDatasetLoader
            # and contains metadata about individual dataset sizes
            dataset_sizes = getattr(self.train_dataset, "dataset_sizes", None)

            if dataset_sizes is None:
                # Fallback: estimate sizes based on total size and number of datasets
                total_size = len(self.train_dataset)
                num_datasets = len(config.data.multi_dataset.dataset_names)
                dataset_sizes = [total_size // num_datasets] * num_datasets

                # Distribute remainder
                remainder = total_size % num_datasets
                for i in range(remainder):
                    dataset_sizes[i] += 1

            # Check if distributed training
            if self.args.local_rank != -1:
                return DistributedMultiDatasetSampler(
                    dataset_sizes=dataset_sizes,
                    num_replicas=self.args.world_size,
                    rank=self.args.local_rank,
                    shuffle=True,
                    strategy=config.data.multi_dataset.mixing_strategy,
                    weights=config.data.multi_dataset.dataset_weights,
                    seed=self.args.seed,
                    batch_size=self.args.per_device_train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                )
            else:
                return MultiDatasetSampler(
                    dataset_sizes=dataset_sizes,
                    strategy=config.data.multi_dataset.mixing_strategy,
                    weights=config.data.multi_dataset.dataset_weights,
                    seed=self.args.seed,
                    batch_size=self.args.per_device_train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                )

    return MultiDatasetTrainer


def create_multi_dataset_trainer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: Config,
) -> tuple[Trainer, Dataset, Dataset]:
    """
    Create trainer with multi-dataset support.

    Args:
        model: The model to train
        tokenizer: The tokenizer
        config: Configuration object

    Returns:
        Tuple of (trainer, train_dataset, eval_dataset)
    """

    from .data import create_data_collator
    from .datasets.multi_dataset import MultiDatasetLoader

    # Create multi-dataset loader
    loader = MultiDatasetLoader(config)

    # Load and prepare datasets
    train_dataset, eval_dataset = loader.create_train_eval_datasets()

    # Log dataset statistics
    stats = loader.get_dataset_statistics()
    logger.info(f"Multi-dataset statistics: {stats}")

    # Store dataset sizes as metadata (for sampler)
    if hasattr(train_dataset, "_dataset_sizes"):
        train_dataset.dataset_sizes = train_dataset._dataset_sizes
    else:
        # Estimate from loader
        train_dataset.dataset_sizes = [len(d) for d in loader.datasets]

    # Create data collator
    data_collator = create_data_collator(tokenizer)

    # Create training manager
    training_manager = TrainingManager(config)

    # Get the appropriate trainer class
    if config.data.multi_dataset.mixing_strategy in ["interleave", "weighted"]:
        # Use custom trainer with multi-dataset sampler
        CustomTrainerClass = create_multi_dataset_trainer_class(config)
        trainer = CustomTrainerClass(
            model=model,
            args=training_manager.create_training_arguments(),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics_factory(config.data.label_names),
            callbacks=training_manager.create_trainer(
                model, tokenizer, train_dataset, eval_dataset, data_collator
            ).callbacks,  # Reuse callbacks
        )
    else:
        # Use standard trainer
        trainer = training_manager.create_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

    return trainer, train_dataset, eval_dataset


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
