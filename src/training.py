"""Training loop and utilities for Mistral NER fine-tuning."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import wandb
from datasets import Dataset
from peft import PeftMixedModel, PeftModel
from transformers.data.data_collator import DataCollatorForTokenClassification
from transformers.integrations.integration_utils import WandbCallback
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer
from transformers.trainer_callback import EarlyStoppingCallback, TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

from .config import Config
from .evaluation import compute_metrics_factory
from .utils import check_gpu_memory, clear_gpu_cache, detect_mixed_precision_support

# Setup logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    pass


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

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any) -> None:
        """Clear cache at end of epoch."""
        clear_gpu_cache()


class CustomWandbCallback(WandbCallback):
    """Custom WandB callback with additional features."""

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: PreTrainedModel | None = None,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Log additional metrics to WandB."""
        super().on_log(args, state, control, model, logs, **kwargs)

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
                checkpoint_folder = Path(f"{output_dir}") / f"checkpoint-{state.global_step}"
                if checkpoint_folder.exists():
                    config_path = checkpoint_folder / "config.yaml"
                    self.config.to_yaml(config_path)
                    logger.info(f"Config saved to checkpoint: {config_path}")

            # Always save to main output directory
            config_path = Path(f"{output_dir}") / "config.yaml"
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
            # Hub settings
            push_to_hub=self.config.training.push_to_hub,
            hub_model_id=self.config.training.hub_model_id,
            hub_strategy=self.config.training.hub_strategy,
        )

        return training_args

    def create_trainer(
        self,
        model: PeftModel | PeftMixedModel,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        data_collator: DataCollatorForTokenClassification,
        compute_metrics: Callable[[Any], dict[str, float]] | None = None,
    ) -> AdvancedLossTrainer:
        """Create trainer with all components."""
        # Create training arguments
        training_args = self.create_training_arguments()

        # Create compute metrics function if not provided
        if compute_metrics is None:
            compute_metrics = compute_metrics_factory(self.config.data.label_names, self.config.training.compute_entity_level_metrics)

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

        # Ensure num_labels is not None
        num_labels = self.config.data.num_labels
        if num_labels is None:
            raise ValueError("num_labels cannot be None for AdvancedLossTrainer")

        # Create custom trainer with focal loss support
        trainer = AdvancedLossTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            num_labels=num_labels,
            class_balanced_loss=self.config.training.class_balanced_loss,
            class_balanced_beta=self.config.training.class_balanced_beta,
            class_balanced_weights=self.config.training.class_balanced_weights,
            focal_loss=self.config.training.focal_loss,
            focal_gamma=self.config.training.focal_gamma,
            label_smoothing_loss=self.config.training.label_smoothing_loss,
            label_smoothing=self.config.training.label_smoothing,
        )

        return trainer

    def train(self, trainer: AdvancedLossTrainer, resume_from_checkpoint: str | None = None) -> dict[str, Any]:
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
                if self.config.training.merge_adapters_on_save:
                    # Save merged model to a separate directory
                    merged_output_dir = str(
                        Path(self.config.training.final_output_dir).parent / f"{Path(self.config.training.final_output_dir).name}-merged"
                    )
                    logger.info(f"Merging LoRA adapters and saving to {merged_output_dir}")

                    from .model import merge_and_save_model

                    merge_and_save_model(trainer.model, trainer.processing_class, merged_output_dir)

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


def compute_class_balanced_weights(
    num_samples_per_class: torch.Tensor, beta: float = 0.99, device: torch.device | None = None
) -> torch.Tensor:
    effective_num = 1.0 - torch.pow(beta, num_samples_per_class.to(device))
    weights = (1.0 - beta) / (effective_num + 1e-12)
    # Normalize so total scale is comparable
    weights = weights / weights.sum() * num_samples_per_class.numel()
    return weights


def label_smoothed_targets(labels: torch.Tensor, num_classes: int, smoothing: float, ignore_index: int | None) -> torch.Tensor:
    with torch.no_grad():
        confidence = 1.0 - smoothing
        low_confidence = smoothing / (num_classes - 1)
        smoothed = torch.full((*labels.shape, num_classes), low_confidence, device=labels.device)
        # Set the true class to confidence
        true_class_mask = labels.unsqueeze(-1) == torch.arange(num_classes, device=labels.device).view(*(1,) * labels.ndim, num_classes)
        num_true_classes = int(true_class_mask.sum())
        smoothed = smoothed.masked_scatter(true_class_mask, torch.full((num_true_classes,), confidence, device=labels.device))
        # Zero out ignore_index positions
        if ignore_index is not None:
            ignore_mask = labels == ignore_index
            if ignore_mask.any():
                smoothed[ignore_mask] = 0.0
        return smoothed  # shape (..., num_classes)


class AdvancedLossTrainer(Trainer):
    def __init__(
        self,
        model: PeftModel | PeftMixedModel,
        args: TrainingArguments,
        data_collator: DataCollatorForTokenClassification,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        processing_class: PreTrainedTokenizerBase,
        compute_metrics: Callable[[Any], dict[str, float]],
        callbacks: list[TrainerCallback],
        num_labels: int,
        class_balanced_loss: bool,
        class_balanced_beta: float,
        class_balanced_weights: list[float],
        focal_loss: bool,
        focal_gamma: float,
        label_smoothing_loss: bool,
        label_smoothing: float,
        ignore_index: int = -100,
        **kwargs: Any,
    ) -> None:
        # Initialize parent class with standard Trainer parameters
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            **kwargs,
        )

        self.num_labels = num_labels
        self.ignore_index = ignore_index
        self.label_smoothing_loss = label_smoothing_loss
        self.label_smoothing = label_smoothing
        self.focal_loss = focal_loss
        self.focal_gamma = focal_gamma
        self.class_balanced_loss = class_balanced_loss
        self.class_balanced_beta = class_balanced_beta
        self.class_balanced_weights = class_balanced_weights

        if self.class_balanced_loss:
            if not self.class_balanced_weights:
                # DOCDEV-TODO # Add auto computation of class weights if not provided
                raise ValueError("Provide class_balanced_weights when class_balanced_loss=True")
            cbw_tensor = torch.tensor(self.class_balanced_weights, dtype=torch.float)
            self.class_weights = compute_class_balanced_weights(cbw_tensor, beta=class_balanced_beta, device=self.args.device)

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("`labels` must be in inputs for loss computation")

        outputs = model(**inputs)
        logits = outputs.logits  # (batch, seq_len, num_labels)
        batch_size, seq_len, num_labels = logits.shape

        logits_flat = logits.view(-1, num_labels)  # (B*S, C)
        labels_flat = labels.view(-1)  # (B*S,)

        active_mask = labels_flat != self.ignore_index  # which tokens to include

        log_probs = torch.nn.functional.log_softmax(logits_flat, dim=-1)  # (B*S, C)
        probs = log_probs.exp()

        # Base cross entropy (with optional label smoothing)
        if self.label_smoothing_loss:
            smoothed = label_smoothed_targets(
                labels_flat, num_classes=num_labels, smoothing=self.label_smoothing, ignore_index=self.ignore_index
            )
            if active_mask.any():
                smoothed_active = smoothed[active_mask]  # (N_active, C)
                logp_active = log_probs[active_mask]  # (N_active, C)
            else:
                smoothed_active = torch.zeros((0, num_labels), device=logits.device)
                logp_active = log_probs[active_mask]
            ce_loss = -(smoothed_active * logp_active).sum(dim=-1)  # (N_active,)
        else:
            weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
            ce_full = torch.nn.functional.nll_loss(log_probs, labels_flat, weight=weight, ignore_index=self.ignore_index, reduction="none")
            ce_loss = ce_full[active_mask] if active_mask.any() else torch.zeros((0,), device=logits.device)

        # If using class-balanced on top of label smoothing, apply per-example scaling
        if self.class_balanced_loss and self.label_smoothing_loss:
            cb_weight_per_label = self.class_weights.to(logits.device)
            true_labels_active = labels_flat[active_mask]
            per_example_cb = cb_weight_per_label[true_labels_active]
            ce_loss = ce_loss * per_example_cb

        # Focal modulation
        if self.focal_loss and self.focal_gamma > 0 and active_mask.any():
            pt = probs[active_mask].gather(1, labels_flat[active_mask].unsqueeze(1)).squeeze(1).clamp(min=1e-8)
            focal_factor = (1 - pt) ** self.focal_gamma
            ce_loss = ce_loss * focal_factor

        loss = ce_loss.mean() if ce_loss.numel() > 0 else torch.tensor(0.0, device=logits.device)
        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(self, *args: Any, **kwargs: Any) -> Any:
        """Custom evaluation loop with GPU memory management."""
        # Clear cache before evaluation
        clear_gpu_cache()

        # Run default evaluation
        result = super().evaluation_loop(*args, **kwargs)

        # Clear cache after evaluation
        clear_gpu_cache()

        return result


# DEVDOC-TODO # Verify and fix multi-dataset support
# Note: This is a placeholder for multi-dataset support logic.
# It should be implemented based on the specific requirements of the project.
# # Multi-dataset support functions
# def create_multi_dataset_trainer_class(config: Config) -> type[Trainer]:
#     """Create a custom trainer class for multi-dataset training."""
#     from .dataset_utils._samplers import DistributedMultiDatasetSampler, MultiDatasetSampler

#     class MultiDatasetTrainer(Trainer):
#         """Custom trainer for multi-dataset training."""

#         def _get_train_sampler(self) -> MultiDatasetSampler | None:
#             """Get custom sampler for multi-dataset training."""
#             if self.train_dataset is None:
#                 return None

#             # Extract dataset sizes from the combined dataset
#             # This assumes the dataset was created by MultiDatasetLoader
#             # and contains metadata about individual dataset sizes
#             dataset_sizes = getattr(self.train_dataset, "dataset_sizes", None)

#             if dataset_sizes is None:
#                 # Fallback: estimate sizes based on total size and number of datasets
#                 total_size = len(self.train_dataset)
#                 num_datasets = len(config.data.multi_dataset.dataset_names)
#                 dataset_sizes = [total_size // num_datasets] * num_datasets

#                 # Distribute remainder
#                 remainder = total_size % num_datasets
#                 for i in range(remainder):
#                     dataset_sizes[i] += 1

#             # Check if distributed training
#             if self.args.local_rank != -1:
#                 return DistributedMultiDatasetSampler(
#                     dataset_sizes=dataset_sizes,
#                     num_replicas=self.args.world_size,
#                     rank=self.args.local_rank,
#                     shuffle=True,
#                     strategy=config.data.multi_dataset.mixing_strategy,
#                     weights=config.data.multi_dataset.dataset_weights,
#                     seed=self.args.seed,
#                     batch_size=self.args.per_device_train_batch_size,
#                     drop_last=self.args.dataloader_drop_last,
#                 )
#             else:
#                 return MultiDatasetSampler(
#                     dataset_sizes=dataset_sizes,
#                     strategy=config.data.multi_dataset.mixing_strategy,
#                     weights=config.data.multi_dataset.dataset_weights,
#                     seed=self.args.seed,
#                     batch_size=self.args.per_device_train_batch_size,
#                     drop_last=self.args.dataloader_drop_last,
#                 )

#     return MultiDatasetTrainer


# def create_multi_dataset_trainer(
#     model: PreTrainedModel,
#     tokenizer: PreTrainedTokenizerBase,
#     config: Config,
# ) -> tuple[Trainer, Dataset, Dataset]:
#     """
#     Create trainer with multi-dataset support.

#     Args:
#         model: The model to train
#         tokenizer: The tokenizer
#         config: Configuration object

#     Returns:
#         Tuple of (trainer, train_dataset, eval_dataset)
#     """

#     from .data import create_data_collator
#     from .dataset_utils import MultiDatasetLoader

#     # Create multi-dataset loader
#     loader = MultiDatasetLoader(config)

#     # Load and prepare datasets
#     train_dataset, eval_dataset = loader.create_train_eval_datasets()

#     # Log dataset statistics
#     stats = loader.get_dataset_statistics()
#     logger.info(f"Multi-dataset statistics: {stats}")

#     # Store dataset sizes as metadata (for sampler)
#     if hasattr(train_dataset, "_dataset_sizes"):
#         train_dataset.dataset_sizes = train_dataset._dataset_sizes
#     else:
#         # Estimate from loader
#         train_dataset.dataset_sizes = [len(d) for d in loader.datasets]

#     # Create data collator
#     data_collator = create_data_collator(tokenizer)

#     # Create training manager
#     training_manager = TrainingManager(config)

#     # Get the appropriate trainer class
#     if config.data.multi_dataset.mixing_strategy in ["interleave", "weighted"]:
#         # Use custom trainer with multi-dataset sampler
#         CustomTrainerClass = create_multi_dataset_trainer_class(config)
#         trainer = CustomTrainerClass(
#             model=model,
#             args=training_manager.create_training_arguments(),
#             train_dataset=train_dataset,
#             eval_dataset=eval_dataset,
#             tokenizer=tokenizer,
#             data_collator=data_collator,
#             compute_metrics=compute_metrics_factory(config.data.label_names),
#             callbacks=training_manager.create_trainer(
#                 model, tokenizer, train_dataset, eval_dataset, data_collator
#             ).callbacks,  # Reuse callbacks
#         )
#     else:
#         # Use standard trainer
#         trainer = training_manager.create_trainer(
#             model=model,
#             tokenizer=tokenizer,
#             train_dataset=train_dataset,
#             eval_dataset=eval_dataset,
#             data_collator=data_collator,
#         )

#     return trainer, train_dataset, eval_dataset


def run_training_pipeline(
    model: PeftModel | PeftMixedModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    data_collator: DataCollatorForTokenClassification,
    config: Config,
) -> dict[str, Any]:
    """Run complete training pipeline."""
    # Setup WandB if enabled
    # DOCDEV-TODO: Validate and fix
    # if config.logging.use_wandb:
    #     config.setup_wandb()
    #     from .utils import setup_wandb_logging

    #     setup_wandb_logging(config)

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
